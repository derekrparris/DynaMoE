//
//  ImageAttachmentStore.swift
//  DynaMoE
//
//  Durable storage for images the user attaches to chat turns.
//
//  Pixels live here; the conversation JSON carries only `ImageAttachmentRef`
//  metadata. Putting base64 in the session file would rewrite megabytes on every
//  streaming save and defeat `ChatSessionStore`'s digest write-skip, so the two
//  stores are deliberately separate and share one retention lifecycle:
//
//  - assets live under `~/Library/Application Support/DynaMoE/sessions/assets/`
//    (`<sessionId>/<attachmentId>.<ext>`), beside the conversation files;
//  - deleting a conversation deletes its asset directory;
//  - the orphan pass removes assets whose ref no longer exists in any session.
//
//  The app is not sandboxed, so writing beside the sessions directory needs no
//  bookmark or entitlement.
//

import Foundation

nonisolated final class ImageAttachmentStore: @unchecked Sendable {
    static let shared = ImageAttachmentStore()

    /// Directory holding every session's asset folders. Nil only when
    /// Application Support itself cannot be resolved, in which case attachments
    /// still work for the current launch (they are prepared in memory) but are
    /// not durable.
    private let root: URL?
    private let ioQueue = DispatchQueue(label: "com.dynamoe.imageattachmentstore.io", qos: .utility)
    private let fileManager: FileManager

    init(root: URL? = nil, fileManager: FileManager = .default) {
        self.fileManager = fileManager
        self.root = root ?? ImageAttachmentStore.defaultRoot(using: fileManager)
    }

    // MARK: - Locations

    nonisolated static func defaultRoot(using fileManager: FileManager = .default) -> URL? {
        guard let sessions = ChatSessionStore.defaultDirectory(using: fileManager) else { return nil }
        return sessions.appendingPathComponent("assets", isDirectory: true)
    }

    /// Absolute location of a ref's bytes, or nil when the store has no durable
    /// root. The ref's relative path is trusted only as a *relative* path: a
    /// hand-edited session file must not be able to point outside the asset tree.
    func absoluteURL(for ref: ImageAttachmentRef) -> URL? {
        guard let root else { return nil }
        let components = ref.relativePath.split(separator: "/", omittingEmptySubsequences: true)
        guard !components.isEmpty,
              !components.contains(".."),
              !components.contains(".") else { return nil }
        var url = root
        for component in components {
            url.appendPathComponent(String(component), isDirectory: false)
        }
        return url
    }

    // MARK: - Writing

    /// Persists one prepared image and returns the ref to store in the message.
    /// Synchronous because the send path needs the ref before it can append the
    /// user message; the write itself is one small atomic file operation.
    func write(_ image: PreparedImage, sessionId: UUID) -> ImageAttachmentRef? {
        guard let root else { return nil }
        let attachmentId = UUID()
        let extensionName = ImageAttachmentStore.fileExtension(forMimeType: image.mimeType)
        let relativePath = "\(sessionId.uuidString.lowercased())/\(attachmentId.uuidString.lowercased()).\(extensionName)"
        let sessionDirectory = root.appendingPathComponent(sessionId.uuidString.lowercased(), isDirectory: true)
        let destination = sessionDirectory.appendingPathComponent(
            "\(attachmentId.uuidString.lowercased()).\(extensionName)",
            isDirectory: false
        )

        var failure: Error?
        ioQueue.sync {
            do {
                try fileManager.createDirectory(at: sessionDirectory, withIntermediateDirectories: true)
                try image.data.write(to: destination, options: .atomic)
            } catch {
                failure = error
            }
        }
        if let failure {
            print("⚠️ [ImageAttachmentStore] could not persist \(image.fileName): \(failure.localizedDescription)")
            return nil
        }

        return ImageAttachmentRef(
            id: attachmentId,
            fileName: image.fileName,
            pixelWidth: image.pixelWidth,
            pixelHeight: image.pixelHeight,
            byteCount: image.data.count,
            sha256: image.sha256,
            relativePath: relativePath,
            mimeType: image.mimeType
        )
    }

    // MARK: - Reading

    func data(for ref: ImageAttachmentRef) -> Data? {
        guard let url = absoluteURL(for: ref) else { return nil }
        return try? Data(contentsOf: url, options: .mappedIfSafe)
    }

    func exists(_ ref: ImageAttachmentRef) -> Bool {
        guard let url = absoluteURL(for: ref) else { return false }
        return fileManager.fileExists(atPath: url.path)
    }

    /// Rebuilds a composer-ready image from a persisted ref. Used when a queued
    /// prompt is pulled back into the composer or sent immediately: the bytes are
    /// already on disk, so nothing needs decoding again.
    func preparedImage(for ref: ImageAttachmentRef) -> PreparedImage? {
        guard let data = data(for: ref) else { return nil }
        return PreparedImage(
            data: data,
            mimeType: ref.mimeType,
            fileName: ref.fileName,
            pixelWidth: ref.pixelWidth,
            pixelHeight: ref.pixelHeight,
            sha256: ref.sha256
        )
    }

    /// Rebuilds every readable image behind a queued prompt's refs. Reading a
    /// handful of megabytes belongs off the main actor; callers wrap this in a
    /// detached task.
    func preparedImages(for refs: [ImageAttachmentRef]) -> [PreparedImage] {
        refs.compactMap { preparedImage(for: $0) }
    }

    // MARK: - Deletion

    /// Removes every asset directory whose conversation is gone.
    func delete(sessionIds: [UUID]) {
        guard let root, !sessionIds.isEmpty else { return }
        let directories = sessionIds.map { root.appendingPathComponent($0.uuidString.lowercased(), isDirectory: true) }
        ioQueue.async {
            let fileManager = FileManager()
            for directory in directories {
                guard fileManager.fileExists(atPath: directory.path) else { continue }
                do {
                    try fileManager.removeItem(at: directory)
                } catch {
                    print("⚠️ [ImageAttachmentStore] could not remove \(directory.lastPathComponent): \(error.localizedDescription)")
                }
            }
        }
    }

    /// Drops specific assets, used when a send that had already persisted its
    /// images is refused before any message references them.
    func remove(_ refs: [ImageAttachmentRef]) {
        guard !refs.isEmpty else { return }
        let urls = refs.compactMap { absoluteURL(for: $0) }
        ioQueue.async {
            let fileManager = FileManager()
            for url in urls {
                try? fileManager.removeItem(at: url)
            }
        }
    }

    /// Removes assets that no conversation references anymore. `liveSessionIds`
    /// bounds the walk (a directory with no live conversation is removed whole,
    /// even if its files were not enumerated), and `liveRelativePaths` keeps the
    /// files an existing message still renders.
    func prune(liveSessionIds: Set<UUID>, liveRelativePaths: Set<String>) {
        guard let root else { return }
        ioQueue.async {
            let fileManager = FileManager()
            guard let sessionDirectories = try? fileManager.contentsOfDirectory(
                at: root,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            ) else { return }

            for directory in sessionDirectories {
                guard let sessionId = UUID(uuidString: directory.lastPathComponent) else { continue }
                guard liveSessionIds.contains(sessionId) else {
                    try? fileManager.removeItem(at: directory)
                    continue
                }
                let files = (try? fileManager.contentsOfDirectory(
                    at: directory,
                    includingPropertiesForKeys: nil,
                    options: [.skipsHiddenFiles]
                )) ?? []
                for file in files {
                    let relativePath = "\(sessionId.uuidString.lowercased())/\(file.lastPathComponent)"
                    if !liveRelativePaths.contains(relativePath) {
                        try? fileManager.removeItem(at: file)
                    }
                }
            }
        }
    }

    // MARK: - Helpers

    /// Blocks until queued deletions/prunes finish. Used by tests, and available
    /// if a caller ever needs a synchronous barrier.
    func flushPendingIO() {
        ioQueue.sync {}
    }

    nonisolated static func fileExtension(forMimeType mimeType: String) -> String {
        mimeType == "image/png" ? "png" : "jpg"
    }
}
