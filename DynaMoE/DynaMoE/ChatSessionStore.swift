//
//  ChatSessionStore.swift
//  DynaMoE
//
//  Phase 2: Chat Session Persistence
//  - One versioned JSON file per conversation, stored under
//    ~/Library/Application Support/DynaMoE/sessions/
//  - Atomic writes with change detection so a debounced save pass only rewrites
//    conversations whose content actually changed during streaming.
//  - Orphan cleanup removes files for conversations the user deleted.
//

import Foundation

/// Keeps conversations on disk across launches. Durable data lives in
/// ~/Library/Application Support so it is backed up and never purged like the
/// regenerable index cache in ~/Library/Caches.
final class ChatSessionStore {
    static let shared = ChatSessionStore()

    /// Stamped into every file so a future format change can migrate old
    /// conversations instead of silently discarding them.
    static let currentSchemaVersion = 1

    private let directory: URL
    private let fileManager: FileManager
    private let ioQueue = DispatchQueue(label: "com.dynamoe.chatsessionstore.io", qos: .utility)
    /// Canonical (`sortedKeys`) encoding is required for the byte signature to be
    /// stable: `JSONEncoder` otherwise emits `ChatSession`'s keys in an unstable
    /// order, which would make every save look like a change.
    private let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }()
    private let decoder = JSONDecoder()

    /// Signature of the bytes last **successfully written** per conversation.
    /// Only updated after a write actually lands, so a failed write (disk full,
    /// permissions) is retried on the next save instead of being treated as
    /// already durable. Access is serialized on `ioQueue` and never touched from
    /// the main actor, which is what makes the unchecked isolation safe.
    nonisolated(unsafe) private var writtenSignatures: [UUID: Int] = [:]

    init(directory: URL? = nil, fileManager: FileManager = .default) {
        self.fileManager = fileManager
        self.directory = directory ?? ChatSessionStore.defaultDirectory(using: fileManager)
    }

    // MARK: - Locations

    nonisolated static func defaultDirectory(using fileManager: FileManager = .default) -> URL {
        let base = (try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )) ?? fileManager.temporaryDirectory
        return base.appendingPathComponent("DynaMoE/sessions", isDirectory: true)
    }

    private func fileURL(for id: UUID) -> URL {
        directory.appendingPathComponent("\(id.uuidString).json", isDirectory: false)
    }

    private func enumerateSessionFiles() -> [URL] {
        guard let urls = try? fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) else { return [] }
        return urls.filter { $0.pathExtension == "json" }
    }

    // MARK: - Loading

    /// Reads every persisted conversation, most-recently-active first to match
    /// the sidebar order.
    ///
    /// A versioned conversation's canonical signature is seeded into
    /// `writtenSignatures` so the first debounced save after launch does not
    /// rewrite files whose contents are already up to date. Legacy bare-session
    /// files are deliberately left unsigned so the next save rewrites them into
    /// the versioned envelope instead of leaving them unstamped forever.
    func loadSessions() -> [ChatSession] {
        var decoded: [(session: ChatSession, signature: Int?)] = []
        for url in enumerateSessionFiles() {
            guard let data = try? Data(contentsOf: url) else { continue }
            if let envelope = try? decoder.decode(PersistedChatSession.self, from: data) {
                decoded.append((envelope.session, canonicalSignature(for: envelope.session)))
            } else if let bare = try? decoder.decode(ChatSession.self, from: data) {
                decoded.append((bare, nil))
            }
        }

        let signatures = decoded.compactMap { pair -> (UUID, Int)? in
            guard let signature = pair.signature else { return nil }
            return (pair.session.id, signature)
        }
        ioQueue.sync { [self] in
            for (id, signature) in signatures where writtenSignatures[id] == nil {
                writtenSignatures[id] = signature
            }
        }

        return decoded
            .map(\.session)
            .map { (session: $0, activity: $0.lastActivityAt) }
            .sorted { $0.activity > $1.activity }
            .map(\.session)
    }

    private func canonicalSignature(for session: ChatSession) -> Int? {
        guard let data = try? encoder.encode(
            PersistedChatSession(schemaVersion: ChatSessionStore.currentSchemaVersion, session: session)
        ) else { return nil }
        return data.hashValue
    }

    // MARK: - Saving

    /// Writes every conversation that changed and removes files for
    /// conversations no longer present. The caller debounces this, so it is
    /// safe to call repeatedly; unchanged conversations are skipped.
    ///
    /// Encoding happens here (the `ChatSession` conformance is main-actor
    /// isolated); the actual compare-write-commit cycle runs on `ioQueue`, where
    /// a signature is recorded only after its write succeeds.
    func saveAll(_ sessions: [ChatSession]) {
        var payloads: [(id: UUID, url: URL, data: Data, signature: Int)] = []
        for session in sessions {
            guard let data = try? encoder.encode(
                PersistedChatSession(schemaVersion: ChatSessionStore.currentSchemaVersion, session: session)
            ) else { continue }
            payloads.append((session.id, fileURL(for: session.id), data, data.hashValue))
        }

        let directory = self.directory
        let liveIds = Set(sessions.map(\.id))

        ioQueue.async { [self] in
            let fm = FileManager()
            try? fm.createDirectory(at: directory, withIntermediateDirectories: true)

            for payload in payloads {
                if writtenSignatures[payload.id] == payload.signature { continue }
                do {
                    try payload.data.write(to: payload.url, options: .atomic)
                    writtenSignatures[payload.id] = payload.signature
                } catch {
                    // Leave no signature so the conversation is retried rather
                    // than assumed durable for the rest of the process.
                    writtenSignatures[payload.id] = nil
                    print("⚠️ [ChatSessionStore] failed to persist session \(payload.id): \(error.localizedDescription)")
                }
            }

            guard let urls = try? fm.contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            ) else { return }
            for url in urls where url.pathExtension == "json" {
                guard let id = UUID(uuidString: url.deletingPathExtension().lastPathComponent),
                      !liveIds.contains(id) else { continue }
                try? fm.removeItem(at: url)
                writtenSignatures[id] = nil
            }
        }
    }

    /// Deletes the given conversations from disk immediately.
    func delete(sessionIds: [UUID]) {
        guard !sessionIds.isEmpty else { return }
        let urls = sessionIds.map { fileURL(for: $0) }
        ioQueue.async { [self] in
            let fm = FileManager()
            for (index, url) in urls.enumerated() {
                try? fm.removeItem(at: url)
                writtenSignatures[sessionIds[index]] = nil
            }
        }
    }

    /// Blocks until queued disk writes finish. Used by tests (and available if a
    /// caller ever needs a synchronous barrier).
    func flushPendingIO() {
        ioQueue.sync {}
    }

    // MARK: - Retention

    /// Decides which conversations survive a retention pass.
    ///
    /// - `limit == nil` means "never auto-delete"; everything is kept.
    /// - Recency comes from `lastActivityAt`, not `updatedAt`, so a conversation
    ///   used recently is not pruned just because it was created long ago.
    /// - The active conversation is always kept even if it falls outside the
    ///   most-recent window, so the user is never yanked out of an open chat.
    /// - Returns the originals in their existing order, split into survivors
    ///   and the removed tail.
    static func retentionPlan(
        sessions: [ChatSession],
        limit: Int?,
        protectedId: UUID?
    ) -> (kept: [ChatSession], removed: [ChatSession]) {
        guard let limit, limit > 0, sessions.count > limit else {
            return (sessions, [])
        }
        let mostRecent = sessions
            .map { (session: $0, activity: $0.lastActivityAt) }
            .sorted { $0.activity > $1.activity }
        var keepIds = Set(mostRecent.prefix(limit).map(\.session.id))
        if let protectedId {
            keepIds.insert(protectedId)
        }
        let kept = sessions.filter { keepIds.contains($0.id) }
        let removed = sessions.filter { !keepIds.contains($0.id) }
        return (kept, removed)
    }
}

/// Versioned on-disk envelope for a single conversation.
private struct PersistedChatSession: Codable {
    let schemaVersion: Int
    let session: ChatSession
}
