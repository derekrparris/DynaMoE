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
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    /// Signature of the bytes last written per conversation. A debounced save
    /// pass compares against this so streaming updates to one conversation do
    /// not rewrite every other file on disk.
    private var writtenSignatures: [UUID: Int] = [:]

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

    private func ensureDirectory() {
        try? fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
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

    /// Reads every persisted conversation, most-recently-updated first to match
    /// the sidebar order.
    func loadSessions() -> [ChatSession] {
        var sessions: [ChatSession] = []
        for url in enumerateSessionFiles() {
            guard let data = try? Data(contentsOf: url) else { continue }
            if let envelope = try? decoder.decode(PersistedChatSession.self, from: data) {
                sessions.append(envelope.session)
            } else if let bare = try? decoder.decode(ChatSession.self, from: data) {
                sessions.append(bare)
            }
        }
        return sessions.sorted { $0.updatedAt > $1.updatedAt }
    }

    // MARK: - Saving

    /// Writes every conversation that changed and removes files for
    /// conversations no longer present. The caller debounces this, so it is
    /// safe to call repeatedly; unchanged conversations are skipped.
    @discardableResult
    func saveAll(_ sessions: [ChatSession]) -> [UUID] {
        ensureDirectory()
        let liveIds = Set(sessions.map(\.id))

        var payloads: [(url: URL, data: Data)] = []
        for session in sessions {
            guard let data = try? encoder.encode(
                PersistedChatSession(schemaVersion: ChatSessionStore.currentSchemaVersion, session: session)
            ) else { continue }
            let signature = data.hashValue
            guard writtenSignatures[session.id] != signature else { continue }
            writtenSignatures[session.id] = signature
            payloads.append((fileURL(for: session.id), data))
        }

        let orphanURLs = orphanURLs(keeping: liveIds)

        if payloads.isEmpty && orphanURLs.isEmpty { return [] }

        ioQueue.async {
            let fm = FileManager()
            for payload in payloads {
                try? payload.data.write(to: payload.url, options: .atomic)
            }
            for url in orphanURLs {
                try? fm.removeItem(at: url)
            }
        }

        return orphanURLs.compactMap { UUID(uuidString: $0.deletingPathExtension().lastPathComponent) }
    }

    /// Deletes the given conversations from disk immediately.
    func delete(sessionIds: [UUID]) {
        let urls = sessionIds.map { fileURL(for: $0) }
        for id in sessionIds {
            writtenSignatures[id] = nil
        }
        guard !urls.isEmpty else { return }
        ioQueue.async {
            let fm = FileManager()
            for url in urls {
                try? fm.removeItem(at: url)
            }
        }
    }

    /// Blocks until queued disk writes finish. Used by tests (and available if a
    /// caller ever needs a synchronous barrier).
    func flushPendingIO() {
        ioQueue.sync {}
    }

    private func orphanURLs(keeping liveIds: Set<UUID>) -> [URL] {
        var orphans: [URL] = []
        for url in enumerateSessionFiles() {
            guard let id = UUID(uuidString: url.deletingPathExtension().lastPathComponent) else { continue }
            if !liveIds.contains(id) {
                writtenSignatures[id] = nil
                orphans.append(url)
            }
        }
        return orphans
    }

    // MARK: - Retention

    /// Decides which conversations survive a retention pass.
    ///
    /// - `limit == nil` means "never auto-delete"; everything is kept.
    /// - The active conversation is always kept even if it falls outside the
    ///   most-recent window, so the user is never yanked out of an open chat.
    /// - Returns the originals in their existing order, split into survivors
    ///   and the removed tail.
    nonisolated static func retentionPlan(
        sessions: [ChatSession],
        limit: Int?,
        protectedId: UUID?
    ) -> (kept: [ChatSession], removed: [ChatSession]) {
        guard let limit, limit > 0, sessions.count > limit else {
            return (sessions, [])
        }
        let mostRecent = sessions.sorted { $0.updatedAt > $1.updatedAt }
        var keepIds = Set(mostRecent.prefix(limit).map(\.id))
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
