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
    nonisolated static let currentSchemaVersion = 1

    /// `nil` when Application Support could not be resolved. In that case the
    /// store refuses to persist rather than silently writing durable history to
    /// a purgeable temporary directory.
    private let directory: URL?
    private let ioQueue = DispatchQueue(label: "com.dynamoe.chatsessionstore.io", qos: .utility)

    /// Signature of the bytes last **successfully written** per conversation.
    /// Only updated after a write actually lands, so a failed write (disk full,
    /// permissions) is retried on the next save instead of being treated as
    /// already durable. Access is serialized on `ioQueue` and never touched from
    /// the main actor, which is what makes the unchecked isolation safe.
    nonisolated(unsafe) private var writtenSignatures: [UUID: Int] = [:]

    /// Conversations this build is allowed to rewrite or delete: the ones it has
    /// loaded or written. Files for newer, unsupported schema versions (or
    /// otherwise unrecognized files) never enter this set, so a save pass leaves
    /// them untouched rather than downgrading or deleting them. Serialized on
    /// `ioQueue` like `writtenSignatures`.
    nonisolated(unsafe) private var managedIds: Set<UUID> = []

    init(directory: URL? = nil, fileManager: FileManager = .default) {
        self.directory = directory ?? ChatSessionStore.defaultDirectory(using: fileManager)
    }

    // MARK: - Locations

    nonisolated static func defaultDirectory(using fileManager: FileManager = .default) -> URL? {
        do {
            let base = try fileManager.url(
                for: .applicationSupportDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: true
            )
            return base.appendingPathComponent("DynaMoE/sessions", isDirectory: true)
        } catch {
            print("⚠️ [ChatSessionStore] Application Support is unavailable, so chat history will not be persisted: \(error.localizedDescription)")
            return nil
        }
    }

    /// False when the store has nowhere durable to write. Callers can surface
    /// this instead of assuming conversations are being saved.
    var isPersistenceAvailable: Bool { directory != nil }

    // MARK: - Loading

    /// Synchronous load. Convenient for tests, but blocks the caller until every
    /// file has been read and decoded, so app code should prefer
    /// `loadSessionsAsync`.
    func loadSessions() -> [ChatSession] {
        guard let directory else { return [] }
        return ioQueue.sync { [self] in
            loadSessionsLocked(in: directory)
        }
    }

    /// Loads conversations without blocking the caller. The read, decode, and
    /// signature work still runs on `ioQueue`; only the result hops back.
    func loadSessionsAsync() async -> [ChatSession] {
        guard let directory else { return [] }
        return await withCheckedContinuation { continuation in
            ioQueue.async { [self] in
                continuation.resume(returning: loadSessionsLocked(in: directory))
            }
        }
    }

    /// Reads every persisted conversation on `ioQueue`, most-recently-active
    /// first to match the sidebar order.
    ///
    /// A versioned conversation's canonical signature is seeded into
    /// `writtenSignatures` so the first debounced save after launch does not
    /// rewrite files whose contents are already up to date. Legacy bare-session
    /// files are deliberately left unsigned so the next save rewrites them into
    /// the versioned envelope instead of leaving them unstamped forever.
    ///
    /// Files stamped with a schema version this build does not understand are
    /// skipped and left on disk untouched, so a newer build's conversations are
    /// never silently downgraded to the current format.
    nonisolated private func loadSessionsLocked(in directory: URL) -> [ChatSession] {
        let fm = FileManager()
        guard let urls = try? fm.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) else { return [] }

        let decoder = JSONDecoder()
        let encoder = JSONEncoder()
        // Canonical (`sortedKeys`) encoding is required for the byte signature to
        // be stable: JSONEncoder otherwise emits ChatSession's keys in an
        // unstable order, making every save look like a change.
        encoder.outputFormatting = [.sortedKeys]

        var decoded: [ChatSession] = []
        for url in urls where url.pathExtension == "json" {
            guard let data = try? Data(contentsOf: url) else { continue }
            if let envelope = try? decoder.decode(PersistedChatSession.self, from: data) {
                guard ChatSessionStore.isSupportedSchema(envelope.schemaVersion) else { continue }
                decoded.append(envelope.session)
                managedIds.insert(envelope.session.id)
                if writtenSignatures[envelope.session.id] == nil,
                   let encoded = try? encoder.encode(envelope) {
                    writtenSignatures[envelope.session.id] = encoded.hashValue
                }
            } else if let bare = try? decoder.decode(ChatSession.self, from: data) {
                decoded.append(bare)
                managedIds.insert(bare.id)
            }
        }

        return decoded
            .map { (session: $0, activity: $0.lastActivityAt) }
            .sorted { $0.activity > $1.activity }
            .map(\.session)
    }

    /// Whether this build can faithfully round-trip a schema version. Newer
    /// versions may carry fields we would drop on rewrite, so they are refused.
    nonisolated static func isSupportedSchema(_ version: Int) -> Bool {
        version <= currentSchemaVersion
    }

    // MARK: - Saving

    /// Writes every conversation that changed and removes files for
    /// conversations no longer present. The caller debounces this, so it is
    /// safe to call repeatedly; unchanged conversations are skipped.
    ///
    /// Serialization and the compare-write-commit cycle all run on `ioQueue`, so
    /// large histories never encode on the main actor. A signature is recorded
    /// only after its write succeeds.
    func saveAll(_ sessions: [ChatSession]) {
        guard let directory else { return }
        let snapshot = sessions
        ioQueue.async { [self] in
            let fm = FileManager()
            try? fm.createDirectory(at: directory, withIntermediateDirectories: true)

            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            let liveIds = Set(snapshot.map(\.id))

            for session in snapshot {
                guard let data = try? encoder.encode(
                    PersistedChatSession(schemaVersion: ChatSessionStore.currentSchemaVersion, session: session)
                ) else { continue }
                let signature = data.hashValue
                if writtenSignatures[session.id] == signature {
                    managedIds.insert(session.id)
                    continue
                }
                let url = directory.appendingPathComponent("\(session.id.uuidString).json", isDirectory: false)
                do {
                    try data.write(to: url, options: .atomic)
                    writtenSignatures[session.id] = signature
                    managedIds.insert(session.id)
                } catch {
                    // Leave no signature so the conversation is retried rather
                    // than assumed durable for the rest of the process.
                    writtenSignatures[session.id] = nil
                    print("⚠️ [ChatSessionStore] failed to persist session \(session.id): \(error.localizedDescription)")
                }
            }

            guard let urls = try? fm.contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            ) else { return }
            for url in urls where url.pathExtension == "json" {
                // Only ever remove conversations this build loaded or wrote.
                // Files from a newer schema (or otherwise unrecognized) are not
                // managed and must survive untouched.
                guard let id = UUID(uuidString: url.deletingPathExtension().lastPathComponent),
                      managedIds.contains(id),
                      !liveIds.contains(id) else { continue }
                // Keep the state only if the file is actually gone, so a failed
                // removal is retried on the next save.
                do {
                    try fm.removeItem(at: url)
                    writtenSignatures[id] = nil
                    managedIds.remove(id)
                } catch {
                    print("⚠️ [ChatSessionStore] failed to remove session \(id): \(error.localizedDescription)")
                }
            }
        }
    }

    /// Deletes the given conversations from disk immediately.
    func delete(sessionIds: [UUID]) {
        guard !sessionIds.isEmpty, let directory else { return }
        let urls = sessionIds.map { (id: $0, url: directory.appendingPathComponent("\($0.uuidString).json", isDirectory: false)) }
        ioQueue.async { [self] in
            let fm = FileManager()
            for entry in urls {
                // Keep the state only if the file is actually gone, so a failed
                // removal is retried on the next save.
                do {
                    try fm.removeItem(at: entry.url)
                    writtenSignatures[entry.id] = nil
                    managedIds.remove(entry.id)
                } catch {
                    print("⚠️ [ChatSessionStore] failed to remove session \(entry.id): \(error.localizedDescription)")
                }
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
    nonisolated static func retentionPlan(
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

/// Versioned on-disk envelope for a single conversation. Internal rather than
/// private so tests can construct envelopes of arbitrary schema versions.
nonisolated struct PersistedChatSession: Codable, Sendable {
    let schemaVersion: Int
    let session: ChatSession
}
