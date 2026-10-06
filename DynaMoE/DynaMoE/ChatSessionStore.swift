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
import Combine
import CryptoKit

/// Keeps conversations on disk across launches. Durable data lives in
/// ~/Library/Application Support so it is backed up and never purged like the
/// regenerable index cache in ~/Library/Caches.
final class ChatSessionStore: ObservableObject {
    static let shared = ChatSessionStore()

    /// Stamped into every file so a future format change can migrate old
    /// conversations instead of silently discarding them.
    nonisolated static let currentSchemaVersion = 1

    /// `nil` only when Application Support itself could not be resolved, in which
    /// case there is no durable location to fall back to and the store refuses to
    /// persist rather than silently writing durable history to a purgeable
    /// temporary directory. A resolved-but-uncreatable directory is kept so the
    /// write paths can retry creation; availability, not this URL, tracks whether
    /// that succeeded.
    private let directory: URL?
    private let ioQueue = DispatchQueue(label: "com.dynamoe.chatsessionstore.io", qos: .utility)

    /// True while the store has a usable directory. Starts from an up-front
    /// create check in `init` and flips out of `ioQueue` when a write or
    /// directory creation later fails (or recovers), so callers can surface the
    /// real state instead of assuming conversations are being saved.
    @Published private(set) var isPersistenceAvailable: Bool

    /// Collision-resistant digest of the canonical bytes last **successfully
    /// written** per conversation. A cryptographic digest (not `hashValue`, whose
    /// 64 bits could collide and silently skip a changed conversation) is what
    /// makes "digest matches" a safe stand-in for "bytes unchanged". Only updated
    /// after a write actually lands, so a failed write (disk full, permissions)
    /// is retried on the next save instead of being treated as already durable.
    /// Access is serialized on `ioQueue` and never touched from the main actor,
    /// which is what makes the unchecked isolation safe.
    nonisolated(unsafe) private var writtenSignatures: [UUID: SHA256Digest] = [:]

    /// Conversations this build is allowed to rewrite or delete: the ones it has
    /// loaded or written. Files for newer, unsupported schema versions (or
    /// otherwise unrecognized files) never enter this set, so a save pass leaves
    /// them untouched rather than downgrading or deleting them. Serialized on
    /// `ioQueue` like `writtenSignatures`.
    nonisolated(unsafe) private var managedIds: Set<UUID> = []

    /// Monotonic id of the most recently *submitted* availability report. Bumped
    /// on `ioQueue`, so it reflects submission order; the main actor compares it
    /// against `appliedAvailabilitySequence` and ignores an older report.
    nonisolated(unsafe) private var availabilitySequence: UInt64 = 0

    /// Highest availability-report id applied on the main actor. Guards against
    /// the unordered delivery of the `Task { @MainActor }` hops, which would
    /// otherwise let a failure arrive after the succeeding retry and restore the
    /// stale warning.
    private var appliedAvailabilitySequence: UInt64 = 0

    init(directory: URL? = nil, fileManager: FileManager = .default) {
        let resolved = directory ?? ChatSessionStore.defaultDirectory(using: fileManager)
        // Retain the resolved URL even if the up-front create check fails: every
        // write path retries `createDirectory` and re-reports availability, so a
        // transient launch-time permission or filesystem failure can recover
        // without a relaunch. `prepared` only seeds the initial UI state.
        self.directory = resolved
        self.isPersistenceAvailable = resolved.flatMap { ChatSessionStore.preparedDirectory($0) } != nil
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

    /// Confirms the sessions directory actually exists and is creatable before it
    /// is reported as usable.
    nonisolated static func preparedDirectory(_ directory: URL) -> URL? {
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            return directory
        } catch {
            print("⚠️ [ChatSessionStore] could not create the sessions directory at \(directory.path), so chat history will not be persisted: \(error.localizedDescription)")
            return nil
        }
    }

    /// Mirrors a write-path outcome onto `isPersistenceAvailable` on the main
    /// actor. No-ops when the value is unchanged.
    ///
    /// Every caller runs on the serial `ioQueue`, but the `Task { @MainActor }`
    /// hops are not ordered, so results carry a sequence id and only the newest
    /// one is applied. Without that, a failure followed by a successful retry
    /// could apply `true` first and stale `false` last, leaving the warning up
    /// after persistence recovered.
    nonisolated private func reportPersistenceAvailability(_ available: Bool) {
        availabilitySequence &+= 1
        let sequence = availabilitySequence
        Task { @MainActor [weak self] in
            guard let self, sequence > self.appliedAvailabilitySequence else { return }
            self.appliedAvailabilitySequence = sequence
            guard self.isPersistenceAvailable != available else { return }
            self.isPersistenceAvailable = available
        }
    }

    // MARK: - Loading

    /// Synchronous load. Convenient for tests, but blocks the caller until every
    /// file has been read and decoded, so app code should prefer
    /// `loadSessionsAsync`. Returns `nil` when the history cannot be read (no
    /// usable directory, or the directory could not be listed), which callers
    /// must treat as a retryable failure rather than an empty history.
    func loadSessions() -> [ChatSession]? {
        guard let directory else { return nil }
        return ioQueue.sync { [self] in
            loadSessionsLocked(in: directory)
        }
    }

    /// Loads conversations without blocking the caller. The read, decode, and
    /// signature work still runs on `ioQueue`; only the result hops back.
    /// Returns `nil` when the history cannot be read; see `loadSessions`.
    func loadSessionsAsync() async -> [ChatSession]? {
        guard let directory else { return nil }
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
    ///
    /// Returns `nil` when the directory itself cannot be listed. That is not an
    /// empty history: reporting it as one would make the caller replace the
    /// visible conversations with nothing and stop retrying, so the failure is
    /// surfaced (and persisted as unavailable) and the caller can retry.
    nonisolated private func loadSessionsLocked(in directory: URL) -> [ChatSession]? {
        let fm = FileManager()
        // Retry directory creation before listing, exactly as the save path does.
        // A launch-time creation failure otherwise makes every retry enumerate a
        // still-missing directory and fail again, so restoring access while the
        // app stays open could never recover the history.
        do {
            try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        } catch {
            print("⚠️ [ChatSessionStore] could not open the sessions directory for reading: \(error.localizedDescription)")
            reportPersistenceAvailability(false)
            return nil
        }
        let urls: [URL]
        do {
            urls = try fm.contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            )
        } catch {
            print("⚠️ [ChatSessionStore] could not read the sessions directory: \(error.localizedDescription)")
            reportPersistenceAvailability(false)
            return nil
        }
        // A readable directory is the store working again. Report it here rather
        // than relying on a later save: the non-empty load path never saves, so a
        // recovered directory would otherwise leave the warning up until the user
        // happened to edit a conversation.
        reportPersistenceAvailability(true)

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
                decoded.append(ChatSessionStore.normalizedForRestore(envelope.session))
                managedIds.insert(envelope.session.id)
                if writtenSignatures[envelope.session.id] == nil,
                   let encoded = try? encoder.encode(envelope) {
                    writtenSignatures[envelope.session.id] = SHA256.hash(data: encoded)
                }
            } else if let bare = try? decoder.decode(ChatSession.self, from: data) {
                decoded.append(ChatSessionStore.normalizedForRestore(bare))
                managedIds.insert(bare.id)
            }
        }

        return decoded
            .map { (session: $0, activity: $0.lastActivityAt) }
            .sorted { $0.activity > $1.activity }
            .map(\.session)
    }

    /// Whether this build can faithfully round-trip a schema version. Newer
    /// versions may carry fields we would drop on rewrite, so they are refused;
    /// version 0 and negatives are not valid stamped versions either, so a
    /// malformed envelope carrying one is preserved as unsupported rather than
    /// ingested and rewritten.
    nonisolated static func isSupportedSchema(_ version: Int) -> Bool {
        version >= 1 && version <= currentSchemaVersion
    }

    /// Resets runtime-only state on a conversation decoded from disk. A chat
    /// saved mid-turn can carry a tool call still marked `running` or
    /// `awaitingApproval` and a transient thinking flag. Restoring those verbatim
    /// makes the next launch show a call permanently stuck in progress (with an
    /// approval action that has no continuation to finish it) and a message that
    /// looks like it is still thinking, so interrupted calls become a terminal
    /// error and transient flags clear.
    nonisolated static func normalizedForRestore(_ session: ChatSession) -> ChatSession {
        var restored = session
        // Transient runtime-only messages (the compaction progress bubble, which
        // streams an uncommitted summary into the array) are dropped entirely:
        // clearing their flags would still leave a partial summary on screen as
        // an ordinary settled system message, and compaction cannot resume.
        restored.messages.removeAll { $0.isTransient == true }
        for idx in restored.messages.indices {
            restored.messages[idx].isThinking = false
            restored.messages[idx].prefillStatus = nil
            guard var calls = restored.messages[idx].toolCalls, !calls.isEmpty else { continue }
            for cIdx in calls.indices {
                switch calls[cIdx].status {
                case .running, .awaitingApproval:
                    calls[cIdx].status = .error
                    calls[cIdx].error = "Interrupted when the app closed."
                    calls[cIdx].output = nil
                case .success, .error, .rejected:
                    break
                }
            }
            restored.messages[idx].toolCalls = calls
        }
        return restored
    }

    // MARK: - Saving

    /// Writes every conversation that changed and removes files for
    /// conversations no longer present. The caller debounces this, so it is
    /// safe to call repeatedly; unchanged conversations are skipped.
    ///
    /// Serialization and the compare-write-commit cycle all run on `ioQueue`, so
    /// large histories never encode on the main actor. A signature is recorded
    /// only after its write succeeds.
    ///
    /// `cleanOrphans` defaults to true. Pass `false` to write conversations
    /// without the deletion pass — needed when persisting in-memory sessions
    /// before a load has populated `managedIds`, since cleanup would otherwise
    /// treat every not-yet-loaded file on disk as a deleted conversation and
    /// remove it.
    func saveAll(_ sessions: [ChatSession], cleanOrphans: Bool = true) {
        guard let directory else { return }
        let snapshot = sessions
        ioQueue.async { [self] in
            let fm = FileManager()
            do {
                try fm.createDirectory(at: directory, withIntermediateDirectories: true)
            } catch {
                print("⚠️ [ChatSessionStore] could not open the sessions directory: \(error.localizedDescription)")
                reportPersistenceAvailability(false)
                return
            }

            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            let liveIds = Set(snapshot.map(\.id))
            // Net outcome for the pass, reported once every write and cleanup has
            // been attempted. Any failure wins: if even one conversation is not
            // durable the pass must not report healthy just because another write
            // succeeded, and a later success must not hide an earlier failure (or
            // vice versa). A pass that only re-runs cleanup can restore
            // availability too, so `removedAny` counts alongside `wroteAny`.
            var wroteAny = false
            var removedAny = false
            var enumeratedAny = false
            var failedAny = false

            for session in snapshot {
                let data: Data
                do {
                    data = try encoder.encode(
                        PersistedChatSession(schemaVersion: ChatSessionStore.currentSchemaVersion, session: session)
                    )
                } catch {
                    // An unencodable conversation (e.g. a non-finite metric) never
                    // reaches disk, so the pass must not report healthy.
                    failedAny = true
                    print("⚠️ [ChatSessionStore] failed to encode session \(session.id): \(error.localizedDescription)")
                    continue
                }
                let signature = SHA256.hash(data: data)
                let url = directory.appendingPathComponent("\(session.id.uuidString).json", isDirectory: false)
                // Only skip a write when the bytes are unchanged *and* a regular
                // file is still on disk: a cached signature alone must not leave
                // the conversation absent if the file (or the whole directory) was
                // removed while the app was running. `fileExists` alone is not
                // enough because it is also true for a directory, which would let
                // a replaced path report the conversation as durable even though
                // it cannot be read back next launch, so confirm the path is not
                // a directory before trusting it.
                var isDirectory: ObjCBool = false
                if writtenSignatures[session.id] == signature,
                   fm.fileExists(atPath: url.path, isDirectory: &isDirectory),
                   !isDirectory.boolValue {
                    managedIds.insert(session.id)
                    continue
                }
                do {
                    try data.write(to: url, options: .atomic)
                    writtenSignatures[session.id] = signature
                    managedIds.insert(session.id)
                    wroteAny = true
                } catch {
                    // Leave no signature so the conversation is retried rather
                    // than assumed durable for the rest of the process.
                    writtenSignatures[session.id] = nil
                    failedAny = true
                    print("⚠️ [ChatSessionStore] failed to persist session \(session.id): \(error.localizedDescription)")
                }
            }

            if cleanOrphans {
                do {
                    let urls = try fm.contentsOfDirectory(
                        at: directory,
                        includingPropertiesForKeys: nil,
                        options: [.skipsHiddenFiles]
                    )
                    enumeratedAny = true
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
                            removedAny = true
                        } catch {
                            // A failed cleanup is not durable state either: surface it so
                            // the warning shows and the removal is retried, and let a
                            // successful retry restore availability below.
                            failedAny = true
                            print("⚠️ [ChatSessionStore] failed to remove session \(id): \(error.localizedDescription)")
                        }
                    }
                } catch {
                    // If the directory cannot be listed, deleted conversations are never
                    // removed and would reappear later, so the pass must not report
                    // healthy. `managedIds` keeps them queued for the next attempt.
                    failedAny = true
                    print("⚠️ [ChatSessionStore] could not enumerate the sessions directory for cleanup: \(error.localizedDescription)")
                }
            }

            if failedAny {
                reportPersistenceAvailability(false)
            } else if wroteAny || removedAny || enumeratedAny {
                reportPersistenceAvailability(true)
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
    /// - Protected conversations are always kept even if they fall outside the
    ///   most-recent window, so the user is never yanked out of an open chat and
    ///   an in-flight generation can still append to its session.
    /// - Returns the originals in their existing order, split into survivors
    ///   and the removed tail.
    nonisolated static func retentionPlan(
        sessions: [ChatSession],
        limit: Int?,
        protectedIds: Set<UUID>
    ) -> (kept: [ChatSession], removed: [ChatSession]) {
        guard let limit, limit > 0, sessions.count > limit else {
            return (sessions, [])
        }
        let mostRecent = sessions
            .map { (session: $0, activity: $0.lastActivityAt) }
            .sorted { $0.activity > $1.activity }
        var keepIds = Set(mostRecent.prefix(limit).map(\.session.id))
        keepIds.formUnion(protectedIds)
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
