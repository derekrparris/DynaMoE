//
//  AgentRunStore.swift
//  DynaMoE
//
//  Durable record of every subagent run. See SUBAGENTS.md §10 and §11.
//
//  One JSON file per run under
//  `~/Library/Application Support/DynaMoE/agent-runs/<sessionId>/<runId>.json`.
//  Application Support rather than Caches for the same reason the chat session
//  store lives there: a run's transcript and report are the only record of work
//  that already happened, and regenerating them is impossible.
//

import CryptoKit
import Foundation

/// Persists `AgentRunRecord`s and answers what survived the last launch.
///
/// Writes are serialized off the main actor and skipped when the bytes have not
/// changed, so a caller can persist on every tool boundary without paying for a
/// re-encode of an unchanged run. Reads are synchronous because they run when a
/// view appears or the app launches, never in a generation loop.
nonisolated final class AgentRunStore: @unchecked Sendable {
    static let shared = AgentRunStore(rootURL: AgentRunStore.processRootURL)

    /// The app writes under Application Support. A test run must not: a case
    /// that finishes a run would otherwise leave a record in the operator's
    /// real history, which is the same class of leak the request-log recorder
    /// guards against.
    static var processRootURL: URL {
        if ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
            || ProcessInfo.processInfo.environment["XCTestBundlePath"] != nil {
            return FileManager.default.temporaryDirectory
                .appendingPathComponent("dynamoe-agent-runs-tests", isDirectory: true)
        }
        return defaultRootURL
    }

    /// A run with no parent session still needs a bucket. The directory name is
    /// deliberate rather than a UUID so it reads as "not part of a conversation"
    /// when someone opens the folder.
    static let unscopedSessionDirectory = "unscoped"

    /// Per-entry budgets applied before writing. A run's transcript is unbounded
    /// by design (a model may read a large file), and a store that writes it
    /// verbatim makes one run the size of the repository it inspected.
    nonisolated enum TranscriptBudget {
        static let toolOutput = 16_384
        static let assistantText = 32_768
        static let thinking = 16_384
        static let status = 2_048
        static let task = 16_384
        /// One tool argument (a `file_write` body is the usual offender). The
        /// record is a summary of what happened, not a second copy of the file:
        /// the arguments and `rawArguments` together used to store a written
        /// file twice in a 0644 JSON file.
        static let argumentValue = 2_048
        static let rawArguments = 8_192
        /// A coordinator message or the spawn's conversation excerpt.
        static let message = 8_192
        static let contextSummary = 16_384
        /// A worktree diff is the report's most useful artifact, so it gets a
        /// larger budget than a single tool output. Applying never reads this
        /// copy: the live worktree supplies the full patch.
        static let reportDiff = 65_536
    }

    /// What retention keeps: the newest `keepNewest` runs, plus anything younger
    /// than `keepDays`, and never a run younger than `minimumAgeHours`.
    nonisolated struct RetentionPolicy: Sendable {
        var keepNewest: Int = 200
        var keepDays: Double = 30
        var minimumAgeHours: Double = 24

        static let standard = RetentionPolicy()
    }

    /// The retention the app runs with, from the operator's Settings. A
    /// non-positive or malformed value keeps the default rather than disabling
    /// the only bound on the store.
    ///
    /// The upper bounds are enforced HERE and not only in the Stepper: a
    /// defaults value written outside the UI (a script, a restored backup)
    /// otherwise made "keep 30 days" mean "keep forever", which is the one
    /// direction the bound exists to prevent.
    static func configuredPolicy(defaults: UserDefaults = .standard) -> RetentionPolicy {
        var policy = RetentionPolicy.standard
        if let days = defaults.object(forKey: "dynamoe_subagent_retention_days") as? NSNumber,
           days.doubleValue > 0 {
            policy.keepDays = min(max(days.doubleValue, 1), 365)
        }
        if let runs = defaults.object(forKey: "dynamoe_subagent_retention_runs") as? NSNumber,
           runs.intValue > 0 {
            policy.keepNewest = min(max(runs.intValue, 20), 2_000)
        }
        return policy
    }

    let rootURL: URL
    private let fileManager: FileManager
    private let ioQueue = DispatchQueue(label: "dynamoe.agent-run-store", qos: .utility)
    /// Marks the store's own queue so a synchronous entry point reached from
    /// code already running on it can execute inline. A same-queue
    /// `ioQueue.sync` is a libdispatch abort, not a deadlock: `deleteSession`
    /// re-scoped its survivors from inside `ioQueue.async` and called
    /// `saveImmediately`, which took the process down with
    /// "dispatch_sync called on queue already owned by current thread".
    private let ioQueueMarker = DispatchSpecificKey<UInt8>()
    private static let ioQueueMarkerValue: UInt8 = 1
    private let signatureLock = NSLock()
    private var writtenSignatures: [String: SHA256Digest] = [:]
    /// Runs and sessions a caller deleted. Writes are queued, so a save that
    /// was captured before its delete can land after it and bring the file
    /// back; the tombstone is what makes the delete final. Bounded because only
    /// the most recent deletions can still have an in-flight save behind them.
    private var deletedRunIds: Set<UUID> = []
    private var deletedRunOrder: [UUID] = []
    private var deletedSessionIds: Set<UUID> = []
    private var deletedSessionOrder: [UUID] = []
    /// Bounded because only the most recent deletions can still have an
    /// in-flight save behind them. Sized so one bulk clear (a conversation's
    /// whole run history) cannot evict its OWN earliest tombstones while the
    /// saves captured before it are still queued; a UUID apiece is cheap.
    private let maximumTombstones = 10_000
    /// Terminal records whose report has not been relayed, keyed by session
    /// bucket. Built once from disk (off the main actor at launch) and kept
    /// current by `writeNow`/`delete`. The report relay asks this instead of
    /// decoding every record in a conversation on each send.
    private var unrelayedBySession: [String: Set<UUID>] = [:]
    private var relayIndexBuilt = false
    /// Runs that became relayed while the initial off-lock scan was reading
    /// them. Publishing the scan cannot resurrect one: `markRelayed` lands
    /// from the relay path, which is not on the store's IO queue.
    private var relaySettledDuringBuild: Set<UUID> = []

    /// Posted on the main queue after a record is written or deleted.
    ///
    /// The inspector reloads on `SubagentManager` changes, and a store-only
    /// mutation has none: a coordinator-path apply on a restored run, a relay
    /// flag, retention, a conversation delete. The pane then kept drawing the
    /// pre-apply record (stale APPLY / UNREAD markers) until something else
    /// published. One name, posted at the two places a record lands or leaves.
    static let recordsDidChangeNotification = Notification.Name("dynamoe.agentRunStoreDidChange")

    init(
        rootURL: URL = AgentRunStore.defaultRootURL,
        fileManager: FileManager = .default
    ) {
        self.rootURL = rootURL
        self.fileManager = fileManager
        ioQueue.setSpecific(key: ioQueueMarker, value: Self.ioQueueMarkerValue)
    }

    /// Serializes `work` on the store's IO queue, or runs it inline when the
    /// caller is ALREADY on that queue. Every synchronous entry point goes
    /// through this: libdispatch aborts the process when a serial queue is
    /// `sync`-ed from within itself, and one such re-entrant call shipped.
    private func onQueue<T>(_ work: () -> T) -> T {
        if DispatchQueue.getSpecific(key: ioQueueMarker) != nil {
            return work()
        }
        return ioQueue.sync(execute: work)
    }

    static var defaultRootURL: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Library/Application Support", isDirectory: true)
        return base
            .appendingPathComponent("DynaMoE", isDirectory: true)
            .appendingPathComponent("agent-runs", isDirectory: true)
    }

    // MARK: - Paths

    func directory(for sessionId: UUID?) -> URL {
        rootURL.appendingPathComponent(
            sessionId?.uuidString ?? Self.unscopedSessionDirectory,
            isDirectory: true
        )
    }

    func fileURL(runId: UUID, sessionId: UUID?) -> URL {
        directory(for: sessionId).appendingPathComponent("\(runId.uuidString).json", isDirectory: false)
    }

    // MARK: - Writing

    /// Queues a write. The caller never waits: a run's bookkeeping must not
    /// stall the execution that produced it.
    func save(_ record: AgentRunRecord) {
        let url = fileURL(runId: record.id, sessionId: record.parentSessionId)
        // Bound on the store's queue, not the caller's. `snapshotRecord` runs
        // on the main actor, and truncating a transcript (string slicing and
        // allocation over every entry) there charged the UI for work that is
        // only needed at write time.
        ioQueue.async { [self] in
            writeNow(Self.bounded(record), to: url)
        }
    }

    /// Synchronous write, used by tests and by the app's termination path where
    /// losing the last run is the one outcome worth pausing for. Safe to call
    /// from the store's own queue: it runs inline there instead of
    /// `ioQueue.sync`-ing the queue it is already executing on.
    @discardableResult
    func saveImmediately(_ record: AgentRunRecord) -> Bool {
        let url = fileURL(runId: record.id, sessionId: record.parentSessionId)
        let bounded = Self.bounded(record)
        return onQueue { [self] in
            writeNow(bounded, to: url)
        }
    }

    /// Blocks until every queued write has landed. Tests use it; so does the
    /// terminate handler before it gives the process back.
    func flush() {
        onQueue {}
    }

    /// Reads the CURRENT record from disk, applies the mutation and writes it
    /// back — all on the store's IO queue, so a caller that loaded a record
    /// seconds ago cannot revert fields other writers changed in between. The
    /// apply path used to restore its stale copy and clobber a
    /// `summaryRelayed` write landing mid-apply, which re-injected an already
    /// relayed report at the next send.
    ///
    /// Returns the record as it stands after the mutation, or nil when no
    /// decodable record exists OR the write was refused (a tombstoned session,
    /// a full disk). Reporting success for a refused write let callers treat
    /// the change as landed when it never reached disk.
    ///
    /// The file is located by id, not assumed to be under `sessionId`: a
    /// conversation delete RE-SCOPES an unapplied isolated run to the unscoped
    /// bucket, and a caller still holding the old session id used to mutate a
    /// path the tombstone refused — losing the apply's `appliedAt` forever.
    @discardableResult
    func mutate(runId: UUID, sessionId: UUID?, _ change: (inout AgentRunRecord) -> Void) -> AgentRunRecord? {
        return onQueue { [self] in
            mutateOnQueue(runId: runId, sessionId: sessionId, change)
        }
    }

    /// The same mutation without blocking the caller's actor.
    ///
    /// The apply path runs on the main actor and needs the written record back
    /// before it can continue. `mutate` reaches that answer through
    /// `ioQueue.sync`, which parks the UI for a decode, a full encode, an
    /// fsync and a rename. Awaiting the same work keeps the ordering (the IO
    /// queue is serial) without holding the main thread.
    @discardableResult
    func mutateAsync(
        runId: UUID,
        sessionId: UUID?,
        _ change: @escaping @Sendable (inout AgentRunRecord) -> Void
    ) async -> AgentRunRecord? {
        await withCheckedContinuation { continuation in
            ioQueue.async { [self] in
                continuation.resume(
                    returning: mutateOnQueue(runId: runId, sessionId: sessionId, change)
                )
            }
        }
    }

    /// Caller is on the IO queue. Locates the record the way `mutate` documents
    /// and answers the bounded record it wrote, or nil when it could not.
    private func mutateOnQueue(
        runId: UUID,
        sessionId: UUID?,
        _ change: (inout AgentRunRecord) -> Void
    ) -> AgentRunRecord? {
        let preferred = fileURL(runId: runId, sessionId: sessionId)
        let url = existingRecordURL(runId: runId, preferredSession: sessionId) ?? preferred
        guard var current = decodeRecord(at: url) else { return nil }
        change(&current)
        let bounded = Self.bounded(current)
        guard writeNow(bounded, to: url) else { return nil }
        return bounded
    }

    /// Marks a report delivered: the relay index learns it NOW, the durable
    /// write lands behind the queue.
    ///
    /// The fast path of `unrelayedRunIds` answers from the index without
    /// waiting for the IO queue, so a `mutateLater` that only wrote the field
    /// left one send's worth of window in which the same run was still listed
    /// as owed and its report was injected twice.
    func markRelayed(runId: UUID, sessionId: UUID?) {
        signatureLock.lock()
        removeRelayEverywhereLocked(runId: runId)
        if !relayIndexBuilt {
            // The initial scan runs without the lock. Remember the settled id
            // so publishing its (stale) disk snapshot cannot re-offer it.
            relaySettledDuringBuild.insert(runId)
        }
        signatureLock.unlock()
        mutateLater(runId: runId, sessionId: sessionId) { current in
            current.summaryRelayed = true
        }
    }

    /// Applies a mutation on the store's IO queue WITHOUT waiting for it.
    ///
    /// For a caller that updates a field nobody reads back synchronously.
    /// Marking a report delivered is the case: the blocking `mutate` was an
    /// `ioQueue.sync` plus a durable `fsync` per record on the main actor,
    /// paid before the next frame could draw.
    func mutateLater(
        runId: UUID,
        sessionId: UUID?,
        _ change: @escaping @Sendable (inout AgentRunRecord) -> Void
    ) {
        ioQueue.async { [self] in
            let preferred = fileURL(runId: runId, sessionId: sessionId)
            let url = existingRecordURL(runId: runId, preferredSession: sessionId) ?? preferred
            guard var current = decodeRecord(at: url) else { return }
            change(&current)
            let bounded = Self.bounded(current)
            _ = writeNow(bounded, to: url)
        }
    }

    /// Whether ANY bucket still holds a file for this run, decodable or not.
    ///
    /// The launch sweep needs this distinction: `loadAny` returns nil for a
    /// corrupt or newer-schema record, and the record is the only thing that
    /// says a worktree belongs to unapplied work. Treating "cannot decode" as
    /// "no record" handed the tree to the 14-day clock.
    func hasRecordFile(runId: UUID) -> Bool {
        onQueue {
            let name = "\(runId.uuidString).json"
            guard let directories = try? fileManager.contentsOfDirectory(atPath: rootURL.path) else {
                return false
            }
            for directory in directories {
                let candidate = rootURL
                    .appendingPathComponent(directory, isDirectory: true)
                    .appendingPathComponent(name, isDirectory: false)
                if fileManager.fileExists(atPath: candidate.path) { return true }
            }
            return false
        }
    }

    /// The record's actual file when the preferred bucket no longer holds it.
    ///
    /// A conversation delete re-scopes a run by writing the new copy BEFORE
    /// removing the old one, so a crash between the two leaves both. Choosing
    /// the preferred bucket here while `loadAny` chose the NEWEST made the two
    /// disagree: the next save wrote the stale copy, `prune`/`delete` removed
    /// the other, and the tombstone then refused the survivor — a record no
    /// read could repair. A write must land on the copy a read returns, so this
    /// picks the newest too, with the preferred bucket winning a tie.
    private func existingRecordURL(runId: UUID, preferredSession: UUID?) -> URL? {
        let preferred = fileURL(runId: runId, sessionId: preferredSession)
        let name = "\(runId.uuidString).json"
        guard let directories = try? fileManager.contentsOfDirectory(atPath: rootURL.path) else {
            return fileManager.fileExists(atPath: preferred.path) ? preferred : nil
        }
        var best: (date: Date, url: URL)?
        for directory in directories {
            let candidate = rootURL
                .appendingPathComponent(directory, isDirectory: true)
                .appendingPathComponent(name, isDirectory: false)
            guard fileManager.fileExists(atPath: candidate.path) else { continue }
            let modified = (try? candidate.resourceValues(forKeys: [.contentModificationDateKey]))?
                .contentModificationDate ?? .distantPast
            let isPreferred = candidate.path == preferred.path
            if let current = best {
                if modified > current.date || (modified == current.date && isPreferred) {
                    best = (modified, candidate)
                }
            } else {
                best = (modified, candidate)
            }
        }
        return best?.url
    }

    @discardableResult
    private func writeNow(_ record: AgentRunRecord, to url: URL) -> Bool {
        signatureLock.lock()
        let wasDeleted = deletedRunIds.contains(record.id)
            || (record.parentSessionId.map { deletedSessionIds.contains($0) } ?? false)
        signatureLock.unlock()
        if wasDeleted { return false }

        let directory = url.deletingLastPathComponent()
        do {
            try fileManager.createDirectory(
                at: directory,
                withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700]
            )
        } catch {
            print("⚠️ [AgentRunStore] could not create \(directory.lastPathComponent): \(error.localizedDescription)")
            return false
        }
        let data: Data
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            data = try encoder.encode(record)
        } catch {
            print("⚠️ [AgentRunStore] failed to encode run \(record.id): \(error.localizedDescription)")
            return false
        }
        let signature = SHA256.hash(data: data)
        let key = url.path
        signatureLock.lock()
        let previous = writtenSignatures[key]
        signatureLock.unlock()
        var isDirectory: ObjCBool = false
        if previous == signature,
           fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory),
           !isDirectory.boolValue {
            noteRelayMembership(record)
            return true
        }
        do {
            // A run's record carries the task, the conversation excerpt,
            // thinking and tool output. Application Support is user-readable by
            // default, which is the wrong floor for that content: the helper
            // creates the file 0600 from its first byte (a plain `.atomic`
            // write creates it 0644 and chmods afterwards, a window in which
            // the record is world-readable) and fsyncs before the rename, so a
            // crash cannot leave a truncated record either.
            try Self.writeOwnerOnlyAtomically(data, to: url, fault: currentWriteFault())
            try? fileManager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
            signatureLock.lock()
            writtenSignatures[key] = signature
            signatureLock.unlock()
            noteRelayMembership(record)
            postRecordsDidChange()
            return true
        } catch {
            print("⚠️ [AgentRunStore] failed to write run \(record.id): \(error.localizedDescription)")
            return false
        }
    }

    /// A failure a test can inject into the durable write, so the recovery
    /// paths (a prior record surviving, a retry succeeding, the digest cache
    /// refusing to skip) can be exercised without a full disk.
    nonisolated enum StoreWriteFault: Sendable, Equatable {
        case write
        case fsync
        case rename
    }

    private let faultLock = NSLock()
    nonisolated(unsafe) private var injectedWriteFault: (@Sendable () -> StoreWriteFault?)?

    /// Test-only. `nil` clears the injection.
    func injectWriteFaultForTesting(_ provider: (@Sendable () -> StoreWriteFault?)?) {
        faultLock.lock()
        injectedWriteFault = provider
        faultLock.unlock()
    }

    private func currentWriteFault() -> StoreWriteFault? {
        faultLock.lock()
        defer { faultLock.unlock() }
        return injectedWriteFault?()
    }

    /// Callers run on the IO queue, so the post hops to the main queue where
    /// the observers (the inspector, the inline run card) live.
    private func postRecordsDidChange() {
        DispatchQueue.main.async {
            NotificationCenter.default.post(name: Self.recordsDidChangeNotification, object: nil)
        }
    }

    /// Atomic, owner-only, durable write. The temp file is created 0600 and
    /// fsynced before the rename, so the record is never world-readable in
    /// flight and never truncated on disk after the call returns.
    private nonisolated static func writeOwnerOnlyAtomically(
        _ data: Data,
        to url: URL,
        fault: StoreWriteFault? = nil
    ) throws {
        let temporary = url.deletingLastPathComponent()
            .appendingPathComponent(".\(url.lastPathComponent).new-\(UUID().uuidString)", isDirectory: false)
        if fault == .write {
            throw NSError(
                domain: NSPOSIXErrorDomain,
                code: Int(EIO),
                userInfo: [NSFilePathErrorKey: temporary.path]
            )
        }
        let descriptor = open(temporary.path, O_WRONLY | O_CREAT | O_EXCL, 0o600)
        guard descriptor >= 0 else {
            throw NSError(
                domain: NSPOSIXErrorDomain,
                code: Int(errno),
                userInfo: [NSFilePathErrorKey: temporary.path]
            )
        }
        var writeErrno: Int32 = 0
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            guard let base = raw.baseAddress else { return }
            var offset = 0
            while offset < raw.count {
                let written = write(descriptor, base + offset, raw.count - offset)
                if written > 0 {
                    offset += written
                    continue
                }
                // A zero return on a blocking descriptor is not progress:
                // nothing was written and the loop would spin. It used to
                // break WITHOUT an error, so a "successful" write could leave
                // a truncated record behind. errno may read 0 here; any
                // zero-byte return is an I/O error by definition.
                writeErrno = errno != 0 ? errno : EIO
                break
            }
        }
        // Both of these are part of the durability claim above, not cleanup:
        // `fsync` reports ENOSPC/IO errors precisely because the data is not
        // on disk yet, and `close` can be the call that surfaces a deferred
        // write error. Ignoring them renamed an unflushed temp file over the
        // previous good record and cached its digest, so the next save of the
        // same bytes was skipped.
        let syncResult = fsync(descriptor)
        let syncErrno = syncResult == 0 ? 0 : errno
        let closeResult = close(descriptor)
        let closeErrno = closeResult == 0 ? 0 : errno
        let durabilityErrno: Int32 = fault == .fsync ? EIO
            : (writeErrno != 0 ? writeErrno
                : (syncErrno != 0 ? syncErrno : closeErrno))
        if durabilityErrno != 0 {
            try? FileManager.default.removeItem(at: temporary)
            throw NSError(
                domain: NSPOSIXErrorDomain,
                code: Int(durabilityErrno),
                userInfo: [NSFilePathErrorKey: temporary.path]
            )
        }
        if fault == .rename {
            try? FileManager.default.removeItem(at: temporary)
            throw NSError(
                domain: NSPOSIXErrorDomain,
                code: Int(EIO),
                userInfo: [NSFilePathErrorKey: url.path]
            )
        }
        do {
            _ = try FileManager.default.replaceItemAt(url, withItemAt: temporary)
            // `replaceItemAt` preserves the DESTINATION's metadata, so a record
            // first written by an older build at 0644 would keep that mode
            // through every rewrite. Re-assert the owner-only mode here; the
            // directory is 0700, which is why this is a hardening step rather
            // than a live exposure.
            try? FileManager.default.setAttributes(
                [.posixPermissions: 0o600],
                ofItemAtPath: url.path
            )
            // The rename is not durable until the DIRECTORY is flushed: a
            // power loss can lose a rename that was reported as a successful
            // write, which is exactly what the header promises it cannot. The
            // file's own fsync above covers its bytes; this covers the entry.
            try Self.fsyncDirectory(url.deletingLastPathComponent())
        } catch {
            try? FileManager.default.removeItem(at: temporary)
            throw error
        }
    }

    /// Flushes a directory entry so a completed rename survives a crash.
    private static func fsyncDirectory(_ directory: URL) throws {
        let descriptor = open(directory.path, O_RDONLY)
        guard descriptor >= 0 else {
            throw NSError(
                domain: NSPOSIXErrorDomain,
                code: Int(errno),
                userInfo: [NSFilePathErrorKey: directory.path]
            )
        }
        let result = fsync(descriptor)
        let syncErrno = result == 0 ? 0 : errno
        _ = close(descriptor)
        guard syncErrno == 0 else {
            throw NSError(
                domain: NSPOSIXErrorDomain,
                code: Int(syncErrno),
                userInfo: [NSFilePathErrorKey: directory.path]
            )
        }
    }

    // MARK: - Unrelayed index

    /// How many records the relay still owes one conversation. Cheap after the
    /// first call: the index is built once and maintained by this store.
    func unrelayedRunIds(sessionId: UUID?) -> Set<UUID> {
        // Steady state: the launch prewarm has finished, so answering is one
        // lock and a set lookup. Before that, a caller on the MAIN THREAD is
        // answered from what the index has learned so far and the scan is
        // started in the background: this is called while a send is being
        // prepared, and a few hundred JSON decodes there blocked the first
        // frame. A record missed on this one send is still unrelayed, so the
        // next send injects it — the report is never dropped, only delayed.
        signatureLock.lock()
        if relayIndexBuilt {
            let ids = unrelayedBySession[Self.relayBucket(sessionId)] ?? []
            signatureLock.unlock()
            return ids
        }
        signatureLock.unlock()
        if Thread.isMainThread {
            prepareRelayIndex()
            signatureLock.lock()
            defer { signatureLock.unlock() }
            return unrelayedBySession[Self.relayBucket(sessionId)] ?? []
        }
        return onQueue { [self] in
            buildRelayIndexIfNeeded()
            signatureLock.lock()
            defer { signatureLock.unlock() }
            return unrelayedBySession[Self.relayBucket(sessionId)] ?? []
        }
    }

    /// Builds the index OFF the caller's thread. The scan decodes every record
    /// in the store; it used to run synchronously wherever the caller happened
    /// to be, so a first send's lookup could pay the whole scan on the main
    /// thread before the first prompt rendered. The IO queue is the write
    /// path's own queue, so a save landing mid-scan waits for it rather than
    /// being missed by the scan's snapshot.
    func prepareRelayIndex() {
        ioQueue.async { [self] in
            buildRelayIndexIfNeeded()
        }
    }

    private static func relayBucket(_ sessionId: UUID?) -> String {
        sessionId?.uuidString ?? unscopedSessionDirectory
    }

    /// Scans the store and publishes what it found, in two phases.
    ///
    /// The decode used to happen while holding `signatureLock`, so every
    /// writer — and a first send's caller — waited behind a few hundred JSON
    /// decodes. The scan now runs with no lock; only the merge at the end
    /// takes it. A run relayed during the scan is remembered in
    /// `relaySettledDuringBuild` so publishing cannot resurrect it.
    ///
    /// Runs on the IO queue (`prepareRelayIndex` / `unrelayedRunIds`), so
    /// saves and deletes cannot interleave with the scan.
    private func buildRelayIndexIfNeeded() {
        signatureLock.lock()
        let alreadyBuilt = relayIndexBuilt
        signatureLock.unlock()
        guard !alreadyBuilt else { return }
        let candidates = scanRelayCandidates()
        signatureLock.lock()
        publishRelayIndexLocked(candidates)
        signatureLock.unlock()
    }

    /// Every terminal, unrelayed record on disk, grouped by session bucket.
    /// No lock is held while this reads.
    private func scanRelayCandidates() -> [String: Set<UUID>] {
        var found: [String: Set<UUID>] = [:]
        guard let directories = try? fileManager.contentsOfDirectory(atPath: rootURL.path) else { return found }
        for name in directories {
            let directory = rootURL.appendingPathComponent(name, isDirectory: true)
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: directory.path, isDirectory: &isDirectory),
                  isDirectory.boolValue,
                  let files = try? fileManager.contentsOfDirectory(atPath: directory.path) else { continue }
            for file in files where file.hasSuffix(".json") {
                guard let data = try? Data(contentsOf: directory.appendingPathComponent(file)),
                      let record = try? JSONDecoder().decode(AgentRunRecord.self, from: data),
                      record.schemaVersion <= AgentRunRecord.currentSchemaVersion,
                      record.status.isTerminal,
                      !record.summaryRelayed else { continue }
                found[Self.relayBucket(record.parentSessionId), default: []].insert(record.id)
            }
        }
        return found
    }

    private func publishRelayIndexLocked(_ candidates: [String: Set<UUID>]) {
        guard !relayIndexBuilt else { return }
        for (bucket, ids) in candidates {
            for id in ids where !relaySettledDuringBuild.contains(id) {
                unrelayedBySession[bucket, default: []].insert(id)
            }
        }
        relaySettledDuringBuild.removeAll()
        relayIndexBuilt = true
    }

    private func addRelayLocked(_ record: AgentRunRecord) {
        let bucket = Self.relayBucket(record.parentSessionId)
        unrelayedBySession[bucket, default: []].insert(record.id)
    }

    private func removeRelayLocked(runId: UUID, sessionId: UUID?) {
        let bucket = Self.relayBucket(sessionId)
        unrelayedBySession[bucket]?.remove(runId)
        if unrelayedBySession[bucket]?.isEmpty == true {
            unrelayedBySession.removeValue(forKey: bucket)
        }
    }

    /// Drops a run from every bucket. A conversation delete can re-scope a
    /// record between a caller reading its bucket and marking it relayed, and
    /// a removal keyed on the old bucket then left the id in the index forever
    /// (a duplicate report on every later send).
    private func removeRelayEverywhereLocked(runId: UUID) {
        for bucket in Array(unrelayedBySession.keys) {
            unrelayedBySession[bucket]?.remove(runId)
            if unrelayedBySession[bucket]?.isEmpty == true {
                unrelayedBySession.removeValue(forKey: bucket)
            }
        }
    }

    /// A save or a delete changes what the relay owes. Called from the write
    /// queue; the index is additive until built, so a save that lands before
    /// the scan cannot be lost by it.
    private func noteRelayMembership(_ record: AgentRunRecord) {
        signatureLock.lock()
        defer { signatureLock.unlock() }
        if record.status.isTerminal, !record.summaryRelayed {
            addRelayLocked(record)
        } else {
            removeRelayLocked(runId: record.id, sessionId: record.parentSessionId)
        }
    }

    // MARK: - Reading

    /// Every run of one conversation, newest first. Unknown newer schema
    /// versions are ignored rather than decoded lossily: a build that writes a
    /// field this one would drop on its next write must not be rewritten.
    ///
    /// Ordered behind pending writes and deletes on the store's IO queue. A
    /// caller that clears and then reloads therefore cannot read the store
    /// before the deletions it just asked for have landed.
    func load(sessionId: UUID?) -> [AgentRunRecord] {
        onQueue { [self] in
            let directory = directory(for: sessionId)
            guard let names = try? fileManager.contentsOfDirectory(atPath: directory.path) else {
                return []
            }
            var records: [AgentRunRecord] = []
            for name in names where name.hasSuffix(".json") {
                guard let record = decodeRecord(at: directory.appendingPathComponent(name)) else { continue }
                records.append(record)
            }
            return records.sorted { Self.sortDate($0) > Self.sortDate($1) }
        }
    }

    func load(runId: UUID, sessionId: UUID?) -> AgentRunRecord? {
        onQueue { [self] in
            decodeRecord(at: fileURL(runId: runId, sessionId: sessionId))
        }
    }

    /// Finds one run without decoding the whole store. The transcript card
    /// knows a run id but not which conversation directory holds it, and a
    /// session with hundreds of finished runs should not be re-parsed to draw
    /// one card.
    func loadAny(runId: UUID) -> AgentRunRecord? {
        onQueue { [self] in
            let name = "\(runId.uuidString).json"
            guard let direct = try? fileManager.contentsOfDirectory(atPath: rootURL.path) else {
                return nil
            }
            // A conversation delete re-scopes a run by writing the new copy
            // before removing the old one, so a crash between the two leaves
            // BOTH files. Returning whichever directory the listing named
            // first was a coin flip; the newer file is the copy that carries
            // the re-scope (and any write after it).
            var newest: (date: Date, record: AgentRunRecord)?
            for directoryName in direct {
                let candidate = rootURL
                    .appendingPathComponent(directoryName, isDirectory: true)
                    .appendingPathComponent(name, isDirectory: false)
                guard fileManager.fileExists(atPath: candidate.path),
                      let record = decodeRecord(at: candidate) else { continue }
                let modified = (try? candidate.resourceValues(forKeys: [.contentModificationDateKey]))?
                    .contentModificationDate ?? .distantPast
                if newest == nil || modified > newest!.date {
                    newest = (modified, record)
                }
            }
            return newest?.record
        }
    }

    /// Every run across every conversation. Retention and the app's launch
    /// sweep read this; the inspector reads one session.
    func loadAll() -> [AgentRunRecord] {
        onQueue { [self] in
            guard let directories = try? fileManager.contentsOfDirectory(atPath: rootURL.path) else {
                return []
            }
            var records: [AgentRunRecord] = []
            for name in directories {
                var isDirectory: ObjCBool = false
                let path = rootURL.appendingPathComponent(name).path
                guard fileManager.fileExists(atPath: path, isDirectory: &isDirectory),
                      isDirectory.boolValue else { continue }
                if let sessionId = UUID(uuidString: name) {
                    records.append(contentsOf: load(sessionId: sessionId))
                } else {
                    records.append(contentsOf: loadSessionDirectory(named: name))
                }
            }
            return records.sorted { Self.sortDate($0) > Self.sortDate($1) }
        }
    }

    /// Marks runs a previous process left mid-flight as failed.
    ///
    /// A record is written `.running` before the work starts and only reaches
    /// a terminal status when the run finishes, so a crash or a force-quit
    /// leaves a record the inspector lists as ACTIVE with no live instance to
    /// Stop and no way to clear it. Nothing can be in flight before the
    /// manager that owns runs exists, so at launch every non-terminal record
    /// older than `graceSeconds` is an orphan; it is rewritten as `.failed`
    /// rather than left claiming to be working. The sweep is ASYNC, though:
    /// a run the operator spawned while this scan was still walking the store
    /// is live, and `isLive` is what keeps it from being rewritten as failed
    /// underneath its own working instance.
    @discardableResult
    func reconcileInterruptedRuns(
        now: Date = Date(),
        graceSeconds: TimeInterval = 60,
        isLive: (UUID) -> Bool = { _ in false }
    ) -> [UUID] {
        var reconciled: [UUID] = []
        for var record in loadAll() where !record.status.isTerminal {
            if isLive(record.id) { continue }
            let started = record.startedAt ?? record.createdAt
            guard now.timeIntervalSince(started) >= graceSeconds else { continue }
            record.status = .failed
            record.completedAt = record.completedAt ?? now
            if record.error == nil {
                record.error = "The app quit before this run finished. Its last recorded state is kept below."
            }
            saveImmediately(record)
            reconciled.append(record.id)
        }
        return reconciled
    }

    /// A directory whose name is not a UUID still belongs to some run (an
    /// unscoped bucket, or a session deleted while its runs were not). It is
    /// read so retention can prune it, never attributed to a conversation.
    private func loadSessionDirectory(named name: String) -> [AgentRunRecord] {
        let directory = rootURL.appendingPathComponent(name, isDirectory: true)
        guard let names = try? fileManager.contentsOfDirectory(atPath: directory.path) else {
            return []
        }
        return names
            .filter { $0.hasSuffix(".json") }
            .compactMap { decodeRecord(at: directory.appendingPathComponent($0)) }
    }

    private func decodeRecord(at url: URL) -> AgentRunRecord? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        guard let record = try? JSONDecoder().decode(AgentRunRecord.self, from: data) else {
            return nil
        }
        guard record.schemaVersion <= AgentRunRecord.currentSchemaVersion else { return nil }
        return record
    }

    // MARK: - Deletion and retention

    func delete(runId: UUID, sessionId: UUID?) {
        ioQueue.async { [self] in
            deleteLocked(runId: runId, sessionId: sessionId)
        }
    }

    /// Every bucket that holds a file for this run, decodable or not.
    ///
    /// A conversation delete re-scopes a rescued run by writing the new copy
    /// BEFORE removing the old one, so a crash between the two leaves the same
    /// run in two buckets. Deleting only the newest copy then tombstoned the id
    /// and left the stale one on disk; the tombstone is in memory, so after a
    /// relaunch the stale record resurfaced — retention-exempt whenever it
    /// still claimed worktree work, even after the operator had discarded the
    /// tree. A delete has to remove the record, not one copy of it.
    private func recordFileURLs(runId: UUID) -> [URL] {
        let name = "\(runId.uuidString).json"
        guard let directories = try? fileManager.contentsOfDirectory(atPath: rootURL.path) else {
            let direct = fileURL(runId: runId, sessionId: nil)
            return fileManager.fileExists(atPath: direct.path) ? [direct] : []
        }
        var urls: [URL] = []
        for directory in directories {
            let candidate = rootURL
                .appendingPathComponent(directory, isDirectory: true)
                .appendingPathComponent(name, isDirectory: false)
            if fileManager.fileExists(atPath: candidate.path) {
                urls.append(candidate)
            }
        }
        return urls
    }

    /// Deletes a record and locates it the way `mutate` does, so a run
    /// re-scoped to the unscoped bucket is removed from where it actually
    /// lives rather than from a path the tombstone refuses to touch.
    private func deleteLocked(runId: UUID, sessionId: UUID?) {
        let preferred = fileURL(runId: runId, sessionId: sessionId)
        let candidates = recordFileURLs(runId: runId)
        let urls = candidates.isEmpty ? [preferred] : candidates
        // Tombstone AFTER the removals succeed. A tombstone written first is
        // never lifted, so a removal that failed (permissions, a locked file)
        // left a record on disk that every later write refused to update: an
        // immortal, invisible run. The queue is serial, so a save enqueued
        // after this still sees the tombstone before it writes.
        //
        // An ABSENT file is the one case that still deserves the tombstone:
        // the record was already gone, and a save captured before this delete
        // is queued behind it. Skipping the tombstone here let that save
        // recreate a run the operator had deleted.
        for url in urls {
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) else { continue }
            guard (try? fileManager.removeItem(at: url)) != nil else { return }
        }
        noteDeletedRun(runId)
        signatureLock.lock()
        for url in urls {
            writtenSignatures[url.path] = nil
        }
        removeRelayEverywhereLocked(runId: runId)
        signatureLock.unlock()
        SubagentNotifier.withdraw(runId: runId)
        postRecordsDidChange()
    }

    /// Deletes every record `clearable` accepts, scanning on the IO queue.
    ///
    /// The cleared-list decision used to be a synchronous load on the main
    /// actor, which decoded an entire conversation's history before the UI
    /// could redraw. The scan runs here instead, and the ordering is still
    /// deterministic: every delete is enqueued before a later `load` (which
    /// `sync`s this same queue) can run, so a reload immediately after a clear
    /// cannot refill the rows.
    func deleteClearableRecords(
        sessionId: UUID?,
        clearable: @escaping @Sendable (AgentRunRecord) -> Bool
    ) {
        ioQueue.async { [self] in
            let stored = load(sessionId: sessionId)
                + (sessionId == nil ? [] : load(sessionId: nil))
            for record in stored where clearable(record) {
                deleteLocked(runId: record.id, sessionId: record.parentSessionId)
            }
        }
    }

    /// Synchronous delete for an explicit per-run discard, where a list that
    /// refills from disk after the call would read as the button failing.
    func deleteImmediately(runId: UUID, sessionId: UUID?) {
        onQueue { [self] in
            deleteLocked(runId: runId, sessionId: sessionId)
        }
    }

    /// Deleting a conversation deletes the runs it spawned. §10 requires it;
    /// leaving them would keep a deleted conversation's work on disk forever.
    ///
    /// One deliberate exception: an UNAPPLIED isolated run's record is the only
    /// thing keeping its worktree out of the launch sweep —
    /// `pruneStaleWorktrees` protects a tree while a decodable record says
    /// unapplied — so deleting it would hand the run's only copy of its work to
    /// a 14-day clock the operator never sees, the same disposal `prune`
    /// refuses to make. Those records are RE-SCOPED to the unscoped bucket
    /// instead of deleted: they ride every inspector with their Apply and Clear
    /// buttons, `prune` keeps skipping them, and the sweep keeps protecting
    /// their trees until the operator lands or clears the work. Callers cancel
    /// the session's live runs first (§10's rule), so a non-terminal survivor
    /// is marked cancelled here rather than left claiming to work for a
    /// conversation that no longer exists, and `summaryRelayed` is set so the
    /// report relay never injects a deleted conversation's run into an
    /// unrelated one.
    func deleteSession(_ sessionId: UUID) {
        let url = directory(for: sessionId)
        ioQueue.async { [self] in
            var rescueFailed = false
            let decoded = load(sessionId: sessionId)
            for var record in decoded where record.holdsWorktreeWork {
                if !record.status.isTerminal {
                    record.status = .cancelled
                    record.completedAt = record.completedAt ?? Date()
                    if record.error == nil {
                        record.error = "The conversation that owned this run was deleted while it was still working."
                    }
                }
                record.parentSessionId = nil
                record.summaryRelayed = true
                // Write-new before remove-old: a crash between the two leaves a
                // duplicate the sweep still protects, while the reverse order
                // leaves the worktree unprotected. If the rescue write FAILS,
                // the original is the only record and must stay: removing it
                // would let the sweep dispose of work the operator never saw.
                // `saveImmediately` is safe from this thread because the store's
                // synchronous entry points run inline when the caller already
                // holds `ioQueue` (see `onQueue`); before that, this line
                // `sync`-ed the queue it was executing on and libdispatch
                // aborted the process.
                guard saveImmediately(record) else {
                    // The rescue write failed, so the ORIGINAL is the only
                    // record. Removing the directory or tombstoning the
                    // session below would delete or hide it, which is exactly
                    // the disposal this branch exists to prevent: leave the
                    // whole conversation's files where they are.
                    rescueFailed = true
                    continue
                }
                let original = fileURL(runId: record.id, sessionId: sessionId)
                try? fileManager.removeItem(at: original)
                signatureLock.lock()
                writtenSignatures[original.path] = nil
                signatureLock.unlock()
                // A PREVIOUS delete that crashed between its two writes (or a
                // save that landed in the unscoped bucket) can leave another
                // copy of this run behind. The rescue copy just written is the
                // current one; every other file for this id is stale and must
                // not outlive the record it duplicates, or the next launch
                // reads work the operator just deleted. The keep copy lives in
                // the unscoped bucket, so it is named explicitly.
                let survivor = fileURL(runId: record.id, sessionId: nil)
                for stale in recordFileURLs(runId: record.id) where stale.path != survivor.path {
                    try? fileManager.removeItem(at: stale)
                    signatureLock.lock()
                    writtenSignatures[stale.path] = nil
                    signatureLock.unlock()
                }
            }
            guard !rescueFailed else { return }
            // A record this build cannot decode is kept, not deleted with the
            // conversation. `pruneStaleWorktrees` asks `hasRecordFile` before
            // disposing of a UUID-named worktree precisely because that file
            // may be a newer schema's record of unapplied work; removing its
            // directory here would hand the tree to the 14-day clock. Move it
            // to the unscoped bucket instead, where it still protects its tree.
            let decodableIds = Set(decoded.map(\.id))
            if let files = try? fileManager.contentsOfDirectory(
                at: url,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            ) {
                let destination = directory(for: nil)
                for file in files where file.pathExtension == "json" {
                    guard let runId = UUID(uuidString: file.deletingPathExtension().lastPathComponent),
                          !decodableIds.contains(runId) else { continue }
                    do {
                        try fileManager.createDirectory(
                            at: destination,
                            withIntermediateDirectories: true,
                            attributes: [.posixPermissions: 0o700]
                        )
                        try? fileManager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: destination.path)
                        try fileManager.moveItem(
                            at: file,
                            to: destination.appendingPathComponent(file.lastPathComponent)
                        )
                    } catch {
                        // Moving failed, so the file is still the only copy of
                        // whatever it says. Leave the whole directory alone.
                        rescueFailed = true
                        break
                    }
                }
            }
            guard !rescueFailed else { return }
            // Removal BEFORE the tombstone, like `deleteLocked`: a tombstone
            // written first is never lifted, so a failed removal would leave a
            // directory every later write refuses to touch. A session that had
            // no directory is still tombstoned, so a save already in flight
            // cannot recreate it.
            if fileManager.fileExists(atPath: url.path) {
                guard (try? fileManager.removeItem(at: url)) != nil else { return }
            }
            noteDeletedSession(sessionId)
            signatureLock.lock()
            for key in writtenSignatures.keys where key.hasPrefix(url.path) {
                writtenSignatures[key] = nil
            }
            unrelayedBySession.removeValue(forKey: Self.relayBucket(sessionId))
            signatureLock.unlock()
            // A delivered banner for a run whose conversation is gone is a tap
            // that lands on nothing (the inspector cannot show the run and the
            // selection pins an id with no row).
            for record in decoded {
                SubagentNotifier.withdraw(runId: record.id)
            }
            postRecordsDidChange()
        }
    }

    /// Caller holds no lock; both helpers take `signatureLock` themselves.
    private func noteDeletedRun(_ runId: UUID) {
        signatureLock.lock()
        defer { signatureLock.unlock() }
        guard deletedRunIds.insert(runId).inserted else { return }
        deletedRunOrder.append(runId)
        while deletedRunOrder.count > maximumTombstones {
            let oldest = deletedRunOrder.removeFirst()
            deletedRunIds.remove(oldest)
        }
    }

    private func noteDeletedSession(_ sessionId: UUID) {
        signatureLock.lock()
        defer { signatureLock.unlock() }
        guard deletedSessionIds.insert(sessionId).inserted else { return }
        deletedSessionOrder.append(sessionId)
        while deletedSessionOrder.count > maximumTombstones {
            let oldest = deletedSessionOrder.removeFirst()
            deletedSessionIds.remove(oldest)
        }
    }

    /// Keeps the newest `policy.keepNewest` runs or `policy.keepDays` of history,
    /// whichever is larger, and never prunes a run younger than
    /// `policy.minimumAgeHours`. Synchronous so the launch sweep can report what
    /// it removed and a test can ask the same question twice.
    @discardableResult
    func prune(now: Date = Date(), policy: RetentionPolicy = .standard) -> [UUID] {
        // The whole pass runs on the store's IO queue, ordered after every
        // write in flight. That ordering is what makes remove-then-tombstone
        // safe: no save can interleave between the two, and a save enqueued
        // after the pass sees the tombstone. Tombstoning first instead left a
        // failed removal as an immortal, invisible file that no later write
        // could update.
        onQueue { [self] in
            let all = loadAll()
            let sorted = all.sorted { Self.sortDate($0) > Self.sortDate($1) }
            let cutoff = now.addingTimeInterval(-policy.keepDays * 86_400)
            let youngestAllowed = now.addingTimeInterval(-policy.minimumAgeHours * 3_600)
            var removed: [UUID] = []
            for (index, record) in sorted.enumerated() {
                let date = Self.sortDate(record)
                if index < policy.keepNewest { continue }
                if date >= cutoff { continue }
                if date >= youngestAllowed { continue }
                // An old record can still be LIVE: the retention clock runs from a
                // run's start, and a long-running job (or a restored record waiting
                // on a slow operator approval) would otherwise be deleted underneath
                // its own instance, which then rewrites it anyway.
                guard record.status.isTerminal else { continue }
                // An isolated run's record is the only thing keeping its
                // WORKTREE alive: the launch sweep protects a tree while the
                // record says unapplied OR names kept ignored files, so
                // pruning the record first means the very next sweep deletes
                // the run's only copy of its work. Retention is for
                // conversation history; that work is the operator's to clear
                // or apply, never the clock's.
                if record.holdsWorktreeWork { continue }
                // Every bucket copy goes, for the same reason `deleteLocked`
                // removes them all: one copy surviving a tombstoned id is a
                // record no later write can repair and the next launch reads
                // as history the operator had pruned.
                let urls = recordFileURLs(runId: record.id)
                guard !urls.isEmpty else { continue }
                var removalFailed = false
                for url in urls {
                    var isDirectory: ObjCBool = false
                    guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) else { continue }
                    guard (try? fileManager.removeItem(at: url)) != nil else {
                        removalFailed = true
                        break
                    }
                }
                // Tombstone only once every copy is gone; see `deleteLocked`.
                guard !removalFailed else { continue }
                noteDeletedRun(record.id)
                removed.append(record.id)
                signatureLock.lock()
                for url in urls {
                    writtenSignatures[url.path] = nil
                }
                removeRelayEverywhereLocked(runId: record.id)
                signatureLock.unlock()
                SubagentNotifier.withdraw(runId: record.id)
            }
            return removed
        }
    }

    /// Removes the temp files an interrupted write leaves behind (a hard
    /// crash between the create and the rename makes `.<run>.json.new-<id>`
    /// orphans nothing else ever looks at) and reports how many records
    /// exist on disk that this build cannot decode. Age-gated so a temp file
    /// an in-flight write is using right now is never touched: a real temp is
    /// milliseconds old, and anything older is an orphan by construction.
    @discardableResult
    func sweepOrphanedTempWrites(now: Date = Date(), olderThan: TimeInterval = 3_600) -> Int {
        onQueue { [self] in
            var removedTemps = 0
            var undecodable = 0
            guard let entryNames = try? fileManager.contentsOfDirectory(atPath: rootURL.path) else {
                return 0
            }
            for entryName in entryNames {
                let directory = rootURL.appendingPathComponent(entryName, isDirectory: true)
                var isDirectory: ObjCBool = false
                guard fileManager.fileExists(atPath: directory.path, isDirectory: &isDirectory),
                      isDirectory.boolValue else { continue }
                guard let fileNames = try? fileManager.contentsOfDirectory(atPath: directory.path) else { continue }
                for fileName in fileNames {
                    if fileName.hasPrefix("."), fileName.contains(".new-") {
                        let url = directory.appendingPathComponent(fileName, isDirectory: false)
                        let modified = (try? fileManager.attributesOfItem(atPath: url.path))?[.modificationDate] as? Date
                        guard let modified, now.timeIntervalSince(modified) >= olderThan else { continue }
                        if (try? fileManager.removeItem(at: url)) != nil { removedTemps += 1 }
                        continue
                    }
                    if fileName.hasSuffix(".json"), fileName.first != "." {
                        if decodeRecord(at: directory.appendingPathComponent(fileName, isDirectory: false)) == nil {
                            undecodable += 1
                        }
                    }
                }
            }
            if undecodable > 0 {
                // Surfaced, never deleted: a file this build cannot decode may
                // be a record a NEWER build wrote, and deleting it would make a
                // downgrade-then-upgrade cycle destroy history.
                print("ℹ️ [AgentRunStore] \(undecodable) record(s) on disk are not decodable by this build; they are kept untouched.")
            }
            return removedTemps
        }
    }

    // MARK: - Bounds

    /// What actually gets written. Tool output is the field that can be
    /// unbounded, so it is truncated at a per-entry budget rather than refusing
    /// the entry: a record missing its transcript is worse than one whose output
    /// was cut.
    nonisolated static func bounded(_ record: AgentRunRecord) -> AgentRunRecord {
        var bounded = record
        bounded.task = truncate(record.task, to: TranscriptBudget.task)
        bounded.contextSummary = record.contextSummary.map {
            truncate($0, to: TranscriptBudget.contextSummary)
        }
        bounded.error = record.error.map { truncate($0, to: TranscriptBudget.status) }
        bounded.messages = record.messages.map { message in
            var message = message
            message.content = truncate(message.content, to: TranscriptBudget.message)
            return message
        }
        bounded.transcript = record.transcript.map { entry in
            var entry = entry
            switch entry.kind {
            case .toolResult, .toolCall:
                if let text = entry.text {
                    entry.text = truncate(text, to: TranscriptBudget.toolOutput)
                }
                if var call = entry.toolRecord {
                    call.output = call.output.map { truncate($0, to: TranscriptBudget.toolOutput) }
                    call.error = call.error.map { truncate($0, to: TranscriptBudget.status) }
                    call.rawArguments = truncate(call.rawArguments, to: TranscriptBudget.rawArguments)
                    call.arguments = call.arguments.mapValues {
                        truncate($0, to: TranscriptBudget.argumentValue)
                    }
                    call.pinnedEndpoint = call.pinnedEndpoint.map {
                        truncate($0, to: TranscriptBudget.status)
                    }
                    entry.toolRecord = call
                }
            case .assistantThinking:
                if let text = entry.text { entry.text = truncate(text, to: TranscriptBudget.thinking) }
            case .assistantText, .task:
                if let text = entry.text { entry.text = truncate(text, to: TranscriptBudget.assistantText) }
            case .status, .usage:
                if let text = entry.text { entry.text = truncate(text, to: TranscriptBudget.status) }
            }
            return entry
        }
        if var report = bounded.report {
            report.summary = truncate(report.summary, to: TranscriptBudget.assistantText)
            report.diff = report.diff.map { truncate($0, to: TranscriptBudget.reportDiff) }
            report.risks = report.risks.map { truncate($0, to: TranscriptBudget.status) }
            report.openQuestions = report.openQuestions.map {
                truncate($0, to: TranscriptBudget.status)
            }
            report.findings = report.findings.map { finding in
                var finding = finding
                finding.claim = truncate(finding.claim, to: TranscriptBudget.status)
                finding.evidence = truncate(finding.evidence, to: TranscriptBudget.message)
                return finding
            }
            bounded.report = report
        }
        return bounded
    }

    private nonisolated static func truncate(_ text: String, to limit: Int) -> String {
        guard text.count > limit else { return text }
        // Idempotent: a value that already carries this function's marker
        // (from the previous save's bound) stays as it is. Re-truncating on
        // every save stacked a new tail with a different character count,
        // which also defeated the store's digest skip: an unchanged record
        // was rewritten on every boundary.
        if isTruncationMarker(text.dropFirst(limit)) { return text }
        let cut = text.prefix(limit)
        return cut + "\n… [truncated \(text.count - limit) characters]"
    }

    /// Recognises this store's own truncation marker, so a value read back off
    /// disk is not truncated a second time.
    private nonisolated static func isTruncationMarker(_ tail: Substring) -> Bool {
        let head = "\n… [truncated "
        let foot = " characters]"
        guard tail.hasPrefix(head), tail.hasSuffix(foot) else { return false }
        let digits = tail.dropFirst(head.count).dropLast(foot.count)
        return !digits.isEmpty && digits.allSatisfy(\.isNumber)
    }

    private nonisolated static func sortDate(_ record: AgentRunRecord) -> Date {
        record.completedAt ?? record.startedAt ?? record.createdAt
    }
}
