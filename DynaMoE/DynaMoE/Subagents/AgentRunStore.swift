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
    static func configuredPolicy(defaults: UserDefaults = .standard) -> RetentionPolicy {
        var policy = RetentionPolicy.standard
        if let days = defaults.object(forKey: "dynamoe_subagent_retention_days") as? NSNumber,
           days.doubleValue > 0 {
            policy.keepDays = days.doubleValue
        }
        if let runs = defaults.object(forKey: "dynamoe_subagent_retention_runs") as? NSNumber,
           runs.intValue > 0 {
            policy.keepNewest = runs.intValue
        }
        return policy
    }

    let rootURL: URL
    private let fileManager: FileManager
    private let ioQueue = DispatchQueue(label: "dynamoe.agent-run-store", qos: .utility)
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
    private let maximumTombstones = 512
    /// Terminal records whose report has not been relayed, keyed by session
    /// bucket. Built once from disk (off the main actor at launch) and kept
    /// current by `writeNow`/`delete`. The report relay asks this instead of
    /// decoding every record in a conversation on each send.
    private var unrelayedBySession: [String: Set<UUID>] = [:]
    private var relayIndexBuilt = false

    init(
        rootURL: URL = AgentRunStore.defaultRootURL,
        fileManager: FileManager = .default
    ) {
        self.rootURL = rootURL
        self.fileManager = fileManager
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
        let bounded = Self.bounded(record)
        ioQueue.async { [self] in
            writeNow(bounded, to: url)
        }
    }

    /// Synchronous write, used by tests and by the app's termination path where
    /// losing the last run is the one outcome worth pausing for.
    func saveImmediately(_ record: AgentRunRecord) {
        let url = fileURL(runId: record.id, sessionId: record.parentSessionId)
        let bounded = Self.bounded(record)
        ioQueue.sync { [self] in
            writeNow(bounded, to: url)
        }
    }

    /// Blocks until every queued write has landed. Tests use it; so does the
    /// terminate handler before it gives the process back.
    func flush() {
        ioQueue.sync {}
    }

    /// Reads the CURRENT record from disk, applies the mutation and writes it
    /// back — all on the store's IO queue, so a caller that loaded a record
    /// seconds ago cannot revert fields other writers changed in between. The
    /// apply path used to restore its stale copy and clobber a
    /// `summaryRelayed` write landing mid-apply, which re-injected an already
    /// relayed report at the next send. Returns the record as it stands after
    /// the mutation, or nil when no decodable record exists.
    @discardableResult
    func mutate(runId: UUID, sessionId: UUID?, _ change: (inout AgentRunRecord) -> Void) -> AgentRunRecord? {
        let url = fileURL(runId: runId, sessionId: sessionId)
        return ioQueue.sync { [self] in
            guard var current = decodeRecord(at: url) else { return nil }
            change(&current)
            let bounded = Self.bounded(current)
            writeNow(bounded, to: url)
            return bounded
        }
    }

    private func writeNow(_ record: AgentRunRecord, to url: URL) {
        signatureLock.lock()
        let wasDeleted = deletedRunIds.contains(record.id)
            || (record.parentSessionId.map { deletedSessionIds.contains($0) } ?? false)
        signatureLock.unlock()
        if wasDeleted { return }

        let directory = url.deletingLastPathComponent()
        do {
            try fileManager.createDirectory(
                at: directory,
                withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700]
            )
        } catch {
            print("⚠️ [AgentRunStore] could not create \(directory.lastPathComponent): \(error.localizedDescription)")
            return
        }
        let data: Data
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            data = try encoder.encode(record)
        } catch {
            print("⚠️ [AgentRunStore] failed to encode run \(record.id): \(error.localizedDescription)")
            return
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
            return
        }
        do {
            // A run's record carries the task, the conversation excerpt,
            // thinking and tool output. Application Support is user-readable by
            // default, which is the wrong floor for that content: the helper
            // creates the file 0600 from its first byte (a plain `.atomic`
            // write creates it 0644 and chmods afterwards, a window in which
            // the record is world-readable) and fsyncs before the rename, so a
            // crash cannot leave a truncated record either.
            try Self.writeOwnerOnlyAtomically(data, to: url)
            try? fileManager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
            signatureLock.lock()
            writtenSignatures[key] = signature
            signatureLock.unlock()
            noteRelayMembership(record)
        } catch {
            print("⚠️ [AgentRunStore] failed to write run \(record.id): \(error.localizedDescription)")
        }
    }

    /// Atomic, owner-only, durable write. The temp file is created 0600 and
    /// fsynced before the rename, so the record is never world-readable in
    /// flight and never truncated on disk after the call returns.
    private nonisolated static func writeOwnerOnlyAtomically(_ data: Data, to url: URL) throws {
        let temporary = url.deletingLastPathComponent()
            .appendingPathComponent(".\(url.lastPathComponent).new-\(UUID().uuidString)", isDirectory: false)
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
        _ = fsync(descriptor)
        close(descriptor)
        if writeErrno != 0 {
            try? FileManager.default.removeItem(at: temporary)
            throw NSError(
                domain: NSPOSIXErrorDomain,
                code: Int(writeErrno),
                userInfo: [NSFilePathErrorKey: temporary.path]
            )
        }
        do {
            _ = try FileManager.default.replaceItemAt(url, withItemAt: temporary)
        } catch {
            try? FileManager.default.removeItem(at: temporary)
            throw error
        }
    }

    // MARK: - Unrelayed index

    /// How many records the relay still owes one conversation. Cheap after the
    /// first call: the index is built once and maintained by this store.
    func unrelayedRunIds(sessionId: UUID?) -> Set<UUID> {
        signatureLock.lock()
        defer { signatureLock.unlock() }
        ensureRelayIndexLocked()
        return unrelayedBySession[Self.relayBucket(sessionId)] ?? []
    }

    /// Builds the index OFF the caller's thread. The scan decodes every record
    /// in the store; it used to run synchronously wherever the caller happened
    /// to be, so a first send's lookup could pay the whole scan on the main
    /// thread under the relay lock before the first prompt rendered. The IO
    /// queue is the write path's own queue, so a save landing mid-scan waits
    /// for it rather than being missed by the scan's snapshot.
    func prepareRelayIndex() {
        ioQueue.async { [self] in
            signatureLock.lock()
            defer { signatureLock.unlock() }
            ensureRelayIndexLocked()
        }
    }

    private static func relayBucket(_ sessionId: UUID?) -> String {
        sessionId?.uuidString ?? unscopedSessionDirectory
    }

    private func ensureRelayIndexLocked() {
        guard !relayIndexBuilt else { return }
        relayIndexBuilt = true
        guard let directories = try? fileManager.contentsOfDirectory(atPath: rootURL.path) else { return }
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
                addRelayLocked(record)
            }
        }
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
    func load(sessionId: UUID?) -> [AgentRunRecord] {
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

    func load(runId: UUID, sessionId: UUID?) -> AgentRunRecord? {
        decodeRecord(at: fileURL(runId: runId, sessionId: sessionId))
    }

    /// Finds one run without decoding the whole store. The transcript card
    /// knows a run id but not which conversation directory holds it, and a
    /// session with hundreds of finished runs should not be re-parsed to draw
    /// one card.
    func loadAny(runId: UUID) -> AgentRunRecord? {
        let name = "\(runId.uuidString).json"
        if let direct = try? fileManager.contentsOfDirectory(atPath: rootURL.path) {
            for directoryName in direct {
                let candidate = rootURL
                    .appendingPathComponent(directoryName, isDirectory: true)
                    .appendingPathComponent(name, isDirectory: false)
                if fileManager.fileExists(atPath: candidate.path) {
                    return decodeRecord(at: candidate)
                }
            }
        }
        return nil
    }

    /// Every run across every conversation. Retention and the app's launch
    /// sweep read this; the inspector reads one session.
    func loadAll() -> [AgentRunRecord] {
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
        let url = fileURL(runId: runId, sessionId: sessionId)
        ioQueue.async { [self] in
            noteDeletedRun(runId)
            try? fileManager.removeItem(at: url)
            signatureLock.lock()
            writtenSignatures[url.path] = nil
            removeRelayLocked(runId: runId, sessionId: sessionId)
            signatureLock.unlock()
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
            for var record in load(sessionId: sessionId) where record.isolation.isIsolated && record.appliedAt == nil {
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
                // leaves the worktree unprotected.
                saveImmediately(record)
                let original = fileURL(runId: record.id, sessionId: sessionId)
                try? fileManager.removeItem(at: original)
                signatureLock.lock()
                writtenSignatures[original.path] = nil
                signatureLock.unlock()
            }
            noteDeletedSession(sessionId)
            try? fileManager.removeItem(at: url)
            signatureLock.lock()
            for key in writtenSignatures.keys where key.hasPrefix(url.path) {
                writtenSignatures[key] = nil
            }
            unrelayedBySession.removeValue(forKey: Self.relayBucket(sessionId))
            signatureLock.unlock()
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
        // write in flight. It used to delete then tombstone from an arbitrary
        // thread, so a `save` that had already passed its `wasDeleted` check
        // could land AFTER the `removeItem` — resurrecting a pruned record
        // that the too-late tombstone then blocked from being rewritten: an
        // immortal, invisible file. `delete(runId:)` already runs on the queue
        // with the tombstone first; prune must answer the same way.
        ioQueue.sync { [self] in
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
                // An unapplied isolated run's record is the only thing keeping its
                // WORKTREE alive: the launch sweep protects a tree while a
                // decodable record says unapplied, so pruning the record first
                // means the very next sweep deletes the run's only copy of its
                // work. Retention is for conversation history; unapplied work is
                // the operator's to clear or apply, never the clock's.
                if record.isolation.isIsolated, record.appliedAt == nil { continue }
                let url = fileURL(runId: record.id, sessionId: record.parentSessionId)
                noteDeletedRun(record.id)
                if (try? fileManager.removeItem(at: url)) != nil {
                    removed.append(record.id)
                    signatureLock.lock()
                    writtenSignatures[url.path] = nil
                    removeRelayLocked(runId: record.id, sessionId: record.parentSessionId)
                    signatureLock.unlock()
                }
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
        ioQueue.sync { [self] in
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
