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

    private func writeNow(_ record: AgentRunRecord, to url: URL) {
        let directory = url.deletingLastPathComponent()
        do {
            try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
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
            return
        }
        do {
            try data.write(to: url, options: .atomic)
            signatureLock.lock()
            writtenSignatures[key] = signature
            signatureLock.unlock()
        } catch {
            print("⚠️ [AgentRunStore] failed to write run \(record.id): \(error.localizedDescription)")
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
            try? fileManager.removeItem(at: url)
            signatureLock.lock()
            writtenSignatures[url.path] = nil
            signatureLock.unlock()
        }
    }

    /// Deleting a conversation deletes the runs it spawned. §10 requires it;
    /// leaving them would keep a deleted conversation's work on disk forever.
    func deleteSession(_ sessionId: UUID) {
        let url = directory(for: sessionId)
        ioQueue.async { [self] in
            try? fileManager.removeItem(at: url)
            signatureLock.lock()
            for key in writtenSignatures.keys where key.hasPrefix(url.path) {
                writtenSignatures[key] = nil
            }
            signatureLock.unlock()
        }
    }

    /// Keeps the newest `policy.keepNewest` runs or `policy.keepDays` of history,
    /// whichever is larger, and never prunes a run younger than
    /// `policy.minimumAgeHours`. Synchronous so the launch sweep can report what
    /// it removed and a test can ask the same question twice.
    @discardableResult
    func prune(now: Date = Date(), policy: RetentionPolicy = .standard) -> [UUID] {
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
            let url = fileURL(runId: record.id, sessionId: record.parentSessionId)
            if (try? fileManager.removeItem(at: url)) != nil {
                removed.append(record.id)
                signatureLock.lock()
                writtenSignatures[url.path] = nil
                signatureLock.unlock()
            }
        }
        return removed
    }

    // MARK: - Bounds

    /// What actually gets written. Tool output is the field that can be
    /// unbounded, so it is truncated at a per-entry budget rather than refusing
    /// the entry: a record missing its transcript is worse than one whose output
    /// was cut.
    nonisolated static func bounded(_ record: AgentRunRecord) -> AgentRunRecord {
        var bounded = record
        bounded.task = truncate(record.task, to: TranscriptBudget.task)
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
            report.diff = report.diff.map { truncate($0, to: TranscriptBudget.toolOutput) }
            bounded.report = report
        }
        return bounded
    }

    private nonisolated static func truncate(_ text: String, to limit: Int) -> String {
        guard text.count > limit else { return text }
        let cut = text.prefix(limit)
        return cut + "\n… [truncated \(text.count - limit) characters]"
    }

    private nonisolated static func sortDate(_ record: AgentRunRecord) -> Date {
        record.completedAt ?? record.startedAt ?? record.createdAt
    }
}
