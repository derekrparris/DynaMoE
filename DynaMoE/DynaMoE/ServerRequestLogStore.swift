//
//  ServerRequestLogStore.swift
//  DynaMoE
//
//  Persisted append-only store for ServerRequestLog rows: one SQLite database
//  at `~/Library/Application Support/DynaMoE/logs/server-requests.sqlite`.
//  Rows are written on a private serial queue (fire-and-forget from the
//  request path), stored as JSON payloads plus the few filterable columns,
//  and pruned by age and count on open and periodically on append so the
//  database can never grow without bound. Store-open failure is non-fatal:
//  the server keeps serving and the unified log keeps recording; the store
//  logs once and silently no-ops (matching ChatSessionStore's degradation
//  philosophy).
//
//  Deliberately plain append-only — no hash chain. This is a single-user
//  local server; retention (30 days / 20,000 rows) replaces tamper evidence
//  as the maintenance policy.
//

import Foundation
import SQLite3

/// Swift's SQLite3 module drops the SQLITE_TRANSIENT macro; this is its
/// value (-1 reinterpreted), telling sqlite to copy bound strings/blobs
/// immediately so the caller's buffers can go out of scope.
private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

final class ServerRequestLogStore: @unchecked Sendable {
    static let shared = ServerRequestLogStore()

    /// Retention: rows older than this are pruned.
    static let retentionDays = 30
    /// Retention: hard cap; oldest rows are dropped beyond it.
    static let maxRows = 20_000
    /// Prune amortization: roll the retention pass every N appended rows.
    private static let pruneEveryAppends = 256
    /// Compaction threshold: retention passes that delete at least this many
    /// rows finish with a VACUUM so the database file itself shrinks. Below
    /// the threshold the freelist keeps the pages warm for append reuse.
    static let compactAfterRowsDeleted = 2_000

    /// Coarse filters over the indexed columns plus a payload text search.
    struct Filter: Sendable, Equatable {
        var errorsOnly = false
        var streamingOnly = false
        var path: String?
        /// Free-text match across the whole JSON payload (model, error
        /// message, tool names, bodies when captured).
        var text = ""

        static let empty = Filter()
    }

    private let queue = DispatchQueue(label: "DRP.DynaMoE.server-request-log")
    private var db: OpaquePointer?
    private let databasePath: String?
    private var openFailureLogged = false
    /// Sticky once the schema pass rejects the database (or itself fails): a
    /// file this build cannot validate must never be written to or pruned
    /// again this session. The store stays a silent no-op (open failures are
    /// non-fatal by design) until the next launch retries the open.
    private var schemaRejected = false
    private var appendCounter = 0
    /// Change observers (the dashboard's log viewer): one signal per append,
    /// clear, or retention prune, so the viewer never polls.
    private let observerLock = NSLock()
    private var observers: [UUID: AsyncStream<Void>.Continuation] = [:]

    /// Path resolution follows ChatSessionStore's convention so the log
    /// lives beside session data (durable, backed up) — not in Caches.
    nonisolated static func defaultDatabasePath() -> String? {
        do {
            let base = try FileManager.default.url(
                for: .applicationSupportDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: true
            )
            let directory = base
                .appendingPathComponent("DynaMoE/logs", isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            return directory
                .appendingPathComponent("server-requests.sqlite")
                .path
        } catch {
            // Never fatal: the server keeps running, unified log still works.
            return nil
        }
    }

    /// The shared on-disk store.
    private init() {
        self.databasePath = Self.defaultDatabasePath()
    }

    /// In-memory store for tests: opens synchronously on the store queue.
    nonisolated static func inMemory() -> ServerRequestLogStore {
        Self.init(path: nil)
    }

    init(path: String?) {
        self.databasePath = path
    }

    deinit {
        if let db {
            sqlite3_close(db)
        }
    }

    private var encoder: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .millisecondsSince1970
        return encoder
    }

    private var decoder: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .millisecondsSince1970
        return decoder
    }

    // MARK: Lifecycle

    /// Opens the database and runs the schema migration + initial retention
    /// pass. Runs on the store queue only. Idempotent.
    private func ensureOpen() {
        guard db == nil, !schemaRejected else { return }
        guard let path = databasePath else {
            // In-memory test store: register the shared in-memory database.
            if sqlite3_open(":memory:", &db) == SQLITE_OK {
                try? runSchema()
                openFailureLogged = false
            }
            return
        }

        var handle: OpaquePointer?
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX
        guard sqlite3_open_v2(path, &handle, flags, nil) == SQLITE_OK, let handle else {
            if !openFailureLogged {
                openFailureLogged = true
                DiagLog.error("server.store", "failed to open request log database at \(path): \(String(cString: sqlite3_errmsg(handle))))")
            }
            sqlite3_close(handle)
            return
        }
        db = handle
        sqlite3_busy_timeout(db, 1_000)
        _ = exec("PRAGMA journal_mode=WAL")
        _ = exec("PRAGMA synchronous=NORMAL")
        do {
            try runSchema()
            openFailureLogged = false
        } catch {
            if !openFailureLogged {
                openFailureLogged = true
                DiagLog.error("server.store", "request log schema setup failed: \(error.localizedDescription)")
            }
            // A rejected database (schema from a newer app, or a schema pass
            // that failed outright) must not fall through to the pruning pass
            // or serve later writes: the handle would stay open, skip
            // validation on every later call, and let this build delete or
            // overwrite rows in a database it cannot understand. Close the
            // handle and leave it closed for the whole session.
            sqlite3_close(db)
            db = nil
            schemaRejected = true
            return
        }
        pruneLocked()
    }

    /// Test-facing retention pass against a controllable clock: prunes rows
    /// older than the retention window measured from `now`, then trims to the
    /// row cap.
    nonisolated func pruneForTesting(now: Date = Date()) async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            queue.async { [weak self] in
                guard let self else {
                    continuation.resume()
                    return
                }
                self.ensureOpen()
                self.pruneLocked(now: now)
                continuation.resume()
            }
        }
    }

    /// Test-facing read of the persisted schema version, so tests can assert
    /// `PRAGMA user_version` survives store opens: the schema pass must run
    /// once per file, not again on every open.
    nonisolated func schemaVersionForTesting() async -> Int64 {
        await withCheckedContinuation { (continuation: CheckedContinuation<Int64, Never>) in
            queue.async { [weak self] in
                guard let self else {
                    continuation.resume(returning: 0)
                    return
                }
                self.ensureOpen()
                continuation.resume(returning: self.userVersion())
            }
        }
    }

    /// Schema versioning via PRAGMA user_version, so a future column change
    /// has a migration base even though v1 is the first.
    private func runSchema() throws {
        let current = userVersion()
        if current > 1 {
            throw StoreError.schemaNewerThanSupported
        }
        if userVersion() < 1 {
            try execChecked("""
                CREATE TABLE IF NOT EXISTS requests (
                    id           TEXT PRIMARY KEY,
                    ts_ms        INTEGER NOT NULL,
                    path         TEXT NOT NULL,
                    http_status  INTEGER NOT NULL,
                    is_error     INTEGER NOT NULL,
                    was_streaming INTEGER NOT NULL,
                    model        TEXT,
                    payload      TEXT NOT NULL
                )
                """)
            try execChecked("CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts_ms)")
            try execChecked("CREATE INDEX IF NOT EXISTS idx_requests_is_error ON requests(is_error)")
            try execChecked("PRAGMA user_version = 1")
        }
    }

    private enum StoreError: LocalizedError {
        case schemaNewerThanSupported
        case exportFailed(String)

        var errorDescription: String? {
            switch self {
            case .schemaNewerThanSupported:
                return "on-disk request log schema is newer than this app supports"
            case .exportFailed(let message):
                return "request log export failed: \(message)"
            }
        }
    }

    /// Reads `PRAGMA user_version`. `sqliteRun` positions the statement on
    /// each row before invoking the callback, so the callback reads the
    /// column without stepping again: stepping here consumes the only row's
    /// SQLITE_DONE and, on SQLite builds that auto-reset, restarts the query
    /// so the outer loop replays the row forever on the store queue.
    private func userVersion() -> Int64 {
        var version: Int64 = 0
        sqliteRun("PRAGMA user_version") { statement in
            version = sqlite3_column_int64(statement, 0)
        }
        return version
    }

    private func exec(_ sql: String) -> Bool {
        var errMsg: UnsafeMutablePointer<CChar>?
        let result = sqlite3_exec(db, sql, nil, nil, &errMsg)
        if let errMsg {
            sqlite3_free(errMsg)
        }
        return result == SQLITE_OK
    }

    private func execChecked(_ sql: String) throws {
        guard exec(sql) else {
            throw NSError(
                domain: "DRP.DynaMoE.ServerRequestLogStore",
                code: Int(sqlite3_errcode(db)),
                userInfo: [NSLocalizedDescriptionKey: String(cString: sqlite3_errmsg(db))]
            )
        }
    }

    /// Prepares, binds nothing, steps through a SELECT; the callback sees the
    /// prepared statement per row.
    private func sqliteRun(_ sql: String, _ body: (OpaquePointer) -> Void) {
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK, let statement else {
            sqlite3_finalize(statement)
            return
        }
        defer { sqlite3_finalize(statement) }
        while sqlite3_step(statement) == SQLITE_ROW {
            body(statement)
        }
    }

    /// Runs a parameterless mutation.
    private func sqliteExec(_ sql: String) -> Bool {
        var statement: OpaquePointer?
        defer { sqlite3_finalize(statement) }
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK else { return false }
        return sqlite3_step(statement) == SQLITE_DONE
    }

    // MARK: Append

    /// Persists one row. Fire-and-forget: dispatches to the store queue; the
    /// caller (the request path) never waits on disk.
    nonisolated func append(_ row: ServerRequestLog) {
        queue.async { [weak self] in
            self?.appendLocked(row)
        }
    }

    private func appendLocked(_ row: ServerRequestLog) {
        ensureOpen()
        guard db != nil else { return }
        guard let payload = try? encoder.encode(row) else {
            DiagLog.error("server.store", "failed to encode request log row \(row.id.uuidString) for persistence")
            return
        }
        var statement: OpaquePointer?
        let sql = """
            INSERT OR REPLACE INTO requests
                (id, ts_ms, path, http_status, is_error, was_streaming, model, payload)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            """
        defer { sqlite3_finalize(statement) }
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK, let statement else { return }
        let id = row.id.uuidString
        let tsMs = Int64(row.timestamp.timeIntervalSince1970 * 1000)
        let path = row.path
        let status = Int64(row.httpStatus)
        let isError = row.isError ? Int64(1) : 0
        let streaming = row.wasStreaming ? Int64(1) : 0
        let model = row.model
        sqlite3_bind_text(statement, 1, id, -1, SQLITE_TRANSIENT)
        sqlite3_bind_int64(statement, 2, tsMs)
        sqlite3_bind_text(statement, 3, path, -1, SQLITE_TRANSIENT)
        sqlite3_bind_int64(statement, 4, status)
        sqlite3_bind_int64(statement, 5, isError)
        sqlite3_bind_int64(statement, 6, streaming)
        if let model {
            sqlite3_bind_text(statement, 7, model, -1, SQLITE_TRANSIENT)
        } else {
            sqlite3_bind_null(statement, 7)
        }
        var bound = false
        payload.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            bound = sqlite3_bind_blob(statement, 8, base, Int32(payload.count), SQLITE_TRANSIENT) == SQLITE_OK
        }
        guard bound, sqlite3_step(statement) == SQLITE_DONE else { return }
        appendCounter += 1
        if appendCounter % Self.pruneEveryAppends == 0 {
            pruneLocked()
        }
        notifyObservers()
    }

    // MARK: Retention

    /// Age- and count-based prune. Runs on the store queue (open-time and
    /// amortized append-time); NOT exposed as a user action. Bulk deletes
    /// finish with a WAL checkpoint so freed pages reach the OS instead of
    /// lingering in the `-wal` file; passes substantial enough to leave the
    /// database mostly empty also VACUUM, so the main file shrinks too and
    /// not just the WAL.
    private func pruneLocked(now: Date = Date()) {
        let cutoffMs = Int64(now.timeIntervalSince1970 * 1000) - Int64(Self.retentionDays) * 86_400_000
        _ = sqliteExec("DELETE FROM requests WHERE ts_ms < \(cutoffMs)")
        let agePrunedRows = Int(sqlite3_changes(db))
        let cap = Self.maxRows
        // -1 LIMIT with OFFSET keeps OFFSET-skip semantics in SQLite.
        _ = sqliteExec("""
            DELETE FROM requests WHERE id IN (
                SELECT id FROM requests
                ORDER BY ts_ms DESC, rowid DESC
                LIMIT -1 OFFSET \(cap)
            )
            """)
        let capPrunedRows = Int(sqlite3_changes(db))
        checkpointLocked()
        if agePrunedRows + capPrunedRows >= Self.compactAfterRowsDeleted {
            compactLocked()
        }
        notifyObservers()
    }

    /// Writes dirty WAL pages back to the database and truncates the `-wal`
    /// file to zero. In WAL journal mode a plain DELETE leaves freed pages in
    /// the `-wal` file until a checkpoint, so without this a cleared log
    /// would keep its size in the WAL.
    private func checkpointLocked() {
        _ = exec("PRAGMA wal_checkpoint(TRUNCATE)")
    }

    /// Rebuilds the database so freed pages leave the main file. DELETE hands
    /// pages to the freelist *inside* the database, and a TRUNCATE checkpoint
    /// only empties the WAL: without a VACUUM, a cleared or mass-pruned log
    /// keeps its peak database-file footprint forever. In WAL mode the
    /// rebuilt image travels through the journal, so a trailing checkpoint
    /// is what actually shrinks the file on disk — vacuum alone leaves both
    /// files at their old sizes.
    private func compactLocked() {
        _ = exec("VACUUM")
        checkpointLocked()
    }

    // MARK: Queries

    /// Most recent rows first, matching the retention cap's view of the log.
    nonisolated func fetch(filter: Filter = .empty, limit: Int = 200, offset: Int = 0) async -> [ServerRequestLog] {
        await withCheckedContinuation { continuation in
            queue.async { [weak self] in
                self?.fetchLocked(filter: filter, limit: limit, offset: offset) { rows in
                    continuation.resume(returning: rows)
                }
            } ?? continuation.resume(returning: [])
        }
    }

    /// The filter's WHERE clauses + binds, shared by `fetchLocked` and the
    /// JSONL export so the two can never drift apart on filter semantics.
    private func filterClauses(_ filter: Filter) -> (whereSQL: String, binds: [Bind]) {
        var clauses: [String] = []
        var binds: [Bind] = []

        func add(_ bind: Bind) -> String {
            binds.append(bind)
            return "?\(binds.count)"
        }
        if filter.errorsOnly {
            clauses.append("is_error = 1")
        }
        if filter.streamingOnly {
            clauses.append("was_streaming = 1")
        }
        if let path = filter.path, !path.isEmpty {
            clauses.append("path = \(add(.text(path)))")
        }
        let text = filter.text.trimmingCharacters(in: .whitespacesAndNewlines)
        if !text.isEmpty {
            // The payload column is JSON, so the term matches only in its
            // stored form: displayed text `invalid "name"` persists as
            // `invalid \"name\"`. Captured API request bodies add one more
            // JSON layer still — the request summary is itself serialized
            // JSON embedded as a string value — so the same displayed text
            // is escaped yet again there (backslashes doubled, quotes
            // re-escaped). Match every encoding level the term can sit at
            // (raw JSON structure, row-level string fields, embedded request
            // summaries), then LIKE-escape each pattern — the encoder's own
            // backslashes first, so `\"` stays literal.
            let likePatterns = jsonEncodedSearchFragments(text).map { fragment in
                let escaped = fragment
                    .replacingOccurrences(of: "\\", with: "\\\\")
                    .replacingOccurrences(of: "%", with: "\\%")
                    .replacingOccurrences(of: "_", with: "\\_")
                return "payload LIKE \(add(.text("%\(escaped)%"))) ESCAPE '\\'"
            }
            clauses.append(likePatterns.count == 1 ? likePatterns[0] : "(" + likePatterns.joined(separator: " OR ") + ")")
        }

        let whereSQL = clauses.isEmpty ? "" : " WHERE " + clauses.joined(separator: " AND ")
        return (whereSQL, binds)
    }

    /// The search term as it appears inside a stored payload: a JSON-encoded
    /// string minus its surrounding quotes, i.e. the exact bytes the term
    /// occupies once a row is persisted. Falls back to the raw term if
    /// encoding fails (it cannot for valid Swift strings).
    private func jsonEncodedSearchFragment(_ text: String) -> String {
        guard let data = try? encoder.encode(text),
              let quoted = String(data: data, encoding: .utf8),
              quoted.hasPrefix("\""), quoted.hasSuffix("\""),
              quoted.count >= 2 else { return text }
        return String(quoted.dropFirst().dropLast())
    }

    /// The term at every JSON-encoding level it can occupy in a stored
    /// payload: the raw term (JSON structure fragments such as `"role":`
    /// appear unescaped in the payload), the term once-encoded (row-level
    /// string fields), and twice-encoded (the request summaries embedded as
    /// strings inside `requestBody`). Encoding stabilizes for plain terms,
    /// collapsing to a single pattern; the ladder stops as soon as a level
    /// repeats.
    private func jsonEncodedSearchFragments(_ text: String) -> [String] {
        var fragments: [String] = []
        var current = text
        for _ in 0..<3 {
            if !fragments.contains(current) { fragments.append(current) }
            let encoded = jsonEncodedSearchFragment(current)
            if encoded == current { break }
            current = encoded
        }
        return fragments
    }

    private func fetchLocked(filter: Filter, limit: Int, offset: Int, completion: ([ServerRequestLog]) -> Void) {
        ensureOpen()
        guard db != nil else {
            completion([])
            return
        }
        let (whereSQL, binds) = filterClauses(filter)
        let sql = "SELECT payload FROM requests\(whereSQL) ORDER BY ts_ms DESC, rowid DESC LIMIT \(max(0, limit)) OFFSET \(max(0, offset))"
        var rows: [ServerRequestLog] = []
        var statement: OpaquePointer?
        defer { sqlite3_finalize(statement) }
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK else {
            completion([])
            return
        }
        for (index, bind) in binds.enumerated() {
            switch bind {
            case .text(let value):
                sqlite3_bind_text(statement, Int32(index + 1), value, -1, SQLITE_TRANSIENT)
            }
        }
        let decoder = decoder
        while sqlite3_step(statement) == SQLITE_ROW {
            guard let blob = sqlite3_column_blob(statement, 0) else { continue }
            let count = Int(sqlite3_column_bytes(statement, 0))
            let data = Data(bytes: blob, count: count)
            if let row = try? decoder.decode(ServerRequestLog.self, from: data) {
                rows.append(row)
            }
        }
        completion(rows)
    }

    nonisolated func count() async -> Int {
        await withCheckedContinuation { continuation in
            queue.async { [weak self] in
                guard let self else {
                    continuation.resume(returning: 0)
                    return
                }
                self.ensureOpen()
                guard self.db != nil else {
                    continuation.resume(returning: 0)
                    return
                }
                var count = 0
                self.sqliteRun("SELECT COUNT(*) FROM requests") { statement in
                    count = Int(sqlite3_column_int64(statement, 0))
                }
                continuation.resume(returning: count)
            }
        }
    }

    /// Removes every row, then checkpoints the WAL and compacts the
    /// database so the freed space is actually returned to the OS (a plain
    /// DELETE would stay in the `-wal` file, its pages in the main file's
    /// freelist). The Server dashboard's Clear Log button.
    nonisolated func clear() async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            queue.async { [weak self] in
                guard let self else {
                    continuation.resume()
                    return
                }
                self.ensureOpen()
                _ = self.sqliteExec("DELETE FROM requests")
                self.checkpointLocked()
                self.compactLocked()
                self.notifyObservers()
                continuation.resume()
            }
        }
    }

    private enum Bind {
        case text(String)
    }

    // MARK: Export

    /// Writes every row matching `filter` to `url` as JSON Lines: one JSON
    /// object per line, byte-identical to each row's persisted payload. The
    /// dashboard's Export button saves the filtered log (bodies included only
    /// per the capture toggle) for bug reports on client-format issues.
    /// Returns the number of exported rows.
    nonisolated func exportJSONL(filter: Filter = .empty, to url: URL) async throws -> Int {
        try await withCheckedThrowingContinuation { continuation in
            queue.async { [weak self] in
                do {
                    guard let self else {
                        continuation.resume(throwing: StoreError.exportFailed("request log store is unavailable"))
                        return
                    }
                    continuation.resume(returning: try self.exportJSONLLocked(filter: filter, to: url))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private func exportJSONLLocked(filter: Filter, to url: URL) throws -> Int {
        ensureOpen()
        guard db != nil else {
            throw StoreError.exportFailed("request log database is unavailable")
        }
        let (whereSQL, binds) = filterClauses(filter)
        let sql = "SELECT payload FROM requests\(whereSQL) ORDER BY ts_ms DESC, rowid DESC"
        var statement: OpaquePointer?
        defer { sqlite3_finalize(statement) }
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK else {
            throw StoreError.exportFailed(String(cString: sqlite3_errmsg(db)))
        }
        for (index, bind) in binds.enumerated() {
            if case .text(let value) = bind {
                sqlite3_bind_text(statement, Int32(index + 1), value, -1, SQLITE_TRANSIENT)
            }
        }

        // Stage the export beside the destination so a failure never touches
        // it: only a complete export moves onto `url`, a mid-write error
        // leaves the previous file intact, and no empty or partial file is
        // ever left standing in for a bug-report attachment.
        let stagingURL = url.deletingLastPathComponent()
            .appendingPathComponent(".\(UUID().uuidString)-export.tmp")
        guard FileManager.default.createFile(atPath: stagingURL.path, contents: nil) else {
            throw StoreError.exportFailed("cannot create staging file at \(stagingURL.path)")
        }
        let encoder = encoder
        let decoder = decoder
        var exported = 0
        do {
            let handle = try FileHandle(forWritingTo: stagingURL)
            defer { try? handle.close() }
            while true {
                let result = sqlite3_step(statement)
                if result == SQLITE_DONE { break }
                guard result == SQLITE_ROW else {
                    // Busy, I/O, and corruption errors are failures, not a
                    // complete export: only SQLITE_DONE means every matching
                    // row made it out.
                    throw StoreError.exportFailed(String(cString: sqlite3_errmsg(db)))
                }
                guard let blob = sqlite3_column_blob(statement, 0) else { continue }
                let payload = Data(bytes: blob, count: Int(sqlite3_column_bytes(statement, 0)))
                // A row that cannot decode (schema drift from an older build) is
                // skipped, matching fetch: the export succeeds with what is there.
                guard let row = try? decoder.decode(ServerRequestLog.self, from: payload) else { continue }
                try handle.write(contentsOf: encoder.encode(row) + Data("\n".utf8))
                exported += 1
            }
        } catch {
            try? FileManager.default.removeItem(at: stagingURL)
            throw error
        }
        do {
            if FileManager.default.fileExists(atPath: url.path) {
                _ = try FileManager.default.removeItem(at: url)
            }
            try FileManager.default.moveItem(at: stagingURL, to: url)
        } catch {
            try? FileManager.default.removeItem(at: stagingURL)
            throw StoreError.exportFailed("cannot move export onto \(url.path): \(error.localizedDescription)")
        }
        return exported
    }

    // MARK: Change observation

    /// A stream that signals whenever stored rows change (append, clear,
    /// retention prune). The dashboard's request-log viewer listens instead
    /// of polling: it refetches only when the log actually changed. Buffered
    /// to the single newest element, so a burst of appends collapses into
    /// one refresh.
    nonisolated func changes() -> AsyncStream<Void> {
        let id = UUID()
        let (stream, continuation) = AsyncStream<Void>.makeStream(of: Void.self, bufferingPolicy: .bufferingNewest(1))
        observerLock.lock()
        observers[id] = continuation
        observerLock.unlock()
        continuation.onTermination = { [weak self] _ in
            guard let self else { return }
            self.observerLock.lock()
            self.observers[id] = nil
            self.observerLock.unlock()
        }
        return stream
    }

    /// Wakes every listener. Called on the store queue, where all row
    /// mutations happen; yields never block the fire-and-forget appends.
    private func notifyObservers() {
        observerLock.lock()
        let continuations = Array(observers.values)
        observerLock.unlock()
        for continuation in continuations {
            continuation.yield(())
        }
    }
}
