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
    private var appendCounter = 0

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
        guard db == nil else { return }
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

        var errorDescription: String? {
            switch self {
            case .schemaNewerThanSupported:
                return "on-disk request log schema is newer than this app supports"
            }
        }
    }

    private func userVersion() -> Int64 {
        var version: Int64 = 0
        sqliteRun("PRAGMA user_version") { statement in
            if sqlite3_step(statement) == SQLITE_ROW {
                version = sqlite3_column_int64(statement, 0)
            }
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
    }

    // MARK: Retention

    /// Age- and count-based prune. Runs on the store queue (open-time and
    /// amortized append-time); NOT exposed as a user action. Bulk deletes
    /// finish with a WAL checkpoint so freed pages reach the OS instead of
    /// lingering in the `-wal` file.
    private func pruneLocked(now: Date = Date()) {
        let cutoffMs = Int64(now.timeIntervalSince1970 * 1000) - Int64(Self.retentionDays) * 86_400_000
        _ = sqliteExec("DELETE FROM requests WHERE ts_ms < \(cutoffMs)")
        let cap = Self.maxRows
        // -1 LIMIT with OFFSET keeps OFFSET-skip semantics in SQLite.
        _ = sqliteExec("""
            DELETE FROM requests WHERE id IN (
                SELECT id FROM requests
                ORDER BY ts_ms DESC, rowid DESC
                LIMIT -1 OFFSET \(cap)
            )
            """)
        checkpointLocked()
    }

    /// Writes dirty WAL pages back to the database and truncates the `-wal`
    /// file to zero, returning the space to the OS. In WAL journal mode a
    /// plain DELETE leaves freed pages in the `-wal` file until a checkpoint,
    /// so without this a cleared log would keep its size on disk.
    private func checkpointLocked() {
        _ = exec("PRAGMA wal_checkpoint(TRUNCATE)")
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

    private func fetchLocked(filter: Filter, limit: Int, offset: Int, completion: ([ServerRequestLog]) -> Void) {
        ensureOpen()
        guard db != nil else {
            completion([])
            return
        }
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
            // Escape LIKE wildcards in the search term; % as the infix what
            // a user expects: substring match across the JSON payload.
            let escaped = text
                .replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "%", with: "\\%")
                .replacingOccurrences(of: "_", with: "\\_")
            clauses.append("payload LIKE \(add(.text("%\(escaped)%"))) ESCAPE '\\'")
        }

        let whereSQL = clauses.isEmpty ? "" : " WHERE " + clauses.joined(separator: " AND ")
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

    /// Removes every row, then checkpoints the WAL so the freed space is
    /// actually returned to the OS (a plain DELETE would stay in the `-wal`
    /// file). The Server dashboard's Clear Log button.
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
                continuation.resume()
            }
        }
    }

    private enum Bind {
        case text(String)
    }
}
