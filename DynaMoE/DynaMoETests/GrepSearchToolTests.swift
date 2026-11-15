//
//  GrepSearchToolTests.swift
//  DynaMoE
//
//  The grep_search contract, pinned without a model. Its command is built and
//  executed on the host, so every case here is pure logic: real files in a
//  temporary directory, the real tool, and assertions on the payload the model
//  would receive.
//
//  A quiet failure is the defect these cases exist for. `rg` is absent on many
//  machines, the tool falls back to `grep`, and a search that never ran must
//  never be reported as "0 results".
//

import XCTest
@testable import DynaMoE

final class GrepSearchToolTests: XCTestCase {

    private var root: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("grep_search_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let root {
            try? FileManager.default.removeItem(at: root)
        }
        root = nil
        try super.tearDownWithError()
    }

    // MARK: - Fixtures

    @discardableResult
    private func writeFile(_ text: String, to relativePath: String) throws -> URL {
        let url = root.appendingPathComponent(relativePath)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try text.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    private static let grepBackend = GrepSearchTool.SearchBackend.grep(URL(fileURLWithPath: "/usr/bin/grep"))

    /// Every backend this machine can run. `nil` is the tool's own resolution,
    /// so a machine with ripgrep covers both real backends and one without it
    /// covers grep twice instead of skipping the contract.
    private func backends() -> [GrepSearchTool.SearchBackend?] {
        var available: [GrepSearchTool.SearchBackend?] = [nil, Self.grepBackend]
        if let ripgrep = Self.ripgrepBinary() {
            available.append(.ripgrep(ripgrep))
        }
        return available
    }

    private func label(_ backend: GrepSearchTool.SearchBackend?) -> String {
        guard let backend else {
            return "auto(\(Self.ripgrepBinary() == nil ? "grep" : "ripgrep"))"
        }
        return backend.isRipgrep ? "ripgrep" : "grep"
    }

    @MainActor
    private func runTool(
        _ arguments: [String: Any],
        maxOutputLength: Int = 4_000,
        backend: GrepSearchTool.SearchBackend? = nil
    ) async throws -> (json: String, stdout: String?) {
        let tool = GrepSearchTool(backendOverride: backend)
        let result = try await tool.execute(
            arguments: arguments,
            workingDirectory: root,
            maxOutputLength: maxOutputLength
        )
        return (json: result.resultJSON, stdout: result.stdout)
    }

    private func gitInit() throws {
        let git = "/usr/bin/git"
        try XCTSkipUnless(FileManager.default.isExecutableFile(atPath: git), "git is unavailable")
        let process = Process()
        process.executableURL = URL(fileURLWithPath: git)
        process.arguments = ["init", "-q"]
        process.currentDirectoryURL = root
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 0)
    }

    /// ripgrep when this machine has it, else a binary named by a marker file.
    /// `xcodebuild` does not forward the shell environment into an app-hosted
    /// test, so a variable cannot carry the path; the marker file is the seam
    /// the DeepSeek measurement cases already use.
    private static func ripgrepBinary() -> URL? {
        let marker = "/tmp/dynamoe_grep_rg_path"
        if let data = FileManager.default.contents(atPath: marker),
           let text = String(data: data, encoding: .utf8) {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty, FileManager.default.isExecutableFile(atPath: trimmed) {
                return URL(fileURLWithPath: trimmed)
            }
        }
        return GrepSearchTool.ripgrepExecutable()
    }

    private func object(in json: String) throws -> [String: Any] {
        let data = try XCTUnwrap(json.data(using: .utf8))
        return try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private func resultObject(in json: String) throws -> [String: Any] {
        let obj = try object(in: json)
        XCTAssertEqual(obj["status"] as? String, "success", "expected a success payload, got: \(json)")
        return try XCTUnwrap(obj["result"] as? [String: Any])
    }

    private func resultLines(in json: String) throws -> [String] {
        try XCTUnwrap(resultObject(in: json)["results"] as? [String])
    }

    private func assertToolError(
        _ json: String,
        context: String = "",
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let suffix = context.isEmpty ? "" : " [\(context)]"
        let obj = try object(in: json)
        XCTAssertEqual(obj["status"] as? String, "error", "expected a tool error\(suffix), got: \(json)", file: file, line: line)
        let error = try XCTUnwrap(obj["error"] as? String, file: file, line: line)
        XCTAssertFalse(error.isEmpty, "error message must name the failure\(suffix)", file: file, line: line)
    }

    // MARK: - Matches

    func testFindsAMatchInTheWorkingDirectory() async throws {
        try writeFile("let answer = 42\n", to: "Sources/a.swift")
        try writeFile("nothing to see\n", to: "Sources/b.swift")

        let (json, stdout) = try await runTool(["query": "let answer = 42"])

        let result = try resultObject(in: json)
        XCTAssertEqual(result["count"] as? Int, 1)
        XCTAssertEqual(result["query"] as? String, "let answer = 42")
        let lines = try resultLines(in: json)
        XCTAssertEqual(lines.count, 1)
        let line = try XCTUnwrap(lines.first, "expected one result line, got \(lines)")
        XCTAssertTrue(line.contains("a.swift"), "line must name the file: \(line)")
        XCTAssertTrue(line.contains("1:"), "line must carry the line number: \(line)")
        XCTAssertTrue(stdout?.contains("a.swift") == true)
    }

    func testNoMatchIsAnEmptySuccess() async throws {
        try writeFile("alpha beta\n", to: "a.swift")

        let (json, _) = try await runTool(["query": "gamma"])

        let result = try resultObject(in: json)
        XCTAssertEqual(result["count"] as? Int, 0)
        XCTAssertEqual(try resultLines(in: json), [])
    }

    func testCaseSensitivityToggle() async throws {
        try writeFile("Needle in a haystack\n", to: "a.swift")

        let insensitive = try await runTool(["query": "needle"])
        XCTAssertEqual(try resultObject(in: insensitive.json)["count"] as? Int, 1)

        let sensitive = try await runTool(["query": "needle", "case_insensitive": false])
        XCTAssertEqual(try resultObject(in: sensitive.json)["count"] as? Int, 0)
    }

    // MARK: - The search actually runs

    /// A query containing the one character POSIX single quotes cannot carry.
    /// Escaping it as `\'` ends the quoted word, so the old shell command died
    /// with "unmatched '" and the tool reported zero results.
    func testApostropheQueryIsSearchedLiterally() async throws {
        try writeFile("print(\"don't stop\")\n", to: "a.swift")

        let (json, _) = try await runTool(["query": "don't"])

        XCTAssertEqual(try resultObject(in: json)["count"] as? Int, 1)
    }

    /// The same broken quoting was a command-injection hole: a quote closes the
    /// pattern and everything after it is shell.
    func testQueryIsNotInterpretedByAShell() async throws {
        let marker = root.appendingPathComponent("injected")
        try writeFile("harmless\n", to: "a.swift")

        let (json, _) = try await runTool(["query": "x'; touch \(marker.path); echo '"])

        XCTAssertFalse(FileManager.default.fileExists(atPath: marker.path), "query reached the shell")
        XCTAssertEqual(try resultObject(in: json)["count"] as? Int, 0)
    }

    /// `foo(bar` is an unterminated group, so both backends exit 2 with
    /// nothing on stdout. Discarding the status turned that into "0 results",
    /// and refusing the query outright would drop a literal code fragment an
    /// agent pasted verbatim (`-E` is stricter than the old BRE fallback, which
    /// matched `foo(` as text). The query is retried as literal text instead,
    /// and the payload says so rather than silently changing the dialect.
    func testAMetacharacterQueryFallsBackToALiteralSearch() async throws {
        try writeFile("if (foo(bar)) {\n", to: "a.swift")

        for backend in backends() {
            let (json, _) = try await runTool(["query": "foo(bar"], backend: backend)

            let result = try resultObject(in: json)
            XCTAssertEqual(result["count"] as? Int, 1, "\(label(backend)) lost a literal match: \(json)")
            XCTAssertTrue(
                try XCTUnwrap(try resultLines(in: json).first).contains("a.swift"),
                "\(label(backend)) must name the file it matched"
            )
            let warning = try XCTUnwrap(
                result["warning"] as? String,
                "\(label(backend)) must say the pattern was not a regex: \(json)"
            )
            XCTAssertTrue(warning.contains("literal"), "\(label(backend)) said: \(warning)")
        }
    }

    /// A literal retry that finds nothing too is a real empty result, not an
    /// error, and the payload still explains which reading it used.
    func testALiteralRetryWithNoMatchIsAnEmptySuccess() async throws {
        try writeFile("alpha beta\n", to: "a.swift")

        let (json, _) = try await runTool(["query": "foo("], backend: Self.grepBackend)

        let result = try resultObject(in: json)
        XCTAssertEqual(result["count"] as? Int, 0)
        XCTAssertNotNil(result["warning"] as? String, "the payload must still name the literal reading")
    }

    func testMissingPathIsReportedAsAnError() async throws {
        let (json, _) = try await runTool([
            "query": "anything",
            "path": root.appendingPathComponent("does-not-exist").path
        ])

        try assertToolError(json)
    }

    // MARK: - Path handling

    /// `grep -r` prints the filename for a single file operand, but ripgrep
    /// defaults to `--no-filename` when explicitly given one file. Pin the
    /// `path:line:text` contract on both backends.
    func testSearchesASingleFilePathOnEitherBackend() async throws {
        try writeFile("needle in this file\n", to: "chosen.swift")
        try writeFile("needle in another file\n", to: "other.swift")

        for backend in backends() {
            let (json, _) = try await runTool([
                "query": "needle",
                "path": root.appendingPathComponent("chosen.swift").path
            ], backend: backend)

            let lines = try resultLines(in: json)
            XCTAssertEqual(lines.count, 1, "\(label(backend)) must search that file only: \(lines)")
            let line = try XCTUnwrap(lines.first)
            XCTAssertTrue(line.hasPrefix("./chosen.swift:1:"), "\(label(backend)) printed: \(line)")
        }
    }

    func testSearchesASubdirectoryPath() async throws {
        try writeFile("needle deep\n", to: "Sub/deep.txt")
        try writeFile("needle shallow\n", to: "shallow.txt")

        let (json, _) = try await runTool([
            "query": "needle",
            "path": "Sub"
        ])

        let lines = try resultLines(in: json)
        XCTAssertEqual(lines.count, 1)
        let line = try XCTUnwrap(lines.first)
        XCTAssertTrue(line.contains("deep.txt"))
    }

    /// Ignored is not absent. Local docs and instruction files are commonly
    /// gitignored, and an agent that searched them used to get a silent zero.
    func testGitIgnoredFileIsSearchedOnEitherBackend() async throws {
        try writeFile("NOTES.md\n", to: ".gitignore")
        try writeFile("the ignored instruction\n", to: "NOTES.md")
        try gitInit()

        for backend in backends() {
            let (json, _) = try await runTool(
                ["query": "ignored instruction"],
                backend: backend
            )

            XCTAssertEqual(
                try resultObject(in: json)["count"] as? Int,
                1,
                "\(label(backend)) skipped a git-ignored file"
            )
        }
    }

    func testHiddenEntriesAreNotSearchedOnEitherBackend() async throws {
        try writeFile("needle only in a dotfile\n", to: ".hidden.txt")
        try writeFile("needle in a nested dotfile\n", to: "Sub/.nested.txt")
        try writeFile("needle in a hidden directory\n", to: "Sub/.hidden/x.txt")
        try writeFile("needle in a visible file\n", to: "Sub/visible.txt")

        for backend in backends() {
            let (json, _) = try await runTool(["query": "needle"], backend: backend)

            let lines = try resultLines(in: json)
            XCTAssertEqual(lines.count, 1, "\(label(backend)) searched a hidden entry: \(lines)")
            XCTAssertTrue(try XCTUnwrap(lines.first).contains("visible.txt"))
        }
    }

    func testTruncationStillReportsSuccess() async throws {
        let lines = (1...80).map { "needle line \($0)" }.joined(separator: "\n")
        try writeFile(lines, to: "many.txt")

        let (json, _) = try await runTool(["query": "needle line"], maxOutputLength: 400)

        let result = try resultObject(in: json)
        XCTAssertEqual(result["truncated"] as? Bool, true)
        let reported = try resultLines(in: json)
        XCTAssertGreaterThan(reported.count, 0)
        XCTAssertFalse(
            reported.contains { $0.contains("<<<TRUNCATED:") },
            "the truncation marker is metadata, not a match: \(reported)"
        )
        XCTAssertEqual(result["count"] as? Int, reported.count)
    }

    // MARK: - grep fallback

    func testTheGrepFallbackFindsMatches() async throws {
        try writeFile("let answer = 42\n", to: "Sources/a.swift")

        let (json, _) = try await runTool(["query": "let answer = 42"], backend: Self.grepBackend)

        let result = try resultObject(in: json)
        XCTAssertEqual(result["count"] as? Int, 1)
        XCTAssertTrue(try XCTUnwrap(try resultLines(in: json).first).hasPrefix("./Sources/a.swift:1:"))
    }

    /// grep and ripgrep exit 2 for a soft error (an unreadable subdirectory).
    /// Matches printed before it are real, so the tool must warn instead of
    /// replacing them with an error.
    func testMatchesFoundBeforeASoftErrorAreReturned() async throws {
        try XCTSkipIf(geteuid() == 0, "root can read a chmod 000 directory")

        try writeFile("needle readable\n", to: "readable.txt")
        let locked = try writeFile("needle locked\n", to: "locked/secret.txt")
            .deletingLastPathComponent()
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o000],
            ofItemAtPath: locked.path
        )
        defer {
            try? FileManager.default.setAttributes(
                [.posixPermissions: 0o755],
                ofItemAtPath: locked.path
            )
        }

        let (json, _) = try await runTool(["query": "needle"], backend: Self.grepBackend)

        let result = try resultObject(in: json)
        XCTAssertEqual(result["count"] as? Int, 1, "a soft error must not discard the matches: \(result)")
        XCTAssertTrue(try XCTUnwrap(try resultLines(in: json).first).contains("readable.txt"))
        XCTAssertNotNil(result["warning"] as? String, "the payload must say the search was incomplete")
    }

    /// An unlistable search root must not read as "0 results". It used to be a
    /// silent success (the root's children were listed with `try?`); now the
    /// backend itself reports the permission failure and the exit code carries
    /// it.
    func testAnUnlistableSearchRootIsReportedAsAnError() async throws {
        try XCTSkipIf(geteuid() == 0, "root can list a chmod 111 directory")

        let locked = root.appendingPathComponent("locked", isDirectory: true)
        try FileManager.default.createDirectory(at: locked, withIntermediateDirectories: true)
        try writeFile("needle\n", to: "locked/a.txt")
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o111],
            ofItemAtPath: locked.path
        )
        defer {
            try? FileManager.default.setAttributes(
                [.posixPermissions: 0o755],
                ofItemAtPath: locked.path
            )
        }

        for backend in backends() {
            let (json, _) = try await runTool([
                "query": "needle",
                "path": locked.path
            ], backend: backend)

            try assertToolError(json, context: label(backend))
        }
    }

    // MARK: - The command line is one contract

    /// Both backends' argv, pinned flag for flag: hidden and binary handling,
    /// the exclude globs, the case toggle, the `--` guard, and exactly one
    /// search target. `rg` is absent on many machines, so this case is the only
    /// one that always exercises its flags.
    func testBothBackendsBuildTheirDocumentedCommandLine() throws {
        let ripgrep = GrepSearchTool.SearchBackend.ripgrep(URL(fileURLWithPath: "/nonexistent/rg"))

        XCTAssertEqual(
            GrepSearchTool.searchArguments(
                backend: ripgrep,
                isDirectory: true,
                caseInsensitive: true,
                literal: false,
                query: "-needle",
                targets: ["."]
            ),
            ["-u", "-n", "--max-count=50", "--with-filename", "-i", "--", "-needle", "."]
        )

        XCTAssertEqual(
            GrepSearchTool.searchArguments(
                backend: Self.grepBackend,
                isDirectory: true,
                caseInsensitive: false,
                literal: false,
                query: "needle",
                targets: ["."]
            ),
            [
                "-E", "-r", "-n", "-I", "-H", "--max-count=50",
                "--exclude-dir=*/.*", "--exclude=*/.*",
                "--", "needle", "."
            ]
        )

        // A single file target needs no exclude rules, and the literal retry
        // swaps the pattern dialect rather than adding a second one.
        XCTAssertEqual(
            GrepSearchTool.searchArguments(
                backend: Self.grepBackend,
                isDirectory: false,
                caseInsensitive: true,
                literal: true,
                query: "foo(",
                targets: ["./a.swift"]
            ),
            ["-F", "-r", "-n", "-I", "-H", "--max-count=50", "-i", "--", "foo(", "./a.swift"]
        )

        XCTAssertEqual(
            GrepSearchTool.searchArguments(
                backend: ripgrep,
                isDirectory: false,
                caseInsensitive: true,
                literal: true,
                query: "foo(",
                targets: ["./a.swift"]
            ),
            ["-F", "-u", "-n", "--max-count=50", "--with-filename", "-i", "--", "foo(", "./a.swift"]
        )
    }

    // MARK: - ripgrep

    /// The primary backend, exercised for real when it is installed. Every
    /// other "either backend" loop here degrades to grep on a machine without
    /// rg, so without this case a search could mean two different things
    /// depending on what a machine happens to have installed.
    func testTheRipgrepBackendHonoursTheSameContract() async throws {
        guard let ripgrep = Self.ripgrepBinary() else {
            throw XCTSkip("ripgrep is not installed; name a binary in /tmp/dynamoe_grep_rg_path to run this case")
        }
        let backend = GrepSearchTool.SearchBackend.ripgrep(ripgrep)

        try writeFile("needle in this file\n", to: "chosen.swift")
        try writeFile("needle in another file\n", to: "other.swift")
        try writeFile("needle in a nested dotfile\n", to: "Sub/.nested.txt")
        try writeFile("needle in a hidden directory\n", to: "Sub/.hidden/x.txt")
        try writeFile("needle in a visible file\n", to: "Sub/visible.txt")
        try writeFile("NOTES.md\n", to: ".gitignore")
        try writeFile("the ignored instruction\n", to: "NOTES.md")
        try gitInit()

        let single = try await runTool([
            "query": "needle",
            "path": root.appendingPathComponent("chosen.swift").path
        ], backend: backend)
        let singleLines = try resultLines(in: single.json)
        XCTAssertEqual(singleLines.count, 1, "a file path must search that file only: \(singleLines)")
        XCTAssertTrue(
            try XCTUnwrap(singleLines.first).hasPrefix("./chosen.swift:1:"),
            "ripgrep printed: \(singleLines)"
        )

        let directory = try await runTool(["query": "needle"], backend: backend)
        let directoryLines = try resultLines(in: directory.json)
        XCTAssertEqual(directoryLines.count, 3, "ripgrep must find only the visible files: \(directoryLines)")
        XCTAssertFalse(
            directoryLines.contains { $0.contains("/.") },
            "ripgrep searched a hidden entry: \(directoryLines)"
        )

        let ignored = try await runTool(["query": "ignored instruction"], backend: backend)
        XCTAssertEqual(
            try resultObject(in: ignored.json)["count"] as? Int,
            1,
            "ripgrep skipped a git-ignored file: \(ignored.json)"
        )
    }

    /// The tool reports a cancelled search as a cancellation rather than a
    /// failure, keyed on the exit code this produces (a cancelled process
    /// reports 130, never the backend's own 2).
    func testACancelledProcessReportsCancellationRatherThanFailure() async throws {
        let start = Date()
        let task = Task {
            try await AgentHarness.runProcess(
                executableURL: URL(fileURLWithPath: "/bin/sleep"),
                arguments: ["30"],
                currentDirectory: root,
                timeoutSeconds: 30
            )
        }
        try await Task.sleep(nanoseconds: 200_000_000)
        task.cancel()

        let result = try await task.value
        XCTAssertEqual(result.exitCode, 130)
        XCTAssertLessThan(Date().timeIntervalSince(start), 10, "a cancelled process must not run to its own exit")
    }

    /// A process that cannot even be launched must fail, not block. The old
    /// cleanup read both pipes to EOF after a failed spawn, and their write ends
    /// were still open, so the caller never resumed: a bad `path` hung the turn.
    func testAnUnlaunchableProcessThrowsInsteadOfHanging() async throws {
        let start = Date()
        do {
            _ = try await AgentHarness.runProcess(
                executableURL: URL(fileURLWithPath: root.appendingPathComponent("no-such-binary").path),
                arguments: [],
                currentDirectory: root,
                timeoutSeconds: 5
            )
            XCTFail("expected the launch to fail")
        } catch {
            XCTAssertFalse(error.localizedDescription.isEmpty)
        }
        XCTAssertLessThan(Date().timeIntervalSince(start), 5)
    }
}
