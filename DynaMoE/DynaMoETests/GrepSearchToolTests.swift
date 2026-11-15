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
//  machines, the tool falls back to `grep` through zsh, and a search that never
//  ran must never be reported as "0 results".
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

    @MainActor
    private func runTool(
        _ arguments: [String: Any],
        maxOutputLength: Int = 4_000
    ) async throws -> (json: String, stdout: String?) {
        let tool = GrepSearchTool()
        let result = try await tool.execute(
            arguments: arguments,
            workingDirectory: root,
            maxOutputLength: maxOutputLength
        )
        return (json: result.resultJSON, stdout: result.stdout)
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
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let obj = try object(in: json)
        XCTAssertEqual(obj["status"] as? String, "error", "expected a tool error, got: \(json)", file: file, line: line)
        let error = try XCTUnwrap(obj["error"] as? String, file: file, line: line)
        XCTAssertFalse(error.isEmpty, "error message must name the failure", file: file, line: line)
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

    /// An invalid pattern makes both grep and ripgrep exit 2 with nothing on
    /// stdout. Discarding the exit code turned that into "0 results", which is
    /// indistinguishable from a real empty result.
    func testInvalidPatternIsReportedAsAnError() async throws {
        try writeFile("foo(bar\n", to: "a.swift")

        let (json, _) = try await runTool(["query": "foo("])

        try assertToolError(json)
    }

    func testMissingPathIsReportedAsAnError() async throws {
        let (json, _) = try await runTool([
            "query": "anything",
            "path": root.appendingPathComponent("does-not-exist").path
        ])

        try assertToolError(json)
    }

    // MARK: - Path handling

    func testSearchesASingleFilePath() async throws {
        try writeFile("needle in this file\n", to: "chosen.swift")
        try writeFile("needle in another file\n", to: "other.swift")

        let (json, _) = try await runTool([
            "query": "needle",
            "path": root.appendingPathComponent("chosen.swift").path
        ])

        let lines = try resultLines(in: json)
        XCTAssertEqual(lines.count, 1, "a file path must search that file only: \(lines)")
        let line = try XCTUnwrap(lines.first)
        XCTAssertTrue(line.contains("chosen.swift"))
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
    func testGitIgnoredFileIsSearched() async throws {
        let git = "/usr/bin/git"
        try XCTSkipUnless(FileManager.default.isExecutableFile(atPath: git), "git is unavailable")

        try writeFile("NOTES.md\n", to: ".gitignore")
        try writeFile("the ignored instruction\n", to: "NOTES.md")

        let process = Process()
        process.executableURL = URL(fileURLWithPath: git)
        process.arguments = ["init", "-q"]
        process.currentDirectoryURL = root
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 0)

        let (json, _) = try await runTool(["query": "ignored instruction"])

        XCTAssertEqual(try resultObject(in: json)["count"] as? Int, 1)
    }

    func testHiddenFileIsNotSearched() async throws {
        try writeFile("needle only in a dotfile\n", to: ".hidden.txt")
        try writeFile("unrelated\n", to: "visible.txt")

        let (json, _) = try await runTool(["query": "needle only in a dotfile"])

        XCTAssertEqual(try resultObject(in: json)["count"] as? Int, 0)
    }

    func testTruncationStillReportsSuccess() async throws {
        let lines = (1...80).map { "needle line \($0)" }.joined(separator: "\n")
        try writeFile(lines, to: "many.txt")

        let (json, _) = try await runTool(["query": "needle line"], maxOutputLength: 400)

        let result = try resultObject(in: json)
        XCTAssertGreaterThanOrEqual(result["count"] as? Int ?? 0, 1)
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
