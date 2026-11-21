//
//  FindFilesToolTests.swift
//  DynaMoE
//
//  The find_files contract, pinned without a model. Every case here runs the
//  real tool against real files in a temporary directory, so the assertions are
//  about the payload the model receives and about what the host did with the
//  pattern it was handed.
//
//  The first case is the defect this file exists for. find_files used to build
//  `zsh -c "find . -iname '<pattern>'"` and escape a quote as `\'`, which does
//  not escape inside a zsh single-quoted word: a pattern could close the quote
//  and run arbitrary commands, and find_files is a read-only tool that no run
//  policy treats as state-changing, so nothing carded it. The tool now passes
//  the pattern as one argv element to `/usr/bin/find`.
//

import XCTest
@testable import DynaMoE

final class FindFilesToolTests: XCTestCase {

    private var root: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("find_files_\(UUID().uuidString)", isDirectory: true)
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
    private func runTool(_ arguments: [String: Any]) async throws -> (json: String, stdout: String?) {
        let tool = FindFilesTool()
        let result = try await tool.execute(
            arguments: arguments,
            workingDirectory: root,
            maxOutputLength: 4_000
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

    private func resultMatches(in json: String) throws -> [String] {
        try XCTUnwrap(resultObject(in: json)["matches"] as? [String])
    }

    // MARK: - The pattern is data, never a command

    func testAPatternCannotRunAShellCommand() async throws {
        try writeFile("hello\n", to: "keep.txt")
        let sentinel = root.appendingPathComponent("pwned").path

        // The exact shape that broke the old quoting: a quote, a command, a
        // comment marker, and a balanced-enough tail. In `zsh -c` with `\'`
        // escaping this ran `touch <sentinel>`.
        let pattern = "a' ; touch \(sentinel) ; #'"
        let (json, _) = try await runTool(["pattern": pattern])

        XCTAssertFalse(
            FileManager.default.fileExists(atPath: sentinel),
            "the model's pattern ran as a shell command"
        )
        XCTAssertEqual(try resultObject(in: json)["count"] as? Int, 0)
    }

    func testACommandSubstitutionPatternIsSearchedLiterally() async throws {
        let sentinel = root.appendingPathComponent("substituted").path
        let (json, _) = try await runTool(["pattern": "$(touch \(sentinel))"])

        XCTAssertFalse(FileManager.default.fileExists(atPath: sentinel))
        XCTAssertEqual(try resultObject(in: json)["count"] as? Int, 0)
    }

    // MARK: - Matching

    func testMatchesAreCaseInsensitiveAndRelative() async throws {
        // Two distinct names, not two spellings of one: the default volume is
        // case-insensitive, so `ContentView.swift` and `contentview.swift` are
        // the same file there and the fixture would silently lose a match.
        try writeFile("", to: "Sources/ContentView.swift")
        try writeFile("", to: "Sources/Other.swift")

        let (json, stdout) = try await runTool(["pattern": "*.SWIFT"])

        let matches = try resultMatches(in: json)
        XCTAssertEqual(Set(matches), ["./Sources/ContentView.swift", "./Sources/Other.swift"], "got \(matches)")
        XCTAssertTrue(matches.allSatisfy { $0.hasPrefix("./") }, "matches are relative to the search root: \(matches)")
        XCTAssertTrue(stdout?.contains("ContentView.swift") == true)
        XCTAssertEqual(try resultObject(in: json)["pattern"] as? String, "*.SWIFT")
    }

    func testHiddenEntriesAreSkipped() async throws {
        try writeFile("", to: "visible.swift")
        try writeFile("", to: ".hidden/secret.swift")

        let (json, _) = try await runTool(["pattern": "*.swift"])

        let matches = try resultMatches(in: json)
        XCTAssertEqual(matches, ["./visible.swift"])
    }

    func testDepthLimitIsHonoured() async throws {
        try writeFile("", to: "one.swift")
        try writeFile("", to: "a/b/two.swift")

        let shallow = try resultMatches(in: try await runTool(["pattern": "*.swift", "max_depth": 1]).json)
        XCTAssertEqual(shallow, ["./one.swift"])

        let deep = try resultMatches(in: try await runTool(["pattern": "*.swift", "max_depth": 3]).json)
        XCTAssertEqual(Set(deep), ["./one.swift", "./a/b/two.swift"])
    }

    /// A negative depth used to reach find as `-maxdepth -4`, which exits with
    /// an error and no results. It is clamped, so the search still answers.
    func testANegativeDepthStillSearchesTheRoot() async throws {
        try writeFile("", to: "root.swift")

        let (json, _) = try await runTool(["pattern": "*.swift", "max_depth": -4])

        XCTAssertEqual(try resultMatches(in: json), ["./root.swift"])
    }

    func testResultsAreCapped() async throws {
        for index in 0..<60 {
            try writeFile("", to: "file\(index).swift")
        }

        let (json, _) = try await runTool(["pattern": "*.swift"])

        XCTAssertEqual(try resultMatches(in: json).count, 50)
        XCTAssertEqual(try resultObject(in: json)["count"] as? Int, 50)
    }

    func testAMissingPatternIsAnError() async throws {
        let (json, _) = try await runTool([:])

        let obj = try object(in: json)
        XCTAssertEqual(obj["status"] as? String, "error")
        XCTAssertTrue((obj["error"] as? String)?.contains("pattern") == true)
    }

    func testAnExplicitPathOverridesTheWorkingDirectory() async throws {
        let nested = root.appendingPathComponent("nested", isDirectory: true)
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        try "".write(to: nested.appendingPathComponent("only.swift"), atomically: true, encoding: .utf8)

        let (json, _) = try await runTool(["pattern": "*.swift", "path": nested.path])

        XCTAssertEqual(try resultMatches(in: json), ["./only.swift"])
        XCTAssertEqual(try resultObject(in: json)["search_path"] as? String, nested.path)
    }
}
