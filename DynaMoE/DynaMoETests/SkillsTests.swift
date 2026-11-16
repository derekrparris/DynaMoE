//
//  SkillsTests.swift
//  DynaMoE
//
//  Phase 8 — Skills, pinned without a model, a GPU, or the user's real skill
//  folders. The frontmatter subset, the validator, discovery precedence,
//  prompt rendering, the config file, the `skill_load` tool's three modes,
//  and the starter skill's parity with the canonical copy in the repository
//  are all pure logic. The one harness-touching case asserts the section
//  survives `refreshingLoadedToolsSection`, which is the byte-for-byte
//  invariant a mid-run `tools_load` depends on.
//

import XCTest
@testable import DynaMoE

// MARK: - Fixtures

private enum SkillFixture {
    static func makeRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("dynamoe-skills-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    @discardableResult
    static func writeSkill(
        in root: URL,
        folder: String,
        frontmatter: String,
        body: String = "Do the thing."
    ) throws -> URL {
        let directory = root.appendingPathComponent(folder, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let text = "---\n\(frontmatter)\n---\n\n\(body)\n"
        try Data(text.utf8).write(to: directory.appendingPathComponent("SKILL.md"))
        return directory
    }

    static func skill(
        name: String,
        directory: URL,
        always: Bool = false,
        description: String = "Does a thing when it applies."
    ) -> Skill {
        Skill(
            name: name,
            description: description,
            always: always,
            userInvocable: false,
            directory: directory,
            rootLabel: "test",
            rootIndex: 0
        )
    }

    static func cleanUp(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }
}

// MARK: - Frontmatter

final class SkillsFrontmatterTests: XCTestCase {

    func testPlainFrontmatterParses() throws {
        let document = try SkillFrontmatterParser.parse("""
        ---
        name: my-skill
        description: Use when the user asks about widgets.
        always: true
        user-invocable: true
        ---

        # Body

        Step one.
        """)
        XCTAssertEqual(document.frontmatter.name, "my-skill")
        XCTAssertEqual(document.frontmatter.description, "Use when the user asks about widgets.")
        XCTAssertTrue(document.frontmatter.always)
        XCTAssertTrue(document.frontmatter.userInvocable)
        XCTAssertEqual(document.body, "# Body\n\nStep one.")
    }

    func testCRLFAndLeadingBlankLinesAreAccepted() throws {
        let document = try SkillFrontmatterParser.parse("\r\n\r\n---\r\nname: crlf-skill\r\ndescription: \"Quoted \\\"value\\\".\"\r\n---\r\nBody.\r\n")
        XCTAssertEqual(document.frontmatter.name, "crlf-skill")
        XCTAssertEqual(document.frontmatter.description, "Quoted \"value\".")
        XCTAssertEqual(document.body, "Body.")
    }

    func testSingleQuotedAndFoldedBlockScalars() throws {
        let single = try SkillFrontmatterParser.parse("""
        ---
        name: single
        description: 'It''s a description.'
        ---
        body
        """)
        XCTAssertEqual(single.frontmatter.description, "It's a description.")

        let folded = try SkillFrontmatterParser.parse("""
        ---
        name: folded
        description: >-
          Use when a long description
          wraps across lines.
        ---
        body
        """)
        XCTAssertEqual(folded.frontmatter.description, "Use when a long description wraps across lines.")

        let literal = try SkillFrontmatterParser.parse("""
        ---
        name: literal
        description: |
          First line.
          Second line.
        ---
        body
        """)
        XCTAssertEqual(literal.frontmatter.description, "First line.\nSecond line.")
    }

    func testNestedMetadataIsCollectedAndNeverMistakenForAName() throws {
        let document = try SkillFrontmatterParser.parse("""
        ---
        name: with-metadata
        description: Metadata is allowed.
        metadata:
          version: 4.9.0
          author: someone
        ---
        body
        """)
        XCTAssertEqual(document.frontmatter.metadata["version"], "4.9.0")
        XCTAssertEqual(document.frontmatter.metadata["author"], "someone")
        XCTAssertEqual(document.frontmatter.name, "with-metadata")
        XCTAssertTrue(document.frontmatter.unknownKeys.isEmpty)
    }

    func testUnknownKeysAreReportedRatherThanSilentlyIgnored() throws {
        let document = try SkillFrontmatterParser.parse("""
        ---
        name: unknown-keys
        description: Fine.
        allowed-tools: bash
        ---
        body
        """)
        XCTAssertEqual(document.frontmatter.unknownKeys, ["allowed-tools"])
        let warnings = SkillValidation.warnings(for: document.frontmatter)
        XCTAssertTrue(warnings.contains { $0.contains("allowed-tools") })
    }

    func testMissingAndUnclosedFrontmatterThrow() {
        XCTAssertThrowsError(try SkillFrontmatterParser.parse("no frontmatter here")) { error in
            XCTAssertEqual(error as? SkillParseError, .missingFrontmatter)
        }
        XCTAssertThrowsError(try SkillFrontmatterParser.parse("---\nname: x\ndescription: y\n")) { error in
            XCTAssertEqual(error as? SkillParseError, .unclosedFrontmatter)
        }
    }

    func testRequiredFieldsThrow() {
        XCTAssertThrowsError(try SkillFrontmatterParser.parse("---\ndescription: y\n---\nbody")) { error in
            XCTAssertEqual(error as? SkillParseError, .missingName)
        }
        XCTAssertThrowsError(try SkillFrontmatterParser.parse("---\nname: x\n---\nbody")) { error in
            XCTAssertEqual(error as? SkillParseError, .missingDescription)
        }
    }

    func testEmptyBooleansAreFalseRatherThanErrors() throws {
        let document = try SkillFrontmatterParser.parse("""
        ---
        name: empty-bools
        description: Fine.
        always:
        ---
        body
        """)
        XCTAssertFalse(document.frontmatter.always)
    }
}

// MARK: - Validation

final class SkillsValidationTests: XCTestCase {

    func testNameRules() {
        XCTAssertTrue(SkillValidation.isValidName("subq-decisions"))
        XCTAssertTrue(SkillValidation.isValidName("impeccable"))
        XCTAssertTrue(SkillValidation.isValidName("Skill2"))
        XCTAssertFalse(SkillValidation.isValidName(""))
        XCTAssertFalse(SkillValidation.isValidName("-leading"))
        XCTAssertFalse(SkillValidation.isValidName("trailing-"))
        XCTAssertFalse(SkillValidation.isValidName("double--dash"))
        XCTAssertFalse(SkillValidation.isValidName("with space"))
        XCTAssertFalse(SkillValidation.isValidName("with.dot"))
        XCTAssertFalse(SkillValidation.isValidName(String(repeating: "a", count: 65)))
    }

    func testNormalizedNameIsCaseInsensitive() {
        XCTAssertEqual(SkillValidation.normalizedName(" SubQ-Decisions "), "subq-decisions")
    }

    func testWarningsCoverLongDescriptions() {
        let frontmatter = SkillFrontmatter(
            name: "long",
            description: String(repeating: "a", count: SkillValidation.maximumDescriptionLength + 1)
        )
        XCTAssertTrue(SkillValidation.warnings(for: frontmatter).contains { $0.contains("description") })
    }
}

// MARK: - Discovery

final class SkillsDiscoveryTests: XCTestCase {
    private var roots: [URL] = []

    override func tearDown() {
        roots.forEach(SkillFixture.cleanUp)
        roots = []
        super.tearDown()
    }

    func testDiscoveryReadsNameAndDescriptionFromEachFolder() throws {
        let root = try SkillFixture.makeRoot()
        roots.append(root)
        try SkillFixture.writeSkill(in: root, folder: "alpha", frontmatter: "name: alpha\ndescription: The alpha skill.")
        try SkillFixture.writeSkill(in: root, folder: "beta", frontmatter: "name: beta\ndescription: The beta skill.")

        let result = SkillDiscovery.scan(roots: [SkillRoot(url: root, label: "test")])
        XCTAssertEqual(result.skills.map(\.name), ["alpha", "beta"])
        XCTAssertEqual(result.skills.first?.description, "The alpha skill.")
        XCTAssertTrue(result.issues.isEmpty)
    }

    func testLaterRootWinsNameCollisionAndTheShadowedCopyIsReported() throws {
        let first = try SkillFixture.makeRoot()
        let second = try SkillFixture.makeRoot()
        roots.append(contentsOf: [first, second])
        try SkillFixture.writeSkill(in: first, folder: "shared", frontmatter: "name: shared\ndescription: From the first root.")
        try SkillFixture.writeSkill(in: second, folder: "shared", frontmatter: "name: shared\ndescription: From the second root.")

        let result = SkillDiscovery.scan(roots: [
            SkillRoot(url: first, label: "first"),
            SkillRoot(url: second, label: "second")
        ])
        XCTAssertEqual(result.skills.count, 1)
        XCTAssertEqual(result.skills.first?.description, "From the second root.")
        XCTAssertEqual(result.shadowed.count, 1)
        XCTAssertEqual(result.shadowed.first?.skill.rootLabel, "first")
        XCTAssertEqual(result.shadowed.first?.winner.rootLabel, "second")
    }

    func testASymlinkedRootIsOneSkillNotAShadowing() throws {
        let root = try SkillFixture.makeRoot()
        let alias = try SkillFixture.makeRoot()
        roots.append(contentsOf: [root, alias])
        let real = try SkillFixture.writeSkill(in: root, folder: "alpha", frontmatter: "name: alpha\ndescription: Alpha.")
        // The common ~/.claude/skills -> ~/.agents/skills layout: an alias that
        // resolves to a skill already scanned must not read as a name clash.
        try FileManager.default.createSymbolicLink(
            at: alias.appendingPathComponent("alpha", isDirectory: true),
            withDestinationURL: real
        )

        let result = SkillDiscovery.scan(roots: [
            SkillRoot(url: root, label: "real"),
            SkillRoot(url: alias, label: "alias")
        ])
        XCTAssertEqual(result.skills.count, 1)
        XCTAssertTrue(result.shadowed.isEmpty)
        XCTAssertEqual(result.skills.first?.rootLabel, "real")
        XCTAssertEqual(result.skills.first?.directory.path, real.resolvingSymlinksInPath().path)
    }

    func testNameMustMatchFolder() throws {
        let root = try SkillFixture.makeRoot()
        roots.append(root)
        try SkillFixture.writeSkill(in: root, folder: "folder", frontmatter: "name: elsewhere\ndescription: Mismatched.")

        let result = SkillDiscovery.scan(roots: [SkillRoot(url: root, label: "test")])
        XCTAssertTrue(result.skills.isEmpty)
        XCTAssertEqual(result.issues.first?.severity, .error)
        XCTAssertTrue(result.issues.first?.message.contains("must match the folder") == true)
    }

    func testEpisodeFolderWithoutASkillManifestIsSkippedSilently() throws {
        let root = try SkillFixture.makeRoot()
        roots.append(root)
        let scripts = root.appendingPathComponent("not-a-skill", isDirectory: true)
        try FileManager.default.createDirectory(at: scripts, withIntermediateDirectories: true)
        try Data("echo hi".utf8).write(to: scripts.appendingPathComponent("run.sh"))

        let result = SkillDiscovery.scan(roots: [SkillRoot(url: root, label: "test")])
        XCTAssertTrue(result.skills.isEmpty)
        XCTAssertTrue(result.issues.isEmpty)
    }

    func testManifestFilenameIsFoundCaseInsensitively() throws {
        let root = try SkillFixture.makeRoot()
        roots.append(root)
        let directory = root.appendingPathComponent("lower", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("---\nname: lower\ndescription: Lowercase manifest.\n---\nbody\n".utf8)
            .write(to: directory.appendingPathComponent("skill.md"))

        let result = SkillDiscovery.scan(roots: [SkillRoot(url: root, label: "test")])
        XCTAssertEqual(result.skills.map(\.name), ["lower"])
    }

    func testOneBrokenManifestDoesNotStopTheOthers() throws {
        let root = try SkillFixture.makeRoot()
        roots.append(root)
        try SkillFixture.writeSkill(in: root, folder: "good", frontmatter: "name: good\ndescription: Fine.")
        try SkillFixture.writeSkill(in: root, folder: "bad", frontmatter: "name: bad")

        let result = SkillDiscovery.scan(roots: [SkillRoot(url: root, label: "test")])
        XCTAssertEqual(result.skills.map(\.name), ["good"])
        XCTAssertEqual(result.issues.count, 1)
        XCTAssertEqual(result.issues.first?.skillName, "bad")
    }

    func testDisabledSkillsAreFilteredByTheConfigNotByDiscovery() throws {
        let root = try SkillFixture.makeRoot()
        roots.append(root)
        try SkillFixture.writeSkill(in: root, folder: "alpha", frontmatter: "name: alpha\ndescription: Alpha.")
        try SkillFixture.writeSkill(in: root, folder: "beta", frontmatter: "name: beta\ndescription: Beta.")

        let result = SkillDiscovery.scan(roots: [SkillRoot(url: root, label: "test")])
        let config = SkillsConfig(disabledSkills: ["ALPHA"])
        XCTAssertEqual(result.skills.count, 2, "discovery must keep disabled skills so Settings can show them")
        XCTAssertEqual(result.skills.filter { config.isEnabled($0) }.map(\.name), ["beta"])
    }
}

// MARK: - Prompt Rendering

final class SkillsPromptRendererTests: XCTestCase {

    func testEmptyRegistryRendersNothing() {
        XCTAssertEqual(SkillsPromptRenderer.section(skills: []), "")
    }

    func testOrderingIsDeterministicAndAlwaysOnLeads() {
        let base = FileManager.default.temporaryDirectory
        let skills = [
            SkillFixture.skill(name: "zeta", directory: base),
            SkillFixture.skill(name: "alpha", directory: base),
            SkillFixture.skill(name: "rule", directory: base, always: true, description: "Always rule.")
        ]
        let section = SkillsPromptRenderer.section(skills: skills)
        XCTAssertTrue(section.hasPrefix("# Skills"))
        let alwaysIndex = section.range(of: "## Always invoke")!.lowerBound
        let availableIndex = section.range(of: "## Available")!.lowerBound
        XCTAssertLessThan(alwaysIndex, availableIndex)
        XCTAssertLessThan(section.range(of: "rule: Always rule.")!.lowerBound, availableIndex)
        let alpha = section.range(of: "alpha:")!.lowerBound
        let zeta = section.range(of: "zeta:")!.lowerBound
        XCTAssertLessThan(alpha, zeta, "available skills are sorted by name")
        XCTAssertFalse(section.contains("<<<"), "no paging markers in the prompt block")
    }

    func testAtMostThreeAlwaysOnSkillsAreRenderedAsRules() {
        let base = FileManager.default.temporaryDirectory
        let skills = (1...5).map { index in
            SkillFixture.skill(name: "rule-\(index)", directory: base, always: true, description: "Rule \(index).")
        }
        let section = SkillsPromptRenderer.section(skills: skills)
        let alwaysPart = section.components(separatedBy: "## Available").first ?? section
        XCTAssertEqual(alwaysPart.components(separatedBy: "- rule-").count - 1, 3)
        XCTAssertTrue(section.contains("rule-4:"), "the fourth always-on skill still appears as an ordinary skill")
    }

    func testDescriptionsAreClipped() {
        let base = FileManager.default.temporaryDirectory
        let long = String(repeating: "word ", count: 200)
        let section = SkillsPromptRenderer.section(
            skills: [SkillFixture.skill(name: "long", directory: base, description: long)]
        )
        let line = section.components(separatedBy: "\n").first { $0.hasPrefix("- long:") } ?? ""
        XCTAssertLessThanOrEqual(line.count, SkillValidation.advertisedDescriptionLength + 20)
        XCTAssertTrue(line.hasSuffix("…"))
    }

    func testCharacterBudgetTruncatesWithADiscoveryHint() {
        let base = FileManager.default.temporaryDirectory
        let skills = (1...12).map { index in
            SkillFixture.skill(
                name: "skill-\(index)",
                directory: base,
                description: String(repeating: "detail ", count: 60)
            )
        }
        let section = SkillsPromptRenderer.section(skills: skills, maxAdvertised: 25, advertisedCharacters: 900)
        XCTAssertTrue(section.contains("more skill(s) are installed but not listed here"))
        XCTAssertTrue(section.contains("skill_load"))
        XCTAssertLessThanOrEqual(section.count, 900 + 400, "a budget breach must be bounded by the final hint")
    }

    func testMaxAdvertisedCapsTheList() {
        let base = FileManager.default.temporaryDirectory
        let skills = (1...5).map { index in
            SkillFixture.skill(name: "skill-\(index)", directory: base, description: "Short.")
        }
        let section = SkillsPromptRenderer.section(skills: skills, maxAdvertised: 2)
        XCTAssertTrue(section.contains("3 more skill(s)"))
    }
}

// MARK: - Config Store

final class SkillsConfigStoreTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("dynamoe-skills-config-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDown() {
        if let directory { SkillFixture.cleanUp(directory) }
        super.tearDown()
    }

    func testRoundTripAndSeed() throws {
        let url = directory.appendingPathComponent("skills.json")
        let seeded = try SkillsConfigStore.loadOrSeed(at: url)
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
        XCTAssertTrue(seeded.enabled)
        XCTAssertEqual(seeded.version, SkillsConfig.currentVersion)

        var updated = seeded
        updated.enabled = false
        updated.extraRoots = ["/tmp/skills"]
        updated.disabledSkills = ["alpha"]
        updated.maxAdvertised = 7
        try SkillsConfigStore.save(updated, to: url)
        XCTAssertEqual(try SkillsConfigStore.load(from: url), updated)
    }

    func testMissingFieldsDecodeToDefaults() throws {
        let url = directory.appendingPathComponent("skills.json")
        try Data("{ \"enabled\": false }".utf8).write(to: url)
        let config = try SkillsConfigStore.load(from: url)
        XCTAssertFalse(config.enabled)
        XCTAssertEqual(config.maxAdvertised, 25)
        XCTAssertEqual(config.advertisedCharacters, 6000)
        XCTAssertEqual(config.extraRoots, [])
    }

    func testRawJSONRejectsMalformedTextAndKeepsTheFile() throws {
        let url = directory.appendingPathComponent("skills.json")
        try SkillsConfigStore.save(SkillsConfig(), to: url)
        XCTAssertThrowsError(try SkillsConfigStore.saveRawJSON("{ nope", to: url))
        XCTAssertEqual(try SkillsConfigStore.load(from: url), SkillsConfig())
    }

    func testRawJSONAcceptsAValidDocument() throws {
        let url = directory.appendingPathComponent("skills.json")
        let text = """
        {
          "version": 1,
          "enabled": true,
          "extraRoots": ["~/work/skills"],
          "disabledSkills": ["noisy"],
          "maxAdvertised": 10,
          "advertisedCharacters": 4000
        }
        """
        let config = try SkillsConfigStore.saveRawJSON(text, to: url)
        XCTAssertEqual(config.extraRoots, ["~/work/skills"])
        XCTAssertEqual(config.disabledSkills, ["noisy"])
        XCTAssertEqual(config.maxAdvertised, 10)
    }

    func testPrettyPrintRoundTrips() throws {
        let pretty = SkillsConfigStore.prettyPrinted("{\"enabled\":true}")
        XCTAssertNotNil(pretty)
        XCTAssertNil(SkillsConfigStore.prettyPrinted("not json"))
    }

    func testValidatorFlagsRangesAndMissingRoots() {
        let config = SkillsConfig(
            extraRoots: ["/definitely/not/a/folder"],
            maxAdvertised: 0,
            advertisedCharacters: 10
        )
        let issues = SkillsConfigValidator.validate(config)
        XCTAssertTrue(issues.contains { $0.severity == .error && $0.message.contains("maxAdvertised") })
        XCTAssertTrue(issues.contains { $0.severity == .error && $0.message.contains("advertisedCharacters") })
        XCTAssertTrue(issues.contains { $0.path == "/definitely/not/a/folder" })
    }
}

// MARK: - The Tool

@MainActor
final class SkillLoadToolTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = try SkillFixture.makeRoot()
    }

    override func tearDown() {
        SkillsRegistry.shared.resetForTesting()
        if let root { SkillFixture.cleanUp(root) }
        super.tearDown()
    }

    private func install(_ skill: Skill) {
        SkillsRegistry.shared.installForTesting(skills: [skill], config: SkillsConfig())
    }

    func testListModeWithNoArguments() async throws {
        let directory = try SkillFixture.writeSkill(
            in: root,
            folder: "alpha",
            frontmatter: "name: alpha\ndescription: Alpha skill."
        )
        install(SkillFixture.skill(name: "alpha", directory: directory, description: "Alpha skill."))

        let tool = SkillLoadTool()
        let result = try await tool.execute(arguments: [:], workingDirectory: nil, maxOutputLength: 20_000)
        XCTAssertTrue(result.resultJSON.contains("\"count\""))
        XCTAssertTrue(result.stdout?.contains("Alpha skill.") == true)
        XCTAssertTrue(result.stdout?.contains("base_dir: \(directory.path)") == true)
    }

    func testListModeWithNoSkillsExplainsWhereItLooked() async throws {
        install(SkillFixture.skill(name: "unused", directory: root))
        SkillsRegistry.shared.installForTesting(skills: [], config: SkillsConfig())

        let tool = SkillLoadTool()
        let result = try await tool.execute(arguments: [:], workingDirectory: nil, maxOutputLength: 20_000)
        XCTAssertTrue(result.resultJSON.contains("No skills are installed"))
    }

    func testLoadReturnsBodyBaseDirectoryAndSupportFiles() async throws {
        let directory = try SkillFixture.writeSkill(
            in: root,
            folder: "alpha",
            frontmatter: "name: alpha\ndescription: Alpha skill.",
            body: "Step one.\nStep two."
        )
        let scripts = directory.appendingPathComponent("scripts", isDirectory: true)
        try FileManager.default.createDirectory(at: scripts, withIntermediateDirectories: true)
        try Data("#!/bin/sh\n".utf8).write(to: scripts.appendingPathComponent("run.sh"))
        install(SkillFixture.skill(name: "alpha", directory: directory))

        let tool = SkillLoadTool()
        let result = try await tool.execute(
            arguments: ["name": "alpha"],
            workingDirectory: nil,
            maxOutputLength: 20_000
        )
        XCTAssertEqual(result.stderr, nil)
        XCTAssertTrue(result.resultJSON.contains("\"base_dir\""))
        // JSONSerialization escapes forward slashes in the payload, so the raw
        // path is asserted on the model-facing text, which is what the model
        // actually reads for the base directory.
        XCTAssertTrue(result.stdout?.contains(directory.path) == true)
        XCTAssertTrue(result.stdout?.contains("Step one.") == true)
        XCTAssertTrue(result.stdout?.contains("scripts/run.sh") == true)
    }

    func testLoadSupportFileAndRejectEscapes() async throws {
        let directory = try SkillFixture.writeSkill(
            in: root,
            folder: "alpha",
            frontmatter: "name: alpha\ndescription: Alpha skill."
        )
        let reference = directory.appendingPathComponent("reference", isDirectory: true)
        try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
        try Data("Reference body.".utf8).write(to: reference.appendingPathComponent("audit.md"))
        install(SkillFixture.skill(name: "alpha", directory: directory))

        let tool = SkillLoadTool()
        let good = try await tool.execute(
            arguments: ["name": "alpha", "file": "reference/audit.md"],
            workingDirectory: nil,
            maxOutputLength: 20_000
        )
        XCTAssertTrue(good.stdout?.contains("Reference body.") == true)

        let escaped = try await tool.execute(
            arguments: ["name": "alpha", "file": "../../etc/passwd"],
            workingDirectory: nil,
            maxOutputLength: 20_000
        )
        XCTAssertNotNil(escaped.stderr)
        XCTAssertTrue(escaped.resultJSON.contains("\"error\""))
        XCTAssertTrue(escaped.resultJSON.contains("\"status\""))

        let absolute = try await tool.execute(
            arguments: ["name": "alpha", "file": "/etc/hosts"],
            workingDirectory: nil,
            maxOutputLength: 20_000
        )
        XCTAssertNotNil(absolute.stderr)
    }

    func testLongBodyPagesWithNextOffset() async throws {
        let body = (1...400).map { "Line \($0) of the procedure." }.joined(separator: "\n")
        let directory = try SkillFixture.writeSkill(
            in: root,
            folder: "long",
            frontmatter: "name: long\ndescription: Long skill.",
            body: body
        )
        install(SkillFixture.skill(name: "long", directory: directory))

        let tool = SkillLoadTool()
        let first = try await tool.execute(
            arguments: ["name": "long", "max_chars": 900],
            workingDirectory: nil,
            maxOutputLength: 20_000
        )
        XCTAssertTrue(first.resultJSON.contains("next_offset"))
        XCTAssertTrue(first.stdout?.contains("<<<TRUNCATED") == true)

        // The reported offset must produce the next slice, not a repeat.
        let offsetText = first.resultJSON
            .components(separatedBy: "\"next_offset\" : ")
            .last?
            .components(separatedBy: ",")
            .first?
            .trimmingCharacters(in: CharacterSet(charactersIn: " \n}")) ?? ""
        let offset = Int(offsetText) ?? 0
        XCTAssertGreaterThan(offset, 0)
        let second = try await tool.execute(
            arguments: ["name": "long", "max_chars": 900, "offset": offset],
            workingDirectory: nil,
            maxOutputLength: 20_000
        )
        XCTAssertNotEqual(first.stdout, second.stdout)
        XCTAssertTrue(second.stdout?.contains("Line \(offset + 1) ") == true)
    }

    func testUnknownNameListsWhatIsInstalled() async throws {
        let directory = try SkillFixture.writeSkill(
            in: root,
            folder: "alpha",
            frontmatter: "name: alpha\ndescription: Alpha skill."
        )
        install(SkillFixture.skill(name: "alpha", directory: directory))

        let tool = SkillLoadTool()
        let result = try await tool.execute(
            arguments: ["name": "nope"],
            workingDirectory: nil,
            maxOutputLength: 20_000
        )
        XCTAssertNotNil(result.stderr)
        XCTAssertTrue(result.stderr?.contains("alpha") == true)
    }

    func testDisabledSkillIsRefused() async throws {
        let directory = try SkillFixture.writeSkill(
            in: root,
            folder: "alpha",
            frontmatter: "name: alpha\ndescription: Alpha skill."
        )
        SkillsRegistry.shared.installForTesting(
            skills: [SkillFixture.skill(name: "alpha", directory: directory)],
            config: SkillsConfig(disabledSkills: ["alpha"])
        )

        let tool = SkillLoadTool()
        let result = try await tool.execute(
            arguments: ["name": "alpha"],
            workingDirectory: nil,
            maxOutputLength: 20_000
        )
        XCTAssertNotNil(result.stderr)
        XCTAssertTrue(result.resultJSON.contains("disabled"))
    }

    func testSkillsDisabledMasterSwitchRefuses() async throws {
        let directory = try SkillFixture.writeSkill(
            in: root,
            folder: "alpha",
            frontmatter: "name: alpha\ndescription: Alpha skill."
        )
        SkillsRegistry.shared.installForTesting(
            skills: [SkillFixture.skill(name: "alpha", directory: directory)],
            config: SkillsConfig(enabled: false)
        )

        let tool = SkillLoadTool()
        let result = try await tool.execute(
            arguments: ["name": "alpha"],
            workingDirectory: nil,
            maxOutputLength: 20_000
        )
        XCTAssertNotNil(result.stderr)
        XCTAssertTrue(result.stderr?.contains("disabled") == true)
    }

    func testPageNeverReturnsNothingForOneEnormousLine() {
        let page = SkillLoadTool.page(String(repeating: "x", count: 5_000), offset: 0, maxCharacters: 100)
        XCTAssertEqual(page.content.count, 100)
        XCTAssertEqual(page.nextOffset, nil)
    }
}

// MARK: - Harness Integration

@MainActor
final class SkillsHarnessIntegrationTests: XCTestCase {
    override func tearDown() {
        SkillsRegistry.shared.resetForTesting()
        AgentHarness.shared.resetLoadedToolsToCore()
        super.tearDown()
    }

    func testSkillLoadIsInstalledAndFundamental() {
        let harness = AgentHarness.shared
        XCTAssertNotNil(harness.tools["skill_load"])
        XCTAssertTrue(AgentHarness.coreLoadedToolNames.contains("skill_load"))
        XCTAssertTrue(AgentHarness.nonUnloadableToolNames.contains("skill_load"))
        XCTAssertTrue(PlanModePolicy.allowedToolNames.contains("skill_load"))
        XCTAssertFalse(PlanModePolicy.withholds("skill_load", in: .plan))
        XCTAssertTrue(PlanModePolicy.allowedToolNames.contains("subq_decide"))
        XCTAssertNotNil(harness.tools["subq_decide"])

        // The decisions tool is seeded only when a key resolves: without one
        // its schema would be paid for on every turn and every call would fail.
        harness.resetLoadedToolsToCore()
        XCTAssertEqual(
            harness.loadedTools["subq_decide"] != nil,
            SubQDecisionKeySource.hasAnyKey(),
            "subq_decide must be core-loaded exactly when a key is configured"
        )
    }

    func testSkillsSectionIsRenderedAndSurvivesTheToolsRefresh() throws {
        let harness = AgentHarness.shared
        harness.resetLoadedToolsToCore()
        defer { harness.resetLoadedToolsToCore() }

        let directory = FileManager.default.temporaryDirectory
        SkillsRegistry.shared.installForTesting(
            skills: [SkillFixture.skill(name: "test-skill", directory: directory, always: true, description: "Use for tests.")],
            config: SkillsConfig()
        )

        let fixedDate = Date(timeIntervalSince1970: 1_800_000_000)
        let before = harness.buildSystemPrompt(baseSystem: "persona", currentDate: fixedDate)
        XCTAssertTrue(before.contains("Installed skill procedures."), "the rendered skills section must be in the prompt")
        XCTAssertTrue(before.contains("test-skill"))
        XCTAssertTrue(before.contains("## Always invoke"))

        // The refresh replaces [# Tools, # Subagent Result Delivery Policy);
        // the skills block sits past that heading, so it must survive intact.
        _ = try? harness.loadTool(named: "git_diff")
        let after = harness.buildSystemPrompt(baseSystem: "persona", currentDate: fixedDate)
        let refreshed = harness.refreshingLoadedToolsSection(inPrompt: before)
        XCTAssertEqual(refreshed, after, "the refreshed prompt must match a freshly built one byte for byte")
        XCTAssertTrue(refreshed.contains("Installed skill procedures."))
        XCTAssertTrue(refreshed.contains("test-skill"))
    }

    func testDisabledMasterSwitchRendersNoSection() {
        let harness = AgentHarness.shared
        SkillsRegistry.shared.installForTesting(
            skills: [SkillFixture.skill(name: "test-skill", directory: FileManager.default.temporaryDirectory)],
            config: SkillsConfig(enabled: false)
        )
        let prompt = harness.buildSystemPrompt(baseSystem: "persona", currentDate: Date(timeIntervalSince1970: 1_800_000_000))
        XCTAssertFalse(prompt.contains("Installed skill procedures."), "a disabled master switch must render no skills section at all")
    }

    func testCloudGuidanceCarriesTheSkillsSection() {
        SkillsRegistry.shared.installForTesting(
            skills: [SkillFixture.skill(name: "test-skill", directory: FileManager.default.temporaryDirectory, description: "Use for tests.")],
            config: SkillsConfig()
        )
        let guidance = AgentHarness.shared.buildCloudToolGuidance(mode: .code)
        XCTAssertTrue(guidance.contains("Installed skill procedures."))
        XCTAssertTrue(guidance.contains("test-skill"))
    }
}

// MARK: - Starter Skill

final class StarterSkillTests: XCTestCase {

    func testEmbeddedStarterMatchesTheRepositoryCopy() throws {
        let testFile = URL(fileURLWithPath: #filePath)
        let repoRoot = testFile
            .deletingLastPathComponent() // DynaMoETests
            .deletingLastPathComponent() // DynaMoE (project dir)
            .deletingLastPathComponent() // repository root
        let canonical = repoRoot.appendingPathComponent("Skills/subq-decisions/SKILL.md")
        guard FileManager.default.fileExists(atPath: canonical.path) else {
            throw XCTSkip("canonical skill file not present at \(canonical.path)")
        }
        let text = try String(contentsOf: canonical, encoding: .utf8)
        XCTAssertEqual(StarterSkills.subqDecisionsMarkdown + "\n", text, "StarterSkills.swift drifted from Skills/subq-decisions/SKILL.md")
    }

    func testInstallWritesTheSkillOnceAndRefusesToOverwrite() throws {
        let root = try SkillFixture.makeRoot()
        defer { SkillFixture.cleanUp(root) }

        let url = try StarterSkills.installSubQDecisions(in: root)
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
        let text = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(text.hasPrefix("---\nname: subq-decisions\n"))
        XCTAssertTrue(text.hasSuffix("\n"))

        XCTAssertThrowsError(try StarterSkills.installSubQDecisions(in: root)) { error in
            guard case StarterSkillError.alreadyInstalled = error else {
                return XCTFail("expected alreadyInstalled, got \(error)")
            }
        }
    }
}

// MARK: - SubQ

final class SubQDecisionTests: XCTestCase {

    private func decode(_ json: String) throws -> SubQDecisionResponse {
        try JSONDecoder().decode(SubQDecisionResponse.self, from: Data(json.utf8))
    }

    func testAllThreeAnswerTypesDecode() throws {
        let response = try decode("""
        {
          "model": "subq-snap-preview",
          "answers": {
            "needs_human": { "type": "noul", "noul": 0.92 },
            "queue": {
              "type": "choice",
              "choice": "billing",
              "probabilities": { "billing": 0.83, "tech_support": 0.15, "fraud": 0.02 },
              "confidence": 0.79
            },
            "priority": {
              "type": "score",
              "score": 1.86,
              "legend": { "0": "Within a week", "1": "Within a day", "2": "Within an hour" },
              "probabilities": { "0": 0.01, "1": 0.12, "2": 0.87 },
              "confidence": 0.79
            }
          },
          "usage": { "input_tokens": 477, "output_tokens": 0 }
        }
        """)
        XCTAssertEqual(response.answers["needs_human"], .noul(probability: 0.92))
        XCTAssertEqual(
            response.answers["queue"],
            .choice(choice: "billing", probabilities: ["billing": 0.83, "tech_support": 0.15, "fraud": 0.02], confidence: 0.79)
        )
        if case .score(let score, let legend, let probabilities, let confidence) = response.answers["priority"] {
            XCTAssertEqual(score ?? 0, 1.86, accuracy: 0.001)
            XCTAssertEqual(legend["2"], "Within an hour")
            XCTAssertEqual(probabilities["2"], 0.87)
            XCTAssertEqual(confidence ?? 0, 0.79, accuracy: 0.001)
        } else {
            XCTFail("expected a score answer")
        }
        XCTAssertEqual(response.usage?.inputTokens, 477)
        XCTAssertEqual(response.usage?.outputTokens, 0)

        let summary = response.readableSummary
        XCTAssertTrue(summary.contains("0.92"))
        XCTAssertTrue(summary.contains("billing"))
        XCTAssertTrue(summary.contains("Within an hour"))
    }

    func testUnknownAnswerTypeDoesNotFailTheWholeResponse() throws {
        let response = try decode("""
        { "answers": { "strange": { "type": "ranking", "score": 1 } } }
        """)
        XCTAssertEqual(response.answers["strange"], .other(type: "ranking"))
    }

    func testErrorMappingMatchesTheSkillPromises() {
        XCTAssertEqual(SubQDecisionError.forStatus(401, detail: ""), .unauthorized(""))
        XCTAssertEqual(SubQDecisionError.forStatus(402, detail: ""), .paymentRequired(""))
        XCTAssertEqual(SubQDecisionError.forStatus(403, detail: ""), .forbidden(""))
        XCTAssertEqual(SubQDecisionError.forStatus(422, detail: ""), .invalidRequest(""))
        XCTAssertEqual(SubQDecisionError.forStatus(429, detail: ""), .rateLimited(""))
        XCTAssertEqual(SubQDecisionError.forStatus(503, detail: ""), .unavailable(""))
        XCTAssertEqual(SubQDecisionError.forStatus(418, detail: ""), .http(status: 418, detail: ""))

        for code in [401, 402, 403, 422, 429, 503] {
            let text = SubQDecisionError.forStatus(code, detail: "detail").localizedDescription
            XCTAssertFalse(text.isEmpty)
        }
        XCTAssertTrue(SubQDecisionError.missingKey.localizedDescription.contains("console.subq.ai"))
    }

    func testQuestionValidationCatchesWhatTheAPIWouldReject() {
        XCTAssertNil(SubQDecideTool.questionProblem([
            "gate": ["type": "noul", "instructions": "Is it safe?", "criteria": ["true": "yes", "false": "no"]]
        ]))
        XCTAssertNotNil(SubQDecideTool.questionProblem([:]))
        XCTAssertNotNil(SubQDecideTool.questionProblem(["gate": ["type": "nope", "instructions": "x"]]))
        XCTAssertNotNil(SubQDecideTool.questionProblem(["gate": ["type": "noul", "instructions": " "]]))
        XCTAssertNotNil(SubQDecideTool.questionProblem(["gate": ["type": "choice", "instructions": "x", "criteria": ["one": 1]]]))
        XCTAssertNotNil(SubQDecideTool.questionProblem(["gate": ["type": "score", "instructions": "x", "criteria": ["only one"]]]))
        XCTAssertNotNil(SubQDecideTool.questionProblem([
            "a": ["type": "noul", "instructions": "x"],
            "b": ["type": "noul", "instructions": "x"],
            "c": ["type": "noul", "instructions": "x"],
            "d": ["type": "noul", "instructions": "x"]
        ]))
    }

    func testAnswerJSONRoundTripsForTheToolPayload() {
        let choice = SubQDecideTool.jsonObject(.choice(
            choice: "billing",
            probabilities: ["billing": 0.9],
            confidence: 0.8
        ))
        XCTAssertEqual(choice["type"] as? String, "choice")
        XCTAssertEqual(choice["choice"] as? String, "billing")
        XCTAssertNotNil(choice["probabilities"] as? [String: Double])

        let noul = SubQDecideTool.jsonObject(.noul(probability: 0.92))
        XCTAssertEqual(noul["probability"] as? Double, 0.92)

        let other = SubQDecideTool.jsonObject(.other(type: ""))
        XCTAssertEqual(other["type"] as? String, "unknown")
    }

    func testSettingsDefaultsAndOverrides() {
        let suite = UserDefaults(suiteName: "dynamoe.subq.tests.\(UUID().uuidString)")!
        XCTAssertEqual(SubQDecisionSettings.endpoint(defaults: suite).absoluteString, SubQDecisionSettings.defaultEndpoint)
        XCTAssertEqual(SubQDecisionSettings.model(defaults: suite), SubQDecisionSettings.defaultModel)

        suite.set("https://example.com/v1/systemone", forKey: SubQDecisionSettings.endpointKey)
        suite.set("subq-snap-next", forKey: SubQDecisionSettings.modelKey)
        XCTAssertEqual(SubQDecisionSettings.endpoint(defaults: suite).absoluteString, "https://example.com/v1/systemone")
        XCTAssertEqual(SubQDecisionSettings.model(defaults: suite), "subq-snap-next")
    }

    func testImageLoaderRejectsUnsupportedFormatsAndMissingFiles() throws {
        XCTAssertThrowsError(try SubQImage.load(path: "/definitely/not/here.png"))

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("dynamoe-subq-\(UUID().uuidString).txt")
        try Data("hello".utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertThrowsError(try SubQImage.load(path: url.path))
    }
}
