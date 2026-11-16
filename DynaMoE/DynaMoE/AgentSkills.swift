//
//  AgentSkills.swift
//  DynaMoE
//
//  Phase 8 — Skills.
//
//  A skill is a folder that contains a `SKILL.md`: YAML-ish frontmatter with a
//  name and a one-line description, then a procedure written as Markdown. The
//  system prompt advertises only the name and the description; the body (and
//  any scripts, references or assets beside it) is read on demand with the
//  `skill_load` tool. That is the whole point of the format: a skill costs a
//  line of prompt until it is used.
//
//  The pieces here are deliberately pure — parsing, validation, discovery,
//  config file I/O and prompt rendering — so the contract can be tested
//  without a model, a GPU, or the user's real skill folders. The live registry
//  (`SkillsRegistry`) and the tool (`SkillLoadTool`) are separate files.
//
//  Frontmatter is a hand-written subset of YAML, not a dependency: the fields
//  are scalars, optional quoted strings, block scalars (`|`, `>` with chomping
//  indicators) and one nested map (`metadata`). Anything else is reported as a
//  warning rather than silently dropped, because a skill author who writes
//  `allowed-tools:` deserves to know it was ignored.
//

import Foundation

// MARK: - Parse Errors

public nonisolated enum SkillParseError: LocalizedError, Equatable {
    case missingFrontmatter
    case unclosedFrontmatter
    case malformedLine(line: Int, text: String)
    case missingName
    case missingDescription
    case unreadable(String)

    public var errorDescription: String? {
        switch self {
        case .missingFrontmatter:
            return "SKILL.md has no frontmatter. The file must begin with a '---' line, the fields, and a closing '---' line."
        case .unclosedFrontmatter:
            return "SKILL.md frontmatter is never closed. Add the closing '---' line before the procedure text."
        case .malformedLine(let line, let text):
            return "SKILL.md frontmatter line \(line) is not 'key: value': \(text)"
        case .missingName:
            return "SKILL.md frontmatter has no 'name'. The name must match the skill's folder name."
        case .missingDescription:
            return "SKILL.md frontmatter has no 'description'. The description is what tells the model when to load the skill."
        case .unreadable(let detail):
            return detail
        }
    }
}

// MARK: - Frontmatter

/// The fields `SKILL.md` may declare. Kept intentionally small: the format's
/// value is that a description is a trigger, not a second prompt.
public nonisolated struct SkillFrontmatter: Equatable, Sendable {
    public var name: String
    public var description: String
    /// Renders an "always invoke" rule into the prompt. This is the local
    /// equivalent of a hand-written agent rule ("route decisions through
    /// subq-decisions"), maintained by the skill rather than by a system
    /// prompt per model profile.
    public var always: Bool
    /// Accepted for parity with other Agent Skills hosts. Reserved for a future
    /// user-invocation command; no composer command reads it yet.
    public var userInvocable: Bool
    /// `disable-model-invocation: true` is an author saying the model must NOT
    /// pick this skill up on its own. This build's only invocation surface is
    /// the model's, so such a skill is not advertised and `skill_load` refuses
    /// it — the opposite of silently auto-invoking it.
    public var modelInvocable: Bool
    public var compatibility: String?
    public var license: String?
    public var metadata: [String: String]
    /// Keys this parser does not understand. Surfaced as warnings so a field
    /// another host honours (for example `allowed-tools`) is never treated as
    /// if it had taken effect.
    public var unknownKeys: [String]

    public init(
        name: String,
        description: String,
        always: Bool = false,
        userInvocable: Bool = false,
        modelInvocable: Bool = true,
        compatibility: String? = nil,
        license: String? = nil,
        metadata: [String: String] = [:],
        unknownKeys: [String] = []
    ) {
        self.name = name
        self.description = description
        self.always = always
        self.userInvocable = userInvocable
        self.modelInvocable = modelInvocable
        self.compatibility = compatibility
        self.license = license
        self.metadata = metadata
        self.unknownKeys = unknownKeys
    }
}

/// Parses one `SKILL.md` file.
public nonisolated enum SkillFrontmatterParser {
    /// A parsed file: the fields plus the Markdown body that follows the
    /// closing delimiter (frontmatter stripped, outer whitespace trimmed).
    public struct Document: Equatable, Sendable {
        public var frontmatter: SkillFrontmatter
        public var body: String
    }

    private static let knownKeys: Set<String> = [
        "name", "description", "always", "always-on", "user-invocable",
        "disable-model-invocation", "license", "compatibility", "metadata"
    ]

    public static func parse(_ text: String) throws -> Document {
        let lines = normalizedLines(text)
        var cursor = 0
        while cursor < lines.count, lines[cursor].trimmingCharacters(in: .whitespaces).isEmpty {
            cursor += 1
        }
        guard cursor < lines.count, lines[cursor].trimmingCharacters(in: .whitespaces) == "---" else {
            throw SkillParseError.missingFrontmatter
        }
        let firstFieldLine = cursor + 1
        var closing: Int?
        var scan = firstFieldLine
        while scan < lines.count {
            if lines[scan].trimmingCharacters(in: .whitespaces) == "---" {
                closing = scan
                break
            }
            scan += 1
        }
        guard let closing else { throw SkillParseError.unclosedFrontmatter }

        let fieldLines = Array(lines[firstFieldLine..<closing])
        let fields = try parseFields(fieldLines, firstAbsoluteLine: firstFieldLine + 1)

        let name = (fields.scalars["name"] ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { throw SkillParseError.missingName }
        let description = (fields.scalars["description"] ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !description.isEmpty else { throw SkillParseError.missingDescription }

        let alwaysValue = fields.scalars["always"] ?? fields.scalars["always-on"]
        let userInvocable = decodeBoolean(fields.scalars["user-invocable"]) ?? false
        // Absent means the model MAY invoke it; only an explicit truthy value
        // opts out.
        let modelInvocable = !(decodeBoolean(fields.scalars["disable-model-invocation"]) ?? false)

        let frontmatter = SkillFrontmatter(
            name: name,
            description: description,
            always: decodeBoolean(alwaysValue) ?? false,
            userInvocable: userInvocable,
            modelInvocable: modelInvocable,
            compatibility: fields.scalars["compatibility"].flatMap { $0.isEmpty ? nil : $0 },
            license: fields.scalars["license"].flatMap { $0.isEmpty ? nil : $0 },
            metadata: fields.metadata,
            unknownKeys: fields.unknownKeys
        )

        let body = lines[(closing + 1)...]
            .joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return Document(frontmatter: frontmatter, body: body)
    }

    // MARK: Field scanning

    private struct Fields {
        var scalars: [String: String] = [:]
        var metadata: [String: String] = [:]
        var unknownKeys: [String] = []
    }

    private static func parseFields(_ lines: [String], firstAbsoluteLine: Int) throws -> Fields {
        var fields = Fields()
        var index = 0
        while index < lines.count {
            let raw = lines[index]
            let trimmed = raw.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty {
                index += 1
                continue
            }
            guard let colon = trimmed.firstIndex(of: ":") else {
                throw SkillParseError.malformedLine(line: firstAbsoluteLine + index, text: trimmed)
            }
            let key = String(trimmed[trimmed.startIndex..<colon])
                .trimmingCharacters(in: .whitespaces)
                .lowercased()
            guard !key.isEmpty, key.unicodeScalars.allSatisfy({ isValidKeyScalar($0) }) else {
                throw SkillParseError.malformedLine(line: firstAbsoluteLine + index, text: trimmed)
            }
            var value = String(trimmed[trimmed.index(after: colon)...])
                .trimmingCharacters(in: .whitespaces)
            index += 1

            if isBlockScalarMarker(value) {
                let marker = value
                let folded = marker.hasPrefix(">")
                let keepTrailingNewlines = marker.hasSuffix("+")
                var collected: [String] = []
                while index < lines.count {
                    let candidate = lines[index]
                    if candidate.trimmingCharacters(in: .whitespaces).isEmpty {
                        collected.append("")
                        index += 1
                        continue
                    }
                    let indent = candidate.prefix(while: { $0 == " " || $0 == "\t" }).count
                    if indent == 0 { break }
                    collected.append(candidate)
                    index += 1
                }
                value = decodeBlockScalar(
                    collected,
                    folded: folded,
                    keepTrailingNewlines: keepTrailingNewlines
                )
                record(key: key, value: value, fields: &fields)
                continue
            }

            if value.isEmpty {
                // An indented block under a bare key. Only `metadata` has a
                // defined meaning here; anything else is reported as ignored.
                var nested: [String] = []
                while index < lines.count {
                    let candidate = lines[index]
                    if candidate.trimmingCharacters(in: .whitespaces).isEmpty { break }
                    let indent = candidate.prefix(while: { $0 == " " || $0 == "\t" }).count
                    if indent == 0 { break }
                    nested.append(candidate)
                    index += 1
                }
                if key == "metadata" {
                    for entry in nested {
                        let line = entry.trimmingCharacters(in: .whitespaces)
                        guard let separator = line.firstIndex(of: ":") else { continue }
                        let nestedKey = String(line[line.startIndex..<separator]).trimmingCharacters(in: .whitespaces)
                        let nestedValue = String(line[line.index(after: separator)...]).trimmingCharacters(in: .whitespaces)
                        guard !nestedKey.isEmpty else { continue }
                        fields.metadata[nestedKey] = decodeScalar(nestedValue)
                    }
                }
                // A bare `always:` with no value is a missing boolean, not an
                // error; it simply means false. `record` reports unknown keys
                // exactly once, so a failed key is not listed twice.
                record(key: key, value: "", fields: &fields)
                continue
            }

            record(key: key, value: decodeScalar(value), fields: &fields)
        }
        return fields
    }

    private static func record(key: String, value: String, fields: inout Fields) {
        if knownKeys.contains(key) {
            if key != "metadata" { fields.scalars[key] = value }
        } else {
            fields.unknownKeys.append(key)
        }
    }

    private static func isValidKeyScalar(_ scalar: Unicode.Scalar) -> Bool {
        let value = scalar.value
        let isDigit = value >= 48 && value <= 57
        let isLower = value >= 97 && value <= 122
        return isDigit || isLower || value == 45 || value == 95
    }

    private static func isBlockScalarMarker(_ value: String) -> Bool {
        switch value {
        case "|", "|-", "|+", ">", ">-", ">+": return true
        default: return false
        }
    }

    // MARK: Scalar decoding

    static func decodeScalar(_ raw: String) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespaces)
        if trimmed.hasPrefix("\"") {
            guard trimmed.hasSuffix("\""), trimmed.count >= 2 else {
                return String(trimmed.dropFirst())
            }
            let inner = String(trimmed.dropFirst().dropLast())
            return unescapeDoubleQuoted(inner)
        }
        if trimmed.hasPrefix("'") {
            guard trimmed.hasSuffix("'"), trimmed.count >= 2 else {
                return String(trimmed.dropFirst())
            }
            let inner = String(trimmed.dropFirst().dropLast())
            return inner.replacingOccurrences(of: "''", with: "'")
        }
        return trimmed
    }

    private static func unescapeDoubleQuoted(_ inner: String) -> String {
        var result = ""
        var escaping = false
        for character in inner {
            if escaping {
                switch character {
                case "n": result.append("\n")
                case "t": result.append("\t")
                case "r": result.append("\r")
                case "\"": result.append("\"")
                case "\\": result.append("\\")
                default:
                    result.append("\\")
                    result.append(character)
                }
                escaping = false
            } else if character == "\\" {
                escaping = true
            } else {
                result.append(character)
            }
        }
        if escaping { result.append("\\") }
        return result
    }

    private static func decodeBoolean(_ raw: String?) -> Bool? {
        guard let raw else { return nil }
        switch raw.trimmingCharacters(in: .whitespaces).lowercased() {
        case "true", "yes", "on", "1": return true
        case "false", "no", "off", "0": return false
        default: return nil
        }
    }

    private static func decodeBlockScalar(
        _ collected: [String],
        folded: Bool,
        keepTrailingNewlines: Bool
    ) -> String {
        let nonEmpty = collected.filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        let minimumIndent = nonEmpty
            .map { $0.prefix(while: { $0 == " " || $0 == "\t" }).count }
            .min() ?? 0
        let stripped = collected.map {
            $0.count >= minimumIndent ? String($0.dropFirst(minimumIndent)) : ""
        }

        var text: String
        if folded {
            var parts: [String] = []
            for line in stripped {
                if line.isEmpty {
                    parts.append("\n")
                } else {
                    parts.append(line + " ")
                }
            }
            text = parts.joined()
        } else {
            text = stripped.joined(separator: "\n")
        }
        if keepTrailingNewlines {
            return text
        }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func normalizedLines(_ text: String) -> [String] {
        text.replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
            .components(separatedBy: "\n")
    }
}

// MARK: - Validation

public nonisolated enum SkillValidation {
    public static let maximumNameLength = 64
    public static let maximumDescriptionLength = 1024
    public static let maximumCompatibilityLength = 500
    /// More always-on skills than this crowd every prompt with rules, and the
    /// fourth is almost always accidental. Reported as an issue rather than
    /// silently dropped.
    public static let maximumAlwaysOnSkills = 3
    /// Per-description clip in the prompt section. Long descriptions are a
    /// trigger to read the file, not a summary of it.
    public static let advertisedDescriptionLength = 320

    /// The same shape other Agent Skills hosts validate: letters, digits and
    /// single dashes. Case is accepted (some folders are capitalised) but
    /// comparisons are case-insensitive so two spellings cannot both load.
    public static func isValidName(_ name: String) -> Bool {
        guard !name.isEmpty, name.count <= maximumNameLength else { return false }
        guard name.first != "-", name.last != "-" else { return false }
        var previousWasDash = false
        for scalar in name.unicodeScalars {
            let value = scalar.value
            let isDigit = value >= 48 && value <= 57
            let isLower = value >= 97 && value <= 122
            let isUpper = value >= 65 && value <= 90
            let isDash = value == 45
            guard isDigit || isLower || isUpper || isDash else { return false }
            if isDash {
                if previousWasDash { return false }
                previousWasDash = true
            } else {
                previousWasDash = false
            }
        }
        return true
    }

    public static func normalizedName(_ name: String) -> String {
        name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }

    /// Non-fatal problems worth showing the operator. Fatal ones (an invalid
    /// name, a name that disagrees with its folder) live in `SkillDiscovery`,
    /// which refuses to load the skill at all.
    public static func warnings(for frontmatter: SkillFrontmatter) -> [String] {
        var warnings: [String] = []
        if frontmatter.description.count > advertisedDescriptionLength {
            warnings.append(
                "description is \(frontmatter.description.count) characters; the prompt clips it to \(advertisedDescriptionLength), so a trailing trigger may never reach the model."
            )
        }
        if frontmatter.description.count > maximumDescriptionLength {
            warnings.append(
                "other Agent Skills hosts may reject a description past \(maximumDescriptionLength) characters."
            )
        }
        if let compatibility = frontmatter.compatibility, compatibility.count > maximumCompatibilityLength {
            warnings.append("compatibility is longer than \(maximumCompatibilityLength) characters and is ignored.")
        }
        if !frontmatter.modelInvocable {
            warnings.append(
                "disable-model-invocation is true. This build has no user-invocation command, so the skill is not advertised and skill_load refuses it; remove the field to let the model use it."
            )
        }
        for key in frontmatter.unknownKeys {
            warnings.append("unknown frontmatter field '\(key)' is ignored — it does not take effect in DynaMoE.")
        }
        return warnings
    }
}

// MARK: - Issues

public nonisolated struct SkillIssue: Identifiable, Equatable, Sendable {
    public enum Severity: Int, Comparable, Sendable {
        case warning = 0
        case error = 1

        public static func < (lhs: Severity, rhs: Severity) -> Bool {
            lhs.rawValue < rhs.rawValue
        }

        public var label: String {
            switch self {
            case .warning: return "Warning"
            case .error: return "Error"
            }
        }
    }

    public let severity: Severity
    public let skillName: String?
    public let path: String?
    public let message: String

    public init(severity: Severity, skillName: String? = nil, path: String? = nil, message: String) {
        self.severity = severity
        self.skillName = skillName
        self.path = path
        self.message = message
    }

    public var id: String {
        "\(severity.rawValue)|\(skillName ?? "")|\(path ?? "")|\(message)"
    }

    public var displayMessage: String {
        var prefix = ""
        if let skillName, !skillName.isEmpty { prefix += "\(skillName): " }
        return prefix + message
    }
}

// MARK: - Skill

/// One name collision between two scanned roots. Published separately from the
/// human-readable issue so a caller (the starter installer, in particular) can
/// tell which copy won instead of parsing a message.
public nonisolated struct SkillShadow: Identifiable, Equatable, Sendable {
    public let loser: Skill
    public let winner: Skill

    public var id: String { "\(loser.id)|\(loser.directory.path)|\(winner.directory.path)" }

    public init(loser: Skill, winner: Skill) {
        self.loser = loser
        self.winner = winner
    }
}

/// One installed skill. Metadata only: the body is re-read from disk on every
/// `skill_load`, so editing a SKILL.md takes effect without a rescan.
public nonisolated struct Skill: Identifiable, Equatable, Sendable {
    public let name: String
    /// Case-insensitive address used for dedupe and for `skill_load`.
    public let id: String
    public let description: String
    public let always: Bool
    public let userInvocable: Bool
    /// False when the frontmatter says `disable-model-invocation: true`.
    public let modelInvocable: Bool
    public let directory: URL
    public let rootLabel: String
    /// Precedence order of the root it came from; a later root wins a name
    /// collision, which is how a personal override shadows a checked-in copy.
    public let rootIndex: Int
    /// The manifest the scanner actually read. Discovery is case-insensitive,
    /// so a checkout whose file is `skill.md` must be loadable too; falling
    /// back to a hardcoded `SKILL.md` would list it and then fail on a
    /// case-sensitive volume.
    public let manifestURL: URL?

    public var skillFile: URL {
        manifestURL ?? directory.appendingPathComponent(SkillDiscovery.skillFileName, isDirectory: false)
    }

    public init(
        name: String,
        description: String,
        always: Bool,
        userInvocable: Bool,
        directory: URL,
        rootLabel: String,
        rootIndex: Int,
        modelInvocable: Bool = true,
        manifestURL: URL? = nil
    ) {
        self.name = name
        self.id = SkillValidation.normalizedName(name)
        self.description = description
        self.always = always
        self.userInvocable = userInvocable
        self.modelInvocable = modelInvocable
        self.directory = directory
        self.rootLabel = rootLabel
        self.rootIndex = rootIndex
        self.manifestURL = manifestURL
    }
}

/// A folder that is scanned for skill directories.
public nonisolated struct SkillRoot: Identifiable, Equatable, Sendable {
    public let url: URL
    /// Short name for the Settings chip ("Application Support", "~/.agents").
    public let label: String

    public var id: String { url.standardizedFileURL.path }

    public init(url: URL, label: String) {
        self.url = url
        self.label = label
    }
}

// MARK: - Discovery

public nonisolated enum SkillDiscovery {
    public struct ScanResult: Sendable {
        /// Every winner by name, including skills the config has disabled; the
        /// registry filters those so Settings can still show them.
        public var skills: [Skill]
        /// A skill that lost a name collision, with the copy that won.
        public var shadowed: [(skill: Skill, winner: Skill)]
        public var issues: [SkillIssue]
        public var scannedRoots: [SkillRoot]

        public init(
            skills: [Skill] = [],
            shadowed: [(skill: Skill, winner: Skill)] = [],
            issues: [SkillIssue] = [],
            scannedRoots: [SkillRoot] = []
        ) {
            self.skills = skills
            self.shadowed = shadowed
            self.issues = issues
            self.scannedRoots = scannedRoots
        }
    }

    public static let skillFileName = "SKILL.md"

    public static func scan(roots: [SkillRoot]) -> ScanResult {
        var result = ScanResult(scannedRoots: roots)
        var winners: [String: Skill] = [:]
        var order: [String] = []
        // The same skill is often reachable twice: ~/.claude/skills commonly
        // holds symlinks into ~/.agents/skills. That is one skill, not a name
        // collision, so the real directory is scanned once. Only two DIFFERENT
        // folders with the same name are a shadowing (which is reported).
        var seenRealDirectories: Set<String> = []

        for (rootIndex, root) in roots.enumerated() {
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: root.url.path, isDirectory: &isDirectory),
                  isDirectory.boolValue else { continue }
            let entries = (try? FileManager.default.contentsOfDirectory(
                at: root.url,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            )) ?? []
            for entry in entries.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
                var entryIsDirectory: ObjCBool = false
                guard FileManager.default.fileExists(atPath: entry.path, isDirectory: &entryIsDirectory),
                      entryIsDirectory.boolValue else { continue }
                // One skill can be reachable through two roots when the second
                // holds a symlink into the first (the common ~/.claude/skills ->
                // ~/.agents/skills layout). The real directory is scanned once;
                // only two DIFFERENT folders sharing a name are a shadowing.
                let realDirectory = entry.resolvingSymlinksInPath().standardizedFileURL
                guard seenRealDirectories.insert(realDirectory.path).inserted else { continue }
                guard let skillFile = skillFileURL(in: entry) else { continue }
                do {
                    let text = try readText(skillFile)
                    let document = try SkillFrontmatterParser.parse(text)
                    let folderName = entry.lastPathComponent
                    var fatal: String?
                    if !SkillValidation.isValidName(document.frontmatter.name) {
                        fatal = "name '\(document.frontmatter.name)' is not a valid skill name (letters, digits and single dashes, at most \(SkillValidation.maximumNameLength) characters)."
                    } else if SkillValidation.normalizedName(document.frontmatter.name) != SkillValidation.normalizedName(folderName) {
                        fatal = "name '\(document.frontmatter.name)' must match the folder '\(folderName)'. Rename one to match the other."
                    }
                    if let fatal {
                        result.issues.append(SkillIssue(
                            severity: .error,
                            skillName: document.frontmatter.name,
                            path: entry.path,
                            message: fatal
                        ))
                        continue
                    }
                    for warning in SkillValidation.warnings(for: document.frontmatter) {
                        result.issues.append(SkillIssue(
                            severity: .warning,
                            skillName: document.frontmatter.name,
                            path: entry.path,
                            message: warning
                        ))
                    }
                    let skill = Skill(
                        name: document.frontmatter.name,
                        description: document.frontmatter.description,
                        always: document.frontmatter.always,
                        userInvocable: document.frontmatter.userInvocable,
                        directory: realDirectory,
                        rootLabel: root.label,
                        rootIndex: rootIndex,
                        modelInvocable: document.frontmatter.modelInvocable,
                        manifestURL: skillFile.resolvingSymlinksInPath()
                    )
                    if let previous = winners[skill.id] {
                        result.shadowed.append((skill: previous, winner: skill))
                    }
                    winners[skill.id] = skill
                    if !order.contains(skill.id) { order.append(skill.id) }
                } catch {
                    result.issues.append(SkillIssue(
                        severity: .error,
                        skillName: entry.lastPathComponent,
                        path: skillFile.path,
                        message: error.localizedDescription
                    ))
                }
            }
        }

        result.skills = order.compactMap { winners[$0] }.sorted { $0.id < $1.id }
        result.issues.sort {
            if $0.severity != $1.severity { return $0.severity > $1.severity }
            return $0.displayMessage < $1.displayMessage
        }
        return result
    }

    /// Locates the manifest case-insensitively so a checkout whose file is
    /// `Skill.md` still loads, while an absent manifest remains a skip (a
    /// folder of scripts is not a skill).
    public static func skillFileURL(in directory: URL) -> URL? {
        let exact = directory.appendingPathComponent(skillFileName, isDirectory: false)
        if FileManager.default.fileExists(atPath: exact.path) { return exact }
        let entries = (try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )) ?? []
        return entries.first { $0.lastPathComponent.lowercased() == skillFileName.lowercased() }
    }

    /// A manifest is a hand-written text file. The cap is generous enough for
    /// any real SKILL.md and small enough that a symlink to a large file (or a
    /// FIFO) cannot hang or exhaust the app: this read runs on the MainActor,
    /// once per skill, at launch and on every watcher event.
    public static let maximumManifestBytes = 512 * 1024

    static func readText(_ url: URL) throws -> String {
        let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard values?.isRegularFile != false else {
            throw SkillParseError.unreadable("\(url.path) is not a regular file")
        }
        if let size = values?.fileSize, size > maximumManifestBytes {
            throw SkillParseError.unreadable(
                "\(url.path) is \(size) bytes; a SKILL.md may be at most \(maximumManifestBytes)."
            )
        }
        let handle: FileHandle
        do {
            handle = try FileHandle(forReadingFrom: url)
        } catch {
            throw SkillParseError.unreadable("\(url.path) could not be opened: \(error.localizedDescription)")
        }
        defer { try? handle.close() }
        let data: Data
        do {
            data = try handle.read(upToCount: maximumManifestBytes + 1) ?? Data()
        } catch {
            throw SkillParseError.unreadable("\(url.path) could not be read: \(error.localizedDescription)")
        }
        guard data.count <= maximumManifestBytes else {
            throw SkillParseError.unreadable(
                "\(url.path) is larger than \(maximumManifestBytes) bytes; a SKILL.md must be a short procedure."
            )
        }
        if let text = String(data: data, encoding: .utf8) { return text }
        if let text = String(data: data, encoding: .isoLatin1) { return text }
        throw SkillParseError.unreadable("\(url.path) is not UTF-8 or Latin-1 text")
    }
}

// MARK: - Config Document

/// `skills.json` — the file-backed knobs for the feature, mirroring `mcp.json`
/// so both capabilities are managed the same way (form plus raw JSON editor).
public nonisolated struct SkillsConfig: Codable, Equatable, Sendable {
    public static let currentVersion = 1

    public var version: Int
    public var enabled: Bool
    /// Extra folders to scan, after the three built-in locations. A workspace
    /// folder belongs here; it is not scanned implicitly.
    public var extraRoots: [String]
    /// Skill names (case-insensitive) hidden from the prompt and the loader.
    public var disabledSkills: [String]
    /// Upper bound on skills listed in the prompt.
    public var maxAdvertised: Int
    /// Total character budget for the advertised section.
    public var advertisedCharacters: Int

    public init(
        version: Int = SkillsConfig.currentVersion,
        enabled: Bool = true,
        extraRoots: [String] = [],
        disabledSkills: [String] = [],
        maxAdvertised: Int = 25,
        advertisedCharacters: Int = 6000
    ) {
        self.version = version
        self.enabled = enabled
        self.extraRoots = extraRoots
        self.disabledSkills = disabledSkills
        self.maxAdvertised = maxAdvertised
        self.advertisedCharacters = advertisedCharacters
    }

    private enum CodingKeys: String, CodingKey {
        case version, enabled, extraRoots, disabledSkills, maxAdvertised, advertisedCharacters
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = SkillsConfig()
        version = try container.decodeIfPresent(Int.self, forKey: .version) ?? defaults.version
        enabled = try container.decodeIfPresent(Bool.self, forKey: .enabled) ?? defaults.enabled
        extraRoots = try container.decodeIfPresent([String].self, forKey: .extraRoots) ?? defaults.extraRoots
        disabledSkills = try container.decodeIfPresent([String].self, forKey: .disabledSkills) ?? defaults.disabledSkills
        maxAdvertised = try container.decodeIfPresent(Int.self, forKey: .maxAdvertised) ?? defaults.maxAdvertised
        advertisedCharacters = try container.decodeIfPresent(Int.self, forKey: .advertisedCharacters) ?? defaults.advertisedCharacters
    }

    /// Case-insensitive membership, because the name is an address.
    public func isEnabled(_ skill: Skill) -> Bool {
        let disabled = Set(disabledSkills.map(SkillValidation.normalizedName))
        return !disabled.contains(skill.id)
    }

    /// The documented bounds are ranges, not suggestions. A hand-edited file
    /// with `advertisedCharacters: 0` used to render a section that ignored
    /// the setting; validation reports the problem, and this makes the running
    /// config obey the same bounds either way.
    public func clamped() -> SkillsConfig {
        var copy = self
        copy.maxAdvertised = min(
            max(maxAdvertised, 1),
            SkillsConfigValidator.maximumAdvertised
        )
        copy.advertisedCharacters = min(
            max(advertisedCharacters, 500),
            SkillsConfigValidator.maximumAdvertisedCharacters
        )
        return copy
    }
}

public nonisolated enum SkillsConfigError: LocalizedError {
    case applicationSupportUnavailable(String)
    case unreadable(path: String, reason: String)
    case malformed(path: String, reason: String)
    case unwritable(path: String, reason: String)

    public var errorDescription: String? {
        switch self {
        case .applicationSupportUnavailable(let detail):
            return "Application Support is unavailable: \(detail)"
        case .unreadable(let path, let reason):
            return "Could not read \(path): \(reason)"
        case .malformed(let path, let reason):
            return "\(path) is not a valid skills file: \(reason)"
        case .unwritable(let path, let reason):
            return "Could not write \(path): \(reason)"
        }
    }
}

public nonisolated enum SkillsConfigStore {
    public static let fileName = "skills.json"

    public static var directoryURL: URL? {
        guard let base = try? FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        ) else { return nil }
        return base.appendingPathComponent("DynaMoE", isDirectory: true)
    }

    /// The app-managed folder skills may be dropped into.
    public static var skillsDirectoryURL: URL? {
        directoryURL?.appendingPathComponent("skills", isDirectory: true)
    }

    public static var defaultURL: URL? {
        directoryURL?.appendingPathComponent(fileName, isDirectory: false)
    }

    public static var defaultPathDescription: String {
        defaultURL?.path ?? "~/Library/Application Support/DynaMoE/\(fileName)"
    }

    public static let templateJSON = """
    {
      "version": 1,
      "enabled": true,
      "extraRoots": [],
      "disabledSkills": [],
      "maxAdvertised": 25,
      "advertisedCharacters": 6000
    }

    """

    public static func load(from url: URL) throws -> SkillsConfig {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw SkillsConfigError.unreadable(path: url.path, reason: error.localizedDescription)
        }
        do {
            return try JSONDecoder().decode(SkillsConfig.self, from: data)
        } catch {
            throw SkillsConfigError.malformed(path: url.path, reason: error.localizedDescription)
        }
    }

    public static func loadOrSeed(at url: URL? = nil) throws -> SkillsConfig {
        guard let target = url ?? defaultURL else {
            throw SkillsConfigError.applicationSupportUnavailable("the directory could not be resolved")
        }
        if !FileManager.default.fileExists(atPath: target.path) {
            try writeAtomically(Data(templateJSON.utf8), to: target)
        }
        return try load(from: target)
    }

    public static func save(_ config: SkillsConfig, to url: URL? = nil) throws {
        guard let target = url ?? defaultURL else {
            throw SkillsConfigError.applicationSupportUnavailable("the directory could not be resolved")
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data: Data
        do {
            data = try encoder.encode(config)
        } catch {
            throw SkillsConfigError.unwritable(path: target.path, reason: error.localizedDescription)
        }
        var payload = data
        payload.append(0x0A)
        try writeAtomically(payload, to: target)
    }

    public static func saveRawJSON(_ text: String, to url: URL? = nil) throws -> SkillsConfig {
        guard let data = text.data(using: .utf8) else {
            throw SkillsConfigError.malformed(path: url?.path ?? defaultPathDescription, reason: "text is not valid UTF-8")
        }
        let config: SkillsConfig
        do {
            config = try JSONDecoder().decode(SkillsConfig.self, from: data)
        } catch {
            throw SkillsConfigError.malformed(path: url?.path ?? defaultPathDescription, reason: error.localizedDescription)
        }
        try save(config, to: url)
        return config
    }

    public static func prettyPrinted(_ text: String) -> String? {
        guard let data = text.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data),
              let pretty = try? JSONSerialization.data(
                withJSONObject: object,
                options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
              ),
              let string = String(data: pretty, encoding: .utf8) else { return nil }
        return string + "\n"
    }

    public static func modificationToken(at url: URL? = nil) -> String? {
        guard let target = url ?? defaultURL,
              let attributes = try? FileManager.default.attributesOfItem(atPath: target.path) else { return nil }
        let modified = (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        let size = (attributes[.size] as? NSNumber)?.intValue ?? 0
        return "\(modified)-\(size)"
    }

    private static func writeAtomically(_ data: Data, to target: URL) throws {
        let directory = target.deletingLastPathComponent()
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        } catch {
            throw SkillsConfigError.unwritable(path: directory.path, reason: error.localizedDescription)
        }
        do {
            try data.write(to: target, options: .atomic)
        } catch {
            throw SkillsConfigError.unwritable(path: target.path, reason: error.localizedDescription)
        }
    }
}

// MARK: - Config Validation

public nonisolated enum SkillsConfigValidator {
    public static let maximumAdvertised = 100
    public static let maximumAdvertisedCharacters = 40_000

    public static func validate(_ config: SkillsConfig) -> [SkillIssue] {
        var issues: [SkillIssue] = []
        if config.version > SkillsConfig.currentVersion {
            issues.append(SkillIssue(
                severity: .warning,
                message: "skills.json declares version \(config.version); this build understands \(SkillsConfig.currentVersion) and ignores the rest."
            ))
        }
        if config.maxAdvertised < 1 || config.maxAdvertised > maximumAdvertised {
            issues.append(SkillIssue(
                severity: .error,
                message: "maxAdvertised must be between 1 and \(maximumAdvertised)."
            ))
        }
        if config.advertisedCharacters < 500 || config.advertisedCharacters > maximumAdvertisedCharacters {
            issues.append(SkillIssue(
                severity: .error,
                message: "advertisedCharacters must be between 500 and \(maximumAdvertisedCharacters)."
            ))
        }

        var seen: Set<String> = []
        for raw in config.extraRoots {
            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty {
                issues.append(SkillIssue(severity: .warning, message: "An extra root is empty and is ignored."))
                continue
            }
            let expanded = (trimmed as NSString).expandingTildeInPath
            guard expanded.hasPrefix("/") else {
                // URL(fileURLWithPath:) resolves a relative path against the
                // process's working directory, which for the app is `/`. That
                // is never what a person means by `my-project/.agents/skills`.
                issues.append(SkillIssue(
                    severity: .warning,
                    path: expanded,
                    message: "This root is not an absolute path and is ignored. Use / or ~/…; relative paths are not resolved against a working directory."
                ))
                continue
            }
            let standardized = URL(fileURLWithPath: expanded, isDirectory: true).standardizedFileURL.path
            if !seen.insert(standardized).inserted {
                issues.append(SkillIssue(
                    severity: .warning,
                    path: standardized,
                    message: "This folder is listed more than once; the duplicate is ignored."
                ))
                continue
            }
            var isDirectory: ObjCBool = false
            if !FileManager.default.fileExists(atPath: standardized, isDirectory: &isDirectory) || !isDirectory.boolValue {
                issues.append(SkillIssue(
                    severity: .warning,
                    path: standardized,
                    message: "Folder does not exist or is not a directory, so nothing is scanned here."
                ))
            }
        }

        let duplicateNames = Dictionary(grouping: config.disabledSkills.map(SkillValidation.normalizedName)) { $0 }
            .filter { !$0.value.isEmpty && $0.key.isEmpty }
            .count
        if duplicateNames > 0 {
            issues.append(SkillIssue(severity: .warning, message: "A disabled skill entry is empty and is ignored."))
        }
        return issues
    }
}

// MARK: - Prompt Rendering

/// Renders the `# Skills` section. Pure and deterministic so a byte-for-byte
/// prompt test, and the prefix-invalidation fingerprint, can rely on it.
public nonisolated enum SkillsPromptRenderer {
    public static func section(
        skills: [Skill],
        maxAdvertised: Int = 25,
        advertisedCharacters: Int = 6000
    ) -> String {
        let ordered = skills.sorted { $0.id < $1.id }
        guard !ordered.isEmpty else { return "" }

        let alwaysOn = Array(ordered.filter(\.always).prefix(SkillValidation.maximumAlwaysOnSkills))
        let alwaysIds = Set(alwaysOn.map(\.id))
        let remaining = ordered.filter { !alwaysIds.contains($0.id) }

        var lines: [String] = []
        lines.append("# Skills")
        lines.append("")
        lines.append(
            "Installed skill procedures. A skill's description says WHEN it applies; the steps live only in its SKILL.md. "
                + "Call `skill_load` with the skill's name before doing work a skill covers — never infer a procedure from its name. "
                + "A loaded skill reports its base directory; its scripts, references and assets live inside that folder."
        )

        // The header is paid too, so it comes out of the configured budget.
        // A floor would silently spend more than the operator allowed, which is
        // why there is none: the validator clamps the value into a range where
        // the header still leaves room for at least a clipped line.
        var budget = max(0, advertisedCharacters - lines.reduce(0) { $0 + $1.count + 1 })
        var listed = 0
        var omitted = 0

        if !alwaysOn.isEmpty {
            var alwaysLines: [String] = []
            for skill in alwaysOn {
                let line = entry(skill)
                if budget - (line.count + 1) >= 0 {
                    alwaysLines.append(line)
                    budget -= line.count + 1
                } else if budget >= 120 {
                    // Keep the rule the author asked for rather than dropping
                    // it, but clip the part that does not fit.
                    let clipped = entry(skill, descriptionLimit: max(40, budget - 40))
                    alwaysLines.append(clipped)
                    budget -= clipped.count + 1
                } else {
                    omitted += 1
                }
            }
            if !alwaysLines.isEmpty {
                lines.append("")
                lines.append("## Always invoke")
                lines.append("")
                lines.append(contentsOf: alwaysLines)
            }
        }

        var availableLines: [String] = []
        for skill in remaining {
            if listed >= max(1, maxAdvertised) {
                omitted += 1
                continue
            }
            let line = entry(skill)
            if budget - (line.count + 1) < 0 {
                // Nothing listed yet: clip this one so the model still learns
                // the skill exists. Otherwise stop and report the remainder.
                if availableLines.isEmpty && budget > 160 {
                    let clipped = entry(skill, descriptionLimit: max(60, budget - 60))
                    availableLines.append(clipped)
                    budget -= clipped.count + 1
                    listed += 1
                }
                omitted += 1
                continue
            }
            availableLines.append(line)
            budget -= line.count + 1
            listed += 1
        }

        if !availableLines.isEmpty {
            lines.append("")
            lines.append("## Available")
            lines.append("")
            lines.append(contentsOf: availableLines)
        }

        if omitted > 0 {
            lines.append("")
            lines.append("\(omitted) more skill(s) are installed but not listed here. Call `skill_load` with no arguments to list them.")
        }

        return lines.joined(separator: "\n")
    }

    private static func entry(_ skill: Skill, descriptionLimit: Int = SkillValidation.advertisedDescriptionLength) -> String {
        // Descriptions may be block scalars; the prompt lists one skill per
        // line, so internal newlines would break the structure.
        var description = skill.description
            .split(whereSeparator: \.isWhitespace)
            .joined(separator: " ")
        if description.count > descriptionLimit {
            description = String(description.prefix(max(0, descriptionLimit - 1))).trimmingCharacters(in: .whitespaces) + "…"
        }
        // A description is pasted into the system block, so a chat-template
        // marker in it would close the system turn and forge another. The
        // name is validated, but it goes through the same pass because a
        // future rule change must not reopen the hole.
        let name = PromptMarkers.sanitizeControlMarkers(skill.name)
        let safeDescription = PromptMarkers.sanitizeControlMarkers(description)
        return "- \(name): \(safeDescription)"
    }
}
