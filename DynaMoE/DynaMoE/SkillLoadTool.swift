//
//  SkillLoadTool.swift
//  DynaMoE
//
//  Phase 8 — Skills.
//
//  `skill_load` is the one tool the skills feature adds. It serves three
//  levels of progressive disclosure:
//
//    1. no arguments        -> every installed skill (name + description)
//    2. name                -> that skill's SKILL.md, its base directory, and
//                              the support files beside it
//    3. name + file         -> one support file (references, scripts, assets)
//
//  The base directory is load-bearing, not decoration: a skill like impeccable
//  resolves its own launcher as `<skill-base-dir>/scripts/impeccable`, so the
//  model cannot follow it without the absolute path.
//
//  Reads are always fresh from disk. Long bodies are paged with the same
//  `offset` / `next_offset` / `continuation_hint` vocabulary `file_read` uses,
//  because a skill body can already exceed the per-call character budget.
//

import Foundation

public final class SkillLoadTool: AgentTool {
    public let definition = ToolDefinition(
        name: "skill_load",
        description: "Reads an installed skill's SKILL.md by name, or one of its support files. Call with no arguments to list every installed skill. The # Skills prompt section lists only names and descriptions; the procedure lives in SKILL.md, so load a skill before following it. Long bodies are paged: the result reports next_offset when more remains.",
        parameters: [
            "type": AnyCodable("object"),
            "properties": AnyCodable([
                "name": [
                    "type": "string",
                    "description": "Exact skill name from the # Skills list. Omit to list every installed skill."
                ],
                "file": [
                    "type": "string",
                    "description": "Optional path inside the skill's folder, relative to its base directory (for example 'reference/audit.md' or 'scripts/run.sh'). Defaults to SKILL.md."
                ],
                "offset": [
                    "type": "integer",
                    "description": "Optional 0-indexed line offset for paging a long body; use the next_offset a previous result reported."
                ],
                "max_chars": [
                    "type": "integer",
                    "description": "Optional character budget for this read. Clamped to the harness output budget."
                ]
            ]),
            "required": AnyCodable([])
        ]
    )

    public init() {}

    public func execute(arguments: [String: Any], workingDirectory: URL?, maxOutputLength: Int) async throws -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        let registry = SkillsRegistry.shared

        guard registry.config.enabled else {
            let err = "Skills are disabled in Settings (Agent & Tools → Skills). Enable the master switch to load a skill."
            return (AgentHarness.toolErrorJSON(tool: "skill_load", error: err), nil, err, false)
        }

        let name = Self.string(arguments["name"])?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let offset = Self.integer(arguments["offset"]) ?? 0
        let requestedBudget = Self.integer(arguments["max_chars"]) ?? maxOutputLength
        let budget = max(1_000, min(requestedBudget, max(4_000, maxOutputLength)))

        if name.isEmpty {
            return list(registry: registry)
        }

        guard let skill = registry.skill(named: name) else {
            let available = registry.enabledSkills.map(\.name).sorted()
            let err = available.isEmpty
                ? "No skill named '\(name)' is installed, and no skills are installed at all. Call skill_load with no arguments to confirm."
                : "No skill named '\(name)' is installed. Installed skills: \(available.joined(separator: ", "))."
            return (AgentHarness.toolErrorJSON(tool: "skill_load", error: err, extra: [
                "installed_skills": available
            ]), nil, err, false)
        }

        guard registry.config.isEnabled(skill) else {
            let err = "Skill '\(skill.name)' is installed but disabled in Settings (Agent & Tools → Skills). Do not follow it; continue without it."
            return (AgentHarness.toolErrorJSON(tool: "skill_load", error: err), nil, err, false)
        }

        let relativeFile = Self.string(arguments["file"])?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !relativeFile.isEmpty {
            return loadSupportFile(
                skill: skill,
                relativePath: relativeFile,
                offset: offset,
                budget: budget
            )
        }
        return loadBody(skill: skill, offset: offset, budget: budget)
    }

    // MARK: List mode

    private func list(registry: SkillsRegistry) -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        let skills = registry.enabledSkills
        guard !skills.isEmpty else {
            let message = registry.allSkills.isEmpty
                ? "No skills are installed. Scanned: \(registry.scannedRoots.map(\.url.path).joined(separator: ", ")). Add a folder to skills.json or install the starter skill in Settings."
                : "All installed skills are disabled in Settings (Agent & Tools → Skills)."
            return (AgentHarness.toolSuccessJSON(tool: "skill_load", data: [
                "skills": [],
                "count": 0,
                "message": message
            ]), message, nil, false)
        }

        var payload: [[String: Any]] = []
        var lines: [String] = ["Installed skills (\(skills.count)):", ""]
        for skill in skills {
            payload.append([
                "name": skill.name,
                "description": skill.description,
                "base_dir": skill.directory.path,
                "always_on": skill.always,
                "source": skill.rootLabel
            ])
            var line = "- \(skill.name) — \(skill.description)"
            if skill.always { line += " [always invoked]" }
            lines.append(line)
            lines.append("  base_dir: \(skill.directory.path)")
        }
        lines.append("")
        lines.append("Call skill_load with {\"name\": \"<name>\"} to read one, or add a \"file\" to read a support file.")

        let message = lines.joined(separator: "\n")
        let res = AgentHarness.toolSuccessJSON(tool: "skill_load", data: [
            "skills": payload,
            "count": skills.count,
            "message": message
        ])
        return (res, message, nil, false)
    }

    // MARK: Load modes

    private func loadBody(
        skill: Skill,
        offset: Int,
        budget: Int
    ) -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        let skillFile = skill.skillFile
        guard FileManager.default.fileExists(atPath: skillFile.path) else {
            let err = "Skill '\(skill.name)' has no readable SKILL.md at \(skillFile.path)."
            return (AgentHarness.toolErrorJSON(tool: "skill_load", error: err), nil, err, false)
        }
        do {
            let text = try SkillDiscovery.readText(skillFile)
            let document: SkillFrontmatterParser.Document
            do {
                document = try SkillFrontmatterParser.parse(text)
            } catch {
                let err = "SKILL.md for '\(skill.name)' could not be parsed: \(error.localizedDescription)"
                return (AgentHarness.toolErrorJSON(tool: "skill_load", error: err), nil, err, false)
            }

            let files = Self.supportFiles(in: skill.directory)
            let page = Self.page(document.body, offset: offset, maxCharacters: budget)

            var header = "[skill] \(skill.name) — \(document.frontmatter.description)\n"
            header += "Base directory: \(skill.directory.path)\n"
            header += "Support files: \(files.isEmpty ? "none" : files.joined(separator: ", "))\n"
            if document.frontmatter.always {
                header += "This skill is declared always-on: apply it whenever its description matches, without being asked.\n"
            }
            header += "\n--- SKILL.md (lines \(page.firstLine + 1)-\(page.lastLine) of \(page.totalLines)) ---\n"

            var observation = header + page.content
            if let hint = page.continuationHint {
                observation += "\n<<<TRUNCATED: more of this skill remains>>>\n" + hint
            }
            if page.content.isEmpty {
                observation += "(The body is empty. Follow the description and the support files.)"
            }

            var data: [String: Any] = [
                "name": skill.name,
                "description": document.frontmatter.description,
                "base_dir": skill.directory.path,
                "skill_file": skillFile.path,
                "source": skill.rootLabel,
                "always_on": document.frontmatter.always,
                "files": files,
                "total_lines": page.totalLines,
                "offset": page.firstLine,
                "content": page.content
            ]
            if let next = page.nextOffset {
                data["next_offset"] = next
                data["continuation_hint"] = page.continuationHint ?? ""
            }
            let res = AgentHarness.toolSuccessJSON(tool: "skill_load", data: data)
            return (res, observation, nil, false)
        } catch {
            let err = "Could not read SKILL.md for '\(skill.name)': \(error.localizedDescription)"
            return (AgentHarness.toolErrorJSON(tool: "skill_load", error: err), nil, err, false)
        }
    }

    private func loadSupportFile(
        skill: Skill,
        relativePath: String,
        offset: Int,
        budget: Int
    ) -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        guard let url = Self.resolve(relativePath: relativePath, in: skill.directory) else {
            let err = "'\(relativePath)' is not a readable path inside the '\(skill.name)' skill folder. Use a path relative to \(skill.directory.path) with no '..' segments."
            return (AgentHarness.toolErrorJSON(tool: "skill_load", error: err), nil, err, false)
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            let files = Self.supportFiles(in: skill.directory)
            let err = "'\(relativePath)' does not exist in '\(skill.name)'. Available: \(files.joined(separator: ", "))."
            return (AgentHarness.toolErrorJSON(tool: "skill_load", error: err, extra: [
                "files": files
            ]), nil, err, false)
        }
        guard !isDirectory.boolValue else {
            let err = "'\(relativePath)' is a folder. Ask for a file inside it, for example '\(relativePath)/file.md'."
            return (AgentHarness.toolErrorJSON(tool: "skill_load", error: err), nil, err, false)
        }
        do {
            let text = try Self.readTextCapped(url, maximumBytes: 1_048_576)
            let page = Self.page(text, offset: offset, maxCharacters: budget)
            var observation = "[skill file] \(skill.name)/\(relativePath)\n"
            observation += "Base directory: \(skill.directory.path)\n\n"
            observation += "--- content (lines \(page.firstLine + 1)-\(page.lastLine) of \(page.totalLines)) ---\n"
            observation += page.content
            if let hint = page.continuationHint {
                observation += "\n<<<TRUNCATED: more of this file remains>>>\n" + hint
            }
            var data: [String: Any] = [
                "name": skill.name,
                "file": relativePath,
                "path": url.path,
                "base_dir": skill.directory.path,
                "total_lines": page.totalLines,
                "offset": page.firstLine,
                "content": page.content
            ]
            if let next = page.nextOffset {
                data["next_offset"] = next
                data["continuation_hint"] = page.continuationHint ?? ""
            }
            let res = AgentHarness.toolSuccessJSON(tool: "skill_load", data: data)
            return (res, observation, nil, false)
        } catch {
            let err = "Could not read '\(relativePath)' from '\(skill.name)': \(error.localizedDescription)"
            return (AgentHarness.toolErrorJSON(tool: "skill_load", error: err), nil, err, false)
        }
    }

    // MARK: Paging

    struct Page {
        var content: String
        var firstLine: Int
        var lastLine: Int
        var totalLines: Int
        var nextOffset: Int?
        var continuationHint: String?
    }

    /// Line-based paging with a character ceiling. Always returns at least one
    /// line so a single oversized line cannot make the tool answer nothing.
    static func page(_ text: String, offset: Int, maxCharacters: Int) -> Page {
        let lines = text.components(separatedBy: "\n")
        let total = lines.count
        let start = max(0, min(offset, max(0, total - 1)))
        var consumed = 0
        var characters = 0
        while start + consumed < total {
            let line = lines[start + consumed]
            let addition = line.count + (consumed == 0 ? 0 : 1)
            if consumed > 0 && characters + addition > maxCharacters { break }
            if consumed == 0 && addition > maxCharacters {
                // One enormous line: clip it and move past it.
                let clipped = String(line.prefix(max(0, maxCharacters)))
                let nextOffset = start + 1 < total ? start + 1 : nil
                return Page(
                    content: clipped,
                    firstLine: start,
                    lastLine: start,
                    totalLines: total,
                    nextOffset: nextOffset,
                    continuationHint: nextOffset.map { "Call skill_load again with \"file\" (if used) and \"offset\": \($0) to continue." }
                )
            }
            characters += addition
            consumed += 1
        }
        if consumed == 0 { consumed = 1 }
        let end = start + consumed
        let content = lines[start..<end].joined(separator: "\n")
        let nextOffset = end < total ? end : nil
        return Page(
            content: content,
            firstLine: start,
            lastLine: max(start + 1, end),
            totalLines: total,
            nextOffset: nextOffset,
            continuationHint: nextOffset.map { "Call skill_load again with \"file\" (if used) and \"offset\": \($0) to continue." }
        )
    }

    // MARK: File helpers

    /// Resolves a relative path inside a skill folder, refusing anything that
    /// escapes it. Symlinks are resolved before the containment check, so a
    /// link inside the folder cannot be used to read outside it.
    static func resolve(relativePath: String, in directory: URL) -> URL? {
        let trimmed = relativePath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !trimmed.hasPrefix("/") else { return nil }
        let base = directory.resolvingSymlinksInPath().standardizedFileURL
        let candidate = base.appendingPathComponent(trimmed).standardizedFileURL
        let resolved = candidate.resolvingSymlinksInPath()
        let basePath = base.path
        guard resolved.path == basePath || resolved.path.hasPrefix(basePath + "/") else { return nil }
        return resolved
    }

    /// Top-level entries plus the contents of the conventional support
    /// folders. Capped so a skill with a large reference tree cannot flood the
    /// result; the model can still ask for a named file.
    static func supportFiles(in directory: URL) -> [String] {
        let fileManager = FileManager.default
        let entries = (try? fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )) ?? []
        var names: [String] = []
        for entry in entries.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            var isDirectory: ObjCBool = false
            fileManager.fileExists(atPath: entry.path, isDirectory: &isDirectory)
            names.append(isDirectory.boolValue ? entry.lastPathComponent + "/" : entry.lastPathComponent)
        }
        for folder in ["scripts", "reference", "references", "assets", "examples"] {
            let folderURL = directory.appendingPathComponent(folder, isDirectory: true)
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: folderURL.path, isDirectory: &isDirectory),
                  isDirectory.boolValue else { continue }
            let children = (try? fileManager.contentsOfDirectory(
                at: folderURL,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            )) ?? []
            let listed = children.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }).prefix(24)
            for child in listed {
                var childIsDirectory: ObjCBool = false
                fileManager.fileExists(atPath: child.path, isDirectory: &childIsDirectory)
                names.append(folder + "/" + child.lastPathComponent + (childIsDirectory.boolValue ? "/" : ""))
            }
            if children.count > 24 {
                names.append("\(folder)/… (\(children.count - 24) more)")
            }
        }
        return names
    }

    private static func readTextCapped(_ url: URL, maximumBytes: Int) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let data = try handle.read(upToCount: maximumBytes) ?? Data()
        if let text = String(data: data, encoding: .utf8) { return text }
        if let text = String(data: data, encoding: .isoLatin1) { return text }
        throw SkillParseError.malformedLine(line: 0, text: "file is not UTF-8 or Latin-1 text")
    }

    // MARK: Argument coercion

    static func string(_ value: Any?) -> String? {
        if let text = value as? String { return text }
        if let number = value as? NSNumber { return number.stringValue }
        return nil
    }

    static func integer(_ value: Any?) -> Int? {
        if let int = value as? Int { return int }
        if let number = value as? NSNumber { return number.intValue }
        if let text = value as? String { return Int(text.trimmingCharacters(in: .whitespaces)) }
        return nil
    }
}
