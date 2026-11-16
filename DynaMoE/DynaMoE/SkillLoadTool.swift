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
        description: "Reads an installed skill's SKILL.md by name, or one of its support files. Call with no arguments to list every enabled skill. The # Skills prompt section lists only names and descriptions; the procedure lives in SKILL.md, so load a skill before following it. Long bodies are paged: the result reports next_offset when more remains.",
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
                    "description": "Optional character budget for this read, in characters. Clamped to the harness output budget; a smaller value is raised to a \(SkillLoadTool.minimumPageCharacters)-character minimum so the page is worth reading."
                ]
            ]),
            "required": AnyCodable([])
        ]
    )

    /// Smallest page worth returning: below this a paged read costs a step and
    /// answers with a fragment. Never allowed to exceed the harness budget.
    static let minimumPageCharacters = 1_000

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
        // The harness budget is a CEILING, not a suggestion: a 100-token budget
        // arrives here as 600 characters, and the old `max(1_000, …)` floor
        // spent 400 more than the caller allowed. The floor is a minimum useful
        // page, so it is clamped by the ceiling rather than applied over it.
        let ceiling = max(0, maxOutputLength)
        let budget = min(ceiling, max(Self.minimumPageCharacters, requestedBudget))

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

        guard skill.modelInvocable else {
            let err = "Skill '\(skill.name)' sets disable-model-invocation: true, so the model must not load it. Continue without it."
            return (AgentHarness.toolErrorJSON(tool: "skill_load", error: err), nil, err, false)
        }

        let relativeFile = Self.string(arguments["file"])?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !relativeFile.isEmpty {
            return await loadSupportFile(
                skill: skill,
                relativePath: relativeFile,
                offset: offset,
                budget: budget
            )
        }
        return await loadBody(skill: skill, offset: offset, budget: budget)
    }

    // MARK: List mode

    private func list(registry: SkillsRegistry) -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        let skills = registry.enabledSkills
        guard !skills.isEmpty else {
            let message = registry.allSkills.isEmpty
                ? "No skills are installed. Scanned: \(registry.scannedRoots.map { AgentHarness.sanitizeText($0.url.path) }.joined(separator: ", ")). Add a folder to skills.json or install the starter skill in Settings."
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
            // A description is untrusted frontmatter, and a folder or root
            // label is a directory name somebody else may have chosen: both go
            // through the same marker pass as every other tool result, so a
            // stored marker cannot forge a boundary in the transcript either.
            let description = AgentHarness.sanitizeText(
                SkillValidation.clippedDescription(skill.description)
            )
            let baseDirectory = AgentHarness.sanitizeText(skill.directory.path)
            let source = AgentHarness.sanitizeText(skill.rootLabel)
            payload.append([
                "name": skill.name,
                "description": description,
                "base_dir": baseDirectory,
                "always_on": skill.always,
                "source": source
            ])
            var line = "- \(skill.name) — \(description)"
            if skill.always { line += " [always invoked]" }
            lines.append(line)
            lines.append("  base_dir: \(baseDirectory)")
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
    ) async -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        let skillFile = skill.skillFile
        guard FileManager.default.fileExists(atPath: skillFile.path) else {
            let err = "Skill '\(skill.name)' has no readable SKILL.md at \(AgentHarness.sanitizeText(skillFile.path))."
            return (AgentHarness.toolErrorJSON(tool: "skill_load", error: err), nil, err, false)
        }
        do {
            // The read runs off the main actor: this tool is MainActor-bound,
            // and a 512 KB manifest plus a stat per entry must not stall the UI.
            let text = try await Self.readOffMain(skillFile, maximumBytes: SkillDiscovery.maximumManifestBytes)
            let document: SkillFrontmatterParser.Document
            do {
                document = try SkillFrontmatterParser.parse(text)
            } catch {
                let err = "SKILL.md for '\(skill.name)' could not be parsed: \(error.localizedDescription)"
                return (AgentHarness.toolErrorJSON(tool: "skill_load", error: err), nil, err, false)
            }

            // The registry listed this skill under its folder name. A manifest
            // edited to declare a DIFFERENT name would otherwise be loaded as
            // the advertised skill, i.e. the model would follow a procedure
            // that no longer belongs to the name it asked for.
            guard SkillValidation.normalizedName(document.frontmatter.name) == skill.id else {
                let declared = AgentHarness.sanitizeText(document.frontmatter.name)
                let err = "SKILL.md for '\(skill.name)' now declares name '\(declared)'. The advertised name and the manifest must match; rename the folder or restore the frontmatter name."
                return (AgentHarness.toolErrorJSON(tool: "skill_load", error: err), nil, err, false)
            }

            let files = (await Task.detached(priority: .userInitiated) {
                Self.supportFiles(in: skill.directory)
            }.value).map(AgentHarness.sanitizeText)
            let safeBody = AgentHarness.sanitizeText(document.body)
            let page = Self.page(safeBody, offset: offset, maxCharacters: budget)
            // Clipped like the prompt clips it: this is a per-call budget, and
            // the description is a trigger to read the file, not a summary of
            // it. The body below is what gets paged.
            let safeDescription = AgentHarness.sanitizeText(
                SkillValidation.clippedDescription(document.frontmatter.description)
            )
            let baseDirectory = AgentHarness.sanitizeText(skill.directory.path)
            let manifestPath = AgentHarness.sanitizeText(skillFile.path)
            let source = AgentHarness.sanitizeText(skill.rootLabel)

            var header = "[skill] \(skill.name) — \(safeDescription)\n"
            header += "Base directory: \(baseDirectory)\n"
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
                "description": safeDescription,
                "base_dir": baseDirectory,
                "skill_file": manifestPath,
                "source": source,
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
    ) async -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        guard let url = Self.resolve(relativePath: relativePath, in: skill.directory) else {
            let safePath = AgentHarness.sanitizeText(relativePath)
            let err = "'\(safePath)' is not a readable path inside the '\(skill.name)' skill folder. Use a path relative to \(AgentHarness.sanitizeText(skill.directory.path)) with no '..' segments."
            return (AgentHarness.toolErrorJSON(tool: "skill_load", error: err), nil, err, false)
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            // Off the caller's actor: this is a stat per entry across every
            // support folder, and the tool is MainActor-bound.
            let files = (await Task.detached(priority: .userInitiated) {
                Self.supportFiles(in: skill.directory)
            }.value).map(AgentHarness.sanitizeText)
            let err = "'\(AgentHarness.sanitizeText(relativePath))' does not exist in '\(skill.name)'. Available: \(files.joined(separator: ", "))."
            return (AgentHarness.toolErrorJSON(tool: "skill_load", error: err, extra: [
                "files": files
            ]), nil, err, false)
        }
        guard !isDirectory.boolValue else {
            let safePath = AgentHarness.sanitizeText(relativePath)
            let err = "'\(safePath)' is a folder. Ask for a file inside it, for example '\(safePath)/file.md'."
            return (AgentHarness.toolErrorJSON(tool: "skill_load", error: err), nil, err, false)
        }
        // A FIFO named like a support file would block the read forever, and a
        // device node is not a document. The shared reader resolves symlinks,
        // opens with O_NONBLOCK and classifies the opened descriptor, so this
        // no longer trusts a check-then-open.
        if let urlValues = try? url.resourceValues(forKeys: [.isRegularFileKey]),
           urlValues.isRegularFile == false {
            let err = "'\(AgentHarness.sanitizeText(relativePath))' is not a regular file, so skill_load will not read it."
            return (AgentHarness.toolErrorJSON(tool: "skill_load", error: err), nil, err, false)
        }
        if let fileKind = AgentHarness.binaryFileKind(at: url) {
            let err = "'\(AgentHarness.sanitizeText(relativePath))' is binary content (detected: \(fileKind)); skill_load returns text only."
            return (AgentHarness.toolErrorJSON(tool: "skill_load", error: err, extra: [
                "file_kind": fileKind
            ]), nil, err, false)
        }
        do {
            let safeRelativePath = AgentHarness.sanitizeText(relativePath)
            let rawText = try await Self.readOffMain(url, maximumBytes: 1_048_576)
            let text = AgentHarness.sanitizeText(rawText)
            let page = Self.page(text, offset: offset, maxCharacters: budget)
            var observation = "[skill file] \(skill.name)/\(safeRelativePath)\n"
            observation += "Base directory: \(AgentHarness.sanitizeText(skill.directory.path))\n\n"
            observation += "--- content (lines \(page.firstLine + 1)-\(page.lastLine) of \(page.totalLines)) ---\n"
            observation += page.content
            if let hint = page.continuationHint {
                observation += "\n<<<TRUNCATED: more of this file remains>>>\n" + hint
            }
            var data: [String: Any] = [
                "name": skill.name,
                "file": safeRelativePath,
                "path": AgentHarness.sanitizeText(url.path),
                "base_dir": AgentHarness.sanitizeText(skill.directory.path),
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
            let err = "Could not read '\(AgentHarness.sanitizeText(relativePath))' from '\(skill.name)': \(error.localizedDescription)"
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
    nonisolated static func supportFiles(in directory: URL) -> [String] {
        let fileManager = FileManager.default
        // Resolve first: `contentsOfDirectory` refuses a symlinked folder and
        // the old `?? []` turned that into "this skill has no support files",
        // which is the exact hazard AGENTS.md records. A skill checked out
        // through a dotfiles link is a normal layout, not an empty one.
        let resolved = directory.resolvingSymlinksInPath().standardizedFileURL
        let entries = (try? fileManager.contentsOfDirectory(
            at: resolved,
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
            let folderURL = resolved.appendingPathComponent(folder, isDirectory: true)
            // `fileExists` follows a link, so a symlinked `scripts/` passes
            // here; listing the RESOLVED path is what makes it list anything.
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: folderURL.path, isDirectory: &isDirectory),
                  isDirectory.boolValue else { continue }
            let resolvedFolder = folderURL.resolvingSymlinksInPath().standardizedFileURL
            let children = (try? fileManager.contentsOfDirectory(
                at: resolvedFolder,
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

    /// Reads on a background task and rethrows on the caller's actor, so the
    /// tool's file access never runs on the MainActor.
    private static func readOffMain(_ url: URL, maximumBytes: Int) async throws -> String {
        let outcome = await Task.detached(priority: .userInitiated) { () -> (text: String?, error: String?) in
            do {
                return (try SkillDiscovery.readTextCapped(url, maximumBytes: maximumBytes), nil)
            } catch {
                return (nil, error.localizedDescription)
            }
        }.value
        if let text = outcome.text { return text }
        throw SkillParseError.unreadable(outcome.error ?? "\(url.path) could not be read")
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
