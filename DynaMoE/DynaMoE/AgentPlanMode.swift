//
//  AgentPlanMode.swift
//  DynaMoE
//
//  Plan mode: a read-only architect pass that analyses the codebase and
//  produces an implementation plan for the user to approve.
//
//  The mode is enforced in three independent layers, because a prompt
//  instruction alone is a suggestion:
//
//   1. `PlanModePolicy.allowedToolNames` withholds every write, execute, MCP
//      and completion tool from the prompt, the cloud tool array and the
//      grammar mask, so the model is never offered a tool it may not call.
//   2. `AgentHarness.executeTool(call:mode:)` refuses a withheld tool even if
//      one is somehow offered, which also covers subagents (they execute
//      through the same harness) and a stale grammar registry.
//   3. `PlanMarkers` brackets the plan document so the UI can render it as a
//      card and ask the user to approve it before the coder runs.
//

import Foundation

/// Which agent persona and tool surface a conversation is running.
/// Persisted per conversation (`ChatSession.agentMode`); `nil` there means `.code`.
nonisolated public enum AgentMode: String, Codable, Equatable, Sendable {
    /// The normal agent: reads, writes, runs shell commands, commits.
    case code
    /// The architect: reads and searches only, then produces a plan.
    case plan
}

/// The sentinels the plan agent wraps its final answer in.
///
/// Detection is line-anchored, not a substring search: a small model that
/// announces "I will end with <!-- DYNAMOE_PLAN_READY -->" must not be read as
/// having produced a finished plan. The markers stay in the persisted message
/// content — they are stripped at render time only — so a reloaded
/// conversation still renders its plan card and history keeps showing the
/// model the convention it was taught.
nonisolated public enum PlanMarkers {
    /// First line of the final plan message.
    public static let start = "<!-- DYNAMOE_PLAN_START -->"
    /// Last line of the final plan message; the UI's cue to ask for approval.
    public static let ready = "<!-- DYNAMOE_PLAN_READY -->"

    public static func startPresent(in text: String) -> Bool {
        lines(of: text).contains { isMarkerLine($0, start) }
    }

    public static func readyPresent(in text: String) -> Bool {
        lines(of: text).contains { isMarkerLine($0, ready) }
    }

    /// Drops marker-only lines and an empty fence pair wrapped around one, so
    /// the rendered plan reads as a document rather than as markup. Prose that
    /// merely mentions a marker is untouched.
    public static func strip(_ text: String) -> String {
        let lines = lines(of: text)
        var out: [String] = []
        var inFence = false
        var index = 0
        while index < lines.count {
            let line = lines[index]
            // A fence pair whose only content is a marker: ``` / marker / ```.
            // Checked only outside a fence, so a real block's closing fence is
            // never mistaken for the opening one.
            if !inFence,
               isFenceLine(line),
               index + 2 < lines.count,
               isMarkerLine(lines[index + 1], start) || isMarkerLine(lines[index + 1], ready),
               isFenceLine(lines[index + 2]) {
                index += 3
                continue
            }
            if isFenceLine(line) { inFence.toggle() }
            if isMarkerLine(line, start) || isMarkerLine(line, ready) {
                index += 1
                continue
            }
            out.append(line)
            index += 1
        }
        return out.joined(separator: "\n")
    }

    /// Marker lines are matched after trimming whitespace and any surrounding
    /// inline-code backticks, which is how a model that half-obeys the
    /// "plain text" instruction still produces a detectable plan.
    private static func isMarkerLine(_ line: String, _ marker: String) -> Bool {
        let unbackticked = line
            .trimmingCharacters(in: .whitespaces)
            .trimmingCharacters(in: CharacterSet(charactersIn: "`"))
        return unbackticked.trimmingCharacters(in: .whitespaces) == marker
    }

    private static func isFenceLine(_ line: String) -> Bool {
        line.trimmingCharacters(in: .whitespaces).hasPrefix("```")
    }

    private static func lines(of text: String) -> [String] {
        text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
    }
}

/// The plan agent's tool surface and its system-prompt section.
///
/// The allow-list is deliberately an allow-list rather than a deny-list: a
/// tool added to the harness later is withheld from plan mode by default
/// instead of silently gaining write access to a read-only promise.
nonisolated public enum PlanModePolicy {
    /// The read tools entering plan mode loads for the turn, so an investigation
    /// does not open with a `tools_load` step.
    ///
    /// Deliberately a small core rather than the whole read surface: every armed
    /// tool is a schema in each plan turn's prompt, and the rest of
    /// `allowedToolNames` stays one `tools_load` away for the turns that need it.
    /// `file_read` plus the two searches cover an ordinary plan.
    public static let coreToolNames: Set<String> = [
        "file_read", "grep_search", "find_files", "codebase_search"
    ]

    /// Tools the plan agent may call. Everything installed and not named here
    /// is withheld.
    public static let allowedToolNames: Set<String> = [
        "file_read", "find_files", "grep_search", "codebase_search",
        "find_symbol_definition", "find_references", "lint_diagnostics",
        "web_search", "web_fetch", "git_status", "git_diff",
        "spawn_subagent", "get_subagent_status", "send_subagent_message", "list_subagents",
        "tools_discover", "tools_load", "tools_unload"
    ]

    /// A coder tool plan mode takes away, and what it is for.
    public struct WithheldTool: Sendable, Hashable {
        public let name: String
        public let purpose: String
    }

    /// The write, execute and completion tools, each with the purpose the
    /// directive prints beside it.
    ///
    /// This is the ONE place those names are written down: the directive's
    /// bullet list is generated from it (`directive(toolNames:)`), so a rename
    /// or an addition here cannot leave the prompt describing a tool that no
    /// longer exists, or omit one that does. It is deliberately not the
    /// enforcement rule — `allowedToolNames` is. A tool that is neither allowed
    /// nor named here is still withheld; the tools section enumerates the
    /// ACTUAL withheld set for the installed catalog every turn.
    public static let withheldTools: [WithheldTool] = [
        WithheldTool(name: "file_write", purpose: "file creation and editing"),
        WithheldTool(name: "file_edit", purpose: "file creation and editing"),
        WithheldTool(name: "shell_run", purpose: "shell execution"),
        WithheldTool(name: "git_commit", purpose: "committing"),
        WithheldTool(name: "complete", purpose: "ending the task with a summary")
    ]

    /// The same set by name, for callers that only need membership.
    public static var withheldToolNames: Set<String> {
        Set(withheldTools.map(\.name))
    }

    /// The `<capabilities>` bullet list, generated from `withheldTools`.
    ///
    /// Names that share a purpose are grouped onto one bullet, which is how the
    /// prompt reads (`- file_write, file_edit (file creation and editing)`) and
    /// keeps the list from spelling the same description out twice. Generation
    /// order follows the constant, so the prompt is stable across runs.
    static var withheldToolBullets: String {
        var order: [String] = []
        var namesByPurpose: [String: [String]] = [:]
        for tool in withheldTools {
            if namesByPurpose[tool.purpose] == nil { order.append(tool.purpose) }
            namesByPurpose[tool.purpose, default: []].append("`\(tool.name)`")
        }
        return order
            .map { "- \(namesByPurpose[$0, default: []].joined(separator: ", ")) (\($0))" }
            .joined(separator: "\n")
    }

    /// True when plan mode must withhold `name`.
    ///
    /// MCP tools are withheld unconditionally, the same posture the approval
    /// gate already takes: a remote server can do anything, and its own
    /// read-only annotations are a claim rather than a guarantee.
    public static func withholds(_ name: String) -> Bool {
        if name.hasPrefix(MCPToolBridge.namePrefix) { return true }
        return !allowedToolNames.contains(name)
    }

    /// The plan-mode system section. Rendered before the tool block so the
    /// model reads its role before its capabilities.
    public static func directive(toolNames: [String]) -> String {
        let available = toolNames.isEmpty
            ? "none"
            : toolNames.joined(separator: ", ")
        return """
        # Plan Mode

        You are DynaMoE in plan mode — an expert architect, senior engineer and planning specialist with meticulous attention to detail.

        Your job is to analyze the codebase and the user's intent, then produce a concrete, actionable implementation plan without modifying files or running state-changing commands.

        <capabilities>
        You do NOT have access to file-modification or shell tools. These tools are physically absent from your environment — calling them fails immediately:
        \(withheldToolBullets)
        - every `mcp__*` tool

        Your available tools are: \(available).

        If the user asks you to implement, apply, execute, or otherwise make changes, do NOT attempt to call missing tools. Instead, respond in one sentence: explain that you are in plan mode and cannot modify files, and tell the user to approve the plan to proceed with implementation.
        </capabilities>

        <critical_rules>
        These rules override everything else. Follow them strictly:

        1. You cannot modify files, create files, delete files, or run write operations — those tools do not exist in plan mode. If asked to implement, say you are in plan mode and direct the user to approve the plan.
        2. Do not attempt commands that change system state. `git_status` and `git_diff` are reads and are expected.
        3. Delegation to subagents is allowed for deeper codebase exploration only; a subagent you spawn inherits this read-only surface.
        4. Provide the most complete analysis you can before proposing implementation steps. Read the real files — never plan against an assumed shape.
        5. Ask a clarifying question only when it is strictly necessary to produce a correct plan, and ask it as an ordinary chat reply. There is no question tool; a reply ends the turn and the user answers normally.
        6. Do not call `complete`; your final plain-text reply IS the deliverable.
        7. Once the investigation is done, output the plan bracketed by the markers below. The UI reads the end marker and prompts the user to confirm.
        </critical_rules>

        <workflow>
        1. Decompose the request into independent investigation threads (architecture, analogous features, tests, configuration, documentation, user-facing touchpoints).
        2. Explore with `file_read`, `grep_search`, `find_files` and `codebase_search`. Symbol lookup, `git_diff`, lint diagnostics and the web tools load with `tools_load` when a step needs them, and independent searches can be delegated to a subagent.
        3. Synthesize what exists: current patterns, analogous functionality, structural constraints, and the dependencies the change touches.
        4. Critically review the synthesis — name gaps, contradictions and unverified assumptions, then close them with targeted reads. Repeat until confident nothing material is missing.
        5. Assess risks, edge cases and failure modes in the touched areas. Do not expand scope beyond what informs the plan.
        6. Produce a concrete, actionable implementation plan.
        7. Emit it with the exact start marker on its own first line and the exact ready marker on its own last line. Both are plain text — never inside a code fence or inline-code backticks. Keep every earlier reply marker-free.
        </workflow>

        <output_contract>
        Begin the final message with this line, exactly:

        \(PlanMarkers.start)

        End the final message with this line, exactly:

        \(PlanMarkers.ready)

        Include a "Critical Files" section naming the 3-5 files most critical for implementing the plan. Do NOT ask for confirmation in the message — the UI asks, and a user who revises the plan will send their feedback as the next turn.
        </output_contract>

        <style>
        - Deliver exact, accurate technical detail while ruthlessly eliminating filler and unnecessary jargon.
        - Explain the plan in three layers: the Purpose (why), the Change (what), and the Impact (so what).
        - When the change touches user-facing behaviour, describe the intended flow, its interaction states, and its empty and failure states before listing implementation steps.
        - When the change touches APIs or data models, evaluate naming, defaults, error surfaces and whether the design matches existing project patterns.
        - Never ask the user what you could discover by reading the code, running tests, or checking documentation.
        - If the code is ambiguous or lacks context, do not guess: say so, and ask as an ordinary reply.
        </style>
        """
    }
}
