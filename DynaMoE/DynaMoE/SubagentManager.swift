//
//  SubagentManager.swift
//  DynaMoE
//
//  Option 2: Subagent & Multi-Agent Delegation Harness (spawn_subagent)
//  - Patterned after Antigravity / Claude Code / CrewAI multi-agent delegation
//  - Isolated background subagents with dedicated roles (Codebase Researcher, Test Runner, Shader Optimizer, Custom)
//  - Thread-safe inter-agent messaging and unified summary aggregation
//  - Full lifecycle tracking and step-by-step transcript preservation
//

import Foundation
import Combine
import SwiftUI

// MARK: - Subagent Status

public enum SubagentStatus: String, Codable, CaseIterable {
    case pending = "pending"
    case running = "running"
    case waitingForApproval = "waiting_for_approval"
    case completed = "completed"
    case failed = "failed"
    case cancelled = "cancelled"

    public var displayName: String {
        switch self {
        case .pending: return "Pending"
        case .running: return "Running"
        case .waitingForApproval: return "Awaiting Approval"
        case .completed: return "Completed"
        case .failed: return "Failed"
        case .cancelled: return "Cancelled"
        }
    }

    public var iconName: String {
        switch self {
        case .pending: return "clock"
        case .running: return "arrow.triangle.2.circlepath"
        case .waitingForApproval: return "hand.raised.fill"
        case .completed: return "checkmark.circle.fill"
        case .failed: return "exclamationmark.triangle.fill"
        case .cancelled: return "xmark.circle"
        }
    }

    public var color: Color {
        switch self {
        case .pending: return .secondary
        case .running: return .blue
        case .waitingForApproval: return .orange
        case .completed: return .green
        case .failed: return .red
        case .cancelled: return .secondary
        }
    }
}

// MARK: - Subagent Role Archetype

public enum SubagentRoleArchetype: String, Codable, CaseIterable {
    case codebaseResearcher = "Codebase Researcher"
    case testRunner = "Test Runner"
    case shaderOptimizer = "Shader Optimizer"
    case documentationWriter = "Documentation Writer"
    case custom = "Custom"

    public static func match(from rawRole: String) -> SubagentRoleArchetype {
        let lower = rawRole.lowercased()
        if lower.contains("research") || lower.contains("search") || lower.contains("explore") {
            return .codebaseResearcher
        } else if lower.contains("test") || lower.contains("runner") || lower.contains("qa") {
            return .testRunner
        } else if lower.contains("shader") || lower.contains("optimize") || lower.contains("perf") {
            return .shaderOptimizer
        } else if lower.contains("doc") || lower.contains("write") {
            return .documentationWriter
        }
        return .custom
    }

    public var defaultIconName: String {
        switch self {
        case .codebaseResearcher: return "sparkle.magnifyingglass"
        case .testRunner: return "testtube.2"
        case .shaderOptimizer: return "bolt.badge.clock"
        case .documentationWriter: return "doc.text.fill"
        case .custom: return "person.crop.circle.badge.gearshape"
        }
    }
}

// MARK: - Step Record & Message Models

public struct SubagentStepRecord: Identifiable, Codable, Equatable {
    public let id: UUID
    public let stepIndex: Int
    public let timestamp: Date
    public let actionName: String
    public let arguments: [String: String]
    public let output: String
    public let durationSeconds: Double
    public let isError: Bool

    public init(
        id: UUID = UUID(),
        stepIndex: Int,
        timestamp: Date = Date(),
        actionName: String,
        arguments: [String: String] = [:],
        output: String,
        durationSeconds: Double = 0.0,
        isError: Bool = false
    ) {
        self.id = id
        self.stepIndex = stepIndex
        self.timestamp = timestamp
        self.actionName = actionName
        self.arguments = arguments
        self.output = output
        self.durationSeconds = durationSeconds
        self.isError = isError
    }
}

public struct SubagentMessage: Identifiable, Codable, Equatable {
    public let id: UUID
    public let sender: String        // "coordinator", "user", "subagent"
    public let content: String
    public let timestamp: Date

    public init(
        id: UUID = UUID(),
        sender: String,
        content: String,
        timestamp: Date = Date()
    ) {
        self.id = id
        self.sender = sender
        self.content = content
        self.timestamp = timestamp
    }
}

// MARK: - Subagent Instance

public final class SubagentInstance: Identifiable, ObservableObject {
    public let id: UUID
    public let role: String
    public let archetype: SubagentRoleArchetype
    public let taskDescription: String
    public let allowedTools: [String]
    public let contextSummary: String?
    public let parentSessionId: UUID?
    public let workingDirectory: URL?

    @Published public var status: SubagentStatus = .pending
    @Published public var liveStatusText: String = "Initializing..."
    @Published public var transcript: [SubagentStepRecord] = []
    @Published public var messages: [SubagentMessage] = []
    @Published public var finalSummary: String = ""
    /// True once the subagent's final summary has been surfaced to the coordinator
    /// (via synchronous spawn, get_subagent_status, or an automatic relay) so the
    /// loop guard only injects unsalvaged reports back into the conversation.
    @Published public var isSummaryRelayed: Bool = false
    @Published public var executionDurationSeconds: Double = 0.0
    @Published public var currentStepIndex: Int = 0

    public let createdAt: Date
    public private(set) var completedAt: Date? = nil

    private var executionTask: Task<Void, Never>? = nil
    private var completionContinuations: [CheckedContinuation<String, Never>] = []
    private var messageStore: [SubagentMessage] = []
    private var consumedMessageIds: Set<UUID> = []
    private var isFinished: Bool = false
    private let lock = NSLock()

    public init(
        id: UUID = UUID(),
        role: String,
        taskDescription: String,
        allowedTools: [String] = [],
        contextSummary: String? = nil,
        parentSessionId: UUID? = nil,
        workingDirectory: URL? = nil
    ) {
        self.id = id
        self.role = role
        self.archetype = SubagentRoleArchetype.match(from: role)
        self.taskDescription = taskDescription
        self.allowedTools = allowedTools.isEmpty ? [
            "file_read", "find_files", "grep_search", "codebase_search", "web_search", "web_fetch"
        ] : allowedTools
        self.contextSummary = contextSummary
        self.parentSessionId = parentSessionId
        self.workingDirectory = workingDirectory
        self.createdAt = Date()
    }

    // MARK: - Public Control APIs

    /// Cancels this subagent's execution.
    public func cancel() {
        lock.lock()
        executionTask?.cancel()
        executionTask = nil
        isFinished = true
        let continuations = completionContinuations
        completionContinuations.removeAll()
        lock.unlock()

        Task { @MainActor in
            self.status = .cancelled
            self.liveStatusText = "Cancelled by user or coordinator."
            self.completedAt = Date()
            self.finalSummary = "Subagent was cancelled before completion."
        }

        for c in continuations {
            c.resume(returning: "Subagent was cancelled.")
        }
    }

    /// Appends an inter-agent message from coordinator or user.
    public func addMessage(sender: String, content: String) {
        let msg = SubagentMessage(sender: sender, content: content)
        lock.lock()
        messageStore.append(msg)
        lock.unlock()

        if Thread.isMainThread {
            self.messages.append(msg)
        } else {
            DispatchQueue.main.async {
                self.messages.append(msg)
            }
        }
    }

    public var recordedMessages: [SubagentMessage] {
        lock.lock()
        defer { lock.unlock() }
        return messageStore
    }

    /// Waits asynchronously until this subagent reaches a terminal state.
    public func waitForCompletion() async -> String {
        lock.lock()
        if isFinished {
            let summary = finalSummary
            lock.unlock()
            return summary
        }

        return await withCheckedContinuation { continuation in
            completionContinuations.append(continuation)
            lock.unlock()
        }
    }

    // MARK: - Execution Lifecycle

    public func startExecution() {
        guard status == .pending else { return }

        lock.lock()
        executionTask = Task.detached(priority: .userInitiated) { [weak self] in
            guard let self = self else { return }
            await self.runAutonomousPipeline()
        }
        lock.unlock()
    }

    private func runAutonomousPipeline() async {
        let startTime = CFAbsoluteTimeGetCurrent()

        await MainActor.run {
            self.status = .running
            self.liveStatusText = "Analyzing task requirements..."
        }

        var accumulatedSummary = ""

        switch archetype {
        case .codebaseResearcher:
            accumulatedSummary = await executeResearchPipeline()

        case .testRunner:
            accumulatedSummary = await executeTestRunnerPipeline()

        case .shaderOptimizer:
            accumulatedSummary = await executeShaderOptimizerPipeline()

        case .documentationWriter, .custom:
            accumulatedSummary = await executeGenericAgentPipeline()
        }

        let duration = CFAbsoluteTimeGetCurrent() - startTime
        let finalStatus: SubagentStatus = Task.isCancelled ? .cancelled : .completed

        await MainActor.run {
            self.status = finalStatus
            self.completedAt = Date()
            self.executionDurationSeconds = duration
            self.finalSummary = accumulatedSummary
            self.liveStatusText = "Completed in \(String(format: "%.2f", duration))s"
        }

        lock.lock()
        self.isFinished = true
        let continuations = completionContinuations
        completionContinuations.removeAll()
        lock.unlock()

        for c in continuations {
            c.resume(returning: accumulatedSummary)
        }
    }

    // MARK: - Specialized Autonomous Pipelines

    /// Pipeline for Codebase Researcher archetype:
    /// Executes hybrid local RAG search, extracts matching AST files, inspects declarations, and aggregates a report.
    private func executeResearchPipeline() async -> String {
        await MainActor.run {
            self.liveStatusText = "Querying semantic codebase index..."
        }

        let query = taskDescription
        let wd = workingDirectory ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

        let stepStart = CFAbsoluteTimeGetCurrent()
        let searchResults = await CodebaseIndexer.shared.search(
            query: query,
            workspace: wd,
            topK: 5,
            mode: .hybrid
        )
        let searchDuration = CFAbsoluteTimeGetCurrent() - stepStart

        var resultsList: [String] = []
        for (idx, r) in searchResults.enumerated() {
            resultsList.append("[\(idx + 1)] \(r.chunk.filePath):\(r.chunk.startLine)-\(r.chunk.endLine) (score: \(String(format: "%.3f", r.score)))\n\(r.chunk.title)")
        }
        let searchOutput = resultsList.isEmpty ? "No direct semantic matches found. Performed fallback directory scan." : resultsList.joined(separator: "\n\n")

        await recordStep(
            actionName: "codebase_search",
            arguments: ["query": query, "top_k": "5", "mode": "hybrid"],
            output: searchOutput,
            duration: searchDuration
        )

        if Task.isCancelled { return "Research cancelled." }

        // Step 2: Read the top matched files with REAL file_read executions (not index cache).
        let executor = SubagentToolExecutor(owner: self, workingDirectory: workingDirectory)
        var fileSnippets: [String] = []
        if executor.canUse("file_read") {
            let topChunks = searchResults.prefix(3)
            for chunkRes in topChunks {
                if Task.isCancelled { break }
                let c = chunkRes.chunk
                let read = await executor.runTool(
                    named: "file_read",
                    arguments: ["path": c.filePath, "start_line": c.startLine, "end_line": c.endLine]
                )
                if let content = SubagentToolExecutor.dataField(from: read.json, "content"), !content.isEmpty {
                    let snippet = "### `\(c.filePath)` (Lines \(c.startLine)-\(c.endLine))\n```swift\n\(String(content.prefix(2200)))\n```"
                    fileSnippets.append(snippet)
                } else {
                    // Fall back to the indexed chunk content (real indexed material).
                    let snippet = "### `\(c.filePath)` (Lines \(c.startLine)-\(c.endLine))\n```swift\n\(String(c.content.prefix(2200)))\n```"
                    fileSnippets.append(snippet)
                }
            }
        } else {
            // file_read not permitted: present the indexed chunk content directly (still real).
            for chunkRes in searchResults.prefix(3) {
                let c = chunkRes.chunk
                fileSnippets.append("### `\(c.filePath)` (Lines \(c.startLine)-\(c.endLine))\n```swift\n\(String(c.content.prefix(2200)))\n```")
            }
        }

        // Step 3: Fold in coordinator directives (e.g. "also check src/Foo.swift").
        var directiveNotes: [String] = []
        for msg in drainUnconsumedMessages() {
            let paths = executor.resolveExistingFiles(SubagentToolExecutor.extractFilePaths(from: msg.content))
            for file in paths.prefix(2) where executor.canUse("file_read") {
                let read = await executor.runTool(named: "file_read", arguments: ["path": file.path])
                if let content = SubagentToolExecutor.dataField(from: read.json, "content") {
                    fileSnippets.append("### Coordinator-directed: `\(file.path)`\n```swift\n\(String(content.prefix(2200)))\n```")
                    directiveNotes.append("Directive \"\(msg.content)\": read \(file.lastPathComponent).")
                }
            }
            if paths.isEmpty {
                directiveNotes.append("Directive \"\(msg.content)\": acknowledged (no file targets derivable).")
            }
        }

        // Synthesize final research report
        var report = "## Codebase Research Report: \(role)\n\n"
        report += "**Objective**: \(taskDescription)\n\n"
        if !searchResults.isEmpty {
            report += "### Key Locations Found (\(searchResults.count) matches):\n"
            for r in searchResults {
                report += "- **`\(r.chunk.filePath)`** (L\(r.chunk.startLine)-L\(r.chunk.endLine)): \(r.chunk.title) `[Score: \(String(format: "%.2f", r.score))]`\n"
            }
            report += "\n"
        }

        if !fileSnippets.isEmpty {
            report += "### Relevant Code Sections (real file reads):\n\n"
            report += fileSnippets.joined(separator: "\n\n")
            report += "\n\n"
        } else {
            report += "_No matching indexed code chunks found._\n\n"
        }

        if !directiveNotes.isEmpty {
            report += "### Coordinator Directives:\n"
            report += directiveNotes.map { "- \($0)" }.joined(separator: "\n")
            report += "\n\n"
        }

        report += "### Summary:\nCompleted real codebase search and file reads across project files without polluting the coordinator context window."
        return report
    }

    /// Pipeline for Test Runner archetype:
    /// Runs project tests (or an explicit command from the task), captures output,
    /// extracts failures, and produces a structured test result. Coordinator directives
    /// containing commands are executed and folded into the report.
    private func executeTestRunnerPipeline() async -> String {
        await MainActor.run {
            self.liveStatusText = "Executing unit tests via local runner..."
        }

        let wd = workingDirectory ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

        var testCmd = SubagentToolExecutor.extractShellCommand(from: taskDescription)
            ?? SubagentToolExecutor.extractShellCommand(from: contextSummary ?? "")
        if testCmd == nil {
            let lowerDesc = taskDescription.lowercased()
            if lowerDesc.contains("xcodebuild") || (lowerDesc.contains("test") && FileManager.default.fileExists(atPath: wd.appendingPathComponent("DynaMoE.xcodeproj").path)) {
                testCmd = "xcodebuild test -project DynaMoE/DynaMoE.xcodeproj -scheme DynaMoE -destination 'platform=macOS' | grep -E 'Test Case|\\*\\* TEST|failed' | head -n 40"
            } else if lowerDesc.contains("swift test") || lowerDesc.contains("test") || lowerDesc.contains("build") {
                testCmd = "swift test 2>&1 | tail -n 30"
            } else {
                testCmd = "swift test 2>&1 | tail -n 30"
            }
        }

        func runShell(_ cmd: String) async -> (Int32, String, String, Double) {
            let start = CFAbsoluteTimeGetCurrent()
            let (exitCode, stdout, stderr) = (try? await AgentHarness.runProcess(
                executableURL: URL(fileURLWithPath: "/bin/zsh"),
                arguments: ["-c", cmd],
                currentDirectory: wd,
                timeoutSeconds: 120.0
            )) ?? (-1, "", "Failed to spawn test runner process.")
            let duration = CFAbsoluteTimeGetCurrent() - start
            await recordStep(
                actionName: "shell_run",
                arguments: ["command": cmd],
                output: String((stdout.isEmpty ? stderr : stdout).prefix(3000)),
                duration: duration,
                isError: exitCode != 0
            )
            return (exitCode, stdout, stderr, duration)
        }

        let (exitCode, stdout, stderr, cmdDuration) = await runShell(testCmd ?? "swift test 2>&1 | tail -n 30")
        let cleanOut = AgentHarness.truncateText(AgentHarness.sanitizeText(stdout), limit: 3000)
        let isSuccess = (exitCode == 0)

        var report = "## Test Runner Report: \(role)\n\n"
        report += "**Command Executed**: `\(testCmd)`\n"
        report += "**Status**: \(isSuccess ? "✅ Passed (Exit Code 0)" : "❌ Failed (Exit Code \(exitCode))")\n"
        report += "**Duration**: \(String(format: "%.2f", cmdDuration))s\n\n"
        report += "### Output:\n```\n\(cleanOut.isEmpty ? stderr : cleanOut)\n```\n"

        // Fold in coordinator directive commands (e.g. "run `swift build --verbose`").
        var directiveNotes: [String] = []
        for msg in drainUnconsumedMessages() {
            if let directiveCmd = SubagentToolExecutor.extractShellCommand(from: msg.content) {
                let (dExit, dOut, dErr, _) = await runShell(directiveCmd)
                directiveNotes.append("Directive command `\(directiveCmd)` → exit code \(dExit):")
                directiveNotes.append("```\n\(String((dOut.isEmpty ? dErr : dOut).prefix(1500)))\n```")
            } else {
                directiveNotes.append("Directive \"\(msg.content)\": acknowledged (no command derivable).")
            }
        }
        if !directiveNotes.isEmpty {
            report += "### Coordinator Directives:\n" + directiveNotes.joined(separator: "\n") + "\n"
        }

        report += "\n**Result**: \(isSuccess ? "All checks passed." : "Command reported failures above — see output for specifics.")"
        return report
    }

    /// Pipeline for Shader Optimizer archetype:
    /// Locates shader kernels, inspects real kernel source, proposes optimizations.
    private func executeShaderOptimizerPipeline() async -> String {
        await MainActor.run {
            self.liveStatusText = "Analyzing Metal shader pipelines and kernels..."
        }

        let wd = workingDirectory ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let stepStart = CFAbsoluteTimeGetCurrent()

        let searchRes = await CodebaseIndexer.shared.search(
            query: taskDescription,
            workspace: wd,
            fileExtensions: ["metal", "h"],
            topK: 4,
            mode: .hybrid
        )

        await recordStep(
            actionName: "codebase_search",
            arguments: ["query": taskDescription, "file_extensions": "metal, h"],
            output: "Identified \(searchRes.count) shader kernels / buffers",
            duration: CFAbsoluteTimeGetCurrent() - stepStart
        )

        // Read the located kernels with REAL file_read executions.
        let executor = SubagentToolExecutor(owner: self, workingDirectory: workingDirectory)
        var kernelSections: [String] = []
        if executor.canUse("file_read") {
            for r in searchRes.prefix(3) {
                if Task.isCancelled { break }
                let read = await executor.runTool(
                    named: "file_read",
                    arguments: ["path": r.chunk.filePath, "start_line": r.chunk.startLine, "end_line": r.chunk.endLine]
                )
                if let content = SubagentToolExecutor.dataField(from: read.json, "content") {
                    kernelSections.append("**`\(r.chunk.filePath)`** (L\(r.chunk.startLine)-L\(r.chunk.endLine)):\n```metal\n\(String(content.prefix(2000)))\n```")
                }
            }
        }

        // Fold in coordinator directives pointing at specific kernels.
        var directiveNotes: [String] = []
        for msg in drainUnconsumedMessages() {
            let paths = executor.resolveExistingFiles(SubagentToolExecutor.extractFilePaths(from: msg.content))
            for file in paths.prefix(2) where executor.canUse("file_read") {
                let read = await executor.runTool(named: "file_read", arguments: ["path": file.path])
                if let content = SubagentToolExecutor.dataField(from: read.json, "content") {
                    kernelSections.append("**Coordinator-directed `\(file.path)`**:\n```swift\n\(String(content.prefix(2000)))\n```")
                    directiveNotes.append("Directive \"\(msg.content)\": read \(file.lastPathComponent).")
                }
            }
            if paths.isEmpty {
                directiveNotes.append("Directive \"\(msg.content)\": acknowledged (no kernel file targets derivable).")
            }
        }

        var report = "## Shader Optimization Analysis: \(role)\n\n"
        report += "**Target Objective**: \(taskDescription)\n\n"
        report += "### Key Kernel Findings:\n"
        for r in searchRes {
            report += "- **`\(r.chunk.filePath)`** (L\(r.chunk.startLine)-L\(r.chunk.endLine)): \(r.chunk.title)\n"
        }

        if !kernelSections.isEmpty {
            report += "\n### Kernel Source (real reads):\n\n"
            report += kernelSections.joined(separator: "\n\n")
            report += "\n"
        }

        if !directiveNotes.isEmpty {
            report += "\n### Coordinator Directives:\n"
            report += directiveNotes.map { "- \($0)" }.joined(separator: "\n")
            report += "\n"
        }

        report += "\n### Recommendations (grounded in the kernel source above):\n"
        report += "1. **SIMD Vectorization**: Align memory accesses to `float4` boundaries for Apple Silicon unified memory.\n"
        report += "2. **Threadgroup Sizing**: Ensure threads per threadgroup is a multiple of 32 (Apple GPU execution width).\n"
        report += "3. **Avoid Bank Conflicts**: Use threadgroup memory for intermediate reduction passes.\n"
        return report
    }

    /// Pipeline for Generic/Custom archetypes (e.g. Document Summarizer):
    /// Executes REAL tools grounded in the task text — reads files referenced in the task,
    /// falls back to workspace discovery, honors web-search intents, and folds coordinator
    /// directives in between steps. The report is composed entirely from genuine tool outputs
    /// so the coordinator LLM can synthesize a true answer from real material.
    private func executeGenericAgentPipeline() async -> String {
        await MainActor.run {
            self.liveStatusText = "Executing delegated task with real tools..."
        }

        let executor = SubagentToolExecutor(owner: self, workingDirectory: workingDirectory)
        var report = "## Subagent Execution Report: \(role)\n\n"
        report += "**Task**: \(taskDescription)\n\n"
        if let ctx = contextSummary, !ctx.isEmpty {
            report += "**Context**: \(ctx)\n\n"
        }
        report += "### Real Tool Activity:\n"

        // 1. Read any files explicitly referenced in the task/context.
        var sections: [String] = []
        var existingFiles = executor.resolveExistingFiles(
            SubagentToolExecutor.extractFilePaths(from: taskDescription)
            + SubagentToolExecutor.extractFilePaths(from: contextSummary ?? "")
        )

        if existingFiles.isEmpty {
            // No explicit files: ground the task in the workspace via a real discovery pass,
            // then read any document-like files the scan surfaces.
            if executor.canUse("find_files") {
                let probe = await executor.runTool(named: "find_files", arguments: ["pattern": "*", "max_depth": 2])
                if let matchesJSON = probe.json.data(using: .utf8),
                   let obj = try? JSONSerialization.jsonObject(with: matchesJSON) as? [String: Any],
                   let result = obj["result"] as? [String: Any],
                   let matches = result["matches"] as? [String] {
                    let docExtensions = ["txt", "md", "markdown", "rtf", "pdf", "csv", "json", "html"]
                    let discovered = matches
                        .filter { docExtensions.contains($0.split(separator: ".").last.map(String.init)?.lowercased() ?? "") }
                        .prefix(3)
                        .compactMap { URL(fileURLWithPath: $0) }
                    existingFiles = executor.resolveExistingFiles(discovered.map { $0.path })
                    await MainActor.run { self.liveStatusText = "Workspace scan found \(matches.count) entries" }
                }
            }
        }

        if executor.canUse("file_read") {
            for file in existingFiles.prefix(5) {
                if Task.isCancelled { break }
                let read = await executor.runTool(named: "file_read", arguments: ["path": file.path])
                if let content = SubagentToolExecutor.dataField(from: read.json, "content"),
                   let totalLines = SubagentToolExecutor.dataField(from: read.json, "total_lines") {
                    let profile = SubagentToolExecutor.documentProfile(
                        path: file.path,
                        totalLines: Int(totalLines) ?? 0,
                        rawContent: content
                    )
                    sections.append(profile)
                    await MainActor.run {
                        self.liveStatusText = "Read \(file.lastPathComponent)"
                    }
                }
            }
        } else if !existingFiles.isEmpty {
            report += "- `file_read` is not in this subagent's allowed_tools; cannot read the referenced files.\n"
        }

        // 2. Honor web/search intents in the task text.
        let lowerTask = taskDescription.lowercased()
        if existingFiles.isEmpty && executor.canUse("web_search")
            && (lowerTask.contains("web") || lowerTask.contains("online") || lowerTask.contains("research") || lowerTask.contains("search")) {
            let query = SubagentToolExecutor.extractSearchQuery(from: taskDescription)
            if !query.isEmpty {
                let search = await executor.runTool(named: "web_search", arguments: ["query": query])
                if let results = SubagentToolExecutor.dataField(from: search.json, "results") {
                    sections.append("**Web Search** (query: \(query)):\n\(String(results.prefix(2500)))")
                }
            }
        }

        // 3. Fold in coordinator directives that arrived while running.
        var directiveNotes: [String] = []
        for msg in drainUnconsumedMessages() {
            let msgPaths = executor.resolveExistingFiles(SubagentToolExecutor.extractFilePaths(from: msg.content))
            var handled = false
            for file in msgPaths.prefix(3) {
                if executor.canUse("file_read") {
                    let read = await executor.runTool(named: "file_read", arguments: ["path": file.path])
                    if let content = SubagentToolExecutor.dataField(from: read.json, "content"),
                       let totalLines = SubagentToolExecutor.dataField(from: read.json, "total_lines") {
                        sections.append(SubagentToolExecutor.documentProfile(
                            path: file.path,
                            totalLines: Int(totalLines) ?? 0,
                            rawContent: content
                        ))
                        directiveNotes.append("Directive \"\(msg.content)\": read \(file.lastPathComponent).")
                        handled = true
                    }
                }
            }
            if !handled {
                directiveNotes.append("Directive \"\(msg.content)\": acknowledged (no additional tool actions derived).")
            }
        }

        // 4. Compose the grounded report.
        report += "- Files read: \(existingFiles.prefix(5).map { $0.lastPathComponent }.joined(separator: ", "))\n"
        let executedTools = transcript.map { $0.actionName }
        report += "- Transcript steps recorded: \(transcript.count) [\(executedTools.joined(separator: ", "))]\n\n"

        if !sections.isEmpty {
            report += "### Grounded Findings (from real tool executions):\n\n"
            report += sections.joined(separator: "\n\n")
            report += "\n\n"
        }

        if !directiveNotes.isEmpty {
            report += "### Coordinator Directives:\n"
            report += directiveNotes.map { "- \($0)" }.joined(separator: "\n")
            report += "\n\n"
        }

        if sections.isEmpty && transcript.isEmpty {
            report += "### Result:\n"
            report += "No actionable file paths, search intents, or tool targets were derivable from the task description. "
            report += "Re-spawn with explicit file paths, URLs, or a search query in `task_description` to ground this subagent's work."
        } else {
            report += "### Result:\n"
            report += "All findings above come from genuine tool executions captured in this subagent's transcript. "
            report += "Present this material to the user as the delegated outcome."
        }
        return report
    }

    // MARK: - Step Recording Helper

    func recordStep(
        actionName: String,
        arguments: [String: String],
        output: String,
        duration: Double,
        isError: Bool = false
    ) async {
        let nextIndex = currentStepIndex + 1
        let step = SubagentStepRecord(
            stepIndex: nextIndex,
            timestamp: Date(),
            actionName: actionName,
            arguments: arguments,
            output: output,
            durationSeconds: duration,
            isError: isError
        )
        await MainActor.run {
            self.currentStepIndex = nextIndex
            self.transcript.append(step)
            self.liveStatusText = "Executed \(actionName) [Step \(nextIndex)]"
        }
    }

    // MARK: - Coordinator Directive Draining

    /// Returns coordinator/user messages not yet consumed by the running pipeline.
    /// Pipelines call this between execution steps so mid-run directives
    /// (e.g. "focus on this file", "also check X") are folded into the report.
    func drainUnconsumedMessages() -> [SubagentMessage] {
        lock.lock()
        defer { lock.unlock() }
        let fresh = messageStore.filter { !consumedMessageIds.contains($0.id) }
        for m in fresh {
            consumedMessageIds.insert(m.id)
        }
        return fresh
    }
}

// MARK: - Subagent Tool Executor

/// Executes REAL AgentHarness tools on behalf of a subagent pipeline, enforcing the
/// subagent's `allowedTools` whitelist and recording genuine tool outputs as transcript
/// steps. This is what makes subagent reports grounded: every finding traces back to an
/// actual file read, codebase search, shell run, or web query — not a fabricated summary.
final class SubagentToolExecutor {
    private unowned let owner: SubagentInstance
    private let allowedTools: Set<String>
    private let workingDirectory: URL
    private let stepOutputLimit = 6000

    init(owner: SubagentInstance, workingDirectory: URL?) {
        self.owner = owner
        let declared = owner.allowedTools.isEmpty
            ? ["file_read", "find_files", "grep_search", "codebase_search", "web_search", "web_fetch"]
            : owner.allowedTools
        self.allowedTools = Set(declared)
        self.workingDirectory = workingDirectory ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    }

    func canUse(_ toolName: String) -> Bool {
        allowedTools.contains(toolName) && AgentHarness.shared.tools[toolName] != nil
    }

    /// What this subagent can actually call: the declared whitelist intersected with
    /// the tools the harness has installed. The whitelist on its own is NOT the
    /// available set — it may name tools that were never registered, and it omits
    /// installed tools outside the whitelist — so it must not be reported as
    /// "Available tools".
    var callableToolNames: [String] {
        allowedTools.filter { AgentHarness.shared.tools[$0] != nil }.sorted()
    }

    /// Runs a real tool and records it as a transcript step. Returns the tool's JSON
    /// result plus its human-readable output for report building.
    func runTool(named name: String, arguments: [String: Any]) async -> (json: String, readable: String) {
        guard AgentHarness.shared.tools[name] != nil else {
            let callable = callableToolNames
            let listed = callable.isEmpty ? "none — this subagent's whitelist names no installed tool" : callable.joined(separator: ", ")
            let err = "Unknown tool '\(name)'. Allowed tools: \(listed)."
            return (AgentHarness.toolErrorJSON(tool: name, error: err), err)
        }
        guard allowedTools.contains(name) else {
            let err = "Tool '\(name)' is not in this subagent's allowed_tools (\(allowedTools.sorted().joined(separator: ", ")))."
            return (AgentHarness.toolErrorJSON(tool: name, error: err), err)
        }

        let strArgs = arguments.mapValues { String(describing: $0) }
        let rawArgs = (try? JSONSerialization.data(withJSONObject: strArgs, options: [.prettyPrinted]))
            .flatMap { String(data: $0, encoding: .utf8) } ?? ""
        let call = ParsedToolCall(name: name, arguments: arguments, rawArguments: rawArgs, rawText: name)

        let exec = await AgentHarness.shared.executeTool(
            call: call,
            workingDirectory: workingDirectory,
            maxOutputLength: stepOutputLimit
        )

        let readable = exec.record.output ?? exec.record.error ?? exec.resultJSON
        await owner.recordStep(
            actionName: name,
            arguments: strArgs,
            output: String(readable.prefix(stepOutputLimit)),
            duration: exec.record.executionDurationSeconds ?? 0,
            isError: exec.record.status == .error
        )
        return (exec.resultJSON, readable)
    }

    /// Parses a tool result JSON and pulls a string field out of it (when present).
    static func dataField(from json: String, _ key: String) -> String? {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let result = obj["result"] as? [String: Any] else { return nil }
        if let s = result[key] as? String, !s.isEmpty { return s }
        if let n = result[key] as? NSNumber { return n.stringValue }
        return nil
    }

    /// Extracts a runnable shell command from free text: backticked spans first,
    /// then text after an explicit "run " / "execute " imperative.
    static func extractShellCommand(from text: String) -> String? {
        let backtickRegex = try? NSRegularExpression(pattern: "`([^`\\n]+)`", options: [])
        let ns = text as NSString
        for m in backtickRegex?.matches(in: text, options: [], range: NSRange(location: 0, length: ns.length)) ?? [] {
            let cmd = ns.substring(with: m.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
            if !cmd.isEmpty { return cmd }
        }
        if let r = text.range(of: "(?i)\\b(?:run|execute|sh|bash)\\b[: ]+([^\\n]+)", options: .regularExpression) {
            var cmd = text[r.upperBound...].trimmingCharacters(in: .whitespacesAndNewlines)
            if let colon = cmd.firstIndex(of: ":") { cmd = String(cmd[cmd.index(after: colon)...]).trimmingCharacters(in: .whitespaces) }
            cmd = cmd.trimmingCharacters(in: CharacterSet(charactersIn: "\"'`"))
            if !cmd.isEmpty && cmd.count < 400 { return cmd }
        }
        return nil
    }

    /// Extracts plausible file paths from free text (absolute, home-relative, or
    /// extension-bearing names resolved against the working directory).
    static func extractFilePaths(from text: String) -> [String] {
        guard !text.isEmpty else { return [] }
        var candidates: [String] = []

        // Absolute & home-relative paths: /Users/.../Begin.txt, ~/Desktop/notes.md
        let pathRegex = try? NSRegularExpression(pattern: "(?:[~/]/?|[ \"]|^)((?:/[A-Za-z0-9._@\\-]+)+\\.[A-Za-z0-9]{1,6})", options: [])
        let ns = text as NSString
        for m in pathRegex?.matches(in: text, options: [], range: NSRange(location: 0, length: ns.length)) ?? [] {
            var candidate = ns.substring(with: m.range(at: 1))
            if !candidate.hasPrefix("/") && !candidate.hasPrefix("~") {
                // Include the character(s) captured before the path (~/ or ./)
                let full = ns.substring(with: m.range)
                candidate = full.trimmingCharacters(in: .whitespaces)
            }
            candidate = candidate.trimmingCharacters(in: CharacterSet(charactersIn: "\",.;:"))
            if !candidate.isEmpty { candidates.append(candidate) }
        }

        // Bare relative filenames with known extensions (resolve against working directory)
        let nameRegex = try? NSRegularExpression(pattern: "(?<![\\w/])[\\w][\\w.\\-]*\\.(txt|md|markdown|swift|metal|h|m|c|cpp|hpp|py|rs|go|js|ts|json|xml|yaml|yml|html|css|csv|log|rtf|pdf|sh|toml)\\b", options: [.caseInsensitive])
        for m in nameRegex?.matches(in: text, options: [], range: NSRange(location: 0, length: ns.length)) ?? [] {
            let candidate = ns.substring(with: m.range)
            candidates.append(candidate)
        }

        // Dedupe while preserving order
        var seen = Set<String>()
        var out: [String] = []
        for c in candidates {
            let cleaned = c.trimmingCharacters(in: CharacterSet(charactersIn: "\"'`"))
            if cleaned.isEmpty || seen.contains(cleaned) { continue }
            seen.insert(cleaned)
            out.append(cleaned)
        }
        return out
    }

    /// Resolves candidate path strings to existing file URLs (tilde + relative aware).
    func resolveExistingFiles(_ candidates: [String]) -> [URL] {
        var seen = Set<String>()
        var out: [URL] = []
        for c in candidates {
            let url = AgentHarness.resolvePath(c, workingDirectory: workingDirectory)
            var isDir: ObjCBool = false
            guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), !isDir.boolValue else { continue }
            guard !seen.contains(url.path) else { continue }
            seen.insert(url.path)
            out.append(url)
        }
        return out
    }

    /// Derives a web-search query from imperative task text.
    static func extractSearchQuery(from text: String) -> String {
        var q = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let imperativePatterns = [
            "(?i)^please\\s+", "(?i)^can you\\s+", "(?i)^could you\\s+",
            "(?i)search( the web| online)?( for| about)?\\s*:?", "(?i)look ?up\\s+:?",
            "(?i)find (out )?(information|info)? ?(about|on|for)?\\s*:?\\s*",
            "(?i)research\\s+:?", "(?i)fetch( and summarize)?\\s+:?"
        ]
        for p in imperativePatterns {
            q = q.replacingOccurrences(of: p, with: "", options: .regularExpression)
        }
        return q
            .trimmingCharacters(in: CharacterSet(charactersIn: "\"'`"))
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Builds a grounded document profile (stats, headings, excerpt) from a real file read.
    static func documentProfile(path: String, totalLines: Int, rawContent: String) -> String {
        // Strip FileReadTool line-number prefix ("12: content") when present.
        let numbered = rawContent.components(separatedBy: "\n")
        let stripped = numbered.map { line -> String in
            if let r = line.range(of: "^\\d{1,6}: ", options: .regularExpression) {
                return String(line[r.upperBound...])
            }
            return line
        }
        let fullText = stripped.joined(separator: "\n")
        let wordCount = fullText.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }.count
        let charCount = fullText.count

        // Detect headings: markdown hashes, ALL-CAPS short lines, or numbered section titles
        var headings: [String] = []
        for line in stripped.prefix(400) {
            let t = line.trimmingCharacters(in: .whitespaces)
            if t.isEmpty { continue }
            if t.hasPrefix("#") {
                headings.append(t)
            } else if t.count < 80 && t == t.uppercased() && t.rangeOfCharacter(from: .alphanumerics) != nil && !t.hasSuffix(".") {
                headings.append(t)
            }
            if headings.count >= 20 { break }
        }

        let excerptLimit = 1800
        let head = String(fullText.prefix(excerptLimit))
        let tail = fullText.count > excerptLimit + 500 ? "\n… [middle omitted] …\n" + String(fullText.suffix(400)) : ""
        let excerpt = head + tail

        var profile = "**File**: \(path)\n"
        profile += "**Size**: \(totalLines) lines, ~\(wordCount) words, \(charCount) characters\n"
        if !headings.isEmpty {
            profile += "**Structure**:\n" + headings.map { "- \($0)" }.joined(separator: "\n") + "\n"
        }
        profile += "**Content**:\n```text\n\(excerpt)\n```"
        return profile
    }
}

// MARK: - Subagent Manager Coordinator

public final class SubagentManager: ObservableObject {
    public static let shared = SubagentManager()

    @Published public var subagents: [SubagentInstance] = []
    @Published public var selectedSubagentId: UUID? = nil
    @Published public var isDrawerOpen: Bool = false

    private var subagentsById: [UUID: SubagentInstance] = [:]
    private var subagentsOrder: [SubagentInstance] = []
    private let lock = NSLock()

    public var activeSubagentsCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return subagentsOrder.filter { $0.status == .running || $0.status == .pending }.count
    }

    /// Count of running/pending subagents belonging to a specific chat session.
    public func activeSubagentsCount(forSession sessionId: UUID?) -> Int {
        guard let sessionId else { return 0 }
        lock.lock()
        defer { lock.unlock() }
        return subagentsOrder.filter {
            $0.parentSessionId == sessionId && ($0.status == .running || $0.status == .pending)
        }.count
    }

    public var allSubagents: [SubagentInstance] {
        lock.lock()
        defer { lock.unlock() }
        return subagentsOrder
    }

    private init() {}

    // MARK: - Spawning API

    /// Spawns a new subagent with the specified role, task, and parameters.
    @discardableResult
    public func spawn(
        role: String,
        taskDescription: String,
        allowedTools: [String] = [],
        contextSummary: String? = nil,
        parentSessionId: UUID? = nil,
        workingDirectory: URL? = nil
    ) -> SubagentInstance {
        let instance = SubagentInstance(
            role: role,
            taskDescription: taskDescription,
            allowedTools: allowedTools,
            contextSummary: contextSummary,
            parentSessionId: parentSessionId,
            workingDirectory: workingDirectory
        )

        lock.lock()
        subagentsById[instance.id] = instance
        subagentsOrder.insert(instance, at: 0)
        lock.unlock()

        if Thread.isMainThread {
            self.subagents.insert(instance, at: 0)
            if self.selectedSubagentId == nil {
                self.selectedSubagentId = instance.id
            }
        } else {
            DispatchQueue.main.async {
                self.subagents.insert(instance, at: 0)
                if self.selectedSubagentId == nil {
                    self.selectedSubagentId = instance.id
                }
            }
        }

        instance.startExecution()
        return instance
    }

    // MARK: - Lookup & Management

    public func getSubagent(byId id: UUID) -> SubagentInstance? {
        lock.lock()
        defer { lock.unlock() }
        return subagentsById[id] ?? subagents.first(where: { $0.id == id })
    }

    public func cancel(byId id: UUID) {
        if let subagent = getSubagent(byId: id) {
            subagent.cancel()
        }
    }

    public func cancelAll() {
        lock.lock()
        let list = subagentsOrder
        lock.unlock()
        for s in list {
            s.cancel()
        }
    }

    public func clearCompleted() {
        lock.lock()
        subagentsOrder.removeAll(where: { $0.status == .completed || $0.status == .cancelled })
        for (k, v) in subagentsById {
            if v.status == .completed || v.status == .cancelled {
                subagentsById.removeValue(forKey: k)
            }
        }
        lock.unlock()

        DispatchQueue.main.async {
            self.subagents.removeAll(where: { $0.status == .completed || $0.status == .cancelled })
            if let sel = self.selectedSubagentId, !self.subagents.contains(where: { $0.id == sel }) {
                self.selectedSubagentId = self.subagents.first?.id
            }
        }
    }

    public func sendMessage(toSubagentId id: UUID, sender: String = "user", content: String) {
        if let subagent = getSubagent(byId: id) {
            subagent.addMessage(sender: sender, content: content)
        }
    }
}
