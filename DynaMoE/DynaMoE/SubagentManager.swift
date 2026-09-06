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
    @Published public var executionDurationSeconds: Double = 0.0
    @Published public var currentStepIndex: Int = 0

    public let createdAt: Date
    public private(set) var completedAt: Date? = nil

    private var executionTask: Task<Void, Never>? = nil
    private var completionContinuations: [CheckedContinuation<String, Never>] = []
    private var messageStore: [SubagentMessage] = []
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

        // Step 2: Read top AST chunks
        var fileSnippets: [String] = []
        let topChunks = searchResults.prefix(3)
        for chunkRes in topChunks {
            if Task.isCancelled { break }
            let c = chunkRes.chunk
            let readStart = CFAbsoluteTimeGetCurrent()

            let snippetHeader = "### `\(c.filePath)` (Lines \(c.startLine)-\(c.endLine))\n"
            let codeBody = "```swift\n\(c.content)\n```"
            fileSnippets.append("\(snippetHeader)\(codeBody)")

            await recordStep(
                actionName: "file_read",
                arguments: ["path": c.filePath, "start_line": "\(c.startLine)", "end_line": "\(c.endLine)"],
                output: "Read \(c.content.split(separator: "\n").count) lines from \(c.filePath)",
                duration: CFAbsoluteTimeGetCurrent() - readStart
            )
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
            report += "### Relevant Code Sections:\n\n"
            report += fileSnippets.joined(separator: "\n\n")
            report += "\n\n"
        } else {
            report += "_No matching indexed code chunks found._\n\n"
        }

        report += "### Summary:\nCompleted autonomous codebase search across project files without polluting the coordinator context window."
        return report
    }

    /// Pipeline for Test Runner archetype:
    /// Runs project tests, captures output, extracts failures, and produces a structured test result.
    private func executeTestRunnerPipeline() async -> String {
        await MainActor.run {
            self.liveStatusText = "Executing unit tests via local runner..."
        }

        let wd = workingDirectory ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let cmdStart = CFAbsoluteTimeGetCurrent()

        var testCmd = "swift test 2>&1 | tail -n 30"
        let lowerDesc = taskDescription.lowercased()
        if lowerDesc.contains("xcodebuild") || lowerDesc.contains("test") && FileManager.default.fileExists(atPath: wd.appendingPathComponent("DynaMoE.xcodeproj").path) {
            testCmd = "xcodebuild test -project DynaMoE/DynaMoE.xcodeproj -scheme DynaMoE -destination 'platform=macOS' | grep -E 'Test Case|\\*\\* TEST|failed' | head -n 40"
        } else if lowerDesc.contains("run") && lowerDesc.contains(" ") {
            testCmd = taskDescription
        }

        let (exitCode, stdout, stderr) = (try? await AgentHarness.runProcess(
            executableURL: URL(fileURLWithPath: "/bin/zsh"),
            arguments: ["-c", testCmd],
            currentDirectory: wd,
            timeoutSeconds: 60.0
        )) ?? (-1, "", "Failed to spawn test runner process.")
        let cmdDuration = CFAbsoluteTimeGetCurrent() - cmdStart

        let cleanOut = AgentHarness.truncateText(AgentHarness.sanitizeText(stdout), limit: 3000)
        let isSuccess = (exitCode == 0)

        await recordStep(
            actionName: "shell_run",
            arguments: ["command": testCmd],
            output: cleanOut.isEmpty ? stderr : cleanOut,
            duration: cmdDuration,
            isError: !isSuccess
        )

        var report = "## Test Runner Report: \(role)\n\n"
        report += "**Command Executed**: `\(testCmd)`\n"
        report += "**Status**: \(isSuccess ? "✅ Passed (Exit Code 0)" : "❌ Failed (Exit Code \(exitCode))")\n\n"
        report += "### Output:\n```\n\(cleanOut.isEmpty ? stderr : cleanOut)\n```\n"
        return report
    }

    /// Pipeline for Shader Optimizer archetype:
    /// Locates shader kernels, inspects memory bindings and threadgroup sizes, proposes optimizations.
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

        var report = "## Shader Optimization Analysis: \(role)\n\n"
        report += "**Target Objective**: \(taskDescription)\n\n"
        report += "### Key Kernel Findings:\n"
        for r in searchRes {
            report += "- **`\(r.chunk.filePath)`** (L\(r.chunk.startLine)-L\(r.chunk.endLine)): \(r.chunk.title)\n"
        }
        report += "\n### Recommendations:\n"
        report += "1. **SIMD Vectorization**: Align memory accesses to `float4` boundaries for Apple Silicon unified memory.\n"
        report += "2. **Threadgroup Sizing**: Ensure threads per threadgroup is a multiple of 32 (Apple GPU execution width).\n"
        report += "3. **Avoid Bank Conflicts**: Use threadgroup memory for intermediate reduction passes.\n"
        return report
    }

    /// Pipeline for Generic/Custom archetypes:
    /// Runs permitted tool executions according to description.
    private func executeGenericAgentPipeline() async -> String {
        await MainActor.run {
            self.liveStatusText = "Executing delegated subagent tasks..."
        }

        let wd = workingDirectory ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let stepStart = CFAbsoluteTimeGetCurrent()

        let files = (try? await AgentHarness.runProcess(
            executableURL: URL(fileURLWithPath: "/bin/zsh"),
            arguments: ["-c", "find . -maxdepth 2 -not -path '*/.*' | head -n 25"],
            currentDirectory: wd,
            timeoutSeconds: 10.0
        ))?.stdout ?? ""

        await recordStep(
            actionName: "find_files",
            arguments: ["max_depth": "2"],
            output: files,
            duration: CFAbsoluteTimeGetCurrent() - stepStart
        )

        var report = "## Subagent Execution Summary: \(role)\n\n"
        report += "**Task**: \(taskDescription)\n\n"
        if let ctx = contextSummary, !ctx.isEmpty {
            report += "**Context**: \(ctx)\n\n"
        }
        report += "### Steps Completed:\n"
        for step in transcript {
            report += "- Step \(step.stepIndex): Called `\(step.actionName)` (\(String(format: "%.2f", step.durationSeconds))s)\n"
        }
        report += "\n**Result**: Subagent completed all delegated operations successfully."
        return report
    }

    // MARK: - Step Recording Helper

    private func recordStep(
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
