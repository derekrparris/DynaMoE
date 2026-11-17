//
//  HeadlessAgentLoop.swift
//  DynaMoE
//
//  Backend-agnostic step loop for a subagent run. See SUBAGENTS.md §5.
//

import Foundation

nonisolated struct AgentLoopInput: Sendable {
    var role: String
    var task: String
    var contextSummary: String?
    var systemPrompt: String
    var budget: AgentBudget
    var sampling: AgentSampling
    var sessionId: UUID
    var thinking: Bool
    var stopSequences: [String]

    init(
        role: String,
        task: String,
        contextSummary: String? = nil,
        systemPrompt: String,
        budget: AgentBudget = .standard,
        sampling: AgentSampling = .default,
        sessionId: UUID,
        thinking: Bool = false,
        stopSequences: [String] = []
    ) {
        self.role = role
        self.task = task
        self.contextSummary = contextSummary
        self.systemPrompt = systemPrompt
        self.budget = budget
        self.sampling = sampling
        self.sessionId = sessionId
        self.thinking = thinking
        self.stopSequences = stopSequences
    }
}

nonisolated struct AgentLoopOutcome: Sendable {
    var status: AgentRunStatus
    var transcript: [AgentTranscriptEntry]
    var report: AgentReport?
    var usage: AgentUsage
    var error: String?
}

nonisolated final class HeadlessAgentLoop: @unchecked Sendable {
    typealias EventSink = @Sendable (AgentRunEvent) -> Void

    func run(
        input: AgentLoopInput,
        backend: any AgentModelBackend,
        context: AgentRunContext,
        emit: @escaping EventSink
    ) async -> AgentLoopOutcome {
        let start = CFAbsoluteTimeGetCurrent()
        var transcript: [AgentTranscriptEntry] = []
        var usage = AgentUsage()
        var index = 0

        @discardableResult
        func append(
            kind: AgentTranscriptEntryKind,
            step: Int,
            text: String? = nil,
            record: ToolCallRecord? = nil,
            id: UUID = UUID()
        ) -> UUID {
            transcript.append(AgentTranscriptEntry(
                id: id,
                index: index,
                stepIndex: step,
                kind: kind,
                text: text,
                toolRecord: record
            ))
            index += 1
            return id
        }

        append(kind: .task, step: 0, text: input.task)
        emit(.status(.running, detail: "Starting \(input.role)"))

        var messages: [AgentBackendMessage] = [.user(Self.initialTaskText(task: input.task, context: input.contextSummary))]
        var finalText = ""
        var completionSummary: String?
        var failure: String?
        var status: AgentRunStatus = .completed
        var finished = false

        let toolDefinitions = await context.toolDefinitions()

        for step in 0..<input.budget.maxSteps {
            if Task.isCancelled {
                status = .cancelled
                break
            }
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            if elapsed >= Double(input.budget.maxWallClockSeconds) {
                status = .failed
                failure = "The run exceeded its \(input.budget.maxWallClockSeconds)-second wall-clock budget."
                break
            }
            let remainingTokens = input.budget.maxCompletionTokens - usage.completionTokens
            if remainingTokens <= 0 {
                status = .failed
                failure = "The run exhausted its \(input.budget.maxCompletionTokens)-token completion budget."
                break
            }
            if let maxCost = input.budget.maxCostUSD, usage.costUSD >= maxCost {
                status = .failed
                failure = "The run exhausted its cost budget."
                break
            }

            let deltaTextId = UUID()
            let deltaThinkingId = UUID()
            let sink = emit
            let request = AgentBackendRequest(
                systemPrompt: input.systemPrompt,
                messages: messages,
                tools: toolDefinitions,
                sampling: input.sampling,
                maxTokens: min(input.sampling.maxTokens, remainingTokens),
                stopSequences: input.stopSequences,
                thinking: input.thinking,
                sessionId: input.sessionId,
                onDelta: { text, isThinking in
                    sink(.assistantDelta(entryId: isThinking ? deltaThinkingId : deltaTextId, text: text, isThinking: isThinking))
                }
            )

            let turn: AgentBackendTurn
            do {
                turn = try await backend.runTurn(request)
            } catch {
                if Task.isCancelled || (error as? AgentBackendError) == .cancelled {
                    status = .cancelled
                    failure = "The run was cancelled."
                } else {
                    status = .failed
                    failure = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
                    append(kind: .status, step: step, text: failure)
                }
                break
            }

            usage.backendRequests += 1
            usage.steps += 1
            usage.merge(turn.usage)
            // A step's prompt size is the run's "last ingested prompt" for the
            // context count; `merge` cannot infer it because backends report
            // only the billed sum, and the sum is not what the drawer shows.
            usage.latestPromptTokens = turn.usage.promptTokens
            usage.wallClockSeconds = CFAbsoluteTimeGetCurrent() - start

            if !turn.thinking.isEmpty {
                append(kind: .assistantThinking, step: step, text: turn.thinking, id: deltaThinkingId)
            }
            if !turn.text.isEmpty {
                append(kind: .assistantText, step: step, text: turn.text, id: deltaTextId)
                finalText = turn.text
            }
            emit(.usage(usage))

            guard !turn.calls.isEmpty else {
                finished = true
                break
            }

            messages.append(.assistant(text: turn.text, thinking: turn.thinking, calls: turn.calls))

            var finishedRun = false
            for call in turn.calls {
                if Task.isCancelled {
                    status = .cancelled
                    failure = "The run was cancelled."
                    finishedRun = true
                    break
                }
                if !context.allowedTools.contains(call.name) {
                    let error = "Tool '\(call.name)' is not in this run's allowed tools (\(context.allowedTools.sorted().joined(separator: ", ")))."
                    let record = ToolCallRecord(
                        name: call.name,
                        arguments: call.arguments.mapValues { String(describing: $0) },
                        rawArguments: call.rawArguments,
                        status: .withheld,
                        output: AgentHarness.renderToolResultForModel(AgentHarness.toolErrorJSON(tool: call.name, error: error)),
                        error: error
                    )
                    append(kind: .toolResult, step: step, record: record)
                    messages.append(.toolResult(callId: call.id, name: call.name, content: error, isError: true))
                    emit(.toolFinished(record))
                    continue
                }

                let running = ToolCallRecord(
                    name: call.name,
                    arguments: call.arguments.mapValues { String(describing: $0) },
                    rawArguments: call.rawArguments,
                    status: .running
                )
                emit(.toolStarted(running))

                let execution = await context.execute(call)
                var record = execution.record
                record.id = running.id
                usage.toolCalls += 1
                await context.noteToolOutcome(record, call: call)
                let rendered = AgentHarness.renderToolResultForModel(execution.resultJSON)
                append(kind: .toolResult, step: step, record: record)
                emit(.toolFinished(record))
                messages.append(.toolResult(
                    callId: call.id,
                    name: call.name,
                    content: rendered,
                    isError: record.status == .error || record.status == .rejected || record.status == .withheld
                ))

                if record.status.endsAgentRun(toolName: call.name, isCompleted: execution.isCompleted) {
                    completionSummary = call.arguments["summary"] as? String
                    finishedRun = true
                    break
                }
            }
            if finishedRun {
                if status != .cancelled { status = .completed }
                finished = true
                break
            }
            if let tripped = await context.trippedGuard() {
                status = .blocked
                failure = "The run was stopped by its loop guard: \(tripped)."
                append(kind: .status, step: step, text: failure)
                finished = true
                break
            }
        }

        if status == .completed && !finished {
            status = .failed
            failure = "The run exhausted its \(input.budget.maxSteps)-step budget before finishing."
        }
        if status == .completed && finalText.isEmpty && completionSummary == nil {
            status = .failed
            failure = "The backend produced no final answer."
        }

        usage.wallClockSeconds = CFAbsoluteTimeGetCurrent() - start
        let summary = Self.bounded(completionSummary ?? finalText, limit: 6000)
        let report = AgentReport(
            status: status,
            summary: summary,
            findings: [],
            changedFiles: Self.changedFiles(in: transcript),
            diff: nil,
            tests: Self.testOutcomes(in: transcript),
            risks: failure.map { [$0] } ?? [],
            openQuestions: [],
            usage: usage,
            backend: backend.descriptor
        )
        emit(.usage(usage))
        if status == .completed {
            emit(.completed(report))
        } else {
            emit(.status(status, detail: failure ?? status.displayName))
        }
        return AgentLoopOutcome(
            status: status,
            transcript: transcript,
            report: report,
            usage: usage,
            error: failure
        )
    }

    private static func initialTaskText(task: String, context: String?) -> String {
        guard let context, !context.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return task
        }
        return task + "\n\nCoordinator context:\n" + context
    }

    private static func bounded(_ text: String, limit: Int) -> String {
        AgentHarness.truncateText(AgentHarness.sanitizeText(text), limit: limit)
    }

    private static func changedFiles(in transcript: [AgentTranscriptEntry]) -> [String] {
        var seen = Set<String>()
        var files: [String] = []
        for entry in transcript {
            guard let record = entry.toolRecord else { continue }
            guard record.name == "file_write" || record.name == "file_edit" else { continue }
            guard let path = record.arguments["path"], !path.isEmpty else { continue }
            if seen.insert(path).inserted {
                files.append(path)
            }
        }
        return files
    }

    private static func testOutcomes(in transcript: [AgentTranscriptEntry]) -> [AgentTestOutcome] {
        var outcomes: [AgentTestOutcome] = []
        for entry in transcript {
            guard let record = entry.toolRecord, record.name == "shell_run" else { continue }
            guard let command = record.arguments["command"] else { continue }
            let lowered = command.lowercased()
            let looksLikeTest = lowered.contains("xcodebuild test")
                || lowered.contains("swift test")
                || lowered.contains("pytest")
                || lowered.contains("npm test")
                || lowered.contains("test")
            guard looksLikeTest else { continue }
            let output = record.output ?? record.error ?? ""
            let exitCode: Int32 = record.status == .success ? 0 : 1
            outcomes.append(AgentTestOutcome(
                command: command,
                exitCode: exitCode,
                passed: Self.count("passed", in: output),
                failed: Self.count("failed", in: output),
                summary: Self.bounded(output, limit: 600)
            ))
        }
        return outcomes
    }

    private static func count(_ word: String, in text: String) -> Int? {
        let lowerString = text.lowercased()
        guard lowerString.contains(word) else { return nil }
        let lower = lowerString as NSString
        let pattern = "([0-9]+)\\s+" + word
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let range = NSRange(location: 0, length: lower.length)
        var total = 0
        for match in regex.matches(in: lowerString, options: [], range: range) {
            let value = lower.substring(with: match.range(at: 1))
            total += Int(value) ?? 0
        }
        return total
    }
}
