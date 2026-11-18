//
//  HeadlessAgentLoop.swift
//  DynaMoE
//
//  Backend-agnostic step loop for a subagent run. See SUBAGENTS.md §5.
//

import Foundation

nonisolated enum AgentLoopError: LocalizedError, Equatable {
    case wallClockExceeded

    var errorDescription: String? {
        switch self {
        case .wallClockExceeded:
            return "The run exceeded its wall-clock budget."
        }
    }
}

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
    /// Steering queued for a run while it works (coordinator directives,
    /// operator notes from the inspector). Drained before each step so a
    /// message the run was told it received actually reaches the model; a nil
    /// drain keeps a context-free caller (and every test input) unchanged.
    var drainSteering: (@Sendable () async -> [(sender: String, content: String)])? = nil

    init(
        role: String,
        task: String,
        contextSummary: String? = nil,
        systemPrompt: String,
        budget: AgentBudget = .standard,
        sampling: AgentSampling = .default,
        sessionId: UUID,
        thinking: Bool = false,
        stopSequences: [String] = [],
        drainSteering: (@Sendable () async -> [(sender: String, content: String)])? = nil
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
        self.drainSteering = drainSteering
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
        // Spend a failed or cancelled cloud attempt incurred is recorded here;
        // it has no turn to carry it. Drained after every step so the budget
        // checks below see it.
        let usageLedger = AgentUsageLedger()
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
            if let maxCost = input.budget.maxCostUSD {
                if usage.costIsUnpriced {
                    status = .failed
                    failure = "The run has a \(Self.currency(maxCost)) cost ceiling, but no price is recorded for its model, so the ceiling cannot be enforced."
                    break
                }
                if usage.costUSD >= maxCost {
                    status = .failed
                    failure = "The run exhausted its cost budget."
                    break
                }
            }

            // Steering queued while the run worked: `send_subagent_message`
            // and the inspector's message box are reported delivered, so the
            // loop has to consume them or the sender is lied to. Each drained
            // note rides as a user turn before this step's request.
            if let drain = input.drainSteering {
                let notes = await drain()
                for note in notes {
                    append(kind: .status, step: step, text: "Steering from \(note.sender): \(note.content)")
                    messages.append(.user(Self.steeringText(sender: note.sender, content: note.content)))
                }
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
                },
                usageLedger: usageLedger
            )

            let turn: AgentBackendTurn
            do {
                // A step is bounded, not merely checked between steps: a
                // provider that stops responding would otherwise run past the
                // wall-clock budget with the scheduler ticket still held.
                let remaining = Double(input.budget.maxWallClockSeconds) - elapsed
                turn = try await Self.runTurn(backend, request, remainingSeconds: remaining)
            } catch AgentLoopError.wallClockExceeded {
                usage.merge(usageLedger.drain())
                status = .failed
                failure = "The run exceeded its \(input.budget.maxWallClockSeconds)-second wall-clock budget."
                append(kind: .status, step: step, text: failure)
                break
            } catch {
                usage.merge(usageLedger.drain())
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
            // Failed or cancelled attempts inside this same step (a retry that
            // never produced a turn) are billed too.
            usage.merge(usageLedger.drain())
            // A step's prompt size is the run's "last ingested prompt" for the
            // context count; `merge` cannot infer it because backends report
            // only the billed sum, and the sum is not what the drawer shows.
            usage.latestPromptTokens = turn.usage.promptTokens
            usage.wallClockSeconds = CFAbsoluteTimeGetCurrent() - start

            // The text lands first even when the turn then fails a budget or a
            // truncation check: what the model managed to say is part of the
            // transcript, it just is not reported as a finished answer.
            if !turn.thinking.isEmpty {
                append(kind: .assistantThinking, step: step, text: turn.thinking, id: deltaThinkingId)
            }
            if !turn.text.isEmpty {
                append(kind: .assistantText, step: step, text: turn.text, id: deltaTextId)
                finalText = turn.text
            }
            emit(.usage(usage))

            // The budget checks at the TOP of the loop only see what earlier
            // steps spent. A final turn that crosses a ceiling must not be
            // reported as a completion, and a truncated answer is not an answer.
            if let problem = Self.postTurnBudgetFailure(
                input: input,
                usage: usage,
                elapsed: CFAbsoluteTimeGetCurrent() - start
            ) {
                status = .failed
                failure = problem
                append(kind: .status, step: step, text: problem)
                break
            }
            let loweredReason = turn.finishReason.lowercased()
            let unusableReason = Self.finishReasonIsTruncated(turn.finishReason)
                || loweredReason == "toolfreeze"
                || loweredReason == "degeneratecycle"
                || loweredReason == "stopped"
            if turn.calls.isEmpty, unusableReason {
                // A token-capped turn is a fragment. A `toolFreeze` with no
                // executable call is the local parser filtering a malformed
                // one: the turn stopped on a call boundary that produced
                // nothing to run. `degenerateCycle` and `stopped` are the
                // decode loop's own "no positive break reason" endings.
                status = .failed
                switch loweredReason {
                case "toolfreeze":
                    failure = "The backend stopped on a tool-call boundary that produced no complete call, so nothing could run."
                case "degeneratecycle":
                    failure = "The backend stopped because it began repeating itself, so its text is not a considered answer."
                case "stopped":
                    failure = "The backend's decode loop stopped without reaching an end-of-turn reason, so its text is a fragment."
                default:
                    failure = "The backend stopped at its token limit (finish reason '\(turn.finishReason)') before the run finished, so its text is a fragment rather than an answer."
                }
                append(kind: .status, step: step, text: failure)
                break
            }

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
                    // A refusal is still a tool call the model made. It used to
                    // `continue` before both loop guards saw it, so a model that
                    // kept calling a tool it was never given ran until its step
                    // budget with no escalation.
                    usage.toolCalls += 1
                    await context.noteToolOutcome(record, call: call)
                    append(kind: .toolResult, step: step, record: record)
                    messages.append(.toolResult(callId: call.id, name: call.name, content: error, isError: true))
                    emit(.toolFinished(record))
                    if let tripped = await context.trippedGuard() {
                        status = .blocked
                        failure = "The run was stopped by its loop guard: \(tripped)."
                        append(kind: .status, step: step, text: failure)
                        finishedRun = true
                        break
                    }
                    continue
                }

                let running = ToolCallRecord(
                    name: call.name,
                    arguments: call.arguments.mapValues { String(describing: $0) },
                    rawArguments: call.rawArguments,
                    status: .running
                )
                emit(.toolStarted(running))

                // The wall-clock budget is a promise about the whole run, not
                // just its model turns. A shell command may take up to two
                // minutes, so a single call could outlive the budget that the
                // next top-of-loop check would then report.
                let remainingBeforeTool = Double(input.budget.maxWallClockSeconds)
                    - (CFAbsoluteTimeGetCurrent() - start)
                let raced = await Self.runTool(
                    context,
                    call,
                    remainingSeconds: remainingBeforeTool
                )
                // The deadline arm fires on CANCELLATION too (`Task.sleep`
                // returns the moment the run's task is stopped), so a Stop
                // racing the deadline otherwise reads as a wall-clock failure
                // and drops the tool's record. Cancellation is the honest
                // outcome, and a tool that managed to return before the Stop
                // keeps its record in the transcript.
                if Task.isCancelled {
                    if let raced {
                        var record = raced.record
                        record.id = running.id
                        usage.toolCalls += 1
                        await context.noteToolOutcome(record, call: call)
                        append(kind: .toolResult, step: step, record: record)
                        emit(.toolFinished(record))
                    } else {
                        append(kind: .status, step: step, text: "The run was stopped while '\(call.name)' was running.")
                    }
                    status = .cancelled
                    failure = "The run was cancelled."
                    finishedRun = true
                    break
                }
                guard let execution = raced else {
                    status = .failed
                    failure = "The run exceeded its \(input.budget.maxWallClockSeconds)-second wall-clock budget while '\(call.name)' was running."
                    append(kind: .status, step: step, text: failure)
                    finishedRun = true
                    break
                }
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
                if status != .cancelled, failure == nil { status = .completed }
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

        usage.merge(usageLedger.drain())
        usage.wallClockSeconds = CFAbsoluteTimeGetCurrent() - start
        // Derive what the run actually said. A `complete` call's summary is the
        // run's considered answer; without one, a completed run's final text is
        // the answer and a failed run's LONGEST text is the partial result it
        // managed to produce (its last text is usually a preamble). Findings and
        // open questions come from the transcript either way, so a read-only
        // run's report is not an empty shell around one paragraph.
        let derivedSummary = AgentReportBuilder.bestEffortSummary(in: transcript)
        let summaryText: String
        if let completionSummary, !completionSummary.isEmpty {
            summaryText = completionSummary
        } else if status == .completed, !finalText.isEmpty {
            summaryText = finalText
        } else {
            summaryText = derivedSummary ?? finalText
        }
        let summary = Self.bounded(summaryText, limit: 6000)
        let report = AgentReport(
            status: status,
            summary: summary,
            findings: AgentReportBuilder.findings(from: transcript),
            changedFiles: Self.changedFiles(in: transcript),
            diff: nil,
            tests: Self.testOutcomes(in: transcript),
            risks: failure.map { [$0] } ?? [],
            openQuestions: AgentReportBuilder.openQuestions(from: transcript),
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

    /// Runs one backend turn under the wall-clock deadline. The timeout task
    /// cancelling its sibling is what stops the HTTP stream or local decode;
    /// the parent run is not cancelled, so the failure is a budget result, not
    /// a Stop.
    private static func runTurn(
        _ backend: any AgentModelBackend,
        _ request: AgentBackendRequest,
        remainingSeconds: TimeInterval
    ) async throws -> AgentBackendTurn {
        try await withThrowingTaskGroup(of: AgentBackendTurn.self) { group in
            group.addTask { try await backend.runTurn(request) }
            group.addTask {
                try await Task.sleep(
                    nanoseconds: Self.deadlineNanoseconds(remainingSeconds)
                )
                throw AgentLoopError.wallClockExceeded
            }
            defer { group.cancelAll() }
            guard let first = try await group.next() else {
                throw AgentBackendError.malformedResponse("The backend produced no turn.")
            }
            return first
        }
    }

    /// A deadline converted to nanoseconds without trapping: a model-supplied
    /// budget can name a `max_wall_clock_seconds` whose nanosecond form
    /// overflows UInt64, and the conversion traps and kills the whole app.
    /// Anything past a day is not a deadline a run can reach anyway.
    private static func deadlineNanoseconds(_ seconds: TimeInterval) -> UInt64 {
        let clamped = min(max(0, seconds), 86_400)
        return UInt64(clamped * 1_000_000_000)
    }

    /// Runs one tool under whatever is left of the wall-clock budget. Nil means
    /// the deadline won; the tool's task is cancelled and the group waits for
    /// it to stop (a `shell_run` child is killed by `runProcess`'s own
    /// cancellation handler).
    private static func runTool(
        _ context: AgentRunContext,
        _ call: AgentToolCall,
        remainingSeconds: TimeInterval
    ) async -> (resultJSON: String, record: ToolCallRecord, isCompleted: Bool)? {
        let outcome = await withTaskGroup(
            of: (resultJSON: String, record: ToolCallRecord, isCompleted: Bool)?.self
        ) { group in
            group.addTask { await context.execute(call) }
            group.addTask {
                try? await Task.sleep(
                    nanoseconds: Self.deadlineNanoseconds(remainingSeconds)
                )
                return nil
            }
            defer { group.cancelAll() }
            return await group.next() ?? nil
        }
        return outcome
    }

    /// What a just-finished turn did to the run's budget, or nil. A cost of
    /// exactly the ceiling is allowed for a turn that then stopped: the ceiling
    /// is the most the run may SPEND, and the next step's pre-check would stop
    /// it from spending more.
    private static func postTurnBudgetFailure(
        input: AgentLoopInput,
        usage: AgentUsage,
        elapsed: TimeInterval
    ) -> String? {
        if elapsed >= Double(input.budget.maxWallClockSeconds) {
            return "The run exceeded its \(input.budget.maxWallClockSeconds)-second wall-clock budget."
        }
        if let maxCost = input.budget.maxCostUSD {
            if usage.costIsUnpriced {
                return "The run has a \(currency(maxCost)) cost ceiling, but no price is recorded for its model, so the ceiling cannot be enforced."
            }
            if usage.costUSD > maxCost {
                return "The run spent \(currency(usage.costUSD)), over its \(currency(maxCost)) cost budget."
            }
        }
        return nil
    }

    static func finishReasonIsTruncated(_ reason: String) -> Bool {
        let lowered = reason.lowercased()
        if lowered == "length" || lowered == "lengthcapped" { return true }
        if lowered.contains("max_tokens") || lowered.contains("max_output_tokens") { return true }
        return false
    }

    private static func initialTaskText(task: String, context: String?) -> String {
        // The coordinator's task and context are model-authored text that lands
        // in the worker's prompt. A chat-template marker in either would open
        // or close a turn and let the text speak as the system.
        let task = AgentHarness.sanitizeText(task)
        guard let context, !context.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return task
        }
        return task + "\n\nCoordinator context:\n" + AgentHarness.sanitizeText(context)
    }

    /// A steering note rendered as a user turn the model cannot mistake for
    /// its own task: the sender is named, the content is sanitised (it is
    /// model- or operator-authored text landing in a prompt) and bounded.
    static func steeringText(sender: String, content: String) -> String {
        let bounded = AgentHarness.truncateText(
            AgentHarness.sanitizeText(content),
            limit: 2_000
        )
        let who = AgentHarness.sanitizeText(sender)
        return "[\(who)] sent while this run was working: \(bounded)"
    }

    private static func bounded(_ text: String, limit: Int) -> String {
        AgentHarness.truncateText(AgentHarness.sanitizeText(text), limit: limit)
    }

    private static func currency(_ amount: Double) -> String {
        String(format: "$%.2f", amount)
    }

    private static func changedFiles(in transcript: [AgentTranscriptEntry]) -> [String] {
        var seen = Set<String>()
        var files: [String] = []
        for entry in transcript {
            guard let record = entry.toolRecord else { continue }
            guard record.name == "file_write" || record.name == "file_edit" else { continue }
            // A failed or refused write changed nothing. Listing it told the
            // operator (and the persisted report) that a file had been edited
            // when the call had been rejected or had errored out.
            guard record.status == .success else { continue }
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
            // The bare `contains("test")` matched "latest" and "contest";
            // the generic arm is word-bounded now.
            let looksLikeTest = lowered.contains("xcodebuild test")
                || lowered.contains("swift test")
                || lowered.contains("pytest")
                || lowered.contains("npm test")
                || Self.containsWord("test", in: lowered)
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

    /// Word-boundary match, so "the latest run" and "contest" do not count a
    /// shell command as a test invocation.
    private static func containsWord(_ word: String, in text: String) -> Bool {
        let pattern = "(^|[^a-z0-9_])" + word + "([^a-z0-9_]|$)"
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return false }
        let range = NSRange(location: 0, length: (text as NSString).length)
        return regex.firstMatch(in: text, options: [], range: range) != nil
    }
}
