//
//  CompletionTypes.swift
//  DynaMoE
//
//  Headless completion API: request/response/event types shared by the
//  in-app chat UI, the local OpenAI/Anthropic-compatible server, and the
//  CompletionEngine executor. Nothing in this file knows about SwiftUI.
//

import Foundation

// MARK: - Sampling Parameters (per turn)

/// Per-turn sampling parameters. Mirrors the profile sliders the chat UI
/// exposes; API requests map their own parameter names onto these fields.
internal struct TurnSampling: Sendable {
    var temperature: Float
    var topP: Float
    var minP: Float
    var topK: Int
    var repetitionPenalty: Float
    var presencePenalty: Float
    var maxTokens: Int

    init(
        temperature: Float,
        topP: Float,
        minP: Float,
        topK: Int,
        repetitionPenalty: Float,
        presencePenalty: Float = 0.0,
        maxTokens: Int
    ) {
        self.temperature = temperature
        self.topP = topP
        self.minP = minP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.presencePenalty = presencePenalty
        self.maxTokens = maxTokens
    }
}

// MARK: - Memory / KV Settings (per turn)

/// Execution-memory knobs the caller resolves from user settings, passed
/// explicitly so a server request can pin its own policy.
internal struct TurnMemorySettings: Sendable {
    var executionMode: MemoryExecutionMode
    var budgetMode: MemoryBudgetMode
    var kvPrecision: KVCachePrecision

    init(
        executionMode: MemoryExecutionMode,
        budgetMode: MemoryBudgetMode,
        kvPrecision: KVCachePrecision
    ) {
        self.executionMode = executionMode
        self.budgetMode = budgetMode
        self.kvPrecision = kvPrecision
    }

    /// Resolves the in-app memory settings the chat UI persists via @AppStorage
    /// ("dynamoe_memory_execution_mode" / "dynamoe_memory_budget_mode" /
    /// "dynamoe_kv_cache_precision"). The local server builds every turn from
    /// this so external clients run under the same budget the UI configured
    /// instead of a hardcoded default.
    nonisolated static func persisted(_ defaults: UserDefaults = .standard) -> TurnMemorySettings {
        let executionMode = MemoryExecutionMode(rawValue: defaults.string(forKey: "dynamoe_memory_execution_mode") ?? "") ?? .autoDetect
        let budgetMode = MemoryBudgetMode(rawValue: defaults.string(forKey: "dynamoe_memory_budget_mode") ?? "") ?? .balanced16GB
        let kvPrecision = KVCachePrecision(rawValue: defaults.string(forKey: "dynamoe_kv_cache_precision") ?? "") ?? .fp16
        return TurnMemorySettings(executionMode: executionMode, budgetMode: budgetMode, kvPrecision: kvPrecision)
    }
}

// MARK: - Turn Request

/// One generation turn: pre-templated, pre-tokenized prompt tokens in, streamed
/// deltas out, then a terminal result. The caller owns templating and
/// tokenization (ChatPromptFormatter + tokenizer); the engine owns the Metal
/// pipeline, KV cache interaction, sampling, guards, backfill and prefix-pin
/// recording.
internal struct CompletionTurnRequest: Sendable {
    /// Fully templated prompt token ids (or agent-spliced continuation ids).
    var promptTokenIds: [UInt32]
    /// The exact templated prompt string (kept for agent continuation splicing,
    /// prompt-dump diagnostics and API response bookkeeping).
    var formattedPrompt: String
    /// Conversation correlation id (session for chat, request-scoped key for
    /// API callers); used for prefix-cache pinning identity.
    var sessionId: UUID?
    /// Per-turn sampling.
    var sampling: TurnSampling
    /// Execution-memory resolution for this turn.
    var memory: TurnMemorySettings
    /// Whether this conversation has the agent tool loop enabled (drives the
    /// mid-stream tool-call freeze and post-turn parsing ranges).
    var agentToolsEnabled: Bool
    /// Thinking was requested in the prompt template (drives decision guards).
    var thinkingEnabled: Bool
    /// Agent step index of this turn within a multi-step tool run.
    var agentStep: Int
    /// Tools are forcibly disabled for this turn (synthesis/relay fallbacks).
    var forceSynthesis: Bool
    /// True when promptTokenIds were spliced from a prior turn's live ids
    /// (agent continuations) rather than encoded from the prompt string —
    /// consumed by the prompt-dump diagnostics and future request tracing.
    var isSplicedContinuation: Bool
    /// New in the API path: arbitrary stop sequences, matched incrementally
    /// against decoded text. Empty for chat turns (EOS/end-tag guards govern).
    var stopSequences: [String]

    init(
        promptTokenIds: [UInt32],
        formattedPrompt: String,
        sessionId: UUID?,
        sampling: TurnSampling,
        memory: TurnMemorySettings,
        agentToolsEnabled: Bool,
        thinkingEnabled: Bool,
        agentStep: Int = 0,
        forceSynthesis: Bool = false,
        isSplicedContinuation: Bool = false,
        stopSequences: [String] = []
    ) {
        self.promptTokenIds = promptTokenIds
        self.formattedPrompt = formattedPrompt
        self.sessionId = sessionId
        self.sampling = sampling
        self.memory = memory
        self.agentToolsEnabled = agentToolsEnabled
        self.thinkingEnabled = thinkingEnabled
        self.agentStep = agentStep
        self.forceSynthesis = forceSynthesis
        self.isSplicedContinuation = isSplicedContinuation
        self.stopSequences = stopSequences
    }
}

// MARK: - Streaming Frame

/// One throttled streaming snapshot, emitted at the same cadence the chat UI
/// used to receive direct state writes (16 ms / 4-token boundaries). Consumers
/// apply these to whatever surface they own (SwiftUI state or SSE frames).
internal struct CompletionFrame: Sendable {
    /// Full decoded text so far (UTF-8 repaired; trailing U+FFFD held back).
    var rawText: String
    var thinkingText: String
    var responseText: String
    var isActivelyThinking: Bool
    var tokensGenerated: Int
    var elapsedMs: Double
    var tokensPerSecond: Double
    var timeToFirstTokenSeconds: Double?
    var thinkingDurationSeconds: Double?
    /// JetSpec live stats (turn-scoped, engine-authoritative).
    var jetSpecActive: Bool
    var jetSpecMeanTau: Double
    var jetSpecDraftAccepted: Int
    var jetSpecDraftProposed: Int
    /// Working-set telemetry for the memory/status readouts.
    var residentMemoryGB: Double
    var residentExpertCount: Int
    var totalExpertCount: Int
    var cacheHitRatePercent: Double
    var prefetchEfficiencyPercent: Double
    var pagingLatencyMs: Double
    /// Prompt tokens ingested minus prefix reuse (odometer seeding input for
    /// consumers that meter tokens); nil when prefill had nothing to do.
    var promptDeltaTokens: Int?

    init(
        rawText: String,
        thinkingText: String,
        responseText: String,
        isActivelyThinking: Bool,
        tokensGenerated: Int,
        elapsedMs: Double,
        tokensPerSecond: Double,
        timeToFirstTokenSeconds: Double?,
        thinkingDurationSeconds: Double?,
        jetSpecActive: Bool,
        jetSpecMeanTau: Double,
        jetSpecDraftAccepted: Int,
        jetSpecDraftProposed: Int,
        residentMemoryGB: Double,
        residentExpertCount: Int,
        totalExpertCount: Int,
        cacheHitRatePercent: Double,
        prefetchEfficiencyPercent: Double,
        pagingLatencyMs: Double
    ) {
        self.rawText = rawText
        self.thinkingText = thinkingText
        self.responseText = responseText
        self.isActivelyThinking = isActivelyThinking
        self.tokensGenerated = tokensGenerated
        self.elapsedMs = elapsedMs
        self.tokensPerSecond = tokensPerSecond
        self.timeToFirstTokenSeconds = timeToFirstTokenSeconds
        self.thinkingDurationSeconds = thinkingDurationSeconds
        self.jetSpecActive = jetSpecActive
        self.jetSpecMeanTau = jetSpecMeanTau
        self.jetSpecDraftAccepted = jetSpecDraftAccepted
        self.jetSpecDraftProposed = jetSpecDraftProposed
        self.residentMemoryGB = residentMemoryGB
        self.residentExpertCount = residentExpertCount
        self.totalExpertCount = totalExpertCount
        self.cacheHitRatePercent = cacheHitRatePercent
        self.prefetchEfficiencyPercent = prefetchEfficiencyPercent
        self.pagingLatencyMs = pagingLatencyMs
        self.promptDeltaTokens = nil
    }
}

// MARK: - Terminal Results

internal enum TurnFinishReason: String, Sendable {
    case eos
    case endTag
    case toolFreeze
    case degenerateCycle
    case lengthCapped
    case stoppedBySequence
    /// The decode loop broke without a positive break reason (cancelled turns
    /// use the failure event instead of a result).
    case stopped
}

/// Terminal failure payload: mirrors what the UI previously wrote inline when
/// a turn died mid-flight (prefill failure, forward failure, backfill failure,
/// cancellation). Consumers translate this into their own state resets.
/// Discriminates the four failure surfaces so consumers can reproduce the
/// historical UI handling exactly (which paths guard on generation
/// ownership, which write odometer, which clear placeholders).
internal enum TurnFailureKind: String, Sendable {
    case prefillFailed
    case forwardPassFailed
    case backfillFailed
    case cancelled
}

internal struct TurnFailure: Sendable {
    var kind: TurnFailureKind
    var message: String
    var cancelled: Bool
    var tokensGenerated: Int
    /// The consumer should treat message as generation status text AND
    /// gpuComputeOutput-style error text (chat behavior); server consumers
    /// may ignore and emit their own error payload.
    var isTechnicalError: Bool
    /// Clear streaming placeholder flags (prefill status, isThinking) on the
    /// live message owned by the consumer.
    var clearStreamingPlaceholders: Bool

    init(
        kind: TurnFailureKind,
        message: String,
        cancelled: Bool,
        tokensGenerated: Int = 0,
        isTechnicalError: Bool = true,
        clearStreamingPlaceholders: Bool = true
    ) {
        self.kind = kind
        self.message = message
        self.cancelled = cancelled
        self.tokensGenerated = tokensGenerated
        self.isTechnicalError = isTechnicalError
        self.clearStreamingPlaceholders = clearStreamingPlaceholders
    }
}

/// Terminal success payload for a completed turn (natural end-of-generation;
/// cancelled turns emit `.failedTurn` instead so finalization never runs).
/// Contains everything the agent-orchestration layer (UI or API adapter) needs
/// to continue a multi-step tool run or finalize the reply.
internal struct TurnResult: @unchecked Sendable {
    var finishReason: TurnFinishReason
    var formattedPrompt: String
    var promptTokenIds: [UInt32]
    var generatedTokenIds: [UInt32]
    /// Full decoded stream text, including any thinking tags still embedded.
    var rawDecodedText: String
    var thinkingText: String
    var responseText: String
    var tokensGenerated: Int
    var elapsedMs: Double
    var tokensPerSecond: Double
    var timeToFirstTokenSeconds: Double?
    var thinkingDurationSeconds: Double?
    var jetSpecActive: Bool
    var jetSpecMeanTau: Double
    var jetSpecDraftAccepted: Int
    var jetSpecDraftProposed: Int
    var residentMemoryGB: Double
    var residentExpertCount: Int
    var totalExpertCount: Int
    var cacheHitRatePercent: Double
    var prefetchEfficiencyPercent: Double
    var pagingLatencyMs: Double
    var prefixTokensReused: Int
    /// Tool calls parsed from the emitted text (agent-enabled turns only).
    var parsedToolCalls: [ParsedToolCall]
    /// Indexes into parsedToolCalls that are actionable (non-gesture).
    var actionableCallIndices: [Int]
    /// Gesture (no-op) call indexes → skip notice text for the transcript.
    var gestureSkipNotices: [Int: String]
    /// Unparseable tool-call fragments (recovery-prompt input for the UI layer).
    var brokenFragments: [String]
    /// The configured stop sequence that ended the turn (when `finishReason`
    /// is `.stoppedBySequence`); protocol adapters surface it to clients.
    var matchedStopSequence: String?

    init(
        finishReason: TurnFinishReason,
        formattedPrompt: String,
        promptTokenIds: [UInt32],
        generatedTokenIds: [UInt32],
        rawDecodedText: String,
        thinkingText: String,
        responseText: String,
        tokensGenerated: Int,
        elapsedMs: Double,
        tokensPerSecond: Double,
        timeToFirstTokenSeconds: Double?,
        thinkingDurationSeconds: Double?,
        jetSpecActive: Bool,
        jetSpecMeanTau: Double,
        jetSpecDraftAccepted: Int,
        jetSpecDraftProposed: Int,
        residentMemoryGB: Double,
        residentExpertCount: Int,
        totalExpertCount: Int,
        cacheHitRatePercent: Double,
        prefetchEfficiencyPercent: Double,
        pagingLatencyMs: Double,
        prefixTokensReused: Int,
        parsedToolCalls: [ParsedToolCall],
        actionableCallIndices: [Int],
        gestureSkipNotices: [Int: String],
        brokenFragments: [String],
        matchedStopSequence: String? = nil
    ) {
        self.finishReason = finishReason
        self.formattedPrompt = formattedPrompt
        self.promptTokenIds = promptTokenIds
        self.generatedTokenIds = generatedTokenIds
        self.rawDecodedText = rawDecodedText
        self.thinkingText = thinkingText
        self.responseText = responseText
        self.tokensGenerated = tokensGenerated
        self.elapsedMs = elapsedMs
        self.tokensPerSecond = tokensPerSecond
        self.timeToFirstTokenSeconds = timeToFirstTokenSeconds
        self.thinkingDurationSeconds = thinkingDurationSeconds
        self.jetSpecActive = jetSpecActive
        self.jetSpecMeanTau = jetSpecMeanTau
        self.jetSpecDraftAccepted = jetSpecDraftAccepted
        self.jetSpecDraftProposed = jetSpecDraftProposed
        self.residentMemoryGB = residentMemoryGB
        self.residentExpertCount = residentExpertCount
        self.totalExpertCount = totalExpertCount
        self.cacheHitRatePercent = cacheHitRatePercent
        self.prefetchEfficiencyPercent = prefetchEfficiencyPercent
        self.pagingLatencyMs = pagingLatencyMs
        self.prefixTokensReused = prefixTokensReused
        self.parsedToolCalls = parsedToolCalls
        self.actionableCallIndices = actionableCallIndices
        self.gestureSkipNotices = gestureSkipNotices
        self.brokenFragments = brokenFragments
        self.matchedStopSequence = matchedStopSequence
    }
}

// MARK: - Events

/// The full event vocabulary of one generation turn. Exactly one terminal
/// event (`.failedTurn` or `.finished`) is emitted per launched turn, and the
/// stream closes immediately after it.
internal enum CompletionEvent: @unchecked Sendable {
    /// The engine began (post-launch); emitted before any prefill work.
    case started
    /// Resident-weight priming pass (Full-RAM mode warmup).
    case priming
    /// Prefill progress: status string plus optional speed/rss; nil status
    /// means "prefill finished — clear the progress indicator".
    case prefillProgress(status: String?, speed: Double?, rssGB: Double?)
    /// Throttled streaming snapshot (16 ms / 4-token cadence, first token
    /// always immediate). same cadence the UI previously received.
    case frame(CompletionFrame)
    /// Terminal failure (prefill/forward/backfill failure or cancellation).
    case failedTurn(TurnFailure)
    /// Terminal success; the engine has finished backfill and recorded the
    /// prefix pin. The stream closes right after this event.
    case finished(TurnResult)
}

// MARK: - Launch Errors

/// Synchronous launch failures surfaced before any compute begins (validation
/// of tensors/pipelines/scratch buffers). The UI maps these to status/output
/// error text exactly like the old inline guards; the server maps them to 4xx.
internal enum CompletionLaunchError: Error, LocalizedError, Sendable {
    case engineNotReady(String)
    /// The prompt cannot fit the KV cache this turn can provision (either
    /// outright, or with the requested max_tokens headroom). Message carries
    /// the token counts and budget numbers the caller can act on.
    case promptTooLong(String)

    /// Chat-shaped launch failures the remote caller can fix (bad request),
    /// as opposed to server-side conditions (model not loaded, engine busy).
    var isClientError: Bool {
        if case .promptTooLong = self { return true }
        return false
    }

    var errorDescription: String? {
        switch self {
        case let .engineNotReady(message):
            return message
        case let .promptTooLong(message):
            return message
        }
    }
}

// MARK: - Turn Handle

/// Cancel/identity handle for a launched turn. Cancellation is cooperative:
/// the engine checks at layer/step boundaries (same tradeoff as the old loop).
internal final class CompletionTurnHandle: @unchecked Sendable {
    let generationId: UInt64
    let task: Task<Void, Never>

    init(generationId: UInt64, task: Task<Void, Never>) {
        self.generationId = generationId
        self.task = task
    }

    func cancel() {
        task.cancel()
    }
}

// MARK: - Preparation Info

/// Synchronous facts resolved during launch, before the engine task starts —
/// needed by callers that meter tokens (the session odometer) and by the server
/// for request accounting.
internal struct TurnPrepInfo: Sendable {
    /// Prefix tokens restored from the pin (delta prefill skip count).
    var prefixTokensReused: Int
    /// Prompt token count (echo of the request).
    var promptTokenCount: Int
    /// KV precision actually in effect after model-specific overrides.
    var effectiveKVPrecision: KVCachePrecision
    /// KV slot length provisioned for this turn.
    var provisionedSequenceLength: Int

    init(
        prefixTokensReused: Int,
        promptTokenCount: Int,
        effectiveKVPrecision: KVCachePrecision,
        provisionedSequenceLength: Int
    ) {
        self.prefixTokensReused = prefixTokensReused
        self.promptTokenCount = promptTokenCount
        self.effectiveKVPrecision = effectiveKVPrecision
        self.provisionedSequenceLength = provisionedSequenceLength
    }
}

/// Launch outcome: either a live stream plus handle, or the prep facts plus
/// handle when the turn should run (success), with `stream` already started.
internal struct CompletionLaunch: Sendable {
    var prep: TurnPrepInfo
    var handle: CompletionTurnHandle
    var stream: AsyncStream<CompletionEvent>

    init(prep: TurnPrepInfo, handle: CompletionTurnHandle, stream: AsyncStream<CompletionEvent>) {
        self.prep = prep
        self.handle = handle
        self.stream = stream
    }
}
