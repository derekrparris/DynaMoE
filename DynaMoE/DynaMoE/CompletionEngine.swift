//
//  CompletionEngine.swift
//  DynaMoE
//
//  Headless single-flight generation executor, extracted from the former
//  `ContentView.startAutoregressiveGeneration` so the chat UI, the local
//  OpenAI/Anthropic-compatible server, and future clients share ONE runtime
//  path. The pipeline/KV/prefill/decode code below was moved VERBATIM from
//  ContentView.swift; the Metal kernels and the bit-31 delta-prefill contract
//  are unchanged (AGENTS.md: read PERF_FINDINGS.md before touching the
//  decode/prefill pipeline).
//
//  Per-turn contract (`launchTurn`):
//   1. Synchronous validation + scratch prep (throws historically-messaged
//      errors; the caller surfaces them as it sees fit).
//   2. Prefix reuse resolution + KV reset + grammar token + turn identity.
//   3. Detached engine task streams CompletionEvents: prefill progress,
//      throttled frames, then exactly ONE terminal `.finished` (backfill +
//      prefix pin complete) or `.failedTurn`. The stream closes after it.
//
//  Single-flight by design: KV buffers, the working set, and scratch are
//  process-wide singletons owned by whichever turn holds the slot.
//

import Foundation
import Metal

// MARK: - Thinking / Response Boundary Splitting (whitespace tolerant)

public enum TurnTextSplitter {
    /// Reasoning open/close marker tags used across model families (Qwen, DeepSeek, Ornith, etc.).
    private static let openTags = ["<think>", "<thought>", "<|thought|>", "<|channel>"]
    private static let closeTags = ["</think>", "</thought>", "</|thought|>", "<channel|>"]

    /// Gemma 4 emits reasoning inside a `<|channel>` block whose first line is the
    /// channel label (`thought`). Strip that label so the accordion shows only the
    /// model's actual reasoning text.
    private static let gemmaChannelOpenTag = "<|channel>"
    static func stripGemmaChannelLabel(_ text: String) -> String {
        let label = "thought"
        guard text.hasPrefix(label) else { return text }
        let rest = text.dropFirst(label.count)
        guard rest.isEmpty || rest.first!.isWhitespace else { return text }
        return String(rest).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Matches a bare "response" delimiter, never the ordinary English word. It must
    /// either start a line (optionally indented) or be glued to a preceding dot with
    /// a non-space following, and when line-leading it must not run into prose
    /// ("response counts"). Without the prose guard, reasoning that mentions the word
    /// ("1. Overall stats subagent: response counts, NPS…") split the thinking block
    /// mid-stream until the real ` response` arrived, briefly leaking reasoning into
    /// the response body.
    private static let responseBoundaryRegex: NSRegularExpression? = {
        try? NSRegularExpression(
            pattern: "(?:(?<=\\.)response(?![a-z@_.0-9\\s])|(?m)^[ \\t]*response(?![a-z@_.0-9])(?![ \\t]+[a-z]))",
            options: []
        )
    }()

    /// Matches a "response" marker on its own line ("\n response \n"). Delimiter emitted
    /// during agent continuations or templates without XML tags (such as Ling 3.0).
    private static let standaloneResponseRegex: NSRegularExpression? = {
        try? NSRegularExpression(pattern: "(?m)^\\s*response\\s*$", options: [])
    }()

    /// Returns the period of a verbatim token cycle at the end of `tokenIds`, or nil
    /// when the tail is not a degenerate repeat.
    ///
    /// Checks only the LAST `period * minCycles` tokens, not a fixed window: a repeat
    /// that begins partway through the recent history (e.g. a heredoc terminator
    /// repeated after a long script) must still be caught. Every accepted cycle must
    /// span at least 24 tokens, so a short verbatim run that is legitimate output (an
    /// alternating data column, repeated table cells, a list of `yes`/`no` values)
    /// never truncates a healthy turn; a real degeneration runs for hundreds. A cycle
    /// is only accepted when its decoded unit carries a letter or digit, so a run of
    /// blank lines, closing braces, or dashes (all legitimate formatting) is ignored.
    static func detectDegenerateCycle(
        tokenIds: [UInt32],
        decodeUnit: ([UInt32]) -> String
    ) -> Int? {
        guard tokenIds.count >= 8 else { return nil }
        for period in 1...8 {
            let minCycles = max((period == 1) ? 24 : 4, Int((24.0 / Double(period)).rounded(.up)))
            let needed = period * minCycles
            guard tokenIds.count >= needed else { continue }
            let window = Array(tokenIds.suffix(needed))
            var isCycle = true
            for i in period..<window.count where window[i] != window[i - period] {
                isCycle = false
                break
            }
            guard isCycle else { continue }
            let unit = Array(window[0..<period])
            let text = decodeUnit(unit)
            guard text.contains(where: { $0.isLetter || $0.isNumber }) else { continue }
            return period
        }
        return nil
    }

    /// Locates the thinking->response boundary in streamed text, or nil when absent.
    static func locateThinkingBoundary(in raw: String, promptRequestsThinking: Bool) -> NSRange? {
        let nsRaw = raw as NSString
        // 1. Explicit formal closing tags first (unambiguous delimiter emitted by models)
        for tag in Self.closeTags {
            let r = nsRaw.range(of: tag)
            if r.location != NSNotFound { return r }
        }
        // 2. Standalone response marker on its own line (for Ling 3.0 / templates without XML tags)
        if let standalone = Self.standaloneResponseRegex {
            var last: NSRange?
            standalone.enumerateMatches(in: raw, options: [], range: NSRange(location: 0, length: nsRaw.length)) { m, _, _ in
                if let m { last = m.range }
            }
            if let last { return last }
        }
        let expectsThinking = Self.openTags.contains { raw.contains($0) }
        guard expectsThinking || promptRequestsThinking else { return nil }
        // 3. Implicit boundary: a tool call emitted while the thinking block is still open means
        //    the model skipped its closing delimiter (Ling 3.0 frequently does). Treat the opening
        //    <tool_call> as the boundary so the call lands in the response half — parsed and
        //    rendered as a tool card — instead of leaking raw XML into the thinking accordion.
        //    A zero-length range keeps the tag itself in the response half.
        let toolCallRange = nsRaw.range(of: StreamingToolParser.qwenToolCallOpen)
        if toolCallRange.location != NSNotFound {
            return NSRange(location: toolCallRange.location, length: 0)
        }
        // 4. Whitespace-tolerant fallback for models that could not emit the special marker token
        if let re = Self.responseBoundaryRegex {
            var last: NSRange?
            re.enumerateMatches(in: raw, options: [], range: NSRange(location: 0, length: nsRaw.length)) { m, _, _ in
                if let m { last = m.range }
            }
            if let last { return last }
        }
        return nil
    }

    /// True when the prompt ends on an open reasoning marker, so untagged streamed
    /// output is the model's reasoning until a closing marker arrives. Covers the
    /// XML/`<thought>` dialects and Gemma 4's channel opener.
    ///
    /// The Gemma 4 agent continuation prompt (`formatGemmaToolResponseTurn`) already
    /// appends `<|channel>thought\n`, so the model does not re-emit the opener and its
    /// reasoning streams with NO open tag, only the closing <channel|> at the end.
    /// Without recognizing the opener as a thinking request, that reasoning was
    /// classified as the response body until the close arrived, then jumped into the
    /// thinking accordion (observed live right after a web_search). The first turn is
    /// unaffected: it ends at `<|turn>model\n` and the model emits its own opener.
    public static func promptRequestsThinking(_ prompt: String) -> Bool {
        let p = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let openSuffixes = ["<think>", "<thought>", "<|thought|>", "<|channel>thought", "<|channel>"]
        let closeSuffixes = ["</think>", "</thought>", "</|thought|>", "<channel|>"]
        guard openSuffixes.contains(where: { p.hasSuffix($0) }) else { return false }
        return !closeSuffixes.contains(where: { p.hasSuffix($0) })
    }

    /// Splits accumulated output into thinking and response halves using a whitespace-tolerant
    /// boundary. `thinkClose` is true when a boundary was found; `thinkOpen` when still inside a
    /// thinking block with no boundary yet.
    public static func splitThinkingAndResponse(raw: String, promptRequestsThinking: Bool) -> (think: String, resp: String, openTag: String, thinkOpen: Bool, thinkClose: Bool) {
        let nsRaw = raw as NSString
        let openTag: String = Self.openTags.first { raw.contains($0) } ?? "<think>"

        if let boundary = locateThinkingBoundary(in: raw, promptRequestsThinking: promptRequestsThinking) {
            var thinking = nsRaw.substring(to: boundary.location)
            for tag in Self.openTags {
                thinking = thinking.replacingOccurrences(of: tag, with: "")
            }
            thinking = thinking.trimmingCharacters(in: .whitespacesAndNewlines)
            if openTag == Self.gemmaChannelOpenTag { thinking = Self.stripGemmaChannelLabel(thinking) }

            let response = nsRaw.substring(from: boundary.location + boundary.length)
                .replacingOccurrences(of: "<|im_end|>", with: "")
                .replacingOccurrences(of: "<|endoftext|>", with: "")
                .replacingOccurrences(of: "<|im_start|>", with: "")
                .replacingOccurrences(of: "<|role_end|>", with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return (thinking, response, openTag, false, true)
        }

        if Self.openTags.contains(where: { raw.contains($0) }) || promptRequestsThinking {
            var thinking = raw
            for tag in Self.openTags {
                thinking = thinking.replacingOccurrences(of: tag, with: "")
            }
            thinking = thinking.trimmingCharacters(in: .whitespaces)
            if openTag == Self.gemmaChannelOpenTag { thinking = Self.stripGemmaChannelLabel(thinking) }
            return (thinking, "", openTag, true, false)
        }

        var response = raw
        for tag in Self.openTags { response = response.replacingOccurrences(of: tag, with: "") }
        for tag in Self.closeTags { response = response.replacingOccurrences(of: tag, with: "") }
        response = response
            .replacingOccurrences(of: "<|im_end|>", with: "")
            .replacingOccurrences(of: "<|endoftext|>", with: "")
            .replacingOccurrences(of: "<|im_start|>", with: "")
            .replacingOccurrences(of: "<|role_end|>", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return ("", response, openTag, false, false)
    }
}

// MARK: - Engine

public final class CompletionEngine: @unchecked Sendable {
    public static let shared = CompletionEngine()

    private let stateLock = NSLock()
    private var currentGenerationTask: Task<Void, Never>?
    private var currentGenerationId: UInt64 = 0
    private var generationCounter: UInt64 = 0

    // Loaded-model state; `configure` is called by the model load/switch path.
    private var summary: ModelSummary?
    private var modelConfig: ModelConfig?
    private var tokenizer: DynaMoeTokenizer?
    private var shardBuffers: [UInt32: MTLBuffer] = [:]
    private var activeLoadedModelPath: String?

    private init() {}

    public var isActive: Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return currentGenerationTask != nil
    }

    public var hasLoadedModel: Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return summary != nil
    }

    public var loadedModelPath: String? {
        stateLock.lock()
        defer { stateLock.unlock() }
        return activeLoadedModelPath
    }

    /// Facts the local API server needs about the loaded model (`/v1/models`,
    /// prompt templating decisions). Snapshot under the state lock.
    public struct LoadedModelInfo: Sendable {
        public var displayName: String
        public var path: String?
        public var supportsThinking: Bool
        public var weightsSizeGB: Double
    }

    public var loadedModelInfo: LoadedModelInfo? {
        stateLock.lock()
        defer { stateLock.unlock() }
        guard let path = activeLoadedModelPath, !path.isEmpty else { return nil }
        let modelName = URL(fileURLWithPath: path).lastPathComponent
        let supports = ModelConfig.supportsThinking(
            config: modelConfig,
            summary: summary,
            modelName: modelName,
            modelPath: path
        )
        return LoadedModelInfo(displayName: modelName, path: path, supportsThinking: supports, weightsSizeGB: summary?.sizeGb ?? 0)
    }

    /// Tokenizer bridge for API callers that build their own prompts
    /// (the server owns templating via ChatPromptFormatter).
    public func tokenize(_ text: String) throws -> [UInt32] {
        stateLock.lock()
        let tok = tokenizer
        stateLock.unlock()
        return try tok?.encode(text: text) ?? []
    }

    public func detokenize(_ ids: [UInt32]) -> String {
        guard !ids.isEmpty else { return "" }
        stateLock.lock()
        let tok = tokenizer
        stateLock.unlock()
        return (try? tok?.decode(ids: ids)) ?? ""
    }

    /// Config/summary snapshots under the state lock, for API callers doing
    /// their own system-prompt assembly and per-family templating.
    public var promptModelConfig: ModelConfig? {
        stateLock.lock()
        defer { stateLock.unlock() }
        return modelConfig
    }

    public var loadedSummary: ModelSummary? {
        stateLock.lock()
        defer { stateLock.unlock() }
        return summary
    }

    func launchTurn(_ request: CompletionTurnRequest) throws -> CompletionLaunch {
        stateLock.lock()
        if currentGenerationTask != nil {
            stateLock.unlock()
            throw CompletionLaunchError.engineNotReady("❌ A generation is already active; one turn may run at a time.")
        }
        guard let summary = summary else {
            stateLock.unlock()
            throw CompletionLaunchError.engineNotReady("❌ Metal or Tokenizer not ready for text generation. Please load model and tokenizer in Settings.")
        }
        guard modelConfig != nil, let tokenizer = tokenizer else {
            stateLock.unlock()
            throw CompletionLaunchError.engineNotReady("❌ Metal or Tokenizer not ready for text generation. Please load model and tokenizer in Settings.")
        }
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let defaultLibrary = device.makeDefaultLibrary() else {
            stateLock.unlock()
            throw CompletionLaunchError.engineNotReady("❌ Metal or Tokenizer not ready for text generation. Please load model and tokenizer in Settings.")
        }
        stateLock.unlock()

        guard !request.promptTokenIds.isEmpty else {
            throw CompletionLaunchError.engineNotReady("⚠️ Prompt tokenization produced 0 tokens.")
        }

        // Local shadow: the detached turn closure captures these by value (struct
        // semantics of the original ContentView context), never `self` directly.
        let modelConfig = self.modelConfig

        let shardBuffers = self.shardBuffers
        let sampling = request.sampling
        let memory = request.memory
        let promptTokenIds = request.promptTokenIds
        let formattedPrompt = request.formattedPrompt
        let sessionId = request.sessionId
        let agentStep = request.agentStep
        let thinkingEnabled = request.thinkingEnabled
        let promptTokens: [UInt32]? = request.isSplicedContinuation ? request.promptTokenIds : nil
        let stopSequences = request.stopSequences.filter { !$0.isEmpty }

        // User-tunable engine settings resolved from the same `dynamoe_*`
        // defaults the chat UI's @AppStorage mirrors read (defaults preserved).
        let userDefaults = UserDefaults.standard
        let speculativePrefetchEnabled = (userDefaults.object(forKey: "dynamoe_speculative_prefetch_enabled") == nil) ? true : userDefaults.bool(forKey: "dynamoe_speculative_prefetch_enabled")
        let prefetchLookaheadDepth = (userDefaults.object(forKey: "dynamoe_prefetch_lookahead_depth") == nil) ? 1 : userDefaults.integer(forKey: "dynamoe_prefetch_lookahead_depth")
        let jetSpecEnabled = (userDefaults.object(forKey: "dynamoe_jetspec_enabled") == nil) ? false : userDefaults.bool(forKey: "dynamoe_jetspec_enabled")
        let jetSpecMaxDepth = (userDefaults.object(forKey: "dynamoe_jetspec_depth") == nil) ? 3 : userDefaults.integer(forKey: "dynamoe_jetspec_depth")
        let jetSpecBranchingFactor = (userDefaults.object(forKey: "dynamoe_jetspec_branching") == nil) ? 2 : userDefaults.integer(forKey: "dynamoe_jetspec_branching")
        let jetSpecMaxExpertCap = (userDefaults.object(forKey: "dynamoe_jetspec_expert_cap") == nil) ? 8 : userDefaults.integer(forKey: "dynamoe_jetspec_expert_cap")

        // Legacy family probe (hyperparameter fallbacks below consume it).
        let isNanbeige = (modelConfig?.modelType?.lowercased().contains("nanbeige") == true) ||
                         (modelConfig?.architectures?.contains(where: { $0.lowercased().contains("nanbeige") }) == true) ||
                         (summary.layerCount == 22 && summary.maxExpertId == 0) ||
                         (summary.tensors.contains(where: { $0.name.contains("dense_gate_up_proj") })) ||
                         (summary.tensors.contains(where: { $0.name.hasPrefix("model.layers.0.mlp.gate_proj") }) && summary.layerCount == 22)

        // Find Embedding Weight & Affine Scales/Biases
        guard let embedWeight = summary.tensors.first(where: {
            !$0.name.contains("visual") && !$0.name.contains("mtp") &&
            ($0.name.contains("embed_tokens") || $0.name.hasSuffix("embed.weight") || $0.name.hasSuffix("embedding.weight") || $0.name.contains("wte") || $0.name.contains("word_embeddings") || $0.category == "Embedding") &&
            !$0.name.contains("scale") && !$0.name.contains("scales") &&
            !$0.name.contains("bias") && !$0.name.contains("biases")
        }), let embedShardBuffer = shardBuffers[embedWeight.shardIndex] else {
            let err = "❌ Embedding weight tensor not found."
            throw CompletionLaunchError.engineNotReady(err)
        }

        let embedScale = summary.tensors.first(where: {
            !$0.name.contains("visual") && !$0.name.contains("mtp") &&
            ($0.name.contains("embed_tokens") || $0.name.hasSuffix("embed.weight") || $0.name.hasSuffix("embedding.weight") || $0.name.contains("wte") || $0.name.contains("word_embeddings") || $0.category == "Embedding") &&
            ($0.name.contains("scale") || $0.name.contains("scales"))
        })
        let embedBias = summary.tensors.first(where: {
            !$0.name.contains("visual") && !$0.name.contains("mtp") &&
            ($0.name.contains("embed_tokens") || $0.name.hasSuffix("embed.weight") || $0.name.hasSuffix("embedding.weight") || $0.name.contains("wte") || $0.name.contains("word_embeddings") || $0.category == "Embedding") &&
            ($0.name.contains("bias") || $0.name.contains("biases"))
        })

        // Find Final RMSNorm or Hyper-Connection Mixer
        let finalHcNormWeight = summary.tensors.first(where: {
            $0.name == "model.language_model.hyper_connection_mixer.hc_norm.weight" ||
            $0.name == "language_model.hyper_connection_mixer.hc_norm.weight"
        })
        let finalHcDownWeight = summary.tensors.first(where: {
            $0.name == "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight" ||
            $0.name == "language_model.hyper_connection_mixer.input_mix_weight_down.weight"
        })
        let finalHcUpWeight = summary.tensors.first(where: {
            $0.name == "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight" ||
            $0.name == "language_model.hyper_connection_mixer.input_mix_weight_up.weight"
        })

        let normTensor = summary.tensors.first(where: {
            $0.name == "model.language_model.norm.weight" ||
            $0.name == "language_model.norm.weight" ||
            $0.name == "model.norm.weight" ||
            ($0.name.hasSuffix(".norm.weight") && !$0.name.contains("layers.") && !$0.name.contains("mixer"))
        })

        let hasFinalHc = (finalHcNormWeight != nil && finalHcDownWeight != nil && finalHcUpWeight != nil)
        guard hasFinalHc || (normTensor != nil && shardBuffers[normTensor!.shardIndex] != nil) else {
            let err = "❌ Final RMSNorm weight / Hyper-Connection mixer not found."
            throw CompletionLaunchError.engineNotReady(err)
        }
        let normShardBuffer = normTensor != nil ? shardBuffers[normTensor!.shardIndex] : nil

        // Find LM Head Weight & Affine Scales/Biases
        let lmHeadTensorCandidate = summary.tensors.first(where: {
            ($0.name == "lm_head.weight" ||
             $0.name == "language_model.lm_head.weight" ||
             $0.name == "model.lm_head.weight" ||
             $0.name == "lm_head") &&
            !$0.name.contains("scale") && !$0.name.contains("scales") &&
            !$0.name.contains("bias") && !$0.name.contains("biases")
        }) ?? embedWeight

        guard let lmHeadShardBuffer = shardBuffers[lmHeadTensorCandidate.shardIndex] else {
            let err = "❌ LM Head weight not found."
            throw CompletionLaunchError.engineNotReady(err)
        }
        let lmHeadTensor = lmHeadTensorCandidate

        let lmHeadScale = summary.tensors.first(where: {
            $0.name.contains("lm_head") && ($0.name.contains("scale") || $0.name.contains("scales"))
        }) ?? (lmHeadTensor == embedWeight ? embedScale : nil)
        let lmHeadBias = summary.tensors.first(where: {
            $0.name.contains("lm_head") && ($0.name.contains("bias") || $0.name.contains("biases"))
        }) ?? (lmHeadTensor == embedWeight ? embedBias : nil)

        // Compile Pipelines with correct kernel names
        guard let embedBF16Function = defaultLibrary.makeFunction(name: "lookup_embeddings_bf16"),
              let routerFunction = defaultLibrary.makeFunction(name: "moe_router_topk_bf16"),
              let rmsnormFunction = defaultLibrary.makeFunction(name: "rmsnorm_bf16"),
              let gemvBF16Function = defaultLibrary.makeFunction(name: "bf16_gemv"),
              let addFunction = defaultLibrary.makeFunction(name: "vector_add_f32"),
              let clearFunction = defaultLibrary.makeFunction(name: "clear_vector_f32") else {
            let err = "❌ Failed to load core Metal compute shaders."
            throw CompletionLaunchError.engineNotReady(err)
        }

        let embedPipeline: MTLComputePipelineState
        let embedQ4Pipeline: MTLComputePipelineState?
        let embedQ8Pipeline: MTLComputePipelineState?
        let embedMXFP8Pipeline: MTLComputePipelineState?
        let routerPipeline: MTLComputePipelineState
        let routerQ4Pipeline: MTLComputePipelineState?
        let routerQ8Pipeline: MTLComputePipelineState?
        let rmsnormPipeline: MTLComputePipelineState
        let rmsnormF16Pipeline: MTLComputePipelineState?
        let rmsnormOffsetPipeline: MTLComputePipelineState?
        let rmsnormOffsetF16Pipeline: MTLComputePipelineState?
        let gemvBF16Pipeline: MTLComputePipelineState
        let bf16GemvSimdPipeline: MTLComputePipelineState?
        let bf16GemvBatchedPipeline: MTLComputePipelineState?
        let fp8GemvBatchedPipeline: MTLComputePipelineState?
        let fp8GemvPipeline: MTLComputePipelineState?
        let mxfp8GemvPipeline: MTLComputePipelineState?
        let mxfp8GemvSimdPipeline: MTLComputePipelineState?
        let q4GemvPipeline: MTLComputePipelineState?
        let q8GemvPipeline: MTLComputePipelineState?
        let addPipeline: MTLComputePipelineState
        let clearPipeline: MTLComputePipelineState
        let fp8GateUpPipeline: MTLComputePipelineState?
        let fp8DownPipeline: MTLComputePipelineState?
        let fp8GateUpSimdPipeline: MTLComputePipelineState?
        let fp8DownSimdPipeline: MTLComputePipelineState?
        let fp8GateUpFusedPipeline: MTLComputePipelineState?
        let fp8DownFusedPipeline: MTLComputePipelineState?
        let mxfp8GateUpPipeline: MTLComputePipelineState?
        let mxfp8DownPipeline: MTLComputePipelineState?
        let mxfp8GateUpSimdPipeline: MTLComputePipelineState?
        let mxfp8DownSimdPipeline: MTLComputePipelineState?
        let bf16GateUpPipeline: MTLComputePipelineState?
        let bf16DownPipeline: MTLComputePipelineState?
        let bf16GateUpSimdPipeline: MTLComputePipelineState?
        let bf16DownSimdPipeline: MTLComputePipelineState?
        let q4GateUpPipeline: MTLComputePipelineState?
        let q4DownPipeline: MTLComputePipelineState?
        let q8GateUpPipeline: MTLComputePipelineState?
        let q8DownPipeline: MTLComputePipelineState?
        let q4GateQ8UpPipeline: MTLComputePipelineState?
        let q8GateQ4UpPipeline: MTLComputePipelineState?
        let headRmsnormPipeline: MTLComputePipelineState?
        let headRmsnormF16Pipeline: MTLComputePipelineState?
        let headRmsnormOffsetPipeline: MTLComputePipelineState?
        let headRmsnormOffsetF16Pipeline: MTLComputePipelineState?
        let ropePipeline: MTLComputePipelineState?
        let ropeProportionalPipeline: MTLComputePipelineState?
        let storeKvCachePipeline: MTLComputePipelineState?
        let storeKvCacheF16Pipeline: MTLComputePipelineState?
        let storeKvCacheFP8Pipeline: MTLComputePipelineState?
        let gqaDecodePipeline: MTLComputePipelineState?
        let gqaDecodeF16Pipeline: MTLComputePipelineState?
        let gqaDecodeFP8Pipeline: MTLComputePipelineState?
        let gqaStandardPipeline: MTLComputePipelineState?
        let gqaStandardF16Pipeline: MTLComputePipelineState?
        let gqaStandardFP8Pipeline: MTLComputePipelineState?
        let gqaHeadGatePipeline: MTLComputePipelineState?
        let gqaHeadGateF16Pipeline: MTLComputePipelineState?
        let gqaHeadGateF16ChunkedPipeline: MTLComputePipelineState?
        let gqaHeadGateF16ChunkedCombinePipeline: MTLComputePipelineState?
        let gqaFusedF16ChunkedPipeline: MTLComputePipelineState?
        let gqaFusedF16ChunkedCombinePipeline: MTLComputePipelineState?
        let gqaFusedF16ChunkedRowsPipeline: MTLComputePipelineState?
        let gqaFusedF16ChunkedRowsCombinePipeline: MTLComputePipelineState?
        let gqaHeadGateFP8Pipeline: MTLComputePipelineState?
        let gqaHeadGateFP8ChunkedPipeline: MTLComputePipelineState?
        let gqaFusedFP8ChunkedPipeline: MTLComputePipelineState?
        let bf16GeluGateUpPipeline: MTLComputePipelineState?
        let bf16GeluGateUpSimdPipeline: MTLComputePipelineState?
        let bf16GeluGateUpBatchedPipeline: MTLComputePipelineState?
        let causalConv1dPipeline: MTLComputePipelineState?
        let l2NormQkPipeline: MTLComputePipelineState?
        let linearAttnStepPipeline: MTLComputePipelineState?
        let linearAttnStepSigmoidPipeline: MTLComputePipelineState?
        let sharedGatePipeline: MTLComputePipelineState?
        let sharedGateQ4Pipeline: MTLComputePipelineState?
        let sharedGateQ8Pipeline: MTLComputePipelineState?
        let router512Pipeline: MTLComputePipelineState?
        let router512Q4Pipeline: MTLComputePipelineState?
        let gdnLinearAttnStepPipeline: MTLComputePipelineState?
        let gdnLinearAttnStepSigmoidPipeline: MTLComputePipelineState?
        let qsaMqaIndexerPipeline: MTLComputePipelineState?
        let gatedResidualBlendPipeline: MTLComputePipelineState?
        let fuseNgramPlePipeline: MTLComputePipelineState?
        let fusedInit4StreamsPipeline: MTLComputePipelineState?
        let extractStream0Pipeline: MTLComputePipelineState?
        let hcNormPipeline: MTLComputePipelineState?
        let hcDownProjPipeline: MTLComputePipelineState?
        let hcUpBlendPipeline: MTLComputePipelineState?
        let hcInjectScalePipeline: MTLComputePipelineState?
        let hcInjectPipeline: MTLComputePipelineState?
        let causalConv1dSeqPipeline: MTLComputePipelineState?
        let l2NormQkSeqPipeline: MTLComputePipelineState?
        let gdnLinearAttnSeqPipeline: MTLComputePipelineState?
        let gdnLinearAttnSeqSigmoidPipeline: MTLComputePipelineState?
        let fp8GemvSimdPipeline: MTLComputePipelineState?
        let fp8GateUpBatchedPipeline: MTLComputePipelineState?
        let fp8DownBatchedPipeline: MTLComputePipelineState?
        let fp8BlockGateUpPipeline: MTLComputePipelineState?
        let fp8BlockDownPipeline: MTLComputePipelineState?
        let fp8BlockGateUpSimdPipeline: MTLComputePipelineState?
        let fp8BlockDownSimdPipeline: MTLComputePipelineState?
        let fp8BlockGemvPipeline: MTLComputePipelineState?
        let fp8BlockGemvSimdPipeline: MTLComputePipelineState?
        let fp8BlockGateUpBatchedPipeline: MTLComputePipelineState?
        let fp8BlockDownBatchedPipeline: MTLComputePipelineState?
        let bf16GateUpBatchedPipeline: MTLComputePipelineState?
        let bf16DownBatchedPipeline: MTLComputePipelineState?
        let kdaLinearAttnStepPipeline: MTLComputePipelineState?
        let ropeInterleavedPipeline: MTLComputePipelineState?
        let routerBailingPipeline: MTLComputePipelineState?
        let storeMlaKvCachePipeline: MTLComputePipelineState?
        let storeMlaKvCacheF16Pipeline: MTLComputePipelineState?
        let mlaAttentionDecodePipeline: MTLComputePipelineState?
        let mlaAttentionDecodeF16Pipeline: MTLComputePipelineState?
        let rmsnormNoScalePipeline: MTLComputePipelineState?
        let gqaGemmaPipeline: MTLComputePipelineState?
        let gqaGemmaF16Pipeline: MTLComputePipelineState?
        let gqaGemmaFP8Pipeline: MTLComputePipelineState?
        let storeKvGemmaPipeline: MTLComputePipelineState?
        let storeKvGemmaF16Pipeline: MTLComputePipelineState?
        let storeKvGemmaFP8Pipeline: MTLComputePipelineState?
        let gqaGemmaTreeVerifyPipeline: MTLComputePipelineState?
        let gqaGemmaTreeVerifyF16Pipeline: MTLComputePipelineState?
        let mulByBf16VectorPipeline: MTLComputePipelineState?
        let scaleVectorInplacePipeline: MTLComputePipelineState?
        let perExpertScalePipeline: MTLComputePipelineState?

        do {
            embedPipeline = try device.makeComputePipelineState(function: embedBF16Function)
            routerPipeline = try device.makeComputePipelineState(function: routerFunction)
            rmsnormPipeline = try device.makeComputePipelineState(function: rmsnormFunction)

            if let rmsF16Func = defaultLibrary.makeFunction(name: "rmsnorm_f16") {
                rmsnormF16Pipeline = try device.makeComputePipelineState(function: rmsF16Func)
            } else { rmsnormF16Pipeline = nil }

            if let rmsOffsetFunc = defaultLibrary.makeFunction(name: "rmsnorm_offset_bf16") {
                rmsnormOffsetPipeline = try device.makeComputePipelineState(function: rmsOffsetFunc)
            } else { rmsnormOffsetPipeline = nil }

            if let rmsOffsetF16Func = defaultLibrary.makeFunction(name: "rmsnorm_offset_f16") {
                rmsnormOffsetF16Pipeline = try device.makeComputePipelineState(function: rmsOffsetF16Func)
            } else { rmsnormOffsetF16Pipeline = nil }

            if let gqaHeadGateFunc = defaultLibrary.makeFunction(name: "gqa_attention_decode_headgate") {
                gqaHeadGatePipeline = try device.makeComputePipelineState(function: gqaHeadGateFunc)
            } else { gqaHeadGatePipeline = nil }

            if let gqaHeadGateF16Func = defaultLibrary.makeFunction(name: "gqa_attention_decode_headgate_f16") {
                gqaHeadGateF16Pipeline = try device.makeComputePipelineState(function: gqaHeadGateF16Func)
            } else { gqaHeadGateF16Pipeline = nil }

            if let gqaHeadGateF16ChunkedFunc = defaultLibrary.makeFunction(name: "gqa_attention_decode_headgate_f16_chunked") {
                gqaHeadGateF16ChunkedPipeline = try device.makeComputePipelineState(function: gqaHeadGateF16ChunkedFunc)
            } else { gqaHeadGateF16ChunkedPipeline = nil }

            if let gqaHeadGateF16ChunkedCombineFunc = defaultLibrary.makeFunction(name: "gqa_attention_decode_headgate_f16_chunked_combine") {
                gqaHeadGateF16ChunkedCombinePipeline = try device.makeComputePipelineState(function: gqaHeadGateF16ChunkedCombineFunc)
            } else { gqaHeadGateF16ChunkedCombinePipeline = nil }

            if let gqaFusedF16ChunkedFunc = defaultLibrary.makeFunction(name: "gqa_attention_decode_fused_f16_chunked") {
                gqaFusedF16ChunkedPipeline = try device.makeComputePipelineState(function: gqaFusedF16ChunkedFunc)
            } else { gqaFusedF16ChunkedPipeline = nil }

            if let gqaFusedF16ChunkedCombineFunc = defaultLibrary.makeFunction(name: "gqa_attention_decode_fused_f16_chunked_combine") {
                gqaFusedF16ChunkedCombinePipeline = try device.makeComputePipelineState(function: gqaFusedF16ChunkedCombineFunc)
            } else { gqaFusedF16ChunkedCombinePipeline = nil }

            if let gqaFusedF16ChunkedRowsFunc = defaultLibrary.makeFunction(name: "gqa_attention_decode_fused_f16_chunked_rows") {
                gqaFusedF16ChunkedRowsPipeline = try device.makeComputePipelineState(function: gqaFusedF16ChunkedRowsFunc)
            } else { gqaFusedF16ChunkedRowsPipeline = nil }

            if let gqaFusedF16ChunkedRowsCombineFunc = defaultLibrary.makeFunction(name: "gqa_attention_decode_fused_f16_chunked_rows_combine") {
                gqaFusedF16ChunkedRowsCombinePipeline = try device.makeComputePipelineState(function: gqaFusedF16ChunkedRowsCombineFunc)
            } else { gqaFusedF16ChunkedRowsCombinePipeline = nil }

            if let gqaHeadGateFP8Func = defaultLibrary.makeFunction(name: "gqa_attention_decode_headgate_fp8") {
                gqaHeadGateFP8Pipeline = try device.makeComputePipelineState(function: gqaHeadGateFP8Func)
            } else { gqaHeadGateFP8Pipeline = nil }

            // Flash-decoding split-K scans for FP8 KV. The combine passes are shared
            // with the F16 chunked family — they only touch FP32 partials plus the
            // gate vectors, so they are KV-precision-agnostic.
            if let gqaHeadGateFP8ChunkedFunc = defaultLibrary.makeFunction(name: "gqa_attention_decode_headgate_fp8_chunked") {
                gqaHeadGateFP8ChunkedPipeline = try device.makeComputePipelineState(function: gqaHeadGateFP8ChunkedFunc)
            } else { gqaHeadGateFP8ChunkedPipeline = nil }

            if let gqaFusedFP8ChunkedFunc = defaultLibrary.makeFunction(name: "gqa_attention_decode_fused_fp8_chunked") {
                gqaFusedFP8ChunkedPipeline = try device.makeComputePipelineState(function: gqaFusedFP8ChunkedFunc)
            } else { gqaFusedFP8ChunkedPipeline = nil }

            if let geluGateUpFunc = defaultLibrary.makeFunction(name: "bf16_gelu_gate_up") {
                bf16GeluGateUpPipeline = try device.makeComputePipelineState(function: geluGateUpFunc)
            } else { bf16GeluGateUpPipeline = nil }

            if let geluGateUpSimdFunc = defaultLibrary.makeFunction(name: "bf16_gelu_gate_up_simd") {
                bf16GeluGateUpSimdPipeline = try device.makeComputePipelineState(function: geluGateUpSimdFunc)
            } else { bf16GeluGateUpSimdPipeline = nil }

            if let geluGateUpBatchedFunc = defaultLibrary.makeFunction(name: "bf16_gelu_gate_up_batched") {
                bf16GeluGateUpBatchedPipeline = try device.makeComputePipelineState(function: geluGateUpBatchedFunc)
            } else { bf16GeluGateUpBatchedPipeline = nil }
            gemvBF16Pipeline = try device.makeComputePipelineState(function: gemvBF16Function)
            addPipeline = try device.makeComputePipelineState(function: addFunction)
            clearPipeline = try device.makeComputePipelineState(function: clearFunction)

            if let rQ4Func = defaultLibrary.makeFunction(name: "moe_router_topk_q4") {
                routerQ4Pipeline = try device.makeComputePipelineState(function: rQ4Func)
            } else { routerQ4Pipeline = nil }

            if let rQ8Func = defaultLibrary.makeFunction(name: "moe_router_topk_q8") {
                routerQ8Pipeline = try device.makeComputePipelineState(function: rQ8Func)
            } else { routerQ8Pipeline = nil }

            if let embedQ4Func = defaultLibrary.makeFunction(name: "lookup_embeddings_q4") {
                embedQ4Pipeline = try device.makeComputePipelineState(function: embedQ4Func)
            } else { embedQ4Pipeline = nil }

            if let embedQ8Func = defaultLibrary.makeFunction(name: "lookup_embeddings_q8") {
                embedQ8Pipeline = try device.makeComputePipelineState(function: embedQ8Func)
            } else { embedQ8Pipeline = nil }

            if let embedMXFP8Func = defaultLibrary.makeFunction(name: "lookup_embeddings_mxfp8") {
                embedMXFP8Pipeline = try device.makeComputePipelineState(function: embedMXFP8Func)
            } else { embedMXFP8Pipeline = nil }

            if let bGemvSimd = defaultLibrary.makeFunction(name: "bf16_gemv_simd") {
                bf16GemvSimdPipeline = try device.makeComputePipelineState(function: bGemvSimd)
            } else { bf16GemvSimdPipeline = nil }

            if let bGemvBatched = defaultLibrary.makeFunction(name: "bf16_gemv_batched") {
                bf16GemvBatchedPipeline = try device.makeComputePipelineState(function: bGemvBatched)
                print("⚡ [GEMV] bf16_gemv_batched pipeline ready — prefill uses weight-stationary reads")
            } else {
                bf16GemvBatchedPipeline = nil
                print("⚠️ [GEMV] bf16_gemv_batched unavailable — prefill GEMVs fall back to per-token weight reads")
            }

            if let fGemvBatched = defaultLibrary.makeFunction(name: "fp8_gemv_batched") {
                fp8GemvBatchedPipeline = try device.makeComputePipelineState(function: fGemvBatched)
                print("⚡ [GEMV] fp8_gemv_batched pipeline ready — attention prefill uses weight-stationary reads")
            } else {
                fp8GemvBatchedPipeline = nil
                print("⚠️ [GEMV] fp8_gemv_batched unavailable — attention prefill GEMVs fall back to per-token weight reads")
            }

            if let fp8GemvFunc = defaultLibrary.makeFunction(name: "fp8_gemv") {
                fp8GemvPipeline = try device.makeComputePipelineState(function: fp8GemvFunc)
            } else { fp8GemvPipeline = nil }

            if let mxfp8GemvFunc = defaultLibrary.makeFunction(name: "mxfp8_gemv") {
                mxfp8GemvPipeline = try device.makeComputePipelineState(function: mxfp8GemvFunc)
            } else { mxfp8GemvPipeline = nil }

            if let mxfp8GemvSimdFunc = defaultLibrary.makeFunction(name: "mxfp8_gemv_simd") {
                mxfp8GemvSimdPipeline = try device.makeComputePipelineState(function: mxfp8GemvSimdFunc)
            } else { mxfp8GemvSimdPipeline = nil }

            if let q4GemvFunc = defaultLibrary.makeFunction(name: "q4_gemv") {
                q4GemvPipeline = try device.makeComputePipelineState(function: q4GemvFunc)
            } else { q4GemvPipeline = nil }

            if let q8GemvFunc = defaultLibrary.makeFunction(name: "q8_gemv") {
                q8GemvPipeline = try device.makeComputePipelineState(function: q8GemvFunc)
            } else { q8GemvPipeline = nil }

            if let fp8GateUpFunc = defaultLibrary.makeFunction(name: "fp8_swiglu_gate_up") {
                fp8GateUpPipeline = try device.makeComputePipelineState(function: fp8GateUpFunc)
            } else { fp8GateUpPipeline = nil }

            if let fp8DownFunc = defaultLibrary.makeFunction(name: "fp8_down_proj_accumulate") {
                fp8DownPipeline = try device.makeComputePipelineState(function: fp8DownFunc)
            } else { fp8DownPipeline = nil }

            if let fp8GateUpSimd = defaultLibrary.makeFunction(name: "fp8_swiglu_gate_up_simd") {
                fp8GateUpSimdPipeline = try device.makeComputePipelineState(function: fp8GateUpSimd)
            } else { fp8GateUpSimdPipeline = nil }

            if let fp8DownSimd = defaultLibrary.makeFunction(name: "fp8_down_proj_accumulate_simd") {
                fp8DownSimdPipeline = try device.makeComputePipelineState(function: fp8DownSimd)
            } else { fp8DownSimdPipeline = nil }

            if let fp8GateUpFused = defaultLibrary.makeFunction(name: "fp8_moe_gate_up_fused") {
                fp8GateUpFusedPipeline = try device.makeComputePipelineState(function: fp8GateUpFused)
            } else { fp8GateUpFusedPipeline = nil }

            if let fp8DownFused = defaultLibrary.makeFunction(name: "fp8_moe_down_fused") {
                fp8DownFusedPipeline = try device.makeComputePipelineState(function: fp8DownFused)
            } else { fp8DownFusedPipeline = nil }

            if let fp8BlockGateFunc = defaultLibrary.makeFunction(name: "fp8_block_swiglu_gate_up") {
                fp8BlockGateUpPipeline = try device.makeComputePipelineState(function: fp8BlockGateFunc)
            } else { fp8BlockGateUpPipeline = nil }

            if let fp8BlockDownFunc = defaultLibrary.makeFunction(name: "fp8_block_down_proj_accumulate") {
                fp8BlockDownPipeline = try device.makeComputePipelineState(function: fp8BlockDownFunc)
            } else { fp8BlockDownPipeline = nil }

            if let fp8BlockGateSimd = defaultLibrary.makeFunction(name: "fp8_block_swiglu_gate_up_simd") {
                fp8BlockGateUpSimdPipeline = try device.makeComputePipelineState(function: fp8BlockGateSimd)
            } else { fp8BlockGateUpSimdPipeline = nil }

            if let fp8BlockDownSimd = defaultLibrary.makeFunction(name: "fp8_block_down_proj_accumulate_simd") {
                fp8BlockDownSimdPipeline = try device.makeComputePipelineState(function: fp8BlockDownSimd)
            } else { fp8BlockDownSimdPipeline = nil }

            if let fp8BlockGemvFunc = defaultLibrary.makeFunction(name: "fp8_block_gemv") {
                fp8BlockGemvPipeline = try device.makeComputePipelineState(function: fp8BlockGemvFunc)
            } else { fp8BlockGemvPipeline = nil }

            if let fp8BlockGemvSimd = defaultLibrary.makeFunction(name: "fp8_block_gemv_simd") {
                fp8BlockGemvSimdPipeline = try device.makeComputePipelineState(function: fp8BlockGemvSimd)
            } else { fp8BlockGemvSimdPipeline = nil }

            if let mxfp8GateFunc = defaultLibrary.makeFunction(name: "mxfp8_swiglu_gate_up") {
                mxfp8GateUpPipeline = try device.makeComputePipelineState(function: mxfp8GateFunc)
            } else { mxfp8GateUpPipeline = nil }

            if let mxfp8DownFunc = defaultLibrary.makeFunction(name: "mxfp8_down_proj_accumulate") {
                mxfp8DownPipeline = try device.makeComputePipelineState(function: mxfp8DownFunc)
            } else { mxfp8DownPipeline = nil }

            if let mxfp8GateSimdFunc = defaultLibrary.makeFunction(name: "mxfp8_swiglu_gate_up_simd") {
                mxfp8GateUpSimdPipeline = try device.makeComputePipelineState(function: mxfp8GateSimdFunc)
            } else { mxfp8GateUpSimdPipeline = nil }

            if let mxfp8DownSimdFunc = defaultLibrary.makeFunction(name: "mxfp8_down_proj_accumulate_simd") {
                mxfp8DownSimdPipeline = try device.makeComputePipelineState(function: mxfp8DownSimdFunc)
            } else { mxfp8DownSimdPipeline = nil }

            if let bGateUp = defaultLibrary.makeFunction(name: "bf16_swiglu_gate_up") {
                bf16GateUpPipeline = try device.makeComputePipelineState(function: bGateUp)
            } else { bf16GateUpPipeline = nil }

            if let bDown = defaultLibrary.makeFunction(name: "bf16_down_proj_accumulate") {
                bf16DownPipeline = try device.makeComputePipelineState(function: bDown)
            } else { bf16DownPipeline = nil }

            if let bGateUpSimd = defaultLibrary.makeFunction(name: "bf16_swiglu_gate_up_simd") {
                bf16GateUpSimdPipeline = try device.makeComputePipelineState(function: bGateUpSimd)
            } else { bf16GateUpSimdPipeline = nil }

            if let bDownSimd = defaultLibrary.makeFunction(name: "bf16_down_proj_accumulate_simd") {
                bf16DownSimdPipeline = try device.makeComputePipelineState(function: bDownSimd)
            } else { bf16DownSimdPipeline = nil }

            if let q4GateUpFunc = defaultLibrary.makeFunction(name: "q4_swiglu_gate_up") {
                q4GateUpPipeline = try device.makeComputePipelineState(function: q4GateUpFunc)
            } else { q4GateUpPipeline = nil }

            if let q4DownFunc = defaultLibrary.makeFunction(name: "q4_down_proj_accumulate") {
                q4DownPipeline = try device.makeComputePipelineState(function: q4DownFunc)
            } else { q4DownPipeline = nil }

            if let q8GateUpFunc = defaultLibrary.makeFunction(name: "q8_swiglu_gate_up") {
                q8GateUpPipeline = try device.makeComputePipelineState(function: q8GateUpFunc)
            } else { q8GateUpPipeline = nil }

            if let q8DownFunc = defaultLibrary.makeFunction(name: "q8_down_proj_accumulate") {
                q8DownPipeline = try device.makeComputePipelineState(function: q8DownFunc)
            } else { q8DownPipeline = nil }

            if let q4q8Func = defaultLibrary.makeFunction(name: "q4_gate_q8_up_swiglu") {
                q4GateQ8UpPipeline = try device.makeComputePipelineState(function: q4q8Func)
            } else { q4GateQ8UpPipeline = nil }

            if let q8q4Func = defaultLibrary.makeFunction(name: "q8_gate_q4_up_swiglu") {
                q8GateQ4UpPipeline = try device.makeComputePipelineState(function: q8q4Func)
            } else { q8GateQ4UpPipeline = nil }

            if let hNormFunc = defaultLibrary.makeFunction(name: "per_head_rmsnorm_bf16") {
                headRmsnormPipeline = try device.makeComputePipelineState(function: hNormFunc)
            } else { headRmsnormPipeline = nil }

            if let hNormF16Func = defaultLibrary.makeFunction(name: "per_head_rmsnorm_f16") {
                headRmsnormF16Pipeline = try device.makeComputePipelineState(function: hNormF16Func)
            } else { headRmsnormF16Pipeline = nil }

            if let hNormOffsetFunc = defaultLibrary.makeFunction(name: "per_head_rmsnorm_offset_bf16") {
                headRmsnormOffsetPipeline = try device.makeComputePipelineState(function: hNormOffsetFunc)
            } else { headRmsnormOffsetPipeline = nil }

            if let hNormOffsetF16Func = defaultLibrary.makeFunction(name: "per_head_rmsnorm_offset_f16") {
                headRmsnormOffsetF16Pipeline = try device.makeComputePipelineState(function: hNormOffsetF16Func)
            } else { headRmsnormOffsetF16Pipeline = nil }

            if let ropeFunc = defaultLibrary.makeFunction(name: "apply_rope_qwen") {
                ropePipeline = try device.makeComputePipelineState(function: ropeFunc)
            } else { ropePipeline = nil }

            if let ropePropFunc = defaultLibrary.makeFunction(name: "apply_rope_proportional") {
                ropeProportionalPipeline = try device.makeComputePipelineState(function: ropePropFunc)
            } else { ropeProportionalPipeline = nil }

            if let storeKvFunc = defaultLibrary.makeFunction(name: "store_kv_cache") {
                storeKvCachePipeline = try device.makeComputePipelineState(function: storeKvFunc)
            } else { storeKvCachePipeline = nil }

            if let storeKvF16Func = defaultLibrary.makeFunction(name: "store_kv_cache_f16") {
                storeKvCacheF16Pipeline = try device.makeComputePipelineState(function: storeKvF16Func)
            } else { storeKvCacheF16Pipeline = nil }

            if let storeKvFP8Func = defaultLibrary.makeFunction(name: "store_kv_cache_fp8") {
                storeKvCacheFP8Pipeline = try device.makeComputePipelineState(function: storeKvFP8Func)
            } else { storeKvCacheFP8Pipeline = nil }

            if let gqaFunc = defaultLibrary.makeFunction(name: "gqa_attention_decode_fused") {
                gqaDecodePipeline = try device.makeComputePipelineState(function: gqaFunc)
            } else { gqaDecodePipeline = nil }

            if let gqaF16Func = defaultLibrary.makeFunction(name: "gqa_attention_decode_fused_f16") {
                gqaDecodeF16Pipeline = try device.makeComputePipelineState(function: gqaF16Func)
            } else { gqaDecodeF16Pipeline = nil }

            if let gqaFP8Func = defaultLibrary.makeFunction(name: "gqa_attention_decode_fused_fp8") {
                gqaDecodeFP8Pipeline = try device.makeComputePipelineState(function: gqaFP8Func)
            } else { gqaDecodeFP8Pipeline = nil }

            if let gqaStdFunc = defaultLibrary.makeFunction(name: "gqa_attention_decode_standard") {
                gqaStandardPipeline = try device.makeComputePipelineState(function: gqaStdFunc)
            } else { gqaStandardPipeline = nil }

            if let gqaStdF16Func = defaultLibrary.makeFunction(name: "gqa_attention_decode_standard_f16") {
                gqaStandardF16Pipeline = try device.makeComputePipelineState(function: gqaStdF16Func)
            } else { gqaStandardF16Pipeline = nil }

            if let gqaStdFP8Func = defaultLibrary.makeFunction(name: "gqa_attention_decode_standard_fp8") {
                gqaStandardFP8Pipeline = try device.makeComputePipelineState(function: gqaStdFP8Func)
            } else { gqaStandardFP8Pipeline = nil }

            if let convFunc = defaultLibrary.makeFunction(name: "causal_conv1d_silu") {
                causalConv1dPipeline = try device.makeComputePipelineState(function: convFunc)
            } else { causalConv1dPipeline = nil }

            if let l2Func = defaultLibrary.makeFunction(name: "l2_norm_qk") {
                l2NormQkPipeline = try device.makeComputePipelineState(function: l2Func)
            } else { l2NormQkPipeline = nil }

            if let linStepFunc = defaultLibrary.makeFunction(name: "linear_attention_recurrent_step") {
                linearAttnStepPipeline = try device.makeComputePipelineState(function: linStepFunc)
            } else { linearAttnStepPipeline = nil }

            if let linStepSigFunc = defaultLibrary.makeFunction(name: "linear_attention_recurrent_step_sigmoid") {
                linearAttnStepSigmoidPipeline = try device.makeComputePipelineState(function: linStepSigFunc)
            } else { linearAttnStepSigmoidPipeline = nil }

            if let sgFunc = defaultLibrary.makeFunction(name: "moe_shared_gate_bf16") {
                sharedGatePipeline = try device.makeComputePipelineState(function: sgFunc)
            } else { sharedGatePipeline = nil }

            if let sgQ4Func = defaultLibrary.makeFunction(name: "moe_shared_gate_q4") {
                sharedGateQ4Pipeline = try device.makeComputePipelineState(function: sgQ4Func)
            } else { sharedGateQ4Pipeline = nil }

            if let sgQ8Func = defaultLibrary.makeFunction(name: "moe_shared_gate_q8") {
                sharedGateQ8Pipeline = try device.makeComputePipelineState(function: sgQ8Func)
            } else { sharedGateQ8Pipeline = nil }

            if let r512Func = defaultLibrary.makeFunction(name: "moe_router_topk_512_bf16") {
                router512Pipeline = try device.makeComputePipelineState(function: r512Func)
            } else { router512Pipeline = nil }

            if let r512Q4Func = defaultLibrary.makeFunction(name: "moe_router_topk_512_q4") {
                router512Q4Pipeline = try device.makeComputePipelineState(function: r512Q4Func)
            } else { router512Q4Pipeline = nil }

            if let gdnFunc = defaultLibrary.makeFunction(name: "gdn_linear_attention_recurrent_step") {
                gdnLinearAttnStepPipeline = try device.makeComputePipelineState(function: gdnFunc)
            } else { gdnLinearAttnStepPipeline = nil }

            if let gdnSigFunc = defaultLibrary.makeFunction(name: "gdn_linear_attention_recurrent_step_sigmoid") {
                gdnLinearAttnStepSigmoidPipeline = try device.makeComputePipelineState(function: gdnSigFunc)
            } else { gdnLinearAttnStepSigmoidPipeline = nil }

            if let qsaIdxFunc = defaultLibrary.makeFunction(name: "qsa_mqa_indexer_score_blocks") {
                qsaMqaIndexerPipeline = try device.makeComputePipelineState(function: qsaIdxFunc)
            } else { qsaMqaIndexerPipeline = nil }

            if let grBlendFunc = defaultLibrary.makeFunction(name: "gated_residual_blend_4stream") {
                gatedResidualBlendPipeline = try device.makeComputePipelineState(function: grBlendFunc)
            } else { gatedResidualBlendPipeline = nil }

            if let pleFunc = defaultLibrary.makeFunction(name: "fuse_ngram_ple_embedding") {
                fuseNgramPlePipeline = try device.makeComputePipelineState(function: pleFunc)
            } else { fuseNgramPlePipeline = nil }

            if let init4StreamsFunc = defaultLibrary.makeFunction(name: "fused_init_4streams") {
                fusedInit4StreamsPipeline = try device.makeComputePipelineState(function: init4StreamsFunc)
            } else { fusedInit4StreamsPipeline = nil }

            if let extract0Func = defaultLibrary.makeFunction(name: "extract_stream0") {
                extractStream0Pipeline = try device.makeComputePipelineState(function: extract0Func)
            } else { extractStream0Pipeline = nil }

            if let hcNormFunc = defaultLibrary.makeFunction(name: "hyper_connection_norm_bf16") {
                hcNormPipeline = try device.makeComputePipelineState(function: hcNormFunc)
            } else { hcNormPipeline = nil }

            if let hcDownFunc = defaultLibrary.makeFunction(name: "hyper_connection_down_proj_bf16") {
                hcDownProjPipeline = try device.makeComputePipelineState(function: hcDownFunc)
            } else { hcDownProjPipeline = nil }

            if let hcUpFunc = defaultLibrary.makeFunction(name: "hyper_connection_up_proj_blend_bf16") {
                hcUpBlendPipeline = try device.makeComputePipelineState(function: hcUpFunc)
            } else { hcUpBlendPipeline = nil }

            if let hcInjScaleFunc = defaultLibrary.makeFunction(name: "hyper_connection_inject_scale_bf16") {
                hcInjectScalePipeline = try device.makeComputePipelineState(function: hcInjScaleFunc)
            } else { hcInjectScalePipeline = nil }

            if let hcInjFunc = defaultLibrary.makeFunction(name: "hyper_connection_inject_bf16") {
                hcInjectPipeline = try device.makeComputePipelineState(function: hcInjFunc)
            } else { hcInjectPipeline = nil }

            if let fp8SimdFunc = defaultLibrary.makeFunction(name: "fp8_gemv_simd") {
                fp8GemvSimdPipeline = try device.makeComputePipelineState(function: fp8SimdFunc)
            } else { fp8GemvSimdPipeline = nil }

            if let convSeqFunc = defaultLibrary.makeFunction(name: "causal_conv1d_sequence_silu") {
                causalConv1dSeqPipeline = try device.makeComputePipelineState(function: convSeqFunc)
            } else { causalConv1dSeqPipeline = nil }

            if let l2SeqFunc = defaultLibrary.makeFunction(name: "l2_norm_qk_sequence") {
                l2NormQkSeqPipeline = try device.makeComputePipelineState(function: l2SeqFunc)
            } else { l2NormQkSeqPipeline = nil }

            if let gdnSeqFunc = defaultLibrary.makeFunction(name: "linear_attention_recurrent_sequence") {
                gdnLinearAttnSeqPipeline = try device.makeComputePipelineState(function: gdnSeqFunc)
            } else { gdnLinearAttnSeqPipeline = nil }

            if let gdnSeqSigFunc = defaultLibrary.makeFunction(name: "linear_attention_recurrent_sequence_sigmoid") {
                gdnLinearAttnSeqSigmoidPipeline = try device.makeComputePipelineState(function: gdnSeqSigFunc)
            } else { gdnLinearAttnSeqSigmoidPipeline = nil }

            if let fp8GateBatchedFunc = defaultLibrary.makeFunction(name: "fp8_swiglu_gate_up_batched") {
                fp8GateUpBatchedPipeline = try device.makeComputePipelineState(function: fp8GateBatchedFunc)
            } else { fp8GateUpBatchedPipeline = nil }

            if let fp8DownBatchedFunc = defaultLibrary.makeFunction(name: "fp8_down_proj_accumulate_batched") {
                fp8DownBatchedPipeline = try device.makeComputePipelineState(function: fp8DownBatchedFunc)
            } else { fp8DownBatchedPipeline = nil }

            if let fp8BlockGateBatchedFunc = defaultLibrary.makeFunction(name: "fp8_block_swiglu_gate_up_batched") {
                fp8BlockGateUpBatchedPipeline = try device.makeComputePipelineState(function: fp8BlockGateBatchedFunc)
            } else { fp8BlockGateUpBatchedPipeline = nil }

            if let fp8BlockDownBatchedFunc = defaultLibrary.makeFunction(name: "fp8_block_down_proj_accumulate_batched") {
                fp8BlockDownBatchedPipeline = try device.makeComputePipelineState(function: fp8BlockDownBatchedFunc)
            } else { fp8BlockDownBatchedPipeline = nil }

            if let bf16GateBatchedFunc = defaultLibrary.makeFunction(name: "bf16_swiglu_gate_up_batched") {
                bf16GateUpBatchedPipeline = try device.makeComputePipelineState(function: bf16GateBatchedFunc)
            } else { bf16GateUpBatchedPipeline = nil }

            if let bf16DownBatchedFunc = defaultLibrary.makeFunction(name: "bf16_down_proj_accumulate_batched") {
                bf16DownBatchedPipeline = try device.makeComputePipelineState(function: bf16DownBatchedFunc)
            } else { bf16DownBatchedPipeline = nil }

            if let kdaFunc = defaultLibrary.makeFunction(name: "kda_linear_attention_recurrent_step") {
                kdaLinearAttnStepPipeline = try device.makeComputePipelineState(function: kdaFunc)
            } else { kdaLinearAttnStepPipeline = nil }

            if let ropeInterFunc = defaultLibrary.makeFunction(name: "apply_rope_interleaved") {
                ropeInterleavedPipeline = try device.makeComputePipelineState(function: ropeInterFunc)
            } else { ropeInterleavedPipeline = nil }

            if let routerBailingFunc = defaultLibrary.makeFunction(name: "moe_router_bailing_topk_bf16") {
                routerBailingPipeline = try device.makeComputePipelineState(function: routerBailingFunc)
            } else { routerBailingPipeline = nil }

            if let storeMlaFunc = defaultLibrary.makeFunction(name: "store_mla_kv_cache") {
                storeMlaKvCachePipeline = try device.makeComputePipelineState(function: storeMlaFunc)
            } else { storeMlaKvCachePipeline = nil }

            if let storeMlaF16Func = defaultLibrary.makeFunction(name: "store_mla_kv_cache_f16") {
                storeMlaKvCacheF16Pipeline = try device.makeComputePipelineState(function: storeMlaF16Func)
            } else { storeMlaKvCacheF16Pipeline = nil }

            if let mlaDecodeFunc = defaultLibrary.makeFunction(name: "mla_attention_decode") {
                mlaAttentionDecodePipeline = try device.makeComputePipelineState(function: mlaDecodeFunc)
            } else { mlaAttentionDecodePipeline = nil }

            if let mlaDecodeF16Func = defaultLibrary.makeFunction(name: "mla_attention_decode_f16") {
                mlaAttentionDecodeF16Pipeline = try device.makeComputePipelineState(function: mlaDecodeF16Func)
            } else { mlaAttentionDecodeF16Pipeline = nil }

            if let f = defaultLibrary.makeFunction(name: "rmsnorm_no_scale_bf16") {
                rmsnormNoScalePipeline = try device.makeComputePipelineState(function: f)
            } else { rmsnormNoScalePipeline = nil }

            if let f = defaultLibrary.makeFunction(name: "gqa_attention_decode_gemma") {
                gqaGemmaPipeline = try device.makeComputePipelineState(function: f)
            } else { gqaGemmaPipeline = nil }

            if let f = defaultLibrary.makeFunction(name: "gqa_attention_decode_gemma_f16") {
                gqaGemmaF16Pipeline = try device.makeComputePipelineState(function: f)
            } else { gqaGemmaF16Pipeline = nil }

            if let f = defaultLibrary.makeFunction(name: "gqa_attention_decode_gemma_fp8") {
                gqaGemmaFP8Pipeline = try device.makeComputePipelineState(function: f)
            } else { gqaGemmaFP8Pipeline = nil }

            if let f = defaultLibrary.makeFunction(name: "store_kv_cache_gemma") {
                storeKvGemmaPipeline = try device.makeComputePipelineState(function: f)
            } else { storeKvGemmaPipeline = nil }

            if let f = defaultLibrary.makeFunction(name: "store_kv_cache_gemma_f16") {
                storeKvGemmaF16Pipeline = try device.makeComputePipelineState(function: f)
            } else { storeKvGemmaF16Pipeline = nil }

            if let f = defaultLibrary.makeFunction(name: "store_kv_cache_gemma_fp8") {
                storeKvGemmaFP8Pipeline = try device.makeComputePipelineState(function: f)
            } else { storeKvGemmaFP8Pipeline = nil }

            if let f = defaultLibrary.makeFunction(name: "gqa_attention_tree_verify_gemma") {
                gqaGemmaTreeVerifyPipeline = try device.makeComputePipelineState(function: f)
            } else { gqaGemmaTreeVerifyPipeline = nil }

            if let f = defaultLibrary.makeFunction(name: "gqa_attention_tree_verify_gemma_f16") {
                gqaGemmaTreeVerifyF16Pipeline = try device.makeComputePipelineState(function: f)
            } else { gqaGemmaTreeVerifyF16Pipeline = nil }

            if let f = defaultLibrary.makeFunction(name: "mul_by_bf16_vector") {
                mulByBf16VectorPipeline = try device.makeComputePipelineState(function: f)
            } else { mulByBf16VectorPipeline = nil }

            if let f = defaultLibrary.makeFunction(name: "scale_vector_inplace") {
                scaleVectorInplacePipeline = try device.makeComputePipelineState(function: f)
            } else { scaleVectorInplacePipeline = nil }

            if let f = defaultLibrary.makeFunction(name: "apply_topk_per_expert_scale") {
                perExpertScalePipeline = try device.makeComputePipelineState(function: f)
            } else { perExpertScalePipeline = nil }
        } catch {
            let err = "❌ Failed to create compute pipelines: \(error.localizedDescription)"
            throw CompletionLaunchError.engineNotReady(err)
        }

        // Model Hyperparameters & Dynamic Sizing
        let hiddenDim: UInt32 = UInt32(modelConfig?.effectiveHiddenSize ?? (isNanbeige ? 3072 : 2048))
        let numHeads: UInt32 = UInt32(modelConfig?.effectiveNumAttentionHeads ?? (isNanbeige ? 48 : 16))
        let numKvHeads: UInt32 = UInt32(modelConfig?.effectiveNumKeyValueHeads ?? (isNanbeige ? 8 : 2))
        let headDim: UInt32 = UInt32(modelConfig?.effectiveHeadDim ?? 128)
        let rotaryDim: UInt32 = UInt32(modelConfig?.effectiveRotaryDim ?? 128)
        let thetaVal: Float = modelConfig?.effectiveRopeTheta ?? (isNanbeige ? 70000000.0 : 10000000.0)
        let totalLoops: Int = max(1, modelConfig?.effectiveNumLoops ?? (isNanbeige ? 2 : 1))
        let eosTokenId: UInt32 = UInt32(modelConfig?.effectiveEosTokenId ?? (isNanbeige ? 166101 : 248044))

        let cachedLayers = InferenceEngine.shared.buildCachedLayers(summary: summary, config: modelConfig)
        let actualLayers = cachedLayers.count
        guard actualLayers > 0 else {
            let err = "❌ No layer tensors found in model."
            throw CompletionLaunchError.engineNotReady(err)
        }

        let arch = modelConfig?.resolveArchitectureType(summary: summary) ?? (summary.maxExpertId > 0 ? .hybridSsmMoe : .denseTransformer)
        let isHybridArch = arch.isHybridSsm
        let isRMSNormOffset = modelConfig?.isRMSNormUnitOffset ?? false

        let isGatedQ: Bool = {
            if let configGate = modelConfig?.effectiveAttnOutputGate {
                return configGate
            }
            if let qTensor = cachedLayers.first(where: { $0.attentionType == .fullAttention })?.qProjTensor {
                let gatedDim = numHeads * headDim * 2
                return qTensor.shapeDisplay.contains("\(gatedDim)")
            }
            return false
        }()
        let isStandardGqa = !isGatedQ
        let qOutDim: UInt32 = isStandardGqa ? (numHeads * headDim) : (numHeads * headDim * 2)
        let kvOutDim: UInt32 = numKvHeads * headDim
        let attnCtxDim: UInt32 = numHeads * headDim
        let kvStride: UInt32 = numKvHeads * headDim

        // Gemma 4: hybrid sliding/global attention. Global layers use
        // global_head_dim with a reduced KV head count and reuse the key
        // projection as the value projection; the shared KV cache is allocated
        // with a single uniform stride (`kvStride`).
        let isGemma4 = (modelConfig?.isGemma4Model == true)
        let globalHeadDim: UInt32 = UInt32(modelConfig?.effectiveGlobalHeadDim ?? 0)
        let numGlobalKvHeads: UInt32 = UInt32(modelConfig?.effectiveNumGlobalKeyValueHeads ?? 0)
        let gemmaExpertInterDim: UInt32 = UInt32(modelConfig?.effectiveMoeIntermediateSize ?? 704)
        let gemmaScaling: Float = 1.0
        let gemmaKvStride: UInt32 = kvStride

        let numExperts: UInt32 = summary.maxExpertId > 0 ? summary.maxExpertId : UInt32(modelConfig?.effectiveNumExperts ?? 0)
        let embedOffset = embedWeight.offsetStart
        let normOffset = normTensor?.offsetStart ?? 0
        let lmHeadOffset = lmHeadTensor.offsetStart
        let eps: Float = modelConfig?.effectiveRmsNormEps ?? (isNanbeige ? 1e-5 : 1e-6)

        var vocabSize: UInt32 = 248320
        let cleanShape = lmHeadTensor.shapeDisplay.replacingOccurrences(of: "[", with: "").replacingOccurrences(of: "]", with: "").replacingOccurrences(of: " ", with: "")
        let shapeParts = cleanShape.split(separator: ",")
        if let first = shapeParts.first, let parsed = UInt32(first), parsed > 0 {
            vocabSize = parsed
        } else if let cfgVocab = modelConfig?.effectiveVocabSize {
            vocabSize = UInt32(cfgVocab)
        }

        let maxInterDim = cachedLayers.map { $0.intermediateDim }.max() ?? 512

        var loadedLayout: FlashMoELayout? = nil
        let packedExpertsDir: URL?
        if let modelPath = activeLoadedModelPath {
            var isDir: ObjCBool = false
            FileManager.default.fileExists(atPath: modelPath, isDirectory: &isDir)
            let baseDir = isDir.boolValue ? URL(fileURLWithPath: modelPath) : URL(fileURLWithPath: modelPath).deletingLastPathComponent()
            let pDir = baseDir.appendingPathComponent("packed_experts")
            let layoutFile = pDir.appendingPathComponent("layout.json")
            if FileManager.default.fileExists(atPath: layoutFile.path),
               let lData = try? Data(contentsOf: layoutFile),
               let lay = try? JSONDecoder().decode(FlashMoELayout.self, from: lData) {
                packedExpertsDir = pDir
                loadedLayout = lay
                ExpertIOThreadPool.shared.initialize(numThreads: 8)
            } else {
                packedExpertsDir = nil
            }
        } else {
            packedExpertsDir = nil
        }

        // FIX #8: 32 slots per staging buffer (kick candidates ≤16 + topK misses ≤8
        // with headroom; ~100.8 MB per buffer for the FP8 packed layout).
        let expertStagingSize = max(32 * 4194304, 32 * Int(loadedLayout?.expert_size ?? 4718592))

        // Allocate Shared Scratch Buffers (Reused across all generation steps)
        guard let singleTokenBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let hCurrBuffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hNextBuffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let xNorm1Buffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let qGateBuffer = device.makeBuffer(length: max(Int(qOutDim), 10240) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let kVectorBuffer = device.makeBuffer(length: max(Int(kvOutDim), 4096) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let vVectorBuffer = device.makeBuffer(length: max(Int(kvOutDim), 6144) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let zGateBuffer = device.makeBuffer(length: 8192 * MemoryLayout<Float>.stride, options: .storageModeShared),
              let aVectorBuffer = device.makeBuffer(length: max(2048, Int(hiddenDim)) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let bVectorBuffer = device.makeBuffer(length: max(2048, Int(hiddenDim)) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let attnCtxBuffer = device.makeBuffer(length: max(Int(attnCtxDim), 8192) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let attnOutBuffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hMidBuffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let xNorm2Buffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hMlpBuffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let routerIndicesBuffer = device.makeBuffer(length: 16 * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let routerWeightsBuffer = device.makeBuffer(length: 16 * MemoryLayout<Float>.stride, options: .storageModeShared),
              let sharedScoreBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared),
              let interBuffer = device.makeBuffer(length: max(Int(maxInterDim), 512) * max(1, Int(modelConfig?.effectiveNumExpertsPerTok ?? (numExperts >= 512 ? 10 : 8))) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let fusedSlotListBuffer = device.makeBuffer(length: 16 * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let fusedSlotWeightsBuffer = device.makeBuffer(length: 16 * MemoryLayout<Float>.stride, options: .storageModeShared),
              // Flash-decoding partials (256 = split-K chunk cap), sized from the
              // model's head count — a fixed 16-head budget overruns on wider models.
              let attnPartialMBuffer = device.makeBuffer(length: Int(numHeads) * 256 * MemoryLayout<Float>.stride, options: .storageModeShared),
              let attnPartialLBuffer = device.makeBuffer(length: Int(numHeads) * 256 * MemoryLayout<Float>.stride, options: .storageModeShared),
              let attnPartialAccBuffer = device.makeBuffer(length: Int(numHeads) * 256 * Int(headDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let expertStagingBuffer = device.makeBuffer(length: expertStagingSize, options: .storageModeShared),
              let expertStagingBufferB = device.makeBuffer(length: expertStagingSize, options: .storageModeShared),
              let hcStreamsBuffer = device.makeBuffer(length: max(4 * Int(hiddenDim), 10240) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hcNormedBuffer = device.makeBuffer(length: max(4 * Int(hiddenDim), 10240) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hcBottleneckBuffer = device.makeBuffer(length: 512 * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hcInjectScaleBuffer = device.makeBuffer(length: 4 * MemoryLayout<Float>.stride, options: .storageModeShared),
              let xFinalBuffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let logitsBuffer = device.makeBuffer(length: Int(vocabSize) * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            let err = "❌ Failed to allocate GPU scratch buffers."
            throw CompletionLaunchError.engineNotReady(err)
        }

        let temp = sampling.temperature
        let topPVal = sampling.topP
        let minPVal = sampling.minP
        let topKVal = sampling.topK
        let repPen = sampling.repetitionPenalty
        let presPen = sampling.presencePenalty
        var maxTokens = sampling.maxTokens
        let buffers = shardBuffers
        let effMode = memory.executionMode.resolveEffectiveMode(modelFootprintGB: summary.sizeGb)
        let isFullRAM = (effMode == .residentRAM || memory.budgetMode == .unrestricted)
        let budgetMode: MemoryBudgetMode = isFullRAM ? .unrestricted : memory.budgetMode

        let isAgentSession = request.agentToolsEnabled
        let minSeq = isAgentSession ? 8192 : 2048
        var neededSeqLen = max(minSeq, min(32768, promptTokenIds.count + maxTokens + 512))
        var kvPrec = memory.kvPrecision

        // Reject (or ratchet) turns whose prompt cannot fit the KV cache this
        // machine can provision. Without this check the prefill kept running
        // past a clamped allocation and the attention/KV kernels indexed rows
        // that were never allocated — silently corrupting cache slots and, in
        // the worst case, walking off the buffer. The check runs before the
        // multi-GB scratch/KV allocations so a rejected turn costs nothing.
        let fitPhysicalGB = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        let fitCheck = KVCacheProvisioning.turnFit(
            isAgentSession: isAgentSession,
            promptTokens: promptTokenIds.count,
            requestedMaxTokens: maxTokens,
            config: modelConfig,
            actualLayers: actualLayers,
            totalLoops: totalLoops,
            numKvHeads: Int(numKvHeads),
            headDim: Int(headDim),
            precision: memory.kvPrecision,
            slidingSlotFlags: nil,
            slidingRingLen: 0,
            budgetMode: memory.budgetMode,
            modelFootprintGB: summary.sizeGb,
            physicalGB: fitPhysicalGB
        )
        switch fitCheck.outcome {
        case .promptTooLong(let provisioned):
            throw CompletionLaunchError.promptTooLong(
                "⚠️ Prompt too long: \(promptTokenIds.count) prompt tokens exceed the \(provisioned)-token KV capacity this turn can provision " +
                "(KV budget \(String(format: "%.2f", fitCheck.budgetGB)) GB on a \(String(format: "%.0f", fitPhysicalGB)) GB machine; " +
                "shorten the prompt, lower max_tokens, or raise the memory budget)."
            )
        case .maxTokensReduced(let provisioned):
            let capped = max(1, min(maxTokens, fitCheck.effectiveMaxTokens))
            if capped < maxTokens {
                print("⚠️ [KVCACHE] clamping max_tokens \(maxTokens) -> \(capped): \(promptTokenIds.count)-token prompt fits a \(provisioned)-token KV provision (budget \(String(format: "%.2f", fitCheck.budgetGB)) GB)")
                maxTokens = capped
                // Keep the rest of the launch consistent with the capped budget: the clamp below re-derives
                // provisioned length from prompt + maxTokens, so mirror the cap there too.
                neededSeqLen = KVCacheProvisioning.neededSeqLen(
                    isAgentSession: isAgentSession,
                    promptTokens: promptTokenIds.count,
                    requestedMaxTokens: capped
                )
            }
        case .ok:
            break
        }
        if modelConfig?.isLingModel == true && kvPrec != .fp16 {
            // Ling MLA KV store/decode kernels are F16-only and their byte-offset math
            // hardcodes MemoryLayout<UInt16>.stride (see the MLA branch in runTokenForward).
            // A lower-precision allocation addresses half the bytes, so the slot-3 offset
            // runs exactly one buffer past the end -> Metal setBuffer assertion at layer 15.
            print("⚠️ [KVCACHE] Ling MLA models require FP16 KV cache — overriding \(kvPrec) -> fp16")
            kvPrec = .fp16
        }
        if modelConfig?.isGemma4Model == true && kvPrec == .fp8 && storeKvGemmaFP8Pipeline == nil {
            // Gemma 4 has dedicated F16/FP32/FP8 store + decode kernels. Only fall back if
            // the FP8 pair failed to load (stale metallib), since the attention branch would
            // otherwise read the INT8 cache with the float kernels.
            print("⚠️ [KVCACHE] Gemma 4 FP8 KV kernels unavailable — overriding fp8 -> fp16")
            kvPrec = .fp16
        }
        let hasRecurrence = (modelConfig?.isLingModel == true || modelConfig?.hasLinearRecurrence == true || cachedLayers.contains { $0.attentionType == .linearAttention })
        // A pin minted under one KV precision cannot be restored under another: the
        // preserve-path memcpy sizes slots with the *new* elementBytes against a cache
        // laid out with the old ones (and FP8 additionally needs its scale buffers).
        // activePrecision still holds the previous turn's value until reset() below.
        if kvPrec != KVCacheManager.shared.activePrecision && PrefixCacheManager.shared.currentPinnedCount > 0 {
            print("🔁 [PREFIX] KV precision changed \(KVCacheManager.shared.activePrecision) -> \(kvPrec) — pinned prefix is unusable, full re-prefill")
            PrefixCacheManager.shared.invalidate()
        }
        // FIX #10: hybrid (GDN) models now reuse the KV prefix too — but only when
        // the ENTIRE pinned prefix matches, because the O(1) recurrent states are
        // only valid at exactly the pinned token count (snapshot is taken at turn
        // end next to recordTurn).
        let reuseCandidate = PrefixCacheManager.shared.findCommonPrefix(promptTokenIds: promptTokenIds, sessionId: sessionId)
        var prefixTokensReused: Int
        // A/B toggle for bisecting the tool-call truncation: defaults write
        // DRP.DynaMoE dynamoe_disable_prefix_reuse -bool YES
        if UserDefaults.standard.bool(forKey: "dynamoe_disable_prefix_reuse") {
            prefixTokensReused = 0
        } else if hasRecurrence {
            if reuseCandidate > 0 && reuseCandidate == PrefixCacheManager.shared.currentPinnedCount {
                prefixTokensReused = reuseCandidate
            } else {
                if reuseCandidate != 0 {
                    print("🔁 [PREFIX] partial match \(reuseCandidate)/\(PrefixCacheManager.shared.currentPinnedCount) pinned — hybrid state cannot rewind; full re-prefill")
                } else if PrefixCacheManager.shared.currentPinnedCount > 0 {
                    print("🔁 [PREFIX] no common prefix with pin of \(PrefixCacheManager.shared.currentPinnedCount) (session mismatch or prompt diverged at token 0) — full re-prefill")
                }
                prefixTokensReused = 0
            }
        } else {
            prefixTokensReused = reuseCandidate
            // Gemma 4's sliding attention is not `linearAttention`, so it takes this
            // non-recurrence branch and a failed reuse is otherwise silent (none of the
            // recurrence-branch prints fire). Name it so a "full re-prefill every turn"
            // report is attributable instead of showing up only as an unexplained loop.
            if isGemma4, prefixTokensReused == 0, PrefixCacheManager.shared.currentPinnedCount > 0 {
                print("🔁 [PREFIX] Gemma pin of \(PrefixCacheManager.shared.currentPinnedCount) not reusable (reuseCandidate=\(reuseCandidate)) — full re-prefill of \(promptTokenIds.count) tokens")
            }
        }

        // Gemma 4 sliding-window ring KV: the sliding layers' ring rows only
        // survive a turn boundary while nothing stored past the pin boundary
        // wrapped onto the reused window's rows. The ring bookkeeping knows
        // exactly, so validate before splicing; a clobbered window falls back
        // to a full re-prefill (same policy as the GDN "cannot rewind" path).
        let gemmaSlidingWindow = Int(modelConfig?.effectiveSlidingWindow ?? 0)
        // Ring slack: must cover the largest batched prefill chunk (the chunk
        // cap below) and the JetSpec tree scratch (maxNodes=8 node slots plus
        // the root re-store) so in-flight batches never overwrite a live row.
        let gemmaRingSlack = max(gemmaPrefillChunkCap, 16)
        let isGemma4Ring = isGemma4 && gemmaSlidingWindow > 0
        let gemmaSlidingRingLen = isGemma4Ring ? (gemmaSlidingWindow + gemmaRingSlack) : 0
        if isGemma4Ring && prefixTokensReused > 0,
           !KVCacheManager.shared.ringPrefixRowsIntact(prefixCount: prefixTokensReused, window: gemmaSlidingWindow) {
            print("🔁 [PREFIX] Gemma sliding-ring window rows were overwritten past the pin (\(PrefixCacheManager.shared.currentPinnedCount) pinned, reusing \(prefixTokensReused)) — full re-prefill")
            prefixTokensReused = 0
        }
        // Per-slot ring flags in KVCacheManager slot order (loopIdx * actualLayers
        // + fullAttnIndex; for Gemma every layer is full-attention, so this is
        // just the layer order).
        var gemmaSlidingSlotFlags: [Bool]? = nil
        if isGemma4Ring {
            var flags = [Bool](repeating: false, count: max(1, totalLoops) * actualLayers)
            for loopIdx in 0..<max(1, totalLoops) {
                for l in 0..<actualLayers {
                    let layer = cachedLayers[l]
                    if layer.attentionType == .fullAttention {
                        let slot = (loopIdx * actualLayers) + layer.fullAttnIndex
                        if slot < flags.count { flags[slot] = layer.isSlidingAttention }
                    }
                }
            }
            gemmaSlidingSlotFlags = flags
        }

        // Prompt-dump diagnostic for the tool-call spiral bisection
        // (defaults write DRP.DynaMoE dynamoe_dump_prompts -bool YES).
        // Writes the EXACT decoded token stream the model attends over this turn
        // to ~/Downloads/DynaMoePromptDumps/, and for spliced continuations also
        // verifies the splice against a fresh encode of the equivalent string
        // prompt. Small divergences (1-2 tokens at a turn boundary) are the
        // known-benign non-round-trip cases the splice exists to avoid; a
        // structural divergence (missing/duplicated block) is the smoking gun.
        // Marker-file gate too: a sandboxed app's prefs live in its container and
        // cfprefsd can serve a stale cached value, so a plain `defaults write`
        // sometimes never reaches a running app. Touch this file to enable:
        //   touch ~/Downloads/DynaMoE-dump-prompts
        let dumpMarker = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Downloads/DynaMoE-dump-prompts")
        let dumpEnabled = UserDefaults.standard.bool(forKey: "dynamoe_dump_prompts")
            || FileManager.default.fileExists(atPath: dumpMarker.path)
        print("🧪 [DUMP] checkpoint: dump_flag=\(UserDefaults.standard.bool(forKey: "dynamoe_dump_prompts")) marker=\(FileManager.default.fileExists(atPath: dumpMarker.path)) prefix_flag=\(UserDefaults.standard.bool(forKey: "dynamoe_disable_prefix_reuse")) tokens=\(promptTokenIds.count)")
        if dumpEnabled {
            let header = """
            turn meta: promptTokens=\(promptTokenIds.count) prefixReused=\(prefixTokensReused) pin=\(PrefixCacheManager.shared.currentPinnedCount) agentStep=\(agentStep) spliced=\(promptTokens != nil) sessionId=\(sessionId?.uuidString ?? "nil")
            """
            var report = header + "\n\n"
            if let encodedStringPrompt = try? tokenizer.encode(text: formattedPrompt), promptTokens != nil {
                if encodedStringPrompt == promptTokenIds {
                    report += "splice-vs-reencode: IDENTICAL (\(promptTokenIds.count) tokens)\n\n"
                } else {
                    let common = zip(encodedStringPrompt, promptTokenIds).prefix(while: { $0 == $1 }).count
                    let winLo = max(0, common - 16)
                    let winHi = min(min(encodedStringPrompt.count, promptTokenIds.count), common + 16)
                    let splicedWin = (try? tokenizer.decode(ids: Array(promptTokenIds[winLo..<min(winHi + 64, promptTokenIds.count)]))) ?? "?"
                    let encodedWin = (try? tokenizer.decode(ids: Array(encodedStringPrompt[winLo..<min(winHi + 64, encodedStringPrompt.count)]))) ?? "?"
                    report += "splice-vs-reencode: DIVERGES at token \(common) (splice=\(promptTokenIds.count) string=\(encodedStringPrompt.count))\n"
                    report += "--- spliced window: \(splicedWin)\n--- re-encoded window: \(encodedWin)\n\n"
                }
            }
            report += (try? tokenizer.decode(ids: promptTokenIds)) ?? "<decode failed>"
            var dumpDir = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Downloads/DynaMoePromptDumps")
            do {
                try FileManager.default.createDirectory(at: dumpDir, withIntermediateDirectories: true)
            } catch {
                dumpDir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("DynaMoePromptDumps")
                try? FileManager.default.createDirectory(at: dumpDir, withIntermediateDirectories: true)
            }
            let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
            let dumpFile = dumpDir.appendingPathComponent("prompt-step\(agentStep)-\(stamp).txt")
            do {
                try report.write(to: dumpFile, atomically: true, encoding: .utf8)
                print("🧪 [DUMP] wrote \(dumpFile.path) (\(promptTokenIds.count) tokens, reused \(prefixTokensReused))")
            } catch {
                print("🧪 [DUMP] FAILED to write \(dumpFile.path): \(error.localizedDescription)")
            }
        }

        // Clamp KV provisioning to what this machine can actually hold. Weights
        // plus a working reserve come off the process target, and "Full RAM"
        // falls back to a 75%-of-physical-RAM ceiling instead of none — without
        // it a 16 GB Mac provisioned 32k tokens of KV for a 4B model (~5 GB)
        // and drove the process past physical RAM. The pinned prefix is a hard
        // floor: the KV relayout preserves it.
        let physicalGB = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        let kvBudgetGB = KVCacheProvisioning.budgetGB(physicalGB: physicalGB, modelFootprintGB: summary.sizeGb, budgetMode: budgetMode)
        let kvGeometry = KVCacheProvisioning.geometry(
            config: modelConfig,
            actualLayers: actualLayers,
            totalLoops: totalLoops,
            numKvHeads: Int(numKvHeads),
            headDim: Int(headDim),
            precision: kvPrec,
            slidingSlotFlags: gemmaSlidingSlotFlags,
            slidingRingLen: gemmaSlidingRingLen
        )
        // The fit check ran before the Ling/Gemma precision overrides, so this
        // launch-side re-derivation is the only one computed from the EFFECTIVE
        // KV precision. Re-gate here: when the budget cannot hold the session
        // floor or the pinned prefix, fail fast in milliseconds instead of
        // over-provisioning the floor size past the budget and thrashing to
        // death mid-generation (the 16 GB repro: a 8,798-token turn accepted,
        // 4.3 s/layer, dead at token 913).
        let preservedRows = max(minSeq, prefixTokensReused)
        let affordability = KVCacheProvisioning.affordableSeqLen(geometry: kvGeometry, budgetGB: kvBudgetGB)
        if affordability < preservedRows {
            throw CompletionLaunchError.promptTooLong(
                "⚠️ KV budget \(String(format: "%.2f", kvBudgetGB)) GB on a \(String(format: "%.0f", physicalGB)) GB machine cannot hold the \(preservedRows)-row floor this turn must provision " +
                "(affords \(affordability) rows: pinned prefix \(prefixTokensReused), session floor \(minSeq), \(kvPrec) KV cache). " +
                "Raise the memory budget mode or lower the KV cache precision."
            )
        }
        let clampedSeqLen = KVCacheProvisioning.clampedSeqLen(
            requested: neededSeqLen,
            floor: preservedRows,
            geometry: kvGeometry,
            budgetGB: kvBudgetGB
        )
        if clampedSeqLen < promptTokenIds.count + 64 {
            throw CompletionLaunchError.promptTooLong(
                "⚠️ \(promptTokenIds.count) prompt tokens exceed the \(clampedSeqLen)-row KV provision the budget affords " +
                "(KV budget \(String(format: "%.2f", kvBudgetGB)) GB on a \(String(format: "%.0f", physicalGB)) GB machine at \(kvPrec) KV cache; " +
                "shorten the prompt, raise the memory budget mode, or lower the KV cache precision)."
            )
        }
        if clampedSeqLen < neededSeqLen {
            let kvGB = Double(KVCacheProvisioning.requiredBytes(maxSeqLen: clampedSeqLen, geometry: kvGeometry)) / 1_073_741_824
            print("⚠️ [KVCACHE] clamping provisioned seq len \(neededSeqLen) -> \(clampedSeqLen) (KV \(String(format: "%.2f", kvGB)) GB against a \(String(format: "%.2f", kvBudgetGB)) GB budget on a \(String(format: "%.1f", physicalGB)) GB machine; set the memory budget mode to raise it)")
            neededSeqLen = clampedSeqLen
        }

        KVCacheManager.shared.reset(
            device: device,
            config: modelConfig,
            actualLayers: actualLayers,
            totalLoops: totalLoops,
            numKvHeads: Int(numKvHeads),
            headDim: Int(headDim),
            maxSeqLen: neededSeqLen,
            precision: kvPrec,
            preservePrefixCount: prefixTokensReused,
            slidingSlotFlags: gemmaSlidingSlotFlags,
            slidingRingLen: gemmaSlidingRingLen
        )

        // Old ordering preserved: KV reset (above) -> grammar token -> turn id.
        let grammarGenerationToken = GrammarConstrainedSampler.shared.beginGeneration()
        stateLock.lock()
        generationCounter &+= 1
        let myGenerationId = generationCounter
        currentGenerationId = myGenerationId
        stateLock.unlock()

        let prep = TurnPrepInfo(
            prefixTokensReused: prefixTokensReused,
            promptTokenCount: promptTokenIds.count,
            effectiveKVPrecision: kvPrec,
            provisionedSequenceLength: neededSeqLen
        )

        let engineSelf = self
        let (stream, continuation) = AsyncStream.makeStream(of: CompletionEvent.self, bufferingPolicy: .unbounded)
        let handleTask = Task.detached(priority: .userInitiated) {
            func emit(_ event: CompletionEvent) { continuation.yield(event) }
            let isCurrentGeneration = { (id: UInt64) in engineSelf.isCurrentGenerationUnchecked(id) }
            engineSelf.turnDidBegin(myGenerationId)
            defer {
                engineSelf.turnDidEnd(id: myGenerationId)
                continuation.finish()
            }
            var contextTokens = promptTokenIds
            let startTime = CFAbsoluteTimeGetCurrent()
            var firstTokenTimestamp: Double? = nil
            var thinkingEndTimestamp: Double? = nil
            var tokensGenerated = 0
            var currentStep: UInt32 = 0

            // Helper for Linear/GEMV Projections (handles Q4, FP8, and BF16 SIMD)
            func dispatchLinear(
                enc: MTLComputeCommandEncoder,
                weight: TensorMetadata?,
                scale: TensorMetadata?,
                bias: TensorMetadata?,
                inBuf: MTLBuffer,
                outBuf: MTLBuffer,
                inDim: UInt32,
                outDim: UInt32,
                groupSize: UInt32 = 64,
                inOffset: Int = 0,
                outOffset: Int = 0,
                batchSize: Int = 1,
                weightOffsetAdd: UInt64 = 0
            ) {
                // Split oversized batched GEMVs into short dispatches. Each row is independent,
                // so slicing is mathematically identical while letting the GPU scheduler
                // interleave WindowServer frames between dispatches (prevents
                // kIOGPUCommandBufferCallbackErrorImpactingInteractivity aborts and UI jank).
                let maxGemmBatchRows = 256
                if batchSize > maxGemmBatchRows {
                    var sliceStart = 0
                    while sliceStart < batchSize {
                        let rows = min(maxGemmBatchRows, batchSize - sliceStart)
                        dispatchLinear(
                            enc: enc, weight: weight, scale: scale, bias: bias,
                            inBuf: inBuf, outBuf: outBuf, inDim: inDim, outDim: outDim,
                            groupSize: groupSize,
                            inOffset: inOffset + sliceStart * Int(inDim) * MemoryLayout<Float>.stride,
                            outOffset: outOffset + sliceStart * Int(outDim) * MemoryLayout<Float>.stride,
                            batchSize: rows,
                            weightOffsetAdd: weightOffsetAdd
                        )
                        sliceStart += rows
                    }
                    return
                }
                guard let w = weight, let wRaw = buffers[w.shardIndex] else { return }
                var wOff = w.offsetStart + weightOffsetAdd
                var inD = inDim
                var outD = outDim
                var grp = groupSize

                let hasScale = (scale != nil)
                let hasBias = (bias != nil)
                let isMXFP8 = hasScale && (scale!.dtype.contains("U8") || scale!.dtype.contains("UINT8") || (!hasBias && (w.dtype.contains("U32") || w.dtype.contains("U8") || w.dtype.contains("FP8")) && (scale!.offsetEnd - scale!.offsetStart) >= UInt64((inDim / 32) * outDim)))
                let isQuantizedAffine = (hasBias || w.dtype.contains("Q4")) && !isMXFP8
                let isFP8 = !isMXFP8 && !isQuantizedAffine && !w.dtype.contains("BF16") && !w.dtype.contains("F16") && !w.dtype.contains("FLOAT")

                if isMXFP8, let sRaw = (scale != nil) ? buffers[scale!.shardIndex] : nil {
                    var sOff = scale!.offsetStart
                    if let simdPipe = mxfp8GemvSimdPipeline {
                        enc.setComputePipelineState(simdPipe)
                        enc.setBuffer(wRaw, offset: 0, index: 0)
                        enc.setBuffer(inBuf, offset: inOffset, index: 1)
                        enc.setBuffer(outBuf, offset: outOffset, index: 2)
                        enc.setBuffer(sRaw, offset: 0, index: 3)
                        enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 4)
                        enc.setBytes(&sOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&inD, length: MemoryLayout<UInt32>.stride, index: 6)
                        enc.setBytes(&outD, length: MemoryLayout<UInt32>.stride, index: 7)
                        enc.dispatchThreadgroups(MTLSize(width: Int(outDim), height: batchSize, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    } else if let mxfp8Pipe = mxfp8GemvPipeline {
                        enc.setComputePipelineState(mxfp8Pipe)
                        enc.setBuffer(wRaw, offset: 0, index: 0)
                        enc.setBuffer(inBuf, offset: inOffset, index: 1)
                        enc.setBuffer(outBuf, offset: outOffset, index: 2)
                        enc.setBuffer(sRaw, offset: 0, index: 3)
                        enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 4)
                        enc.setBytes(&sOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&inD, length: MemoryLayout<UInt32>.stride, index: 6)
                        enc.setBytes(&outD, length: MemoryLayout<UInt32>.stride, index: 7)
                        enc.dispatchThreads(MTLSize(width: Int(outDim), height: batchSize, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, mxfp8Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    }
                } else if isQuantizedAffine {
                    let sRaw = (scale != nil) ? buffers[scale!.shardIndex] : nil
                    let bRaw = (bias != nil) ? buffers[bias!.shardIndex] : nil
                    guard let sBuffer = sRaw, let bBuffer = bRaw else { return }
                    var sOff = scale!.offsetStart
                    var bOff = bias!.offsetStart

                    let is8Bit = (w.offsetEnd - w.offsetStart) >= UInt64(outDim) * UInt64(inDim)
                    if is8Bit, let q8Pipe = q8GemvPipeline {
                        enc.setComputePipelineState(q8Pipe)
                        enc.setBuffer(wRaw, offset: 0, index: 0)
                        enc.setBuffer(sBuffer, offset: 0, index: 1)
                        enc.setBuffer(bBuffer, offset: 0, index: 2)
                        enc.setBuffer(inBuf, offset: inOffset, index: 3)
                        enc.setBuffer(outBuf, offset: outOffset, index: 4)
                        enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&sOff, length: MemoryLayout<UInt64>.stride, index: 6)
                        enc.setBytes(&bOff, length: MemoryLayout<UInt64>.stride, index: 7)
                        enc.setBytes(&inD, length: MemoryLayout<UInt32>.stride, index: 8)
                        enc.setBytes(&outD, length: MemoryLayout<UInt32>.stride, index: 9)
                        enc.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 10)
                        enc.dispatchThreadgroups(MTLSize(width: Int(outDim), height: batchSize, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    } else if let q4Pipe = q4GemvPipeline {
                        enc.setComputePipelineState(q4Pipe)
                        enc.setBuffer(wRaw, offset: 0, index: 0)
                        enc.setBuffer(sBuffer, offset: 0, index: 1)
                        enc.setBuffer(bBuffer, offset: 0, index: 2)
                        enc.setBuffer(inBuf, offset: inOffset, index: 3)
                        enc.setBuffer(outBuf, offset: outOffset, index: 4)
                        enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&sOff, length: MemoryLayout<UInt64>.stride, index: 6)
                        enc.setBytes(&bOff, length: MemoryLayout<UInt64>.stride, index: 7)
                        enc.setBytes(&inD, length: MemoryLayout<UInt32>.stride, index: 8)
                        enc.setBytes(&outD, length: MemoryLayout<UInt32>.stride, index: 9)
                        enc.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 10)
                        enc.dispatchThreadgroups(MTLSize(width: Int(outDim), height: batchSize, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    }
                } else if isFP8 {
                    let sRaw = (scale != nil) ? buffers[scale!.shardIndex] : wRaw
                    var sOff = scale?.offsetStart ?? 0
                    let isBlockScale = (scale != nil) && (scale!.name.contains("scale_inv") || ((scale!.offsetEnd - scale!.offsetStart) < UInt64(outDim * 2)))
                    if isBlockScale, let bSimdPipe = fp8BlockGemvSimdPipeline ?? fp8BlockGemvPipeline {
                        enc.setComputePipelineState(bSimdPipe)
                        enc.setBuffer(wRaw, offset: 0, index: 0)
                        enc.setBuffer(inBuf, offset: inOffset, index: 1)
                        enc.setBuffer(outBuf, offset: outOffset, index: 2)
                        enc.setBuffer(sRaw, offset: 0, index: 3)
                        enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 4)
                        enc.setBytes(&sOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&inD, length: MemoryLayout<UInt32>.stride, index: 6)
                        enc.setBytes(&outD, length: MemoryLayout<UInt32>.stride, index: 7)
                        enc.dispatchThreadgroups(MTLSize(width: Int(outDim), height: batchSize, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    } else if batchSize > 1, let fBatchedPipe = fp8GemvBatchedPipeline {
                    // FIX #9b: weight-stationary batched FP8 GEMV for prefill (P > 1).
                    var batchVal = UInt32(batchSize)
                    enc.setComputePipelineState(fBatchedPipe)
                    enc.setBuffer(wRaw, offset: 0, index: 0)
                    enc.setBuffer(inBuf, offset: inOffset, index: 1)
                    enc.setBuffer(outBuf, offset: outOffset, index: 2)
                    enc.setBuffer(sRaw, offset: 0, index: 3)
                    enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 4)
                    enc.setBytes(&sOff, length: MemoryLayout<UInt64>.stride, index: 5)
                    enc.setBytes(&inD, length: MemoryLayout<UInt32>.stride, index: 6)
                    enc.setBytes(&outD, length: MemoryLayout<UInt32>.stride, index: 7)
                    enc.setBytes(&batchVal, length: MemoryLayout<UInt32>.stride, index: 8)
                    enc.dispatchThreadgroups(MTLSize(width: Int(outDim), height: (batchSize + 15) / 16, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                } else if let simdPipe = fp8GemvSimdPipeline {
                        enc.setComputePipelineState(simdPipe)
                        enc.setBuffer(wRaw, offset: 0, index: 0)
                        enc.setBuffer(inBuf, offset: inOffset, index: 1)
                        enc.setBuffer(outBuf, offset: outOffset, index: 2)
                        enc.setBuffer(sRaw, offset: 0, index: 3)
                        enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 4)
                        enc.setBytes(&sOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&inD, length: MemoryLayout<UInt32>.stride, index: 6)
                        enc.setBytes(&outD, length: MemoryLayout<UInt32>.stride, index: 7)
                        enc.dispatchThreadgroups(MTLSize(width: Int(outDim), height: batchSize, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    } else if let fp8Pipe = fp8GemvPipeline {
                        enc.setComputePipelineState(fp8Pipe)
                        enc.setBuffer(wRaw, offset: 0, index: 0)
                        enc.setBuffer(inBuf, offset: inOffset, index: 1)
                        enc.setBuffer(outBuf, offset: outOffset, index: 2)
                        enc.setBuffer(sRaw, offset: 0, index: 3)
                        enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 4)
                        enc.setBytes(&sOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&inD, length: MemoryLayout<UInt32>.stride, index: 6)
                        enc.setBytes(&outD, length: MemoryLayout<UInt32>.stride, index: 7)
                        enc.dispatchThreads(MTLSize(width: Int(outDim), height: batchSize, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(outDim), fp8Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    }
                } else if batchSize > 1, let bBatchedPipe = bf16GemvBatchedPipeline {
                    // FIX #9: weight-stationary batched GEMV for prefill (P > 1) —
                    // weight traffic ceil(P/16) instead of P; verified T18b 1.6-2x.
                    var batchVal = UInt32(batchSize)
                    enc.setComputePipelineState(bBatchedPipe)
                    enc.setBuffer(wRaw, offset: 0, index: 0)
                    enc.setBuffer(inBuf, offset: inOffset, index: 1)
                    enc.setBuffer(outBuf, offset: outOffset, index: 2)
                    enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 3)
                    enc.setBytes(&inD, length: MemoryLayout<UInt32>.stride, index: 4)
                    enc.setBytes(&outD, length: MemoryLayout<UInt32>.stride, index: 5)
                    enc.setBytes(&batchVal, length: MemoryLayout<UInt32>.stride, index: 6)
                    enc.dispatchThreadgroups(MTLSize(width: Int(outDim), height: (batchSize + 15) / 16, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                } else if let bSimdPipe = bf16GemvSimdPipeline {
                    enc.setComputePipelineState(bSimdPipe)
                    enc.setBuffer(wRaw, offset: 0, index: 0)
                    enc.setBuffer(inBuf, offset: inOffset, index: 1)
                    enc.setBuffer(outBuf, offset: outOffset, index: 2)
                    enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 3)
                    enc.setBytes(&inD, length: MemoryLayout<UInt32>.stride, index: 4)
                    enc.setBytes(&outD, length: MemoryLayout<UInt32>.stride, index: 5)
                    enc.dispatchThreadgroups(MTLSize(width: Int(outDim), height: batchSize, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                } else {
                    enc.setComputePipelineState(gemvBF16Pipeline)
                    enc.setBuffer(wRaw, offset: 0, index: 0)
                    enc.setBuffer(inBuf, offset: inOffset, index: 1)
                    enc.setBuffer(outBuf, offset: outOffset, index: 2)
                    enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 3)
                    enc.setBytes(&inD, length: MemoryLayout<UInt32>.stride, index: 4)
                    enc.setBytes(&outD, length: MemoryLayout<UInt32>.stride, index: 5)
                    enc.dispatchThreads(MTLSize(width: Int(outDim), height: batchSize, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(outDim), gemvBF16Pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                }
                enc.memoryBarrier(scope: .buffers)
            }

            // Helper for MoE Expert MLP (SwiGLU Gate/Up + Down Proj Accumulate)
            func dispatchExpertMlp(
                enc: MTLComputeCommandEncoder,
                gateW: TensorMetadata,
                gateS: TensorMetadata?,
                gateB: TensorMetadata?,
                upW: TensorMetadata,
                upS: TensorMetadata?,
                upB: TensorMetadata?,
                downW: TensorMetadata,
                downS: TensorMetadata?,
                downB: TensorMetadata?,
                inBuf: MTLBuffer,
                interBuf: MTLBuffer,
                accumBuf: MTLBuffer,
                inDim: UInt32,
                interDim: UInt32,
                routingWeight: Float,
                groupSize: UInt32 = 64,
                inOffset: Int = 0,
                interOffset: Int = 0,
                accumOffset: Int = 0
            ) {
                guard let gRaw = buffers[gateW.shardIndex],
                      let uRaw = buffers[upW.shardIndex],
                      let dRaw = buffers[downW.shardIndex] else { return }

                var gWOff = gateW.offsetStart
                var uWOff = upW.offsetStart
                var dWOff = downW.offsetStart
                var hDimVal = inDim
                var interDimVal = interDim
                var p_k = routingWeight
                var grp = groupSize

                let hasGateScale = (gateS != nil)
                let hasGateBias = (gateB != nil)
                let isMXFP8 = hasGateScale && (gateS!.dtype.contains("U8") || gateS!.dtype.contains("UINT8") || (!hasGateBias && (gateW.dtype.contains("U32") || gateW.dtype.contains("U8") || gateW.dtype.contains("FP8")) && (gateS!.offsetEnd - gateS!.offsetStart) >= UInt64((inDim / 32) * interDim)))
                let isQuantizedAffine = (hasGateBias || gateW.dtype.contains("Q4") || gateW.dtype.contains("Q8")) && !isMXFP8
                let isFP8 = !isMXFP8 && !isQuantizedAffine && !gateW.dtype.contains("BF16") && !gateW.dtype.contains("F16") && !gateW.dtype.contains("FLOAT")

                if isMXFP8 {
                    guard let gS = gateS, let gSRaw = buffers[gS.shardIndex],
                          let uS = upS, let uSRaw = buffers[uS.shardIndex],
                          let dS = downS, let dSRaw = buffers[dS.shardIndex] else { return }

                    var gSOff = gS.offsetStart
                    var uSOff = uS.offsetStart
                    var dSOff = dS.offsetStart

                    if let gateSimd = mxfp8GateUpSimdPipeline, let downSimd = mxfp8DownSimdPipeline {
                        enc.setComputePipelineState(gateSimd)
                        enc.setBuffer(gRaw, offset: 0, index: 0)
                        enc.setBuffer(uRaw, offset: 0, index: 1)
                        enc.setBuffer(inBuf, offset: inOffset, index: 2)
                        enc.setBuffer(interBuf, offset: interOffset, index: 3)
                        enc.setBuffer(gSRaw, offset: 0, index: 4)
                        enc.setBuffer(uSRaw, offset: 0, index: 5)
                        enc.setBytes(&gWOff, length: MemoryLayout<UInt64>.stride, index: 6)
                        enc.setBytes(&gSOff, length: MemoryLayout<UInt64>.stride, index: 7)
                        enc.setBytes(&uWOff, length: MemoryLayout<UInt64>.stride, index: 8)
                        enc.setBytes(&uSOff, length: MemoryLayout<UInt64>.stride, index: 9)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 10)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 11)
                        enc.dispatchThreadgroups(MTLSize(width: Int(interDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)

                        enc.setComputePipelineState(downSimd)
                        enc.setBuffer(dRaw, offset: 0, index: 0)
                        enc.setBuffer(interBuf, offset: interOffset, index: 1)
                        enc.setBuffer(accumBuf, offset: accumOffset, index: 2)
                        enc.setBuffer(dSRaw, offset: 0, index: 3)
                        enc.setBytes(&dWOff, length: MemoryLayout<UInt64>.stride, index: 4)
                        enc.setBytes(&dSOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 6)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 7)
                        enc.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 8)
                        enc.dispatchThreadgroups(MTLSize(width: Int(inDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)
                    } else if let mxfp8GatePipe = mxfp8GateUpPipeline, let mxfp8DownPipe = mxfp8DownPipeline {
                        enc.setComputePipelineState(mxfp8GatePipe)
                        enc.setBuffer(gRaw, offset: 0, index: 0)
                        enc.setBuffer(uRaw, offset: 0, index: 1)
                        enc.setBuffer(inBuf, offset: inOffset, index: 2)
                        enc.setBuffer(interBuf, offset: interOffset, index: 3)
                        enc.setBuffer(gSRaw, offset: 0, index: 4)
                        enc.setBuffer(uSRaw, offset: 0, index: 5)
                        enc.setBytes(&gWOff, length: MemoryLayout<UInt64>.stride, index: 6)
                        enc.setBytes(&gSOff, length: MemoryLayout<UInt64>.stride, index: 7)
                        enc.setBytes(&uWOff, length: MemoryLayout<UInt64>.stride, index: 8)
                        enc.setBytes(&uSOff, length: MemoryLayout<UInt64>.stride, index: 9)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 10)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 11)
                        enc.dispatchThreads(MTLSize(width: Int(interDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, mxfp8GatePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)

                        enc.setComputePipelineState(mxfp8DownPipe)
                        enc.setBuffer(dRaw, offset: 0, index: 0)
                        enc.setBuffer(interBuf, offset: interOffset, index: 1)
                        enc.setBuffer(accumBuf, offset: accumOffset, index: 2)
                        enc.setBuffer(dSRaw, offset: 0, index: 3)
                        enc.setBytes(&dWOff, length: MemoryLayout<UInt64>.stride, index: 4)
                        enc.setBytes(&dSOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 6)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 7)
                        enc.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 8)
                        enc.dispatchThreads(MTLSize(width: Int(inDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, mxfp8DownPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)
                    }
                } else if isQuantizedAffine {
                    guard let gS = gateS, let gSRaw = buffers[gS.shardIndex],
                          let gB = gateB, let gBRaw = buffers[gB.shardIndex],
                          let uS = upS, let uSRaw = buffers[uS.shardIndex],
                          let uB = upB, let uBRaw = buffers[uB.shardIndex],
                          let dS = downS, let dSRaw = buffers[dS.shardIndex],
                          let dB = downB, let dBRaw = buffers[dB.shardIndex] else { return }

                    var gSOff = gS.offsetStart
                    var gBOff = gB.offsetStart
                    var uSOff = uS.offsetStart
                    var uBOff = uB.offsetStart
                    var dSOff = dS.offsetStart
                    var dBOff = dB.offsetStart

                    let isGate8Bit = (gateW.offsetEnd - gateW.offsetStart) >= UInt64(interDim) * UInt64(inDim)
                    let isUp8Bit = (upW.offsetEnd - upW.offsetStart) >= UInt64(interDim) * UInt64(inDim)
                    let isDown8Bit = (downW.offsetEnd - downW.offsetStart) >= UInt64(inDim) * UInt64(interDim)

                    let swigluPipe: MTLComputePipelineState?
                    if isGate8Bit && isUp8Bit {
                        swigluPipe = q8GateUpPipeline
                    } else if !isGate8Bit && !isUp8Bit {
                        swigluPipe = q4GateUpPipeline
                    } else if !isGate8Bit && isUp8Bit {
                        swigluPipe = q4GateQ8UpPipeline
                    } else {
                        swigluPipe = q8GateQ4UpPipeline
                    }

                    let downPipe: MTLComputePipelineState?
                    if isDown8Bit {
                        downPipe = q8DownPipeline
                    } else {
                        downPipe = q4DownPipeline
                    }

                    if let sPipe = swigluPipe, let dPipe = downPipe {
                        // Step 1: SwiGLU Gate & Up Proj
                        enc.setComputePipelineState(sPipe)
                        enc.setBuffer(gRaw, offset: 0, index: 0)
                        enc.setBuffer(gSRaw, offset: 0, index: 1)
                        enc.setBuffer(gBRaw, offset: 0, index: 2)
                        enc.setBuffer(uRaw, offset: 0, index: 3)
                        enc.setBuffer(uSRaw, offset: 0, index: 4)
                        enc.setBuffer(uBRaw, offset: 0, index: 5)
                        enc.setBuffer(inBuf, offset: inOffset, index: 6)
                        enc.setBuffer(interBuf, offset: interOffset, index: 7)
                        enc.setBytes(&gWOff, length: MemoryLayout<UInt64>.stride, index: 8)
                        enc.setBytes(&gSOff, length: MemoryLayout<UInt64>.stride, index: 9)
                        enc.setBytes(&gBOff, length: MemoryLayout<UInt64>.stride, index: 10)
                        enc.setBytes(&uWOff, length: MemoryLayout<UInt64>.stride, index: 11)
                        enc.setBytes(&uSOff, length: MemoryLayout<UInt64>.stride, index: 12)
                        enc.setBytes(&uBOff, length: MemoryLayout<UInt64>.stride, index: 13)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 14)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 15)
                        enc.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 16)
                        enc.dispatchThreadgroups(MTLSize(width: Int(interDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)

                        // Step 2: Down Proj Accumulate
                        enc.setComputePipelineState(dPipe)
                        enc.setBuffer(dRaw, offset: 0, index: 0)
                        enc.setBuffer(dSRaw, offset: 0, index: 1)
                        enc.setBuffer(dBRaw, offset: 0, index: 2)
                        enc.setBuffer(interBuf, offset: interOffset, index: 3)
                        enc.setBuffer(accumBuf, offset: accumOffset, index: 4)
                        enc.setBytes(&dWOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&dSOff, length: MemoryLayout<UInt64>.stride, index: 6)
                        enc.setBytes(&dBOff, length: MemoryLayout<UInt64>.stride, index: 7)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 8)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 9)
                        enc.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 10)
                        enc.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 11)
                        enc.dispatchThreadgroups(MTLSize(width: Int(inDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)
                    }
                } else if isFP8 {
                    guard let gS = gateS, let gSRaw = buffers[gS.shardIndex],
                          let uS = upS, let uSRaw = buffers[uS.shardIndex],
                          let dS = downS, let dSRaw = buffers[dS.shardIndex] else { return }

                    var gSOff = gS.offsetStart
                    var uSOff = uS.offsetStart
                    var dSOff = dS.offsetStart

                    let isBlockScale = ((gS.offsetEnd - gS.offsetStart) < UInt64(interDim * 2)) || gS.name.contains("scale_inv")
                    let gatePipe = isBlockScale ? (fp8BlockGateUpSimdPipeline ?? fp8GateUpSimdPipeline) : fp8GateUpSimdPipeline
                    let downPipe = isBlockScale ? (fp8BlockDownSimdPipeline ?? fp8DownSimdPipeline) : fp8DownSimdPipeline

                    if let fp8GateSimd = gatePipe, let fp8DownSimd = downPipe {
                        enc.setComputePipelineState(fp8GateSimd)
                        enc.setBuffer(gRaw, offset: 0, index: 0)
                        enc.setBuffer(uRaw, offset: 0, index: 1)
                        enc.setBuffer(inBuf, offset: inOffset, index: 2)
                        enc.setBuffer(interBuf, offset: interOffset, index: 3)
                        enc.setBuffer(gSRaw, offset: 0, index: 4)
                        enc.setBuffer(uSRaw, offset: 0, index: 5)
                        enc.setBytes(&gWOff, length: MemoryLayout<UInt64>.stride, index: 6)
                        enc.setBytes(&gSOff, length: MemoryLayout<UInt64>.stride, index: 7)
                        enc.setBytes(&uWOff, length: MemoryLayout<UInt64>.stride, index: 8)
                        enc.setBytes(&uSOff, length: MemoryLayout<UInt64>.stride, index: 9)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 10)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 11)
                        enc.dispatchThreadgroups(MTLSize(width: Int(interDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)

                        enc.setComputePipelineState(fp8DownSimd)
                        enc.setBuffer(dRaw, offset: 0, index: 0)
                        enc.setBuffer(interBuf, offset: interOffset, index: 1)
                        enc.setBuffer(accumBuf, offset: accumOffset, index: 2)
                        enc.setBuffer(dSRaw, offset: 0, index: 3)
                        enc.setBytes(&dWOff, length: MemoryLayout<UInt64>.stride, index: 4)
                        enc.setBytes(&dSOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 6)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 7)
                        enc.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 8)
                        enc.dispatchThreadgroups(MTLSize(width: Int(inDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)
                    } else if let fp8GatePipe = fp8GateUpPipeline, let fp8DownPipe = fp8DownPipeline {
                        enc.setComputePipelineState(fp8GatePipe)
                        enc.setBuffer(gRaw, offset: 0, index: 0)
                        enc.setBuffer(uRaw, offset: 0, index: 1)
                        enc.setBuffer(inBuf, offset: inOffset, index: 2)
                        enc.setBuffer(interBuf, offset: interOffset, index: 3)
                        enc.setBuffer(gSRaw, offset: 0, index: 4)
                        enc.setBuffer(uSRaw, offset: 0, index: 5)
                        enc.setBytes(&gWOff, length: MemoryLayout<UInt64>.stride, index: 6)
                        enc.setBytes(&gSOff, length: MemoryLayout<UInt64>.stride, index: 7)
                        enc.setBytes(&uWOff, length: MemoryLayout<UInt64>.stride, index: 8)
                        enc.setBytes(&uSOff, length: MemoryLayout<UInt64>.stride, index: 9)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 10)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 11)
                        enc.dispatchThreads(MTLSize(width: Int(interDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, fp8GatePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)

                        enc.setComputePipelineState(fp8DownPipe)
                        enc.setBuffer(dRaw, offset: 0, index: 0)
                        enc.setBuffer(interBuf, offset: interOffset, index: 1)
                        enc.setBuffer(accumBuf, offset: accumOffset, index: 2)
                        enc.setBuffer(dSRaw, offset: 0, index: 3)
                        enc.setBytes(&dWOff, length: MemoryLayout<UInt64>.stride, index: 4)
                        enc.setBytes(&dSOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 6)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 7)
                        enc.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 8)
                        enc.dispatchThreads(MTLSize(width: Int(inDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, fp8DownPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)
                    }
                } else {
                    // BF16
                    if let bGateSimd = bf16GateUpSimdPipeline, let bDownSimd = bf16DownSimdPipeline {
                        enc.setComputePipelineState(bGateSimd)
                        enc.setBuffer(gRaw, offset: 0, index: 0)
                        enc.setBuffer(uRaw, offset: 0, index: 1)
                        enc.setBuffer(inBuf, offset: inOffset, index: 2)
                        enc.setBuffer(interBuf, offset: interOffset, index: 3)
                        enc.setBytes(&gWOff, length: MemoryLayout<UInt64>.stride, index: 4)
                        enc.setBytes(&uWOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 6)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 7)
                        enc.dispatchThreadgroups(MTLSize(width: Int(interDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)

                        enc.setComputePipelineState(bDownSimd)
                        enc.setBuffer(dRaw, offset: 0, index: 0)
                        enc.setBuffer(interBuf, offset: interOffset, index: 1)
                        enc.setBuffer(accumBuf, offset: accumOffset, index: 2)
                        enc.setBytes(&dWOff, length: MemoryLayout<UInt64>.stride, index: 3)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 4)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 5)
                        enc.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 6)
                        enc.dispatchThreadgroups(MTLSize(width: Int(inDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)
                    } else if let bGatePipe = bf16GateUpPipeline, let bDownPipe = bf16DownPipeline {
                        enc.setComputePipelineState(bGatePipe)
                        enc.setBuffer(gRaw, offset: 0, index: 0)
                        enc.setBuffer(uRaw, offset: 0, index: 1)
                        enc.setBuffer(inBuf, offset: inOffset, index: 2)
                        enc.setBuffer(interBuf, offset: interOffset, index: 3)
                        enc.setBytes(&gWOff, length: MemoryLayout<UInt64>.stride, index: 4)
                        enc.setBytes(&uWOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 6)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 7)
                        enc.dispatchThreads(MTLSize(width: Int(interDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(interDim), bGatePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)

                        enc.setComputePipelineState(bDownPipe)
                        enc.setBuffer(dRaw, offset: 0, index: 0)
                        enc.setBuffer(interBuf, offset: interOffset, index: 1)
                        enc.setBuffer(accumBuf, offset: accumOffset, index: 2)
                        enc.setBytes(&dWOff, length: MemoryLayout<UInt64>.stride, index: 3)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 4)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 5)
                        enc.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 6)
                        enc.dispatchThreads(MTLSize(width: Int(inDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(inDim), bDownPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)
                    }
                }
            }

            // Helper for Single Token Forward Pass
            var previousLayerActiveExperts: [Int: [Int]] = [:]
            // 2-deep per-layer expert history for the temporal predictor:
            // layerExpertHistory[l][0] = previous token's set, [1] = the one before.
            var layerExpertHistory: [Int: [[Int]]] = [:]
            // Pipelined-expert adaptive gate: persistent across tokens; disables the
            // prefetcher for the rest of the generation if the predictor proves useless
            // (<20% hit rate over 64 consumed experts), since wasted prefetch then
            // competes with critical-path miss IO for NVMe. Judgement is deferred until
            // temporal history exists (token >= 2): the cold-start token legitimately
            // produces 0% and must never latch the gate off.
            var packedPrefetchAdaptiveEnabled = speculativePrefetchEnabled && packedExpertsDir != nil
            var packedPrefetchObservedHits = 0
            var packedPrefetchObservedTotal = 0
            func runTokenForward(tokenId: UInt32, step: UInt32, computeLogits: Bool, wait: Bool = true) -> Bool {
                if Int(step) >= KVCacheManager.shared.allocatedSeqLen {
                    print("❌ [KVCACHE] step \(step) exceeds the provisioned row length \(KVCacheManager.shared.allocatedSeqLen) — refusing forward (KV precision \(KVCacheManager.shared.activePrecision))")
                    return false
                }
                let tTokenStart = CFAbsoluteTimeGetCurrent()
                let hadTemporalHistory = !previousLayerActiveExperts.isEmpty
                var diagLayerRouterMs: Double = 0
                var diagRouterGpuMs: Double = 0
                var diagIoMs: Double = 0
                var diagMoeGpuMs: Double = 0
                var diagOtherMs: Double = 0
                var diagPrefetchHits: Int = 0
                var diagPrefetchTotal: Int = 0
                let singleTokenPtr = singleTokenBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)
                singleTokenPtr[0] = tokenId
                var hDim = hiddenDim
                let isGatedQ: Bool = {
                    if let configGate = modelConfig?.effectiveAttnOutputGate {
                        return configGate
                    }
                    if let qTensor = cachedLayers.first(where: { $0.attentionType == .fullAttention })?.qProjTensor {
                        let gatedDim = numHeads * headDim * 2
                        return qTensor.shapeDisplay.contains("\(gatedDim)")
                    }
                    return false
                }()
                let isStandardGqa = !isGatedQ

                // 2. Multi-Layer Transformer Backbone (0..<actualLayers, looped totalLoops times)
                var currentH = hCurrBuffer
                var nextH = hNextBuffer

                guard var activeCmd = commandQueue.makeCommandBuffer() else {
                    print("❌ [METAL] token forward: command buffer allocation failed (step=\(step))")
                    return false
                }
                // Pipelined packed-expert streaming state (per token, packed decode only).
                // After layer l's router resolves, the l+1 predicted experts are pread into the
                // OTHER staging buffer while layer l's MoE MLP runs on the GPU; layer l's own
                // experts are consumed from prefetched slots, paying synchronous IO only for
                // predictor misses.
                let packedExpertSize = Int(loadedLayout?.expert_size ?? 1769472)
                let maxTopK = Int(modelConfig?.effectiveNumExpertsPerTok ?? (numExperts >= 512 ? 10 : 8))
                let packedPrefetchLock = NSLock()

                // FIX #8: per-slot completion tracking. A kick publishes its candidate
                // list and per-slot semaphores synchronously; the consuming layer waits
                // ONLY for the slots it needs (signaled by the pool as each pread
                // completes), while false-positive reads run to completion in the
                // background. A kick's slot space is reclaimed at the next kick into
                // the same staging buffer (2 layers later), which first waits for ALL
                // of the old kick's reads — that retire-wait overlaps the current
                // layer's MoE GPU command.
                final class PackedExpertKick {
                    let targetLayer: Int
                    let bufIdx: Int
                    let buffer: MTLBuffer
                    let slotBase: Int               // first slot available for consume-side misses
                    let candidates: [Int]           // slot i holds candidates[i]
                    let slotSems: [DispatchSemaphore]
                    let allDoneSem = DispatchSemaphore(value: 0)
                    var slotOK: [Bool]              // guarded by packedPrefetchLock

                    init(targetLayer: Int, bufIdx: Int, buffer: MTLBuffer, slotBase: Int, candidates: [Int]) {
                        self.targetLayer = targetLayer
                        self.bufIdx = bufIdx
                        self.buffer = buffer
                        self.slotBase = slotBase
                        self.candidates = candidates
                        self.slotSems = (0..<candidates.count).map { _ in DispatchSemaphore(value: 0) }
                        self.slotOK = Array(repeating: false, count: candidates.count)
                    }
                }

                var packedKickInFlight: [PackedExpertKick?] = [nil, nil]
                let packedPrefetchQueue = DispatchQueue(label: "com.dynamoe.packed.expertprefetch", qos: .userInitiated, attributes: .concurrent)

                // Kicks the background pread of predicted experts for `targetLayer` into
                // the opposite staging buffer. Safe while the GPU reads the current
                // buffer: the prefetch writes only buffer (targetLayer & 1), whose
                // previous MoE pass completed before this layer's router resolved.
                // Must be called AFTER the current layer's MoE command is committed so
                // the previous-same-parity-kick retire wait overlaps GPU execution.
                func kickPackedPrefetch(_ predicted: [Int], targetLayer: Int, fd: Int32) {
                    guard !predicted.isEmpty else { return }
                    let targetBuf = targetLayer & 1
                    let stagingBuf = (targetBuf == 0) ? expertStagingBuffer : expertStagingBufferB
                    let capacity = expertStagingSize / packedExpertSize
                    let slots = min(predicted.count, 10, max(0, capacity - maxTopK))
                    guard slots > 0 else { return }
                    let kicked = Array(predicted.prefix(slots))
                    let kick = PackedExpertKick(targetLayer: targetLayer, bufIdx: targetBuf, buffer: stagingBuf, slotBase: slots, candidates: kicked)

                    packedPrefetchLock.lock()
                    let old = packedKickInFlight[targetBuf]
                    packedKickInFlight[targetBuf] = kick
                    packedPrefetchLock.unlock()
                    // Retire the previous kick that owned this buffer's slot space
                    // before its bytes can be overwritten (wait overlaps GPU work).
                    if let old = old { _ = old.allDoneSem.wait(timeout: .now() + 30.0) }

                    packedPrefetchQueue.async {
                        var tasks: [ExpertPreadTask] = []
                        for (slotIdx, expId) in kicked.enumerated() {
                            tasks.append(ExpertPreadTask(fd: fd,
                                                         dst: stagingBuf.contents().advanced(by: slotIdx * packedExpertSize),
                                                         offset: off_t(expId * packedExpertSize),
                                                         size: packedExpertSize))
                        }
                        ExpertIOThreadPool.shared.dispatchAsync(tasks: tasks, onTaskDone: { idx in
                            packedPrefetchLock.lock()
                            kick.slotOK[idx] = true
                            packedPrefetchLock.unlock()
                            kick.slotSems[idx].signal()
                        }, done: kick.allDoneSem)
                    }
                }

                for loopIdx in 0..<totalLoops {
                    for l in 0..<actualLayers {
                        if Task.isCancelled { return false }
                        let layer = cachedLayers[l]

                        // Asynchronous Layer Lookahead Backbone Prefetching
                        // (mmap models only — FlashMoE backbone is pinned in RAM at load)
                        if speculativePrefetchEnabled && packedExpertsDir == nil {
                            let nextL = (l + 1) < actualLayers ? (l + 1) : 0
                            WorkingSetManager.shared.prefetchLayerBackbone(layer: cachedLayers[nextL], shardBuffers: buffers)
                            if prefetchLookaheadDepth >= 2 {
                                let nextNextL = (l + 2) < actualLayers ? (l + 2) : ((l + 2) % actualLayers)
                                WorkingSetManager.shared.prefetchLayerBackbone(layer: cachedLayers[nextNextL], shardBuffers: buffers)
                            }
                        }

                        // --- Phase A: Attention & Routing Sub-Block ---
                        guard let layerEnc1 = activeCmd.makeComputeCommandEncoder() else {
                            print("❌ [METAL] token forward: Phase A encoder allocation failed (layer \(l), step \(step))")
                            return false
                        }

                        // Step 0: Embed Token on Loop 0, Layer 0
                        if loopIdx == 0 && l == 0 {
                            let hasEmbedScale = (embedScale != nil)
                            let hasEmbedBias = (embedBias != nil)
                            let isEmbedMXFP8 = hasEmbedScale && (embedScale!.dtype.contains("U8") || embedScale!.dtype.contains("UINT8") || (!hasEmbedBias && (embedWeight.dtype.contains("U32") || embedWeight.dtype.contains("U8") || embedWeight.dtype.contains("FP8")) && (embedScale!.offsetEnd - embedScale!.offsetStart) >= UInt64(hiddenDim / 32)))
                            let isEmbedAffine = (hasEmbedBias || embedWeight.dtype.contains("Q4") || embedWeight.dtype.contains("Q8") || hasEmbedScale) && !isEmbedMXFP8

                            if isEmbedMXFP8, let embedMXFP8Pipe = embedMXFP8Pipeline,
                               let embedScaleRaw = buffers[embedScale!.shardIndex] {
                                var wOffset = embedOffset
                                var sOffset = embedScale!.offsetStart
                                var hDimVal = hiddenDim
                                layerEnc1.setComputePipelineState(embedMXFP8Pipe)
                                layerEnc1.setBuffer(embedShardBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(singleTokenBuffer, offset: 0, index: 1)
                                layerEnc1.setBuffer(currentH, offset: 0, index: 2)
                                layerEnc1.setBuffer(embedScaleRaw, offset: 0, index: 3)
                                layerEnc1.setBytes(&wOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                                layerEnc1.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                                layerEnc1.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 6)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), embedMXFP8Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            } else if isEmbedAffine,
                               let embedScaleRaw = (embedScale != nil) ? buffers[embedScale!.shardIndex] : nil,
                               let embedBiasRaw = (embedBias != nil) ? buffers[embedBias!.shardIndex] : nil {
                                var wOffset = embedOffset
                                var sOffset = embedScale!.offsetStart
                                var bOffset = embedBias!.offsetStart
                                var tok = tokenId
                                var grpSize: UInt32 = 64
                                let is8Bit = (embedWeight.offsetEnd - embedWeight.offsetStart) >= (UInt64(vocabSize) * UInt64(hiddenDim) * 3) / 4
                                if is8Bit, let embedQ8Pipe = embedQ8Pipeline {
                                    layerEnc1.setComputePipelineState(embedQ8Pipe)
                                    layerEnc1.setBuffer(embedShardBuffer, offset: 0, index: 0)
                                    layerEnc1.setBuffer(embedScaleRaw, offset: 0, index: 1)
                                    layerEnc1.setBuffer(embedBiasRaw, offset: 0, index: 2)
                                    layerEnc1.setBuffer(currentH, offset: 0, index: 3)
                                    layerEnc1.setBytes(&tok, length: MemoryLayout<UInt32>.stride, index: 4)
                                    layerEnc1.setBytes(&wOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                                    layerEnc1.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                                    layerEnc1.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                                    layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 8)
                                    layerEnc1.setBytes(&grpSize, length: MemoryLayout<UInt32>.stride, index: 9)
                                    layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), embedQ8Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                } else if let embedQ4Pipe = embedQ4Pipeline {
                                    layerEnc1.setComputePipelineState(embedQ4Pipe)
                                    layerEnc1.setBuffer(embedShardBuffer, offset: 0, index: 0)
                                    layerEnc1.setBuffer(embedScaleRaw, offset: 0, index: 1)
                                    layerEnc1.setBuffer(embedBiasRaw, offset: 0, index: 2)
                                    layerEnc1.setBuffer(currentH, offset: 0, index: 3)
                                    layerEnc1.setBytes(&tok, length: MemoryLayout<UInt32>.stride, index: 4)
                                    layerEnc1.setBytes(&wOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                                    layerEnc1.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                                    layerEnc1.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                                    layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 8)
                                    layerEnc1.setBytes(&grpSize, length: MemoryLayout<UInt32>.stride, index: 9)
                                    layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), embedQ4Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                }
                            } else {
                                var wOffset = embedOffset
                                var tokCount: UInt32 = 1
                                layerEnc1.setComputePipelineState(embedPipeline)
                                layerEnc1.setBuffer(embedShardBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(singleTokenBuffer, offset: 0, index: 1)
                                layerEnc1.setBuffer(currentH, offset: 0, index: 2)
                                layerEnc1.setBytes(&wOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                                layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&tokCount, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), embedPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }
                            if isGemma4, let scalePipe = scaleVectorInplacePipeline {
                                // Gemma 4 scales token embeddings by sqrt(hidden_size).
                                var dimV = hiddenDim
                                var sc = Float(Double(hiddenDim).squareRoot())
                                layerEnc1.setComputePipelineState(scalePipe)
                                layerEnc1.setBuffer(currentH, offset: 0, index: 0)
                                layerEnc1.setBytes(&dimV, length: MemoryLayout<UInt32>.stride, index: 1)
                                layerEnc1.setBytes(&sc, length: MemoryLayout<Float>.stride, index: 2)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), scalePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }
                            if (cachedLayers.first?.attnHcDownWeight != nil), let initStreams = fusedInit4StreamsPipeline {
                                layerEnc1.memoryBarrier(scope: .buffers)
                                layerEnc1.setComputePipelineState(initStreams)
                                layerEnc1.setBuffer(currentH, offset: 0, index: 0)
                                layerEnc1.setBuffer(hcStreamsBuffer, offset: 0, index: 1)
                                layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 2)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), initStreams.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }
                        }

                        // Step 1: Pre-Attention RMSNorm or Hyper-Connection Mixing (currentH / hcStreams -> xNorm1)
                        if let hcDown = layer.attnHcDownWeight, let hcDownRaw = buffers[hcDown.shardIndex],
                           let hcUp = layer.attnHcUpWeight, let hcUpRaw = buffers[hcUp.shardIndex],
                           let hcNorm = layer.attnHcNorm, let hcNormRaw = buffers[hcNorm.shardIndex],
                           let normPipe = hcNormPipeline, let downPipe = hcDownProjPipeline, let upPipe = hcUpBlendPipeline {
                            var nOff = hcNorm.offsetStart
                            var dOff = hcDown.offsetStart
                            var uOff = hcUp.offsetStart
                            var totDim: UInt32 = 4 * hiddenDim
                            var rank: UInt32 = UInt32(modelConfig?.effectiveHcLowrank ?? 320)
                            var epsVal = eps

                            // 1a. Group RMSNorm on 4 Streams
                            layerEnc1.setComputePipelineState(normPipe)
                            layerEnc1.setBuffer(hcStreamsBuffer, offset: 0, index: 0)
                            layerEnc1.setBuffer(hcNormRaw, offset: 0, index: 1)
                            layerEnc1.setBuffer(hcNormedBuffer, offset: 0, index: 2)
                            layerEnc1.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: 4, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 1b. Down-Projection
                            layerEnc1.setComputePipelineState(downPipe)
                            layerEnc1.setBuffer(hcNormedBuffer, offset: 0, index: 0)
                            layerEnc1.setBuffer(hcDownRaw, offset: 0, index: 1)
                            layerEnc1.setBuffer(hcBottleneckBuffer, offset: 0, index: 2)
                            layerEnc1.setBytes(&dOff, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc1.setBytes(&totDim, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc1.setBytes(&rank, length: MemoryLayout<UInt32>.stride, index: 5)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: Int(rank), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 1c. Up-Projection and Stream Blend
                            layerEnc1.setComputePipelineState(upPipe)
                            layerEnc1.setBuffer(hcNormedBuffer, offset: 0, index: 0)
                            layerEnc1.setBuffer(hcBottleneckBuffer, offset: 0, index: 1)
                            layerEnc1.setBuffer(hcUpRaw, offset: 0, index: 2)
                            layerEnc1.setBuffer(xNorm1Buffer, offset: 0, index: 3)
                            layerEnc1.setBytes(&uOff, length: MemoryLayout<UInt64>.stride, index: 4)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 5)
                            layerEnc1.setBytes(&rank, length: MemoryLayout<UInt32>.stride, index: 6)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 1d. Compute Injection Scale if attnHcInject exists
                            if let attnHcInj = layer.attnHcInjectWeight, let attnHcInjRaw = buffers[attnHcInj.shardIndex],
                               let injScalePipe = hcInjectScalePipeline {
                                var iOff = attnHcInj.offsetStart
                                layerEnc1.setComputePipelineState(injScalePipe)
                                layerEnc1.setBuffer(hcNormedBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(attnHcInjRaw, offset: 0, index: 1)
                                layerEnc1.setBuffer(hcInjectScaleBuffer, offset: 0, index: 2)
                                layerEnc1.setBytes(&iOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                layerEnc1.setBytes(&totDim, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.dispatchThreadgroups(MTLSize(width: 4, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }
                        } else if let norm1 = layer.norm1Tensor, let norm1Raw = buffers[norm1.shardIndex] {
                            var gammaOff = norm1.offsetStart
                            var epsVal = eps
                            let isNorm1F16 = (norm1.dtype.contains("F16") || norm1.dtype.contains("HALF") || norm1.dtype.contains("FLOAT16")) && !norm1.dtype.contains("BF16") && !norm1.dtype.contains("BFLOAT")
                            let norm1Pipe: MTLComputePipelineState
                            if isRMSNormOffset {
                                norm1Pipe = (isNorm1F16 && rmsnormOffsetF16Pipeline != nil) ? rmsnormOffsetF16Pipeline! : (rmsnormOffsetPipeline ?? rmsnormPipeline)
                            } else {
                                norm1Pipe = (isNorm1F16 && rmsnormF16Pipeline != nil) ? rmsnormF16Pipeline! : rmsnormPipeline
                            }
                            layerEnc1.setComputePipelineState(norm1Pipe)
                            layerEnc1.setBuffer(currentH, offset: 0, index: 0)
                            layerEnc1.setBuffer(norm1Raw, offset: 0, index: 1)
                            layerEnc1.setBuffer(xNorm1Buffer, offset: 0, index: 2)
                            layerEnc1.setBytes(&gammaOff, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                            layerEnc1.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)
                        }

                        // Step 2: Attention Computation
                        if layer.isMLA {
                            layerEnc1.setComputePipelineState(clearPipeline)
                            layerEnc1.setBuffer(attnOutBuffer, offset: 0, index: 0)
                            layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 1. Q projection: hiddenDim (1536) -> q_lora_rank (256) into kVectorBuffer
                            dispatchLinear(enc: layerEnc1, weight: layer.mlaQAProj, scale: nil, bias: nil, inBuf: xNorm1Buffer, outBuf: kVectorBuffer, inDim: hiddenDim, outDim: 256)
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 2. Q LayerNorm (RMSNorm 256) in-place on kVectorBuffer
                            if let qALN = layer.mlaQALayernorm, let qALNRaw = buffers[qALN.shardIndex] {
                                var qLNGammaOff = qALN.offsetStart
                                var qLNDim: UInt32 = 256
                                var epsVal = eps
                                layerEnc1.setComputePipelineState(rmsnormPipeline)
                                layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(qALNRaw, offset: 0, index: 1)
                                layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 2)
                                layerEnc1.setBytes(&qLNGammaOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                layerEnc1.setBytes(&qLNDim, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                                layerEnc1.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                                layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(qLNDim)), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            // 3. Q_B projection: 256 -> 3072 (16 heads x 192) into qGateBuffer
                            dispatchLinear(enc: layerEnc1, weight: layer.mlaQBProj, scale: nil, bias: nil, inBuf: kVectorBuffer, outBuf: qGateBuffer, inDim: 256, outDim: 3072)
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 4. Apply interleaved RoPE to Q rotary part (elements 128..191 of each 192-dim head in qGateBuffer)
                            if let ropePipe = ropeInterleavedPipeline {
                                var pos = UInt32(step)
                                var nHeads: UInt32 = 16
                                var hDimTotal: UInt32 = 192
                                var rotDim: UInt32 = 64
                                var hStride: UInt32 = 192
                                var thetaVal: Float = Float(modelConfig?.ropeTheta ?? 10000.0)
                                var qRopeOffset: UInt32 = 128
                                layerEnc1.setComputePipelineState(ropePipe)
                                layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 1)
                                layerEnc1.setBytes(&nHeads, length: MemoryLayout<UInt32>.stride, index: 2)
                                layerEnc1.setBytes(&hDimTotal, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc1.setBytes(&rotDim, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&hStride, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc1.setBytes(&thetaVal, length: MemoryLayout<Float>.stride, index: 6)
                                layerEnc1.setBytes(&qRopeOffset, length: MemoryLayout<UInt32>.stride, index: 7)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(nHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(32, Int(nHeads)), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            // 5. KV_A projection with MQA: hiddenDim (1536) -> 576 (512 compressed KV + 64 k_rope) into kVectorBuffer
                            dispatchLinear(enc: layerEnc1, weight: layer.mlaKVAProjWithMqa, scale: nil, bias: nil, inBuf: xNorm1Buffer, outBuf: kVectorBuffer, inDim: hiddenDim, outDim: 576)
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 6. Apply interleaved RoPE to K rotary part (elements 512..575 of kVectorBuffer)
                            if let ropePipe = ropeInterleavedPipeline {
                                var pos = UInt32(step)
                                var nHeads: UInt32 = 1
                                var hDimTotal: UInt32 = 64
                                var rotDim: UInt32 = 64
                                var hStride: UInt32 = 64
                                var thetaVal: Float = Float(modelConfig?.ropeTheta ?? 10000.0)
                                var kRopeOffset: UInt32 = 512
                                layerEnc1.setComputePipelineState(ropePipe)
                                layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 0)
                                layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 1)
                                layerEnc1.setBytes(&nHeads, length: MemoryLayout<UInt32>.stride, index: 2)
                                layerEnc1.setBytes(&hDimTotal, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc1.setBytes(&rotDim, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&hStride, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc1.setBytes(&thetaVal, length: MemoryLayout<Float>.stride, index: 6)
                                layerEnc1.setBytes(&kRopeOffset, length: MemoryLayout<UInt32>.stride, index: 7)
                                layerEnc1.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            // 7. KV LayerNorm (RMSNorm 512) on first 512 elements of kVectorBuffer into zGateBuffer
                            if let kvALN = layer.mlaKVALayernorm, let kvALNRaw = buffers[kvALN.shardIndex] {
                                var kvLNGammaOff = kvALN.offsetStart
                                var kvLNDim: UInt32 = 512
                                var epsVal = eps
                                layerEnc1.setComputePipelineState(rmsnormPipeline)
                                layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(kvALNRaw, offset: 0, index: 1)
                                layerEnc1.setBuffer(zGateBuffer, offset: 0, index: 2)
                                layerEnc1.setBytes(&kvLNGammaOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                layerEnc1.setBytes(&kvLNDim, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                                layerEnc1.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                                layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(kvLNDim)), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            // 8. KV_B projection: 512 (zGateBuffer) -> 4096 (16 heads x (128 k_nope + 128 v)) into vVectorBuffer
                            dispatchLinear(enc: layerEnc1, weight: layer.mlaKVBProj, scale: nil, bias: nil, inBuf: zGateBuffer, outBuf: vVectorBuffer, inDim: 512, outDim: 4096)
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 9. Store into MLA KV cache (F16)
                            let isLingMla = (modelConfig?.isLingModel == true)
                            let slot = isLingMla ? ((loopIdx * 6) + layer.fullAttnIndex) : ((loopIdx * actualLayers) + layer.fullAttnIndex)
                            let maxSeq = KVCacheManager.shared.allocatedSeqLen
                            let kStride = isLingMla ? 3072 : 4096
                            let vStride = isLingMla ? 2048 : 4096
                            let kLayerByteOffset = slot * maxSeq * kStride * MemoryLayout<UInt16>.stride
                            let vLayerByteOffset = slot * maxSeq * vStride * MemoryLayout<UInt16>.stride
                            if let kvLenCheck = KVCacheManager.shared.kCacheBuffer, kLayerByteOffset + maxSeq * kStride * 2 > kvLenCheck.length {
                                print("❌ [KVCACHE] MLA K-cache overflow: layer \(l) slot \(slot) offset \(kLayerByteOffset) + span \(maxSeq * kStride * 2) > buffer \(kvLenCheck.length) bytes | maxSeq=\(maxSeq) kStride=\(kStride) precision=\(KVCacheManager.shared.activePrecision)")
                                return false
                            }
                            if let storePipe = storeMlaKvCacheF16Pipeline,
                               let kCache = KVCacheManager.shared.kCacheBuffer,
                               let vCache = KVCacheManager.shared.vCacheBuffer {
                                var pos = UInt32(step)
                                var nHeads: UInt32 = 16
                                layerEnc1.setComputePipelineState(storePipe)
                                layerEnc1.setBuffer(vVectorBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(kVectorBuffer, offset: 512 * MemoryLayout<Float>.stride, index: 1)
                                layerEnc1.setBuffer(kCache, offset: kLayerByteOffset, index: 2)
                                layerEnc1.setBuffer(vCache, offset: vLayerByteOffset, index: 3)
                                layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&nHeads, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(nHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(32, Int(nHeads)), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            // 10. Gate projection (hiddenDim -> 16 heads) into bVectorBuffer
                            if let gateProj = layer.mlaGateProj {
                                dispatchLinear(enc: layerEnc1, weight: gateProj, scale: nil, bias: nil, inBuf: xNorm1Buffer, outBuf: bVectorBuffer, inDim: hiddenDim, outDim: 16)
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            // 11. MLA Attention Decode (F16)
                            if let mlaPipe = mlaAttentionDecodeF16Pipeline,
                               let kCache = KVCacheManager.shared.kCacheBuffer,
                               let vCache = KVCacheManager.shared.vCacheBuffer {
                                var seqLen = UInt32(step + 1)
                                var nHeads: UInt32 = 16
                                var qkHeadDim: UInt32 = 192
                                var vDim: UInt32 = 128
                                layerEnc1.setComputePipelineState(mlaPipe)
                                layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(kCache, offset: kLayerByteOffset, index: 1)
                                layerEnc1.setBuffer(vCache, offset: vLayerByteOffset, index: 2)
                                layerEnc1.setBuffer(attnCtxBuffer, offset: 0, index: 3)
                                layerEnc1.setBuffer(bVectorBuffer, offset: 0, index: 4)
                                layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc1.setBytes(&nHeads, length: MemoryLayout<UInt32>.stride, index: 6)
                                layerEnc1.setBytes(&qkHeadDim, length: MemoryLayout<UInt32>.stride, index: 7)
                                layerEnc1.setBytes(&vDim, length: MemoryLayout<UInt32>.stride, index: 8)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(nHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(32, Int(nHeads)), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            // 12. Output projection: 2048 (16 heads x 128 v_dim) -> hiddenDim (1536) into attnOutBuffer
                            dispatchLinear(enc: layerEnc1, weight: layer.oProjTensor, scale: nil, bias: nil, inBuf: attnCtxBuffer, outBuf: attnOutBuffer, inDim: 2048, outDim: hiddenDim)
                            layerEnc1.memoryBarrier(scope: .buffers)
                        } else if isGemma4 {
                            // Gemma 4 hybrid attention: sliding layers use head_dim / num_key_value_heads;
                            // global layers use global_head_dim / num_global_key_value_heads, reuse the
                            // key projection as the value, and apply proportional RoPE. Attention scaling
                            // is 1.0 (no 1/sqrt(head_dim)).
                            layerEnc1.setComputePipelineState(clearPipeline)
                            layerEnc1.setBuffer(attnOutBuffer, offset: 0, index: 0)
                            layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            let gSliding = layer.isSlidingAttention
                            let gHeadDim: UInt32 = gSliding ? headDim : max(headDim, globalHeadDim)
                            let gNumKv: UInt32 = gSliding ? numKvHeads : max(1, numGlobalKvHeads)
                            let gQDim: UInt32 = numHeads * gHeadDim
                            let gKvDim: UInt32 = gNumKv * gHeadDim

                            dispatchLinear(enc: layerEnc1, weight: layer.qProjTensor, scale: layer.qScaleTensor, bias: layer.qBiasTensor, inBuf: xNorm1Buffer, outBuf: qGateBuffer, inDim: hiddenDim, outDim: gQDim)
                            dispatchLinear(enc: layerEnc1, weight: layer.kProjTensor, scale: layer.kScaleTensor, bias: layer.kBiasTensor, inBuf: xNorm1Buffer, outBuf: kVectorBuffer, inDim: hiddenDim, outDim: gKvDim)
                            // Sliding layers have their own value projection; global layers
                            // (attention_k_eq_v) reuse the key projection as the value.
                            if gSliding {
                                dispatchLinear(enc: layerEnc1, weight: layer.vProjTensor, scale: layer.vScaleTensor, bias: layer.vBiasTensor, inBuf: xNorm1Buffer, outBuf: vVectorBuffer, inDim: hiddenDim, outDim: gKvDim)
                            }
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // Value states: sliding layers norm their v_proj output; global layers
                            // reuse the (pre-norm) key projection output. Gemma 4's v_norm has no scale.
                            if let vNormPipe = rmsnormNoScalePipeline {
                                var vDim = gHeadDim
                                var vEps = eps
                                layerEnc1.setComputePipelineState(vNormPipe)
                                layerEnc1.setBuffer(gSliding ? vVectorBuffer : kVectorBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(vVectorBuffer, offset: 0, index: 1)
                                layerEnc1.setBytes(&vDim, length: MemoryLayout<UInt32>.stride, index: 2)
                                layerEnc1.setBytes(&vEps, length: MemoryLayout<Float>.stride, index: 3)
                                layerEnc1.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                                layerEnc1.dispatchThreadgroups(MTLSize(width: Int(gNumKv), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(gHeadDim)), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            // Q-Norm (per head, with scale)
                            if let qNorm = layer.qNormTensor, let qNormRaw = buffers[qNorm.shardIndex], let headNormPipe = headRmsnormPipeline {
                                var qNormOff = qNorm.offsetStart
                                var nQ = numHeads
                                var hD = gHeadDim
                                var hStride = gHeadDim
                                var epsVal = eps
                                layerEnc1.setComputePipelineState(headNormPipe)
                                layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(qNormRaw, offset: 0, index: 1)
                                layerEnc1.setBytes(&qNormOff, length: MemoryLayout<UInt64>.stride, index: 2)
                                layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&hStride, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 6)
                                layerEnc1.dispatchThreadgroups(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            // K-Norm (per head, with scale)
                            if let kNorm = layer.kNormTensor, let kNormRaw = buffers[kNorm.shardIndex], let headNormPipe = headRmsnormPipeline {
                                var kNormOff = kNorm.offsetStart
                                var nK = gNumKv
                                var hD = gHeadDim
                                var hStride = gHeadDim
                                var epsVal = eps
                                layerEnc1.setComputePipelineState(headNormPipe)
                                layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(kNormRaw, offset: 0, index: 1)
                                layerEnc1.setBytes(&kNormOff, length: MemoryLayout<UInt64>.stride, index: 2)
                                layerEnc1.setBytes(&nK, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&hStride, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 6)
                                layerEnc1.dispatchThreadgroups(MTLSize(width: Int(gNumKv), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            // RoPE (per-layer-type theta + proportional rotary dim).
                            // Sliding layers use standard RoPE (rotate i with i+headDim/2,
                            // denominator headDim); global layers use p-RoPE where only the
                            // first partial_rotary_factor * headDim dims rotate.
                            let gemmaRopePipe: MTLComputePipelineState? = gSliding ? ropePipeline : (ropeProportionalPipeline ?? ropePipeline)
                            if let ropePipe = gemmaRopePipe {
                                var pos = step
                                var nQ = numHeads
                                var nK = gNumKv
                                var hD = gHeadDim
                                var rD = UInt32(modelConfig?.effectiveRotaryDim(layerIndex: l, headDim: Int(gHeadDim)) ?? Int(rotaryDim))
                                var qStr = gHeadDim
                                var kStr = gHeadDim
                                var theta = modelConfig?.effectiveRopeTheta(layerIndex: l) ?? thetaVal
                                layerEnc1.setComputePipelineState(ropePipe)
                                layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 1)
                                layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 2)
                                layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc1.setBytes(&rD, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&qStr, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc1.setBytes(&theta, length: MemoryLayout<Float>.stride, index: 6)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), ropePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 0)
                                layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 1)
                                layerEnc1.setBytes(&nK, length: MemoryLayout<UInt32>.stride, index: 2)
                                layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc1.setBytes(&rD, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&kStr, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc1.setBytes(&theta, length: MemoryLayout<Float>.stride, index: 6)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(gNumKv), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(gNumKv), ropePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            if let kCache = KVCacheManager.shared.kCacheBuffer,
                               let vCache = KVCacheManager.shared.vCacheBuffer {
                                let slot = (loopIdx * actualLayers) + layer.fullAttnIndex
                                let kvMgr = KVCacheManager.shared
                                let prec = kvMgr.activePrecision
                                let layerByteOffset = kvMgr.kvSlotByteOffset(slot: slot, stride: Int(gemmaKvStride), elementBytes: prec.bytesPerElement)
                                var pos = step
                                var nKv = gNumKv
                                var hD = gHeadDim
                                var cacheStride = gemmaKvStride
                                // Sliding layers address a ring-sized slot region.
                                var ringLen: UInt32 = kvMgr.isRingSlot(slot) ? UInt32(kvMgr.slidingRingLen) : 0

                                if prec == .fp8,
                                   let kScale = KVCacheManager.shared.kScaleBuffer,
                                   let vScale = KVCacheManager.shared.vScaleBuffer,
                                   let storePipe = storeKvGemmaFP8Pipeline,
                                   let attnPipe = gqaGemmaFP8Pipeline {
                                    let scaleByteOffset = kvMgr.kvSlotScaleByteOffset(slot: slot, layoutHeads: Int(numKvHeads))
                                    layerEnc1.setComputePipelineState(storePipe)
                                    layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 0)
                                    layerEnc1.setBuffer(vVectorBuffer, offset: 0, index: 1)
                                    layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 2)
                                    layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 3)
                                    layerEnc1.setBuffer(kScale, offset: scaleByteOffset, index: 4)
                                    layerEnc1.setBuffer(vScale, offset: scaleByteOffset, index: 5)
                                    layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 6)
                                    layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 7)
                                    layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 8)
                                    layerEnc1.setBytes(&cacheStride, length: MemoryLayout<UInt32>.stride, index: 9)
                                    layerEnc1.setBytes(&ringLen, length: MemoryLayout<UInt32>.stride, index: 10)
                                    layerEnc1.dispatchThreadgroups(MTLSize(width: Int(gNumKv), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)
                                    // Ring bookkeeping: this layer just stored position `step`.
                                    kvMgr.noteRingStores(range: Int(step)..<Int(step) + 1)

                                    var seqLen = step + 1
                                    var nQ = numHeads
                                    var windowSize: UInt32 = gSliding ? UInt32(modelConfig?.effectiveSlidingWindow ?? 0) : 0
                                    var scaling = gemmaScaling
                                    layerEnc1.setComputePipelineState(attnPipe)
                                    layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                    layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                    layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                    layerEnc1.setBuffer(kScale, offset: scaleByteOffset, index: 3)
                                    layerEnc1.setBuffer(vScale, offset: scaleByteOffset, index: 4)
                                    layerEnc1.setBuffer(attnCtxBuffer, offset: 0, index: 5)
                                    layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 6)
                                    layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 7)
                                    layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 8)
                                    layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 9)
                                    layerEnc1.setBytes(&windowSize, length: MemoryLayout<UInt32>.stride, index: 10)
                                    layerEnc1.setBytes(&scaling, length: MemoryLayout<Float>.stride, index: 11)
                                    layerEnc1.setBytes(&cacheStride, length: MemoryLayout<UInt32>.stride, index: 12)
                                    layerEnc1.setBytes(&ringLen, length: MemoryLayout<UInt32>.stride, index: 13)
                                    layerEnc1.dispatchThreadgroups(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)
                                } else {
                                let storePipe: MTLComputePipelineState? = (prec == .fp16) ? (storeKvGemmaF16Pipeline ?? storeKvGemmaPipeline) : storeKvGemmaPipeline
                                if let storePipe {
                                    layerEnc1.setComputePipelineState(storePipe)
                                    layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 0)
                                    layerEnc1.setBuffer(vVectorBuffer, offset: 0, index: 1)
                                    layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 2)
                                    layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 3)
                                    layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 4)
                                    layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 5)
                                    layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 6)
                                    layerEnc1.setBytes(&cacheStride, length: MemoryLayout<UInt32>.stride, index: 7)
                                    layerEnc1.setBytes(&ringLen, length: MemoryLayout<UInt32>.stride, index: 8)
                                    layerEnc1.dispatchThreads(MTLSize(width: Int(gKvDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(gKvDim), storePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)
                                    // Ring bookkeeping: this layer just stored position `step`.
                                    kvMgr.noteRingStores(range: Int(step)..<Int(step) + 1)
                                }

                                var seqLen = step + 1
                                var nQ = numHeads
                                var nKvA = gNumKv
                                var hDA = gHeadDim
                                var windowSize: UInt32 = gSliding ? UInt32(modelConfig?.effectiveSlidingWindow ?? 0) : 0
                                var scaling = gemmaScaling
                                let attnPipe: MTLComputePipelineState? = (prec == .fp16) ? (gqaGemmaF16Pipeline ?? gqaGemmaPipeline) : gqaGemmaPipeline
                                if let attnPipe {
                                    layerEnc1.setComputePipelineState(attnPipe)
                                    layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                    layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                    layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                    layerEnc1.setBuffer(attnCtxBuffer, offset: 0, index: 3)
                                    layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 4)
                                    layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 5)
                                    layerEnc1.setBytes(&nKvA, length: MemoryLayout<UInt32>.stride, index: 6)
                                    layerEnc1.setBytes(&hDA, length: MemoryLayout<UInt32>.stride, index: 7)
                                    layerEnc1.setBytes(&windowSize, length: MemoryLayout<UInt32>.stride, index: 8)
                                    layerEnc1.setBytes(&scaling, length: MemoryLayout<Float>.stride, index: 9)
                                    layerEnc1.setBytes(&cacheStride, length: MemoryLayout<UInt32>.stride, index: 10)
                                    layerEnc1.setBytes(&ringLen, length: MemoryLayout<UInt32>.stride, index: 11)
                                    layerEnc1.dispatchThreadgroups(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)
                                }
                                }
                            }

                            dispatchLinear(enc: layerEnc1, weight: layer.oProjTensor, scale: layer.oScaleTensor, bias: layer.oBiasTensor, inBuf: attnCtxBuffer, outBuf: attnOutBuffer, inDim: gQDim, outDim: hiddenDim)
                            layerEnc1.memoryBarrier(scope: .buffers)
                        } else if layer.attentionType == .fullAttention {
                            layerEnc1.setComputePipelineState(clearPipeline)
                            layerEnc1.setBuffer(attnOutBuffer, offset: 0, index: 0)
                            layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            let currentQDim = isStandardGqa ? (numHeads * headDim) : (numHeads * headDim * 2)
                            let currentKvDim = kvStride

                            if let fusedQKV = layer.fusedQKVTensor {
                                // Spark 2.5: single fused q_k_v_proj [qDim + 2*kvDim, hiddenDim]
                                dispatchLinear(enc: layerEnc1, weight: fusedQKV, scale: nil, bias: nil, inBuf: xNorm1Buffer, outBuf: qGateBuffer, inDim: hiddenDim, outDim: currentQDim)
                                dispatchLinear(enc: layerEnc1, weight: fusedQKV, scale: nil, bias: nil, inBuf: xNorm1Buffer, outBuf: kVectorBuffer, inDim: hiddenDim, outDim: currentKvDim, weightOffsetAdd: UInt64(currentQDim) * UInt64(hiddenDim) * 2)
                                dispatchLinear(enc: layerEnc1, weight: fusedQKV, scale: nil, bias: nil, inBuf: xNorm1Buffer, outBuf: vVectorBuffer, inDim: hiddenDim, outDim: currentKvDim, weightOffsetAdd: (UInt64(currentQDim) + UInt64(currentKvDim)) * UInt64(hiddenDim) * 2)
                                if let gateProj = layer.attnGateProjTensor {
                                    dispatchLinear(enc: layerEnc1, weight: gateProj, scale: nil, bias: nil, inBuf: xNorm1Buffer, outBuf: bVectorBuffer, inDim: hiddenDim, outDim: numHeads)
                                }
                            } else {
                                dispatchLinear(enc: layerEnc1, weight: layer.qProjTensor, scale: layer.qScaleTensor, bias: layer.qBiasTensor, inBuf: xNorm1Buffer, outBuf: qGateBuffer, inDim: hiddenDim, outDim: currentQDim)
                                dispatchLinear(enc: layerEnc1, weight: layer.kProjTensor, scale: layer.kScaleTensor, bias: layer.kBiasTensor, inBuf: xNorm1Buffer, outBuf: kVectorBuffer, inDim: hiddenDim, outDim: currentKvDim)
                                dispatchLinear(enc: layerEnc1, weight: layer.vProjTensor, scale: layer.vScaleTensor, bias: layer.vBiasTensor, inBuf: xNorm1Buffer, outBuf: vVectorBuffer, inDim: hiddenDim, outDim: currentKvDim)
                            }
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // Q-Norm (if present)
                            if let qNorm = layer.qNormTensor, let qNormRaw = buffers[qNorm.shardIndex] {
                                let isQNormF16 = (qNorm.dtype.contains("F16") || qNorm.dtype.contains("HALF") || qNorm.dtype.contains("FLOAT16")) && !qNorm.dtype.contains("BF16") && !qNorm.dtype.contains("BFLOAT")
                                let qHeadNormPipe: MTLComputePipelineState?
                                if isRMSNormOffset {
                                    qHeadNormPipe = (isQNormF16 && headRmsnormOffsetF16Pipeline != nil) ? headRmsnormOffsetF16Pipeline : (headRmsnormOffsetPipeline ?? headRmsnormPipeline)
                                } else {
                                    qHeadNormPipe = (isQNormF16 && headRmsnormF16Pipeline != nil) ? headRmsnormF16Pipeline : headRmsnormPipeline
                                }
                                if let headNormPipe = qHeadNormPipe {
                                    var qNormOff = qNorm.offsetStart
                                    var nQ = numHeads
                                    var hD = headDim
                                    var hStride: UInt32 = isStandardGqa ? headDim : (headDim * 2)
                                    var epsVal = eps
                                    layerEnc1.setComputePipelineState(headNormPipe)
                                    layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                    layerEnc1.setBuffer(qNormRaw, offset: 0, index: 1)
                                    layerEnc1.setBytes(&qNormOff, length: MemoryLayout<UInt64>.stride, index: 2)
                                    layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 3)
                                    layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                                    layerEnc1.setBytes(&hStride, length: MemoryLayout<UInt32>.stride, index: 5)
                                    layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 6)
                                    layerEnc1.dispatchThreadgroups(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)
                                }
                            }

                            // K-Norm (if present)
                            if let kNorm = layer.kNormTensor, let kNormRaw = buffers[kNorm.shardIndex] {
                                let isKNormF16 = (kNorm.dtype.contains("F16") || kNorm.dtype.contains("HALF") || kNorm.dtype.contains("FLOAT16")) && !kNorm.dtype.contains("BF16") && !kNorm.dtype.contains("BFLOAT")
                                let kHeadNormPipe: MTLComputePipelineState?
                                if isRMSNormOffset {
                                    kHeadNormPipe = (isKNormF16 && headRmsnormOffsetF16Pipeline != nil) ? headRmsnormOffsetF16Pipeline : (headRmsnormOffsetPipeline ?? headRmsnormPipeline)
                                } else {
                                    kHeadNormPipe = (isKNormF16 && headRmsnormF16Pipeline != nil) ? headRmsnormF16Pipeline : headRmsnormPipeline
                                }
                                if let headNormPipe = kHeadNormPipe {
                                    var kNormOff = kNorm.offsetStart
                                    var nK = numKvHeads
                                    var hD = headDim
                                    var hStride = headDim
                                    var epsVal = eps
                                    layerEnc1.setComputePipelineState(headNormPipe)
                                    layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 0)
                                    layerEnc1.setBuffer(kNormRaw, offset: 0, index: 1)
                                    layerEnc1.setBytes(&kNormOff, length: MemoryLayout<UInt64>.stride, index: 2)
                                    layerEnc1.setBytes(&nK, length: MemoryLayout<UInt32>.stride, index: 3)
                                    layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                                    layerEnc1.setBytes(&hStride, length: MemoryLayout<UInt32>.stride, index: 5)
                                    layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 6)
                                    layerEnc1.dispatchThreadgroups(MTLSize(width: Int(numKvHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)
                                }
                            }

                            // RoPE on Q and K
                            if let ropePipe = ropePipeline {
                                var pos = step
                                var nQ = numHeads
                                var nK = numKvHeads
                                var hD = headDim
                                var rD = UInt32(modelConfig?.effectiveRotaryDim(layerIndex: l, headDim: Int(headDim)) ?? Int(rotaryDim))
                                var qStr: UInt32 = isStandardGqa ? headDim : (headDim * 2)
                                var kStr = headDim
                                var theta = modelConfig?.effectiveRopeTheta(layerIndex: l) ?? thetaVal

                                layerEnc1.setComputePipelineState(ropePipe)
                                layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 1)
                                layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 2)
                                layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc1.setBytes(&rD, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&qStr, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc1.setBytes(&theta, length: MemoryLayout<Float>.stride, index: 6)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), ropePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                                layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 0)
                                layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 1)
                                layerEnc1.setBytes(&nK, length: MemoryLayout<UInt32>.stride, index: 2)
                                layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc1.setBytes(&rD, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&kStr, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc1.setBytes(&theta, length: MemoryLayout<Float>.stride, index: 6)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(numKvHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numKvHeads), ropePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            // Store KV-Cache and Fused GQA Decode
                            if let kCache = KVCacheManager.shared.kCacheBuffer,
                               let vCache = KVCacheManager.shared.vCacheBuffer {
                                let slot = (loopIdx * actualLayers) + layer.fullAttnIndex
                                let maxSeq = KVCacheManager.shared.allocatedSeqLen
                                let prec = KVCacheManager.shared.activePrecision
                                let layerByteOffset = slot * maxSeq * Int(kvStride) * prec.bytesPerElement
                                var pos = step
                                var nKv = numKvHeads
                                var hD = headDim
                                var nQ = numHeads
                                var seqLen = step + 1

                                switch prec {
                                case .fp16:
                                    if let storePipe = storeKvCacheF16Pipeline ?? storeKvCachePipeline {
                                        layerEnc1.setComputePipelineState(storePipe)
                                        layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 0)
                                        layerEnc1.setBuffer(vVectorBuffer, offset: 0, index: 1)
                                        layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 2)
                                        layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 3)
                                        layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 4)
                                        layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.dispatchThreads(MTLSize(width: Int(kvStride), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(kvStride), storePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    }

if layer.attnGateProjTensor != nil,
                                       let chunkedPipe = gqaHeadGateF16ChunkedPipeline,
                                       let chunkedCombinePipe = gqaHeadGateF16ChunkedCombinePipeline {
                                        // Flash-decoding style: split the KV range across chunks so
                                        // the attention runs on hundreds of threads instead of one
                                        // thread per head (latency-bound at long contexts).
                                        var windowSize: UInt32 = layer.isSlidingAttention ? UInt32(modelConfig?.effectiveSlidingWindow ?? 0) : 0
                                        let curLen: UInt32 = (seqLen == 0) ? 1 : ((seqLen & 0x80000000) != 0 ? ((seqLen & 0x7FFFFFFF) + 1) : seqLen)
                                        let winStart: UInt32 = (windowSize > 0 && curLen > windowSize) ? (curLen - windowSize) : 0
                                        let spanLen = curLen - winStart
                                        let numChunks: UInt32 = max(1, min(256, (spanLen + 63) / 64))
                                        let chunkSize: UInt32 = (spanLen + numChunks - 1) / numChunks
                                        var seqLenVal = seqLen
                                        var nQv = nQ
                                        var nKvv = nKv
                                        var hDv = hD
                                        var winVal = windowSize
                                        var chunkVal = chunkSize
                                        layerEnc1.setComputePipelineState(chunkedPipe)
                                        layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                        layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                        layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                        layerEnc1.setBuffer(attnPartialMBuffer, offset: 0, index: 3)
                                        layerEnc1.setBuffer(attnPartialLBuffer, offset: 0, index: 4)
                                        layerEnc1.setBuffer(attnPartialAccBuffer, offset: 0, index: 5)
                                        layerEnc1.setBytes(&seqLenVal, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.setBytes(&nQv, length: MemoryLayout<UInt32>.stride, index: 7)
                                        layerEnc1.setBytes(&nKvv, length: MemoryLayout<UInt32>.stride, index: 8)
                                        layerEnc1.setBytes(&hDv, length: MemoryLayout<UInt32>.stride, index: 9)
                                        layerEnc1.setBytes(&winVal, length: MemoryLayout<UInt32>.stride, index: 10)
                                        layerEnc1.setBytes(&chunkVal, length: MemoryLayout<UInt32>.stride, index: 11)
                                        layerEnc1.dispatchThreadgroups(MTLSize(width: Int(numHeads), height: Int(numChunks), depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                                        layerEnc1.memoryBarrier(scope: .buffers)

                                        var numChunkVal = numChunks
                                        layerEnc1.setComputePipelineState(chunkedCombinePipe)
                                        layerEnc1.setBuffer(attnPartialMBuffer, offset: 0, index: 0)
                                        layerEnc1.setBuffer(attnPartialLBuffer, offset: 0, index: 1)
                                        layerEnc1.setBuffer(attnPartialAccBuffer, offset: 0, index: 2)
                                        layerEnc1.setBuffer(attnCtxBuffer, offset: 0, index: 3)
                                        // Spark 2.5: the per-head sigmoid gate lives in the g_proj
                                        // output (bVectorBuffer), NOT the fused QKV vector — the old
                                        // headgate kernel binds bVectorBuffer here too. Binding
                                        // qGateBuffer read Q values as gate logits (token soup).
                                        layerEnc1.setBuffer(bVectorBuffer, offset: 0, index: 4)
                                        layerEnc1.setBytes(&numChunkVal, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc1.setBytes(&nQv, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.setBytes(&hDv, length: MemoryLayout<UInt32>.stride, index: 7)
                                        layerEnc1.dispatchThreadgroups(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                                    } else if layer.attnGateProjTensor != nil, let headGatePipe = gqaHeadGateF16Pipeline ?? gqaHeadGatePipeline {
                                        // Spark 2.5: per-head sigmoid output gate + optional sliding window
                                        var windowSize: UInt32 = layer.isSlidingAttention ? UInt32(modelConfig?.effectiveSlidingWindow ?? 0) : 0
                                        layerEnc1.setComputePipelineState(headGatePipe)
                                        layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                        layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                        layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                        layerEnc1.setBuffer(attnCtxBuffer, offset: 0, index: 3)
                                        layerEnc1.setBuffer(bVectorBuffer, offset: 0, index: 4)
                                        layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 7)
                                        layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 8)
                                        layerEnc1.setBytes(&windowSize, length: MemoryLayout<UInt32>.stride, index: 9)
                                        layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), headGatePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    } else if isStandardGqa, let gqaStdPipe = gqaStandardF16Pipeline ?? gqaStandardPipeline {
                                        layerEnc1.setComputePipelineState(gqaStdPipe)
                                        layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                        layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                        layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                        layerEnc1.setBuffer(attnCtxBuffer, offset: 0, index: 3)
                                        layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 4)
                                        layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 7)
                                        layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), gqaStdPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    } else if let gqaPipe = gqaDecodeF16Pipeline ?? gqaDecodePipeline {
                                        if let chunkedPipe = gqaFusedF16ChunkedPipeline,
                                           let chunkedCombinePipe = gqaFusedF16ChunkedCombinePipeline {
                                            // Flash-decoding split-K for the fused Q+Gate path — the
                                            // branch Ornith-class models actually take (no g_proj tensor).
                                            var chunkedWin: UInt32 = 0
                                            let curLen: UInt32 = (seqLen == 0) ? 1 : ((seqLen & 0x80000000) != 0 ? ((seqLen & 0x7FFFFFFF) + 1) : seqLen)
                                            let numChunks: UInt32 = max(1, min(256, (curLen + 63) / 64))
                                            let chunkSize: UInt32 = (curLen + numChunks - 1) / numChunks
                                            var seqLenVal = seqLen
                                            var nQv = nQ
                                            var nKvv = nKv
                                            var hDv = hD
                                            var chunkVal = chunkSize
                                            layerEnc1.setComputePipelineState(chunkedPipe)
                                            layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                            layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                            layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                            layerEnc1.setBuffer(attnPartialMBuffer, offset: 0, index: 3)
                                            layerEnc1.setBuffer(attnPartialLBuffer, offset: 0, index: 4)
                                            layerEnc1.setBuffer(attnPartialAccBuffer, offset: 0, index: 5)
                                            layerEnc1.setBytes(&seqLenVal, length: MemoryLayout<UInt32>.stride, index: 6)
                                            layerEnc1.setBytes(&nQv, length: MemoryLayout<UInt32>.stride, index: 7)
                                            layerEnc1.setBytes(&nKvv, length: MemoryLayout<UInt32>.stride, index: 8)
                                            layerEnc1.setBytes(&hDv, length: MemoryLayout<UInt32>.stride, index: 9)
                                            layerEnc1.setBytes(&chunkVal, length: MemoryLayout<UInt32>.stride, index: 10)
                                            layerEnc1.dispatchThreadgroups(MTLSize(width: Int(numHeads), height: Int(numChunks), depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                                            layerEnc1.memoryBarrier(scope: .buffers)

                                            var numChunkVal = numChunks
                                            layerEnc1.setComputePipelineState(chunkedCombinePipe)
                                            layerEnc1.setBuffer(attnPartialMBuffer, offset: 0, index: 0)
                                            layerEnc1.setBuffer(attnPartialLBuffer, offset: 0, index: 1)
                                            layerEnc1.setBuffer(attnPartialAccBuffer, offset: 0, index: 2)
                                            layerEnc1.setBuffer(attnCtxBuffer, offset: 0, index: 3)
                                            layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 4)
                                            layerEnc1.setBytes(&numChunkVal, length: MemoryLayout<UInt32>.stride, index: 5)
                                            layerEnc1.setBytes(&nQv, length: MemoryLayout<UInt32>.stride, index: 6)
                                            layerEnc1.setBytes(&hDv, length: MemoryLayout<UInt32>.stride, index: 7)
                                            layerEnc1.dispatchThreadgroups(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                                        } else {
                                            layerEnc1.setComputePipelineState(gqaPipe)
                                            layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                            layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                            layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                            layerEnc1.setBuffer(attnCtxBuffer, offset: 0, index: 3)
                                            layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 4)
                                            layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 5)
                                            layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 6)
                                            layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 7)
                                            layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), gqaPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                        }
                                    }

                                case .fp8:
                                    let scaleByteOffset = slot * maxSeq * Int(numKvHeads) * MemoryLayout<UInt16>.stride
                                    if let quantStorePipe = storeKvCacheFP8Pipeline,
                                       let kScale = KVCacheManager.shared.kScaleBuffer,
                                       let vScale = KVCacheManager.shared.vScaleBuffer {
                                        layerEnc1.setComputePipelineState(quantStorePipe)
                                        layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 0)
                                        layerEnc1.setBuffer(vVectorBuffer, offset: 0, index: 1)
                                        layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 2)
                                        layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 3)
                                        layerEnc1.setBuffer(kScale, offset: scaleByteOffset, index: 4)
                                        layerEnc1.setBuffer(vScale, offset: scaleByteOffset, index: 5)
                                        layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 7)
                                        layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 8)
                                        let threadgroups = MTLSize(width: Int(numKvHeads), height: 1, depth: 1)
                                        let threadsPerTG = MTLSize(width: 32, height: 1, depth: 1)
                                        layerEnc1.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerTG)
                                    } else if let storePipe = storeKvCachePipeline {
                                        layerEnc1.setComputePipelineState(storePipe)
                                        layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 0)
                                        layerEnc1.setBuffer(vVectorBuffer, offset: 0, index: 1)
                                        layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 2)
                                        layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 3)
                                        layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 4)
                                        layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.dispatchThreads(MTLSize(width: Int(kvStride), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(kvStride), storePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    }

                                    if let kScale = KVCacheManager.shared.kScaleBuffer,
                                       let vScale = KVCacheManager.shared.vScaleBuffer {
                                        if layer.attnGateProjTensor != nil,
                                           let chunkedPipe = gqaHeadGateFP8ChunkedPipeline,
                                           let chunkedCombinePipe = gqaHeadGateF16ChunkedCombinePipeline {
                                            // Flash-decoding split-K for FP8 KV (same shape as the F16
                                            // path): each (head, chunk) thread dequantizes its INT8
                                            // slice, so the scan runs on hundreds of threads instead
                                            // of one per head (latency-bound at long contexts).
                                            var windowSize: UInt32 = layer.isSlidingAttention ? UInt32(modelConfig?.effectiveSlidingWindow ?? 0) : 0
                                            let curLen: UInt32 = (seqLen == 0) ? 1 : ((seqLen & 0x80000000) != 0 ? ((seqLen & 0x7FFFFFFF) + 1) : seqLen)
                                            let winStart: UInt32 = (windowSize > 0 && curLen > windowSize) ? (curLen - windowSize) : 0
                                            let spanLen = curLen - winStart
                                            let numChunks: UInt32 = max(1, min(256, (spanLen + 63) / 64))
                                            let chunkSize: UInt32 = (spanLen + numChunks - 1) / numChunks
                                            var seqLenVal = seqLen
                                            var nQv = nQ
                                            var nKvv = nKv
                                            var hDv = hD
                                            var winVal = windowSize
                                            var chunkVal = chunkSize
                                            layerEnc1.setComputePipelineState(chunkedPipe)
                                            layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                            layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                            layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                            layerEnc1.setBuffer(kScale, offset: scaleByteOffset, index: 3)
                                            layerEnc1.setBuffer(vScale, offset: scaleByteOffset, index: 4)
                                            layerEnc1.setBuffer(attnPartialMBuffer, offset: 0, index: 5)
                                            layerEnc1.setBuffer(attnPartialLBuffer, offset: 0, index: 6)
                                            layerEnc1.setBuffer(attnPartialAccBuffer, offset: 0, index: 7)
                                            layerEnc1.setBytes(&seqLenVal, length: MemoryLayout<UInt32>.stride, index: 8)
                                            layerEnc1.setBytes(&nQv, length: MemoryLayout<UInt32>.stride, index: 9)
                                            layerEnc1.setBytes(&nKvv, length: MemoryLayout<UInt32>.stride, index: 10)
                                            layerEnc1.setBytes(&hDv, length: MemoryLayout<UInt32>.stride, index: 11)
                                            layerEnc1.setBytes(&winVal, length: MemoryLayout<UInt32>.stride, index: 12)
                                            layerEnc1.setBytes(&chunkVal, length: MemoryLayout<UInt32>.stride, index: 13)
                                            layerEnc1.dispatchThreadgroups(MTLSize(width: Int(numHeads), height: Int(numChunks), depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                                            layerEnc1.memoryBarrier(scope: .buffers)

                                            var numChunkVal = numChunks
                                            layerEnc1.setComputePipelineState(chunkedCombinePipe)
                                            layerEnc1.setBuffer(attnPartialMBuffer, offset: 0, index: 0)
                                            layerEnc1.setBuffer(attnPartialLBuffer, offset: 0, index: 1)
                                            layerEnc1.setBuffer(attnPartialAccBuffer, offset: 0, index: 2)
                                            layerEnc1.setBuffer(attnCtxBuffer, offset: 0, index: 3)
                                            // Spark 2.5: the per-head sigmoid gate lives in the g_proj
                                            // output (bVectorBuffer), NOT the fused QKV vector — binding
                                            // qGateBuffer here would read Q values as gate logits.
                                            layerEnc1.setBuffer(bVectorBuffer, offset: 0, index: 4)
                                            layerEnc1.setBytes(&numChunkVal, length: MemoryLayout<UInt32>.stride, index: 5)
                                            layerEnc1.setBytes(&nQv, length: MemoryLayout<UInt32>.stride, index: 6)
                                            layerEnc1.setBytes(&hDv, length: MemoryLayout<UInt32>.stride, index: 7)
                                            layerEnc1.dispatchThreadgroups(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                                        } else if layer.attnGateProjTensor != nil, let headGatePipe = gqaHeadGateFP8Pipeline {
                                            // Spark 2.5: per-head sigmoid output gate + optional sliding window (FP8 KV)
                                            var windowSize: UInt32 = layer.isSlidingAttention ? UInt32(modelConfig?.effectiveSlidingWindow ?? 0) : 0
                                            layerEnc1.setComputePipelineState(headGatePipe)
                                            layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                            layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                            layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                            layerEnc1.setBuffer(kScale, offset: scaleByteOffset, index: 3)
                                            layerEnc1.setBuffer(vScale, offset: scaleByteOffset, index: 4)
                                            layerEnc1.setBuffer(attnCtxBuffer, offset: 0, index: 5)
                                            layerEnc1.setBuffer(bVectorBuffer, offset: 0, index: 6)
                                            layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 7)
                                            layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 8)
                                            layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 9)
                                            layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 10)
                                            layerEnc1.setBytes(&windowSize, length: MemoryLayout<UInt32>.stride, index: 11)
                                            layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), headGatePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                        } else if isStandardGqa, let gqaStdPipe = gqaStandardFP8Pipeline {
                                            layerEnc1.setComputePipelineState(gqaStdPipe)
                                            layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                            layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                            layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                            layerEnc1.setBuffer(kScale, offset: scaleByteOffset, index: 3)
                                            layerEnc1.setBuffer(vScale, offset: scaleByteOffset, index: 4)
                                            layerEnc1.setBuffer(attnCtxBuffer, offset: 0, index: 5)
                                            layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 6)
                                            layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 7)
                                            layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 8)
                                            layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 9)
                                            layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), gqaStdPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                        } else if let gqaPipe = gqaDecodeFP8Pipeline {
                                            if let chunkedPipe = gqaFusedFP8ChunkedPipeline,
                                               let chunkedCombinePipe = gqaFusedF16ChunkedCombinePipeline {
                                                // Flash-decoding split-K for the fused Q+Gate FP8 path —
                                                // the branch Ornith-class models take (no g_proj tensor).
                                                let curLen: UInt32 = (seqLen == 0) ? 1 : ((seqLen & 0x80000000) != 0 ? ((seqLen & 0x7FFFFFFF) + 1) : seqLen)
                                                let numChunks: UInt32 = max(1, min(256, (curLen + 63) / 64))
                                                let chunkSize: UInt32 = (curLen + numChunks - 1) / numChunks
                                                var seqLenVal = seqLen
                                                var nQv = nQ
                                                var nKvv = nKv
                                                var hDv = hD
                                                var chunkVal = chunkSize
                                                layerEnc1.setComputePipelineState(chunkedPipe)
                                                layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                                layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                                layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                                layerEnc1.setBuffer(kScale, offset: scaleByteOffset, index: 3)
                                                layerEnc1.setBuffer(vScale, offset: scaleByteOffset, index: 4)
                                                layerEnc1.setBuffer(attnPartialMBuffer, offset: 0, index: 5)
                                                layerEnc1.setBuffer(attnPartialLBuffer, offset: 0, index: 6)
                                                layerEnc1.setBuffer(attnPartialAccBuffer, offset: 0, index: 7)
                                                layerEnc1.setBytes(&seqLenVal, length: MemoryLayout<UInt32>.stride, index: 8)
                                                layerEnc1.setBytes(&nQv, length: MemoryLayout<UInt32>.stride, index: 9)
                                                layerEnc1.setBytes(&nKvv, length: MemoryLayout<UInt32>.stride, index: 10)
                                                layerEnc1.setBytes(&hDv, length: MemoryLayout<UInt32>.stride, index: 11)
                                                layerEnc1.setBytes(&chunkVal, length: MemoryLayout<UInt32>.stride, index: 12)
                                                layerEnc1.dispatchThreadgroups(MTLSize(width: Int(numHeads), height: Int(numChunks), depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                                                layerEnc1.memoryBarrier(scope: .buffers)

                                                var numChunkVal = numChunks
                                                layerEnc1.setComputePipelineState(chunkedCombinePipe)
                                                layerEnc1.setBuffer(attnPartialMBuffer, offset: 0, index: 0)
                                                layerEnc1.setBuffer(attnPartialLBuffer, offset: 0, index: 1)
                                                layerEnc1.setBuffer(attnPartialAccBuffer, offset: 0, index: 2)
                                                layerEnc1.setBuffer(attnCtxBuffer, offset: 0, index: 3)
                                                layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 4)
                                                layerEnc1.setBytes(&numChunkVal, length: MemoryLayout<UInt32>.stride, index: 5)
                                                layerEnc1.setBytes(&nQv, length: MemoryLayout<UInt32>.stride, index: 6)
                                                layerEnc1.setBytes(&hDv, length: MemoryLayout<UInt32>.stride, index: 7)
                                                layerEnc1.dispatchThreadgroups(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                                            } else {
                                                layerEnc1.setComputePipelineState(gqaPipe)
                                                layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                                layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                                layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                                layerEnc1.setBuffer(kScale, offset: scaleByteOffset, index: 3)
                                                layerEnc1.setBuffer(vScale, offset: scaleByteOffset, index: 4)
                                                layerEnc1.setBuffer(attnCtxBuffer, offset: 0, index: 5)
                                                layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 6)
                                                layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 7)
                                                layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 8)
                                                layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 9)
                                                layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), gqaPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                            }
                                        }
                                    }

                                case .fp32:
                                    if let storePipe = storeKvCachePipeline {
                                        layerEnc1.setComputePipelineState(storePipe)
                                        layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 0)
                                        layerEnc1.setBuffer(vVectorBuffer, offset: 0, index: 1)
                                        layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 2)
                                        layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 3)
                                        layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 4)
                                        layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.dispatchThreads(MTLSize(width: Int(kvStride), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(kvStride), storePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    }

                                    if layer.attnGateProjTensor != nil, let headGatePipe = gqaHeadGatePipeline {
                                        // Spark 2.5: per-head sigmoid output gate + optional sliding window
                                        var windowSize: UInt32 = layer.isSlidingAttention ? UInt32(modelConfig?.effectiveSlidingWindow ?? 0) : 0
                                        layerEnc1.setComputePipelineState(headGatePipe)
                                        layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                        layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                        layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                        layerEnc1.setBuffer(attnCtxBuffer, offset: 0, index: 3)
                                        layerEnc1.setBuffer(bVectorBuffer, offset: 0, index: 4)
                                        layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 7)
                                        layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 8)
                                        layerEnc1.setBytes(&windowSize, length: MemoryLayout<UInt32>.stride, index: 9)
                                        layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), headGatePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    } else if isStandardGqa, let gqaStdPipe = gqaStandardPipeline {
                                        layerEnc1.setComputePipelineState(gqaStdPipe)
                                        layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                        layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                        layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                        layerEnc1.setBuffer(attnCtxBuffer, offset: 0, index: 3)
                                        layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 4)
                                        layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 7)
                                        layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), gqaStdPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    } else if let gqaPipe = gqaDecodePipeline {
                                        layerEnc1.setComputePipelineState(gqaPipe)
                                        layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                        layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                        layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                        layerEnc1.setBuffer(attnCtxBuffer, offset: 0, index: 3)
                                        layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 4)
                                        layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 7)
                                        layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), gqaPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    }
                                }
                            }
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // Attention Output Projection (o_proj)
                            dispatchLinear(enc: layerEnc1, weight: layer.oProjTensor, scale: layer.oScaleTensor, bias: layer.oBiasTensor, inBuf: attnCtxBuffer, outBuf: attnOutBuffer, inDim: attnCtxDim, outDim: hiddenDim)
                            layerEnc1.memoryBarrier(scope: .buffers)
                        } else if layer.isKDA {
                            // KDA Linear Attention (Key-Decay Attention with recurrent associative state)
                            layerEnc1.setComputePipelineState(clearPipeline)
                            layerEnc1.setBuffer(attnOutBuffer, offset: 0, index: 0)
                            layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 1. Q, K, V projections (hiddenDim 1536 -> 2048)
                            dispatchLinear(enc: layerEnc1, weight: layer.kdaQProj ?? layer.qProjTensor, scale: nil, bias: nil, inBuf: xNorm1Buffer, outBuf: qGateBuffer, inDim: hiddenDim, outDim: 2048)
                            dispatchLinear(enc: layerEnc1, weight: layer.kdaKProj ?? layer.kProjTensor, scale: nil, bias: nil, inBuf: xNorm1Buffer, outBuf: kVectorBuffer, inDim: hiddenDim, outDim: 2048)
                            dispatchLinear(enc: layerEnc1, weight: layer.kdaVProj ?? layer.vProjTensor, scale: nil, bias: nil, inBuf: xNorm1Buffer, outBuf: vVectorBuffer, inDim: hiddenDim, outDim: 2048)
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 2. Causal 1D Convolutions with SiLU
                            let linIdx = layer.linAttnIndex
                            if let convPipe = causalConv1dPipeline, let convState = KVCacheManager.shared.convStateBuffer {
                                var numChan: UInt32 = 2048
                                // Q conv
                                if let qc = layer.kdaQConv1d, let qcRaw = buffers[qc.shardIndex] {
                                    var cOff = qc.offsetStart
                                    let qStateOffset = (linIdx * 6144 + 0) * 4 * MemoryLayout<Float>.stride
                                    layerEnc1.setComputePipelineState(convPipe)
                                    layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                    layerEnc1.setBuffer(qcRaw, offset: 0, index: 1)
                                    layerEnc1.setBuffer(convState, offset: qStateOffset, index: 2)
                                    layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 3)
                                    layerEnc1.setBytes(&cOff, length: MemoryLayout<UInt64>.stride, index: 4)
                                    layerEnc1.setBytes(&numChan, length: MemoryLayout<UInt32>.stride, index: 5)
                                    layerEnc1.dispatchThreads(MTLSize(width: 2048, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, convPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)
                                }
                                // K conv
                                if let kc = layer.kdaKConv1d, let kcRaw = buffers[kc.shardIndex] {
                                    var cOff = kc.offsetStart
                                    let kStateOffset = (linIdx * 6144 + 2048) * 4 * MemoryLayout<Float>.stride
                                    layerEnc1.setComputePipelineState(convPipe)
                                    layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 0)
                                    layerEnc1.setBuffer(kcRaw, offset: 0, index: 1)
                                    layerEnc1.setBuffer(convState, offset: kStateOffset, index: 2)
                                    layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 3)
                                    layerEnc1.setBytes(&cOff, length: MemoryLayout<UInt64>.stride, index: 4)
                                    layerEnc1.setBytes(&numChan, length: MemoryLayout<UInt32>.stride, index: 5)
                                    layerEnc1.dispatchThreads(MTLSize(width: 2048, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, convPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)
                                }
                                // V conv
                                if let vc = layer.kdaVConv1d, let vcRaw = buffers[vc.shardIndex] {
                                    var cOff = vc.offsetStart
                                    let vStateOffset = (linIdx * 6144 + 4096) * 4 * MemoryLayout<Float>.stride
                                    layerEnc1.setComputePipelineState(convPipe)
                                    layerEnc1.setBuffer(vVectorBuffer, offset: 0, index: 0)
                                    layerEnc1.setBuffer(vcRaw, offset: 0, index: 1)
                                    layerEnc1.setBuffer(convState, offset: vStateOffset, index: 2)
                                    layerEnc1.setBuffer(vVectorBuffer, offset: 0, index: 3)
                                    layerEnc1.setBytes(&cOff, length: MemoryLayout<UInt64>.stride, index: 4)
                                    layerEnc1.setBytes(&numChan, length: MemoryLayout<UInt32>.stride, index: 5)
                                    layerEnc1.dispatchThreads(MTLSize(width: 2048, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, convPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)
                                }
                            }

                            // 3. F, B, G projections
                            // F projection: decay gate (2048) into aVectorBuffer
                            dispatchLinear(enc: layerEnc1, weight: layer.inProjA, scale: nil, bias: nil, inBuf: xNorm1Buffer, outBuf: aVectorBuffer, inDim: hiddenDim, outDim: 2048)
                            // B projection: beta gate (16) into bVectorBuffer
                            dispatchLinear(enc: layerEnc1, weight: layer.inProjB, scale: nil, bias: nil, inBuf: xNorm1Buffer, outBuf: bVectorBuffer, inDim: hiddenDim, outDim: 16)
                            // G projection: output gate (2048) into zGateBuffer
                            dispatchLinear(enc: layerEnc1, weight: layer.inProjZ, scale: nil, bias: nil, inBuf: xNorm1Buffer, outBuf: zGateBuffer, inDim: hiddenDim, outDim: 2048)
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 4. KDA Recurrent Step
                            if let kdaPipe = kdaLinearAttnStepPipeline,
                               let sBuf = KVCacheManager.shared.linearStateBuffer,
                               let aLog = layer.aLogTensor, let aLogRaw = buffers[aLog.shardIndex],
                               let dtBias = layer.dtBiasTensor, let dtBiasRaw = buffers[dtBias.shardIndex],
                               let oNorm = layer.linearNormTensor, let oNormRaw = buffers[oNorm.shardIndex] {
                                let stateByteOffset = linIdx * Int(16 * 128 * 128) * MemoryLayout<Float>.stride
                                var aLogOff = aLog.offsetStart
                                var dtBiasOff = dtBias.offsetStart
                                var oNormOff = oNorm.offsetStart
                                var numHeads: UInt32 = 16
                                var headDim: UInt32 = 128
                                var lowerBound: Float = Float(modelConfig?.kdaLowerBound ?? -5.0)
                                var epsVal = eps
                                var isF32Params: UInt32 = (aLog.dtype.contains("F32") || (aLog.dtype.contains("FLOAT") && !aLog.dtype.contains("BF16"))) ? 1 : 0

                                layerEnc1.setComputePipelineState(kdaPipe)
                                layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 1)
                                layerEnc1.setBuffer(vVectorBuffer, offset: 0, index: 2)
                                layerEnc1.setBuffer(aVectorBuffer, offset: 0, index: 3)
                                layerEnc1.setBuffer(bVectorBuffer, offset: 0, index: 4)
                                layerEnc1.setBuffer(zGateBuffer, offset: 0, index: 5)
                                layerEnc1.setBuffer(aLogRaw, offset: 0, index: 6)
                                layerEnc1.setBuffer(dtBiasRaw, offset: 0, index: 7)
                                layerEnc1.setBuffer(oNormRaw, offset: 0, index: 8)
                                layerEnc1.setBuffer(sBuf, offset: stateByteOffset, index: 9)
                                layerEnc1.setBuffer(attnCtxBuffer, offset: 0, index: 10)
                                layerEnc1.setBytes(&aLogOff, length: MemoryLayout<UInt64>.stride, index: 11)
                                layerEnc1.setBytes(&dtBiasOff, length: MemoryLayout<UInt64>.stride, index: 12)
                                layerEnc1.setBytes(&oNormOff, length: MemoryLayout<UInt64>.stride, index: 13)
                                layerEnc1.setBytes(&numHeads, length: MemoryLayout<UInt32>.stride, index: 14)
                                layerEnc1.setBytes(&headDim, length: MemoryLayout<UInt32>.stride, index: 15)
                                layerEnc1.setBytes(&lowerBound, length: MemoryLayout<Float>.stride, index: 16)
                                layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 17)
                                layerEnc1.setBytes(&isF32Params, length: MemoryLayout<UInt32>.stride, index: 18)
                                layerEnc1.dispatchThreadgroups(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            // 5. Output projection: 2048 -> hiddenDim (1536) into attnOutBuffer
                            dispatchLinear(enc: layerEnc1, weight: layer.linearOutProjTensor ?? layer.oProjTensor, scale: nil, bias: nil, inBuf: attnCtxBuffer, outBuf: attnOutBuffer, inDim: 2048, outDim: hiddenDim)
                            layerEnc1.memoryBarrier(scope: .buffers)
                        } else {
                            // Linear Attention (GatedDeltaNet Recurrent State)
                            let linValHeads: UInt32 = UInt32(modelConfig?.effectiveLinearNumValueHeads ?? (layer.inProjA != nil && (layer.inProjA!.offsetEnd - layer.inProjA!.offsetStart) > 32 * 2560 ? 48 : 32))
                            let linKeyHeads: UInt32 = UInt32(modelConfig?.effectiveLinearNumKeyHeads ?? 16)
                            let qkvDim: UInt32 = (linKeyHeads + linKeyHeads + linValHeads) * 128
                            let zDim: UInt32 = linValHeads * 128
                            let aDim: UInt32 = linValHeads
                            let bDim: UInt32 = linValHeads

                            dispatchLinear(enc: layerEnc1, weight: layer.inProjQKV, scale: layer.inProjQKVScale, bias: layer.inProjQKVBias, inBuf: xNorm1Buffer, outBuf: qGateBuffer, inDim: hiddenDim, outDim: qkvDim)
                            layerEnc1.memoryBarrier(scope: .buffers)

                            if let conv1d = layer.conv1dTensor, let convRaw = buffers[conv1d.shardIndex],
                               let convPipe = causalConv1dPipeline, let convState = KVCacheManager.shared.convStateBuffer {
                                let convStateByteOffset = layer.linAttnIndex * Int(qkvDim) * 4 * MemoryLayout<Float>.stride
                                var cOff = conv1d.offsetStart
                                var numChannels: UInt32 = qkvDim
                                layerEnc1.setComputePipelineState(convPipe)
                                layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(convRaw, offset: 0, index: 1)
                                layerEnc1.setBuffer(convState, offset: convStateByteOffset, index: 2)
                                layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 3)
                                layerEnc1.setBytes(&cOff, length: MemoryLayout<UInt64>.stride, index: 4)
                                layerEnc1.setBytes(&numChannels, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(qkvDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, convPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            if let l2Pipe = l2NormQkPipeline {
                                var numHeads: UInt32 = linKeyHeads
                                var headDim: UInt32 = 128
                                layerEnc1.setComputePipelineState(l2Pipe)
                                layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                layerEnc1.setBytes(&numHeads, length: MemoryLayout<UInt32>.stride, index: 1)
                                layerEnc1.setBytes(&headDim, length: MemoryLayout<UInt32>.stride, index: 2)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), l2Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            dispatchLinear(enc: layerEnc1, weight: layer.inProjZ, scale: layer.inProjZScale, bias: layer.inProjZBias, inBuf: xNorm1Buffer, outBuf: zGateBuffer, inDim: hiddenDim, outDim: zDim)
                            dispatchLinear(enc: layerEnc1, weight: layer.inProjA, scale: layer.inProjAScale, bias: layer.inProjABias, inBuf: xNorm1Buffer, outBuf: aVectorBuffer, inDim: hiddenDim, outDim: aDim)
                            dispatchLinear(enc: layerEnc1, weight: layer.inProjB, scale: layer.inProjBScale, bias: layer.inProjBBias, inBuf: xNorm1Buffer, outBuf: bVectorBuffer, inDim: hiddenDim, outDim: bDim)
                            layerEnc1.memoryBarrier(scope: .buffers)

                            let useSigmoidGate = (modelConfig?.effectiveOutputGateType.lowercased() == "sigmoid")
                            let selectedStepPipe = useSigmoidGate
                                ? (gdnLinearAttnStepSigmoidPipeline ?? linearAttnStepSigmoidPipeline ?? gdnLinearAttnStepPipeline ?? linearAttnStepPipeline)
                                : (gdnLinearAttnStepPipeline ?? linearAttnStepPipeline)

                            if let linPipe = selectedStepPipe,
                               let sBuf = KVCacheManager.shared.linearStateBuffer,
                               let aLog = layer.aLogTensor, let aLogRaw = buffers[aLog.shardIndex],
                               let dtBias = layer.dtBiasTensor, let dtBiasRaw = buffers[dtBias.shardIndex],
                               let linNorm = layer.linearNormTensor, let linNormRaw = buffers[linNorm.shardIndex] {
                                let linIdx = layer.linAttnIndex
                                let stateByteOffset = linIdx * Int(linValHeads * 128 * 128) * MemoryLayout<Float>.stride
                                var aLogOff = aLog.offsetStart
                                var dtBiasOff = dtBias.offsetStart
                                var linNormOff = linNorm.offsetStart
                                var numValHeads: UInt32 = linValHeads
                                var numKeyHeads: UInt32 = linKeyHeads
                                var headDim: UInt32 = 128
                                var epsVal = eps
                                
                                layerEnc1.setComputePipelineState(linPipe)
                                layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(zGateBuffer, offset: 0, index: 1)
                                layerEnc1.setBuffer(aVectorBuffer, offset: 0, index: 2)
                                layerEnc1.setBuffer(bVectorBuffer, offset: 0, index: 3)
                                layerEnc1.setBuffer(aLogRaw, offset: 0, index: 4)
                                layerEnc1.setBuffer(dtBiasRaw, offset: 0, index: 5)
                                layerEnc1.setBuffer(linNormRaw, offset: 0, index: 6)
                                layerEnc1.setBuffer(sBuf, offset: stateByteOffset, index: 7)
                                layerEnc1.setBuffer(attnCtxBuffer, offset: 0, index: 8)
                                layerEnc1.setBytes(&aLogOff, length: MemoryLayout<UInt64>.stride, index: 9)
                                layerEnc1.setBytes(&dtBiasOff, length: MemoryLayout<UInt64>.stride, index: 10)
                                layerEnc1.setBytes(&linNormOff, length: MemoryLayout<UInt64>.stride, index: 11)
                                layerEnc1.setBytes(&numValHeads, length: MemoryLayout<UInt32>.stride, index: 12)
                                layerEnc1.setBytes(&numKeyHeads, length: MemoryLayout<UInt32>.stride, index: 13)
                                layerEnc1.setBytes(&headDim, length: MemoryLayout<UInt32>.stride, index: 14)
                                layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 15)
                                layerEnc1.dispatchThreadgroups(MTLSize(width: Int(numValHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            dispatchLinear(enc: layerEnc1, weight: layer.linearOutProjTensor ?? layer.oProjTensor, scale: layer.linearOutProjScale ?? layer.oScaleTensor, bias: layer.linearOutProjBias ?? layer.oBiasTensor, inBuf: attnCtxBuffer, outBuf: attnOutBuffer, inDim: zDim, outDim: hiddenDim)
                            layerEnc1.memoryBarrier(scope: .buffers)
                        }

                        if isGemma4 {
                            // Gemma 4 decoder layer (text):
                            //   hMid = x + post_attention_layernorm(attn)
                            //   dense = mlp(pre_feedforward_layernorm(hMid))
                            //   moe   = experts(pre_feedforward_layernorm_2(hMid), router(hMid))
                            //   h    = hMid + post_feedforward_layernorm(
                            //              post_feedforward_layernorm_1(dense) + post_feedforward_layernorm_2(moe))
                            //   out  = h * layer_scalar
                            let gemmaScale: Float = 1.0 / Float(Double(hiddenDim).squareRoot())
                            var hDimV = hiddenDim
                            var epsV = eps

                            // post_attention_layernorm(attnOut) -> xNorm2Buffer
                            if let norm2 = layer.norm2Tensor, let norm2Raw = buffers[norm2.shardIndex] {
                                var gOff = norm2.offsetStart
                                layerEnc1.setComputePipelineState(rmsnormPipeline)
                                layerEnc1.setBuffer(attnOutBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(norm2Raw, offset: 0, index: 1)
                                layerEnc1.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                                layerEnc1.setBytes(&gOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                layerEnc1.setBytes(&hDimV, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&epsV, length: MemoryLayout<Float>.stride, index: 5)
                                layerEnc1.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                                layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }
                            // hMid = currentH + xNorm2
                            layerEnc1.setComputePipelineState(addPipeline)
                            layerEnc1.setBuffer(currentH, offset: 0, index: 0)
                            layerEnc1.setBuffer(xNorm2Buffer, offset: 0, index: 1)
                            layerEnc1.setBuffer(hMidBuffer, offset: 0, index: 2)
                            layerEnc1.setBytes(&hDimV, length: MemoryLayout<UInt32>.stride, index: 3)
                            layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // pre_feedforward_layernorm(hMid) -> xNorm2Buffer
                            if let preFfn = layer.preFfnNormTensor, let preFfnRaw = buffers[preFfn.shardIndex] {
                                var gOff = preFfn.offsetStart
                                layerEnc1.setComputePipelineState(rmsnormPipeline)
                                layerEnc1.setBuffer(hMidBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(preFfnRaw, offset: 0, index: 1)
                                layerEnc1.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                                layerEnc1.setBytes(&gOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                layerEnc1.setBytes(&hDimV, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&epsV, length: MemoryLayout<Float>.stride, index: 5)
                                layerEnc1.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                                layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            // Dense shared MLP: GeGLU(gate,up)(xNorm2) -> interBuffer; down -> hMlpBuffer
                            layerEnc1.setComputePipelineState(clearPipeline)
                            layerEnc1.setBuffer(hMlpBuffer, offset: 0, index: 0)
                            layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)
                            if let dG = layer.denseGateWeight, let dU = layer.denseUpWeight, let dD = layer.denseDownWeight,
                               let dGRaw = buffers[dG.shardIndex], let dURaw = buffers[dU.shardIndex], let dDRaw = buffers[dD.shardIndex],
                               let geluPipe = bf16GeluGateUpSimdPipeline ?? bf16GeluGateUpPipeline {
                                var gOff = dG.offsetStart
                                var uOff = dU.offsetStart
                                var dOff = dD.offsetStart
                                var interDimV = layer.intermediateDim
                                var pk: Float = 1.0
                                layerEnc1.setComputePipelineState(geluPipe)
                                layerEnc1.setBuffer(dGRaw, offset: 0, index: 0)
                                layerEnc1.setBuffer(dURaw, offset: 0, index: 1)
                                layerEnc1.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                                layerEnc1.setBuffer(interBuffer, offset: 0, index: 3)
                                layerEnc1.setBytes(&gOff, length: MemoryLayout<UInt64>.stride, index: 4)
                                layerEnc1.setBytes(&uOff, length: MemoryLayout<UInt64>.stride, index: 5)
                                layerEnc1.setBytes(&hDimV, length: MemoryLayout<UInt32>.stride, index: 6)
                                layerEnc1.setBytes(&interDimV, length: MemoryLayout<UInt32>.stride, index: 7)
                                if bf16GeluGateUpSimdPipeline != nil {
                                    layerEnc1.dispatchThreadgroups(MTLSize(width: Int(interDimV), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                } else {
                                    layerEnc1.dispatchThreads(MTLSize(width: Int(interDimV), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(interDimV), geluPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                }
                                layerEnc1.memoryBarrier(scope: .buffers)
                                let downPipe = bf16DownSimdPipeline ?? bf16DownPipeline
                                if let downPipe {
                                    layerEnc1.setComputePipelineState(downPipe)
                                    layerEnc1.setBuffer(dDRaw, offset: 0, index: 0)
                                    layerEnc1.setBuffer(interBuffer, offset: 0, index: 1)
                                    layerEnc1.setBuffer(hMlpBuffer, offset: 0, index: 2)
                                    layerEnc1.setBytes(&dOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                    layerEnc1.setBytes(&interDimV, length: MemoryLayout<UInt32>.stride, index: 4)
                                    layerEnc1.setBytes(&hDimV, length: MemoryLayout<UInt32>.stride, index: 5)
                                    layerEnc1.setBytes(&pk, length: MemoryLayout<Float>.stride, index: 6)
                                    if bf16DownSimdPipeline != nil {
                                        layerEnc1.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    } else {
                                        layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), downPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    }
                                    layerEnc1.memoryBarrier(scope: .buffers)
                                }
                            }
                            // post_feedforward_layernorm_1(dense) in place on hMlpBuffer
                            if let pf1 = layer.postFfnNorm1Tensor, let pf1Raw = buffers[pf1.shardIndex] {
                                var gOff = pf1.offsetStart
                                layerEnc1.setComputePipelineState(rmsnormPipeline)
                                layerEnc1.setBuffer(hMlpBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(pf1Raw, offset: 0, index: 1)
                                layerEnc1.setBuffer(hMlpBuffer, offset: 0, index: 2)
                                layerEnc1.setBytes(&gOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                layerEnc1.setBytes(&hDimV, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&epsV, length: MemoryLayout<Float>.stride, index: 5)
                                layerEnc1.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                                layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            // Router: rmsnorm_no_scale(hMid) * router.scale * hidden^-0.5 -> bVectorBuffer
                            if let vNormPipe = rmsnormNoScalePipeline {
                                var dimV = hiddenDim
                                var eV = eps
                                layerEnc1.setComputePipelineState(vNormPipe)
                                layerEnc1.setBuffer(hMidBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(attnOutBuffer, offset: 0, index: 1)
                                layerEnc1.setBytes(&dimV, length: MemoryLayout<UInt32>.stride, index: 2)
                                layerEnc1.setBytes(&eV, length: MemoryLayout<Float>.stride, index: 3)
                                layerEnc1.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                                layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }
                            if let rsv = layer.routerScaleVec, let rsvRaw = buffers[rsv.shardIndex], let mulPipe = mulByBf16VectorPipeline {
                                var rsvOff = rsv.offsetStart
                                var dimV = hiddenDim
                                var sc = gemmaScale
                                layerEnc1.setComputePipelineState(mulPipe)
                                layerEnc1.setBuffer(attnOutBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(rsvRaw, offset: 0, index: 1)
                                layerEnc1.setBuffer(bVectorBuffer, offset: 0, index: 2)
                                layerEnc1.setBytes(&rsvOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                layerEnc1.setBytes(&dimV, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&sc, length: MemoryLayout<Float>.stride, index: 5)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), mulPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }
                            if let routerTensor = layer.routerTensor, let routerRaw = buffers[routerTensor.shardIndex] {
                                var rOffset = routerTensor.offsetStart
                                var nExp = numExperts
                                var kVal: UInt32 = UInt32(modelConfig?.effectiveNumExpertsPerTok ?? 8)
                                layerEnc1.setComputePipelineState(routerPipeline)
                                layerEnc1.setBuffer(routerRaw, offset: 0, index: 0)
                                layerEnc1.setBuffer(bVectorBuffer, offset: 0, index: 1)
                                layerEnc1.setBuffer(routerIndicesBuffer, offset: 0, index: 2)
                                layerEnc1.setBuffer(routerWeightsBuffer, offset: 0, index: 3)
                                layerEnc1.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                                layerEnc1.setBytes(&hDimV, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc1.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 6)
                                layerEnc1.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 7)
                                layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: Int(numExperts), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                                // per-expert routing scale
                                if let pes = layer.routerPerExpertScale, let pesRaw = buffers[pes.shardIndex], let pesPipe = perExpertScalePipeline {
                                    var pesOff = pes.offsetStart
                                    var kTop = kVal
                                    layerEnc1.setComputePipelineState(pesPipe)
                                    layerEnc1.setBuffer(routerWeightsBuffer, offset: 0, index: 0)
                                    layerEnc1.setBuffer(routerIndicesBuffer, offset: 0, index: 1)
                                    layerEnc1.setBuffer(pesRaw, offset: 0, index: 2)
                                    layerEnc1.setBytes(&pesOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                    layerEnc1.setBytes(&kTop, length: MemoryLayout<UInt32>.stride, index: 4)
                                    layerEnc1.dispatchThreads(MTLSize(width: Int(kVal), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(kVal), pesPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)
                                }
                            }

                            layerEnc1.endEncoding()
                            let tGemmaRouter = CFAbsoluteTimeGetCurrent()
                            activeCmd.commit()
                            activeCmd.waitUntilCompleted()
                            diagRouterGpuMs += (CFAbsoluteTimeGetCurrent() - tGemmaRouter) * 1000.0
                            if let err = activeCmd.error {
                                print("❌ [METAL ERROR] gemma4 router cmd failed at layer \(l): \(err)")
                                return false
                            }

                            // Read selected experts
                            let gemmaTopK = max(1, Int(modelConfig?.effectiveNumExpertsPerTok ?? 8))
                            var gemmaActiveExperts: [(id: Int, weight: Float)] = []
                            let gIndPtr = routerIndicesBuffer.contents().bindMemory(to: UInt32.self, capacity: gemmaTopK)
                            let gWPtr = routerWeightsBuffer.contents().bindMemory(to: Float.self, capacity: gemmaTopK)
                            for i in 0..<gemmaTopK {
                                let w = gWPtr[i]
                                if w > 0.00001 { gemmaActiveExperts.append((id: Int(gIndPtr[i]), weight: w)) }
                            }
                            if !gemmaActiveExperts.isEmpty {
                                WorkingSetManager.shared.touchAndEvict(layer: l, activeExpertIds: gemmaActiveExperts.map { $0.id }, mode: budgetMode, shardBuffers: buffers, isPrefill: !computeLogits)
                            }

                            // FIX #8: predict layer l+1's experts from temporal history; the
                            // kick is issued after this layer's MoE commit so the retire wait
                            // overlaps GPU execution.
                            var deferredPredictedExperts: [Int]? = nil
                            if speculativePrefetchEnabled {
                                let activeIds = gemmaActiveExperts.map { $0.id }
                                if l > 0, let prevIds = previousLayerActiveExperts[l - 1] {
                                    WorkingSetManager.shared.transitionTracker.recordTransition(fromLayer: l - 1, fromExperts: prevIds, toLayer: l, toExperts: activeIds)
                                }
                                layerExpertHistory[l] = [previousLayerActiveExperts[l] ?? [], layerExpertHistory[l]?.first ?? []].filter { !$0.isEmpty }
                                previousLayerActiveExperts[l] = activeIds
                                if l + 1 < actualLayers, packedPrefetchAdaptiveEnabled {
                                    let budget = min(10, max(0, (expertStagingSize / packedExpertSize) - maxTopK))
                                    var seen = Set<Int>()
                                    var predicted: [Int] = []
                                    for hist in [previousLayerActiveExperts[l + 1] ?? [], layerExpertHistory[l + 1]?.first ?? []] {
                                        for id in hist where !seen.contains(id) && predicted.count < budget {
                                            seen.insert(id)
                                            predicted.append(id)
                                        }
                                    }
                                    let markov = WorkingSetManager.shared.predictNextLayerExperts(currentLayer: l, currentActiveExperts: activeIds, topN: 10)
                                    for id in markov where !seen.contains(id) && predicted.count < budget {
                                        seen.insert(id)
                                        predicted.append(id)
                                    }
                                    deferredPredictedExperts = predicted
                                }
                            }

                            guard let gemmaMoeCmd = commandQueue.makeCommandBuffer(),
                                  let gemmaEnc = gemmaMoeCmd.makeComputeCommandEncoder() else {
                                print("❌ [METAL] token forward: gemma MoE command buffer failed (layer \(l), step \(step))")
                                return false
                            }
                            // pre_feedforward_layernorm_2(hMid) -> xNorm2Buffer
                            if let pf2 = layer.preFfnNorm2Tensor, let pf2Raw = buffers[pf2.shardIndex] {
                                var gOff = pf2.offsetStart
                                gemmaEnc.setComputePipelineState(rmsnormPipeline)
                                gemmaEnc.setBuffer(hMidBuffer, offset: 0, index: 0)
                                gemmaEnc.setBuffer(pf2Raw, offset: 0, index: 1)
                                gemmaEnc.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                                gemmaEnc.setBytes(&gOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                gemmaEnc.setBytes(&hDimV, length: MemoryLayout<UInt32>.stride, index: 4)
                                gemmaEnc.setBytes(&epsV, length: MemoryLayout<Float>.stride, index: 5)
                                gemmaEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                                gemmaEnc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                                gemmaEnc.memoryBarrier(scope: .buffers)
                            }

                            // Expert accumulation into xNorm1Buffer
                            gemmaEnc.setComputePipelineState(clearPipeline)
                            gemmaEnc.setBuffer(xNorm1Buffer, offset: 0, index: 0)
                            gemmaEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            gemmaEnc.memoryBarrier(scope: .buffers)

                            // FlashMoE-packed experts live in a 46 GB file-backed mmap. Reading
                            // them directly from that mapping makes the GPU fault pages in on
                            // demand (documented collapse to ~0.3 tok/s), so pread this token's
                            // selected experts into the pinned staging buffer first — the same
                            // anonymous-shared-memory path the dense/standard MoE branch uses.
                            let gemmaExpertSize = Int(loadedLayout?.expert_size ?? 0)
                            var gemmaSlotOf: [Int: Int] = [:]
                            var gemmaStaging: MTLBuffer? = nil
                            if let packedDir = packedExpertsDir, gemmaExpertSize > 0,
                               let gemmaFd = ExpertIOThreadPool.shared.getOrOpenLayerFD(layerIndex: l, packedExpertsDir: packedDir) {
                                let bufIdx = l & 1
                                let stagingBuf = (bufIdx == 0) ? expertStagingBuffer : expertStagingBufferB
                                let stagingCapacity = expertStagingSize / gemmaExpertSize
                                gemmaStaging = stagingBuf

                                // FIX #8: consume this layer's predicted experts from the in-flight
                                // kick (waiting per needed slot); synchronously pread only the misses.
                                var kick: PackedExpertKick? = nil
                                var staleKick: PackedExpertKick? = nil
                                packedPrefetchLock.lock()
                                if let k = packedKickInFlight[bufIdx] {
                                    if k.targetLayer == l {
                                        kick = k
                                    } else {
                                        staleKick = k
                                        packedKickInFlight[bufIdx] = nil
                                    }
                                }
                                packedPrefetchLock.unlock()
                                if let old = staleKick { _ = old.allDoneSem.wait(timeout: .now() + 30.0) }

                                var nextSlot: Int
                                var missList: [(id: Int, weight: Float)] = []
                                if let k = kick {
                                    nextSlot = k.slotBase
                                    for exp in gemmaActiveExperts {
                                        if let idx = k.candidates.firstIndex(of: exp.id), idx < k.slotSems.count {
                                            _ = k.slotSems[idx].wait(timeout: .now() + 15.0)
                                            packedPrefetchLock.lock()
                                            let ok = k.slotOK[idx]
                                            packedPrefetchLock.unlock()
                                            if ok { gemmaSlotOf[exp.id] = idx } else { missList.append(exp) }
                                        } else {
                                            missList.append(exp)
                                        }
                                    }
                                } else {
                                    nextSlot = 0
                                    missList = gemmaActiveExperts
                                }

                                var missTasks: [ExpertPreadTask] = []
                                for exp in missList {
                                    guard nextSlot < stagingCapacity else {
                                        print("❌ [Prefetch] slot overflow at gemma layer \(l) — aborting generation")
                                        return false
                                    }
                                    let slot = nextSlot
                                    nextSlot += 1
                                    gemmaSlotOf[exp.id] = slot
                                    missTasks.append(ExpertPreadTask(fd: gemmaFd,
                                                                     dst: stagingBuf.contents().advanced(by: slot * gemmaExpertSize),
                                                                     offset: off_t(exp.id * gemmaExpertSize),
                                                                     size: gemmaExpertSize))
                                }
                                if !missTasks.isEmpty {
                                    ExpertIOThreadPool.shared.dispatchSync(tasks: &missTasks)
                                }
                            }
                            // Component offsets within one packed expert slot. Gemma packs the
                            // fused gate+up under "up_proj" (gate = first half, up = second half)
                            // with no separate gate_proj; down_proj follows.
                            let gemmaCompUpOff = UInt64(loadedLayout?.components.first(where: { $0.name.contains("up_proj") && !$0.name.contains("scale") && !$0.name.contains("bias") })?.offset ?? 0)
                            let gemmaCompDownOff = UInt64(loadedLayout?.components.first(where: { $0.name.contains("down_proj") && !$0.name.contains("scale") && !$0.name.contains("bias") })?.offset ?? 0)

                            if let geluPipe = bf16GeluGateUpSimdPipeline ?? bf16GeluGateUpPipeline {
                                let downPipe = bf16DownSimdPipeline ?? bf16DownPipeline
                                var interDimV = gemmaExpertInterDim
                                let upByteOffset = UInt64(gemmaExpertInterDim) * UInt64(hiddenDim) * 2
                                for (expId, weight) in gemmaActiveExperts {
                                    // Packed: both halves sit in one staging slot. Unpacked
                                    // safetensors: fall back to the mmap'd per-expert tensors.
                                    let fusedRaw: MTLBuffer
                                    let downRaw: MTLBuffer
                                    let gOff: UInt64
                                    let uOff: UInt64
                                    let dOff: UInt64
                                    if let staging = gemmaStaging, let slot = gemmaSlotOf[expId] {
                                        let slotOffset = UInt64(slot * gemmaExpertSize)
                                        fusedRaw = staging
                                        downRaw = staging
                                        gOff = slotOffset + gemmaCompUpOff
                                        uOff = slotOffset + gemmaCompUpOff + upByteOffset
                                        dOff = slotOffset + gemmaCompDownOff
                                    } else if let fused = layer.expertFusedGateUpWeights[expId], let down = layer.expertDownWeights[expId],
                                              let fusedBuf = buffers[fused.shardIndex], let downBuf = buffers[down.shardIndex] {
                                        fusedRaw = fusedBuf
                                        downRaw = downBuf
                                        gOff = fused.offsetStart
                                        uOff = fused.offsetStart + upByteOffset
                                        dOff = down.offsetStart
                                    } else { continue }
                                    var gOffV = gOff
                                    var uOffV = uOff
                                    var dOffV = dOff
                                    var pk = weight
                                    gemmaEnc.setComputePipelineState(geluPipe)
                                    gemmaEnc.setBuffer(fusedRaw, offset: 0, index: 0)
                                    gemmaEnc.setBuffer(fusedRaw, offset: 0, index: 1)
                                    gemmaEnc.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                                    gemmaEnc.setBuffer(interBuffer, offset: 0, index: 3)
                                    gemmaEnc.setBytes(&gOffV, length: MemoryLayout<UInt64>.stride, index: 4)
                                    gemmaEnc.setBytes(&uOffV, length: MemoryLayout<UInt64>.stride, index: 5)
                                    gemmaEnc.setBytes(&hDimV, length: MemoryLayout<UInt32>.stride, index: 6)
                                    gemmaEnc.setBytes(&interDimV, length: MemoryLayout<UInt32>.stride, index: 7)
                                    if bf16GeluGateUpSimdPipeline != nil {
                                        gemmaEnc.dispatchThreadgroups(MTLSize(width: Int(interDimV), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    } else {
                                        gemmaEnc.dispatchThreads(MTLSize(width: Int(interDimV), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(interDimV), geluPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    }
                                    gemmaEnc.memoryBarrier(scope: .buffers)
                                    if let downPipe {
                                        gemmaEnc.setComputePipelineState(downPipe)
                                        gemmaEnc.setBuffer(downRaw, offset: 0, index: 0)
                                        gemmaEnc.setBuffer(interBuffer, offset: 0, index: 1)
                                        gemmaEnc.setBuffer(xNorm1Buffer, offset: 0, index: 2)
                                        gemmaEnc.setBytes(&dOffV, length: MemoryLayout<UInt64>.stride, index: 3)
                                        gemmaEnc.setBytes(&interDimV, length: MemoryLayout<UInt32>.stride, index: 4)
                                        gemmaEnc.setBytes(&hDimV, length: MemoryLayout<UInt32>.stride, index: 5)
                                        gemmaEnc.setBytes(&pk, length: MemoryLayout<Float>.stride, index: 6)
                                        if bf16DownSimdPipeline != nil {
                                            gemmaEnc.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        } else {
                                            gemmaEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), downPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                        }
                                        gemmaEnc.memoryBarrier(scope: .buffers)
                                    }
                                }
                            }

                            // post_feedforward_layernorm_2(moe) -> bVectorBuffer
                            if let pf2 = layer.postFfnNorm2Tensor, let pf2Raw = buffers[pf2.shardIndex] {
                                var gOff = pf2.offsetStart
                                gemmaEnc.setComputePipelineState(rmsnormPipeline)
                                gemmaEnc.setBuffer(xNorm1Buffer, offset: 0, index: 0)
                                gemmaEnc.setBuffer(pf2Raw, offset: 0, index: 1)
                                gemmaEnc.setBuffer(bVectorBuffer, offset: 0, index: 2)
                                gemmaEnc.setBytes(&gOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                gemmaEnc.setBytes(&hDimV, length: MemoryLayout<UInt32>.stride, index: 4)
                                gemmaEnc.setBytes(&epsV, length: MemoryLayout<Float>.stride, index: 5)
                                gemmaEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                                gemmaEnc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                                gemmaEnc.memoryBarrier(scope: .buffers)
                            }

                            // hMlp = postFfnNorm1(dense) + postFfnNorm2(moe)
                            gemmaEnc.setComputePipelineState(addPipeline)
                            gemmaEnc.setBuffer(hMlpBuffer, offset: 0, index: 0)
                            gemmaEnc.setBuffer(bVectorBuffer, offset: 0, index: 1)
                            gemmaEnc.setBuffer(hMlpBuffer, offset: 0, index: 2)
                            gemmaEnc.setBytes(&hDimV, length: MemoryLayout<UInt32>.stride, index: 3)
                            gemmaEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            gemmaEnc.memoryBarrier(scope: .buffers)

                            // post_feedforward_layernorm(combined) -> xNorm2Buffer
                            if let pf = layer.postFfnNormTensor, let pfRaw = buffers[pf.shardIndex] {
                                var gOff = pf.offsetStart
                                gemmaEnc.setComputePipelineState(rmsnormPipeline)
                                gemmaEnc.setBuffer(hMlpBuffer, offset: 0, index: 0)
                                gemmaEnc.setBuffer(pfRaw, offset: 0, index: 1)
                                gemmaEnc.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                                gemmaEnc.setBytes(&gOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                gemmaEnc.setBytes(&hDimV, length: MemoryLayout<UInt32>.stride, index: 4)
                                gemmaEnc.setBytes(&epsV, length: MemoryLayout<Float>.stride, index: 5)
                                gemmaEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                                gemmaEnc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                                gemmaEnc.memoryBarrier(scope: .buffers)
                            }
                            // nextH = hMid + xNorm2
                            gemmaEnc.setComputePipelineState(addPipeline)
                            gemmaEnc.setBuffer(hMidBuffer, offset: 0, index: 0)
                            gemmaEnc.setBuffer(xNorm2Buffer, offset: 0, index: 1)
                            gemmaEnc.setBuffer(nextH, offset: 0, index: 2)
                            gemmaEnc.setBytes(&hDimV, length: MemoryLayout<UInt32>.stride, index: 3)
                            gemmaEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            gemmaEnc.memoryBarrier(scope: .buffers)

                            // nextH *= layer_scalar
                            if let ls = layer.layerScalarTensor, let lsRaw = buffers[ls.shardIndex], let scalePipe = scaleVectorInplacePipeline {
                                var dimV = hiddenDim
                                let lsBits = lsRaw.contents().load(fromByteOffset: Int(ls.offsetStart), as: UInt16.self)
                                var scalarVal = Float(bitPattern: UInt32(lsBits) << 16)
                                gemmaEnc.setComputePipelineState(scalePipe)
                                gemmaEnc.setBuffer(nextH, offset: 0, index: 0)
                                gemmaEnc.setBytes(&dimV, length: MemoryLayout<UInt32>.stride, index: 1)
                                gemmaEnc.setBytes(&scalarVal, length: MemoryLayout<Float>.stride, index: 2)
                                gemmaEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), scalePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                gemmaEnc.memoryBarrier(scope: .buffers)
                            }

                            gemmaEnc.endEncoding()
                            let tGemmaMoe = CFAbsoluteTimeGetCurrent()
                            gemmaMoeCmd.commit()

                            // FIX #8: kick layer l+1's predicted expert reads while this layer's
                            // MoE runs on the GPU (the retire wait inside overlaps GPU execution).
                            if let predicted = deferredPredictedExperts, !predicted.isEmpty,
                               let packedDir = packedExpertsDir,
                               let nextFd = ExpertIOThreadPool.shared.getOrOpenLayerFD(layerIndex: l + 1, packedExpertsDir: packedDir) {
                                kickPackedPrefetch(predicted, targetLayer: l + 1, fd: nextFd)
                            }

                            gemmaMoeCmd.waitUntilCompleted()
                            diagMoeGpuMs += (CFAbsoluteTimeGetCurrent() - tGemmaMoe) * 1000.0
                            if let err = gemmaMoeCmd.error {
                                print("❌ [METAL ERROR] gemma4 moe cmd failed at layer \(l): \(err)")
                                return false
                            }

                            guard let nextCmd = commandQueue.makeCommandBuffer() else {
                                print("❌ [METAL] token forward: command buffer allocation failed at gemma layer \(l) (step \(step))")
                                return false
                            }
                            activeCmd = nextCmd
                        } else {
                        // Step 3: Residual Connection 1 (hMid = currentH + attnOut or 4-stream inject)
                        if (layer.attnHcInjectWeight != nil || layer.attnHcDownWeight != nil),
                           let injPipe = hcInjectPipeline {
                            layerEnc1.setComputePipelineState(injPipe)
                            layerEnc1.setBuffer(hcStreamsBuffer, offset: 0, index: 0)
                            layerEnc1.setBuffer(attnOutBuffer, offset: 0, index: 1)
                            layerEnc1.setBuffer(hcInjectScaleBuffer, offset: 0, index: 2)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                            layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), injPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)
                        } else {
                            layerEnc1.setComputePipelineState(addPipeline)
                            layerEnc1.setBuffer(currentH, offset: 0, index: 0)
                            layerEnc1.setBuffer(attnOutBuffer, offset: 0, index: 1)
                            layerEnc1.setBuffer(hMidBuffer, offset: 0, index: 2)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                            layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)
                        }

                        // Step 4: Post-Attention RMSNorm / HC Stream Mixing (hMid/hcStreams -> xNorm2)
                        if let hcDown = layer.mlpHcDownWeight, let hcDownRaw = buffers[hcDown.shardIndex],
                           let hcUp = layer.mlpHcUpWeight, let hcUpRaw = buffers[hcUp.shardIndex],
                           let hcNorm = layer.mlpHcNorm, let hcNormRaw = buffers[hcNorm.shardIndex],
                           let normPipe = hcNormPipeline, let downPipe = hcDownProjPipeline, let upPipe = hcUpBlendPipeline {
                            var nOff = hcNorm.offsetStart
                            var dOff = hcDown.offsetStart
                            var uOff = hcUp.offsetStart
                            var totDim: UInt32 = 4 * hiddenDim
                            var rank: UInt32 = UInt32(modelConfig?.effectiveHcLowrank ?? 320)
                            var epsVal = eps

                            // 4a. Group RMSNorm on 4 Streams
                            layerEnc1.setComputePipelineState(normPipe)
                            layerEnc1.setBuffer(hcStreamsBuffer, offset: 0, index: 0)
                            layerEnc1.setBuffer(hcNormRaw, offset: 0, index: 1)
                            layerEnc1.setBuffer(hcNormedBuffer, offset: 0, index: 2)
                            layerEnc1.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: 4, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 4b. Down-Projection
                            layerEnc1.setComputePipelineState(downPipe)
                            layerEnc1.setBuffer(hcNormedBuffer, offset: 0, index: 0)
                            layerEnc1.setBuffer(hcDownRaw, offset: 0, index: 1)
                            layerEnc1.setBuffer(hcBottleneckBuffer, offset: 0, index: 2)
                            layerEnc1.setBytes(&dOff, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc1.setBytes(&totDim, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc1.setBytes(&rank, length: MemoryLayout<UInt32>.stride, index: 5)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: Int(rank), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 4c. Up-Projection and Stream Blend
                            layerEnc1.setComputePipelineState(upPipe)
                            layerEnc1.setBuffer(hcNormedBuffer, offset: 0, index: 0)
                            layerEnc1.setBuffer(hcBottleneckBuffer, offset: 0, index: 1)
                            layerEnc1.setBuffer(hcUpRaw, offset: 0, index: 2)
                            layerEnc1.setBuffer(xNorm2Buffer, offset: 0, index: 3)
                            layerEnc1.setBytes(&uOff, length: MemoryLayout<UInt64>.stride, index: 4)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 5)
                            layerEnc1.setBytes(&rank, length: MemoryLayout<UInt32>.stride, index: 6)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 4d. Compute Injection Scale if mlpHcInject exists
                            if let mlpHcInj = layer.mlpHcInjectWeight, let mlpHcInjRaw = buffers[mlpHcInj.shardIndex],
                               let injScalePipe = hcInjectScalePipeline {
                                var iOff = mlpHcInj.offsetStart
                                layerEnc1.setComputePipelineState(injScalePipe)
                                layerEnc1.setBuffer(hcNormedBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(mlpHcInjRaw, offset: 0, index: 1)
                                layerEnc1.setBuffer(hcInjectScaleBuffer, offset: 0, index: 2)
                                layerEnc1.setBytes(&iOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                layerEnc1.setBytes(&totDim, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.dispatchThreadgroups(MTLSize(width: 4, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }
                        } else if let norm2 = layer.norm2Tensor, let norm2Raw = buffers[norm2.shardIndex] {
                            var gammaOff = norm2.offsetStart
                            var epsVal = eps
                            let isNorm2F16 = (norm2.dtype.contains("F16") || norm2.dtype.contains("HALF") || norm2.dtype.contains("FLOAT16")) && !norm2.dtype.contains("BF16") && !norm2.dtype.contains("BFLOAT")
                            let norm2Pipe: MTLComputePipelineState
                            if isRMSNormOffset {
                                norm2Pipe = (isNorm2F16 && rmsnormOffsetF16Pipeline != nil) ? rmsnormOffsetF16Pipeline! : (rmsnormOffsetPipeline ?? rmsnormPipeline)
                            } else {
                                norm2Pipe = (isNorm2F16 && rmsnormF16Pipeline != nil) ? rmsnormF16Pipeline! : rmsnormPipeline
                            }
                            layerEnc1.setComputePipelineState(norm2Pipe)
                            layerEnc1.setBuffer(hMidBuffer, offset: 0, index: 0)
                            layerEnc1.setBuffer(norm2Raw, offset: 0, index: 1)
                            layerEnc1.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                            layerEnc1.setBytes(&gammaOff, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                            layerEnc1.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)
                        }

                        // Step 5: Dense Feed-Forward OR MoE
                        let intermediateDim = layer.intermediateDim

                        if layer.mlpType == .denseMlp || layer.routerTensor == nil {
                            // Dense SwiGLU Feed-Forward (Unified within layerEnc1)
                            layerEnc1.setComputePipelineState(clearPipeline)
                            layerEnc1.setBuffer(hMlpBuffer, offset: 0, index: 0)
                            layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            if let gateW = layer.denseGateWeight,
                               let upW = layer.denseUpWeight,
                               let downW = layer.denseDownWeight {
                                let gateS = layer.denseGateScale
                                let gateB = layer.denseGateBias
                                let upS = layer.denseUpScale
                                let upB = layer.denseUpBias
                                let downS = layer.denseDownScale
                                let downB = layer.denseDownBias

                                if modelConfig?.isSparkModel == true,
                                   let dRaw = buffers[downW.shardIndex],
                                   let geluSimd = bf16GeluGateUpSimdPipeline ?? bf16GeluGateUpPipeline {
                                    // Spark 2.5: GeGLU (exact GeLU gate * up) + down-projection accumulate
                                    var gWOff = gateW.offsetStart
                                    var uWOff = upW.offsetStart
                                    var dWOff = downW.offsetStart
                                    var hDimVal = hiddenDim
                                    var interDimVal = intermediateDim
                                    var pkVal: Float = 1.0

                                    layerEnc1.setComputePipelineState(geluSimd)
                                    layerEnc1.setBuffer(buffers[gateW.shardIndex]!, offset: 0, index: 0)
                                    layerEnc1.setBuffer(buffers[upW.shardIndex]!, offset: 0, index: 1)
                                    layerEnc1.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                                    layerEnc1.setBuffer(interBuffer, offset: 0, index: 3)
                                    layerEnc1.setBytes(&gWOff, length: MemoryLayout<UInt64>.stride, index: 4)
                                    layerEnc1.setBytes(&uWOff, length: MemoryLayout<UInt64>.stride, index: 5)
                                    layerEnc1.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 6)
                                    layerEnc1.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 7)
                                    if bf16GeluGateUpSimdPipeline != nil {
                                        layerEnc1.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    } else {
                                        layerEnc1.dispatchThreads(MTLSize(width: Int(intermediateDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(intermediateDim), geluSimd.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    }
                                    layerEnc1.memoryBarrier(scope: .buffers)

                                    if let downSimd = bf16DownSimdPipeline {
                                        layerEnc1.setComputePipelineState(downSimd)
                                        layerEnc1.setBuffer(dRaw, offset: 0, index: 0)
                                        layerEnc1.setBuffer(interBuffer, offset: 0, index: 1)
                                        layerEnc1.setBuffer(hMlpBuffer, offset: 0, index: 2)
                                        layerEnc1.setBytes(&dWOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                        layerEnc1.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 4)
                                        layerEnc1.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc1.setBytes(&pkVal, length: MemoryLayout<Float>.stride, index: 6)
                                        layerEnc1.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    } else if let downUnq = bf16DownPipeline {
                                        layerEnc1.setComputePipelineState(downUnq)
                                        layerEnc1.setBuffer(dRaw, offset: 0, index: 0)
                                        layerEnc1.setBuffer(interBuffer, offset: 0, index: 1)
                                        layerEnc1.setBuffer(hMlpBuffer, offset: 0, index: 2)
                                        layerEnc1.setBytes(&dWOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                        layerEnc1.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 4)
                                        layerEnc1.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc1.setBytes(&pkVal, length: MemoryLayout<Float>.stride, index: 6)
                                        layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), downUnq.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    }
                                } else {
                                    dispatchExpertMlp(
                                        enc: layerEnc1,
                                        gateW: gateW,
                                        gateS: gateS,
                                        gateB: gateB,
                                        upW: upW,
                                        upS: upS,
                                        upB: upB,
                                        downW: downW,
                                        downS: downS,
                                        downB: downB,
                                        inBuf: xNorm2Buffer,
                                        interBuf: interBuffer,
                                        accumBuf: hMlpBuffer,
                                        inDim: hiddenDim,
                                        interDim: intermediateDim,
                                        routingWeight: 1.0
                                    )
                                }
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            // Step 6: Residual Connection 2 (nextH = hMid + hMlp or 4-stream inject)
                            if (layer.mlpHcInjectWeight != nil || layer.mlpHcDownWeight != nil),
                               let injPipe = hcInjectPipeline {
                                layerEnc1.setComputePipelineState(injPipe)
                                layerEnc1.setBuffer(hcStreamsBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(hMlpBuffer, offset: 0, index: 1)
                                layerEnc1.setBuffer(hcInjectScaleBuffer, offset: 0, index: 2)
                                layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), injPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            } else {
                                layerEnc1.setComputePipelineState(addPipeline)
                                layerEnc1.setBuffer(hMidBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(hMlpBuffer, offset: 0, index: 1)
                                layerEnc1.setBuffer(nextH, offset: 0, index: 2)
                                layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            layerEnc1.endEncoding()
                        } else {
                            // MoE Model with dynamic routing
                            if let routerTensor = layer.routerTensor, let routerRaw = buffers[routerTensor.shardIndex] {
                                var rOffset = routerTensor.offsetStart
                                var nExp = numExperts
                                var kVal: UInt32 = UInt32(modelConfig?.effectiveNumExpertsPerTok ?? (numExperts >= 512 ? 10 : 8))
                                var grp: UInt32 = 64

                                if let rScale = layer.routerScale, let rBias = layer.routerBias,
                                   let rScaleRaw = buffers[rScale.shardIndex], let rBiasRaw = buffers[rBias.shardIndex] {
                                    var sOffset = rScale.offsetStart
                                    var bOffset = rBias.offsetStart

                                    if (routerTensor.shapeDisplay.contains("512") || numExperts >= 512), let rQ8 = routerQ8Pipeline {
                                        layerEnc1.setComputePipelineState(rQ8)
                                        layerEnc1.setBuffer(routerRaw, offset: 0, index: 0)
                                        layerEnc1.setBuffer(rScaleRaw, offset: 0, index: 1)
                                        layerEnc1.setBuffer(rBiasRaw, offset: 0, index: 2)
                                        layerEnc1.setBuffer(xNorm2Buffer, offset: 0, index: 3)
                                        layerEnc1.setBuffer(routerIndicesBuffer, offset: 0, index: 4)
                                        layerEnc1.setBuffer(routerWeightsBuffer, offset: 0, index: 5)
                                        layerEnc1.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                                        layerEnc1.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                                        layerEnc1.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 8)
                                        layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 9)
                                        layerEnc1.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 10)
                                        layerEnc1.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 11)
                                        layerEnc1.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 12)
                                        layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
                                    } else if let rQ4 = routerQ4Pipeline {
                                        layerEnc1.setComputePipelineState(rQ4)
                                        layerEnc1.setBuffer(routerRaw, offset: 0, index: 0)
                                        layerEnc1.setBuffer(rScaleRaw, offset: 0, index: 1)
                                        layerEnc1.setBuffer(rBiasRaw, offset: 0, index: 2)
                                        layerEnc1.setBuffer(xNorm2Buffer, offset: 0, index: 3)
                                        layerEnc1.setBuffer(routerIndicesBuffer, offset: 0, index: 4)
                                        layerEnc1.setBuffer(routerWeightsBuffer, offset: 0, index: 5)
                                        layerEnc1.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                                        layerEnc1.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                                        layerEnc1.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 8)
                                        layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 9)
                                        layerEnc1.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 10)
                                        layerEnc1.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 11)
                                        layerEnc1.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 12)
                                        layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: Int(numExperts), height: 1, depth: 1))
                                    }
                                } else {
                                    if (routerTensor.shapeDisplay.contains("512") || numExperts >= 512), let r512 = router512Pipeline {
                                        layerEnc1.setComputePipelineState(r512)
                                        layerEnc1.setBuffer(routerRaw, offset: 0, index: 0)
                                        layerEnc1.setBuffer(xNorm2Buffer, offset: 0, index: 1)
                                        layerEnc1.setBuffer(routerIndicesBuffer, offset: 0, index: 2)
                                        layerEnc1.setBuffer(routerWeightsBuffer, offset: 0, index: 3)
                                        layerEnc1.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                                        layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc1.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 7)
                                        layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                                    } else if let rBias = layer.routerExpertBias, let rBiasRaw = buffers[rBias.shardIndex],
                                               let bailingPipe = routerBailingPipeline {
                                        var bOffset = rBias.offsetStart
                                        var nGrp: UInt32 = UInt32(modelConfig?.nGroup ?? 8)
                                        var topkGrp: UInt32 = UInt32(modelConfig?.topkGroup ?? 4)
                                        var scaleFactor: Float = Float(modelConfig?.routedScalingFactor ?? 2.5)

                                        layerEnc1.setComputePipelineState(bailingPipe)
                                        layerEnc1.setBuffer(routerRaw, offset: 0, index: 0)
                                        layerEnc1.setBuffer(rBiasRaw, offset: 0, index: 1)
                                        layerEnc1.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                                        layerEnc1.setBuffer(routerIndicesBuffer, offset: 0, index: 3)
                                        layerEnc1.setBuffer(routerWeightsBuffer, offset: 0, index: 4)
                                        layerEnc1.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                                        layerEnc1.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                                        layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 7)
                                        layerEnc1.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 8)
                                        layerEnc1.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 9)
                                        layerEnc1.setBytes(&nGrp, length: MemoryLayout<UInt32>.stride, index: 10)
                                        layerEnc1.setBytes(&topkGrp, length: MemoryLayout<UInt32>.stride, index: 11)
                                        layerEnc1.setBytes(&scaleFactor, length: MemoryLayout<Float>.stride, index: 12)
                                        layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: Int(numExperts), height: 1, depth: 1))
                                    } else {
                                        layerEnc1.setComputePipelineState(routerPipeline)
                                        layerEnc1.setBuffer(routerRaw, offset: 0, index: 0)
                                        layerEnc1.setBuffer(xNorm2Buffer, offset: 0, index: 1)
                                        layerEnc1.setBuffer(routerIndicesBuffer, offset: 0, index: 2)
                                        layerEnc1.setBuffer(routerWeightsBuffer, offset: 0, index: 3)
                                        layerEnc1.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                                        layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc1.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 7)
                                        layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: Int(numExperts), height: 1, depth: 1))
                                    }
                                }
                                if let sg = layer.sharedGateTensor, let sgRaw = buffers[sg.shardIndex] {
                                    var sgOff = sg.offsetStart
                                    var grp: UInt32 = 64
                                    if let sgScale = layer.sharedGateTensorScale, let sgBias = layer.sharedGateTensorBias,
                                       let sgScaleRaw = buffers[sgScale.shardIndex], let sgBiasRaw = buffers[sgBias.shardIndex] {
                                        var sOff = sgScale.offsetStart
                                        var bOff = sgBias.offsetStart
                                        let is8Bit = (sg.offsetEnd - sg.offsetStart) >= UInt64(hiddenDim)
                                        if is8Bit, let pipe = sharedGateQ8Pipeline {
                                            layerEnc1.setComputePipelineState(pipe)
                                            layerEnc1.setBuffer(sgRaw, offset: 0, index: 0)
                                            layerEnc1.setBuffer(sgScaleRaw, offset: 0, index: 1)
                                            layerEnc1.setBuffer(sgBiasRaw, offset: 0, index: 2)
                                            layerEnc1.setBuffer(xNorm2Buffer, offset: 0, index: 3)
                                            layerEnc1.setBuffer(sharedScoreBuffer, offset: 0, index: 4)
                                            layerEnc1.setBytes(&sgOff, length: MemoryLayout<UInt64>.stride, index: 5)
                                            layerEnc1.setBytes(&sOff, length: MemoryLayout<UInt64>.stride, index: 6)
                                            layerEnc1.setBytes(&bOff, length: MemoryLayout<UInt64>.stride, index: 7)
                                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 8)
                                            layerEnc1.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 9)
                                            layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        } else if let pipe = sharedGateQ4Pipeline {
                                            layerEnc1.setComputePipelineState(pipe)
                                            layerEnc1.setBuffer(sgRaw, offset: 0, index: 0)
                                            layerEnc1.setBuffer(sgScaleRaw, offset: 0, index: 1)
                                            layerEnc1.setBuffer(sgBiasRaw, offset: 0, index: 2)
                                            layerEnc1.setBuffer(xNorm2Buffer, offset: 0, index: 3)
                                            layerEnc1.setBuffer(sharedScoreBuffer, offset: 0, index: 4)
                                            layerEnc1.setBytes(&sgOff, length: MemoryLayout<UInt64>.stride, index: 5)
                                            layerEnc1.setBytes(&sOff, length: MemoryLayout<UInt64>.stride, index: 6)
                                            layerEnc1.setBytes(&bOff, length: MemoryLayout<UInt64>.stride, index: 7)
                                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 8)
                                            layerEnc1.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 9)
                                            layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        }
                                    } else if let sgPipe = sharedGatePipeline {
                                        layerEnc1.setComputePipelineState(sgPipe)
                                        layerEnc1.setBuffer(sgRaw, offset: 0, index: 0)
                                        layerEnc1.setBuffer(xNorm2Buffer, offset: 0, index: 1)
                                        layerEnc1.setBuffer(sharedScoreBuffer, offset: 0, index: 2)
                                        layerEnc1.setBytes(&sgOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                        layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                                        layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    }
                                }
                            }

                            layerEnc1.endEncoding()
                            let tRouterGpuStart = CFAbsoluteTimeGetCurrent()
                            activeCmd.commit()
                            activeCmd.waitUntilCompleted()
                            let layerRouterMs = (CFAbsoluteTimeGetCurrent() - tRouterGpuStart) * 1000.0
                            diagRouterGpuMs += layerRouterMs
                            diagLayerRouterMs = layerRouterMs
                            if let err = activeCmd.error {
                                print("❌ [METAL ERROR] activeCmd failed at layer \(l): \(err)")
                                return false
                            }

                            var activeExperts: [(id: Int, weight: Float)] = []
                            let topKCount = Int(modelConfig?.effectiveNumExpertsPerTok ?? (numExperts >= 512 ? 10 : 8))
                            if layer.routerTensor != nil {
                                let indPtr = routerIndicesBuffer.contents().bindMemory(to: UInt32.self, capacity: topKCount)
                                let wPtr = routerWeightsBuffer.contents().bindMemory(to: Float.self, capacity: topKCount)
                                for i in 0..<topKCount {
                                    let p_k = wPtr[i]
                                    if p_k > 0.00001 {
                                        activeExperts.append((id: Int(indPtr[i]), weight: p_k))
                                    }
                                }
                            } else {
                                activeExperts = (0..<topKCount).map { (id: $0, weight: 1.0 / Float(topKCount)) }
                            }

                            var sharedW: Float = 1.0
                            if layer.sharedGateTensor != nil {
                                sharedW = sharedScoreBuffer.contents().load(as: Float.self)
                            }

                            let activeIds = activeExperts.map { $0.id }
                            // FIX #8: predicted experts for layer l+1, kicked after this
                            // layer's MoE commit so the retire wait overlaps GPU execution.
                            var deferredPredictedExperts: [Int]? = nil
                            if speculativePrefetchEnabled {
                                // 1. Transition correlation tracking + 2-token history rotation
                                if l > 0, let prevIds = previousLayerActiveExperts[l - 1] {
                                    WorkingSetManager.shared.transitionTracker.recordTransition(fromLayer: l - 1, fromExperts: prevIds, toLayer: l, toExperts: activeIds)
                                }
                                // Age the per-layer history: [l] currently holds the previous
                                // token's set; shift it to depth 2 before overwriting.
                                layerExpertHistory[l] = [previousLayerActiveExperts[l] ?? [], layerExpertHistory[l]?.first ?? []].filter { !$0.isEmpty }
                                previousLayerActiveExperts[l] = activeIds

                                // 2. Speculatively prefetch layer l + 1 experts. Temporal signal
                                //    first: the last TWO tokens' expert sets at layer l+1
                                //    (consecutive tokens in fluent text route to largely the same
                                //    experts at the same layer — the measured signal; the
                                //    cross-layer Markov signal measured ~0% and only fills
                                //    remaining budget). Router top-k order is weight-ranked, so
                                //    earlier entries are the higher-confidence candidates.
                                // FIX #8: the kick itself moved AFTER this layer's MoE commit
                                // (see below) so the previous-same-parity-kick retire wait
                                // overlaps GPU execution instead of serializing the CPU.
                                if l + 1 < actualLayers, packedPrefetchAdaptiveEnabled {
                                    // FIX #8: budget capped at 10 candidates. QA #12: 16-candidate
                                    // kicks (50MB/layer, ~2.0GB/token) saturate the SSD and inflate
                                    // GPU-phase wall-times (driver starvation) — candidates beyond
                                    // the window's read capacity buy late hits, not speed.
                                    // 10 candidates ≈ 31.5MB ≈ 5.6ms ≈ the 2-layer overlap window.
                                    let budget = min(10, max(0, (expertStagingSize / packedExpertSize) - maxTopK))
                                    var seen = Set<Int>()
                                    var predicted: [Int] = []
                                    for hist in [previousLayerActiveExperts[l + 1] ?? [], layerExpertHistory[l + 1]?.first ?? []] {
                                        for id in hist where !seen.contains(id) && predicted.count < budget {
                                            seen.insert(id)
                                            predicted.append(id)
                                        }
                                    }
                                    let markov = WorkingSetManager.shared.predictNextLayerExperts(currentLayer: l, currentActiveExperts: activeIds, topN: 10)
                                    for id in markov where !seen.contains(id) && predicted.count < budget {
                                        seen.insert(id)
                                        predicted.append(id)
                                    }
                                    // FIX #8: defer the kick until after moeCmd.commit() — store
                                    // the predicted list for this layer.
                                    deferredPredictedExperts = predicted
                                }

                                // 3. Lookahead prefetch layer l + 2 dense backbone — mmap models only.
                                //    FlashMoE backbone is pinned in RAM at load; the WSM fallback here
                                //    would just CPU-fault safetensors pages and compete with the
                                //    critical-path expert preads for NVMe.
                                // FIX #8: moved after moeCmd.commit() alongside the kick.
                            }
                            WorkingSetManager.shared.touchAndEvict(layer: l, activeExpertIds: activeIds, mode: budgetMode, shardBuffers: buffers, isPrefill: !computeLogits)

                            if let packedDir = packedExpertsDir,
                               let fd = ExpertIOThreadPool.shared.getOrOpenLayerFD(layerIndex: l, packedExpertsDir: packedDir) {
                                let expertSize = Int(loadedLayout?.expert_size ?? 1769472)
                                let isFP8Layout = loadedLayout?.components.contains { $0.dtype.contains("F8") || $0.name.contains("weight_scale") } ?? false

                                // Dynamic component offset discovery from layout
                                let compGateW = loadedLayout?.components.first(where: { $0.name.contains("gate_proj") && $0.name.contains("weight") && !$0.name.contains("scale") && !$0.name.contains("bias") })
                                let compGateS = loadedLayout?.components.first(where: { $0.name.contains("gate_proj") && ($0.name.contains("scale") || $0.name.contains("scales")) })
                                let compGateB = loadedLayout?.components.first(where: { $0.name.contains("gate_proj") && ($0.name.contains("bias") || $0.name.contains("biases")) })

                                let compUpW = loadedLayout?.components.first(where: { $0.name.contains("up_proj") && $0.name.contains("weight") && !$0.name.contains("scale") && !$0.name.contains("bias") })
                                let compUpS = loadedLayout?.components.first(where: { $0.name.contains("up_proj") && ($0.name.contains("scale") || $0.name.contains("scales")) })
                                let compUpB = loadedLayout?.components.first(where: { $0.name.contains("up_proj") && ($0.name.contains("bias") || $0.name.contains("biases")) })

                                let compDownW = loadedLayout?.components.first(where: { $0.name.contains("down_proj") && $0.name.contains("weight") && !$0.name.contains("scale") && !$0.name.contains("bias") })
                                let compDownS = loadedLayout?.components.first(where: { $0.name.contains("down_proj") && ($0.name.contains("scale") || $0.name.contains("scales")) })
                                let compDownB = loadedLayout?.components.first(where: { $0.name.contains("down_proj") && ($0.name.contains("bias") || $0.name.contains("biases")) })

                                let isQuantizedAffine = (compGateB != nil || compGateW?.name.contains("Q4") == true || compGateW?.name.contains("Q8") == true || compGateW?.dtype.contains("Q4") == true || compGateW?.dtype.contains("Q8") == true || (compGateS != nil && !isFP8Layout))

                                // 1. Per-slot expert consumption (FIX #8): wait only for the
                                //    needed slots of the in-flight kick; false-positive reads
                                //    continue in the background. Synchronously pread misses
                                //    into slots after the kick's.
                                let bufIdx = l & 1
                                let stagingBuf = (bufIdx == 0) ? expertStagingBuffer : expertStagingBufferB
                                let stagingCapacity = expertStagingSize / expertSize

                                let tIoStart = CFAbsoluteTimeGetCurrent()

                                var kick: PackedExpertKick? = nil
                                var staleKick: PackedExpertKick? = nil
                                packedPrefetchLock.lock()
                                if let k = packedKickInFlight[bufIdx] {
                                    if k.targetLayer == l {
                                        kick = k
                                    } else {
                                        // Stale kick for this parity (e.g. a skipped dense layer
                                        // or the token boundary): retire it fully before the
                                        // slot space is reused.
                                        staleKick = k
                                        packedKickInFlight[bufIdx] = nil
                                    }
                                }
                                packedPrefetchLock.unlock()
                                if let old = staleKick {
                                    _ = old.allDoneSem.wait(timeout: .now() + 30.0)
                                }

                                var slotOf: [Int: Int] = [:]
                                var prefetchHits = 0
                                var nextSlot: Int
                                var missList: [(id: Int, weight: Float)] = []
                                if let k = kick {
                                    nextSlot = k.slotBase
                                    for exp in activeExperts {
                                        if let idx = k.candidates.firstIndex(of: exp.id), idx < k.slotSems.count {
                                            // Wait for THIS slot's pread only (bounded; a stalled
                                            // read falls back to a synchronous duplicate below).
                                            _ = k.slotSems[idx].wait(timeout: .now() + 15.0)
                                            packedPrefetchLock.lock()
                                            let ok = k.slotOK[idx]
                                            packedPrefetchLock.unlock()
                                            if ok {
                                                slotOf[exp.id] = idx
                                                prefetchHits += 1
                                            } else {
                                                missList.append(exp)
                                            }
                                        } else {
                                            missList.append(exp)
                                        }
                                    }
                                } else {
                                    nextSlot = 0
                                    missList = activeExperts
                                }

                                var missTasks: [ExpertPreadTask] = []
                                for exp in missList {
                                    // Unreachable by construction (misses ≤ maxTopK and the slot
                                    // space starts at 0 or after the kick's candidates); abort
                                    // rather than ever feed the GPU a wrong expert's slot.
                                    guard nextSlot < stagingCapacity else {
                                        print("❌ [Prefetch] slot overflow at layer \(l) — aborting generation")
                                        return false
                                    }
                                    let slot = nextSlot
                                    nextSlot += 1
                                    slotOf[exp.id] = slot
                                    missTasks.append(ExpertPreadTask(fd: fd,
                                                                     dst: stagingBuf.contents().advanced(by: slot * expertSize),
                                                                     offset: off_t(exp.id * expertSize),
                                                                     size: expertSize))
                                }
                                if !missTasks.isEmpty {
                                    ExpertIOThreadPool.shared.dispatchSync(tasks: &missTasks)
                                }
                                let tIoElapsed = (CFAbsoluteTimeGetCurrent() - tIoStart) * 1000.0
                                diagIoMs += tIoElapsed
                                diagPrefetchHits += prefetchHits
                                diagPrefetchTotal += activeExperts.count
                                if hadTemporalHistory {
                                    packedPrefetchObservedHits += prefetchHits
                                    packedPrefetchObservedTotal += activeExperts.count
                                }
                                if hadTemporalHistory, packedPrefetchAdaptiveEnabled, packedPrefetchObservedTotal >= 64,
                                   Double(packedPrefetchObservedHits) / Double(packedPrefetchObservedTotal) < 0.2 {
                                    packedPrefetchAdaptiveEnabled = false
                                    print("⚠️ [Prefetch] hit rate \(String(format: "%.0f", Double(packedPrefetchObservedHits) / Double(packedPrefetchObservedTotal) * 100))% below 20% after temporal warmup — disabling expert prefetch for this generation")
                                }
                                print("[DIAG] Layer \(l)\(layer.attentionType == .fullAttention ? "(gqa)" : ""): PhaseA=\(String(format: "%.2f", diagLayerRouterMs))ms IO=\(String(format: "%.1f", tIoElapsed))ms (prefetch \(prefetchHits)/\(activeExperts.count))")

                                guard let moeCmd = commandQueue.makeCommandBuffer(),
                                      let layerEnc2 = moeCmd.makeComputeCommandEncoder() else {
                                    print("❌ [METAL] token forward: MoE command buffer failed (layer \(l), step \(step), packed)")
                                    return false
                                }

                                layerEnc2.setComputePipelineState(clearPipeline)
                                layerEnc2.setBuffer(hMlpBuffer, offset: 0, index: 0)
                                layerEnc2.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc2.memoryBarrier(scope: .buffers)

                                // Fused fast path: all routed experts in 2 dispatches instead of
                                // 2 per expert (16 dispatches + barriers). Applies to per-row-scaled
                                // FP8 layouts (block-scaled and quantized-affine keep the per-expert
                                // path). slotOf maps expert -> staging slot; the kernels walk the
                                // router-rank list explicitly.
                                let isBlockScaleFP8 = (compGateS?.size ?? 1024) < (intermediateDim * 2) || (compGateS?.name.contains("scale_inv") ?? false)
                                let canFuseMoE = isFP8Layout && !isBlockScaleFP8 && activeExperts.count > 0 && fp8GateUpFusedPipeline != nil && fp8DownFusedPipeline != nil
                                if l == 0 {
                                    print("⚡ [MoE] fused expert path: \(canFuseMoE ? "ACTIVE (2 dispatches/expert-layer)" : "off (isFP8=\(isFP8Layout) blockScale=\(isBlockScaleFP8) pipelines=\(fp8GateUpFusedPipeline != nil)/\(fp8DownFusedPipeline != nil))")")
                                }
                                if canFuseMoE, let fusedGatePipe = fp8GateUpFusedPipeline, let fusedDownPipe = fp8DownFusedPipeline {
                                    let slPtr = fusedSlotListBuffer.contents().bindMemory(to: UInt32.self, capacity: 16)
                                    let swPtr = fusedSlotWeightsBuffer.contents().bindMemory(to: Float.self, capacity: 16)
                                    for (rank, expert) in activeExperts.enumerated() {
                                        slPtr[rank] = UInt32(slotOf[expert.id] ?? 0)
                                        swPtr[rank] = expert.weight
                                    }
                                    var topKVal: UInt32 = UInt32(activeExperts.count)
                                    var esVal: UInt64 = UInt64(expertSize)
                                    var gsVal: UInt64 = compGateS?.offset ?? 1048576
                                    var uwVal: UInt64 = compUpW?.offset ?? 1049600
                                    var usVal: UInt64 = compUpS?.offset ?? 2098176
                                    var dwVal: UInt64 = compDownW?.offset ?? 2099200
                                    var dsVal: UInt64 = compDownS?.offset ?? 3147776
                                    var hDimVal: UInt32 = UInt32(hiddenDim)
                                    var interDimVal: UInt32 = UInt32(intermediateDim)

                                    layerEnc2.setComputePipelineState(fusedGatePipe)
                                    layerEnc2.setBuffer(stagingBuf, offset: 0, index: 0)
                                    layerEnc2.setBuffer(xNorm2Buffer, offset: 0, index: 1)
                                    layerEnc2.setBuffer(interBuffer, offset: 0, index: 2)
                                    layerEnc2.setBuffer(fusedSlotListBuffer, offset: 0, index: 3)
                                    layerEnc2.setBytes(&esVal, length: 8, index: 4)
                                    layerEnc2.setBytes(&gsVal, length: 8, index: 5)
                                    layerEnc2.setBytes(&uwVal, length: 8, index: 6)
                                    layerEnc2.setBytes(&usVal, length: 8, index: 7)
                                    layerEnc2.setBytes(&hDimVal, length: 4, index: 8)
                                    layerEnc2.setBytes(&interDimVal, length: 4, index: 9)
                                    layerEnc2.setBytes(&topKVal, length: 4, index: 10)
                                    layerEnc2.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: activeExperts.count, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    layerEnc2.memoryBarrier(scope: .buffers)

                                    layerEnc2.setComputePipelineState(fusedDownPipe)
                                    layerEnc2.setBuffer(stagingBuf, offset: 0, index: 0)
                                    layerEnc2.setBuffer(interBuffer, offset: 0, index: 1)
                                    layerEnc2.setBuffer(hMlpBuffer, offset: 0, index: 2)
                                    layerEnc2.setBuffer(fusedSlotWeightsBuffer, offset: 0, index: 3)
                                    layerEnc2.setBuffer(fusedSlotListBuffer, offset: 0, index: 4)
                                    layerEnc2.setBytes(&esVal, length: 8, index: 5)
                                    layerEnc2.setBytes(&dwVal, length: 8, index: 6)
                                    layerEnc2.setBytes(&dsVal, length: 8, index: 7)
                                    layerEnc2.setBytes(&interDimVal, length: 4, index: 8)
                                    layerEnc2.setBytes(&hDimVal, length: 4, index: 9)
                                    layerEnc2.setBytes(&topKVal, length: 4, index: 10)
                                    layerEnc2.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: activeExperts.count, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    layerEnc2.memoryBarrier(scope: .buffers)
                                }

                                if !canFuseMoE {
                                for (slot, expert) in activeExperts.enumerated() {
                                    let pk = expert.weight
                                    if pk <= 0.00001 { continue }
                                    let slotOffset = UInt64((slotOf[expert.id] ?? slot) * expertSize)
                                    let gWOff = slotOffset + (compGateW?.offset ?? 0)
                                    let gSOff = slotOffset + (compGateS?.offset ?? 524288)
                                    let gBOff = slotOffset + (compGateB?.offset ?? 557056)
                                    let uWOff = slotOffset + (compUpW?.offset ?? 589824)
                                    let uSOff = slotOffset + (compUpS?.offset ?? 1114112)
                                    let uBOff = slotOffset + (compUpB?.offset ?? 1146880)
                                    let dWOff = slotOffset + (compDownW?.offset ?? 1179648)
                                    let dSOff = slotOffset + (compDownS?.offset ?? 1703936)
                                    let dBOff = slotOffset + (compDownB?.offset ?? 1736704)

                                    if isFP8Layout {
                                        let isBlockScale = (compGateS?.size ?? 1024) < (intermediateDim * 2) || (compGateS?.name.contains("scale_inv") ?? false)
                                        let gateSimd = isBlockScale ? (fp8BlockGateUpSimdPipeline ?? fp8GateUpSimdPipeline ?? fp8GateUpPipeline) : (fp8GateUpSimdPipeline ?? fp8GateUpPipeline)
                                        let downSimd = isBlockScale ? (fp8BlockDownSimdPipeline ?? fp8DownSimdPipeline ?? fp8DownPipeline) : (fp8DownSimdPipeline ?? fp8DownPipeline)

                                        if let gateSimd = gateSimd,
                                           let downSimd = downSimd {
                                            var gWOffU = gWOff
                                            var gSOffU = gSOff
                                            var uWOffU = uWOff
                                            var uSOffU = uSOff
                                            var dWOffU = dWOff
                                            var dSOffU = dSOff
                                            var hDimVal: UInt32 = UInt32(hiddenDim)
                                            var interDimVal: UInt32 = UInt32(intermediateDim)
                                            var pkVal = pk

                                            layerEnc2.setComputePipelineState(gateSimd)
                                            layerEnc2.setBuffer(stagingBuf, offset: 0, index: 0)
                                            layerEnc2.setBuffer(stagingBuf, offset: 0, index: 1)
                                            layerEnc2.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                                            layerEnc2.setBuffer(interBuffer, offset: 0, index: 3)
                                            layerEnc2.setBuffer(stagingBuf, offset: 0, index: 4)
                                            layerEnc2.setBuffer(stagingBuf, offset: 0, index: 5)
                                            layerEnc2.setBytes(&gWOffU, length: 8, index: 6)
                                            layerEnc2.setBytes(&gSOffU, length: 8, index: 7)
                                            layerEnc2.setBytes(&uWOffU, length: 8, index: 8)
                                            layerEnc2.setBytes(&uSOffU, length: 8, index: 9)
                                            layerEnc2.setBytes(&hDimVal, length: 4, index: 10)
                                            layerEnc2.setBytes(&interDimVal, length: 4, index: 11)
                                            layerEnc2.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                            layerEnc2.memoryBarrier(scope: .buffers)

                                            layerEnc2.setComputePipelineState(downSimd)
                                            layerEnc2.setBuffer(stagingBuf, offset: 0, index: 0)
                                            layerEnc2.setBuffer(interBuffer, offset: 0, index: 1)
                                            layerEnc2.setBuffer(hMlpBuffer, offset: 0, index: 2)
                                            layerEnc2.setBuffer(stagingBuf, offset: 0, index: 3)
                                            layerEnc2.setBytes(&dWOffU, length: 8, index: 4)
                                            layerEnc2.setBytes(&dSOffU, length: 8, index: 5)
                                            layerEnc2.setBytes(&interDimVal, length: 4, index: 6)
                                            layerEnc2.setBytes(&hDimVal, length: 4, index: 7)
                                            layerEnc2.setBytes(&pkVal, length: 4, index: 8)
                                            layerEnc2.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                            layerEnc2.memoryBarrier(scope: .buffers)
                                        }
                                    } else if isQuantizedAffine,
                                               let q4GatePipe = q4GateUpPipeline,
                                               let q4DownPipe = q4DownPipeline {
                                        var gWOffU = gWOff
                                        var gSOffU = gSOff
                                        var gBOffU = gBOff
                                        var uWOffU = uWOff
                                        var uSOffU = uSOff
                                        var uBOffU = uBOff
                                        var dWOffU = dWOff
                                        var dSOffU = dSOff
                                        var dBOffU = dBOff
                                        var hDimVal: UInt32 = UInt32(hiddenDim)
                                        var interDimVal: UInt32 = UInt32(intermediateDim)
                                        var grp: UInt32 = 64
                                        var pkVal = pk

                                        layerEnc2.setComputePipelineState(q4GatePipe)
                                        layerEnc2.setBuffer(stagingBuf, offset: 0, index: 0)
                                        layerEnc2.setBuffer(stagingBuf, offset: 0, index: 1)
                                        layerEnc2.setBuffer(stagingBuf, offset: 0, index: 2)
                                        layerEnc2.setBuffer(stagingBuf, offset: 0, index: 3)
                                        layerEnc2.setBuffer(stagingBuf, offset: 0, index: 4)
                                        layerEnc2.setBuffer(stagingBuf, offset: 0, index: 5)
                                        layerEnc2.setBuffer(xNorm2Buffer, offset: 0, index: 6)
                                        layerEnc2.setBuffer(interBuffer, offset: 0, index: 7)
                                        layerEnc2.setBytes(&gWOffU, length: 8, index: 8)
                                        layerEnc2.setBytes(&gSOffU, length: 8, index: 9)
                                        layerEnc2.setBytes(&gBOffU, length: 8, index: 10)
                                        layerEnc2.setBytes(&uWOffU, length: 8, index: 11)
                                        layerEnc2.setBytes(&uSOffU, length: 8, index: 12)
                                        layerEnc2.setBytes(&uBOffU, length: 8, index: 13)
                                        layerEnc2.setBytes(&hDimVal, length: 4, index: 14)
                                        layerEnc2.setBytes(&interDimVal, length: 4, index: 15)
                                        layerEnc2.setBytes(&grp, length: 4, index: 16)
                                        layerEnc2.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        layerEnc2.memoryBarrier(scope: .buffers)

                                        layerEnc2.setComputePipelineState(q4DownPipe)
                                        layerEnc2.setBuffer(stagingBuf, offset: 0, index: 0)
                                        layerEnc2.setBuffer(stagingBuf, offset: 0, index: 1)
                                        layerEnc2.setBuffer(stagingBuf, offset: 0, index: 2)
                                        layerEnc2.setBuffer(stagingBuf, offset: 0, index: 3)
                                        layerEnc2.setBuffer(interBuffer, offset: 0, index: 4)
                                        layerEnc2.setBuffer(hMlpBuffer, offset: 0, index: 5)
                                        layerEnc2.setBytes(&dWOffU, length: 8, index: 6)
                                        layerEnc2.setBytes(&dSOffU, length: 8, index: 7)
                                        layerEnc2.setBytes(&dBOffU, length: 8, index: 8)
                                        layerEnc2.setBytes(&interDimVal, length: 4, index: 9)
                                        layerEnc2.setBytes(&hDimVal, length: 4, index: 10)
                                        layerEnc2.setBytes(&grp, length: 4, index: 11)
                                        layerEnc2.setBytes(&pkVal, length: 4, index: 12)
                                        layerEnc2.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        layerEnc2.memoryBarrier(scope: .buffers)
                                    } else if let gateSimd = bf16GateUpSimdPipeline,
                                               let downSimd = bf16DownSimdPipeline {
                                        var gWOffU = gWOff
                                        var uWOffU = uWOff
                                        var dWOffU = dWOff
                                        var hDimVal: UInt32 = UInt32(hiddenDim)
                                        var interDimVal: UInt32 = UInt32(intermediateDim)
                                        var pkVal = pk

                                        layerEnc2.setComputePipelineState(gateSimd)
                                        layerEnc2.setBuffer(stagingBuf, offset: 0, index: 0)
                                        layerEnc2.setBuffer(stagingBuf, offset: 0, index: 1)
                                        layerEnc2.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                                        layerEnc2.setBuffer(interBuffer, offset: 0, index: 3)
                                        layerEnc2.setBytes(&gWOffU, length: 8, index: 4)
                                        layerEnc2.setBytes(&uWOffU, length: 8, index: 5)
                                        layerEnc2.setBytes(&hDimVal, length: 4, index: 6)
                                        layerEnc2.setBytes(&interDimVal, length: 4, index: 7)
                                        layerEnc2.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        layerEnc2.memoryBarrier(scope: .buffers)

                                        layerEnc2.setComputePipelineState(downSimd)
                                        layerEnc2.setBuffer(stagingBuf, offset: 0, index: 0)
                                        layerEnc2.setBuffer(interBuffer, offset: 0, index: 1)
                                        layerEnc2.setBuffer(hMlpBuffer, offset: 0, index: 2)
                                        layerEnc2.setBytes(&dWOffU, length: 8, index: 3)
                                        layerEnc2.setBytes(&interDimVal, length: 4, index: 4)
                                        layerEnc2.setBytes(&hDimVal, length: 4, index: 5)
                                        layerEnc2.setBytes(&pkVal, length: 4, index: 6)
                                        layerEnc2.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        layerEnc2.memoryBarrier(scope: .buffers)
                                    } else if let gateUnq = bf16GateUpPipeline,
                                               let downUnq = bf16DownPipeline {
                                        var gWOffU = gWOff
                                        var uWOffU = uWOff
                                        var dWOffU = dWOff
                                        var hDimVal: UInt32 = UInt32(hiddenDim)
                                        var interDimVal: UInt32 = UInt32(intermediateDim)
                                        var pkVal = pk

                                        layerEnc2.setComputePipelineState(gateUnq)
                                        layerEnc2.setBuffer(stagingBuf, offset: 0, index: 0)
                                        layerEnc2.setBuffer(stagingBuf, offset: 0, index: 1)
                                        layerEnc2.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                                        layerEnc2.setBuffer(interBuffer, offset: 0, index: 3)
                                        layerEnc2.setBytes(&gWOffU, length: 8, index: 4)
                                        layerEnc2.setBytes(&uWOffU, length: 8, index: 5)
                                        layerEnc2.setBytes(&hDimVal, length: 4, index: 6)
                                        layerEnc2.setBytes(&interDimVal, length: 4, index: 7)
                                        layerEnc2.dispatchThreads(MTLSize(width: Int(intermediateDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(intermediateDim), gateUnq.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                        layerEnc2.memoryBarrier(scope: .buffers)

                                        layerEnc2.setComputePipelineState(downUnq)
                                        layerEnc2.setBuffer(stagingBuf, offset: 0, index: 0)
                                        layerEnc2.setBuffer(interBuffer, offset: 0, index: 1)
                                        layerEnc2.setBuffer(hMlpBuffer, offset: 0, index: 2)
                                        layerEnc2.setBytes(&dWOffU, length: 8, index: 3)
                                        layerEnc2.setBytes(&interDimVal, length: 4, index: 4)
                                        layerEnc2.setBytes(&hDimVal, length: 4, index: 5)
                                        layerEnc2.setBytes(&pkVal, length: 4, index: 6)
                                        layerEnc2.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), downUnq.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                        layerEnc2.memoryBarrier(scope: .buffers)
                                    }
                                }
                                }

                                if let gateW = layer.sharedGateWeight,
                                   let upW = layer.sharedUpWeight,
                                   let downW = layer.sharedDownWeight {
                                    let gateS = layer.sharedGateScale
                                    let gateB = layer.sharedGateBias
                                    let upS = layer.sharedUpScale
                                    let upB = layer.sharedUpBias
                                    let downS = layer.sharedDownScale
                                    let downB = layer.sharedDownBias

                                    dispatchExpertMlp(
                                        enc: layerEnc2,
                                        gateW: gateW,
                                        gateS: gateS,
                                        gateB: gateB,
                                        upW: upW,
                                        upS: upS,
                                        upB: upB,
                                        downW: downW,
                                        downS: downS,
                                        downB: downB,
                                        inBuf: xNorm2Buffer,
                                        interBuf: interBuffer,
                                        accumBuf: hMlpBuffer,
                                        inDim: hiddenDim,
                                        interDim: intermediateDim,
                                        routingWeight: sharedW
                                    )
                                    layerEnc2.memoryBarrier(scope: .buffers)
                                }

                                // Step 6: Residual Connection 2 (nextH = hMid + hMlp or 4-stream inject)
                                if (layer.mlpHcInjectWeight != nil || layer.mlpHcDownWeight != nil),
                                   let injPipe = hcInjectPipeline {
                                    layerEnc2.setComputePipelineState(injPipe)
                                    layerEnc2.setBuffer(hcStreamsBuffer, offset: 0, index: 0)
                                    layerEnc2.setBuffer(hMlpBuffer, offset: 0, index: 1)
                                    layerEnc2.setBuffer(hcInjectScaleBuffer, offset: 0, index: 2)
                                    layerEnc2.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                                    layerEnc2.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), injPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    layerEnc2.memoryBarrier(scope: .buffers)
                                } else {
                                    layerEnc2.setComputePipelineState(addPipeline)
                                    layerEnc2.setBuffer(hMidBuffer, offset: 0, index: 0)
                                    layerEnc2.setBuffer(hMlpBuffer, offset: 0, index: 1)
                                    layerEnc2.setBuffer(nextH, offset: 0, index: 2)
                                    layerEnc2.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                                    layerEnc2.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    layerEnc2.memoryBarrier(scope: .buffers)
                                }

                                layerEnc2.endEncoding()
                                let tMoeGpuStart = CFAbsoluteTimeGetCurrent()
                                moeCmd.commit()

                                // FIX #8: kick layer l+1's predicted expert reads now that the
                                // GPU is busy with this layer's MoE. The retire-wait for the
                                // previous same-parity kick (inside kickPackedPrefetch)
                                // overlaps this GPU execution instead of serializing the CPU.
                                if let predicted = deferredPredictedExperts, !predicted.isEmpty {
                                    if let packedDir = packedExpertsDir,
                                       let nextFd = ExpertIOThreadPool.shared.getOrOpenLayerFD(layerIndex: l + 1, packedExpertsDir: packedDir) {
                                        // Packed: stream predicted experts into the opposite staging
                                        // buffer, overlapping this layer's MoE GPU work.
                                        kickPackedPrefetch(predicted, targetLayer: l + 1, fd: nextFd)
                                    } else if packedExpertsDir == nil {
                                        // Unpacked: WSM mmap-path prefetch
                                        WorkingSetManager.shared.prefetchLayerExperts(layer: l + 1, expertIds: predicted, shardBuffers: buffers)
                                    }
                                }
                                // 3. Lookahead prefetch layer l + 2 dense backbone — mmap models only.
                                //    FlashMoE backbone is pinned in RAM at load; the WSM fallback here
                                //    would just CPU-fault safetensors pages and compete with the
                                //    critical-path expert preads for NVMe.
                                if packedExpertsDir == nil && l + 2 < actualLayers {
                                    let nextNextL = l + 2
                                    if nextNextL < cachedLayers.count {
                                        WorkingSetManager.shared.prefetchLayerBackbone(layer: cachedLayers[nextNextL], shardBuffers: buffers)
                                    }
                                }

                                moeCmd.waitUntilCompleted()
                                if let moeErr = moeCmd.error {
                                    print("❌ [METAL ERROR] prefill MoE cmd aborted: \(moeErr)")
                                    return false
                                }
                                diagMoeGpuMs += (CFAbsoluteTimeGetCurrent() - tMoeGpuStart) * 1000.0
                                if let err = moeCmd.error {
                                    print("❌ [METAL ERROR] moeCmd (packed) failed at layer \(l): \(err)")
                                    return false
                                }
                            } else {
                                guard let moeCmd = commandQueue.makeCommandBuffer(),
                                      let layerEnc2 = moeCmd.makeComputeCommandEncoder() else {
                                    print("❌ [METAL] token forward: MoE command buffer failed (layer \(l), step \(step), unpacked)")
                                    return false
                                }

                                layerEnc2.setComputePipelineState(clearPipeline)
                                layerEnc2.setBuffer(hMlpBuffer, offset: 0, index: 0)
                                layerEnc2.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc2.memoryBarrier(scope: .buffers)

                                for expert in activeExperts {
                                    let expId = expert.id
                                    let p_k = expert.weight
                                    if p_k <= 0.00001 { continue }

                                    if let gateW = layer.expertGateWeights[expId],
                                       let upW = layer.expertUpWeights[expId],
                                       let downW = layer.expertDownWeights[expId] {
                                        let gateS = layer.expertGateScales[expId]
                                        let gateB = layer.expertGateBiases[expId]
                                        let upS = layer.expertUpScales[expId]
                                        let upB = layer.expertUpBiases[expId]
                                        let downS = layer.expertDownScales[expId]
                                        let downB = layer.expertDownBiases[expId]

                                        dispatchExpertMlp(
                                            enc: layerEnc2,
                                            gateW: gateW,
                                            gateS: gateS,
                                            gateB: gateB,
                                            upW: upW,
                                            upS: upS,
                                            upB: upB,
                                            downW: downW,
                                            downS: downS,
                                            downB: downB,
                                            inBuf: xNorm2Buffer,
                                            interBuf: interBuffer,
                                            accumBuf: hMlpBuffer,
                                            inDim: hiddenDim,
                                            interDim: intermediateDim,
                                            routingWeight: p_k
                                        )
                                        layerEnc2.memoryBarrier(scope: .buffers)
                                    }
                                }

                                if let gateW = layer.sharedGateWeight,
                                   let upW = layer.sharedUpWeight,
                                   let downW = layer.sharedDownWeight {
                                    let gateS = layer.sharedGateScale
                                    let gateB = layer.sharedGateBias
                                    let upS = layer.sharedUpScale
                                    let upB = layer.sharedUpBias
                                    let downS = layer.sharedDownScale
                                    let downB = layer.sharedDownBias

                                    dispatchExpertMlp(
                                        enc: layerEnc2,
                                        gateW: gateW,
                                        gateS: gateS,
                                        gateB: gateB,
                                        upW: upW,
                                        upS: upS,
                                        upB: upB,
                                        downW: downW,
                                        downS: downS,
                                        downB: downB,
                                        inBuf: xNorm2Buffer,
                                        interBuf: interBuffer,
                                        accumBuf: hMlpBuffer,
                                        inDim: hiddenDim,
                                        interDim: intermediateDim,
                                        routingWeight: sharedW
                                    )
                                    layerEnc2.memoryBarrier(scope: .buffers)
                                }

                                // Step 6: Residual Connection 2 (nextH = hMid + hMlp or 4-stream inject)
                                if (layer.mlpHcInjectWeight != nil || layer.mlpHcDownWeight != nil),
                                   let injPipe = hcInjectPipeline {
                                    layerEnc2.setComputePipelineState(injPipe)
                                    layerEnc2.setBuffer(hcStreamsBuffer, offset: 0, index: 0)
                                    layerEnc2.setBuffer(hMlpBuffer, offset: 0, index: 1)
                                    layerEnc2.setBuffer(hcInjectScaleBuffer, offset: 0, index: 2)
                                    layerEnc2.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                                    layerEnc2.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), injPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    layerEnc2.memoryBarrier(scope: .buffers)
                                } else {
                                    layerEnc2.setComputePipelineState(addPipeline)
                                    layerEnc2.setBuffer(hMidBuffer, offset: 0, index: 0)
                                    layerEnc2.setBuffer(hMlpBuffer, offset: 0, index: 1)
                                    layerEnc2.setBuffer(nextH, offset: 0, index: 2)
                                    layerEnc2.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                                    layerEnc2.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    layerEnc2.memoryBarrier(scope: .buffers)
                                }

                                layerEnc2.endEncoding()
                                let tMoeGpuStart2 = CFAbsoluteTimeGetCurrent()
                                moeCmd.commit()
                                moeCmd.waitUntilCompleted()
                                diagMoeGpuMs += (CFAbsoluteTimeGetCurrent() - tMoeGpuStart2) * 1000.0
                                if let err = moeCmd.error {
                                    print("❌ [METAL ERROR] moeCmd (unpacked) failed at layer \(l): \(err)")
                                    return false
                                }
                            }

                            guard let nextCmd = commandQueue.makeCommandBuffer() else {
                                print("❌ [METAL] token forward: command buffer allocation failed at MoE layer \(l) (step \(step))")
                                return false
                            }
                            activeCmd = nextCmd
                        }
                        }

                        let tempBuf = currentH
                        currentH = nextH
                        nextH = tempBuf
                    }

                    // Apply intermediate loop final norm if multiple loops exist
                    if loopIdx + 1 < totalLoops, let normTensor = normTensor, let normShardBuffer = normShardBuffer {
                        guard let loopNormEnc = activeCmd.makeComputeCommandEncoder() else {
                            print("❌ [METAL] token forward: loop-norm encoder allocation failed (loop \(loopIdx), step \(step))")
                            return false
                        }

                        var nOff = normOffset
                        var epsVal = eps
                        let isFinalF16 = (normTensor.dtype.contains("F16") || normTensor.dtype.contains("HALF") || normTensor.dtype.contains("FLOAT16")) && !normTensor.dtype.contains("BF16") && !normTensor.dtype.contains("BFLOAT")
                        let finalNormPipe: MTLComputePipelineState
                        if isRMSNormOffset {
                            finalNormPipe = (isFinalF16 && rmsnormOffsetF16Pipeline != nil) ? rmsnormOffsetF16Pipeline! : (rmsnormOffsetPipeline ?? rmsnormPipeline)
                        } else {
                            finalNormPipe = (isFinalF16 && rmsnormF16Pipeline != nil) ? rmsnormF16Pipeline! : rmsnormPipeline
                        }
                        loopNormEnc.setComputePipelineState(finalNormPipe)
                        loopNormEnc.setBuffer(currentH, offset: 0, index: 0)
                        loopNormEnc.setBuffer(normShardBuffer, offset: 0, index: 1)
                        loopNormEnc.setBuffer(nextH, offset: 0, index: 2)
                        loopNormEnc.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                        loopNormEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                        loopNormEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                        loopNormEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                        loopNormEnc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))

                        loopNormEnc.endEncoding()

                        let tempBuf = currentH
                        currentH = nextH
                        nextH = tempBuf
                    }
                }

                guard computeLogits else {
                    activeCmd.commit()
                    activeCmd.waitUntilCompleted()
                    if let err = activeCmd.error {
                        print("❌ [METAL ERROR] activeCmd (prefill) failed: \(err)")
                        return false
                    }
                    return true
                }

                // 3. Final RMSNorm & LM Head
                guard let finalEnc = activeCmd.makeComputeCommandEncoder() else {
                    print("❌ [METAL] token forward: final norm encoder allocation failed (step \(step))")
                    return false
                }

                if let finalHcDown = finalHcDownWeight, let finalHcDownRaw = buffers[finalHcDown.shardIndex],
                   let finalHcUp = finalHcUpWeight, let finalHcUpRaw = buffers[finalHcUp.shardIndex],
                   let finalHcNorm = finalHcNormWeight, let finalHcNormRaw = buffers[finalHcNorm.shardIndex],
                   let normPipe = hcNormPipeline, let downPipe = hcDownProjPipeline, let upPipe = hcUpBlendPipeline {
                    var nOff = finalHcNorm.offsetStart
                    var dOff = finalHcDown.offsetStart
                    var uOff = finalHcUp.offsetStart
                    var totDim: UInt32 = 4 * hiddenDim
                    var rank: UInt32 = UInt32(modelConfig?.effectiveHcLowrank ?? 320)
                    var epsVal = eps

                    // Final 1a. Group RMSNorm on 4 Streams
                    finalEnc.setComputePipelineState(normPipe)
                    finalEnc.setBuffer(hcStreamsBuffer, offset: 0, index: 0)
                    finalEnc.setBuffer(finalHcNormRaw, offset: 0, index: 1)
                    finalEnc.setBuffer(hcNormedBuffer, offset: 0, index: 2)
                    finalEnc.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                    finalEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                    finalEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                    finalEnc.dispatchThreadgroups(MTLSize(width: 4, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    finalEnc.memoryBarrier(scope: .buffers)

                    // Final 1b. Down-Projection
                    finalEnc.setComputePipelineState(downPipe)
                    finalEnc.setBuffer(hcNormedBuffer, offset: 0, index: 0)
                    finalEnc.setBuffer(finalHcDownRaw, offset: 0, index: 1)
                    finalEnc.setBuffer(hcBottleneckBuffer, offset: 0, index: 2)
                    finalEnc.setBytes(&dOff, length: MemoryLayout<UInt64>.stride, index: 3)
                    finalEnc.setBytes(&totDim, length: MemoryLayout<UInt32>.stride, index: 4)
                    finalEnc.setBytes(&rank, length: MemoryLayout<UInt32>.stride, index: 5)
                    finalEnc.dispatchThreadgroups(MTLSize(width: Int(rank), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    finalEnc.memoryBarrier(scope: .buffers)

                    // Final 1c. Up-Projection and Stream Blend
                    finalEnc.setComputePipelineState(upPipe)
                    finalEnc.setBuffer(hcNormedBuffer, offset: 0, index: 0)
                    finalEnc.setBuffer(hcBottleneckBuffer, offset: 0, index: 1)
                    finalEnc.setBuffer(finalHcUpRaw, offset: 0, index: 2)
                    finalEnc.setBuffer(xFinalBuffer, offset: 0, index: 3)
                    finalEnc.setBytes(&uOff, length: MemoryLayout<UInt64>.stride, index: 4)
                    finalEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 5)
                    finalEnc.setBytes(&rank, length: MemoryLayout<UInt32>.stride, index: 6)
                    finalEnc.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    finalEnc.memoryBarrier(scope: .buffers)
                } else if let normTensor = normTensor, let normShardBuffer = normShardBuffer {
                    if (cachedLayers.first?.attnHcDownWeight != nil), let extractPipe = extractStream0Pipeline {
                        finalEnc.setComputePipelineState(extractPipe)
                        finalEnc.setBuffer(hcStreamsBuffer, offset: 0, index: 0)
                        finalEnc.setBuffer(currentH, offset: 0, index: 1)
                        finalEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 2)
                        finalEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), extractPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                        finalEnc.memoryBarrier(scope: .buffers)
                    }

                    var nOff = normOffset
                    var epsVal = eps
                    let isFinalF16 = (normTensor.dtype.contains("F16") || normTensor.dtype.contains("HALF") || normTensor.dtype.contains("FLOAT16")) && !normTensor.dtype.contains("BF16") && !normTensor.dtype.contains("BFLOAT")
                    let finalNormPipe: MTLComputePipelineState
                    if isRMSNormOffset {
                        finalNormPipe = (isFinalF16 && rmsnormOffsetF16Pipeline != nil) ? rmsnormOffsetF16Pipeline! : (rmsnormOffsetPipeline ?? rmsnormPipeline)
                    } else {
                        finalNormPipe = (isFinalF16 && rmsnormF16Pipeline != nil) ? rmsnormF16Pipeline! : rmsnormPipeline
                    }
                    finalEnc.setComputePipelineState(finalNormPipe)
                    finalEnc.setBuffer(currentH, offset: 0, index: 0)
                    finalEnc.setBuffer(normShardBuffer, offset: 0, index: 1)
                    finalEnc.setBuffer(xFinalBuffer, offset: 0, index: 2)
                    finalEnc.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                    finalEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                    finalEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                    finalEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                    finalEnc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                    finalEnc.memoryBarrier(scope: .buffers)
                }

                dispatchLinear(enc: finalEnc, weight: lmHeadTensor, scale: lmHeadScale, bias: lmHeadBias, inBuf: xFinalBuffer, outBuf: logitsBuffer, inDim: hiddenDim, outDim: vocabSize)

                finalEnc.endEncoding()
                activeCmd.commit()
                if wait {
                    activeCmd.waitUntilCompleted()
                    if let err = activeCmd.error {
                        print("❌ [METAL ERROR] activeCmd (decode) failed: \(err)")
                        return false
                    }
                }

                if computeLogits {
                    let tTokenElapsed = (CFAbsoluteTimeGetCurrent() - tTokenStart) * 1000.0
                    diagOtherMs = tTokenElapsed - diagRouterGpuMs - diagIoMs - diagMoeGpuMs
                    let hitStr = diagPrefetchTotal > 0 ? " | Prefetch \(diagPrefetchHits)/\(diagPrefetchTotal) (\(String(format: "%.0f", Double(diagPrefetchHits) / Double(max(diagPrefetchTotal, 1)) * 100.0))%)" : ""
                    print("[DIAG] Token \(step) total=\(String(format: "%.1f", tTokenElapsed))ms | RouterGPU=\(String(format: "%.1f", diagRouterGpuMs))ms | IO=\(String(format: "%.1f", diagIoMs))ms | MoEGPU=\(String(format: "%.1f", diagMoeGpuMs))ms | Other=\(String(format: "%.1f", diagOtherMs))ms\(hitStr)")
                }

                return true
            }

            // Helper for Layer-Wise Streaming Prompt Prefill (O(Layers) SSD reads instead of O(Tokens * Layers))
            func runLayerWisePrefill(promptTokens: [UInt32], startPos: UInt32 = 0, chunkIndex: Int = 0, chunkCount: Int = 1, prefillProgress: ((String, Double, Double) -> Void)? = nil) -> Bool {
                let P = promptTokens.count
                guard P > 0 else { return true }
                if Int(startPos) + P > KVCacheManager.shared.allocatedSeqLen {
                    print("❌ [KVCACHE] prefill range \(startPos)..<\(Int(startPos) + P) exceeds the provisioned row length \(KVCacheManager.shared.allocatedSeqLen) — refusing chunk (KV precision \(KVCacheManager.shared.activePrecision))")
                    return false
                }

                let topKCount = max(Int(modelConfig?.effectiveNumExpertsPerTok ?? (numExperts >= 512 ? 10 : (numExperts > 0 ? 8 : 1))), 1)
                let hasHc = (cachedLayers.first?.attnHcDownWeight != nil)
                let totalHBytes = max(P * Int(hiddenDim) * MemoryLayout<Float>.stride, 64)
                let totalHcBytes = hasHc ? max(P * 4 * Int(hiddenDim) * MemoryLayout<Float>.stride, 64) : 64

                let maxInterDim = max(cachedLayers.map { $0.intermediateDim }.max() ?? 2560, 2560)
                let linKeyHeads = UInt32(modelConfig?.effectiveLinearNumKeyHeads ?? 16)
                let linValHeads = UInt32(modelConfig?.effectiveLinearNumValueHeads ?? (cachedLayers.first?.inProjA != nil && (cachedLayers.first!.inProjA!.offsetEnd - cachedLayers.first!.inProjA!.offsetStart) > 32 * 2560 ? 48 : 32))
                let gdnQkvDim = (linKeyHeads + linKeyHeads + linValHeads) * 128
                let isGatedQ: Bool = {
                    if let configGate = modelConfig?.effectiveAttnOutputGate {
                        return configGate
                    }
                    if let qTensor = cachedLayers.first(where: { $0.attentionType == .fullAttention })?.qProjTensor {
                        let gatedDim = numHeads * headDim * 2
                        return qTensor.shapeDisplay.contains("\(gatedDim)")
                    }
                    return false
                }()
                let isStandardGqa = !isGatedQ
                let gqaQDim: UInt32 = isStandardGqa ? (numHeads * headDim) : (numHeads * headDim * 2)
                // Gemma 4's global layers widen the per-token head dim (512 vs 256) and reuse
                // K as V, so size the batched q/context buffers for the widest case.
                let isGemma4Prefill = (modelConfig?.isGemma4Model == true)
                let gemmaQDimAll = numHeads * max(headDim, globalHeadDim)
                let maxQkvDim = max(gdnQkvDim, gqaQDim, hiddenDim, 2560, isGemma4Prefill ? gemmaQDimAll : 0)
                let maxZDim = max(linValHeads * 128, attnCtxDim, 128, isGemma4Prefill ? gemmaQDimAll : 0)
                // Max unique experts a single layer can activate for this prompt; the
                // double-buffered staging holds exactly one layer's set per buffer.
                let prefillStagingSlotCap = min(max(P * topKCount, 64), numExperts > 0 ? Int(numExperts) : 512)

                // Per-call transients come from the prefill pool: the chunked Gemma
                // prefill re-enters this function per chunk, so per-call allocation
                // would re-allocate the multi-GB expert staging on every chunk while
                // the weight residency grows toward the budget. A nil allocation is
                // reported by name — it is the silent killer of chunked prefill runs.
                func pooled(_ name: String, _ byteCount: Int) -> MTLBuffer? {
                    let bytes = max(byteCount, 64)
                    guard let buf = PrefillTransientPool.shared.buffer(device: device, name: name, byteCount: bytes) else {
                        print("❌ [PREFILL] transient buffer '\(name)' (\(bytes / 1_048_576) MB) failed to allocate — pool already held \(PrefillTransientPool.shared.heldBytes / 1_048_576) MB; memory pressure (budget-capped weight residency + staging)")
                        return nil
                    }
                    return buf
                }
                guard let hBufA = pooled("prefill.hBufA", totalHBytes),
                      let hBufB = pooled("prefill.hBufB", totalHBytes),
                      let hMidBuffer_all = pooled("prefill.hMid", totalHBytes),
                      let xNorm1Buffer_all = pooled("prefill.xNorm1", totalHBytes),
                      let xNorm2Buffer_all = pooled("prefill.xNorm2", totalHBytes),
                      let attnOutBuffer_all = pooled("prefill.attnOut", totalHBytes),
                      let hMlpBuffer_all = pooled("prefill.hMlp", totalHBytes),
                      let qGateBuffer_all = pooled("prefill.qGate", max(P * Int(maxQkvDim), 1) * MemoryLayout<Float>.stride),
                      let zGateBuffer_all = pooled("prefill.zGate", max(P * Int(maxZDim), 1) * MemoryLayout<Float>.stride),
                      let attnCtxBuffer_all = pooled("prefill.attnCtx", max(P * Int(maxZDim), 1) * MemoryLayout<Float>.stride),
                      let kVectorBuffer_all = pooled("prefill.kVec", max(P * max(Int(kvStride), 128), 1) * MemoryLayout<Float>.stride),
                      let vVectorBuffer_all = pooled("prefill.vVec", max(P * max(Int(kvStride), 128), 1) * MemoryLayout<Float>.stride),
                      let aVectorBuffer_all = pooled("prefill.aVec", max(P * 128, 1) * MemoryLayout<Float>.stride),
                      let bVectorBuffer_all = pooled("prefill.bVec", max(P * (isGemma4Prefill ? Int(hiddenDim) : 128), 1) * MemoryLayout<Float>.stride),
                      let interBuffer_all = pooled("prefill.inter", max(P * Int(maxInterDim), 1) * MemoryLayout<Float>.stride),
                      let routerIndicesBuffer_all = pooled("prefill.routerIdx", max(P * topKCount, 16) * MemoryLayout<UInt32>.stride),
                      let routerWeightsBuffer_all = pooled("prefill.routerW", max(P * topKCount, 16) * MemoryLayout<Float>.stride),
                      let sharedScoreBuffer_all = pooled("prefill.sharedScore", max(P, 1) * MemoryLayout<Float>.stride),
                      let hcStreamsBuffer_all = pooled("prefill.hcStreams", max(totalHcBytes, 64)),
                      let hcNormedBuffer_all = pooled("prefill.hcNormed", max(totalHcBytes, 64)),
                      let hcBottleneckBuffer_all = pooled("prefill.hcBottleneck", max(P * 512, 64) * MemoryLayout<Float>.stride),
                      let hcInjectScaleBuffer_all = pooled("prefill.hcInjectScale", max(P * 4, 64) * MemoryLayout<Float>.stride),
                      let denseActiveTokensBuffer = pooled("prefill.denseTok", max(P, 1) * MemoryLayout<UInt32>.stride),
                      let denseActiveWeightsBuffer = pooled("prefill.denseW", max(P, 1) * MemoryLayout<Float>.stride),
                      let expertActiveTokensBuffer = pooled("prefill.expertTok", max(P * topKCount, 64) * MemoryLayout<UInt32>.stride),
                      let expertActiveWeightsBuffer = pooled("prefill.expertW", max(P * topKCount, 64) * MemoryLayout<Float>.stride),
                      let prefillStagingBufferA = pooled("prefill.stagingA", max(prefillStagingSlotCap * Int(loadedLayout?.expert_size ?? 3151872), 64)),
                      let prefillStagingBufferB = pooled("prefill.stagingB", max(prefillStagingSlotCap * Int(loadedLayout?.expert_size ?? 3151872), 64)) else {
                    return false
                }
                // Packs a layer's active experts into one staging buffer: reused predicted
                // slots first, then the lowest free slots. Bounded by the buffer capacity,
                // so a mispredicted prefetch can never write past the end (the predicted
                // set and this layer's set can be disjoint, which previously needed a 2x
                // buffer; packing instead keeps a single-layer buffer sufficient).
                func packPrefillExpertSlots(predicted: [Int: Int]?, activeIds: [Int]) -> (map: [Int: Int], misses: [Int]) {
                    var map: [Int: Int] = [:]
                    var used = [Bool](repeating: false, count: prefillStagingSlotCap)
                    if let predicted {
                        for expId in activeIds {
                            if let slot = predicted[expId], slot >= 0, slot < prefillStagingSlotCap, !used[slot] {
                                map[expId] = slot
                                used[slot] = true
                            }
                        }
                    }
                    var misses: [Int] = []
                    var cursor = 0
                    for expId in activeIds where map[expId] == nil {
                        while cursor < prefillStagingSlotCap && used[cursor] { cursor += 1 }
                        if cursor >= prefillStagingSlotCap { break }
                        map[expId] = cursor
                        used[cursor] = true
                        misses.append(expId)
                    }
                    return (map, misses)
                }
                // FIX #6a: flash-decoding partials for multi-row prefill attention.
                let prefillAttnChunkRows = 64
                let prefillAttnSeqMax = max(Int(KVCacheManager.shared.allocatedSeqLen) + 64, 1)
                let prefillAttnChunksMax = max(1, min(64, (prefillAttnSeqMax + 1023) / 1024))
                guard let attnPartialMBuffer = device.makeBuffer(length: max(prefillAttnChunkRows * Int(numHeads) * prefillAttnChunksMax, 64) * MemoryLayout<Float>.stride, options: .storageModeShared),
                      let attnPartialLBuffer = device.makeBuffer(length: max(prefillAttnChunkRows * Int(numHeads) * prefillAttnChunksMax, 64) * MemoryLayout<Float>.stride, options: .storageModeShared),
                      let attnPartialAccBuffer = device.makeBuffer(length: max(prefillAttnChunkRows * Int(numHeads) * prefillAttnChunksMax * Int(headDim), 64) * MemoryLayout<Float>.stride, options: .storageModeShared) else {
                    return false
                }
                // FIX #5: double-buffered expert staging — layer l's MoE GPU reads one
                // buffer while layer l+1's speculative preads fill the other.
                var prefillStagingBuffer = prefillStagingBufferA
                var lastMoeStageIsA = false
                let prefillPipelineEnabled = UserDefaults.standard.object(forKey: "dynamoe_prefill_pipeline") as? Bool ?? true
                // In-flight speculative kick state, keyed by the target layer index.
                var prefetchedFor: [Int: (buf: MTLBuffer, slotMap: [Int: Int], slotBase: Int, sem: DispatchSemaphore)] = [:]
                defer {
                    for (_, pf) in prefetchedFor { pf.sem.wait() }
                }
                var prefillDiagIoMs = 0.0
                var prefillDiagGpuMs = 0.0
                var prefillDiagPhaseAMs = 0.0
                var prefillDiagLayers = 0
                var prefillDiagHits = 0
                var prefillDiagTotal = 0

                let denseTokPtr = denseActiveTokensBuffer.contents().bindMemory(to: UInt32.self, capacity: P)
                for i in 0..<P { denseTokPtr[i] = UInt32(i) }
                let denseWgtPtr = denseActiveWeightsBuffer.contents().bindMemory(to: Float.self, capacity: P)
                for i in 0..<P { denseWgtPtr[i] = 1.0 }

                var hDim = hiddenDim
                let prefillStartTime = CFAbsoluteTimeGetCurrent()
                var lastUIUpdateTime = CFAbsoluteTimeGetCurrent()

                // Step 0: Initial Token Embeddings for all P prompt tokens into hBufA
                guard let embedCmd = commandQueue.makeCommandBuffer(),
                      let embedEnc = embedCmd.makeComputeCommandEncoder() else {
                    print("❌ [PREFILL] embed command buffer allocation failed (chunk startPos=\(startPos), P=\(P))")
                    return false
                }

                let hasEmbedScale = (embedScale != nil)
                let hasEmbedBias = (embedBias != nil)
                let isEmbedMXFP8 = hasEmbedScale && (embedScale!.dtype.contains("U8") || embedScale!.dtype.contains("UINT8") || (!hasEmbedBias && (embedWeight.dtype.contains("U32") || embedWeight.dtype.contains("U8") || embedWeight.dtype.contains("FP8")) && (embedScale!.offsetEnd - embedScale!.offsetStart) >= UInt64(hiddenDim / 32)))
                let isEmbedAffine = (hasEmbedBias || embedWeight.dtype.contains("Q4") || embedWeight.dtype.contains("Q8") || hasEmbedScale) && !isEmbedMXFP8

                let promptTokensBuffer = device.makeBuffer(bytes: promptTokens, length: P * MemoryLayout<UInt32>.stride, options: .storageModeShared)

                for p in 0..<P {
                    let tokOffset = p * MemoryLayout<UInt32>.stride
                    let outOffset = p * Int(hiddenDim) * MemoryLayout<Float>.stride
                    let tok = promptTokens[p]

                    if isEmbedMXFP8, let embedMXFP8Pipe = embedMXFP8Pipeline,
                       let embedScaleRaw = buffers[embedScale!.shardIndex] {
                        var wOffset = embedOffset
                        var sOffset = embedScale!.offsetStart
                        var hDimVal = hiddenDim
                        embedEnc.setComputePipelineState(embedMXFP8Pipe)
                        embedEnc.setBuffer(embedShardBuffer, offset: 0, index: 0)
                        embedEnc.setBuffer(promptTokensBuffer, offset: tokOffset, index: 1)
                        embedEnc.setBuffer(hBufA, offset: outOffset, index: 2)
                        embedEnc.setBuffer(embedScaleRaw, offset: 0, index: 3)
                        embedEnc.setBytes(&wOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                        embedEnc.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                        embedEnc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 6)
                        embedEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), embedMXFP8Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    } else if isEmbedAffine,
                       let embedScaleRaw = (embedScale != nil) ? buffers[embedScale!.shardIndex] : nil,
                       let embedBiasRaw = (embedBias != nil) ? buffers[embedBias!.shardIndex] : nil {
                        var wOffset = embedOffset
                        var sOffset = embedScale!.offsetStart
                        var bOffset = embedBias!.offsetStart
                        var tokVal = tok
                        var grpSize: UInt32 = 64
                        let is8Bit = (embedWeight.offsetEnd - embedWeight.offsetStart) >= (UInt64(vocabSize) * UInt64(hiddenDim) * 3) / 4
                        if is8Bit, let embedQ8Pipe = embedQ8Pipeline {
                            embedEnc.setComputePipelineState(embedQ8Pipe)
                            embedEnc.setBuffer(embedShardBuffer, offset: 0, index: 0)
                            embedEnc.setBuffer(embedScaleRaw, offset: 0, index: 1)
                            embedEnc.setBuffer(embedBiasRaw, offset: 0, index: 2)
                            embedEnc.setBuffer(hBufA, offset: outOffset, index: 3)
                            embedEnc.setBytes(&tokVal, length: MemoryLayout<UInt32>.stride, index: 4)
                            embedEnc.setBytes(&wOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                            embedEnc.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                            embedEnc.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                            embedEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 8)
                            embedEnc.setBytes(&grpSize, length: MemoryLayout<UInt32>.stride, index: 9)
                            embedEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), embedQ8Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                        } else if let embedQ4Pipe = embedQ4Pipeline {
                            embedEnc.setComputePipelineState(embedQ4Pipe)
                            embedEnc.setBuffer(embedShardBuffer, offset: 0, index: 0)
                            embedEnc.setBuffer(embedScaleRaw, offset: 0, index: 1)
                            embedEnc.setBuffer(embedBiasRaw, offset: 0, index: 2)
                            embedEnc.setBuffer(hBufA, offset: outOffset, index: 3)
                            embedEnc.setBytes(&tokVal, length: MemoryLayout<UInt32>.stride, index: 4)
                            embedEnc.setBytes(&wOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                            embedEnc.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                            embedEnc.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                            embedEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 8)
                            embedEnc.setBytes(&grpSize, length: MemoryLayout<UInt32>.stride, index: 9)
                            embedEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), embedQ4Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                        }
                    } else {
                        var wOffset = embedOffset
                        var tokCount: UInt32 = 1
                        embedEnc.setComputePipelineState(embedPipeline)
                        embedEnc.setBuffer(embedShardBuffer, offset: 0, index: 0)
                        embedEnc.setBuffer(promptTokensBuffer, offset: tokOffset, index: 1)
                        embedEnc.setBuffer(hBufA, offset: outOffset, index: 2)
                        embedEnc.setBytes(&wOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                        embedEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                        embedEnc.setBytes(&tokCount, length: MemoryLayout<UInt32>.stride, index: 5)
                        embedEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), embedPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    }
                }

                if isGemma4Prefill, let scalePipe = scaleVectorInplacePipeline {
                    // Gemma 4 scales token embeddings by sqrt(hidden_size) (per token).
                    var dimV = hiddenDim
                    var sc = Float(Double(hiddenDim).squareRoot())
                    embedEnc.setComputePipelineState(scalePipe)
                    for p in 0..<P {
                        let off = p * Int(hiddenDim) * MemoryLayout<Float>.stride
                        embedEnc.setBuffer(hBufA, offset: off, index: 0)
                        embedEnc.setBytes(&dimV, length: MemoryLayout<UInt32>.stride, index: 1)
                        embedEnc.setBytes(&sc, length: MemoryLayout<Float>.stride, index: 2)
                        embedEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), scalePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    }
                    embedEnc.memoryBarrier(scope: .buffers)
                }

                if hasHc, let initStreams = fusedInit4StreamsPipeline {
                    var hDimVal = hiddenDim
                    embedEnc.setComputePipelineState(initStreams)
                    embedEnc.setBuffer(hBufA, offset: 0, index: 0)
                    embedEnc.setBuffer(hcStreamsBuffer_all, offset: 0, index: 1)
                    embedEnc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 2)
                    embedEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), initStreams.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                }

                embedEnc.endEncoding()
                embedCmd.commit()
                embedCmd.waitUntilCompleted()
                if let embedErr = embedCmd.error {
                    print("❌ [METAL ERROR] prefill embedCmd aborted: \(embedErr)")
                    return false
                }

                var currHBuf = hBufA
                var nextHBuf = hBufB

                // Step 1: Layer-Wise Forward Pass: Outer Loop Layers, Inner Loop Tokens
                let totalPasses = totalLoops * actualLayers
                var passIdx = 0
                // Chunked Gemma prefill re-enters this function once per token chunk. Report
                // progress against the whole turn's layer count so the UI advances monotonically
                // instead of restarting at layer 1 for every chunk (which reads as a re-prefill).
                let prefillLayerBase = max(0, chunkIndex) * totalPasses
                let prefillLayerTotal = max(1, chunkCount) * totalPasses

                for loopIdx in 0..<totalLoops {
                    for l in 0..<actualLayers {
                        if Task.isCancelled { return false }
                        let layer = cachedLayers[l]
                        let intermediateDim = layer.intermediateDim

                        if speculativePrefetchEnabled && packedExpertsDir == nil {
                            let nextL = (l + 1) < actualLayers ? (l + 1) : 0
                            WorkingSetManager.shared.prefetchLayerBackbone(layer: cachedLayers[nextL], shardBuffers: buffers)
                        }

                        // ================= Gemma 4 hybrid-attention MoE decoder layer =================
                        // Mirrors the per-token Gemma path (see runTokenForward) but processes every
                        // prompt token per layer, so each layer's packed-expert reads are amortized
                        // over the whole prompt instead of being repeated once per token.
                        if isGemma4Prefill {
                            let gemmaScale: Float = 1.0 / Float(Double(hiddenDim).squareRoot())
                            var hDimV = hiddenDim
                            var epsV = eps
                            let gSliding = layer.isSlidingAttention
                            let gHeadDim: UInt32 = gSliding ? headDim : max(headDim, globalHeadDim)
                            let gNumKv: UInt32 = gSliding ? numKvHeads : max(1, numGlobalKvHeads)
                            let gQDim: UInt32 = numHeads * gHeadDim
                            let gKvDim: UInt32 = gNumKv * gHeadDim

                            guard let gCmd = commandQueue.makeCommandBuffer(),
                                  let gEnc = gCmd.makeComputeCommandEncoder() else {
                                print("❌ [PREFILL] Gemma layer \(l) Phase A command buffer allocation failed (chunk startPos=\(startPos), P=\(P))")
                                return false
                            }

                            // 1. RMSNorm1(currH) -> xNorm1Buffer_all
                            if let norm1 = layer.norm1Tensor, let norm1Raw = buffers[norm1.shardIndex] {
                                var gOff = norm1.offsetStart
                                gEnc.setComputePipelineState(rmsnormPipeline)
                                gEnc.setBuffer(currHBuf, offset: 0, index: 0)
                                gEnc.setBuffer(norm1Raw, offset: 0, index: 1)
                                gEnc.setBuffer(xNorm1Buffer_all, offset: 0, index: 2)
                                gEnc.setBytes(&gOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                gEnc.setBytes(&hDimV, length: MemoryLayout<UInt32>.stride, index: 4)
                                gEnc.setBytes(&epsV, length: MemoryLayout<Float>.stride, index: 5)
                                gEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                                gEnc.dispatchThreadgroups(MTLSize(width: P, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                                gEnc.memoryBarrier(scope: .buffers)
                            }

                            // 2. Q/K/V projections (batched across P tokens)
                            dispatchLinear(enc: gEnc, weight: layer.qProjTensor, scale: layer.qScaleTensor, bias: layer.qBiasTensor, inBuf: xNorm1Buffer_all, outBuf: qGateBuffer_all, inDim: hiddenDim, outDim: gQDim, batchSize: P)
                            dispatchLinear(enc: gEnc, weight: layer.kProjTensor, scale: layer.kScaleTensor, bias: layer.kBiasTensor, inBuf: xNorm1Buffer_all, outBuf: kVectorBuffer_all, inDim: hiddenDim, outDim: gKvDim, batchSize: P)
                            if gSliding {
                                dispatchLinear(enc: gEnc, weight: layer.vProjTensor, scale: layer.vScaleTensor, bias: layer.vBiasTensor, inBuf: xNorm1Buffer_all, outBuf: vVectorBuffer_all, inDim: hiddenDim, outDim: gKvDim, batchSize: P)
                            }
                            gEnc.memoryBarrier(scope: .buffers)

                            // 3. Value norm (no scale) per head, per token. Sliding layers norm the
                            // v_proj output; global layers (attention_k_eq_v) norm the raw key output.
                            if let vNormPipe = rmsnormNoScalePipeline {
                                var vDim = gHeadDim
                                var vEps = epsV
                                gEnc.setComputePipelineState(vNormPipe)
                                let vInBuf = gSliding ? vVectorBuffer_all : kVectorBuffer_all
                                for r in 0..<P {
                                    let base = r * Int(gKvDim) * MemoryLayout<Float>.stride
                                    gEnc.setBuffer(vInBuf, offset: base, index: 0)
                                    gEnc.setBuffer(vVectorBuffer_all, offset: base, index: 1)
                                    gEnc.setBytes(&vDim, length: MemoryLayout<UInt32>.stride, index: 2)
                                    gEnc.setBytes(&vEps, length: MemoryLayout<Float>.stride, index: 3)
                                    gEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                                    gEnc.dispatchThreadgroups(MTLSize(width: Int(gNumKv), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(gHeadDim)), height: 1, depth: 1))
                                }
                                gEnc.memoryBarrier(scope: .buffers)
                            }

                            // 4. Q/K per-head RMSNorm (batched across P tokens)
                            if let qNorm = layer.qNormTensor, let qNormRaw = buffers[qNorm.shardIndex], let headNormPipe = headRmsnormPipeline {
                                var qNormOff = qNorm.offsetStart
                                var nQ = numHeads
                                var hD = gHeadDim
                                var hStride = gHeadDim
                                gEnc.setComputePipelineState(headNormPipe)
                                gEnc.setBuffer(qGateBuffer_all, offset: 0, index: 0)
                                gEnc.setBuffer(qNormRaw, offset: 0, index: 1)
                                gEnc.setBytes(&qNormOff, length: MemoryLayout<UInt64>.stride, index: 2)
                                gEnc.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 3)
                                gEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                                gEnc.setBytes(&hStride, length: MemoryLayout<UInt32>.stride, index: 5)
                                gEnc.setBytes(&epsV, length: MemoryLayout<Float>.stride, index: 6)
                                gEnc.dispatchThreadgroups(MTLSize(width: Int(numHeads), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                gEnc.memoryBarrier(scope: .buffers)
                            }
                            if let kNorm = layer.kNormTensor, let kNormRaw = buffers[kNorm.shardIndex], let headNormPipe = headRmsnormPipeline {
                                var kNormOff = kNorm.offsetStart
                                var nK = gNumKv
                                var hD = gHeadDim
                                var hStride = gHeadDim
                                gEnc.setComputePipelineState(headNormPipe)
                                gEnc.setBuffer(kVectorBuffer_all, offset: 0, index: 0)
                                gEnc.setBuffer(kNormRaw, offset: 0, index: 1)
                                gEnc.setBytes(&kNormOff, length: MemoryLayout<UInt64>.stride, index: 2)
                                gEnc.setBytes(&nK, length: MemoryLayout<UInt32>.stride, index: 3)
                                gEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                                gEnc.setBytes(&hStride, length: MemoryLayout<UInt32>.stride, index: 5)
                                gEnc.setBytes(&epsV, length: MemoryLayout<Float>.stride, index: 6)
                                gEnc.dispatchThreadgroups(MTLSize(width: Int(gNumKv), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                gEnc.memoryBarrier(scope: .buffers)
                            }

                            // 5. RoPE (batched; per-layer-type theta + proportional rotary dim)
                            let gemmaRopePipe: MTLComputePipelineState? = gSliding ? ropePipeline : (ropeProportionalPipeline ?? ropePipeline)
                            if let ropePipe = gemmaRopePipe {
                                var pos = startPos
                                var nQ = numHeads
                                var nK = gNumKv
                                var hD = gHeadDim
                                var rD = UInt32(modelConfig?.effectiveRotaryDim(layerIndex: l, headDim: Int(gHeadDim)) ?? Int(rotaryDim))
                                var qStr = gHeadDim
                                var kStr = gHeadDim
                                var theta = modelConfig?.effectiveRopeTheta(layerIndex: l) ?? thetaVal
                                gEnc.setComputePipelineState(ropePipe)
                                gEnc.setBuffer(qGateBuffer_all, offset: 0, index: 0)
                                gEnc.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 1)
                                gEnc.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 2)
                                gEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 3)
                                gEnc.setBytes(&rD, length: MemoryLayout<UInt32>.stride, index: 4)
                                gEnc.setBytes(&qStr, length: MemoryLayout<UInt32>.stride, index: 5)
                                gEnc.setBytes(&theta, length: MemoryLayout<Float>.stride, index: 6)
                                gEnc.dispatchThreads(MTLSize(width: Int(numHeads), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), ropePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                gEnc.setBuffer(kVectorBuffer_all, offset: 0, index: 0)
                                gEnc.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 1)
                                gEnc.setBytes(&nK, length: MemoryLayout<UInt32>.stride, index: 2)
                                gEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 3)
                                gEnc.setBytes(&rD, length: MemoryLayout<UInt32>.stride, index: 4)
                                gEnc.setBytes(&kStr, length: MemoryLayout<UInt32>.stride, index: 5)
                                gEnc.setBytes(&theta, length: MemoryLayout<Float>.stride, index: 6)
                                gEnc.dispatchThreads(MTLSize(width: Int(gNumKv), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(gNumKv), ropePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                gEnc.memoryBarrier(scope: .buffers)
                            }

                            // 6. KV store (batched) + per-row attention decode
                            if let kCache = KVCacheManager.shared.kCacheBuffer,
                               let vCache = KVCacheManager.shared.vCacheBuffer {
                                let slot = (loopIdx * actualLayers) + layer.fullAttnIndex
                                let kvMgr = KVCacheManager.shared
                                let prec = kvMgr.activePrecision
                                let layerByteOffset = kvMgr.kvSlotByteOffset(slot: slot, stride: Int(gemmaKvStride), elementBytes: prec.bytesPerElement)
                                var cacheStride = gemmaKvStride
                                var pos = startPos
                                var nKv = gNumKv
                                var hD = gHeadDim
                                // Sliding layers address a ring-sized slot region; the
                                // chunk cap bounds P so the batch never overwrites a
                                // row a later row in this pass still reads.
                                var ringLen: UInt32 = kvMgr.isRingSlot(slot) ? UInt32(kvMgr.slidingRingLen) : 0
                                if prec == .fp8,
                                   let kScale = KVCacheManager.shared.kScaleBuffer,
                                   let vScale = KVCacheManager.shared.vScaleBuffer,
                                   let storePipe = storeKvGemmaFP8Pipeline,
                                   let attnPipe = gqaGemmaFP8Pipeline {
                                    let scaleByteOffset = kvMgr.kvSlotScaleByteOffset(slot: slot, layoutHeads: Int(numKvHeads))
                                    gEnc.setComputePipelineState(storePipe)
                                    gEnc.setBuffer(kVectorBuffer_all, offset: 0, index: 0)
                                    gEnc.setBuffer(vVectorBuffer_all, offset: 0, index: 1)
                                    gEnc.setBuffer(kCache, offset: layerByteOffset, index: 2)
                                    gEnc.setBuffer(vCache, offset: layerByteOffset, index: 3)
                                    gEnc.setBuffer(kScale, offset: scaleByteOffset, index: 4)
                                    gEnc.setBuffer(vScale, offset: scaleByteOffset, index: 5)
                                    gEnc.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 6)
                                    gEnc.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 7)
                                    gEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 8)
                                    gEnc.setBytes(&cacheStride, length: MemoryLayout<UInt32>.stride, index: 9)
                                    gEnc.setBytes(&ringLen, length: MemoryLayout<UInt32>.stride, index: 10)
                                    gEnc.dispatchThreadgroups(MTLSize(width: Int(gNumKv), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    gEnc.memoryBarrier(scope: .buffers)
                                    // Ring bookkeeping: this layer just stored [startPos, startPos+P).
                                    kvMgr.noteRingStores(range: Int(startPos)..<Int(startPos) + P)

                                    var nQ = numHeads
                                    var windowSize: UInt32 = gSliding ? UInt32(modelConfig?.effectiveSlidingWindow ?? 0) : 0
                                    var scaling = gemmaScaling
                                    gEnc.setComputePipelineState(attnPipe)
                                    for r in 0..<P {
                                        let qOff = r * Int(gQDim) * MemoryLayout<Float>.stride
                                        var seqLen = startPos + UInt32(r) + 1
                                        gEnc.setBuffer(qGateBuffer_all, offset: qOff, index: 0)
                                        gEnc.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                        gEnc.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                        gEnc.setBuffer(kScale, offset: scaleByteOffset, index: 3)
                                        gEnc.setBuffer(vScale, offset: scaleByteOffset, index: 4)
                                        gEnc.setBuffer(attnCtxBuffer_all, offset: qOff, index: 5)
                                        gEnc.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 6)
                                        gEnc.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 7)
                                        gEnc.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 8)
                                        gEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 9)
                                        gEnc.setBytes(&windowSize, length: MemoryLayout<UInt32>.stride, index: 10)
                                        gEnc.setBytes(&scaling, length: MemoryLayout<Float>.stride, index: 11)
                                        gEnc.setBytes(&cacheStride, length: MemoryLayout<UInt32>.stride, index: 12)
                                        gEnc.setBytes(&ringLen, length: MemoryLayout<UInt32>.stride, index: 13)
                                        gEnc.dispatchThreadgroups(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    }
                                    gEnc.memoryBarrier(scope: .buffers)
                                } else {
                                let storePipe: MTLComputePipelineState? = (prec == .fp16) ? (storeKvGemmaF16Pipeline ?? storeKvGemmaPipeline) : storeKvGemmaPipeline
                                if let storePipe {
                                    gEnc.setComputePipelineState(storePipe)
                                    gEnc.setBuffer(kVectorBuffer_all, offset: 0, index: 0)
                                    gEnc.setBuffer(vVectorBuffer_all, offset: 0, index: 1)
                                    gEnc.setBuffer(kCache, offset: layerByteOffset, index: 2)
                                    gEnc.setBuffer(vCache, offset: layerByteOffset, index: 3)
                                    gEnc.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 4)
                                    gEnc.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 5)
                                    gEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 6)
                                    gEnc.setBytes(&cacheStride, length: MemoryLayout<UInt32>.stride, index: 7)
                                    gEnc.setBytes(&ringLen, length: MemoryLayout<UInt32>.stride, index: 8)
                                    gEnc.dispatchThreads(MTLSize(width: Int(gKvDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(gKvDim), storePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    gEnc.memoryBarrier(scope: .buffers)
                                    // Ring bookkeeping: this layer just stored [startPos, startPos+P).
                                    kvMgr.noteRingStores(range: Int(startPos)..<Int(startPos) + P)
                                }

                                var nQ = numHeads
                                var windowSize: UInt32 = gSliding ? UInt32(modelConfig?.effectiveSlidingWindow ?? 0) : 0
                                var scaling = gemmaScaling
                                let attnPipe: MTLComputePipelineState? = (prec == .fp16) ? (gqaGemmaF16Pipeline ?? gqaGemmaPipeline) : gqaGemmaPipeline
                                if let attnPipe {
                                    gEnc.setComputePipelineState(attnPipe)
                                    for r in 0..<P {
                                        let qOff = r * Int(gQDim) * MemoryLayout<Float>.stride
                                        var seqLen = startPos + UInt32(r) + 1
                                        gEnc.setBuffer(qGateBuffer_all, offset: qOff, index: 0)
                                        gEnc.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                        gEnc.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                        gEnc.setBuffer(attnCtxBuffer_all, offset: qOff, index: 3)
                                        gEnc.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 4)
                                        gEnc.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 5)
                                        gEnc.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 6)
                                        gEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 7)
                                        gEnc.setBytes(&windowSize, length: MemoryLayout<UInt32>.stride, index: 8)
                                        gEnc.setBytes(&scaling, length: MemoryLayout<Float>.stride, index: 9)
                                        gEnc.setBytes(&cacheStride, length: MemoryLayout<UInt32>.stride, index: 10)
                                        gEnc.setBytes(&ringLen, length: MemoryLayout<UInt32>.stride, index: 11)
                                        gEnc.dispatchThreadgroups(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    }
                                    gEnc.memoryBarrier(scope: .buffers)
                                }
                                }
                            }

                            // 7. o_proj
                            dispatchLinear(enc: gEnc, weight: layer.oProjTensor, scale: layer.oScaleTensor, bias: layer.oBiasTensor, inBuf: attnCtxBuffer_all, outBuf: attnOutBuffer_all, inDim: gQDim, outDim: hiddenDim, batchSize: P)
                            gEnc.memoryBarrier(scope: .buffers)

                            // 8. hMid = currH + post_attention_layernorm(attn)
                            if let norm2 = layer.norm2Tensor, let norm2Raw = buffers[norm2.shardIndex] {
                                var gOff = norm2.offsetStart
                                gEnc.setComputePipelineState(rmsnormPipeline)
                                gEnc.setBuffer(attnOutBuffer_all, offset: 0, index: 0)
                                gEnc.setBuffer(norm2Raw, offset: 0, index: 1)
                                gEnc.setBuffer(xNorm2Buffer_all, offset: 0, index: 2)
                                gEnc.setBytes(&gOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                gEnc.setBytes(&hDimV, length: MemoryLayout<UInt32>.stride, index: 4)
                                gEnc.setBytes(&epsV, length: MemoryLayout<Float>.stride, index: 5)
                                gEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                                gEnc.dispatchThreadgroups(MTLSize(width: P, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                                gEnc.memoryBarrier(scope: .buffers)
                            }
                            gEnc.setComputePipelineState(addPipeline)
                            gEnc.setBuffer(currHBuf, offset: 0, index: 0)
                            gEnc.setBuffer(xNorm2Buffer_all, offset: 0, index: 1)
                            gEnc.setBuffer(hMidBuffer_all, offset: 0, index: 2)
                            gEnc.setBytes(&hDimV, length: MemoryLayout<UInt32>.stride, index: 3)
                            gEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            gEnc.memoryBarrier(scope: .buffers)

                            // 9. pre_feedforward_layernorm(hMid) -> xNorm2Buffer_all
                            if let preFfn = layer.preFfnNormTensor, let preFfnRaw = buffers[preFfn.shardIndex] {
                                var gOff = preFfn.offsetStart
                                gEnc.setComputePipelineState(rmsnormPipeline)
                                gEnc.setBuffer(hMidBuffer_all, offset: 0, index: 0)
                                gEnc.setBuffer(preFfnRaw, offset: 0, index: 1)
                                gEnc.setBuffer(xNorm2Buffer_all, offset: 0, index: 2)
                                gEnc.setBytes(&gOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                gEnc.setBytes(&hDimV, length: MemoryLayout<UInt32>.stride, index: 4)
                                gEnc.setBytes(&epsV, length: MemoryLayout<Float>.stride, index: 5)
                                gEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                                gEnc.dispatchThreadgroups(MTLSize(width: P, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                                gEnc.memoryBarrier(scope: .buffers)
                            }

                            // 10. Dense shared MLP (GeGLU) batched -> hMlpBuffer_all; postFfnNorm1 in place
                            gEnc.setComputePipelineState(clearPipeline)
                            gEnc.setBuffer(hMlpBuffer_all, offset: 0, index: 0)
                            gEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            gEnc.memoryBarrier(scope: .buffers)
                            if let dG = layer.denseGateWeight, let dU = layer.denseUpWeight, let dD = layer.denseDownWeight,
                               let dGRaw = buffers[dG.shardIndex], let dURaw = buffers[dU.shardIndex], let dDRaw = buffers[dD.shardIndex],
                               let geluBatched = bf16GeluGateUpBatchedPipeline,
                               let downBatched = bf16DownBatchedPipeline {
                                var gOff = dG.offsetStart
                                var uOff = dU.offsetStart
                                var dOff = dD.offsetStart
                                var interDimV = layer.intermediateDim
                                gEnc.setComputePipelineState(geluBatched)
                                gEnc.setBuffer(dGRaw, offset: 0, index: 0)
                                gEnc.setBuffer(dURaw, offset: 0, index: 1)
                                gEnc.setBuffer(xNorm2Buffer_all, offset: 0, index: 2)
                                gEnc.setBuffer(interBuffer_all, offset: 0, index: 3)
                                gEnc.setBytes(&gOff, length: MemoryLayout<UInt64>.stride, index: 4)
                                gEnc.setBytes(&uOff, length: MemoryLayout<UInt64>.stride, index: 5)
                                gEnc.setBytes(&hDimV, length: MemoryLayout<UInt32>.stride, index: 6)
                                gEnc.setBytes(&interDimV, length: MemoryLayout<UInt32>.stride, index: 7)
                                gEnc.setBuffer(denseActiveTokensBuffer, offset: 0, index: 8)
                                gEnc.dispatchThreadgroups(MTLSize(width: Int(layer.intermediateDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                gEnc.memoryBarrier(scope: .buffers)
                                var pkVal: Float = 1.0
                                gEnc.setComputePipelineState(downBatched)
                                gEnc.setBuffer(dDRaw, offset: 0, index: 0)
                                gEnc.setBuffer(interBuffer_all, offset: 0, index: 1)
                                gEnc.setBuffer(hMlpBuffer_all, offset: 0, index: 2)
                                gEnc.setBytes(&dOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                gEnc.setBytes(&interDimV, length: MemoryLayout<UInt32>.stride, index: 4)
                                gEnc.setBytes(&hDimV, length: MemoryLayout<UInt32>.stride, index: 5)
                                gEnc.setBytes(&pkVal, length: MemoryLayout<Float>.stride, index: 6)
                                gEnc.setBuffer(denseActiveTokensBuffer, offset: 0, index: 7)
                                gEnc.setBuffer(denseActiveWeightsBuffer, offset: 0, index: 8)
                                gEnc.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                gEnc.memoryBarrier(scope: .buffers)
                            }
                            if let pf1 = layer.postFfnNorm1Tensor, let pf1Raw = buffers[pf1.shardIndex] {
                                var gOff = pf1.offsetStart
                                gEnc.setComputePipelineState(rmsnormPipeline)
                                gEnc.setBuffer(hMlpBuffer_all, offset: 0, index: 0)
                                gEnc.setBuffer(pf1Raw, offset: 0, index: 1)
                                gEnc.setBuffer(hMlpBuffer_all, offset: 0, index: 2)
                                gEnc.setBytes(&gOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                gEnc.setBytes(&hDimV, length: MemoryLayout<UInt32>.stride, index: 4)
                                gEnc.setBytes(&epsV, length: MemoryLayout<Float>.stride, index: 5)
                                gEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                                gEnc.dispatchThreadgroups(MTLSize(width: P, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                                gEnc.memoryBarrier(scope: .buffers)
                            }

                            // 11. Router: rmsnorm_no_scale(hMid) * router.scale * hidden^-0.5 -> top-k
                            if let vNormPipe = rmsnormNoScalePipeline {
                                var dimV = hiddenDim
                                var eV = epsV
                                gEnc.setComputePipelineState(vNormPipe)
                                gEnc.setBuffer(hMidBuffer_all, offset: 0, index: 0)
                                gEnc.setBuffer(attnOutBuffer_all, offset: 0, index: 1)
                                gEnc.setBytes(&dimV, length: MemoryLayout<UInt32>.stride, index: 2)
                                gEnc.setBytes(&eV, length: MemoryLayout<Float>.stride, index: 3)
                                gEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                                gEnc.dispatchThreadgroups(MTLSize(width: P, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                                gEnc.memoryBarrier(scope: .buffers)
                            }
                            if let rsv = layer.routerScaleVec, let rsvRaw = buffers[rsv.shardIndex], let mulPipe = mulByBf16VectorPipeline {
                                var rsvOff = rsv.offsetStart
                                var dimV = hiddenDim
                                var sc = gemmaScale
                                gEnc.setComputePipelineState(mulPipe)
                                for r in 0..<P {
                                    let off = r * Int(hiddenDim) * MemoryLayout<Float>.stride
                                    gEnc.setBuffer(attnOutBuffer_all, offset: off, index: 0)
                                    gEnc.setBuffer(rsvRaw, offset: 0, index: 1)
                                    gEnc.setBuffer(bVectorBuffer_all, offset: off, index: 2)
                                    gEnc.setBytes(&rsvOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                    gEnc.setBytes(&dimV, length: MemoryLayout<UInt32>.stride, index: 4)
                                    gEnc.setBytes(&sc, length: MemoryLayout<Float>.stride, index: 5)
                                    gEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), mulPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                }
                                gEnc.memoryBarrier(scope: .buffers)
                            }
                            if let routerTensor = layer.routerTensor, let routerRaw = buffers[routerTensor.shardIndex] {
                                var rOffset = routerTensor.offsetStart
                                var nExp = numExperts
                                var kVal: UInt32 = UInt32(modelConfig?.effectiveNumExpertsPerTok ?? 8)
                                gEnc.setComputePipelineState(routerPipeline)
                                for r in 0..<P {
                                    let tOff = r * Int(hiddenDim) * MemoryLayout<Float>.stride
                                    let rIndexOffset = r * topKCount * MemoryLayout<UInt32>.stride
                                    let rWeightOffset = r * topKCount * MemoryLayout<Float>.stride
                                    gEnc.setBuffer(routerRaw, offset: 0, index: 0)
                                    gEnc.setBuffer(bVectorBuffer_all, offset: tOff, index: 1)
                                    gEnc.setBuffer(routerIndicesBuffer_all, offset: rIndexOffset, index: 2)
                                    gEnc.setBuffer(routerWeightsBuffer_all, offset: rWeightOffset, index: 3)
                                    gEnc.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                                    gEnc.setBytes(&hDimV, length: MemoryLayout<UInt32>.stride, index: 5)
                                    gEnc.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 6)
                                    gEnc.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 7)
                                    gEnc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: Int(numExperts), height: 1, depth: 1))
                                }
                                gEnc.memoryBarrier(scope: .buffers)
                                if let pes = layer.routerPerExpertScale, let pesRaw = buffers[pes.shardIndex], let pesPipe = perExpertScalePipeline {
                                    var pesOff = pes.offsetStart
                                    var kTop = kVal
                                    gEnc.setComputePipelineState(pesPipe)
                                    for r in 0..<P {
                                        let rIndexOffset = r * topKCount * MemoryLayout<UInt32>.stride
                                        let rWeightOffset = r * topKCount * MemoryLayout<Float>.stride
                                        gEnc.setBuffer(routerWeightsBuffer_all, offset: rWeightOffset, index: 0)
                                        gEnc.setBuffer(routerIndicesBuffer_all, offset: rIndexOffset, index: 1)
                                        gEnc.setBuffer(pesRaw, offset: 0, index: 2)
                                        gEnc.setBytes(&pesOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                        gEnc.setBytes(&kTop, length: MemoryLayout<UInt32>.stride, index: 4)
                                        gEnc.dispatchThreads(MTLSize(width: Int(kVal), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(kVal), pesPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    }
                                    gEnc.memoryBarrier(scope: .buffers)
                                }
                            }

                            gEnc.endEncoding()
                            gCmd.commit()
                            let tGemmaPhaseAStart = CFAbsoluteTimeGetCurrent()
                            gCmd.waitUntilCompleted()
                            prefillDiagPhaseAMs += (CFAbsoluteTimeGetCurrent() - tGemmaPhaseAStart) * 1000.0
                            prefillDiagLayers += 1
                            if let gErr = gCmd.error {
                                print("❌ [METAL ERROR] gemma4 prefill Phase A layer \(l): \(gErr)")
                                return false
                            }

                            // ---- Phase B: expert MLPs, expert-major batched over all P tokens ----
                            let indStride = topKCount
                            let indAllPtr = routerIndicesBuffer_all.contents().bindMemory(to: UInt32.self, capacity: P * indStride)
                            let wAllPtr = routerWeightsBuffer_all.contents().bindMemory(to: Float.self, capacity: P * indStride)

                            var expertTokenMap: [Int: [(tokenIdx: UInt32, weight: Float)]] = [:]
                            for p in 0..<P {
                                let rBase = p * indStride
                                for i in 0..<topKCount {
                                    let expId = Int(indAllPtr[rBase + i])
                                    let pk = wAllPtr[rBase + i]
                                    if pk > 0.00001 {
                                        expertTokenMap[expId, default: []].append((tokenIdx: UInt32(p), weight: pk))
                                    }
                                }
                            }

                            guard let gemmaMoeCmd = commandQueue.makeCommandBuffer(),
                                  let gemmaMoeEnc = gemmaMoeCmd.makeComputeCommandEncoder() else {
                                print("❌ [PREFILL] Gemma layer \(l) Phase B command buffer allocation failed (chunk startPos=\(startPos), P=\(P))")
                                return false
                            }

                            // pre_feedforward_layernorm_2(hMid) -> xNorm2Buffer_all
                            if let pf2 = layer.preFfnNorm2Tensor, let pf2Raw = buffers[pf2.shardIndex] {
                                var gOff = pf2.offsetStart
                                gemmaMoeEnc.setComputePipelineState(rmsnormPipeline)
                                gemmaMoeEnc.setBuffer(hMidBuffer_all, offset: 0, index: 0)
                                gemmaMoeEnc.setBuffer(pf2Raw, offset: 0, index: 1)
                                gemmaMoeEnc.setBuffer(xNorm2Buffer_all, offset: 0, index: 2)
                                gemmaMoeEnc.setBytes(&gOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                gemmaMoeEnc.setBytes(&hDimV, length: MemoryLayout<UInt32>.stride, index: 4)
                                gemmaMoeEnc.setBytes(&epsV, length: MemoryLayout<Float>.stride, index: 5)
                                gemmaMoeEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                                gemmaMoeEnc.dispatchThreadgroups(MTLSize(width: P, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                                gemmaMoeEnc.memoryBarrier(scope: .buffers)
                            }

                            // clear MoE accumulator (xNorm1Buffer_all, free after attention)
                            gemmaMoeEnc.setComputePipelineState(clearPipeline)
                            gemmaMoeEnc.setBuffer(xNorm1Buffer_all, offset: 0, index: 0)
                            gemmaMoeEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            gemmaMoeEnc.memoryBarrier(scope: .buffers)

                            if let packedDir = packedExpertsDir,
                               let fd = ExpertIOThreadPool.shared.getOrOpenLayerFD(layerIndex: l, packedExpertsDir: packedDir) {
                                let expertSize = Int(loadedLayout?.expert_size ?? 0)
                                let uniqueActiveExpIds = Array(expertTokenMap.keys).sorted()
                                if !uniqueActiveExpIds.isEmpty && expertSize > 0 {
                                    // FIX #5: consume the speculative kick that prefetched this
                                    // layer's predicted experts during the previous layer's MoE
                                    // GPU window; synchronously pread only the misses.
                                    let tIoStart = CFAbsoluteTimeGetCurrent()
                                    var predHits = 0
                                    for (idx, stale) in prefetchedFor where idx < l {
                                        stale.sem.wait()
                                        prefetchedFor[idx] = nil
                                    }
                                    let staging: MTLBuffer
                                    var predictedSlots: [Int: Int]? = nil
                                    if prefillPipelineEnabled, let pf = prefetchedFor[l] {
                                        pf.sem.wait()
                                        prefetchedFor[l] = nil
                                        predictedSlots = pf.slotMap
                                        staging = pf.buf
                                        lastMoeStageIsA = (pf.buf === prefillStagingBufferA)
                                    } else {
                                        staging = lastMoeStageIsA ? prefillStagingBufferB : prefillStagingBufferA
                                        lastMoeStageIsA = !lastMoeStageIsA
                                    }
                                    let packed = packPrefillExpertSlots(predicted: predictedSlots, activeIds: uniqueActiveExpIds)
                                    let slotMap = packed.map
                                    predHits = uniqueActiveExpIds.count - packed.misses.count
                                    let stagingPtr = staging.contents()
                                    var tasks: [ExpertPreadTask] = []
                                    for expId in packed.misses {
                                        let slot = slotMap[expId]!
                                        tasks.append(ExpertPreadTask(fd: fd, dst: stagingPtr.advanced(by: slot * expertSize), offset: off_t(expId * expertSize), size: expertSize))
                                    }
                                    if !tasks.isEmpty {
                                        ExpertIOThreadPool.shared.dispatchSync(tasks: &tasks)
                                    }
                                    prefillDiagIoMs += (CFAbsoluteTimeGetCurrent() - tIoStart) * 1000.0
                                    prefillDiagHits += predHits
                                    prefillDiagTotal += uniqueActiveExpIds.count

                                    // Component offsets within one packed expert slot. Gemma packs the
                                    // fused gate+up under "up_proj" (gate = first half, up = second
                                    // half) with no separate gate_proj; down_proj follows. Match the
                                    // per-token path's filter (names carry no ".weight" suffix).
                                    let compUpW = loadedLayout?.components.first(where: { $0.name.contains("up_proj") && !$0.name.contains("scale") && !$0.name.contains("bias") })
                                    let compDownW = loadedLayout?.components.first(where: { $0.name.contains("down_proj") && !$0.name.contains("scale") && !$0.name.contains("bias") })
                                    let upByteOffset = UInt64(gemmaExpertInterDim) * UInt64(hiddenDim) * 2

                                    var expertBufferOffsetMap: [Int: Int] = [:]
                                    var currentExpOffset = 0
                                    let expTokPtr = expertActiveTokensBuffer.contents().bindMemory(to: UInt32.self, capacity: max(P * topKCount, 64))
                                    let expWgtPtr = expertActiveWeightsBuffer.contents().bindMemory(to: Float.self, capacity: max(P * topKCount, 64))
                                    for expId in uniqueActiveExpIds {
                                        guard let tokenAssignments = expertTokenMap[expId], !tokenAssignments.isEmpty else { continue }
                                        expertBufferOffsetMap[expId] = currentExpOffset * MemoryLayout<UInt32>.stride
                                        for item in tokenAssignments {
                                            expTokPtr[currentExpOffset] = item.tokenIdx
                                            expWgtPtr[currentExpOffset] = item.weight
                                            currentExpOffset += 1
                                        }
                                    }

                                    if let geluBatched = bf16GeluGateUpBatchedPipeline, let downBatched = bf16DownBatchedPipeline {
                                        var interDimV = gemmaExpertInterDim
                                        for expId in uniqueActiveExpIds {
                                            guard let tokenAssignments = expertTokenMap[expId], let slot = slotMap[expId], !tokenAssignments.isEmpty else { continue }
                                            let count = tokenAssignments.count
                                            let expOffset = expertBufferOffsetMap[expId] ?? 0
                                            let slotOffset = UInt64(slot * expertSize)
                                            var gOff = slotOffset + (compUpW?.offset ?? 0)
                                            var uOff = slotOffset + (compUpW?.offset ?? 0) + upByteOffset
                                            var dOff = slotOffset + (compDownW?.offset ?? 0)

                                            gemmaMoeEnc.setComputePipelineState(geluBatched)
                                            gemmaMoeEnc.setBuffer(staging, offset: 0, index: 0)
                                            gemmaMoeEnc.setBuffer(staging, offset: 0, index: 1)
                                            gemmaMoeEnc.setBuffer(xNorm2Buffer_all, offset: 0, index: 2)
                                            gemmaMoeEnc.setBuffer(interBuffer_all, offset: 0, index: 3)
                                            gemmaMoeEnc.setBytes(&gOff, length: MemoryLayout<UInt64>.stride, index: 4)
                                            gemmaMoeEnc.setBytes(&uOff, length: MemoryLayout<UInt64>.stride, index: 5)
                                            gemmaMoeEnc.setBytes(&hDimV, length: MemoryLayout<UInt32>.stride, index: 6)
                                            gemmaMoeEnc.setBytes(&interDimV, length: MemoryLayout<UInt32>.stride, index: 7)
                                            gemmaMoeEnc.setBuffer(expertActiveTokensBuffer, offset: expOffset, index: 8)
                                            gemmaMoeEnc.dispatchThreadgroups(MTLSize(width: Int(interDimV), height: count, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                            gemmaMoeEnc.memoryBarrier(scope: .buffers)

                                            var pkVal: Float = 1.0
                                            gemmaMoeEnc.setComputePipelineState(downBatched)
                                            gemmaMoeEnc.setBuffer(staging, offset: 0, index: 0)
                                            gemmaMoeEnc.setBuffer(interBuffer_all, offset: 0, index: 1)
                                            gemmaMoeEnc.setBuffer(xNorm1Buffer_all, offset: 0, index: 2)
                                            gemmaMoeEnc.setBytes(&dOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                            gemmaMoeEnc.setBytes(&interDimV, length: MemoryLayout<UInt32>.stride, index: 4)
                                            gemmaMoeEnc.setBytes(&hDimV, length: MemoryLayout<UInt32>.stride, index: 5)
                                            gemmaMoeEnc.setBytes(&pkVal, length: MemoryLayout<Float>.stride, index: 6)
                                            gemmaMoeEnc.setBuffer(expertActiveTokensBuffer, offset: expOffset, index: 7)
                                            gemmaMoeEnc.setBuffer(expertActiveWeightsBuffer, offset: expOffset, index: 8)
                                            gemmaMoeEnc.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: count, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                            gemmaMoeEnc.memoryBarrier(scope: .buffers)
                                        }
                                    }
                                }
                            }

                            // post_feedforward_layernorm_2(moe) -> bVectorBuffer_all
                            if let pf2 = layer.postFfnNorm2Tensor, let pf2Raw = buffers[pf2.shardIndex] {
                                var gOff = pf2.offsetStart
                                gemmaMoeEnc.setComputePipelineState(rmsnormPipeline)
                                gemmaMoeEnc.setBuffer(xNorm1Buffer_all, offset: 0, index: 0)
                                gemmaMoeEnc.setBuffer(pf2Raw, offset: 0, index: 1)
                                gemmaMoeEnc.setBuffer(bVectorBuffer_all, offset: 0, index: 2)
                                gemmaMoeEnc.setBytes(&gOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                gemmaMoeEnc.setBytes(&hDimV, length: MemoryLayout<UInt32>.stride, index: 4)
                                gemmaMoeEnc.setBytes(&epsV, length: MemoryLayout<Float>.stride, index: 5)
                                gemmaMoeEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                                gemmaMoeEnc.dispatchThreadgroups(MTLSize(width: P, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                                gemmaMoeEnc.memoryBarrier(scope: .buffers)
                            }

                            // hMlp = postFfnNorm1(dense) + postFfnNorm2(moe)
                            gemmaMoeEnc.setComputePipelineState(addPipeline)
                            gemmaMoeEnc.setBuffer(hMlpBuffer_all, offset: 0, index: 0)
                            gemmaMoeEnc.setBuffer(bVectorBuffer_all, offset: 0, index: 1)
                            gemmaMoeEnc.setBuffer(hMlpBuffer_all, offset: 0, index: 2)
                            gemmaMoeEnc.setBytes(&hDimV, length: MemoryLayout<UInt32>.stride, index: 3)
                            gemmaMoeEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            gemmaMoeEnc.memoryBarrier(scope: .buffers)

                            // post_feedforward_layernorm(combined) -> xNorm2Buffer_all
                            if let pf = layer.postFfnNormTensor, let pfRaw = buffers[pf.shardIndex] {
                                var gOff = pf.offsetStart
                                gemmaMoeEnc.setComputePipelineState(rmsnormPipeline)
                                gemmaMoeEnc.setBuffer(hMlpBuffer_all, offset: 0, index: 0)
                                gemmaMoeEnc.setBuffer(pfRaw, offset: 0, index: 1)
                                gemmaMoeEnc.setBuffer(xNorm2Buffer_all, offset: 0, index: 2)
                                gemmaMoeEnc.setBytes(&gOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                gemmaMoeEnc.setBytes(&hDimV, length: MemoryLayout<UInt32>.stride, index: 4)
                                gemmaMoeEnc.setBytes(&epsV, length: MemoryLayout<Float>.stride, index: 5)
                                gemmaMoeEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                                gemmaMoeEnc.dispatchThreadgroups(MTLSize(width: P, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                                gemmaMoeEnc.memoryBarrier(scope: .buffers)
                            }

                            // nextH = hMid + xNorm2
                            gemmaMoeEnc.setComputePipelineState(addPipeline)
                            gemmaMoeEnc.setBuffer(hMidBuffer_all, offset: 0, index: 0)
                            gemmaMoeEnc.setBuffer(xNorm2Buffer_all, offset: 0, index: 1)
                            gemmaMoeEnc.setBuffer(nextHBuf, offset: 0, index: 2)
                            gemmaMoeEnc.setBytes(&hDimV, length: MemoryLayout<UInt32>.stride, index: 3)
                            gemmaMoeEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            gemmaMoeEnc.memoryBarrier(scope: .buffers)

                            // nextH *= layer_scalar
                            if let ls = layer.layerScalarTensor, let lsRaw = buffers[ls.shardIndex], let scalePipe = scaleVectorInplacePipeline {
                                var dimV = hiddenDim
                                let lsBits = lsRaw.contents().load(fromByteOffset: Int(ls.offsetStart), as: UInt16.self)
                                var scalarVal = Float(bitPattern: UInt32(lsBits) << 16)
                                gemmaMoeEnc.setComputePipelineState(scalePipe)
                                for r in 0..<P {
                                    let off = r * Int(hiddenDim) * MemoryLayout<Float>.stride
                                    gemmaMoeEnc.setBuffer(nextHBuf, offset: off, index: 0)
                                    gemmaMoeEnc.setBytes(&dimV, length: MemoryLayout<UInt32>.stride, index: 1)
                                    gemmaMoeEnc.setBytes(&scalarVal, length: MemoryLayout<Float>.stride, index: 2)
                                    gemmaMoeEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), scalePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                }
                                gemmaMoeEnc.memoryBarrier(scope: .buffers)
                            }

                            gemmaMoeEnc.endEncoding()
                            gemmaMoeCmd.commit()

                            // FIX #5: while this layer's MoE runs on the GPU, kick layer l+1's
                            // predicted expert reads (temporal prediction: this layer's active
                            // set) into the other staging buffer.
                            if prefillPipelineEnabled, l + 1 < actualLayers,
                               let packedDir = packedExpertsDir,
                               let nextFD = ExpertIOThreadPool.shared.getOrOpenLayerFD(layerIndex: l + 1, packedExpertsDir: packedDir),
                               let layoutExpertSize = loadedLayout?.expert_size, layoutExpertSize > 0,
                               !expertTokenMap.isEmpty {
                                let targetBuf = lastMoeStageIsA ? prefillStagingBufferB : prefillStagingBufferA
                                let kickSem = DispatchSemaphore(value: 0)
                                var kickMap: [Int: Int] = [:]
                                var kickTasks: [ExpertPreadTask] = []
                                let targetPtr = targetBuf.contents()
                                let predictedIds = Array(expertTokenMap.keys).sorted()
                                for (slot, expId) in predictedIds.enumerated() {
                                    kickMap[expId] = slot
                                    kickTasks.append(ExpertPreadTask(fd: nextFD, dst: targetPtr.advanced(by: slot * Int(layoutExpertSize)), offset: off_t(expId * Int(layoutExpertSize)), size: Int(layoutExpertSize)))
                                }
                                prefetchedFor[l + 1] = (buf: targetBuf, slotMap: kickMap, slotBase: predictedIds.count, sem: kickSem)
                                ExpertIOThreadPool.shared.dispatchAsync(tasks: kickTasks, done: kickSem)
                            }

                            let tGemmaMoeGpuStart = CFAbsoluteTimeGetCurrent()
                            gemmaMoeCmd.waitUntilCompleted()
                            prefillDiagGpuMs += (CFAbsoluteTimeGetCurrent() - tGemmaMoeGpuStart) * 1000.0
                            if let gErr = gemmaMoeCmd.error {
                                print("❌ [METAL ERROR] gemma4 prefill Phase B layer \(l): \(gErr)")
                                return false
                            }

                            // Swap ping-pong hidden buffers and report progress.
                            let tmpG = currHBuf
                            currHBuf = nextHBuf
                            nextHBuf = tmpG
                            passIdx += 1
                            let nowG = CFAbsoluteTimeGetCurrent()
                            if nowG - lastUIUpdateTime >= 0.1 || passIdx == totalPasses {
                                lastUIUpdateTime = nowG
                                let elapsedG = max(0.001, nowG - prefillStartTime)
                                let effG = Double(P) * (Double(passIdx) / Double(totalPasses))
                                let spdG = effG / elapsedG
                                let dispLayerIdxG = prefillLayerBase + passIdx
                                let pctG = Int((Double(dispLayerIdxG) / Double(max(1, prefillLayerTotal))) * 100)
                                let sStrG = spdG >= 10 ? String(format: "%.0f", spdG) : String(format: "%.1f", spdG)
                                let pStrG = "Ingesting prompt: Layer \(dispLayerIdxG)/\(prefillLayerTotal) (\(pctG)%) • \(sStrG) tok/s"
                                let rssG = WorkingSetManager.shared.effectiveResidentMemoryGB
                                prefillProgress?(pStrG, spdG, rssG)
                            }
                            continue
                        }

                        // Phase A: RMSNorm1/HC Blend + Attention/GDN + Residual1/HC Inject + RMSNorm2/HC Blend + Router (batched across all P tokens in ONE encoder)
                        guard let activeCmd = commandQueue.makeCommandBuffer(),
                              let layerEnc1 = activeCmd.makeComputeCommandEncoder() else {
                            print("❌ [METAL] prefill: Phase A command buffer allocation failed (chunk startPos=\(startPos), P=\(P))")
                            return false
                        }

                        // 1. RMSNorm 1 OR Pre-Attention Hyper-Connection Mixing (dispatched across all P tokens)
                        if let hcDown = layer.attnHcDownWeight, let hcDownRaw = buffers[hcDown.shardIndex],
                           let hcUp = layer.attnHcUpWeight, let hcUpRaw = buffers[hcUp.shardIndex],
                           let hcNorm = layer.attnHcNorm, let hcNormRaw = buffers[hcNorm.shardIndex],
                           let normPipe = hcNormPipeline, let downPipe = hcDownProjPipeline, let upPipe = hcUpBlendPipeline {
                            var nOff = hcNorm.offsetStart
                            var dOff = hcDown.offsetStart
                            var uOff = hcUp.offsetStart
                            var totDim: UInt32 = 4 * hiddenDim
                            var rank: UInt32 = UInt32(modelConfig?.effectiveHcLowrank ?? 320)
                            var epsVal = eps

                            // 1a. Group RMSNorm on 4 Streams
                            layerEnc1.setComputePipelineState(normPipe)
                            layerEnc1.setBuffer(hcStreamsBuffer_all, offset: 0, index: 0)
                            layerEnc1.setBuffer(hcNormRaw, offset: 0, index: 1)
                            layerEnc1.setBuffer(hcNormedBuffer_all, offset: 0, index: 2)
                            layerEnc1.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: 4, height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 1b. Down-Projection
                            layerEnc1.setComputePipelineState(downPipe)
                            layerEnc1.setBuffer(hcNormedBuffer_all, offset: 0, index: 0)
                            layerEnc1.setBuffer(hcDownRaw, offset: 0, index: 1)
                            layerEnc1.setBuffer(hcBottleneckBuffer_all, offset: 0, index: 2)
                            layerEnc1.setBytes(&dOff, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc1.setBytes(&totDim, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc1.setBytes(&rank, length: MemoryLayout<UInt32>.stride, index: 5)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: Int(rank), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 1c. Up-Projection and Stream Blend
                            layerEnc1.setComputePipelineState(upPipe)
                            layerEnc1.setBuffer(hcNormedBuffer_all, offset: 0, index: 0)
                            layerEnc1.setBuffer(hcBottleneckBuffer_all, offset: 0, index: 1)
                            layerEnc1.setBuffer(hcUpRaw, offset: 0, index: 2)
                            layerEnc1.setBuffer(xNorm1Buffer_all, offset: 0, index: 3)
                            layerEnc1.setBytes(&uOff, length: MemoryLayout<UInt64>.stride, index: 4)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 5)
                            layerEnc1.setBytes(&rank, length: MemoryLayout<UInt32>.stride, index: 6)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 1d. Compute Injection Scale if attnHcInject exists
                            if let attnHcInj = layer.attnHcInjectWeight, let attnHcInjRaw = buffers[attnHcInj.shardIndex],
                               let injScalePipe = hcInjectScalePipeline {
                                var iOff = attnHcInj.offsetStart
                                layerEnc1.setComputePipelineState(injScalePipe)
                                layerEnc1.setBuffer(hcNormedBuffer_all, offset: 0, index: 0)
                                layerEnc1.setBuffer(attnHcInjRaw, offset: 0, index: 1)
                                layerEnc1.setBuffer(hcInjectScaleBuffer_all, offset: 0, index: 2)
                                layerEnc1.setBytes(&iOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                layerEnc1.setBytes(&totDim, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.dispatchThreadgroups(MTLSize(width: 4, height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }
                        } else if let norm1 = layer.norm1Tensor, let norm1Raw = buffers[norm1.shardIndex] {
                            var gammaOff = norm1.offsetStart
                            var epsVal = eps
                            let isNorm1F16 = (norm1.dtype.contains("F16") || norm1.dtype.contains("HALF") || norm1.dtype.contains("FLOAT16")) && !norm1.dtype.contains("BF16") && !norm1.dtype.contains("BFLOAT")
                            let norm1Pipe: MTLComputePipelineState
                            if isRMSNormOffset {
                                norm1Pipe = (isNorm1F16 && rmsnormOffsetF16Pipeline != nil) ? rmsnormOffsetF16Pipeline! : (rmsnormOffsetPipeline ?? rmsnormPipeline)
                            } else {
                                norm1Pipe = (isNorm1F16 && rmsnormF16Pipeline != nil) ? rmsnormF16Pipeline! : rmsnormPipeline
                            }
                            layerEnc1.setComputePipelineState(norm1Pipe)
                            layerEnc1.setBuffer(currHBuf, offset: 0, index: 0)
                            layerEnc1.setBuffer(norm1Raw, offset: 0, index: 1)
                            layerEnc1.setBuffer(xNorm1Buffer_all, offset: 0, index: 2)
                            layerEnc1.setBytes(&gammaOff, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                            layerEnc1.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: P, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)
                        }

                        // 2. Attention / GDN Computation
                        if layer.attentionType == .fullAttention {
                            let currentQDim = isStandardGqa ? (numHeads * headDim) : (numHeads * headDim * 2)
                            let currentKvDim = kvStride

                            if let fusedQKV = layer.fusedQKVTensor {
                                // Spark 2.5: single fused q_k_v_proj [qDim + 2*kvDim, hiddenDim]
                                dispatchLinear(enc: layerEnc1, weight: fusedQKV, scale: nil, bias: nil, inBuf: xNorm1Buffer_all, outBuf: qGateBuffer_all, inDim: hiddenDim, outDim: currentQDim, batchSize: P)
                                dispatchLinear(enc: layerEnc1, weight: fusedQKV, scale: nil, bias: nil, inBuf: xNorm1Buffer_all, outBuf: kVectorBuffer_all, inDim: hiddenDim, outDim: currentKvDim, batchSize: P, weightOffsetAdd: UInt64(currentQDim) * UInt64(hiddenDim) * 2)
                                dispatchLinear(enc: layerEnc1, weight: fusedQKV, scale: nil, bias: nil, inBuf: xNorm1Buffer_all, outBuf: vVectorBuffer_all, inDim: hiddenDim, outDim: currentKvDim, batchSize: P, weightOffsetAdd: (UInt64(currentQDim) + UInt64(currentKvDim)) * UInt64(hiddenDim) * 2)
                                if let gateProj = layer.attnGateProjTensor {
                                    dispatchLinear(enc: layerEnc1, weight: gateProj, scale: nil, bias: nil, inBuf: xNorm1Buffer_all, outBuf: bVectorBuffer_all, inDim: hiddenDim, outDim: numHeads, batchSize: P)
                                }
                            } else {
                                dispatchLinear(enc: layerEnc1, weight: layer.qProjTensor, scale: layer.qScaleTensor, bias: layer.qBiasTensor, inBuf: xNorm1Buffer_all, outBuf: qGateBuffer_all, inDim: hiddenDim, outDim: currentQDim, batchSize: P)
                                dispatchLinear(enc: layerEnc1, weight: layer.kProjTensor, scale: layer.kScaleTensor, bias: layer.kBiasTensor, inBuf: xNorm1Buffer_all, outBuf: kVectorBuffer_all, inDim: hiddenDim, outDim: currentKvDim, batchSize: P)
                                dispatchLinear(enc: layerEnc1, weight: layer.vProjTensor, scale: layer.vScaleTensor, bias: layer.vBiasTensor, inBuf: xNorm1Buffer_all, outBuf: vVectorBuffer_all, inDim: hiddenDim, outDim: currentKvDim, batchSize: P)
                            }
                            layerEnc1.memoryBarrier(scope: .buffers)

                            if let qNorm = layer.qNormTensor, let qNormRaw = buffers[qNorm.shardIndex] {
                                let isQNormF16 = (qNorm.dtype.contains("F16") || qNorm.dtype.contains("HALF") || qNorm.dtype.contains("FLOAT16")) && !qNorm.dtype.contains("BF16") && !qNorm.dtype.contains("BFLOAT")
                                let qHeadNormPipe: MTLComputePipelineState?
                                if isRMSNormOffset {
                                    qHeadNormPipe = (isQNormF16 && headRmsnormOffsetF16Pipeline != nil) ? headRmsnormOffsetF16Pipeline : (headRmsnormOffsetPipeline ?? headRmsnormPipeline)
                                } else {
                                    qHeadNormPipe = (isQNormF16 && headRmsnormF16Pipeline != nil) ? headRmsnormF16Pipeline : headRmsnormPipeline
                                }
                                if let headNormPipe = qHeadNormPipe {
                                    var qNormOff = qNorm.offsetStart
                                    var nQ = numHeads
                                    var hD = headDim
                                    var hStride: UInt32 = isStandardGqa ? headDim : (headDim * 2)
                                    var epsVal = eps
                                    layerEnc1.setComputePipelineState(headNormPipe)
                                    layerEnc1.setBuffer(qGateBuffer_all, offset: 0, index: 0)
                                    layerEnc1.setBuffer(qNormRaw, offset: 0, index: 1)
                                    layerEnc1.setBytes(&qNormOff, length: MemoryLayout<UInt64>.stride, index: 2)
                                    layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 3)
                                    layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                                    layerEnc1.setBytes(&hStride, length: MemoryLayout<UInt32>.stride, index: 5)
                                    layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 6)
                                    layerEnc1.dispatchThreadgroups(MTLSize(width: Int(numHeads), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                }
                            }

                            if let kNorm = layer.kNormTensor, let kNormRaw = buffers[kNorm.shardIndex] {
                                let isKNormF16 = (kNorm.dtype.contains("F16") || kNorm.dtype.contains("HALF") || kNorm.dtype.contains("FLOAT16")) && !kNorm.dtype.contains("BF16") && !kNorm.dtype.contains("BFLOAT")
                                let kHeadNormPipe: MTLComputePipelineState?
                                if isRMSNormOffset {
                                    kHeadNormPipe = (isKNormF16 && headRmsnormOffsetF16Pipeline != nil) ? headRmsnormOffsetF16Pipeline : (headRmsnormOffsetPipeline ?? headRmsnormPipeline)
                                } else {
                                    kHeadNormPipe = (isKNormF16 && headRmsnormF16Pipeline != nil) ? headRmsnormF16Pipeline : headRmsnormPipeline
                                }
                                if let headNormPipe = kHeadNormPipe {
                                    var kNormOff = kNorm.offsetStart
                                    var nK = numKvHeads
                                    var hD = headDim
                                    var hStride = headDim
                                    var epsVal = eps
                                    layerEnc1.setComputePipelineState(headNormPipe)
                                    layerEnc1.setBuffer(kVectorBuffer_all, offset: 0, index: 0)
                                    layerEnc1.setBuffer(kNormRaw, offset: 0, index: 1)
                                    layerEnc1.setBytes(&kNormOff, length: MemoryLayout<UInt64>.stride, index: 2)
                                    layerEnc1.setBytes(&nK, length: MemoryLayout<UInt32>.stride, index: 3)
                                    layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                                    layerEnc1.setBytes(&hStride, length: MemoryLayout<UInt32>.stride, index: 5)
                                    layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 6)
                                    layerEnc1.dispatchThreadgroups(MTLSize(width: Int(numKvHeads), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                }
                            }
                            layerEnc1.memoryBarrier(scope: .buffers)

                            if let ropePipe = ropePipeline {
                                var pos: UInt32 = startPos
                                var nQ = numHeads
                                var nK = numKvHeads
                                var hD = headDim
                                var rD = UInt32(modelConfig?.effectiveRotaryDim(layerIndex: l, headDim: Int(headDim)) ?? Int(rotaryDim))
                                var qStr: UInt32 = isStandardGqa ? headDim : (headDim * 2)
                                var kStr = headDim
                                var theta = modelConfig?.effectiveRopeTheta(layerIndex: l) ?? thetaVal

                                layerEnc1.setComputePipelineState(ropePipe)
                                layerEnc1.setBuffer(qGateBuffer_all, offset: 0, index: 0)
                                layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 1)
                                layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 2)
                                layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc1.setBytes(&rD, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&qStr, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc1.setBytes(&theta, length: MemoryLayout<Float>.stride, index: 6)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), ropePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                                layerEnc1.setBuffer(kVectorBuffer_all, offset: 0, index: 0)
                                layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 1)
                                layerEnc1.setBytes(&nK, length: MemoryLayout<UInt32>.stride, index: 2)
                                layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc1.setBytes(&rD, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&kStr, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc1.setBytes(&theta, length: MemoryLayout<Float>.stride, index: 6)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(numKvHeads), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numKvHeads), ropePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            }
                            layerEnc1.memoryBarrier(scope: .buffers)

                            if let kCache = KVCacheManager.shared.kCacheBuffer,
                               let vCache = KVCacheManager.shared.vCacheBuffer {
                                let slot = (loopIdx * actualLayers) + layer.fullAttnIndex
                                let maxSeq = KVCacheManager.shared.allocatedSeqLen
                                let prec = KVCacheManager.shared.activePrecision
                                let layerByteOffset = slot * maxSeq * Int(kvStride) * prec.bytesPerElement
                                var pos: UInt32 = startPos
                                var nKv = numKvHeads
                                var hD = headDim
                                var nQ = numHeads
                                var seqLen: UInt32 = (startPos > 0) ? (0x80000000 | startPos) : 0

                                switch prec {
                                case .fp16:
                                    // Slice the O(P^2) attention into short dispatches so no single
                                    // kernel starves WindowServer (Impacting Interactivity aborts).
                                    var sliceStart16 = 0
                                    while sliceStart16 < P {
                                        let sliceRows = min(prefillAttnChunkRows, P - sliceStart16)
                                        let kVecOff = sliceStart16 * Int(kvStride) * MemoryLayout<Float>.stride
                                        let qOff = sliceStart16 * Int(gqaQDim) * MemoryLayout<Float>.stride
                                        let ctxOff = sliceStart16 * Int(numHeads) * Int(headDim) * MemoryLayout<Float>.stride
                                        let gateOff = sliceStart16 * Int(numHeads) * MemoryLayout<Float>.stride
                                        let sliceBase = startPos + UInt32(sliceStart16)
                                        var slicePos: UInt32 = sliceBase
                                        var sliceSeq: UInt32 = (sliceBase > 0) ? (0x80000000 | sliceBase) : 0
                                        if let storePipe = storeKvCacheF16Pipeline ?? storeKvCachePipeline {
                                            layerEnc1.setComputePipelineState(storePipe)
                                            layerEnc1.setBuffer(kVectorBuffer_all, offset: kVecOff, index: 0)
                                            layerEnc1.setBuffer(vVectorBuffer_all, offset: kVecOff, index: 1)
                                            layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 2)
                                            layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 3)
                                            layerEnc1.setBytes(&slicePos, length: MemoryLayout<UInt32>.stride, index: 4)
                                            layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 5)
                                            layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 6)
                                            layerEnc1.dispatchThreads(MTLSize(width: Int(kvStride), height: sliceRows, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(kvStride), storePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                        }

                                        layerEnc1.memoryBarrier(scope: .buffers)

                                        if layer.attnGateProjTensor != nil, let headGatePipe = gqaHeadGateF16Pipeline ?? gqaHeadGatePipeline {
                                            // Spark 2.5: per-head sigmoid output gate + optional sliding window
                                            var windowSize: UInt32 = layer.isSlidingAttention ? UInt32(modelConfig?.effectiveSlidingWindow ?? 0) : 0
                                            layerEnc1.setComputePipelineState(headGatePipe)
                                            layerEnc1.setBuffer(qGateBuffer_all, offset: qOff, index: 0)
                                            layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                            layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                            layerEnc1.setBuffer(attnCtxBuffer_all, offset: ctxOff, index: 3)
                                            layerEnc1.setBuffer(bVectorBuffer_all, offset: gateOff, index: 4)
                                            layerEnc1.setBytes(&sliceSeq, length: MemoryLayout<UInt32>.stride, index: 5)
                                            layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 6)
                                            layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 7)
                                            layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 8)
                                            layerEnc1.setBytes(&windowSize, length: MemoryLayout<UInt32>.stride, index: 9)
                                            layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: sliceRows, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), headGatePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                        } else if isStandardGqa, let gqaStdPipe = gqaStandardF16Pipeline ?? gqaStandardPipeline {
                                            layerEnc1.setComputePipelineState(gqaStdPipe)
                                            layerEnc1.setBuffer(qGateBuffer_all, offset: qOff, index: 0)
                                            layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                            layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                            layerEnc1.setBuffer(attnCtxBuffer_all, offset: ctxOff, index: 3)
                                            layerEnc1.setBytes(&sliceSeq, length: MemoryLayout<UInt32>.stride, index: 4)
                                            layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 5)
                                            layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 6)
                                            layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 7)
                                            layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: sliceRows, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), gqaStdPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                        } else if let gqaPipe = gqaDecodeF16Pipeline ?? gqaDecodePipeline {
                                            // FIX #6a NOTE: multi-row chunked kernels verified SLOWER
                                            // than this path at prefill shapes (T17c) — see
                                            // Benchmarks/PERF_FINDINGS.md. Reverted to the old kernel
                                            // until a SIMD-tiled variant exists.
                                            layerEnc1.setComputePipelineState(gqaPipe)
                                            layerEnc1.setBuffer(qGateBuffer_all, offset: qOff, index: 0)
                                            layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                            layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                            layerEnc1.setBuffer(attnCtxBuffer_all, offset: ctxOff, index: 3)
                                            layerEnc1.setBytes(&sliceSeq, length: MemoryLayout<UInt32>.stride, index: 4)
                                            layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 5)
                                            layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 6)
                                            layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 7)
                                            layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: sliceRows, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), gqaPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                        }
                                        sliceStart16 += sliceRows
                                    }

                                case .fp8:
                                    let scaleByteOffset = slot * maxSeq * Int(numKvHeads) * MemoryLayout<UInt16>.stride
                                    var sliceStart8 = 0
                                    while sliceStart8 < P {
                                        let sliceRows = min(prefillAttnChunkRows, P - sliceStart8)
                                        let kVecOff = sliceStart8 * Int(kvStride) * MemoryLayout<Float>.stride
                                        let qOff = sliceStart8 * Int(gqaQDim) * MemoryLayout<Float>.stride
                                        let ctxOff = sliceStart8 * Int(numHeads) * Int(headDim) * MemoryLayout<Float>.stride
                                        let gateOff = sliceStart8 * Int(numHeads) * MemoryLayout<Float>.stride
                                        let sliceBase = startPos + UInt32(sliceStart8)
                                        var slicePos: UInt32 = sliceBase
                                        var sliceSeq: UInt32 = (sliceBase > 0) ? (0x80000000 | sliceBase) : 0
                                        if let storePipe = storeKvCacheFP8Pipeline,
                                           let kScale = KVCacheManager.shared.kScaleBuffer,
                                           let vScale = KVCacheManager.shared.vScaleBuffer {
                                            layerEnc1.setComputePipelineState(storePipe)
                                            layerEnc1.setBuffer(kVectorBuffer_all, offset: kVecOff, index: 0)
                                            layerEnc1.setBuffer(vVectorBuffer_all, offset: kVecOff, index: 1)
                                            layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 2)
                                            layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 3)
                                            layerEnc1.setBuffer(kScale, offset: scaleByteOffset, index: 4)
                                            layerEnc1.setBuffer(vScale, offset: scaleByteOffset, index: 5)
                                            layerEnc1.setBytes(&slicePos, length: MemoryLayout<UInt32>.stride, index: 6)
                                            layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 7)
                                            layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 8)
                                            let threadgroups = MTLSize(width: Int(numKvHeads), height: sliceRows, depth: 1)
                                            let threadsPerTG = MTLSize(width: 32, height: 1, depth: 1)
                                            layerEnc1.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerTG)
                                        }

                                        layerEnc1.memoryBarrier(scope: .buffers)

                                        if let kScale = KVCacheManager.shared.kScaleBuffer,
                                           let vScale = KVCacheManager.shared.vScaleBuffer {
                                            if layer.attnGateProjTensor != nil, let headGatePipe = gqaHeadGateFP8Pipeline {
                                                // Spark 2.5: per-head sigmoid output gate + optional sliding window (FP8 KV)
                                                var windowSize: UInt32 = layer.isSlidingAttention ? UInt32(modelConfig?.effectiveSlidingWindow ?? 0) : 0
                                                layerEnc1.setComputePipelineState(headGatePipe)
                                                layerEnc1.setBuffer(qGateBuffer_all, offset: qOff, index: 0)
                                                layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                                layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                                layerEnc1.setBuffer(kScale, offset: scaleByteOffset, index: 3)
                                                layerEnc1.setBuffer(vScale, offset: scaleByteOffset, index: 4)
                                                layerEnc1.setBuffer(attnCtxBuffer_all, offset: ctxOff, index: 5)
                                                layerEnc1.setBuffer(bVectorBuffer_all, offset: gateOff, index: 6)
                                                layerEnc1.setBytes(&sliceSeq, length: MemoryLayout<UInt32>.stride, index: 7)
                                                layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 8)
                                                layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 9)
                                                layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 10)
                                                layerEnc1.setBytes(&windowSize, length: MemoryLayout<UInt32>.stride, index: 11)
                                                layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: sliceRows, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), headGatePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                            } else if isStandardGqa, let gqaStdPipe = gqaStandardFP8Pipeline {
                                                layerEnc1.setComputePipelineState(gqaStdPipe)
                                                layerEnc1.setBuffer(qGateBuffer_all, offset: qOff, index: 0)
                                                layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                                layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                                layerEnc1.setBuffer(kScale, offset: scaleByteOffset, index: 3)
                                                layerEnc1.setBuffer(vScale, offset: scaleByteOffset, index: 4)
                                                layerEnc1.setBuffer(attnCtxBuffer_all, offset: ctxOff, index: 5)
                                                layerEnc1.setBytes(&sliceSeq, length: MemoryLayout<UInt32>.stride, index: 6)
                                                layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 7)
                                                layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 8)
                                                layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 9)
                                                layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: sliceRows, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), gqaStdPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                            } else if let gqaPipe = gqaDecodeFP8Pipeline {
                                                layerEnc1.setComputePipelineState(gqaPipe)
                                                layerEnc1.setBuffer(qGateBuffer_all, offset: qOff, index: 0)
                                                layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                                layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                                layerEnc1.setBuffer(kScale, offset: scaleByteOffset, index: 3)
                                                layerEnc1.setBuffer(vScale, offset: scaleByteOffset, index: 4)
                                                layerEnc1.setBuffer(attnCtxBuffer_all, offset: ctxOff, index: 5)
                                                layerEnc1.setBytes(&sliceSeq, length: MemoryLayout<UInt32>.stride, index: 6)
                                                layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 7)
                                                layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 8)
                                                layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 9)
                                                layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: sliceRows, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), gqaPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                            }
                                        }
                                        sliceStart8 += sliceRows
                                    }

                                case .fp32:
                                    var sliceStart32 = 0
                                    while sliceStart32 < P {
                                        let sliceRows = min(prefillAttnChunkRows, P - sliceStart32)
                                        let kVecOff = sliceStart32 * Int(kvStride) * MemoryLayout<Float>.stride
                                        let qOff = sliceStart32 * Int(gqaQDim) * MemoryLayout<Float>.stride
                                        let ctxOff = sliceStart32 * Int(numHeads) * Int(headDim) * MemoryLayout<Float>.stride
                                        let gateOff = sliceStart32 * Int(numHeads) * MemoryLayout<Float>.stride
                                        let sliceBase = startPos + UInt32(sliceStart32)
                                        var slicePos: UInt32 = sliceBase
                                        var sliceSeq: UInt32 = (sliceBase > 0) ? (0x80000000 | sliceBase) : 0
                                        if let storePipe = storeKvCachePipeline {
                                            layerEnc1.setComputePipelineState(storePipe)
                                            layerEnc1.setBuffer(kVectorBuffer_all, offset: kVecOff, index: 0)
                                            layerEnc1.setBuffer(vVectorBuffer_all, offset: kVecOff, index: 1)
                                            layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 2)
                                            layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 3)
                                            layerEnc1.setBytes(&slicePos, length: MemoryLayout<UInt32>.stride, index: 4)
                                            layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 5)
                                            layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 6)
                                            layerEnc1.dispatchThreads(MTLSize(width: Int(kvStride), height: sliceRows, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(kvStride), storePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                        }

                                        layerEnc1.memoryBarrier(scope: .buffers)

                                        if layer.attnGateProjTensor != nil, let headGatePipe = gqaHeadGatePipeline {
                                            // Spark 2.5: per-head sigmoid output gate + optional sliding window
                                            var windowSize: UInt32 = layer.isSlidingAttention ? UInt32(modelConfig?.effectiveSlidingWindow ?? 0) : 0
                                            layerEnc1.setComputePipelineState(headGatePipe)
                                            layerEnc1.setBuffer(qGateBuffer_all, offset: qOff, index: 0)
                                            layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                            layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                            layerEnc1.setBuffer(attnCtxBuffer_all, offset: ctxOff, index: 3)
                                            layerEnc1.setBuffer(bVectorBuffer_all, offset: gateOff, index: 4)
                                            layerEnc1.setBytes(&sliceSeq, length: MemoryLayout<UInt32>.stride, index: 5)
                                            layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 6)
                                            layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 7)
                                            layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 8)
                                            layerEnc1.setBytes(&windowSize, length: MemoryLayout<UInt32>.stride, index: 9)
                                            layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: sliceRows, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), headGatePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                        } else if isStandardGqa, let gqaStdPipe = gqaStandardPipeline {
                                            layerEnc1.setComputePipelineState(gqaStdPipe)
                                            layerEnc1.setBuffer(qGateBuffer_all, offset: qOff, index: 0)
                                            layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                            layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                            layerEnc1.setBuffer(attnCtxBuffer_all, offset: ctxOff, index: 3)
                                            layerEnc1.setBytes(&sliceSeq, length: MemoryLayout<UInt32>.stride, index: 4)
                                            layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 5)
                                            layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 6)
                                            layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 7)
                                            layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: sliceRows, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), gqaStdPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                        } else if let gqaPipe = gqaDecodePipeline {
                                            layerEnc1.setComputePipelineState(gqaPipe)
                                            layerEnc1.setBuffer(qGateBuffer_all, offset: qOff, index: 0)
                                            layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                            layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                            layerEnc1.setBuffer(attnCtxBuffer_all, offset: ctxOff, index: 3)
                                            layerEnc1.setBytes(&sliceSeq, length: MemoryLayout<UInt32>.stride, index: 4)
                                            layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 5)
                                            layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 6)
                                            layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 7)
                                            layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: sliceRows, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), gqaPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                        }
                                        sliceStart32 += sliceRows
                                    }
                                }
                            }

                            layerEnc1.memoryBarrier(scope: .buffers)
                            dispatchLinear(enc: layerEnc1, weight: layer.oProjTensor, scale: layer.oScaleTensor, bias: layer.oBiasTensor, inBuf: attnCtxBuffer_all, outBuf: attnOutBuffer_all, inDim: attnCtxDim, outDim: hiddenDim, batchSize: P)
                        } else {
                                // Linear Attention / GDN (Gated DeltaNet) Sequence Dispatch
                                let qkvDim: UInt32 = (linKeyHeads + linKeyHeads + linValHeads) * 128
                                let zDim: UInt32 = linValHeads * 128
                                let aDim: UInt32 = linValHeads
                                let bDim: UInt32 = linValHeads

                                dispatchLinear(enc: layerEnc1, weight: layer.inProjQKV, scale: layer.inProjQKVScale, bias: layer.inProjQKVBias, inBuf: xNorm1Buffer_all, outBuf: qGateBuffer_all, inDim: hiddenDim, outDim: qkvDim, batchSize: P)
                                layerEnc1.memoryBarrier(scope: .buffers)

                                if let conv1d = layer.conv1dTensor, let convRaw = buffers[conv1d.shardIndex],
                                   let convPipe = causalConv1dSeqPipeline ?? causalConv1dPipeline, let convState = KVCacheManager.shared.convStateBuffer {
                                    let convStateByteOffset = layer.linAttnIndex * Int(qkvDim) * 4 * MemoryLayout<Float>.stride
                                    var cOff = conv1d.offsetStart
                                    var numChannels: UInt32 = qkvDim
                                    var seqLen: UInt32 = UInt32(P)
                                    layerEnc1.setComputePipelineState(convPipe)
                                    layerEnc1.setBuffer(qGateBuffer_all, offset: 0, index: 0)
                                    layerEnc1.setBuffer(convRaw, offset: 0, index: 1)
                                    layerEnc1.setBuffer(convState, offset: convStateByteOffset, index: 2)
                                    layerEnc1.setBuffer(qGateBuffer_all, offset: 0, index: 3)
                                    layerEnc1.setBytes(&cOff, length: MemoryLayout<UInt64>.stride, index: 4)
                                    layerEnc1.setBytes(&numChannels, length: MemoryLayout<UInt32>.stride, index: 5)
                                    layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 6)
                                    layerEnc1.dispatchThreads(MTLSize(width: Int(qkvDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, convPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)
                                }

                                if let l2Pipe = l2NormQkSeqPipeline ?? l2NormQkPipeline {
                                    var numHeads: UInt32 = linKeyHeads
                                    var headDimVal: UInt32 = 128
                                    var qkvDimVal: UInt32 = qkvDim
                                    layerEnc1.setComputePipelineState(l2Pipe)
                                    layerEnc1.setBuffer(qGateBuffer_all, offset: 0, index: 0)
                                    layerEnc1.setBytes(&numHeads, length: MemoryLayout<UInt32>.stride, index: 1)
                                    layerEnc1.setBytes(&headDimVal, length: MemoryLayout<UInt32>.stride, index: 2)
                                    layerEnc1.setBytes(&qkvDimVal, length: MemoryLayout<UInt32>.stride, index: 3)
                                    layerEnc1.dispatchThreads(MTLSize(width: Int(linKeyHeads), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(linKeyHeads), l2Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)
                                }

                                dispatchLinear(enc: layerEnc1, weight: layer.inProjZ, scale: layer.inProjZScale, bias: layer.inProjZBias, inBuf: xNorm1Buffer_all, outBuf: zGateBuffer_all, inDim: hiddenDim, outDim: zDim, batchSize: P)
                                dispatchLinear(enc: layerEnc1, weight: layer.inProjA, scale: layer.inProjAScale, bias: layer.inProjABias, inBuf: xNorm1Buffer_all, outBuf: aVectorBuffer_all, inDim: hiddenDim, outDim: aDim, batchSize: P)
                                dispatchLinear(enc: layerEnc1, weight: layer.inProjB, scale: layer.inProjBScale, bias: layer.inProjBBias, inBuf: xNorm1Buffer_all, outBuf: bVectorBuffer_all, inDim: hiddenDim, outDim: bDim, batchSize: P)
                                layerEnc1.memoryBarrier(scope: .buffers)

                                let useSigmoidGate = (modelConfig?.effectiveOutputGateType.lowercased() == "sigmoid")
                                let selectedSeqPipe = useSigmoidGate
                                    ? (gdnLinearAttnSeqSigmoidPipeline ?? gdnLinearAttnStepSigmoidPipeline ?? linearAttnStepSigmoidPipeline ?? gdnLinearAttnSeqPipeline ?? gdnLinearAttnStepPipeline ?? linearAttnStepPipeline)
                                    : (gdnLinearAttnSeqPipeline ?? gdnLinearAttnStepPipeline ?? linearAttnStepPipeline)

                                if let linPipe = selectedSeqPipe,
                                   let sBuf = KVCacheManager.shared.linearStateBuffer,
                                   let aLog = layer.aLogTensor, let aLogRaw = buffers[aLog.shardIndex],
                                   let dtBias = layer.dtBiasTensor, let dtBiasRaw = buffers[dtBias.shardIndex],
                                   let linNorm = layer.linearNormTensor, let linNormRaw = buffers[linNorm.shardIndex] {
                                    let linIdx = layer.linAttnIndex
                                    let stateByteOffset = linIdx * Int(linValHeads * 128 * 128) * MemoryLayout<Float>.stride
                                    var aLogOff = aLog.offsetStart
                                    var dtBiasOff = dtBias.offsetStart
                                    var linNormOff = linNorm.offsetStart
                                    var numValHeads: UInt32 = linValHeads
                                    var numKeyHeads: UInt32 = linKeyHeads
                                    var headDim: UInt32 = 128
                                    var epsVal = eps
                                    var seqLen: UInt32 = UInt32(P)

                                    layerEnc1.setComputePipelineState(linPipe)
                                    layerEnc1.setBuffer(qGateBuffer_all, offset: 0, index: 0)
                                    layerEnc1.setBuffer(zGateBuffer_all, offset: 0, index: 1)
                                    layerEnc1.setBuffer(aVectorBuffer_all, offset: 0, index: 2)
                                    layerEnc1.setBuffer(bVectorBuffer_all, offset: 0, index: 3)
                                    layerEnc1.setBuffer(aLogRaw, offset: 0, index: 4)
                                    layerEnc1.setBuffer(dtBiasRaw, offset: 0, index: 5)
                                    layerEnc1.setBuffer(linNormRaw, offset: 0, index: 6)
                                    layerEnc1.setBuffer(sBuf, offset: stateByteOffset, index: 7)
                                    layerEnc1.setBuffer(attnCtxBuffer_all, offset: 0, index: 8)
                                    layerEnc1.setBytes(&aLogOff, length: MemoryLayout<UInt64>.stride, index: 9)
                                    layerEnc1.setBytes(&dtBiasOff, length: MemoryLayout<UInt64>.stride, index: 10)
                                    layerEnc1.setBytes(&linNormOff, length: MemoryLayout<UInt64>.stride, index: 11)
                                    layerEnc1.setBytes(&numValHeads, length: MemoryLayout<UInt32>.stride, index: 12)
                                    layerEnc1.setBytes(&numKeyHeads, length: MemoryLayout<UInt32>.stride, index: 13)
                                    layerEnc1.setBytes(&headDim, length: MemoryLayout<UInt32>.stride, index: 14)
                                    layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 15)
                                    layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 16)
                                    layerEnc1.dispatchThreadgroups(MTLSize(width: Int(numValHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)
                                }

                                dispatchLinear(enc: layerEnc1, weight: layer.linearOutProjTensor ?? layer.oProjTensor, scale: layer.linearOutProjScale ?? layer.oScaleTensor, bias: layer.linearOutProjBias ?? layer.oBiasTensor, inBuf: attnCtxBuffer_all, outBuf: attnOutBuffer_all, inDim: zDim, outDim: hiddenDim, batchSize: P)
                            }
                            layerEnc1.memoryBarrier(scope: .buffers)

                        // 3. Post-Attention HyperConnection Injection / Residual 1
                        if (layer.attnHcInjectWeight != nil || layer.attnHcDownWeight != nil),
                           let injPipe = hcInjectPipeline {
                            layerEnc1.setComputePipelineState(injPipe)
                            layerEnc1.setBuffer(hcStreamsBuffer_all, offset: 0, index: 0)
                            layerEnc1.setBuffer(attnOutBuffer_all, offset: 0, index: 1)
                            layerEnc1.setBuffer(hcInjectScaleBuffer_all, offset: 0, index: 2)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                            layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), injPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                        } else {
                            layerEnc1.setComputePipelineState(addPipeline)
                            layerEnc1.setBuffer(currHBuf, offset: 0, index: 0)
                            layerEnc1.setBuffer(attnOutBuffer_all, offset: 0, index: 1)
                            layerEnc1.setBuffer(hMidBuffer_all, offset: 0, index: 2)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                            layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                        }
                        layerEnc1.memoryBarrier(scope: .buffers)

                        // 4. Pre-MLP HyperConnection Mixing / RMSNorm 2
                        if let hcDown = layer.mlpHcDownWeight, let hcDownRaw = buffers[hcDown.shardIndex],
                           let hcUp = layer.mlpHcUpWeight, let hcUpRaw = buffers[hcUp.shardIndex],
                           let hcNorm = layer.mlpHcNorm, let hcNormRaw = buffers[hcNorm.shardIndex],
                           let normPipe = hcNormPipeline, let downPipe = hcDownProjPipeline, let upPipe = hcUpBlendPipeline {
                            var nOff = hcNorm.offsetStart
                            var dOff = hcDown.offsetStart
                            var uOff = hcUp.offsetStart
                            var totDim: UInt32 = 4 * hiddenDim
                            var rank: UInt32 = UInt32(modelConfig?.effectiveHcLowrank ?? 320)
                            var epsVal = eps

                            // 4a. Group RMSNorm on 4 Streams
                            layerEnc1.setComputePipelineState(normPipe)
                            layerEnc1.setBuffer(hcStreamsBuffer_all, offset: 0, index: 0)
                            layerEnc1.setBuffer(hcNormRaw, offset: 0, index: 1)
                            layerEnc1.setBuffer(hcNormedBuffer_all, offset: 0, index: 2)
                            layerEnc1.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: 4, height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 4b. Down-Projection
                            layerEnc1.setComputePipelineState(downPipe)
                            layerEnc1.setBuffer(hcNormedBuffer_all, offset: 0, index: 0)
                            layerEnc1.setBuffer(hcDownRaw, offset: 0, index: 1)
                            layerEnc1.setBuffer(hcBottleneckBuffer_all, offset: 0, index: 2)
                            layerEnc1.setBytes(&dOff, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc1.setBytes(&totDim, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc1.setBytes(&rank, length: MemoryLayout<UInt32>.stride, index: 5)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: Int(rank), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 4c. Up-Projection and Stream Blend
                            layerEnc1.setComputePipelineState(upPipe)
                            layerEnc1.setBuffer(hcNormedBuffer_all, offset: 0, index: 0)
                            layerEnc1.setBuffer(hcBottleneckBuffer_all, offset: 0, index: 1)
                            layerEnc1.setBuffer(hcUpRaw, offset: 0, index: 2)
                            layerEnc1.setBuffer(xNorm2Buffer_all, offset: 0, index: 3)
                            layerEnc1.setBytes(&uOff, length: MemoryLayout<UInt64>.stride, index: 4)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 5)
                            layerEnc1.setBytes(&rank, length: MemoryLayout<UInt32>.stride, index: 6)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 4d. Compute Injection Scale if mlpHcInject exists
                            if let mlpHcInj = layer.mlpHcInjectWeight, let mlpHcInjRaw = buffers[mlpHcInj.shardIndex],
                               let injScalePipe = hcInjectScalePipeline {
                                var iOff = mlpHcInj.offsetStart
                                layerEnc1.setComputePipelineState(injScalePipe)
                                layerEnc1.setBuffer(hcNormedBuffer_all, offset: 0, index: 0)
                                layerEnc1.setBuffer(mlpHcInjRaw, offset: 0, index: 1)
                                layerEnc1.setBuffer(hcInjectScaleBuffer_all, offset: 0, index: 2)
                                layerEnc1.setBytes(&iOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                layerEnc1.setBytes(&totDim, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.dispatchThreadgroups(MTLSize(width: 4, height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }
                        } else if let norm2 = layer.norm2Tensor, let norm2Raw = buffers[norm2.shardIndex] {
                            var gammaOff = norm2.offsetStart
                            var epsVal = eps
                            let isNorm2F16 = (norm2.dtype.contains("F16") || norm2.dtype.contains("HALF") || norm2.dtype.contains("FLOAT16")) && !norm2.dtype.contains("BF16") && !norm2.dtype.contains("BFLOAT")
                            let norm2Pipe: MTLComputePipelineState
                            if isRMSNormOffset {
                                norm2Pipe = (isNorm2F16 && rmsnormOffsetF16Pipeline != nil) ? rmsnormOffsetF16Pipeline! : (rmsnormOffsetPipeline ?? rmsnormPipeline)
                            } else {
                                norm2Pipe = (isNorm2F16 && rmsnormF16Pipeline != nil) ? rmsnormF16Pipeline! : rmsnormPipeline
                            }
                            layerEnc1.setComputePipelineState(norm2Pipe)
                            layerEnc1.setBuffer(hMidBuffer_all, offset: 0, index: 0)
                            layerEnc1.setBuffer(norm2Raw, offset: 0, index: 1)
                            layerEnc1.setBuffer(xNorm2Buffer_all, offset: 0, index: 2)
                            layerEnc1.setBytes(&gammaOff, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                            layerEnc1.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: P, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)
                        }

                        // 5. Dense MLP or MoE Routing
                        if layer.mlpType == .denseMlp || layer.routerTensor == nil {
                            layerEnc1.setComputePipelineState(clearPipeline)
                            layerEnc1.setBuffer(hMlpBuffer_all, offset: 0, index: 0)
                            layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            if let gateW = layer.denseGateWeight,
                               let upW = layer.denseUpWeight,
                               let downW = layer.denseDownWeight,
                               let gRaw = buffers[gateW.shardIndex],
                               let uRaw = buffers[upW.shardIndex],
                               let dRaw = buffers[downW.shardIndex] {
                                var gWOff = gateW.offsetStart
                                var uWOff = upW.offsetStart
                                var dWOff = downW.offsetStart
                                var hDimVal: UInt32 = UInt32(hiddenDim)
                                var interDimVal: UInt32 = UInt32(intermediateDim)

                                let gateS = layer.denseGateScale
                                let upS = layer.denseUpScale
                                let downS = layer.denseDownScale
                                let isBlockScale = (gateS != nil) && (gateS!.name.contains("scale_inv") || ((gateS!.offsetEnd - gateS!.offsetStart) < UInt64(intermediateDim * 2)))
                                let isQuantizedAffine = (layer.denseGateBias != nil || gateW.dtype.contains("Q4") || gateW.dtype.contains("Q8") || (gateS != nil && !isBlockScale))

                                if modelConfig?.isSparkModel == true,
                                   let geluBatched = bf16GeluGateUpBatchedPipeline,
                                   let downBatched = bf16DownBatchedPipeline {
                                    // Spark 2.5: GeGLU (exact GeLU gate * up), batched across all P tokens
                                    layerEnc1.setComputePipelineState(geluBatched)
                                    layerEnc1.setBuffer(gRaw, offset: 0, index: 0)
                                    layerEnc1.setBuffer(uRaw, offset: 0, index: 1)
                                    layerEnc1.setBuffer(xNorm2Buffer_all, offset: 0, index: 2)
                                    layerEnc1.setBuffer(interBuffer_all, offset: 0, index: 3)
                                    layerEnc1.setBytes(&gWOff, length: 8, index: 4)
                                    layerEnc1.setBytes(&uWOff, length: 8, index: 5)
                                    layerEnc1.setBytes(&hDimVal, length: 4, index: 6)
                                    layerEnc1.setBytes(&interDimVal, length: 4, index: 7)
                                    layerEnc1.setBuffer(denseActiveTokensBuffer, offset: 0, index: 8)
                                    layerEnc1.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)

                                    layerEnc1.setComputePipelineState(downBatched)
                                    layerEnc1.setBuffer(dRaw, offset: 0, index: 0)
                                    layerEnc1.setBuffer(interBuffer_all, offset: 0, index: 1)
                                    layerEnc1.setBuffer(hMlpBuffer_all, offset: 0, index: 2)
                                    layerEnc1.setBytes(&dWOff, length: 8, index: 3)
                                    layerEnc1.setBytes(&interDimVal, length: 4, index: 4)
                                    layerEnc1.setBytes(&hDimVal, length: 4, index: 5)
                                    var pkVal: Float = 1.0
                                    layerEnc1.setBytes(&pkVal, length: 4, index: 6)
                                    layerEnc1.setBuffer(denseActiveTokensBuffer, offset: 0, index: 7)
                                    layerEnc1.setBuffer(denseActiveWeightsBuffer, offset: 0, index: 8)
                                    layerEnc1.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)
                                } else if isBlockScale,
                                   let gateBatched = fp8BlockGateUpBatchedPipeline ?? fp8BlockGateUpSimdPipeline ?? fp8GateUpBatchedPipeline,
                                   let downBatched = fp8BlockDownBatchedPipeline ?? fp8BlockDownSimdPipeline ?? fp8DownBatchedPipeline,
                                   let gsRaw = gateS != nil ? buffers[gateS!.shardIndex] : nil,
                                   let usRaw = upS != nil ? buffers[upS!.shardIndex] : nil,
                                   let dsRaw = downS != nil ? buffers[downS!.shardIndex] : nil {
                                    var gSOff = gateS?.offsetStart ?? 0
                                    var uSOff = upS?.offsetStart ?? 0
                                    var dSOff = downS?.offsetStart ?? 0

                                    layerEnc1.setComputePipelineState(gateBatched)
                                    layerEnc1.setBuffer(gRaw, offset: 0, index: 0)
                                    layerEnc1.setBuffer(uRaw, offset: 0, index: 1)
                                    layerEnc1.setBuffer(xNorm2Buffer_all, offset: 0, index: 2)
                                    layerEnc1.setBuffer(interBuffer_all, offset: 0, index: 3)
                                    layerEnc1.setBuffer(gsRaw, offset: 0, index: 4)
                                    layerEnc1.setBuffer(usRaw, offset: 0, index: 5)
                                    layerEnc1.setBytes(&gWOff, length: 8, index: 6)
                                    layerEnc1.setBytes(&gSOff, length: 8, index: 7)
                                    layerEnc1.setBytes(&uWOff, length: 8, index: 8)
                                    layerEnc1.setBytes(&uSOff, length: 8, index: 9)
                                    layerEnc1.setBytes(&hDimVal, length: 4, index: 10)
                                    layerEnc1.setBytes(&interDimVal, length: 4, index: 11)
                                    layerEnc1.setBuffer(denseActiveTokensBuffer, offset: 0, index: 12)
                                    layerEnc1.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)

                                    layerEnc1.setComputePipelineState(downBatched)
                                    layerEnc1.setBuffer(dRaw, offset: 0, index: 0)
                                    layerEnc1.setBuffer(interBuffer_all, offset: 0, index: 1)
                                    layerEnc1.setBuffer(hMlpBuffer_all, offset: 0, index: 2)
                                    layerEnc1.setBuffer(dsRaw, offset: 0, index: 3)
                                    layerEnc1.setBytes(&dWOff, length: 8, index: 4)
                                    layerEnc1.setBytes(&dSOff, length: 8, index: 5)
                                    layerEnc1.setBytes(&interDimVal, length: 4, index: 6)
                                    layerEnc1.setBytes(&hDimVal, length: 4, index: 7)
                                    var pkVal: Float = 1.0
                                    layerEnc1.setBytes(&pkVal, length: 4, index: 8)
                                    layerEnc1.setBuffer(denseActiveTokensBuffer, offset: 0, index: 9)
                                    layerEnc1.setBuffer(denseActiveWeightsBuffer, offset: 0, index: 10)
                                    layerEnc1.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)
                                } else if !isQuantizedAffine,
                                           let gateBatched = bf16GateUpBatchedPipeline ?? bf16GateUpSimdPipeline,
                                           let downBatched = bf16DownBatchedPipeline ?? bf16DownSimdPipeline {
                                    layerEnc1.setComputePipelineState(gateBatched)
                                    layerEnc1.setBuffer(gRaw, offset: 0, index: 0)
                                    layerEnc1.setBuffer(uRaw, offset: 0, index: 1)
                                    layerEnc1.setBuffer(xNorm2Buffer_all, offset: 0, index: 2)
                                    layerEnc1.setBuffer(interBuffer_all, offset: 0, index: 3)
                                    layerEnc1.setBytes(&gWOff, length: 8, index: 4)
                                    layerEnc1.setBytes(&uWOff, length: 8, index: 5)
                                    layerEnc1.setBytes(&hDimVal, length: 4, index: 6)
                                    layerEnc1.setBytes(&interDimVal, length: 4, index: 7)
                                    layerEnc1.setBuffer(denseActiveTokensBuffer, offset: 0, index: 8)
                                    layerEnc1.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)

                                    layerEnc1.setComputePipelineState(downBatched)
                                    layerEnc1.setBuffer(dRaw, offset: 0, index: 0)
                                    layerEnc1.setBuffer(interBuffer_all, offset: 0, index: 1)
                                    layerEnc1.setBuffer(hMlpBuffer_all, offset: 0, index: 2)
                                    layerEnc1.setBytes(&dWOff, length: 8, index: 3)
                                    layerEnc1.setBytes(&interDimVal, length: 4, index: 4)
                                    layerEnc1.setBytes(&hDimVal, length: 4, index: 5)
                                    var pkVal: Float = 1.0
                                    layerEnc1.setBytes(&pkVal, length: 4, index: 6)
                                    layerEnc1.setBuffer(denseActiveTokensBuffer, offset: 0, index: 7)
                                    layerEnc1.setBuffer(denseActiveWeightsBuffer, offset: 0, index: 8)
                                    layerEnc1.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)
                                } else {
                                    for p in 0..<P {
                                        let tokenOffset = p * Int(hiddenDim) * MemoryLayout<Float>.stride
                                        let interTokenOffset = p * Int(intermediateDim) * MemoryLayout<Float>.stride

                                        if modelConfig?.isSparkModel == true,
                                           let geluPipe = bf16GeluGateUpSimdPipeline ?? bf16GeluGateUpPipeline,
                                           let downPipe = bf16DownSimdPipeline ?? bf16DownPipeline {
                                            // Spark 2.5: GeGLU per-token fallback
                                            layerEnc1.setComputePipelineState(geluPipe)
                                            layerEnc1.setBuffer(gRaw, offset: 0, index: 0)
                                            layerEnc1.setBuffer(uRaw, offset: 0, index: 1)
                                            layerEnc1.setBuffer(xNorm2Buffer_all, offset: tokenOffset, index: 2)
                                            layerEnc1.setBuffer(interBuffer_all, offset: interTokenOffset, index: 3)
                                            layerEnc1.setBytes(&gWOff, length: 8, index: 4)
                                            layerEnc1.setBytes(&uWOff, length: 8, index: 5)
                                            layerEnc1.setBytes(&hDimVal, length: 4, index: 6)
                                            layerEnc1.setBytes(&interDimVal, length: 4, index: 7)
                                            if bf16GeluGateUpSimdPipeline != nil {
                                                layerEnc1.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                            } else {
                                                layerEnc1.dispatchThreads(MTLSize(width: Int(intermediateDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(intermediateDim), geluPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                            }
                                            layerEnc1.memoryBarrier(scope: .buffers)

                                            layerEnc1.setComputePipelineState(downPipe)
                                            layerEnc1.setBuffer(dRaw, offset: 0, index: 0)
                                            layerEnc1.setBuffer(interBuffer_all, offset: interTokenOffset, index: 1)
                                            layerEnc1.setBuffer(hMlpBuffer_all, offset: tokenOffset, index: 2)
                                            layerEnc1.setBytes(&dWOff, length: 8, index: 3)
                                            layerEnc1.setBytes(&interDimVal, length: 4, index: 4)
                                            layerEnc1.setBytes(&hDimVal, length: 4, index: 5)
                                            var pkVal: Float = 1.0
                                            layerEnc1.setBytes(&pkVal, length: 4, index: 6)
                                            layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), downPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                            layerEnc1.memoryBarrier(scope: .buffers)
                                            continue
                                        }

                                        dispatchExpertMlp(
                                            enc: layerEnc1,
                                            gateW: gateW,
                                            gateS: gateS,
                                            gateB: layer.denseGateBias,
                                            upW: upW,
                                            upS: upS,
                                            upB: layer.denseUpBias,
                                            downW: downW,
                                            downS: downS,
                                            downB: layer.denseDownBias,
                                            inBuf: xNorm2Buffer_all,
                                            interBuf: interBuffer_all,
                                            accumBuf: hMlpBuffer_all,
                                            inDim: hiddenDim,
                                            interDim: intermediateDim,
                                            routingWeight: 1.0,
                                            inOffset: tokenOffset,
                                            interOffset: interTokenOffset,
                                            accumOffset: tokenOffset
                                        )
                                    }
                                    layerEnc1.memoryBarrier(scope: .buffers)
                                }
                            }

                            if (layer.mlpHcInjectWeight != nil || layer.mlpHcDownWeight != nil),
                               let injPipe = hcInjectPipeline {
                                layerEnc1.setComputePipelineState(injPipe)
                                layerEnc1.setBuffer(hcStreamsBuffer_all, offset: 0, index: 0)
                                layerEnc1.setBuffer(hMlpBuffer_all, offset: 0, index: 1)
                                layerEnc1.setBuffer(hcInjectScaleBuffer_all, offset: 0, index: 2)
                                layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), injPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            } else {
                                layerEnc1.setComputePipelineState(addPipeline)
                                layerEnc1.setBuffer(hMidBuffer_all, offset: 0, index: 0)
                                layerEnc1.setBuffer(hMlpBuffer_all, offset: 0, index: 1)
                                layerEnc1.setBuffer(nextHBuf, offset: 0, index: 2)
                                layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            }
                        } else {
                            // MoE Router & Shared Gate
                            if let routerTensor = layer.routerTensor, let routerRaw = buffers[routerTensor.shardIndex] {
                                var rOffset = routerTensor.offsetStart
                                var nExp = numExperts
                                var kVal: UInt32 = UInt32(topKCount)
                                var grp: UInt32 = 64

                                if let rScale = layer.routerScale, let rBias = layer.routerBias,
                                   let rScaleRaw = buffers[rScale.shardIndex], let rBiasRaw = buffers[rBias.shardIndex] {
                                    var sOffset = rScale.offsetStart
                                    var bOffset = rBias.offsetStart

                                    if (routerTensor.shapeDisplay.contains("512") || numExperts >= 512), let rQ8 = routerQ8Pipeline {
                                        for p in 0..<P {
                                            let tokenOffset = p * Int(hiddenDim) * MemoryLayout<Float>.stride
                                            let rIndexOffset = p * topKCount * MemoryLayout<UInt32>.stride
                                            let rWeightOffset = p * topKCount * MemoryLayout<Float>.stride
                                            layerEnc1.setComputePipelineState(rQ8)
                                            layerEnc1.setBuffer(routerRaw, offset: 0, index: 0)
                                            layerEnc1.setBuffer(rScaleRaw, offset: 0, index: 1)
                                            layerEnc1.setBuffer(rBiasRaw, offset: 0, index: 2)
                                            layerEnc1.setBuffer(xNorm2Buffer_all, offset: tokenOffset, index: 3)
                                            layerEnc1.setBuffer(routerIndicesBuffer_all, offset: rIndexOffset, index: 4)
                                            layerEnc1.setBuffer(routerWeightsBuffer_all, offset: rWeightOffset, index: 5)
                                            layerEnc1.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                                            layerEnc1.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                                            layerEnc1.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 8)
                                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 9)
                                            layerEnc1.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 10)
                                            layerEnc1.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 11)
                                            layerEnc1.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 12)
                                            layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
                                        }
                                    } else if let rQ4 = routerQ4Pipeline {
                                        for p in 0..<P {
                                            let tokenOffset = p * Int(hiddenDim) * MemoryLayout<Float>.stride
                                            let rIndexOffset = p * topKCount * MemoryLayout<UInt32>.stride
                                            let rWeightOffset = p * topKCount * MemoryLayout<Float>.stride
                                            layerEnc1.setComputePipelineState(rQ4)
                                            layerEnc1.setBuffer(routerRaw, offset: 0, index: 0)
                                            layerEnc1.setBuffer(rScaleRaw, offset: 0, index: 1)
                                            layerEnc1.setBuffer(rBiasRaw, offset: 0, index: 2)
                                            layerEnc1.setBuffer(xNorm2Buffer_all, offset: tokenOffset, index: 3)
                                            layerEnc1.setBuffer(routerIndicesBuffer_all, offset: rIndexOffset, index: 4)
                                            layerEnc1.setBuffer(routerWeightsBuffer_all, offset: rWeightOffset, index: 5)
                                            layerEnc1.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                                            layerEnc1.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                                            layerEnc1.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 8)
                                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 9)
                                            layerEnc1.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 10)
                                            layerEnc1.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 11)
                                            layerEnc1.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 12)
                                            layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: Int(numExperts), height: 1, depth: 1))
                                        }
                                    }
                                } else {
                                    if (routerTensor.shapeDisplay.contains("512") || numExperts >= 512), let r512 = router512Pipeline {
                                        layerEnc1.setComputePipelineState(r512)
                                        layerEnc1.setBuffer(routerRaw, offset: 0, index: 0)
                                        layerEnc1.setBuffer(xNorm2Buffer_all, offset: 0, index: 1)
                                        layerEnc1.setBuffer(routerIndicesBuffer_all, offset: 0, index: 2)
                                        layerEnc1.setBuffer(routerWeightsBuffer_all, offset: 0, index: 3)
                                        layerEnc1.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                                        layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc1.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 7)
                                        layerEnc1.dispatchThreadgroups(MTLSize(width: P, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                                    } else if let rBias = layer.routerExpertBias, let rBiasRaw = buffers[rBias.shardIndex],
                                               let bailingPipe = routerBailingPipeline {
                                        var bOffset = rBias.offsetStart
                                        var nGrp: UInt32 = UInt32(modelConfig?.nGroup ?? 8)
                                        var topkGrp: UInt32 = UInt32(modelConfig?.topkGroup ?? 4)
                                        var scaleFactor: Float = Float(modelConfig?.routedScalingFactor ?? 2.5)

                                        for p in 0..<P {
                                            let tokenOffset = p * Int(hiddenDim) * MemoryLayout<Float>.stride
                                            let rIndexOffset = p * topKCount * MemoryLayout<UInt32>.stride
                                            let rWeightOffset = p * topKCount * MemoryLayout<Float>.stride
                                            layerEnc1.setComputePipelineState(bailingPipe)
                                            layerEnc1.setBuffer(routerRaw, offset: 0, index: 0)
                                            layerEnc1.setBuffer(rBiasRaw, offset: 0, index: 1)
                                            layerEnc1.setBuffer(xNorm2Buffer_all, offset: tokenOffset, index: 2)
                                            layerEnc1.setBuffer(routerIndicesBuffer_all, offset: rIndexOffset, index: 3)
                                            layerEnc1.setBuffer(routerWeightsBuffer_all, offset: rWeightOffset, index: 4)
                                            layerEnc1.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                                            layerEnc1.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 7)
                                            layerEnc1.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 8)
                                            layerEnc1.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 9)
                                            layerEnc1.setBytes(&nGrp, length: MemoryLayout<UInt32>.stride, index: 10)
                                            layerEnc1.setBytes(&topkGrp, length: MemoryLayout<UInt32>.stride, index: 11)
                                            layerEnc1.setBytes(&scaleFactor, length: MemoryLayout<Float>.stride, index: 12)
                                            layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: Int(numExperts), height: 1, depth: 1))
                                        }
                                    } else {
                                        for p in 0..<P {
                                            let tokenOffset = p * Int(hiddenDim) * MemoryLayout<Float>.stride
                                            let rIndexOffset = p * topKCount * MemoryLayout<UInt32>.stride
                                            let rWeightOffset = p * topKCount * MemoryLayout<Float>.stride
                                            layerEnc1.setComputePipelineState(routerPipeline)
                                            layerEnc1.setBuffer(routerRaw, offset: 0, index: 0)
                                            layerEnc1.setBuffer(xNorm2Buffer_all, offset: tokenOffset, index: 1)
                                            layerEnc1.setBuffer(routerIndicesBuffer_all, offset: rIndexOffset, index: 2)
                                            layerEnc1.setBuffer(routerWeightsBuffer_all, offset: rWeightOffset, index: 3)
                                            layerEnc1.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 5)
                                            layerEnc1.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 6)
                                            layerEnc1.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 7)
                                            layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: Int(numExperts), height: 1, depth: 1))
                                        }
                                    }
                                }

                                if let sg = layer.sharedGateTensor, let sgRaw = buffers[sg.shardIndex] {
                                    var sgOff = sg.offsetStart
                                    var grp: UInt32 = 64
                                    if let sgScale = layer.sharedGateTensorScale, let sgBias = layer.sharedGateTensorBias,
                                       let sgScaleRaw = buffers[sgScale.shardIndex], let sgBiasRaw = buffers[sgBias.shardIndex] {
                                        var sOff = sgScale.offsetStart
                                        var bOff = sgBias.offsetStart
                                        let is8Bit = (sg.offsetEnd - sg.offsetStart) >= UInt64(hiddenDim)
                                        if is8Bit, let pipe = sharedGateQ8Pipeline {
                                            for p in 0..<P {
                                                let tokenOffset = p * Int(hiddenDim) * MemoryLayout<Float>.stride
                                                let sScoreOffset = p * MemoryLayout<Float>.stride
                                                layerEnc1.setComputePipelineState(pipe)
                                                layerEnc1.setBuffer(sgRaw, offset: 0, index: 0)
                                                layerEnc1.setBuffer(sgScaleRaw, offset: 0, index: 1)
                                                layerEnc1.setBuffer(sgBiasRaw, offset: 0, index: 2)
                                                layerEnc1.setBuffer(xNorm2Buffer_all, offset: tokenOffset, index: 3)
                                                layerEnc1.setBuffer(sharedScoreBuffer_all, offset: sScoreOffset, index: 4)
                                                layerEnc1.setBytes(&sgOff, length: MemoryLayout<UInt64>.stride, index: 5)
                                                layerEnc1.setBytes(&sOff, length: MemoryLayout<UInt64>.stride, index: 6)
                                                layerEnc1.setBytes(&bOff, length: MemoryLayout<UInt64>.stride, index: 7)
                                                layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 8)
                                                layerEnc1.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 9)
                                                layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                            }
                                        } else if let pipe = sharedGateQ4Pipeline {
                                            for p in 0..<P {
                                                let tokenOffset = p * Int(hiddenDim) * MemoryLayout<Float>.stride
                                                let sScoreOffset = p * MemoryLayout<Float>.stride
                                                layerEnc1.setComputePipelineState(pipe)
                                                layerEnc1.setBuffer(sgRaw, offset: 0, index: 0)
                                                layerEnc1.setBuffer(sgScaleRaw, offset: 0, index: 1)
                                                layerEnc1.setBuffer(sgBiasRaw, offset: 0, index: 2)
                                                layerEnc1.setBuffer(xNorm2Buffer_all, offset: tokenOffset, index: 3)
                                                layerEnc1.setBuffer(sharedScoreBuffer_all, offset: sScoreOffset, index: 4)
                                                layerEnc1.setBytes(&sgOff, length: MemoryLayout<UInt64>.stride, index: 5)
                                                layerEnc1.setBytes(&sOff, length: MemoryLayout<UInt64>.stride, index: 6)
                                                layerEnc1.setBytes(&bOff, length: MemoryLayout<UInt64>.stride, index: 7)
                                                layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 8)
                                                layerEnc1.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 9)
                                                layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                            }
                                        }
                                    } else if let sgPipe = sharedGatePipeline {
                                        layerEnc1.setComputePipelineState(sgPipe)
                                        layerEnc1.setBuffer(sgRaw, offset: 0, index: 0)
                                        layerEnc1.setBuffer(xNorm2Buffer_all, offset: 0, index: 1)
                                        layerEnc1.setBuffer(sharedScoreBuffer_all, offset: 0, index: 2)
                                        layerEnc1.setBytes(&sgOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                        layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                                        layerEnc1.dispatchThreadgroups(MTLSize(width: P, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    }
                                }
                            }
                        }

                        layerEnc1.endEncoding()
                        activeCmd.commit()
                        let tPhaseAStart = CFAbsoluteTimeGetCurrent()
                        activeCmd.waitUntilCompleted()
                        let phaseAMs = (CFAbsoluteTimeGetCurrent() - tPhaseAStart) * 1000.0
                        prefillDiagPhaseAMs += phaseAMs
                        prefillDiagLayers += 1
                        if let actErr = activeCmd.error {
                            print("❌ [METAL ERROR] prefill layer activeCmd aborted: \(actErr)")
                            return false
                        }

                        // Phase B: Expert MLPs for MoE layers (Expert-Major Batching)
                        if layer.mlpType != .denseMlp && layer.routerTensor != nil {
                            let indStride = topKCount
                            let indAllPtr = routerIndicesBuffer_all.contents().bindMemory(to: UInt32.self, capacity: P * indStride)
                            let wAllPtr = routerWeightsBuffer_all.contents().bindMemory(to: Float.self, capacity: P * indStride)
                            let hasSharedGate = (layer.sharedGateTensor != nil)
                            let sharedScorePtr = sharedScoreBuffer_all.contents().bindMemory(to: Float.self, capacity: P)

                            var expertTokenMap: [Int: [(tokenIdx: UInt32, weight: Float)]] = [:]
                            for p in 0..<P {
                                let rBase = p * indStride
                                for i in 0..<topKCount {
                                    let expId = Int(indAllPtr[rBase + i])
                                    let pk = wAllPtr[rBase + i]
                                    if pk > 0.00001 {
                                        expertTokenMap[expId, default: []].append((tokenIdx: UInt32(p), weight: pk))
                                    }
                                }
                            }

                            var expertBufferOffsetMap: [Int: Int] = [:]
                            var currentExpOffset = 0
                            let expTokPtr = expertActiveTokensBuffer.contents().bindMemory(to: UInt32.self, capacity: max(P * topKCount, 64))
                            let expWgtPtr = expertActiveWeightsBuffer.contents().bindMemory(to: Float.self, capacity: max(P * topKCount, 64))
                            for (expId, tokenAssignments) in expertTokenMap {
                                guard !tokenAssignments.isEmpty else { continue }
                                expertBufferOffsetMap[expId] = currentExpOffset * MemoryLayout<UInt32>.stride
                                for item in tokenAssignments {
                                    expTokPtr[currentExpOffset] = item.tokenIdx
                                    expWgtPtr[currentExpOffset] = item.weight
                                    currentExpOffset += 1
                                }
                            }

                            if let packedDir = packedExpertsDir,
                               let fd = ExpertIOThreadPool.shared.getOrOpenLayerFD(layerIndex: l, packedExpertsDir: packedDir) {
                                let expertSize = Int(loadedLayout?.expert_size ?? 3151872)
                                let isFP8Layout = loadedLayout?.components.contains { $0.dtype.contains("F8") || $0.name.contains("weight_scale") } ?? false

                                let compGateW = loadedLayout?.components.first(where: { $0.name.contains("gate_proj") && $0.name.contains("weight") && !$0.name.contains("scale") && !$0.name.contains("bias") })
                                let compGateS = loadedLayout?.components.first(where: { $0.name.contains("gate_proj") && ($0.name.contains("scale") || $0.name.contains("scales")) })
                                let compGateB = loadedLayout?.components.first(where: { $0.name.contains("gate_proj") && ($0.name.contains("bias") || $0.name.contains("biases")) })

                                let compUpW = loadedLayout?.components.first(where: { $0.name.contains("up_proj") && $0.name.contains("weight") && !$0.name.contains("scale") && !$0.name.contains("bias") })
                                let compUpS = loadedLayout?.components.first(where: { $0.name.contains("up_proj") && ($0.name.contains("scale") || $0.name.contains("scales")) })
                                let compUpB = loadedLayout?.components.first(where: { $0.name.contains("up_proj") && ($0.name.contains("bias") || $0.name.contains("biases")) })

                                let compDownW = loadedLayout?.components.first(where: { $0.name.contains("down_proj") && $0.name.contains("weight") && !$0.name.contains("scale") && !$0.name.contains("bias") })
                                let compDownS = loadedLayout?.components.first(where: { $0.name.contains("down_proj") && ($0.name.contains("scale") || $0.name.contains("scales")) })
                                let compDownB = loadedLayout?.components.first(where: { $0.name.contains("down_proj") && ($0.name.contains("bias") || $0.name.contains("biases")) })

                                let uniqueActiveExpIds = Array(expertTokenMap.keys).sorted()
                                let numActive = uniqueActiveExpIds.count

                                if numActive > 0 {
                                    // FIX #5: consume the speculative kick that filled this
                                    // layer's predicted experts during the previous layer's
                                    // GPU window, then synchronously read only the misses.
                                    let tIoStart = CFAbsoluteTimeGetCurrent()
                                    var predHits = 0
                                    // FIX #5: drain any stale kicks aimed at earlier layers
                                    // (e.g. skipped dense layers) so their staging buffers
                                    // are quiescent before reuse.
                                    for (idx, stale) in prefetchedFor where idx < l {
                                        stale.sem.wait()
                                        prefetchedFor[idx] = nil
                                    }
                                    var predictedSlots: [Int: Int]? = nil
                                    if prefillPipelineEnabled, let pf = prefetchedFor[l] {
                                        pf.sem.wait()
                                        prefetchedFor[l] = nil
                                        predictedSlots = pf.slotMap
                                        prefillStagingBuffer = pf.buf
                                        lastMoeStageIsA = (pf.buf === prefillStagingBufferA)
                                    } else {
                                        // No speculative data landed for this layer: read into
                                        // the buffer NOT used by the last MoE GPU command.
                                        prefillStagingBuffer = lastMoeStageIsA ? prefillStagingBufferB : prefillStagingBufferA
                                        lastMoeStageIsA = !lastMoeStageIsA
                                    }
                                    let packed = packPrefillExpertSlots(predicted: predictedSlots, activeIds: uniqueActiveExpIds)
                                    let expSlotMap = packed.map
                                    predHits = uniqueActiveExpIds.count - packed.misses.count
                                    var tasks: [ExpertPreadTask] = []
                                    let rawStagingPtr = prefillStagingBuffer.contents()
                                    for expId in packed.misses {
                                        let slot = expSlotMap[expId]!
                                        let offset = off_t(expId * expertSize)
                                        let dst = rawStagingPtr.advanced(by: slot * expertSize)
                                        tasks.append(ExpertPreadTask(fd: fd, dst: dst, offset: offset, size: expertSize))
                                    }
                                    if !tasks.isEmpty {
                                        ExpertIOThreadPool.shared.dispatchSync(tasks: &tasks)
                                    }
                                    let ioWaitMs = (CFAbsoluteTimeGetCurrent() - tIoStart) * 1000.0
                                    prefillDiagIoMs += ioWaitMs
                                    prefillDiagHits += predHits
                                    prefillDiagTotal += numActive
                                    print("[PREFILL-DIAG] Layer \(l): act=\(numActive) hits=\(predHits) miss=\(numActive - predHits) PhaseA=\(String(format: "%.1f", phaseAMs))ms ioWait=\(String(format: "%.1f", ioWaitMs))ms")

                                    guard let moeCmd = commandQueue.makeCommandBuffer(),
                                          let layerEnc2 = moeCmd.makeComputeCommandEncoder() else {
                                        return false
                                    }

                                    // Clear hMlpBuffer_all for all P tokens
                                    layerEnc2.setComputePipelineState(clearPipeline)
                                    layerEnc2.setBuffer(hMlpBuffer_all, offset: 0, index: 0)
                                    layerEnc2.dispatchThreads(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    layerEnc2.memoryBarrier(scope: .buffers)

                                    for expId in uniqueActiveExpIds {
                                        guard let tokenAssignments = expertTokenMap[expId], let slot = expSlotMap[expId] else { continue }
                                        let count = tokenAssignments.count
                                        if count == 0 { continue }
                                        let expOffset = expertBufferOffsetMap[expId] ?? 0

                                        let slotOffset = UInt64(slot * expertSize)
                                        let gWOff = slotOffset + (compGateW?.offset ?? 0)
                                        let gSOff = slotOffset + (compGateS?.offset ?? 1048576)
                                        let gBOff = slotOffset + (compGateB?.offset ?? 0)
                                        let uWOff = slotOffset + (compUpW?.offset ?? 1049600)
                                        let uSOff = slotOffset + (compUpS?.offset ?? 2098176)
                                        let uBOff = slotOffset + (compUpB?.offset ?? 0)
                                        let dWOff = slotOffset + (compDownW?.offset ?? 2099200)
                                        let dSOff = slotOffset + (compDownS?.offset ?? 3147776)
                                        let dBOff = slotOffset + (compDownB?.offset ?? 0)
                                        let isQuantizedAffine = (compGateB != nil || compGateW?.name.contains("Q4") == true || compGateW?.name.contains("Q8") == true || compGateW?.dtype.contains("Q4") == true || compGateW?.dtype.contains("Q8") == true || (compGateS != nil && !isFP8Layout))

                                        if isFP8Layout {
                                            let isBlockScale = (compGateS?.size ?? 1024) < (intermediateDim * 2) || (compGateS?.name.contains("scale_inv") ?? false)
                                            let gateBatched = isBlockScale ? (fp8BlockGateUpBatchedPipeline ?? fp8BlockGateUpSimdPipeline ?? fp8GateUpBatchedPipeline) : (fp8GateUpBatchedPipeline ?? fp8GateUpSimdPipeline ?? fp8GateUpPipeline)
                                            let downBatched = isBlockScale ? (fp8BlockDownBatchedPipeline ?? fp8BlockDownSimdPipeline ?? fp8DownBatchedPipeline) : (fp8DownBatchedPipeline ?? fp8DownSimdPipeline ?? fp8DownPipeline)

                                            if let gateBatched = gateBatched,
                                               let downBatched = downBatched {
                                                var gWOffU = gWOff
                                                var gSOffU = gSOff
                                                var uWOffU = uWOff
                                                var uSOffU = uSOff
                                                var dWOffU = dWOff
                                                var dSOffU = dSOff
                                                var hDimVal: UInt32 = UInt32(hiddenDim)
                                                var interDimVal: UInt32 = UInt32(intermediateDim)

                                                layerEnc2.setComputePipelineState(gateBatched)
                                                layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 0)
                                                layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 1)
                                                layerEnc2.setBuffer(xNorm2Buffer_all, offset: 0, index: 2)
                                                layerEnc2.setBuffer(interBuffer_all, offset: 0, index: 3)
                                                layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 4)
                                                layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 5)
                                                layerEnc2.setBytes(&gWOffU, length: 8, index: 6)
                                                layerEnc2.setBytes(&gSOffU, length: 8, index: 7)
                                                layerEnc2.setBytes(&uWOffU, length: 8, index: 8)
                                                layerEnc2.setBytes(&uSOffU, length: 8, index: 9)
                                                layerEnc2.setBytes(&hDimVal, length: 4, index: 10)
                                                layerEnc2.setBytes(&interDimVal, length: 4, index: 11)
                                                layerEnc2.setBuffer(expertActiveTokensBuffer, offset: expOffset, index: 12)
                                                layerEnc2.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: count, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                                layerEnc2.memoryBarrier(scope: .buffers)

                                                layerEnc2.setComputePipelineState(downBatched)
                                                layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 0)
                                                layerEnc2.setBuffer(interBuffer_all, offset: 0, index: 1)
                                                layerEnc2.setBuffer(hMlpBuffer_all, offset: 0, index: 2)
                                                layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 3)
                                                layerEnc2.setBytes(&dWOffU, length: 8, index: 4)
                                                layerEnc2.setBytes(&dSOffU, length: 8, index: 5)
                                                layerEnc2.setBytes(&interDimVal, length: 4, index: 6)
                                                layerEnc2.setBytes(&hDimVal, length: 4, index: 7)
                                                var pkVal: Float = 1.0
                                                layerEnc2.setBytes(&pkVal, length: 4, index: 8)
                                                layerEnc2.setBuffer(expertActiveTokensBuffer, offset: expOffset, index: 9)
                                                layerEnc2.setBuffer(expertActiveWeightsBuffer, offset: expOffset, index: 10)
                                                layerEnc2.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: count, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                                layerEnc2.memoryBarrier(scope: .buffers)
                                            }
                                        } else if !isQuantizedAffine,
                                                  let gateBatched = bf16GateUpBatchedPipeline ?? bf16GateUpSimdPipeline,
                                                  let downBatched = bf16DownBatchedPipeline ?? bf16DownSimdPipeline {
                                            var gWOffU = gWOff
                                            var uWOffU = uWOff
                                            var dWOffU = dWOff
                                            var hDimVal: UInt32 = UInt32(hiddenDim)
                                            var interDimVal: UInt32 = UInt32(intermediateDim)

                                            layerEnc2.setComputePipelineState(gateBatched)
                                            layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 0)
                                            layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 1)
                                            layerEnc2.setBuffer(xNorm2Buffer_all, offset: 0, index: 2)
                                            layerEnc2.setBuffer(interBuffer_all, offset: 0, index: 3)
                                            layerEnc2.setBytes(&gWOffU, length: 8, index: 4)
                                            layerEnc2.setBytes(&uWOffU, length: 8, index: 5)
                                            layerEnc2.setBytes(&hDimVal, length: 4, index: 6)
                                            layerEnc2.setBytes(&interDimVal, length: 4, index: 7)
                                            layerEnc2.setBuffer(expertActiveTokensBuffer, offset: expOffset, index: 8)
                                            layerEnc2.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: count, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                            layerEnc2.memoryBarrier(scope: .buffers)

                                            layerEnc2.setComputePipelineState(downBatched)
                                            layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 0)
                                            layerEnc2.setBuffer(interBuffer_all, offset: 0, index: 1)
                                            layerEnc2.setBuffer(hMlpBuffer_all, offset: 0, index: 2)
                                            layerEnc2.setBytes(&dWOffU, length: 8, index: 3)
                                            layerEnc2.setBytes(&interDimVal, length: 4, index: 4)
                                            layerEnc2.setBytes(&hDimVal, length: 4, index: 5)
                                            var pkVal: Float = 1.0
                                            layerEnc2.setBytes(&pkVal, length: 4, index: 6)
                                            layerEnc2.setBuffer(expertActiveTokensBuffer, offset: expOffset, index: 7)
                                            layerEnc2.setBuffer(expertActiveWeightsBuffer, offset: expOffset, index: 8)
                                            layerEnc2.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: count, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                            layerEnc2.memoryBarrier(scope: .buffers)
                                        } else if let q4Gate = q4GateUpPipeline,
                                                  let q4Down = q4DownPipeline {
                                            for item in tokenAssignments {
                                                let tOff = Int(item.tokenIdx) * Int(hiddenDim) * MemoryLayout<Float>.stride
                                                let iOff = Int(item.tokenIdx) * Int(intermediateDim) * MemoryLayout<Float>.stride
                                                var gWOffU = gWOff
                                                var gSOffU = gSOff
                                                var gBOffU = gBOff
                                                var uWOffU = uWOff
                                                var uSOffU = uSOff
                                                var uBOffU = uBOff
                                                var dWOffU = dWOff
                                                var dSOffU = dSOff
                                                var dBOffU = dBOff
                                                var hDimVal: UInt32 = UInt32(hiddenDim)
                                                var interDimVal: UInt32 = UInt32(intermediateDim)
                                                var grp: UInt32 = 64
                                                var pkVal = item.weight

                                                layerEnc2.setComputePipelineState(q4Gate)
                                                layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 0)
                                                layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 1)
                                                layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 2)
                                                layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 3)
                                                layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 4)
                                                layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 5)
                                                layerEnc2.setBuffer(xNorm2Buffer_all, offset: tOff, index: 6)
                                                layerEnc2.setBuffer(interBuffer_all, offset: iOff, index: 7)
                                                layerEnc2.setBytes(&gWOffU, length: 8, index: 8)
                                                layerEnc2.setBytes(&gSOffU, length: 8, index: 9)
                                                layerEnc2.setBytes(&gBOffU, length: 8, index: 10)
                                                layerEnc2.setBytes(&uWOffU, length: 8, index: 11)
                                                layerEnc2.setBytes(&uSOffU, length: 8, index: 12)
                                                layerEnc2.setBytes(&uBOffU, length: 8, index: 13)
                                                layerEnc2.setBytes(&hDimVal, length: 4, index: 14)
                                                layerEnc2.setBytes(&interDimVal, length: 4, index: 15)
                                                layerEnc2.setBytes(&grp, length: 4, index: 16)
                                                layerEnc2.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                                layerEnc2.memoryBarrier(scope: .buffers)

                                                layerEnc2.setComputePipelineState(q4Down)
                                                layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 0)
                                                layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 1)
                                                layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 2)
                                                layerEnc2.setBuffer(interBuffer_all, offset: iOff, index: 3)
                                                layerEnc2.setBuffer(hMlpBuffer_all, offset: tOff, index: 4)
                                                layerEnc2.setBytes(&dWOffU, length: 8, index: 5)
                                                layerEnc2.setBytes(&dSOffU, length: 8, index: 6)
                                                layerEnc2.setBytes(&dBOffU, length: 8, index: 7)
                                                layerEnc2.setBytes(&interDimVal, length: 4, index: 8)
                                                layerEnc2.setBytes(&hDimVal, length: 4, index: 9)
                                                layerEnc2.setBytes(&grp, length: 4, index: 10)
                                                layerEnc2.setBytes(&pkVal, length: 4, index: 11)
                                                layerEnc2.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                                layerEnc2.memoryBarrier(scope: .buffers)
                                            }
                                        }
                                    }

                                    if let gateW = layer.sharedGateWeight,
                                       let upW = layer.sharedUpWeight,
                                       let downW = layer.sharedDownWeight {
                                        for p in 0..<P {
                                            let tokenOffset = p * Int(hiddenDim) * MemoryLayout<Float>.stride
                                            let interTokenOffset = p * Int(intermediateDim) * MemoryLayout<Float>.stride
                                            let sharedW = hasSharedGate ? sharedScorePtr[p] : 1.0
                                            let gateS = layer.sharedGateScale
                                            let gateB = layer.sharedGateBias
                                            let upS = layer.sharedUpScale
                                            let upB = layer.sharedUpBias
                                            let downS = layer.sharedDownScale
                                            let downB = layer.sharedDownBias

                                            dispatchExpertMlp(
                                                enc: layerEnc2,
                                                gateW: gateW,
                                                gateS: gateS,
                                                gateB: gateB,
                                                upW: upW,
                                                upS: upS,
                                                upB: upB,
                                                downW: downW,
                                                downS: downS,
                                                downB: downB,
                                                inBuf: xNorm2Buffer_all,
                                                interBuf: interBuffer_all,
                                                accumBuf: hMlpBuffer_all,
                                                inDim: hiddenDim,
                                                interDim: intermediateDim,
                                                routingWeight: sharedW,
                                                inOffset: tokenOffset,
                                                interOffset: interTokenOffset,
                                                accumOffset: tokenOffset
                                            )
                                        }
                                        layerEnc2.memoryBarrier(scope: .buffers)
                                    }

                                    if (layer.mlpHcInjectWeight != nil || layer.mlpHcDownWeight != nil),
                                       let injPipe = hcInjectPipeline {
                                        layerEnc2.setComputePipelineState(injPipe)
                                        layerEnc2.setBuffer(hcStreamsBuffer_all, offset: 0, index: 0)
                                        layerEnc2.setBuffer(hMlpBuffer_all, offset: 0, index: 1)
                                        layerEnc2.setBuffer(hcInjectScaleBuffer_all, offset: 0, index: 2)
                                        layerEnc2.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                                        layerEnc2.dispatchThreads(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), injPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    } else {
                                        layerEnc2.setComputePipelineState(addPipeline)
                                        layerEnc2.setBuffer(hMidBuffer_all, offset: 0, index: 0)
                                        layerEnc2.setBuffer(hMlpBuffer_all, offset: 0, index: 1)
                                        layerEnc2.setBuffer(nextHBuf, offset: 0, index: 2)
                                        layerEnc2.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                                        layerEnc2.dispatchThreads(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    }

                                    layerEnc2.endEncoding()
                                    moeCmd.commit()

                                    // FIX #5: while this layer's MoE runs on the GPU, kick
                                    // layer l+1's predicted expert reads (temporal prediction:
                                    // this layer's active set) into the other staging buffer.
                                    if prefillPipelineEnabled, l + 1 < actualLayers,
                                       let nextFD = ExpertIOThreadPool.shared.getOrOpenLayerFD(layerIndex: l + 1, packedExpertsDir: packedDir) {
                                        let targetBuf = lastMoeStageIsA ? prefillStagingBufferB : prefillStagingBufferA
                                        let kickSem = DispatchSemaphore(value: 0)
                                        var kickMap: [Int: Int] = [:]
                                        var kickTasks: [ExpertPreadTask] = []
                                        let targetPtr = targetBuf.contents()
                                        for (slot, expId) in uniqueActiveExpIds.enumerated() {
                                            kickMap[expId] = slot
                                            kickTasks.append(ExpertPreadTask(fd: nextFD, dst: targetPtr.advanced(by: slot * expertSize), offset: off_t(expId * expertSize), size: expertSize))
                                        }
                                        prefetchedFor[l + 1] = (buf: targetBuf, slotMap: kickMap, slotBase: uniqueActiveExpIds.count, sem: kickSem)
                                        ExpertIOThreadPool.shared.dispatchAsync(tasks: kickTasks, done: kickSem)
                                    }

                                    let tGpuStart = CFAbsoluteTimeGetCurrent()
                                    moeCmd.waitUntilCompleted()
                                    prefillDiagGpuMs += (CFAbsoluteTimeGetCurrent() - tGpuStart) * 1000.0
                                    if let moeErr = moeCmd.error {
                                        print("❌ [METAL ERROR] prefill MoE cmd (speculative) aborted: \(moeErr)")
                                        return false
                                    }
                                }
                            } else {
                                let activeExpIds = Array(expertTokenMap.keys)
                                if speculativePrefetchEnabled && packedExpertsDir == nil {
                                    let nextL = (l + 1) < actualLayers ? (l + 1) : 0
                                    WorkingSetManager.shared.prefetchLayerBackbone(layer: cachedLayers[nextL], shardBuffers: buffers)
                                }
                                WorkingSetManager.shared.touchAndEvict(layer: l, activeExpertIds: activeExpIds, mode: budgetMode, shardBuffers: buffers, isPrefill: true)

                                guard let moeCmd = commandQueue.makeCommandBuffer(),
                                      let layerEnc2 = moeCmd.makeComputeCommandEncoder() else {
                                    print("❌ [METAL] prefill: MoE command buffer allocation failed (layer \(l), chunk startPos=\(startPos), P=\(P))")
                                    return false
                                }

                                layerEnc2.setComputePipelineState(clearPipeline)
                                layerEnc2.setBuffer(hMlpBuffer_all, offset: 0, index: 0)
                                layerEnc2.dispatchThreads(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc2.memoryBarrier(scope: .buffers)

                                for (expId, tokenAssignments) in expertTokenMap {
                                    guard let gateW = layer.expertGateWeights[expId],
                                          let upW = layer.expertUpWeights[expId],
                                          let downW = layer.expertDownWeights[expId],
                                          let gRaw = buffers[gateW.shardIndex],
                                          let uRaw = buffers[upW.shardIndex],
                                          let dRaw = buffers[downW.shardIndex] else { continue }

                                    let count = tokenAssignments.count
                                    if count == 0 { continue }
                                    let expOffset = expertBufferOffsetMap[expId] ?? 0

                                    var gWOff = gateW.offsetStart
                                    var uWOff = upW.offsetStart
                                    var dWOff = downW.offsetStart
                                    var hDimVal: UInt32 = UInt32(hiddenDim)
                                    var interDimVal: UInt32 = UInt32(intermediateDim)

                                    let gateS = layer.expertGateScales[expId]
                                    let upS = layer.expertUpScales[expId]
                                    let downS = layer.expertDownScales[expId]
                                    let isBlockScale = (gateS != nil) && (gateS!.name.contains("scale_inv") || ((gateS!.offsetEnd - gateS!.offsetStart) < UInt64(intermediateDim * 2)))
                                    let isQuantizedAffine = (layer.expertGateBiases[expId] != nil || gateW.dtype.contains("Q4") || gateW.dtype.contains("Q8") || (gateS != nil && !isBlockScale))

                                    if isBlockScale,
                                       let gateBatched = fp8BlockGateUpBatchedPipeline ?? fp8BlockGateUpSimdPipeline ?? fp8GateUpBatchedPipeline,
                                       let downBatched = fp8BlockDownBatchedPipeline ?? fp8BlockDownSimdPipeline ?? fp8DownBatchedPipeline,
                                       let gsRaw = gateS != nil ? buffers[gateS!.shardIndex] : nil,
                                       let usRaw = upS != nil ? buffers[upS!.shardIndex] : nil,
                                       let dsRaw = downS != nil ? buffers[downS!.shardIndex] : nil {
                                        var gSOff = gateS?.offsetStart ?? 0
                                        var uSOff = upS?.offsetStart ?? 0
                                        var dSOff = downS?.offsetStart ?? 0

                                        layerEnc2.setComputePipelineState(gateBatched)
                                        layerEnc2.setBuffer(gRaw, offset: 0, index: 0)
                                        layerEnc2.setBuffer(uRaw, offset: 0, index: 1)
                                        layerEnc2.setBuffer(xNorm2Buffer_all, offset: 0, index: 2)
                                        layerEnc2.setBuffer(interBuffer_all, offset: 0, index: 3)
                                        layerEnc2.setBuffer(gsRaw, offset: 0, index: 4)
                                        layerEnc2.setBuffer(usRaw, offset: 0, index: 5)
                                        layerEnc2.setBytes(&gWOff, length: 8, index: 6)
                                        layerEnc2.setBytes(&gSOff, length: 8, index: 7)
                                        layerEnc2.setBytes(&uWOff, length: 8, index: 8)
                                        layerEnc2.setBytes(&uSOff, length: 8, index: 9)
                                        layerEnc2.setBytes(&hDimVal, length: 4, index: 10)
                                        layerEnc2.setBytes(&interDimVal, length: 4, index: 11)
                                        layerEnc2.setBuffer(expertActiveTokensBuffer, offset: expOffset, index: 12)
                                        layerEnc2.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: count, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        layerEnc2.memoryBarrier(scope: .buffers)

                                        layerEnc2.setComputePipelineState(downBatched)
                                        layerEnc2.setBuffer(dRaw, offset: 0, index: 0)
                                        layerEnc2.setBuffer(interBuffer_all, offset: 0, index: 1)
                                        layerEnc2.setBuffer(hMlpBuffer_all, offset: 0, index: 2)
                                        layerEnc2.setBuffer(dsRaw, offset: 0, index: 3)
                                        layerEnc2.setBytes(&dWOff, length: 8, index: 4)
                                        layerEnc2.setBytes(&dSOff, length: 8, index: 5)
                                        layerEnc2.setBytes(&interDimVal, length: 4, index: 6)
                                        layerEnc2.setBytes(&hDimVal, length: 4, index: 7)
                                        var pkVal: Float = 1.0
                                        layerEnc2.setBytes(&pkVal, length: 4, index: 8)
                                        layerEnc2.setBuffer(expertActiveTokensBuffer, offset: expOffset, index: 9)
                                        layerEnc2.setBuffer(expertActiveWeightsBuffer, offset: expOffset, index: 10)
                                        layerEnc2.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: count, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        layerEnc2.memoryBarrier(scope: .buffers)
                                    } else if !isQuantizedAffine,
                                              let gateBatched = bf16GateUpBatchedPipeline ?? bf16GateUpSimdPipeline,
                                              let downBatched = bf16DownBatchedPipeline ?? bf16DownSimdPipeline {
                                        layerEnc2.setComputePipelineState(gateBatched)
                                        layerEnc2.setBuffer(gRaw, offset: 0, index: 0)
                                        layerEnc2.setBuffer(uRaw, offset: 0, index: 1)
                                        layerEnc2.setBuffer(xNorm2Buffer_all, offset: 0, index: 2)
                                        layerEnc2.setBuffer(interBuffer_all, offset: 0, index: 3)
                                        layerEnc2.setBytes(&gWOff, length: 8, index: 4)
                                        layerEnc2.setBytes(&uWOff, length: 8, index: 5)
                                        layerEnc2.setBytes(&hDimVal, length: 4, index: 6)
                                        layerEnc2.setBytes(&interDimVal, length: 4, index: 7)
                                        layerEnc2.setBuffer(expertActiveTokensBuffer, offset: expOffset, index: 8)
                                        layerEnc2.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: count, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        layerEnc2.memoryBarrier(scope: .buffers)

                                        layerEnc2.setComputePipelineState(downBatched)
                                        layerEnc2.setBuffer(dRaw, offset: 0, index: 0)
                                        layerEnc2.setBuffer(interBuffer_all, offset: 0, index: 1)
                                        layerEnc2.setBuffer(hMlpBuffer_all, offset: 0, index: 2)
                                        layerEnc2.setBytes(&dWOff, length: 8, index: 3)
                                        layerEnc2.setBytes(&interDimVal, length: 4, index: 4)
                                        layerEnc2.setBytes(&hDimVal, length: 4, index: 5)
                                        var pkVal: Float = 1.0
                                        layerEnc2.setBytes(&pkVal, length: 4, index: 6)
                                        layerEnc2.setBuffer(expertActiveTokensBuffer, offset: expOffset, index: 7)
                                        layerEnc2.setBuffer(expertActiveWeightsBuffer, offset: expOffset, index: 8)
                                        layerEnc2.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: count, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        layerEnc2.memoryBarrier(scope: .buffers)
                                    } else {
                                        for item in tokenAssignments {
                                            let tokenOffset = Int(item.tokenIdx) * Int(hiddenDim) * MemoryLayout<Float>.stride
                                            let interTokenOffset = Int(item.tokenIdx) * Int(intermediateDim) * MemoryLayout<Float>.stride
                                            dispatchExpertMlp(
                                                enc: layerEnc2,
                                                gateW: gateW,
                                                gateS: gateS,
                                                gateB: layer.expertGateBiases[expId],
                                                upW: upW,
                                                upS: upS,
                                                upB: layer.expertUpBiases[expId],
                                                downW: downW,
                                                downS: downS,
                                                downB: layer.expertDownBiases[expId],
                                                inBuf: xNorm2Buffer_all,
                                                interBuf: interBuffer_all,
                                                accumBuf: hMlpBuffer_all,
                                                inDim: hiddenDim,
                                                interDim: intermediateDim,
                                                routingWeight: item.weight,
                                                inOffset: tokenOffset,
                                                interOffset: interTokenOffset,
                                                accumOffset: tokenOffset
                                            )
                                        }
                                    }
                                }
                                layerEnc2.memoryBarrier(scope: .buffers)

                                if let gateW = layer.sharedGateWeight,
                                   let upW = layer.sharedUpWeight,
                                   let downW = layer.sharedDownWeight,
                                   let gRaw = buffers[gateW.shardIndex],
                                   let uRaw = buffers[upW.shardIndex],
                                   let dRaw = buffers[downW.shardIndex] {
                                    var gWOff = gateW.offsetStart
                                    var uWOff = upW.offsetStart
                                    var dWOff = downW.offsetStart
                                    var hDimVal: UInt32 = UInt32(hiddenDim)
                                    var interDimVal: UInt32 = UInt32(intermediateDim)
                                    let sharedWeightsBuf = hasSharedGate ? sharedScoreBuffer_all : denseActiveWeightsBuffer

                                    let gateS = layer.sharedGateScale
                                    let upS = layer.sharedUpScale
                                    let downS = layer.sharedDownScale
                                    let isBlockScale = (gateS != nil) && (gateS!.name.contains("scale_inv") || ((gateS!.offsetEnd - gateS!.offsetStart) < UInt64(intermediateDim * 2)))
                                    let isQuantizedAffine = (layer.sharedGateBias != nil || gateW.dtype.contains("Q4") || gateW.dtype.contains("Q8") || (gateS != nil && !isBlockScale))

                                    if isBlockScale,
                                       let gateBatched = fp8BlockGateUpBatchedPipeline ?? fp8BlockGateUpSimdPipeline ?? fp8GateUpBatchedPipeline,
                                       let downBatched = fp8BlockDownBatchedPipeline ?? fp8BlockDownSimdPipeline ?? fp8DownBatchedPipeline,
                                       let gsRaw = gateS != nil ? buffers[gateS!.shardIndex] : nil,
                                       let usRaw = upS != nil ? buffers[upS!.shardIndex] : nil,
                                       let dsRaw = downS != nil ? buffers[downS!.shardIndex] : nil {
                                        var gSOff = gateS?.offsetStart ?? 0
                                        var uSOff = upS?.offsetStart ?? 0
                                        var dSOff = downS?.offsetStart ?? 0

                                        layerEnc2.setComputePipelineState(gateBatched)
                                        layerEnc2.setBuffer(gRaw, offset: 0, index: 0)
                                        layerEnc2.setBuffer(uRaw, offset: 0, index: 1)
                                        layerEnc2.setBuffer(xNorm2Buffer_all, offset: 0, index: 2)
                                        layerEnc2.setBuffer(interBuffer_all, offset: 0, index: 3)
                                        layerEnc2.setBuffer(gsRaw, offset: 0, index: 4)
                                        layerEnc2.setBuffer(usRaw, offset: 0, index: 5)
                                        layerEnc2.setBytes(&gWOff, length: 8, index: 6)
                                        layerEnc2.setBytes(&gSOff, length: 8, index: 7)
                                        layerEnc2.setBytes(&uWOff, length: 8, index: 8)
                                        layerEnc2.setBytes(&uSOff, length: 8, index: 9)
                                        layerEnc2.setBytes(&hDimVal, length: 4, index: 10)
                                        layerEnc2.setBytes(&interDimVal, length: 4, index: 11)
                                        layerEnc2.setBuffer(denseActiveTokensBuffer, offset: 0, index: 12)
                                        layerEnc2.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        layerEnc2.memoryBarrier(scope: .buffers)

                                        layerEnc2.setComputePipelineState(downBatched)
                                        layerEnc2.setBuffer(dRaw, offset: 0, index: 0)
                                        layerEnc2.setBuffer(interBuffer_all, offset: 0, index: 1)
                                        layerEnc2.setBuffer(hMlpBuffer_all, offset: 0, index: 2)
                                        layerEnc2.setBuffer(dsRaw, offset: 0, index: 3)
                                        layerEnc2.setBytes(&dWOff, length: 8, index: 4)
                                        layerEnc2.setBytes(&dSOff, length: 8, index: 5)
                                        layerEnc2.setBytes(&interDimVal, length: 4, index: 6)
                                        layerEnc2.setBytes(&hDimVal, length: 4, index: 7)
                                        var pkVal: Float = 1.0
                                        layerEnc2.setBytes(&pkVal, length: 4, index: 8)
                                        layerEnc2.setBuffer(denseActiveTokensBuffer, offset: 0, index: 9)
                                        layerEnc2.setBuffer(sharedWeightsBuf, offset: 0, index: 10)
                                        layerEnc2.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        layerEnc2.memoryBarrier(scope: .buffers)
                                    } else if !isQuantizedAffine,
                                              let gateBatched = bf16GateUpBatchedPipeline ?? bf16GateUpSimdPipeline,
                                              let downBatched = bf16DownBatchedPipeline ?? bf16DownSimdPipeline {
                                        layerEnc2.setComputePipelineState(gateBatched)
                                        layerEnc2.setBuffer(gRaw, offset: 0, index: 0)
                                        layerEnc2.setBuffer(uRaw, offset: 0, index: 1)
                                        layerEnc2.setBuffer(xNorm2Buffer_all, offset: 0, index: 2)
                                        layerEnc2.setBuffer(interBuffer_all, offset: 0, index: 3)
                                        layerEnc2.setBytes(&gWOff, length: 8, index: 4)
                                        layerEnc2.setBytes(&uWOff, length: 8, index: 5)
                                        layerEnc2.setBytes(&hDimVal, length: 4, index: 6)
                                        layerEnc2.setBytes(&interDimVal, length: 4, index: 7)
                                        layerEnc2.setBuffer(denseActiveTokensBuffer, offset: 0, index: 8)
                                        layerEnc2.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        layerEnc2.memoryBarrier(scope: .buffers)

                                        layerEnc2.setComputePipelineState(downBatched)
                                        layerEnc2.setBuffer(dRaw, offset: 0, index: 0)
                                        layerEnc2.setBuffer(interBuffer_all, offset: 0, index: 1)
                                        layerEnc2.setBuffer(hMlpBuffer_all, offset: 0, index: 2)
                                        layerEnc2.setBytes(&dWOff, length: 8, index: 3)
                                        layerEnc2.setBytes(&interDimVal, length: 4, index: 4)
                                        layerEnc2.setBytes(&hDimVal, length: 4, index: 5)
                                        var pkVal: Float = 1.0
                                        layerEnc2.setBytes(&pkVal, length: 4, index: 6)
                                        layerEnc2.setBuffer(denseActiveTokensBuffer, offset: 0, index: 7)
                                        layerEnc2.setBuffer(sharedWeightsBuf, offset: 0, index: 8)
                                        layerEnc2.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        layerEnc2.memoryBarrier(scope: .buffers)
                                    } else {
                                        for p in 0..<P {
                                            let tokenOffset = p * Int(hiddenDim) * MemoryLayout<Float>.stride
                                            let interTokenOffset = p * Int(intermediateDim) * MemoryLayout<Float>.stride
                                            let sharedW = hasSharedGate ? sharedScorePtr[p] : 1.0
                                            dispatchExpertMlp(
                                                enc: layerEnc2,
                                                gateW: gateW,
                                                gateS: gateS,
                                                gateB: layer.sharedGateBias,
                                                upW: upW,
                                                upS: upS,
                                                upB: layer.sharedUpBias,
                                                downW: downW,
                                                downS: downS,
                                                downB: layer.sharedDownBias,
                                                inBuf: xNorm2Buffer_all,
                                                interBuf: interBuffer_all,
                                                accumBuf: hMlpBuffer_all,
                                                inDim: hiddenDim,
                                                interDim: intermediateDim,
                                                routingWeight: sharedW,
                                                inOffset: tokenOffset,
                                                interOffset: interTokenOffset,
                                                accumOffset: tokenOffset
                                            )
                                        }
                                    }
                                }
                                layerEnc2.memoryBarrier(scope: .buffers)

                                if (layer.mlpHcInjectWeight != nil || layer.mlpHcDownWeight != nil),
                                   let injPipe = hcInjectPipeline {
                                    layerEnc2.setComputePipelineState(injPipe)
                                    layerEnc2.setBuffer(hcStreamsBuffer_all, offset: 0, index: 0)
                                    layerEnc2.setBuffer(hMlpBuffer_all, offset: 0, index: 1)
                                    layerEnc2.setBuffer(hcInjectScaleBuffer_all, offset: 0, index: 2)
                                    layerEnc2.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                                    layerEnc2.dispatchThreads(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), injPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                } else {
                                    layerEnc2.setComputePipelineState(addPipeline)
                                    layerEnc2.setBuffer(hMidBuffer_all, offset: 0, index: 0)
                                    layerEnc2.setBuffer(hMlpBuffer_all, offset: 0, index: 1)
                                    layerEnc2.setBuffer(nextHBuf, offset: 0, index: 2)
                                    layerEnc2.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                                    layerEnc2.dispatchThreads(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                }

                                layerEnc2.endEncoding()
                                moeCmd.commit()
                                moeCmd.waitUntilCompleted()
                            }
                        }

                        // Swap ping-pong hidden buffers for next layer
                        let tmp = currHBuf
                        currHBuf = nextHBuf
                        nextHBuf = tmp

                        if packedExpertsDir == nil && WorkingSetManager.shared.residentExpertsCount > budgetMode.maxResidentExperts {
                            WorkingSetManager.shared.trimToBudget(mode: budgetMode, shardBuffers: buffers)
                        }

                        passIdx += 1
                        let now = CFAbsoluteTimeGetCurrent()
                        if now - lastUIUpdateTime >= 0.1 || passIdx == totalPasses {
                            lastUIUpdateTime = now
                            let elapsed = max(0.001, now - prefillStartTime)
                            let effectiveTokensProcessed = Double(P) * (Double(passIdx) / Double(totalPasses))
                            let promptSpeed = effectiveTokensProcessed / elapsed
                            let dispLayerIdx = prefillLayerBase + passIdx
                            let pct = Int((Double(dispLayerIdx) / Double(max(1, prefillLayerTotal))) * 100)
                            let speedStr = promptSpeed >= 10 ? String(format: "%.0f", promptSpeed) : String(format: "%.1f", promptSpeed)
                            let prefillStr = "Ingesting prompt: Layer \(dispLayerIdx)/\(prefillLayerTotal) (\(pct)%) • \(speedStr) tok/s"
                            let currentRss = WorkingSetManager.shared.effectiveResidentMemoryGB

                            prefillProgress?(prefillStr, promptSpeed, currentRss)
                        }
                    }

                    // Apply intermediate loop final norm if multiple loops exist
                    if loopIdx + 1 < totalLoops, let normTensor = normTensor, let normShardBuffer = normShardBuffer {
                        guard let loopNormCmd = commandQueue.makeCommandBuffer(),
                              let loopNormEnc = loopNormCmd.makeComputeCommandEncoder() else {
                            print("❌ [METAL] prefill: loop-norm command buffer allocation failed (chunk startPos=\(startPos), P=\(P))")
                            return false
                        }

                        var nOff = normOffset
                        var epsVal = eps
                        let isFinalF16 = (normTensor.dtype.contains("F16") || normTensor.dtype.contains("HALF") || normTensor.dtype.contains("FLOAT16")) && !normTensor.dtype.contains("BF16") && !normTensor.dtype.contains("BFLOAT")
                        let finalNormPipe: MTLComputePipelineState
                        if isRMSNormOffset {
                            finalNormPipe = (isFinalF16 && rmsnormOffsetF16Pipeline != nil) ? rmsnormOffsetF16Pipeline! : (rmsnormOffsetPipeline ?? rmsnormPipeline)
                        } else {
                            finalNormPipe = (isFinalF16 && rmsnormF16Pipeline != nil) ? rmsnormF16Pipeline! : rmsnormPipeline
                        }

                        loopNormEnc.setComputePipelineState(finalNormPipe)
                        loopNormEnc.setBuffer(currHBuf, offset: 0, index: 0)
                        loopNormEnc.setBuffer(normShardBuffer, offset: 0, index: 1)
                        loopNormEnc.setBuffer(nextHBuf, offset: 0, index: 2)
                        loopNormEnc.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                        loopNormEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                        loopNormEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                        loopNormEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                        loopNormEnc.dispatchThreadgroups(MTLSize(width: P, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))

                        loopNormEnc.endEncoding()
                        loopNormCmd.commit()
                        loopNormCmd.waitUntilCompleted()
                        if let lnErr = loopNormCmd.error {
                            print("❌ [METAL ERROR] prefill loopNormCmd aborted: \(lnErr)")
                            return false
                        }

                        let tmp = currHBuf
                        currHBuf = nextHBuf
                        nextHBuf = tmp
                    }
                }

                if hasHc, let ext0 = extractStream0Pipeline {
                    guard let extCmd = commandQueue.makeCommandBuffer(),
                          let extEnc = extCmd.makeComputeCommandEncoder() else {
                        print("❌ [METAL] prefill: HC extract command buffer allocation failed")
                        return false
                    }
                    let lastHcTokenOffset = (P - 1) * 4 * Int(hiddenDim) * MemoryLayout<Float>.stride
                    extEnc.setComputePipelineState(ext0)
                    extEnc.setBuffer(hcStreamsBuffer_all, offset: lastHcTokenOffset, index: 0)
                    extEnc.setBuffer(hCurrBuffer, offset: 0, index: 1)
                    extEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 2)
                    extEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), ext0.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    extEnc.endEncoding()
                    extCmd.commit()
                    extCmd.waitUntilCompleted()
                    if let extErr = extCmd.error {
                        print("❌ [METAL ERROR] prefill extCmd aborted: \(extErr)")
                        return false
                    }
                } else {
                    // Copy final prompt token (P - 1)'s hidden state into hCurrBuffer so generation continues seamlessly
                    let lastTokenOffset = (P - 1) * Int(hiddenDim) * MemoryLayout<Float>.stride
                    guard let copyCmd = commandQueue.makeCommandBuffer(),
                          let blit = copyCmd.makeBlitCommandEncoder() else {
                        print("❌ [METAL] prefill: buffer-copy command allocation failed")
                        return false
                    }
                    blit.copy(from: currHBuf, sourceOffset: lastTokenOffset, to: hCurrBuffer, destinationOffset: 0, size: Int(hiddenDim) * MemoryLayout<Float>.stride)
                    blit.endEncoding()
                    copyCmd.commit()
                    copyCmd.waitUntilCompleted()
                    if let cpErr = copyCmd.error {
                        print("❌ [METAL ERROR] prefill copyCmd aborted: \(cpErr)")
                        return false
                    }
                }

                if prefillDiagTotal > 0 {
                    let elapsed = max(0.001, CFAbsoluteTimeGetCurrent() - prefillStartTime)
                    print("[PREFILL-DIAG] Summary: P=\(P) startPos=\(startPos) experts=\(prefillDiagTotal) layers=\(prefillDiagLayers) PhaseA=\(String(format: "%.2f", prefillDiagPhaseAMs / 1000.0))s ioWait=\(String(format: "%.2f", prefillDiagIoMs / 1000.0))s gpuWait=\(String(format: "%.2f", prefillDiagGpuMs / 1000.0))s other=\(String(format: "%.2f", max(0, elapsed - prefillDiagPhaseAMs / 1000.0 - prefillDiagIoMs / 1000.0 - prefillDiagGpuMs / 1000.0)))s hitRate=\(String(format: "%.0f", Double(prefillDiagHits) * 100.0 / Double(max(1, prefillDiagTotal))))% prefillTotal=\(String(format: "%.2f", elapsed))s")
                }

                return true
            }

            // Allocate JetSpec Staging Buffers for Tree Drafting & Verification
            // Speculative tree forward pass is enabled for resident in-memory standard attention
            // models and for Gemma 4 (hybrid sliding/global attention MoE, verified by the
            // dedicated Gemma tree kernels). Linear Recurrence / Gated DeltaNet hybrid models
            // (maintaining continuous causal convolution and O(1) recurrent states) execute via
            // the direct single-token path for maximum throughput and state integrity.
            let hasLinearRecurrence = cachedLayers.contains { $0.attentionType == .linearAttention }
            let effectiveJetSpec = isJetSpecEligible(
                jetSpecEnabled: jetSpecEnabled,
                hasLinearRecurrence: hasLinearRecurrence,
                isSparkModel: modelConfig?.isSparkModel == true,
                kvPrecision: KVCacheManager.shared.activePrecision
            )
            let effectiveInterDim = modelConfig?.intermediateSize ?? Int(cachedLayers.first?.intermediateDim ?? 14336)
            let effectiveQkvDim = Int(max(hiddenDim * 2, 8192))
            let effectiveZDim = Int(max(hiddenDim * 2, 8192))
            let effectiveTopK = max(1, Int(modelConfig?.effectiveNumExpertsPerTok ?? 8))
            let linLayers = cachedLayers.filter { $0.attentionType == .linearAttention }.count
            let linValHeads = max(32, modelConfig?.effectiveLinearNumValueHeads ?? 32)
            let jetspecStaging = effectiveJetSpec ? InferenceEngine.shared.allocateJetSpecBuffers(
                device: device,
                maxNodes: 8,
                vocabSize: Int(vocabSize),
                hiddenDim: Int(hiddenDim),
                intermediateDim: effectiveInterDim,
                maxQkvDim: effectiveQkvDim,
                maxZDim: effectiveZDim,
                kvStride: Int(kvStride),
                topK: effectiveTopK,
                maxLinearLayers: max(1, linLayers),
                linValHeads: linValHeads
            ) : nil

            // Multi-Node Parallel Tree Forward Pass (Target Backbone Execution)
            func runJetSpecTreeForward(treeMask: JetSpecTreeMask, nodeScores: [Float], step: UInt32) -> Bool {
                guard let jb = jetspecStaging else { return false }
                let N = Int(treeMask.nodeCount)
                if N == 0 { return false }
                let maxTreeDepth = Int(treeMask.depths.max() ?? 0)

                let getRmsNormPipe = { (dtype: String) -> MTLComputePipelineState in
                    let isF16 = (dtype.contains("F16") || dtype.contains("HALF") || dtype.contains("FLOAT16")) && !dtype.contains("BF16") && !dtype.contains("BFLOAT")
                    if isRMSNormOffset {
                        return (isF16 && rmsnormOffsetF16Pipeline != nil) ? rmsnormOffsetF16Pipeline! : (rmsnormOffsetPipeline ?? rmsnormPipeline)
                    } else {
                        return (isF16 && rmsnormF16Pipeline != nil) ? rmsnormF16Pipeline! : rmsnormPipeline
                    }
                }

                // 1. Stage Tree Metadata in JetSpec buffers
                let maskPtr = jb.treeMaskBuffer.contents().bindMemory(to: Float.self, capacity: N * N)
                for i in 0..<(N * N) {
                    maskPtr[i] = i < treeMask.mask.count ? treeMask.mask[i] : -1e9
                }

                let depthsPtr = jb.depthsBuffer.contents().bindMemory(to: UInt32.self, capacity: N)
                for i in 0..<N {
                    depthsPtr[i] = i < treeMask.depths.count ? treeMask.depths[i] : 0
                }

                let parentsPtr = jb.parentIndicesBuffer.contents().bindMemory(to: UInt32.self, capacity: N)
                for i in 0..<N {
                    parentsPtr[i] = i < treeMask.parentIndices.count ? treeMask.parentIndices[i] : 0
                }

                let tokensPtr = jb.candidateTokensBuffer.contents().bindMemory(to: UInt32.self, capacity: N)
                for i in 0..<N {
                    tokensPtr[i] = i < treeMask.tokenIds.count ? treeMask.tokenIds[i] : 0
                }

                guard var activeCmd = commandQueue.makeCommandBuffer() else { return false }

                // 2. Embed all N candidate tree tokens into jb.treeHiddenBuffer
                guard let embedEnc = activeCmd.makeComputeCommandEncoder() else { return false }
                if let rawBaseBuffer = buffers[embedWeight.shardIndex] {
                    var baseOff = embedWeight.offsetStart
                    var hDim = hiddenDim
                    let hasEmbedScale = (embedScale != nil)
                    let hasEmbedBias = (embedBias != nil)
                    let isEmbedMXFP8 = hasEmbedScale && (embedScale!.dtype.contains("U8") || embedScale!.dtype.contains("UINT8") || (!hasEmbedBias && (embedWeight.dtype.contains("U32") || embedWeight.dtype.contains("U8") || embedWeight.dtype.contains("FP8")) && (embedScale!.offsetEnd - embedScale!.offsetStart) >= UInt64(hiddenDim / 32)))
                    let isEmbedAffine = (hasEmbedBias || embedWeight.dtype.contains("Q4") || embedWeight.dtype.contains("Q8") || hasEmbedScale) && !isEmbedMXFP8

                    if isEmbedMXFP8, let embedMXFP8Pipe = embedMXFP8Pipeline,
                       let embedScaleRaw = buffers[embedScale!.shardIndex] {
                        var wOffset = embedOffset
                        var sOffset = embedScale!.offsetStart
                        embedEnc.setComputePipelineState(embedMXFP8Pipe)
                        embedEnc.setBuffer(rawBaseBuffer, offset: 0, index: 0)
                        embedEnc.setBuffer(jb.candidateTokensBuffer, offset: 0, index: 1)
                        embedEnc.setBuffer(jb.treeHiddenBuffer, offset: 0, index: 2)
                        embedEnc.setBuffer(embedScaleRaw, offset: 0, index: 3)
                        embedEnc.setBytes(&wOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                        embedEnc.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                        embedEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 6)
                        embedEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), embedMXFP8Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    } else if isEmbedAffine,
                              let embedScaleRaw = (embedScale != nil) ? buffers[embedScale!.shardIndex] : nil,
                              let embedBiasRaw = (embedBias != nil) ? buffers[embedBias!.shardIndex] : nil {
                        var wOffset = embedOffset
                        var sOffset = embedScale!.offsetStart
                        var bOffset = embedBias!.offsetStart
                        var grpSize: UInt32 = 64
                        let is8Bit = (embedWeight.offsetEnd - embedWeight.offsetStart) >= (UInt64(vocabSize) * UInt64(hiddenDim) * 3) / 4

                        for n in 0..<N {
                            var tok = treeMask.tokenIds[n]
                            let outOffset = n * Int(hiddenDim) * MemoryLayout<Float>.stride
                            if is8Bit, let embedQ8Pipe = embedQ8Pipeline {
                                embedEnc.setComputePipelineState(embedQ8Pipe)
                                embedEnc.setBuffer(rawBaseBuffer, offset: 0, index: 0)
                                embedEnc.setBuffer(embedScaleRaw, offset: 0, index: 1)
                                embedEnc.setBuffer(embedBiasRaw, offset: 0, index: 2)
                                embedEnc.setBuffer(jb.treeHiddenBuffer, offset: outOffset, index: 3)
                                embedEnc.setBytes(&tok, length: MemoryLayout<UInt32>.stride, index: 4)
                                embedEnc.setBytes(&wOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                                embedEnc.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                                embedEnc.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                                embedEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 8)
                                embedEnc.setBytes(&grpSize, length: MemoryLayout<UInt32>.stride, index: 9)
                                embedEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), embedQ8Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            } else if let embedQ4Pipe = embedQ4Pipeline {
                                embedEnc.setComputePipelineState(embedQ4Pipe)
                                embedEnc.setBuffer(rawBaseBuffer, offset: 0, index: 0)
                                embedEnc.setBuffer(embedScaleRaw, offset: 0, index: 1)
                                embedEnc.setBuffer(embedBiasRaw, offset: 0, index: 2)
                                embedEnc.setBuffer(jb.treeHiddenBuffer, offset: outOffset, index: 3)
                                embedEnc.setBytes(&tok, length: MemoryLayout<UInt32>.stride, index: 4)
                                embedEnc.setBytes(&wOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                                embedEnc.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                                embedEnc.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                                embedEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 8)
                                embedEnc.setBytes(&grpSize, length: MemoryLayout<UInt32>.stride, index: 9)
                                embedEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), embedQ4Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            }
                        }
                    } else {
                        let embedPipe = embedPipeline
                        var wOffset = embedOffset
                        var tokCount: UInt32 = UInt32(N)
                        embedEnc.setComputePipelineState(embedPipe)
                        embedEnc.setBuffer(rawBaseBuffer, offset: 0, index: 0)
                        embedEnc.setBuffer(jb.candidateTokensBuffer, offset: 0, index: 1)
                        embedEnc.setBuffer(jb.treeHiddenBuffer, offset: 0, index: 2)
                        embedEnc.setBytes(&wOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                        embedEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                        embedEnc.setBytes(&tokCount, length: MemoryLayout<UInt32>.stride, index: 5)
                        embedEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), embedPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    }
                }
                embedEnc.endEncoding()

                // 3. Forward Pass through Backbone Layers
                for loopIdx in 0..<totalLoops {
                    for l in 0..<actualLayers {
                        if Task.isCancelled { return false }
                        let layer = cachedLayers[l]
                        // ================= Gemma 4 hybrid-attention MoE decoder layer =================
                        // Verifies all N tree nodes in one pass. Gemma differs from the generic tree
                        // path in every op: per-layer head dims (sliding 256 / global 512), per-head
                        // q/k norms, value norm, proportional RoPE, a sliding window, a dual
                        // dense+MoE FFN (GeGLU), and the per-layer output scalar.
                        if isGemma4 {
                            let gSliding = layer.isSlidingAttention
                            let gHeadDim: UInt32 = gSliding ? headDim : max(headDim, globalHeadDim)
                            let gNumKv: UInt32 = gSliding ? numKvHeads : max(1, numGlobalKvHeads)
                            let gQDim: UInt32 = numHeads * gHeadDim
                            let gKvDim: UInt32 = gNumKv * gHeadDim
                            let gRmsScale: Float = 1.0 / Float(Double(hiddenDim).squareRoot())
                            let gTopK = Int(modelConfig?.effectiveNumExpertsPerTok ?? 8)

                            guard let gCmd = commandQueue.makeCommandBuffer(),
                                  let gEnc = gCmd.makeComputeCommandEncoder() else { return false }

                            // 1. pre-attention RMSNorm -> xNorm1
                            if let norm1 = layer.norm1Tensor, let norm1Raw = buffers[norm1.shardIndex] {
                                var nOff = norm1.offsetStart
                                var hD = hiddenDim
                                var epsVal = eps
                                gEnc.setComputePipelineState(rmsnormPipeline)
                                gEnc.setBuffer(jb.treeHiddenBuffer, offset: 0, index: 0)
                                gEnc.setBuffer(norm1Raw, offset: 0, index: 1)
                                gEnc.setBuffer(jb.treeXNorm1Buffer, offset: 0, index: 2)
                                gEnc.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                gEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                                gEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                                gEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                                gEnc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                                gEnc.memoryBarrier(scope: .buffers)
                            }

                            // 2. Q/K/V projections (batched over N nodes). Global layers reuse the key
                            // projection as the value (attention_k_eq_v).
                            dispatchLinear(enc: gEnc, weight: layer.qProjTensor, scale: layer.qScaleTensor, bias: layer.qBiasTensor, inBuf: jb.treeXNorm1Buffer, outBuf: jb.treeQGateBuffer, inDim: hiddenDim, outDim: gQDim, batchSize: N)
                            dispatchLinear(enc: gEnc, weight: layer.kProjTensor, scale: layer.kScaleTensor, bias: layer.kBiasTensor, inBuf: jb.treeXNorm1Buffer, outBuf: jb.treeKVectorBuffer, inDim: hiddenDim, outDim: gKvDim, batchSize: N)
                            if gSliding {
                                dispatchLinear(enc: gEnc, weight: layer.vProjTensor, scale: layer.vScaleTensor, bias: layer.vBiasTensor, inBuf: jb.treeXNorm1Buffer, outBuf: jb.treeVVectorBuffer, inDim: hiddenDim, outDim: gKvDim, batchSize: N)
                            }
                            gEnc.memoryBarrier(scope: .buffers)

                            // 3. Value norm (sliding: v_proj output; global: raw key output), no scale.
                            if let vNormPipe = rmsnormNoScalePipeline {
                                var vDim = gHeadDim
                                var vEps = eps
                                gEnc.setComputePipelineState(vNormPipe)
                                let vInBuf = gSliding ? jb.treeVVectorBuffer : jb.treeKVectorBuffer
                                for r in 0..<N {
                                    let base = r * Int(gKvDim) * MemoryLayout<Float>.stride
                                    gEnc.setBuffer(vInBuf, offset: base, index: 0)
                                    gEnc.setBuffer(jb.treeVVectorBuffer, offset: base, index: 1)
                                    gEnc.setBytes(&vDim, length: MemoryLayout<UInt32>.stride, index: 2)
                                    gEnc.setBytes(&vEps, length: MemoryLayout<Float>.stride, index: 3)
                                    gEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                                    gEnc.dispatchThreadgroups(MTLSize(width: Int(gNumKv), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(gHeadDim)), height: 1, depth: 1))
                                }
                                gEnc.memoryBarrier(scope: .buffers)
                            }

                            // 4. Per-head Q/K RMSNorm (with scale), batched over N nodes.
                            if let qNorm = layer.qNormTensor, let qNormRaw = buffers[qNorm.shardIndex], let headNormPipe = headRmsnormPipeline {
                                var qNormOff = qNorm.offsetStart
                                var nQ = numHeads
                                var hD = gHeadDim
                                var hStride = gHeadDim
                                var epsVal = eps
                                gEnc.setComputePipelineState(headNormPipe)
                                gEnc.setBuffer(jb.treeQGateBuffer, offset: 0, index: 0)
                                gEnc.setBuffer(qNormRaw, offset: 0, index: 1)
                                gEnc.setBytes(&qNormOff, length: MemoryLayout<UInt64>.stride, index: 2)
                                gEnc.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 3)
                                gEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                                gEnc.setBytes(&hStride, length: MemoryLayout<UInt32>.stride, index: 5)
                                gEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 6)
                                gEnc.dispatchThreadgroups(MTLSize(width: Int(numHeads), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                gEnc.memoryBarrier(scope: .buffers)
                            }
                            if let kNorm = layer.kNormTensor, let kNormRaw = buffers[kNorm.shardIndex], let headNormPipe = headRmsnormPipeline {
                                var kNormOff = kNorm.offsetStart
                                var nK = gNumKv
                                var hD = gHeadDim
                                var hStride = gHeadDim
                                var epsVal = eps
                                gEnc.setComputePipelineState(headNormPipe)
                                gEnc.setBuffer(jb.treeKVectorBuffer, offset: 0, index: 0)
                                gEnc.setBuffer(kNormRaw, offset: 0, index: 1)
                                gEnc.setBytes(&kNormOff, length: MemoryLayout<UInt64>.stride, index: 2)
                                gEnc.setBytes(&nK, length: MemoryLayout<UInt32>.stride, index: 3)
                                gEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                                gEnc.setBytes(&hStride, length: MemoryLayout<UInt32>.stride, index: 5)
                                gEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 6)
                                gEnc.dispatchThreadgroups(MTLSize(width: Int(gNumKv), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                gEnc.memoryBarrier(scope: .buffers)
                            }

                            // 5. Tree RoPE: position = step + depth[node]. Sliding layers use standard
                            // RoPE over the full head dim; global layers use proportional RoPE.
                            let gRopePipe = gSliding ? InferenceEngine.shared.applyRopeTreePipeline : InferenceEngine.shared.applyRopeTreeProportionalPipeline
                            if let ropePipe = gRopePipe {
                                var tPos = step
                                var nQ = numHeads
                                var nK = gNumKv
                                var hD = gHeadDim
                                var rD = UInt32(modelConfig?.effectiveRotaryDim(layerIndex: l, headDim: Int(gHeadDim)) ?? Int(rotaryDim))
                                var qStr = gHeadDim
                                var kStr = gHeadDim
                                var theta = modelConfig?.effectiveRopeTheta(layerIndex: l) ?? thetaVal
                                gEnc.setComputePipelineState(ropePipe)
                                gEnc.setBuffer(jb.treeQGateBuffer, offset: 0, index: 0)
                                gEnc.setBuffer(jb.depthsBuffer, offset: 0, index: 1)
                                gEnc.setBytes(&tPos, length: MemoryLayout<UInt32>.stride, index: 2)
                                gEnc.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 3)
                                gEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                                gEnc.setBytes(&rD, length: MemoryLayout<UInt32>.stride, index: 5)
                                gEnc.setBytes(&qStr, length: MemoryLayout<UInt32>.stride, index: 6)
                                gEnc.setBytes(&theta, length: MemoryLayout<Float>.stride, index: 7)
                                gEnc.dispatchThreads(MTLSize(width: Int(numHeads), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), ropePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                gEnc.setBuffer(jb.treeKVectorBuffer, offset: 0, index: 0)
                                gEnc.setBytes(&nK, length: MemoryLayout<UInt32>.stride, index: 3)
                                gEnc.setBytes(&kStr, length: MemoryLayout<UInt32>.stride, index: 6)
                                gEnc.dispatchThreads(MTLSize(width: Int(gNumKv), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(gNumKv), ropePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                gEnc.memoryBarrier(scope: .buffers)
                            }

                            // 6. KV store + tree-causal attention. Tree node k occupies KV slot
                            // (step + k); only its RoPE position uses step + depth[k].
                            if let kCache = KVCacheManager.shared.kCacheBuffer,
                               let vCache = KVCacheManager.shared.vCacheBuffer {
                                let slotIndex = (loopIdx * actualLayers) + layer.fullAttnIndex
                                let kvMgr = KVCacheManager.shared
                                let prec = kvMgr.activePrecision
                                let layerByteOffset = kvMgr.kvSlotByteOffset(slot: slotIndex, stride: Int(gemmaKvStride), elementBytes: prec.bytesPerElement)
                                var basePos = step
                                var nKv = gNumKv
                                var hD = gHeadDim
                                var cacheStride = gemmaKvStride
                                // Sliding layers address a ring-sized slot region; tree
                                // node k's ring slot is (step + k) % ringLen (N <= ring
                                // slack, so the scratch never lands on a live window row).
                                var ringLen: UInt32 = kvMgr.isRingSlot(slotIndex) ? UInt32(kvMgr.slidingRingLen) : 0

                                if prec == .fp16, let storePipe = storeKvGemmaF16Pipeline {
                                    gEnc.setComputePipelineState(storePipe)
                                    gEnc.setBuffer(jb.treeKVectorBuffer, offset: 0, index: 0)
                                    gEnc.setBuffer(jb.treeVVectorBuffer, offset: 0, index: 1)
                                    gEnc.setBuffer(kCache, offset: layerByteOffset, index: 2)
                                    gEnc.setBuffer(vCache, offset: layerByteOffset, index: 3)
                                    gEnc.setBytes(&basePos, length: MemoryLayout<UInt32>.stride, index: 4)
                                    gEnc.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 5)
                                    gEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 6)
                                    gEnc.setBytes(&cacheStride, length: MemoryLayout<UInt32>.stride, index: 7)
                                    gEnc.setBytes(&ringLen, length: MemoryLayout<UInt32>.stride, index: 8)
                                    gEnc.dispatchThreads(MTLSize(width: Int(gKvDim), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(gKvDim), storePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    gEnc.memoryBarrier(scope: .buffers)
                                    kvMgr.noteRingStores(range: Int(step)..<Int(step) + N)
                                } else if let storePipe = storeKvGemmaPipeline {
                                    gEnc.setComputePipelineState(storePipe)
                                    gEnc.setBuffer(jb.treeKVectorBuffer, offset: 0, index: 0)
                                    gEnc.setBuffer(jb.treeVVectorBuffer, offset: 0, index: 1)
                                    gEnc.setBuffer(kCache, offset: layerByteOffset, index: 2)
                                    gEnc.setBuffer(vCache, offset: layerByteOffset, index: 3)
                                    gEnc.setBytes(&basePos, length: MemoryLayout<UInt32>.stride, index: 4)
                                    gEnc.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 5)
                                    gEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 6)
                                    gEnc.setBytes(&cacheStride, length: MemoryLayout<UInt32>.stride, index: 7)
                                    gEnc.setBytes(&ringLen, length: MemoryLayout<UInt32>.stride, index: 8)
                                    gEnc.dispatchThreads(MTLSize(width: Int(gKvDim), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(gKvDim), storePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    gEnc.memoryBarrier(scope: .buffers)
                                    kvMgr.noteRingStores(range: Int(step)..<Int(step) + N)
                                }

                                var pLen = step
                                var numNodesVal = UInt32(N)
                                var nQ = numHeads
                                var windowSize: UInt32 = gSliding ? UInt32(modelConfig?.effectiveSlidingWindow ?? 0) : 0
                                var scaling = gemmaScaling
                                let attnPipe: MTLComputePipelineState? = (prec == .fp16) ? gqaGemmaTreeVerifyF16Pipeline : gqaGemmaTreeVerifyPipeline
                                if let attnPipe {
                                    gEnc.setComputePipelineState(attnPipe)
                                    gEnc.setBuffer(jb.treeQGateBuffer, offset: 0, index: 0)
                                    gEnc.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                    gEnc.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                    gEnc.setBuffer(jb.treeMaskBuffer, offset: 0, index: 3)
                                    gEnc.setBuffer(jb.depthsBuffer, offset: 0, index: 4)
                                    gEnc.setBuffer(jb.treeAttnCtxBuffer, offset: 0, index: 5)
                                    gEnc.setBytes(&pLen, length: MemoryLayout<UInt32>.stride, index: 6)
                                    gEnc.setBytes(&numNodesVal, length: MemoryLayout<UInt32>.stride, index: 7)
                                    gEnc.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 8)
                                    gEnc.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 9)
                                    gEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 10)
                                    gEnc.setBytes(&windowSize, length: MemoryLayout<UInt32>.stride, index: 11)
                                    gEnc.setBytes(&scaling, length: MemoryLayout<Float>.stride, index: 12)
                                    gEnc.setBytes(&cacheStride, length: MemoryLayout<UInt32>.stride, index: 13)
                                    gEnc.setBytes(&ringLen, length: MemoryLayout<UInt32>.stride, index: 14)
                                    gEnc.dispatchThreadgroups(MTLSize(width: Int(numHeads), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    gEnc.memoryBarrier(scope: .buffers)
                                }
                            }

                            // 7. o_proj
                            dispatchLinear(enc: gEnc, weight: layer.oProjTensor, scale: layer.oScaleTensor, bias: layer.oBiasTensor, inBuf: jb.treeAttnCtxBuffer, outBuf: jb.treeAttnOutBuffer, inDim: gQDim, outDim: hiddenDim, batchSize: N)
                            gEnc.memoryBarrier(scope: .buffers)

                            // 8. hMid = hidden + post_attention_layernorm(attn)
                            if let norm2 = layer.norm2Tensor, let norm2Raw = buffers[norm2.shardIndex] {
                                var nOff = norm2.offsetStart
                                var hD = hiddenDim
                                var epsVal = eps
                                gEnc.setComputePipelineState(rmsnormPipeline)
                                gEnc.setBuffer(jb.treeAttnOutBuffer, offset: 0, index: 0)
                                gEnc.setBuffer(norm2Raw, offset: 0, index: 1)
                                gEnc.setBuffer(jb.treeXNorm2Buffer, offset: 0, index: 2)
                                gEnc.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                gEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                                gEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                                gEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                                gEnc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                                gEnc.memoryBarrier(scope: .buffers)
                            }
                            gEnc.setComputePipelineState(addPipeline)
                            gEnc.setBuffer(jb.treeHiddenBuffer, offset: 0, index: 0)
                            gEnc.setBuffer(jb.treeXNorm2Buffer, offset: 0, index: 1)
                            gEnc.setBuffer(jb.treeHMidBuffer, offset: 0, index: 2)
                            var hDimAdd = hiddenDim
                            gEnc.setBytes(&hDimAdd, length: MemoryLayout<UInt32>.stride, index: 3)
                            gEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            gEnc.memoryBarrier(scope: .buffers)

                            // 9. pre_feedforward_layernorm(hMid) -> xNorm2
                            if let preFfn = layer.preFfnNormTensor, let preFfnRaw = buffers[preFfn.shardIndex] {
                                var nOff = preFfn.offsetStart
                                var hD = hiddenDim
                                var epsVal = eps
                                gEnc.setComputePipelineState(rmsnormPipeline)
                                gEnc.setBuffer(jb.treeHMidBuffer, offset: 0, index: 0)
                                gEnc.setBuffer(preFfnRaw, offset: 0, index: 1)
                                gEnc.setBuffer(jb.treeXNorm2Buffer, offset: 0, index: 2)
                                gEnc.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                gEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                                gEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                                gEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                                gEnc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                                gEnc.memoryBarrier(scope: .buffers)
                            }

                            // 10. Dense shared MLP (GeGLU) -> hMlp; postFfnNorm1 in place
                            gEnc.setComputePipelineState(clearPipeline)
                            gEnc.setBuffer(jb.treeHMlpBuffer, offset: 0, index: 0)
                            gEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            gEnc.memoryBarrier(scope: .buffers)
                            if let dG = layer.denseGateWeight, let dU = layer.denseUpWeight, let dD = layer.denseDownWeight,
                               let dGRaw = buffers[dG.shardIndex], let dURaw = buffers[dU.shardIndex], let dDRaw = buffers[dD.shardIndex],
                               let geluPipe = bf16GeluGateUpSimdPipeline, let downPipe = bf16DownSimdPipeline {
                                var gOff = dG.offsetStart
                                var uOff = dU.offsetStart
                                var dOff = dD.offsetStart
                                var hDimVal = hiddenDim
                                var interDimVal = layer.intermediateDim
                                var oneVal: Float = 1.0
                                for r in 0..<N {
                                    let inOff = r * Int(hiddenDim) * MemoryLayout<Float>.stride
                                    let interOff = r * Int(layer.intermediateDim) * MemoryLayout<Float>.stride
                                    let accumOff = r * Int(hiddenDim) * MemoryLayout<Float>.stride
                                    gEnc.setComputePipelineState(geluPipe)
                                    gEnc.setBuffer(dGRaw, offset: 0, index: 0)
                                    gEnc.setBuffer(dURaw, offset: 0, index: 1)
                                    gEnc.setBuffer(jb.treeXNorm2Buffer, offset: inOff, index: 2)
                                    gEnc.setBuffer(jb.treeInterBuffer, offset: interOff, index: 3)
                                    gEnc.setBytes(&gOff, length: MemoryLayout<UInt64>.stride, index: 4)
                                    gEnc.setBytes(&uOff, length: MemoryLayout<UInt64>.stride, index: 5)
                                    gEnc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 6)
                                    gEnc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 7)
                                    gEnc.dispatchThreadgroups(MTLSize(width: Int(layer.intermediateDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    gEnc.memoryBarrier(scope: .buffers)
                                    gEnc.setComputePipelineState(downPipe)
                                    gEnc.setBuffer(dDRaw, offset: 0, index: 0)
                                    gEnc.setBuffer(jb.treeInterBuffer, offset: interOff, index: 1)
                                    gEnc.setBuffer(jb.treeHMlpBuffer, offset: accumOff, index: 2)
                                    gEnc.setBytes(&dOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                    gEnc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 4)
                                    gEnc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 5)
                                    gEnc.setBytes(&oneVal, length: MemoryLayout<Float>.stride, index: 6)
                                    gEnc.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    gEnc.memoryBarrier(scope: .buffers)
                                }
                            }
                            if let pf1 = layer.postFfnNorm1Tensor, let pf1Raw = buffers[pf1.shardIndex] {
                                var nOff = pf1.offsetStart
                                var hD = hiddenDim
                                var epsVal = eps
                                gEnc.setComputePipelineState(rmsnormPipeline)
                                gEnc.setBuffer(jb.treeHMlpBuffer, offset: 0, index: 0)
                                gEnc.setBuffer(pf1Raw, offset: 0, index: 1)
                                gEnc.setBuffer(jb.treeHMlpBuffer, offset: 0, index: 2)
                                gEnc.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                gEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                                gEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                                gEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                                gEnc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                                gEnc.memoryBarrier(scope: .buffers)
                            }

                            // 11. Router: rmsnorm_no_scale(hMid) * router.scale * hidden^-0.5 -> top-k
                            if let vNormPipe = rmsnormNoScalePipeline {
                                var dimV = hiddenDim
                                var eV = eps
                                gEnc.setComputePipelineState(vNormPipe)
                                gEnc.setBuffer(jb.treeHMidBuffer, offset: 0, index: 0)
                                gEnc.setBuffer(jb.treeAttnOutBuffer, offset: 0, index: 1)
                                gEnc.setBytes(&dimV, length: MemoryLayout<UInt32>.stride, index: 2)
                                gEnc.setBytes(&eV, length: MemoryLayout<Float>.stride, index: 3)
                                gEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                                gEnc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                                gEnc.memoryBarrier(scope: .buffers)
                            }
                            if let rsv = layer.routerScaleVec, let rsvRaw = buffers[rsv.shardIndex], let mulPipe = mulByBf16VectorPipeline {
                                var rsvOff = rsv.offsetStart
                                var dimV = hiddenDim
                                var sc = gRmsScale
                                gEnc.setComputePipelineState(mulPipe)
                                for r in 0..<N {
                                    let off = r * Int(hiddenDim) * MemoryLayout<Float>.stride
                                    gEnc.setBuffer(jb.treeAttnOutBuffer, offset: off, index: 0)
                                    gEnc.setBuffer(rsvRaw, offset: 0, index: 1)
                                    gEnc.setBuffer(jb.treeZGateBuffer, offset: off, index: 2)
                                    gEnc.setBytes(&rsvOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                    gEnc.setBytes(&dimV, length: MemoryLayout<UInt32>.stride, index: 4)
                                    gEnc.setBytes(&sc, length: MemoryLayout<Float>.stride, index: 5)
                                    gEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), mulPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                }
                                gEnc.memoryBarrier(scope: .buffers)
                            }
                            if let routerTensor = layer.routerTensor, let routerRaw = buffers[routerTensor.shardIndex] {
                                var rOffset = routerTensor.offsetStart
                                var nExp = numExperts
                                var kVal: UInt32 = UInt32(gTopK)
                                var hDimR = hiddenDim
                                gEnc.setComputePipelineState(routerPipeline)
                                for r in 0..<N {
                                    let tOff = r * Int(hiddenDim) * MemoryLayout<Float>.stride
                                    let rIndexOffset = r * gTopK * MemoryLayout<UInt32>.stride
                                    let rWeightOffset = r * gTopK * MemoryLayout<Float>.stride
                                    gEnc.setBuffer(routerRaw, offset: 0, index: 0)
                                    gEnc.setBuffer(jb.treeZGateBuffer, offset: tOff, index: 1)
                                    gEnc.setBuffer(jb.treeRouterIndicesBuffer, offset: rIndexOffset, index: 2)
                                    gEnc.setBuffer(jb.treeRouterWeightsBuffer, offset: rWeightOffset, index: 3)
                                    gEnc.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                                    gEnc.setBytes(&hDimR, length: MemoryLayout<UInt32>.stride, index: 5)
                                    gEnc.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 6)
                                    gEnc.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 7)
                                    gEnc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: Int(numExperts), height: 1, depth: 1))
                                }
                                gEnc.memoryBarrier(scope: .buffers)
                                if let pes = layer.routerPerExpertScale, let pesRaw = buffers[pes.shardIndex], let pesPipe = perExpertScalePipeline {
                                    var pesOff = pes.offsetStart
                                    var kTop = kVal
                                    gEnc.setComputePipelineState(pesPipe)
                                    for r in 0..<N {
                                        let rIndexOffset = r * gTopK * MemoryLayout<UInt32>.stride
                                        let rWeightOffset = r * gTopK * MemoryLayout<Float>.stride
                                        gEnc.setBuffer(jb.treeRouterWeightsBuffer, offset: rWeightOffset, index: 0)
                                        gEnc.setBuffer(jb.treeRouterIndicesBuffer, offset: rIndexOffset, index: 1)
                                        gEnc.setBuffer(pesRaw, offset: 0, index: 2)
                                        gEnc.setBytes(&pesOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                        gEnc.setBytes(&kTop, length: MemoryLayout<UInt32>.stride, index: 4)
                                        gEnc.dispatchThreads(MTLSize(width: Int(kVal), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(kVal), pesPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    }
                                    gEnc.memoryBarrier(scope: .buffers)
                                }
                            }

                            gEnc.endEncoding()
                            gCmd.commit()
                            gCmd.waitUntilCompleted()
                            if let gErr = gCmd.error {
                                print("❌ [METAL ERROR] gemma4 tree Phase A layer \(l): \(gErr)")
                                return false
                            }

                            // ---- Phase B: expert-major MoE over all tree nodes ----
                            let indAllPtr = jb.treeRouterIndicesBuffer.contents().bindMemory(to: UInt32.self, capacity: N * gTopK)
                            let wAllPtr = jb.treeRouterWeightsBuffer.contents().bindMemory(to: Float.self, capacity: N * gTopK)
                            var candidateExpertsPerNode: [[(id: Int, weight: Float)]] = []
                            var flatCandidateExperts: [UInt32] = []
                            var uniqueActiveExperts = Set<UInt32>()
                            for n in 0..<N {
                                var nodeExp: [(id: Int, weight: Float)] = []
                                for i in 0..<gTopK {
                                    let expId = indAllPtr[n * gTopK + i]
                                    let pk = wAllPtr[n * gTopK + i]
                                    flatCandidateExperts.append(expId)
                                    if pk > 0.00001 {
                                        nodeExp.append((id: Int(expId), weight: pk))
                                        uniqueActiveExperts.insert(expId)
                                    }
                                }
                                candidateExpertsPerNode.append(nodeExp)
                            }

                            // Prune the tree to fit the expert budget (and the staging buffer).
                            let expertStagingCap = max(1, expertStagingSize / max(1, Int(loadedLayout?.expert_size ?? 1)))
                            let moeExpertCap = max(1, min(jetSpecMaxExpertCap, expertStagingCap))
                            var activeNodeIndices = Array(0..<N)
                            if uniqueActiveExperts.count > moeExpertCap && N > 1 {
                                let prunedMask = pruneJetspecTreeMoe(
                                    treeTokens: treeMask.tokenIds,
                                    parentIndices: treeMask.parentIndices,
                                    draftScores: nodeScores,
                                    candidateExpertsFlat: flatCandidateExperts,
                                    expertsPerNode: UInt32(gTopK),
                                    maxUniqueExperts: UInt32(moeExpertCap)
                                )
                                var keptIndices: [Int] = []
                                var searchStart = 0
                                for tok in prunedMask.tokenIds {
                                    for origIdx in searchStart..<N {
                                        if treeMask.tokenIds[origIdx] == tok {
                                            keptIndices.append(origIdx)
                                            searchStart = origIdx + 1
                                            break
                                        }
                                    }
                                }
                                if !keptIndices.isEmpty { activeNodeIndices = keptIndices }
                            }

                            var expertTokenMap: [Int: [(nodeIdx: UInt32, weight: Float)]] = [:]
                            for n in activeNodeIndices {
                                for item in candidateExpertsPerNode[n] {
                                    expertTokenMap[item.id, default: []].append((nodeIdx: UInt32(n), weight: item.weight))
                                }
                            }
                            let uniqueActiveExpIds = Array(expertTokenMap.keys).sorted()

                            guard let gemmaMoeCmd = commandQueue.makeCommandBuffer(),
                                  let gemmaMoeEnc = gemmaMoeCmd.makeComputeCommandEncoder() else { return false }

                            // pre_feedforward_layernorm_2(hMid) -> xNorm2
                            if let pf2 = layer.preFfnNorm2Tensor, let pf2Raw = buffers[pf2.shardIndex] {
                                var nOff = pf2.offsetStart
                                var hD = hiddenDim
                                var epsVal = eps
                                gemmaMoeEnc.setComputePipelineState(rmsnormPipeline)
                                gemmaMoeEnc.setBuffer(jb.treeHMidBuffer, offset: 0, index: 0)
                                gemmaMoeEnc.setBuffer(pf2Raw, offset: 0, index: 1)
                                gemmaMoeEnc.setBuffer(jb.treeXNorm2Buffer, offset: 0, index: 2)
                                gemmaMoeEnc.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                gemmaMoeEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                                gemmaMoeEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                                gemmaMoeEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                                gemmaMoeEnc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                                gemmaMoeEnc.memoryBarrier(scope: .buffers)
                            }
                            // clear MoE accumulator (treeAttnOutBuffer, free after router)
                            gemmaMoeEnc.setComputePipelineState(clearPipeline)
                            gemmaMoeEnc.setBuffer(jb.treeAttnOutBuffer, offset: 0, index: 0)
                            gemmaMoeEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            gemmaMoeEnc.memoryBarrier(scope: .buffers)

                            if let packedDir = packedExpertsDir,
                               let fd = ExpertIOThreadPool.shared.getOrOpenLayerFD(layerIndex: l, packedExpertsDir: packedDir) {
                                let expertSize = Int(loadedLayout?.expert_size ?? 0)
                                if !uniqueActiveExpIds.isEmpty && expertSize > 0 && uniqueActiveExpIds.count <= expertStagingCap {
                                    let rawStagingPtr = expertStagingBuffer.contents()
                                    var tasks: [ExpertPreadTask] = []
                                    for (slot, expId) in uniqueActiveExpIds.enumerated() {
                                        tasks.append(ExpertPreadTask(fd: fd, dst: rawStagingPtr.advanced(by: slot * expertSize), offset: off_t(expId * expertSize), size: expertSize))
                                    }
                                    ExpertIOThreadPool.shared.dispatchSync(tasks: &tasks)

                                    let compUpW = loadedLayout?.components.first(where: { $0.name.contains("up_proj") && !$0.name.contains("scale") && !$0.name.contains("bias") })
                                    let compDownW = loadedLayout?.components.first(where: { $0.name.contains("down_proj") && !$0.name.contains("scale") && !$0.name.contains("bias") })
                                    let moeInterDim = Int(gemmaExpertInterDim)
                                    let upByteOffset = UInt64(moeInterDim) * UInt64(hiddenDim) * 2

                                    if let geluPipe = bf16GeluGateUpSimdPipeline, let downPipe = bf16DownSimdPipeline {
                                        var hDimVal = hiddenDim
                                        var interDimVal = UInt32(moeInterDim)
                                        var slotLookup: [Int: Int] = [:]
                                        for (slot, expId) in uniqueActiveExpIds.enumerated() { slotLookup[expId] = slot }
                                        for expId in uniqueActiveExpIds {
                                            guard let assignments = expertTokenMap[expId], let slot = slotLookup[expId], !assignments.isEmpty else { continue }
                                            let slotOffset = UInt64(slot * expertSize)
                                            var gOff = slotOffset + (compUpW?.offset ?? 0)
                                            var uOff = gOff + upByteOffset
                                            var dOff = slotOffset + (compDownW?.offset ?? 0)
                                            for item in assignments {
                                                let n = Int(item.nodeIdx)
                                                let inOff = n * Int(hiddenDim) * MemoryLayout<Float>.stride
                                                let interOff = n * moeInterDim * MemoryLayout<Float>.stride
                                                let accumOff = n * Int(hiddenDim) * MemoryLayout<Float>.stride
                                                var pkVal = item.weight
                                                gemmaMoeEnc.setComputePipelineState(geluPipe)
                                                gemmaMoeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 0)
                                                gemmaMoeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 1)
                                                gemmaMoeEnc.setBuffer(jb.treeXNorm2Buffer, offset: inOff, index: 2)
                                                gemmaMoeEnc.setBuffer(jb.treeInterBuffer, offset: interOff, index: 3)
                                                gemmaMoeEnc.setBytes(&gOff, length: MemoryLayout<UInt64>.stride, index: 4)
                                                gemmaMoeEnc.setBytes(&uOff, length: MemoryLayout<UInt64>.stride, index: 5)
                                                gemmaMoeEnc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 6)
                                                gemmaMoeEnc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 7)
                                                gemmaMoeEnc.dispatchThreadgroups(MTLSize(width: moeInterDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                                gemmaMoeEnc.memoryBarrier(scope: .buffers)
                                                gemmaMoeEnc.setComputePipelineState(downPipe)
                                                gemmaMoeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 0)
                                                gemmaMoeEnc.setBuffer(jb.treeInterBuffer, offset: interOff, index: 1)
                                                gemmaMoeEnc.setBuffer(jb.treeAttnOutBuffer, offset: accumOff, index: 2)
                                                gemmaMoeEnc.setBytes(&dOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                                gemmaMoeEnc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 4)
                                                gemmaMoeEnc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 5)
                                                gemmaMoeEnc.setBytes(&pkVal, length: MemoryLayout<Float>.stride, index: 6)
                                                gemmaMoeEnc.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                                gemmaMoeEnc.memoryBarrier(scope: .buffers)
                                            }
                                        }
                                    }
                                }
                            }

                            // post_feedforward_layernorm_2(moe) -> treeZGateBuffer
                            if let pf2 = layer.postFfnNorm2Tensor, let pf2Raw = buffers[pf2.shardIndex] {
                                var nOff = pf2.offsetStart
                                var hD = hiddenDim
                                var epsVal = eps
                                gemmaMoeEnc.setComputePipelineState(rmsnormPipeline)
                                gemmaMoeEnc.setBuffer(jb.treeAttnOutBuffer, offset: 0, index: 0)
                                gemmaMoeEnc.setBuffer(pf2Raw, offset: 0, index: 1)
                                gemmaMoeEnc.setBuffer(jb.treeZGateBuffer, offset: 0, index: 2)
                                gemmaMoeEnc.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                gemmaMoeEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                                gemmaMoeEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                                gemmaMoeEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                                gemmaMoeEnc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                                gemmaMoeEnc.memoryBarrier(scope: .buffers)
                            }
                            // hMlp = postFfnNorm1(dense) + postFfnNorm2(moe)
                            gemmaMoeEnc.setComputePipelineState(addPipeline)
                            gemmaMoeEnc.setBuffer(jb.treeHMlpBuffer, offset: 0, index: 0)
                            gemmaMoeEnc.setBuffer(jb.treeZGateBuffer, offset: 0, index: 1)
                            gemmaMoeEnc.setBuffer(jb.treeHMlpBuffer, offset: 0, index: 2)
                            var hDimA2 = hiddenDim
                            gemmaMoeEnc.setBytes(&hDimA2, length: MemoryLayout<UInt32>.stride, index: 3)
                            gemmaMoeEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            gemmaMoeEnc.memoryBarrier(scope: .buffers)

                            // post_feedforward_layernorm(combined) -> xNorm2
                            if let pf = layer.postFfnNormTensor, let pfRaw = buffers[pf.shardIndex] {
                                var nOff = pf.offsetStart
                                var hD = hiddenDim
                                var epsVal = eps
                                gemmaMoeEnc.setComputePipelineState(rmsnormPipeline)
                                gemmaMoeEnc.setBuffer(jb.treeHMlpBuffer, offset: 0, index: 0)
                                gemmaMoeEnc.setBuffer(pfRaw, offset: 0, index: 1)
                                gemmaMoeEnc.setBuffer(jb.treeXNorm2Buffer, offset: 0, index: 2)
                                gemmaMoeEnc.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                gemmaMoeEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                                gemmaMoeEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                                gemmaMoeEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                                gemmaMoeEnc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                                gemmaMoeEnc.memoryBarrier(scope: .buffers)
                            }
                            // hidden = hMid + xNorm2
                            gemmaMoeEnc.setComputePipelineState(addPipeline)
                            gemmaMoeEnc.setBuffer(jb.treeHMidBuffer, offset: 0, index: 0)
                            gemmaMoeEnc.setBuffer(jb.treeXNorm2Buffer, offset: 0, index: 1)
                            gemmaMoeEnc.setBuffer(jb.treeHiddenBuffer, offset: 0, index: 2)
                            var hDimF = hiddenDim
                            gemmaMoeEnc.setBytes(&hDimF, length: MemoryLayout<UInt32>.stride, index: 3)
                            gemmaMoeEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            gemmaMoeEnc.memoryBarrier(scope: .buffers)

                            // hidden *= layer_scalar
                            if let ls = layer.layerScalarTensor, let lsRaw = buffers[ls.shardIndex], let scalePipe = scaleVectorInplacePipeline {
                                var dimV = hiddenDim
                                let lsBits = lsRaw.contents().load(fromByteOffset: Int(ls.offsetStart), as: UInt16.self)
                                var scalarVal = Float(bitPattern: UInt32(lsBits) << 16)
                                gemmaMoeEnc.setComputePipelineState(scalePipe)
                                for r in 0..<N {
                                    let off = r * Int(hiddenDim) * MemoryLayout<Float>.stride
                                    gemmaMoeEnc.setBuffer(jb.treeHiddenBuffer, offset: off, index: 0)
                                    gemmaMoeEnc.setBytes(&dimV, length: MemoryLayout<UInt32>.stride, index: 1)
                                    gemmaMoeEnc.setBytes(&scalarVal, length: MemoryLayout<Float>.stride, index: 2)
                                    gemmaMoeEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), scalePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                }
                                gemmaMoeEnc.memoryBarrier(scope: .buffers)
                            }

                            gemmaMoeEnc.endEncoding()
                            gemmaMoeCmd.commit()
                            gemmaMoeCmd.waitUntilCompleted()
                            if let gErr = gemmaMoeCmd.error {
                                print("❌ [METAL ERROR] gemma4 tree Phase B layer \(l): \(gErr)")
                                return false
                            }
                            continue
                        }

                        guard let layerEnc = activeCmd.makeComputeCommandEncoder() else { return false }

                        // Step A: Pre-Attention RMSNorm 1
                        if let norm1 = layer.norm1Tensor, let normRaw = buffers[norm1.shardIndex] {
                            let rmsPipe = getRmsNormPipe(norm1.dtype)
                            var nOff = norm1.offsetStart
                            var hD = hiddenDim
                            var epsVal = eps
                            layerEnc.setComputePipelineState(rmsPipe)
                            layerEnc.setBuffer(jb.treeHiddenBuffer, offset: 0, index: 0)
                            layerEnc.setBuffer(normRaw, offset: 0, index: 1)
                            layerEnc.setBuffer(jb.treeXNorm1Buffer, offset: 0, index: 2)
                            layerEnc.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                            layerEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                            layerEnc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                            layerEnc.memoryBarrier(scope: .buffers)
                        }

                        // Step B: Attention (Full Attention or Linear Attention)
                        if layer.attentionType == .fullAttention {
                            let currentQDim = isStandardGqa ? (numHeads * headDim) : (numHeads * headDim * 2)
                            let currentKvDim = numKvHeads * headDim
                            let attnCtxDim = numHeads * headDim

                            // Q, K, V Projections for all N candidate tree nodes
                            dispatchLinear(enc: layerEnc, weight: layer.qProjTensor, scale: layer.qScaleTensor, bias: layer.qBiasTensor, inBuf: jb.treeXNorm1Buffer, outBuf: jb.treeQGateBuffer, inDim: hiddenDim, outDim: currentQDim, batchSize: N)
                            dispatchLinear(enc: layerEnc, weight: layer.kProjTensor, scale: layer.kScaleTensor, bias: layer.kBiasTensor, inBuf: jb.treeXNorm1Buffer, outBuf: jb.treeKVectorBuffer, inDim: hiddenDim, outDim: currentKvDim, batchSize: N)
                            dispatchLinear(enc: layerEnc, weight: layer.vProjTensor, scale: layer.vScaleTensor, bias: layer.vBiasTensor, inBuf: jb.treeXNorm1Buffer, outBuf: jb.treeVVectorBuffer, inDim: hiddenDim, outDim: currentKvDim, batchSize: N)
                            layerEnc.memoryBarrier(scope: .buffers)

                            // Q / K Normalization if present
                            if let qNorm = layer.qNormTensor, let qNormRaw = buffers[qNorm.shardIndex],
                               let headNormPipe = headRmsnormPipeline {
                                var qNormOff = qNorm.offsetStart
                                var nQ = numHeads
                                var hD = headDim
                                var hStride = isStandardGqa ? headDim : (headDim * 2)
                                var epsVal = eps
                                layerEnc.setComputePipelineState(headNormPipe)
                                layerEnc.setBuffer(jb.treeQGateBuffer, offset: 0, index: 0)
                                layerEnc.setBuffer(qNormRaw, offset: 0, index: 1)
                                layerEnc.setBytes(&qNormOff, length: MemoryLayout<UInt64>.stride, index: 2)
                                layerEnc.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc.setBytes(&hStride, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 6)
                                layerEnc.dispatchThreadgroups(MTLSize(width: Int(numHeads), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                layerEnc.memoryBarrier(scope: .buffers)
                            }
                            if let kNorm = layer.kNormTensor, let kNormRaw = buffers[kNorm.shardIndex],
                               let headNormPipe = headRmsnormPipeline {
                                var kNormOff = kNorm.offsetStart
                                var nK = numKvHeads
                                var hD = headDim
                                var hStride = headDim
                                var epsVal = eps
                                layerEnc.setComputePipelineState(headNormPipe)
                                layerEnc.setBuffer(jb.treeKVectorBuffer, offset: 0, index: 0)
                                layerEnc.setBuffer(kNormRaw, offset: 0, index: 1)
                                layerEnc.setBytes(&kNormOff, length: MemoryLayout<UInt64>.stride, index: 2)
                                layerEnc.setBytes(&nK, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc.setBytes(&hStride, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 6)
                                layerEnc.dispatchThreadgroups(MTLSize(width: Int(numKvHeads), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                layerEnc.memoryBarrier(scope: .buffers)
                            }

                            // Tree-Aware RoPE Rotation: effectivePos = tokenPos + nodeDepths[nodeIdx]
                            if let ropePipe = InferenceEngine.shared.applyRopeTreePipeline {
                                var tPos = step
                                var rDim = rotaryDim
                                var hDimVal = headDim
                                var thVal = thetaVal
                                var qStr = isStandardGqa ? headDim : (headDim * 2)
                                var numNodesVal = UInt32(N)

                                // Rotate Q
                                layerEnc.setComputePipelineState(ropePipe)
                                layerEnc.setBuffer(jb.treeQGateBuffer, offset: 0, index: 0)
                                layerEnc.setBuffer(jb.depthsBuffer, offset: 0, index: 1)
                                layerEnc.setBytes(&tPos, length: MemoryLayout<UInt32>.stride, index: 2)
                                layerEnc.setBytes(&rDim, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc.setBytes(&thVal, length: MemoryLayout<Float>.stride, index: 5)
                                layerEnc.setBytes(&qStr, length: MemoryLayout<UInt32>.stride, index: 6)
                                layerEnc.setBytes(&numNodesVal, length: MemoryLayout<UInt32>.stride, index: 7)
                                layerEnc.dispatchThreads(MTLSize(width: Int(numHeads), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), ropePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                                // Rotate K
                                var kStr = headDim
                                layerEnc.setBuffer(jb.treeKVectorBuffer, offset: 0, index: 0)
                                layerEnc.setBytes(&kStr, length: MemoryLayout<UInt32>.stride, index: 6)
                                layerEnc.dispatchThreads(MTLSize(width: Int(numKvHeads), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numKvHeads), ropePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc.memoryBarrier(scope: .buffers)
                            }

                            // Store K and V vectors into KV Cache at slots [step ..< step + N]
                            let slotIndex = (loopIdx * actualLayers) + layer.fullAttnIndex
                            let maxSeq = KVCacheManager.shared.allocatedSeqLen
                            let prec = KVCacheManager.shared.activePrecision

                            if prec == .fp16, let storePipe = storeKvCacheF16Pipeline,
                               let kCache = KVCacheManager.shared.kCacheBuffer,
                               let vCache = KVCacheManager.shared.vCacheBuffer {
                                let layerByteOffset = slotIndex * maxSeq * Int(kvStride) * MemoryLayout<Float16>.stride
                                var tPos = step
                                var nKv = numKvHeads
                                var hD = headDim

                                layerEnc.setComputePipelineState(storePipe)
                                layerEnc.setBuffer(jb.treeKVectorBuffer, offset: 0, index: 0)
                                layerEnc.setBuffer(jb.treeVVectorBuffer, offset: 0, index: 1)
                                layerEnc.setBuffer(kCache, offset: layerByteOffset, index: 2)
                                layerEnc.setBuffer(vCache, offset: layerByteOffset, index: 3)
                                layerEnc.setBytes(&tPos, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 6)
                                layerEnc.dispatchThreads(MTLSize(width: Int(kvStride), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(kvStride), storePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc.memoryBarrier(scope: .buffers)
                            } else if let storePipe = storeKvCachePipeline,
                                      let kCache = KVCacheManager.shared.kCacheBuffer,
                                      let vCache = KVCacheManager.shared.vCacheBuffer {
                                let layerByteOffset = slotIndex * maxSeq * Int(kvStride) * MemoryLayout<Float>.stride
                                var tPos = step
                                var nKv = numKvHeads
                                var hD = headDim

                                layerEnc.setComputePipelineState(storePipe)
                                layerEnc.setBuffer(jb.treeKVectorBuffer, offset: 0, index: 0)
                                layerEnc.setBuffer(jb.treeVVectorBuffer, offset: 0, index: 1)
                                layerEnc.setBuffer(kCache, offset: layerByteOffset, index: 2)
                                layerEnc.setBuffer(vCache, offset: layerByteOffset, index: 3)
                                layerEnc.setBytes(&tPos, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 6)
                                layerEnc.dispatchThreads(MTLSize(width: Int(kvStride), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(kvStride), storePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc.memoryBarrier(scope: .buffers)
                            }

                            // Grouped-Query Attention Tree Causal Verification
                            if prec == .fp16, let treeAttnPipe = InferenceEngine.shared.gqaAttentionTreeVerifyStandardF16Pipeline,
                               let kCache = KVCacheManager.shared.kCacheBuffer,
                               let vCache = KVCacheManager.shared.vCacheBuffer {
                                let layerByteOffset = slotIndex * maxSeq * Int(kvStride) * MemoryLayout<Float16>.stride
                                var pLen = step
                                var numNodesVal = UInt32(N)
                                var nQ = numHeads
                                var nKv = numKvHeads
                                var hD = headDim

                                layerEnc.setComputePipelineState(treeAttnPipe)
                                layerEnc.setBuffer(jb.treeQGateBuffer, offset: 0, index: 0)
                                layerEnc.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                layerEnc.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                layerEnc.setBuffer(jb.treeMaskBuffer, offset: 0, index: 3)
                                layerEnc.setBuffer(jb.treeAttnCtxBuffer, offset: 0, index: 4)
                                layerEnc.setBytes(&pLen, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc.setBytes(&numNodesVal, length: MemoryLayout<UInt32>.stride, index: 6)
                                layerEnc.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 7)
                                layerEnc.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 8)
                                layerEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 9)
                                layerEnc.dispatchThreads(MTLSize(width: Int(numHeads), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), treeAttnPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc.memoryBarrier(scope: .buffers)
                            } else if let treeAttnPipe = InferenceEngine.shared.gqaAttentionTreeVerifyStandardPipeline,
                                      let kCache = KVCacheManager.shared.kCacheBuffer,
                                      let vCache = KVCacheManager.shared.vCacheBuffer {
                                let layerByteOffset = slotIndex * maxSeq * Int(kvStride) * MemoryLayout<Float>.stride
                                var pLen = step
                                var numNodesVal = UInt32(N)
                                var nQ = numHeads
                                var nKv = numKvHeads
                                var hD = headDim

                                layerEnc.setComputePipelineState(treeAttnPipe)
                                layerEnc.setBuffer(jb.treeQGateBuffer, offset: 0, index: 0)
                                layerEnc.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                layerEnc.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                layerEnc.setBuffer(jb.treeMaskBuffer, offset: 0, index: 3)
                                layerEnc.setBuffer(jb.treeAttnCtxBuffer, offset: 0, index: 4)
                                layerEnc.setBytes(&pLen, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc.setBytes(&numNodesVal, length: MemoryLayout<UInt32>.stride, index: 6)
                                layerEnc.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 7)
                                layerEnc.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 8)
                                layerEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 9)
                                layerEnc.dispatchThreads(MTLSize(width: Int(numHeads), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), treeAttnPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc.memoryBarrier(scope: .buffers)
                            }

                            // Attention Output Projection (o_proj)
                            dispatchLinear(enc: layerEnc, weight: layer.oProjTensor, scale: layer.oScaleTensor, bias: layer.oBiasTensor, inBuf: jb.treeAttnCtxBuffer, outBuf: jb.treeAttnOutBuffer, inDim: attnCtxDim, outDim: hiddenDim, batchSize: N)
                            layerEnc.memoryBarrier(scope: .buffers)
                        } else {
                            // Linear Attention (GatedDeltaNet Recurrent State)
                            let linValHeads: UInt32 = UInt32(modelConfig?.effectiveLinearNumValueHeads ?? (layer.inProjA != nil && (layer.inProjA!.offsetEnd - layer.inProjA!.offsetStart) > 32 * 2560 ? 48 : 32))
                            let linKeyHeads: UInt32 = UInt32(modelConfig?.effectiveLinearNumKeyHeads ?? 16)
                            let qkvDim: UInt32 = (linKeyHeads + linKeyHeads + linValHeads) * 128
                            let zDim: UInt32 = linValHeads * 128
                            let aDim: UInt32 = linValHeads
                            let bDim: UInt32 = linValHeads

                            dispatchLinear(enc: layerEnc, weight: layer.inProjQKV, scale: layer.inProjQKVScale, bias: layer.inProjQKVBias, inBuf: jb.treeXNorm1Buffer, outBuf: jb.treeQGateBuffer, inDim: hiddenDim, outDim: qkvDim, batchSize: N)
                            layerEnc.memoryBarrier(scope: .buffers)

                            if let l2Pipe = l2NormQkSeqPipeline ?? l2NormQkPipeline {
                                var numHeads: UInt32 = linKeyHeads
                                var headDimVal: UInt32 = 128
                                var qkvDimVal: UInt32 = qkvDim
                                layerEnc.setComputePipelineState(l2Pipe)
                                layerEnc.setBuffer(jb.treeQGateBuffer, offset: 0, index: 0)
                                layerEnc.setBytes(&numHeads, length: MemoryLayout<UInt32>.stride, index: 1)
                                layerEnc.setBytes(&headDimVal, length: MemoryLayout<UInt32>.stride, index: 2)
                                layerEnc.setBytes(&qkvDimVal, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc.dispatchThreads(MTLSize(width: Int(numHeads), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), l2Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc.memoryBarrier(scope: .buffers)
                            }

                            dispatchLinear(enc: layerEnc, weight: layer.inProjZ, scale: layer.inProjZScale, bias: layer.inProjZBias, inBuf: jb.treeXNorm1Buffer, outBuf: jb.treeZGateBuffer, inDim: hiddenDim, outDim: zDim, batchSize: N)
                            dispatchLinear(enc: layerEnc, weight: layer.inProjA, scale: layer.inProjAScale, bias: layer.inProjABias, inBuf: jb.treeXNorm1Buffer, outBuf: jb.treeAVectorBuffer, inDim: hiddenDim, outDim: aDim, batchSize: N)
                            dispatchLinear(enc: layerEnc, weight: layer.inProjB, scale: layer.inProjBScale, bias: layer.inProjBBias, inBuf: jb.treeXNorm1Buffer, outBuf: jb.treeBVectorBuffer, inDim: hiddenDim, outDim: bDim, batchSize: N)
                            layerEnc.memoryBarrier(scope: .buffers)

                            if let sBuf = KVCacheManager.shared.linearStateBuffer,
                               let aLog = layer.aLogTensor, let aLogRaw = buffers[aLog.shardIndex],
                               let dtBias = layer.dtBiasTensor, let dtBiasRaw = buffers[dtBias.shardIndex],
                               let linNorm = layer.linearNormTensor, let linNormRaw = buffers[linNorm.shardIndex] {
                                let linIdx = layer.linAttnIndex
                                let stateFloatsPerNode = Int(linValHeads * 128 * 128)
                                let baseStateByteOffset = linIdx * stateFloatsPerNode * MemoryLayout<Float>.stride
                                let layerOutStateByteOffset = linIdx * jb.maxNodes * stateFloatsPerNode * MemoryLayout<Float>.stride
                                var aLogOff = aLog.offsetStart
                                var dtBiasOff = dtBias.offsetStart
                                var linNormOff = linNorm.offsetStart
                                var numValHeads: UInt32 = linValHeads
                                var numKeyHeads: UInt32 = linKeyHeads
                                var headDim: UInt32 = 128
                                var epsVal = eps
                                var numNodesVal = UInt32(N)
                                var stateSizeFloats = UInt32(stateFloatsPerNode)

                                for d in 0...maxTreeDepth {
                                    var dVal = UInt32(d)
                                    // 1. Gather parent states for candidate nodes at depth d
                                    if let gatherPipe = InferenceEngine.shared.gatherGdnTreeParentStatesPipeline {
                                        layerEnc.setComputePipelineState(gatherPipe)
                                        layerEnc.setBuffer(sBuf, offset: baseStateByteOffset, index: 0)
                                        layerEnc.setBuffer(jb.treeGdnOutStateBuffer, offset: layerOutStateByteOffset, index: 1)
                                        layerEnc.setBuffer(jb.treeGdnParentStateBuffer, offset: 0, index: 2)
                                        layerEnc.setBuffer(jb.parentIndicesBuffer, offset: 0, index: 3)
                                        layerEnc.setBuffer(jb.depthsBuffer, offset: 0, index: 4)
                                        layerEnc.setBytes(&dVal, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc.setBytes(&numNodesVal, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc.setBytes(&stateSizeFloats, length: MemoryLayout<UInt32>.stride, index: 7)
                                        let vec4Count = Int(stateSizeFloats / 4)
                                        layerEnc.dispatchThreads(MTLSize(width: vec4Count, height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, gatherPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                        layerEnc.memoryBarrier(scope: .buffers)
                                    }

                                    // 2. Compute tree step for candidate nodes at depth d
                                    let useSigmoidGate = (modelConfig?.effectiveOutputGateType.lowercased() == "sigmoid")
                                    let selectedTreeStepPipe = useSigmoidGate
                                        ? (InferenceEngine.shared.gdnLinearAttnTreeStepSigmoidPipeline ?? InferenceEngine.shared.gdnLinearAttnTreeStepPipeline)
                                        : InferenceEngine.shared.gdnLinearAttnTreeStepPipeline

                                    if let treeStepPipe = selectedTreeStepPipe {
                                        layerEnc.setComputePipelineState(treeStepPipe)
                                        layerEnc.setBuffer(jb.treeQGateBuffer, offset: 0, index: 0)
                                        layerEnc.setBuffer(jb.treeZGateBuffer, offset: 0, index: 1)
                                        layerEnc.setBuffer(jb.treeAVectorBuffer, offset: 0, index: 2)
                                        layerEnc.setBuffer(jb.treeBVectorBuffer, offset: 0, index: 3)
                                        layerEnc.setBuffer(aLogRaw, offset: 0, index: 4)
                                        layerEnc.setBuffer(dtBiasRaw, offset: 0, index: 5)
                                        layerEnc.setBuffer(linNormRaw, offset: 0, index: 6)
                                        layerEnc.setBuffer(jb.treeGdnParentStateBuffer, offset: 0, index: 7)
                                        layerEnc.setBuffer(jb.treeGdnOutStateBuffer, offset: layerOutStateByteOffset, index: 8)
                                        layerEnc.setBuffer(jb.treeAttnCtxBuffer, offset: 0, index: 9)
                                        layerEnc.setBytes(&aLogOff, length: MemoryLayout<UInt64>.stride, index: 10)
                                        layerEnc.setBytes(&dtBiasOff, length: MemoryLayout<UInt64>.stride, index: 11)
                                        layerEnc.setBytes(&linNormOff, length: MemoryLayout<UInt64>.stride, index: 12)
                                        layerEnc.setBytes(&numValHeads, length: MemoryLayout<UInt32>.stride, index: 13)
                                        layerEnc.setBytes(&numKeyHeads, length: MemoryLayout<UInt32>.stride, index: 14)
                                        layerEnc.setBytes(&headDim, length: MemoryLayout<UInt32>.stride, index: 15)
                                        layerEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 16)
                                        layerEnc.setBytes(&dVal, length: MemoryLayout<UInt32>.stride, index: 17)
                                        layerEnc.setBuffer(jb.depthsBuffer, offset: 0, index: 18)
                                        layerEnc.dispatchThreadgroups(MTLSize(width: Int(linValHeads), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        layerEnc.memoryBarrier(scope: .buffers)
                                    }
                                }
                            }

                            let linAttnCtxDim = linValHeads * 128
                            dispatchLinear(enc: layerEnc, weight: layer.linearOutProjTensor ?? layer.oProjTensor, scale: layer.linearOutProjScale ?? layer.oScaleTensor, bias: layer.linearOutProjBias ?? layer.oBiasTensor, inBuf: jb.treeAttnCtxBuffer, outBuf: jb.treeAttnOutBuffer, inDim: linAttnCtxDim, outDim: hiddenDim, batchSize: N)
                            layerEnc.memoryBarrier(scope: .buffers)
                        }

                        // Residual 1: treeHMidBuffer = treeHiddenBuffer + treeAttnOutBuffer
                        let addPipe = addPipeline
                        layerEnc.setComputePipelineState(addPipe)
                        layerEnc.setBuffer(jb.treeHiddenBuffer, offset: 0, index: 0)
                        layerEnc.setBuffer(jb.treeAttnOutBuffer, offset: 0, index: 1)
                        layerEnc.setBuffer(jb.treeHMidBuffer, offset: 0, index: 2)
                        var hD = hiddenDim
                        layerEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 3)
                        layerEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                        layerEnc.memoryBarrier(scope: .buffers)

                        // Step C: Post-Attention RMSNorm 2
                        if let norm2 = layer.norm2Tensor, let normRaw = buffers[norm2.shardIndex] {
                            let rmsPipe = getRmsNormPipe(norm2.dtype)
                            var nOff = norm2.offsetStart
                            var hD = hiddenDim
                            var epsVal = eps
                            layerEnc.setComputePipelineState(rmsPipe)
                            layerEnc.setBuffer(jb.treeHMidBuffer, offset: 0, index: 0)
                            layerEnc.setBuffer(normRaw, offset: 0, index: 1)
                            layerEnc.setBuffer(jb.treeXNorm2Buffer, offset: 0, index: 2)
                            layerEnc.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                            layerEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                            layerEnc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                            layerEnc.memoryBarrier(scope: .buffers)
                        }

                        // Step D: Feed-Forward MLP (Dense or MoE)
                        layerEnc.setComputePipelineState(clearPipeline)
                        layerEnc.setBuffer(jb.treeHMlpBuffer, offset: 0, index: 0)
                        layerEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                        layerEnc.memoryBarrier(scope: .buffers)

                        if layer.mlpType == .denseMlp || layer.routerTensor == nil {
                            if let gateW = layer.denseGateWeight,
                               let upW = layer.denseUpWeight,
                               let downW = layer.denseDownWeight {
                                for n in 0..<N {
                                    let inOff = n * Int(hiddenDim) * MemoryLayout<Float>.stride
                                    let interOff = n * Int(layer.intermediateDim) * MemoryLayout<Float>.stride
                                    let accumOff = n * Int(hiddenDim) * MemoryLayout<Float>.stride
                                    dispatchExpertMlp(
                                        enc: layerEnc,
                                        gateW: gateW,
                                        gateS: layer.denseGateScale,
                                        gateB: layer.denseGateBias,
                                        upW: upW,
                                        upS: layer.denseUpScale,
                                        upB: layer.denseUpBias,
                                        downW: downW,
                                        downS: layer.denseDownScale,
                                        downB: layer.denseDownBias,
                                        inBuf: jb.treeXNorm2Buffer,
                                        interBuf: jb.treeInterBuffer,
                                        accumBuf: jb.treeHMlpBuffer,
                                        inDim: hiddenDim,
                                        interDim: layer.intermediateDim,
                                        routingWeight: 1.0,
                                        inOffset: inOff,
                                        interOffset: interOff,
                                        accumOffset: accumOff
                                    )
                                }
                                layerEnc.memoryBarrier(scope: .buffers)
                            }

                            // Residual 2: treeHiddenBuffer = treeHMidBuffer + treeHMlpBuffer
                            layerEnc.setComputePipelineState(addPipe)
                            layerEnc.setBuffer(jb.treeHMidBuffer, offset: 0, index: 0)
                            layerEnc.setBuffer(jb.treeHMlpBuffer, offset: 0, index: 1)
                            layerEnc.setBuffer(jb.treeHiddenBuffer, offset: 0, index: 2)
                            var hD2 = hiddenDim
                            layerEnc.setBytes(&hD2, length: MemoryLayout<UInt32>.stride, index: 3)
                            layerEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            layerEnc.memoryBarrier(scope: .buffers)

                            layerEnc.endEncoding()
                        } else {
                            // MoE Expert Evaluation across candidate tree nodes using real router pre-pass
                            let topKCount = Int(modelConfig?.effectiveNumExpertsPerTok ?? (numExperts >= 512 ? 10 : 8))

                            if let routerTensor = layer.routerTensor, let routerRaw = buffers[routerTensor.shardIndex] {
                                var rOffset = routerTensor.offsetStart
                                var nExp = numExperts
                                var kVal: UInt32 = UInt32(topKCount)
                                var grp: UInt32 = 64
                                var hDimVal = hiddenDim

                                if let rScale = layer.routerScale, let rBias = layer.routerBias,
                                   let rScaleRaw = buffers[rScale.shardIndex], let rBiasRaw = buffers[rBias.shardIndex] {
                                    var sOffset = rScale.offsetStart
                                    var bOffset = rBias.offsetStart

                                    if (routerTensor.shapeDisplay.contains("512") || numExperts >= 512), let rQ8 = routerQ8Pipeline {
                                        layerEnc.setComputePipelineState(rQ8)
                                        layerEnc.setBuffer(routerRaw, offset: 0, index: 0)
                                        layerEnc.setBuffer(rScaleRaw, offset: 0, index: 1)
                                        layerEnc.setBuffer(rBiasRaw, offset: 0, index: 2)
                                        layerEnc.setBuffer(jb.treeXNorm2Buffer, offset: 0, index: 3)
                                        layerEnc.setBuffer(jb.treeRouterIndicesBuffer, offset: 0, index: 4)
                                        layerEnc.setBuffer(jb.treeRouterWeightsBuffer, offset: 0, index: 5)
                                        layerEnc.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                                        layerEnc.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                                        layerEnc.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 8)
                                        layerEnc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 9)
                                        layerEnc.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 10)
                                        layerEnc.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 11)
                                        layerEnc.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 12)
                                        layerEnc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
                                    } else if let rQ4 = routerQ4Pipeline {
                                        layerEnc.setComputePipelineState(rQ4)
                                        layerEnc.setBuffer(routerRaw, offset: 0, index: 0)
                                        layerEnc.setBuffer(rScaleRaw, offset: 0, index: 1)
                                        layerEnc.setBuffer(rBiasRaw, offset: 0, index: 2)
                                        layerEnc.setBuffer(jb.treeXNorm2Buffer, offset: 0, index: 3)
                                        layerEnc.setBuffer(jb.treeRouterIndicesBuffer, offset: 0, index: 4)
                                        layerEnc.setBuffer(jb.treeRouterWeightsBuffer, offset: 0, index: 5)
                                        layerEnc.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                                        layerEnc.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                                        layerEnc.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 8)
                                        layerEnc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 9)
                                        layerEnc.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 10)
                                        layerEnc.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 11)
                                        layerEnc.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 12)
                                        layerEnc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: Int(numExperts), height: 1, depth: 1))
                                    }
                                } else {
                                    if (routerTensor.shapeDisplay.contains("512") || numExperts >= 512), let r512 = router512Pipeline {
                                        layerEnc.setComputePipelineState(r512)
                                        layerEnc.setBuffer(routerRaw, offset: 0, index: 0)
                                        layerEnc.setBuffer(jb.treeXNorm2Buffer, offset: 0, index: 1)
                                        layerEnc.setBuffer(jb.treeRouterIndicesBuffer, offset: 0, index: 2)
                                        layerEnc.setBuffer(jb.treeRouterWeightsBuffer, offset: 0, index: 3)
                                        layerEnc.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                                        layerEnc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 7)
                                        layerEnc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                                    } else if let rBias = layer.routerExpertBias, let rBiasRaw = buffers[rBias.shardIndex],
                                               let bailingPipe = routerBailingPipeline {
                                        var bOffset = rBias.offsetStart
                                        var nGrp: UInt32 = UInt32(modelConfig?.nGroup ?? 8)
                                        var topkGrp: UInt32 = UInt32(modelConfig?.topkGroup ?? 4)
                                        var scaleFactor: Float = Float(modelConfig?.routedScalingFactor ?? 2.5)

                                        for n in 0..<N {
                                            let tokenOffset = n * Int(hiddenDim) * MemoryLayout<Float>.stride
                                            let rIndexOffset = n * topKCount * MemoryLayout<UInt32>.stride
                                            let rWeightOffset = n * topKCount * MemoryLayout<Float>.stride
                                            layerEnc.setComputePipelineState(bailingPipe)
                                            layerEnc.setBuffer(routerRaw, offset: 0, index: 0)
                                            layerEnc.setBuffer(rBiasRaw, offset: 0, index: 1)
                                            layerEnc.setBuffer(jb.treeXNorm2Buffer, offset: tokenOffset, index: 2)
                                            layerEnc.setBuffer(jb.treeRouterIndicesBuffer, offset: rIndexOffset, index: 3)
                                            layerEnc.setBuffer(jb.treeRouterWeightsBuffer, offset: rWeightOffset, index: 4)
                                            layerEnc.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                                            layerEnc.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                                            layerEnc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 7)
                                            layerEnc.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 8)
                                            layerEnc.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 9)
                                            layerEnc.setBytes(&nGrp, length: MemoryLayout<UInt32>.stride, index: 10)
                                            layerEnc.setBytes(&topkGrp, length: MemoryLayout<UInt32>.stride, index: 11)
                                            layerEnc.setBytes(&scaleFactor, length: MemoryLayout<Float>.stride, index: 12)
                                            layerEnc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: Int(numExperts), height: 1, depth: 1))
                                        }
                                    } else {
                                        layerEnc.setComputePipelineState(routerPipeline)
                                        layerEnc.setBuffer(routerRaw, offset: 0, index: 0)
                                        layerEnc.setBuffer(jb.treeXNorm2Buffer, offset: 0, index: 1)
                                        layerEnc.setBuffer(jb.treeRouterIndicesBuffer, offset: 0, index: 2)
                                        layerEnc.setBuffer(jb.treeRouterWeightsBuffer, offset: 0, index: 3)
                                        layerEnc.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                                        layerEnc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 7)
                                        layerEnc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: Int(numExperts), height: 1, depth: 1))
                                    }
                                }
                            }

                            layerEnc.endEncoding()
                            activeCmd.commit()
                            activeCmd.waitUntilCompleted()
                            if let treeErr = activeCmd.error {
                                print("❌ [METAL ERROR] tree verify activeCmd aborted: \(treeErr)")
                                return false
                            }

                            let indPtr = jb.treeRouterIndicesBuffer.contents().bindMemory(to: UInt32.self, capacity: N * topKCount)
                            let wPtr = jb.treeRouterWeightsBuffer.contents().bindMemory(to: Float.self, capacity: N * topKCount)

                            var candidateExpertsPerNode: [[(id: Int, weight: Float)]] = []
                            var uniqueActiveExperts = Set<UInt32>()
                            var flatCandidateExperts: [UInt32] = []

                            for n in 0..<N {
                                var nodeExp: [(id: Int, weight: Float)] = []
                                for k in 0..<topKCount {
                                    let expId = indPtr[n * topKCount + k]
                                    let w = wPtr[n * topKCount + k]
                                    flatCandidateExperts.append(expId)
                                    if w > 0.00001 {
                                        nodeExp.append((id: Int(expId), weight: w))
                                        uniqueActiveExperts.insert(expId)
                                    }
                                }
                                candidateExpertsPerNode.append(nodeExp)
                            }

                            var activeNodeIndices = Array(0..<N)
                            if uniqueActiveExperts.count > jetSpecMaxExpertCap && N > 1 {
                                let prunedMask = pruneJetspecTreeMoe(
                                    treeTokens: treeMask.tokenIds,
                                    parentIndices: treeMask.parentIndices,
                                    draftScores: nodeScores,
                                    candidateExpertsFlat: flatCandidateExperts,
                                    expertsPerNode: UInt32(topKCount),
                                    maxUniqueExperts: UInt32(jetSpecMaxExpertCap)
                                )
                                var keptIndices: [Int] = []
                                var searchStart = 0
                                for tok in prunedMask.tokenIds {
                                    for origIdx in searchStart..<N {
                                        if treeMask.tokenIds[origIdx] == tok {
                                            keptIndices.append(origIdx)
                                            searchStart = origIdx + 1
                                            break
                                        }
                                    }
                                }
                                if !keptIndices.isEmpty {
                                    activeNodeIndices = keptIndices
                                }
                            }

                            let prunedUniqueExperts = Array(Set(activeNodeIndices.flatMap { n in candidateExpertsPerNode[n].map { $0.id } })).sorted()
                            let expertSize = Int(loadedLayout?.expert_size ?? 1769472)

                            if let packedDir = packedExpertsDir,
                               let fd = ExpertIOThreadPool.shared.getOrOpenLayerFD(layerIndex: l, packedExpertsDir: packedDir) {
                                var tasks: [ExpertPreadTask] = []
                                let rawStagingPtr = expertStagingBuffer.contents()
                                for (slot, expId) in prunedUniqueExperts.enumerated() {
                                    let offset = off_t(expId * expertSize)
                                    let dst = rawStagingPtr.advanced(by: slot * expertSize)
                                    tasks.append(ExpertPreadTask(fd: fd, dst: dst, offset: offset, size: expertSize))
                                }
                                if !tasks.isEmpty {
                                    ExpertIOThreadPool.shared.dispatchSync(tasks: &tasks)
                                }
                            }

                            guard let nextCmd = commandQueue.makeCommandBuffer(),
                                  let moeEnc = nextCmd.makeComputeCommandEncoder() else { return false }
                            activeCmd = nextCmd

                            if packedExpertsDir != nil {
                                let isFP8Layout = loadedLayout?.components.contains { $0.dtype.contains("F8") || $0.name.contains("weight_scale") } ?? false

                                let compGateW = loadedLayout?.components.first(where: { $0.name.contains("gate_proj") && $0.name.contains("weight") && !$0.name.contains("scale") && !$0.name.contains("bias") })
                                let compGateS = loadedLayout?.components.first(where: { $0.name.contains("gate_proj") && ($0.name.contains("scale") || $0.name.contains("scales")) })
                                let compGateB = loadedLayout?.components.first(where: { $0.name.contains("gate_proj") && ($0.name.contains("bias") || $0.name.contains("biases")) })

                                let compUpW = loadedLayout?.components.first(where: { $0.name.contains("up_proj") && $0.name.contains("weight") && !$0.name.contains("scale") && !$0.name.contains("bias") })
                                let compUpS = loadedLayout?.components.first(where: { $0.name.contains("up_proj") && ($0.name.contains("scale") || $0.name.contains("scales")) })
                                let compUpB = loadedLayout?.components.first(where: { $0.name.contains("up_proj") && ($0.name.contains("bias") || $0.name.contains("biases")) })

                                let compDownW = loadedLayout?.components.first(where: { $0.name.contains("down_proj") && $0.name.contains("weight") && !$0.name.contains("scale") && !$0.name.contains("bias") })
                                let compDownS = loadedLayout?.components.first(where: { $0.name.contains("down_proj") && ($0.name.contains("scale") || $0.name.contains("scales")) })
                                let compDownB = loadedLayout?.components.first(where: { $0.name.contains("down_proj") && ($0.name.contains("bias") || $0.name.contains("biases")) })

                                let isQuantizedAffine = (compGateB != nil || compGateW?.name.contains("Q4") == true || compGateW?.name.contains("Q8") == true || compGateW?.dtype.contains("Q4") == true || compGateW?.dtype.contains("Q8") == true || (compGateS != nil && !isFP8Layout))

                                for n in activeNodeIndices {
                                    let inOff = n * Int(hiddenDim) * MemoryLayout<Float>.stride
                                    let interOff = n * Int(layer.intermediateDim) * MemoryLayout<Float>.stride
                                    let accumOff = n * Int(hiddenDim) * MemoryLayout<Float>.stride

                                    for expert in candidateExpertsPerNode[n] {
                                        let expId = expert.id
                                        let p_k = expert.weight
                                        if p_k <= 0.00001 { continue }
                                        guard let slot = prunedUniqueExperts.firstIndex(of: expId) else { continue }

                                        let slotOffset = UInt64(slot * expertSize)
                                        let gWOff = slotOffset + (compGateW?.offset ?? 0)
                                        let gSOff = slotOffset + (compGateS?.offset ?? 524288)
                                        let gBOff = slotOffset + (compGateB?.offset ?? 557056)
                                        let uWOff = slotOffset + (compUpW?.offset ?? 589824)
                                        let uSOff = slotOffset + (compUpS?.offset ?? 1114112)
                                        let uBOff = slotOffset + (compUpB?.offset ?? 1146880)
                                        let dWOff = slotOffset + (compDownW?.offset ?? 1179648)
                                        let dSOff = slotOffset + (compDownS?.offset ?? 1703936)
                                        let dBOff = slotOffset + (compDownB?.offset ?? 1736704)

                                        if isFP8Layout {
                                            let isBlockScale = (compGateS?.size ?? 1024) < (layer.intermediateDim * 2) || (compGateS?.name.contains("scale_inv") ?? false)
                                            let gateSimd = isBlockScale ? (fp8BlockGateUpSimdPipeline ?? fp8GateUpSimdPipeline ?? fp8GateUpPipeline) : (fp8GateUpSimdPipeline ?? fp8GateUpPipeline)
                                            let downSimd = isBlockScale ? (fp8BlockDownSimdPipeline ?? fp8DownSimdPipeline ?? fp8DownPipeline) : (fp8DownSimdPipeline ?? fp8DownPipeline)

                                            if let gateSimd = gateSimd, let downSimd = downSimd {
                                                var gWOffU = gWOff
                                                var gSOffU = gSOff
                                                var uWOffU = uWOff
                                                var uSOffU = uSOff
                                                var dWOffU = dWOff
                                                var dSOffU = dSOff
                                                var hDimVal: UInt32 = UInt32(hiddenDim)
                                                var interDimVal: UInt32 = UInt32(layer.intermediateDim)
                                                var pkVal = p_k

                                                moeEnc.setComputePipelineState(gateSimd)
                                                moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 0)
                                                moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 1)
                                                moeEnc.setBuffer(jb.treeXNorm2Buffer, offset: inOff, index: 2)
                                                moeEnc.setBuffer(jb.treeInterBuffer, offset: interOff, index: 3)
                                                moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 4)
                                                moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 5)
                                                moeEnc.setBytes(&gWOffU, length: 8, index: 6)
                                                moeEnc.setBytes(&gSOffU, length: 8, index: 7)
                                                moeEnc.setBytes(&uWOffU, length: 8, index: 8)
                                                moeEnc.setBytes(&uSOffU, length: 8, index: 9)
                                                moeEnc.setBytes(&hDimVal, length: 4, index: 10)
                                                moeEnc.setBytes(&interDimVal, length: 4, index: 11)
                                                moeEnc.dispatchThreadgroups(MTLSize(width: Int(layer.intermediateDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                                moeEnc.memoryBarrier(scope: .buffers)

                                                moeEnc.setComputePipelineState(downSimd)
                                                moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 0)
                                                moeEnc.setBuffer(jb.treeInterBuffer, offset: interOff, index: 1)
                                                moeEnc.setBuffer(jb.treeHMlpBuffer, offset: accumOff, index: 2)
                                                moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 3)
                                                moeEnc.setBytes(&dWOffU, length: 8, index: 4)
                                                moeEnc.setBytes(&dSOffU, length: 8, index: 5)
                                                moeEnc.setBytes(&interDimVal, length: 4, index: 6)
                                                moeEnc.setBytes(&hDimVal, length: 4, index: 7)
                                                moeEnc.setBytes(&pkVal, length: 4, index: 8)
                                                moeEnc.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                                moeEnc.memoryBarrier(scope: .buffers)
                                            }
                                        } else if isQuantizedAffine, let q4GatePipe = q4GateUpPipeline, let q4DownPipe = q4DownPipeline {
                                            var gWOffU = gWOff
                                            var gSOffU = gSOff
                                            var gBOffU = gBOff
                                            var uWOffU = uWOff
                                            var uSOffU = uSOff
                                            var uBOffU = uBOff
                                            var dWOffU = dWOff
                                            var dSOffU = dSOff
                                            var dBOffU = dBOff
                                            var hDimVal: UInt32 = UInt32(hiddenDim)
                                            var interDimVal: UInt32 = UInt32(layer.intermediateDim)
                                            var grp: UInt32 = 64
                                            var pkVal = p_k

                                            moeEnc.setComputePipelineState(q4GatePipe)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 0)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 1)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 2)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 3)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 4)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 5)
                                            moeEnc.setBuffer(jb.treeXNorm2Buffer, offset: inOff, index: 6)
                                            moeEnc.setBuffer(jb.treeInterBuffer, offset: interOff, index: 7)
                                            moeEnc.setBytes(&gWOffU, length: 8, index: 8)
                                            moeEnc.setBytes(&gSOffU, length: 8, index: 9)
                                            moeEnc.setBytes(&gBOffU, length: 8, index: 10)
                                            moeEnc.setBytes(&uWOffU, length: 8, index: 11)
                                            moeEnc.setBytes(&uSOffU, length: 8, index: 12)
                                            moeEnc.setBytes(&uBOffU, length: 8, index: 13)
                                            moeEnc.setBytes(&hDimVal, length: 4, index: 14)
                                            moeEnc.setBytes(&interDimVal, length: 4, index: 15)
                                            moeEnc.setBytes(&grp, length: 4, index: 16)
                                            moeEnc.dispatchThreadgroups(MTLSize(width: Int(layer.intermediateDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                            moeEnc.memoryBarrier(scope: .buffers)

                                            moeEnc.setComputePipelineState(q4DownPipe)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 0)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 1)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 2)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 3)
                                            moeEnc.setBuffer(jb.treeInterBuffer, offset: interOff, index: 4)
                                            moeEnc.setBuffer(jb.treeHMlpBuffer, offset: accumOff, index: 5)
                                            moeEnc.setBytes(&dWOffU, length: 8, index: 6)
                                            moeEnc.setBytes(&dSOffU, length: 8, index: 7)
                                            moeEnc.setBytes(&dBOffU, length: 8, index: 8)
                                            moeEnc.setBytes(&interDimVal, length: 4, index: 9)
                                            moeEnc.setBytes(&hDimVal, length: 4, index: 10)
                                            moeEnc.setBytes(&grp, length: 4, index: 11)
                                            moeEnc.setBytes(&pkVal, length: 4, index: 12)
                                            moeEnc.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                            moeEnc.memoryBarrier(scope: .buffers)
                                        } else if let gateSimd = bf16GateUpSimdPipeline, let downSimd = bf16DownSimdPipeline {
                                            var gWOffU = gWOff
                                            var uWOffU = uWOff
                                            var dWOffU = dWOff
                                            var hDimVal: UInt32 = UInt32(hiddenDim)
                                            var interDimVal: UInt32 = UInt32(layer.intermediateDim)
                                            var pkVal = p_k

                                            moeEnc.setComputePipelineState(gateSimd)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 0)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 1)
                                            moeEnc.setBuffer(jb.treeXNorm2Buffer, offset: inOff, index: 2)
                                            moeEnc.setBuffer(jb.treeInterBuffer, offset: interOff, index: 3)
                                            moeEnc.setBytes(&gWOffU, length: 8, index: 4)
                                            moeEnc.setBytes(&uWOffU, length: 8, index: 5)
                                            moeEnc.setBytes(&hDimVal, length: 4, index: 6)
                                            moeEnc.setBytes(&interDimVal, length: 4, index: 7)
                                            moeEnc.dispatchThreadgroups(MTLSize(width: Int(layer.intermediateDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                            moeEnc.memoryBarrier(scope: .buffers)

                                            moeEnc.setComputePipelineState(downSimd)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 0)
                                            moeEnc.setBuffer(jb.treeInterBuffer, offset: interOff, index: 1)
                                            moeEnc.setBuffer(jb.treeHMlpBuffer, offset: accumOff, index: 2)
                                            moeEnc.setBytes(&dWOffU, length: 8, index: 3)
                                            moeEnc.setBytes(&interDimVal, length: 4, index: 4)
                                            moeEnc.setBytes(&hDimVal, length: 4, index: 5)
                                            moeEnc.setBytes(&pkVal, length: 4, index: 6)
                                            moeEnc.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                            moeEnc.memoryBarrier(scope: .buffers)
                                        } else if let gateUnq = bf16GateUpPipeline, let downUnq = bf16DownPipeline {
                                            var gWOffU = gWOff
                                            var uWOffU = uWOff
                                            var dWOffU = dWOff
                                            var hDimVal: UInt32 = UInt32(hiddenDim)
                                            var interDimVal: UInt32 = UInt32(layer.intermediateDim)
                                            var pkVal = p_k

                                            moeEnc.setComputePipelineState(gateUnq)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 0)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 1)
                                            moeEnc.setBuffer(jb.treeXNorm2Buffer, offset: inOff, index: 2)
                                            moeEnc.setBuffer(jb.treeInterBuffer, offset: interOff, index: 3)
                                            moeEnc.setBytes(&gWOffU, length: 8, index: 4)
                                            moeEnc.setBytes(&uWOffU, length: 8, index: 5)
                                            moeEnc.setBytes(&hDimVal, length: 4, index: 6)
                                            moeEnc.setBytes(&interDimVal, length: 4, index: 7)
                                            moeEnc.dispatchThreads(MTLSize(width: Int(layer.intermediateDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(layer.intermediateDim), gateUnq.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                            moeEnc.memoryBarrier(scope: .buffers)

                                            moeEnc.setComputePipelineState(downUnq)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 0)
                                            moeEnc.setBuffer(jb.treeInterBuffer, offset: interOff, index: 1)
                                            moeEnc.setBuffer(jb.treeHMlpBuffer, offset: accumOff, index: 2)
                                            moeEnc.setBytes(&dWOffU, length: 8, index: 3)
                                            moeEnc.setBytes(&interDimVal, length: 4, index: 4)
                                            moeEnc.setBytes(&hDimVal, length: 4, index: 5)
                                            moeEnc.setBytes(&pkVal, length: 4, index: 6)
                                            moeEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), downUnq.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                            moeEnc.memoryBarrier(scope: .buffers)
                                        }
                                    }
                                }
                            } else {
                                for n in activeNodeIndices {
                                    let inOff = n * Int(hiddenDim) * MemoryLayout<Float>.stride
                                    let interOff = n * Int(layer.intermediateDim) * MemoryLayout<Float>.stride
                                    let accumOff = n * Int(hiddenDim) * MemoryLayout<Float>.stride

                                    for expert in candidateExpertsPerNode[n] {
                                        let expId = expert.id
                                        let p_k = expert.weight
                                        if p_k <= 0.00001 { continue }

                                        if let gW = layer.expertGateWeights[expId],
                                           let uW = layer.expertUpWeights[expId],
                                           let dW = layer.expertDownWeights[expId] {
                                            dispatchExpertMlp(
                                                enc: moeEnc,
                                                gateW: gW,
                                                gateS: layer.expertGateScales[expId],
                                                gateB: layer.expertGateBiases[expId],
                                                upW: uW,
                                                upS: layer.expertUpScales[expId],
                                                upB: layer.expertUpBiases[expId],
                                                downW: dW,
                                                downS: layer.expertDownScales[expId],
                                                downB: layer.expertDownBiases[expId],
                                                inBuf: jb.treeXNorm2Buffer,
                                                interBuf: jb.treeInterBuffer,
                                                accumBuf: jb.treeHMlpBuffer,
                                                inDim: hiddenDim,
                                                interDim: layer.intermediateDim,
                                                routingWeight: p_k,
                                                inOffset: inOff,
                                                interOffset: interOff,
                                                accumOffset: accumOff
                                            )
                                        }
                                    }
                                }
                            }
                            moeEnc.memoryBarrier(scope: .buffers)

                            if let gateW = layer.sharedGateWeight,
                               let upW = layer.sharedUpWeight,
                               let downW = layer.sharedDownWeight {
                                for n in activeNodeIndices {
                                    let inOff = n * Int(hiddenDim) * MemoryLayout<Float>.stride
                                    let interOff = n * Int(layer.intermediateDim) * MemoryLayout<Float>.stride
                                    let accumOff = n * Int(hiddenDim) * MemoryLayout<Float>.stride
                                    dispatchExpertMlp(
                                        enc: moeEnc,
                                        gateW: gateW,
                                        gateS: layer.sharedGateScale,
                                        gateB: layer.sharedGateBias,
                                        upW: upW,
                                        upS: layer.sharedUpScale,
                                        upB: layer.sharedUpBias,
                                        downW: downW,
                                        downS: layer.sharedDownScale,
                                        downB: layer.sharedDownBias,
                                        inBuf: jb.treeXNorm2Buffer,
                                        interBuf: jb.treeInterBuffer,
                                        accumBuf: jb.treeHMlpBuffer,
                                        inDim: hiddenDim,
                                        interDim: layer.intermediateDim,
                                        routingWeight: 1.0,
                                        inOffset: inOff,
                                        interOffset: interOff,
                                        accumOffset: accumOff
                                    )
                                }
                                moeEnc.memoryBarrier(scope: .buffers)
                            }

                            // Residual 2: treeHiddenBuffer = treeHMidBuffer + treeHMlpBuffer
                            moeEnc.setComputePipelineState(addPipe)
                            moeEnc.setBuffer(jb.treeHMidBuffer, offset: 0, index: 0)
                            moeEnc.setBuffer(jb.treeHMlpBuffer, offset: 0, index: 1)
                            moeEnc.setBuffer(jb.treeHiddenBuffer, offset: 0, index: 2)
                            var hD2 = hiddenDim
                            moeEnc.setBytes(&hD2, length: MemoryLayout<UInt32>.stride, index: 3)
                            moeEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            moeEnc.memoryBarrier(scope: .buffers)

                            moeEnc.endEncoding()
                        }
                    }
                }

                // 4. Final RMSNorm & LM Head Output
                guard let finalEnc = activeCmd.makeComputeCommandEncoder() else { return false }
                if let normTensor = normTensor, let normShardBuffer = normShardBuffer {
                    var nOff = normOffset
                    var epsVal = eps
                    var hD = hiddenDim
                    let finalNormPipe = getRmsNormPipe(normTensor.dtype)
                    finalEnc.setComputePipelineState(finalNormPipe)
                    finalEnc.setBuffer(jb.treeHiddenBuffer, offset: 0, index: 0)
                    finalEnc.setBuffer(normShardBuffer, offset: 0, index: 1)
                    finalEnc.setBuffer(jb.treeXNorm1Buffer, offset: 0, index: 2)
                    finalEnc.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                    finalEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                    finalEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                    finalEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                    finalEnc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                    finalEnc.memoryBarrier(scope: .buffers)
                }

                // LM Head Projection across all N candidate tree nodes
                let lmHead = lmHeadTensor
                dispatchLinear(enc: finalEnc, weight: lmHead, scale: lmHeadScale, bias: lmHeadBias, inBuf: jb.treeXNorm1Buffer, outBuf: jb.targetLogitsBuffer, inDim: hiddenDim, outDim: vocabSize, batchSize: N)
                finalEnc.endEncoding()

                activeCmd.commit()
                activeCmd.waitUntilCompleted()
                if let treeErr = activeCmd.error {
                    print("❌ [METAL ERROR] tree forward final cmd aborted: \(treeErr)")
                    return false
                }

                return true
            }

            func runJetSpecTreeStep(rootToken: UInt32, step: UInt32, temperature: Float) -> (acceptedTokens: [UInt32], newStep: UInt32)? {
                guard effectiveJetSpec, let jb = jetspecStaging else { return nil }

                // 1. Obtain draft candidate proposals (Draft Head or N-gram Prompt Lookup + Top-K Logits fallback)
                var topDraftTokens: [UInt32] = []
                var topDraftScores: [Float] = []

                if let dhTensor = summary.tensors.first(where: { $0.name.contains("draft_head") || $0.name.contains("speculative_head") }),
                   let dhRaw = buffers[dhTensor.shardIndex],
                   let dhPipe = InferenceEngine.shared.jetDraftHeadPredictPipeline,
                   let cmd = commandQueue.makeCommandBuffer(),
                   let enc = cmd.makeComputeCommandEncoder() {
                    var hDimVal = hiddenDim
                    var vDimVal = vocabSize
                    enc.setComputePipelineState(dhPipe)
                    enc.setBuffer(hCurrBuffer, offset: 0, index: 0)
                    enc.setBuffer(dhRaw, offset: Int(dhTensor.offsetStart), index: 1)
                    enc.setBuffer(jb.draftLogitsBuffer, offset: 0, index: 2)
                    enc.setBuffer(jb.draftLogitsBuffer, offset: 0, index: 3)
                    enc.setBytes(&hDimVal, length: 4, index: 4)
                    enc.setBytes(&vDimVal, length: 4, index: 5)
                    enc.dispatchThreads(MTLSize(width: Int(vocabSize), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, dhPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    enc.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                    if let dhErr = cmd.error {
                        print("❌ [METAL ERROR] draft head cmd aborted: \(dhErr)")
                        return nil
                    }

                    let dhLogitsPtr = jb.draftLogitsBuffer.contents().bindMemory(to: Float.self, capacity: Int(vocabSize))
                    let kCount = min(8, Int(vocabSize))
                    var topCandidates: [(token: UInt32, score: Float)] = []
                    topCandidates.reserveCapacity(kCount + 1)
                    for v in 0..<Int(vocabSize) {
                        let score = dhLogitsPtr[v]
                        if topCandidates.count < kCount {
                            topCandidates.append((token: UInt32(v), score: score))
                            if topCandidates.count == kCount {
                                topCandidates.sort { $0.score > $1.score }
                            }
                        } else if score > topCandidates.last!.score {
                            topCandidates[kCount - 1] = (token: UInt32(v), score: score)
                            var idx = kCount - 1
                            while idx > 0 && topCandidates[idx].score > topCandidates[idx - 1].score {
                                topCandidates.swapAt(idx, idx - 1)
                                idx -= 1
                            }
                        }
                    }
                    let maxScore = topCandidates.map { $0.score }.max() ?? 0.0
                    let effTemp = max(0.01, temperature)
                    let expScores = topCandidates.map { exp(($0.score - maxScore) / effTemp) }
                    let sumExp = expScores.reduce(0.0, +)
                    for (idx, cand) in topCandidates.enumerated() {
                        topDraftTokens.append(cand.token)
                        let prob = sumExp > 0 ? (expScores[idx] / sumExp) : (1.0 / Float(topCandidates.count))
                        topDraftScores.append(prob)
                    }
                }

                let ctxLen = contextTokens.count
                if ctxLen >= 3 {
                    let ngramSize = min(3, ctxLen - 1)
                    let pattern = Array(contextTokens.suffix(ngramSize))
                    for i in stride(from: ctxLen - ngramSize - 1, through: 0, by: -1) {
                        if Array(contextTokens[i..<(i + ngramSize)]) == pattern {
                            let matchEnd = i + ngramSize
                            let numDraft = min(8, ctxLen - matchEnd)
                            for j in 0..<numDraft {
                                topDraftTokens.append(contextTokens[matchEnd + j])
                                topDraftScores.append(max(0.1, 1.0 - Float(j) * 0.1))
                            }
                            break
                        }
                    }
                }

                if topDraftTokens.isEmpty {
                    let logitsPtr = logitsBuffer.contents().bindMemory(to: Float.self, capacity: Int(vocabSize))
                    let kCount = min(8, Int(vocabSize))
                    var topCandidates: [(token: UInt32, score: Float)] = []
                    topCandidates.reserveCapacity(kCount + 1)

                    for v in 0..<Int(vocabSize) {
                        let score = logitsPtr[v]
                        if topCandidates.count < kCount {
                            topCandidates.append((token: UInt32(v), score: score))
                            if topCandidates.count == kCount {
                                topCandidates.sort { $0.score > $1.score }
                            }
                        } else if score > topCandidates.last!.score {
                            topCandidates[kCount - 1] = (token: UInt32(v), score: score)
                            var idx = kCount - 1
                            while idx > 0 && topCandidates[idx].score > topCandidates[idx - 1].score {
                                topCandidates.swapAt(idx, idx - 1)
                                idx -= 1
                            }
                        }
                    }

                    let maxScore = topCandidates.map { $0.score }.max() ?? 0.0
                    let effTemp = max(0.01, temperature)
                    let expScores = topCandidates.map { exp(($0.score - maxScore) / effTemp) }
                    let sumExp = expScores.reduce(0.0, +)
                    for (idx, cand) in topCandidates.enumerated() {
                        topDraftTokens.append(cand.token)
                        let prob = sumExp > 0 ? (expScores[idx] / sumExp) : (1.0 / Float(topCandidates.count))
                        topDraftScores.append(prob)
                    }
                }

                if topDraftTokens.isEmpty { return nil }

                // 2. Build Tree Topology using Rust Engine
                var treeMask = buildJetspecCandidateTree(
                    rootTokenId: rootToken,
                    draftTokens: topDraftTokens,
                    draftScores: topDraftScores,
                    depth: UInt32(jetSpecMaxDepth),
                    branchingFactor: UInt32(jetSpecBranchingFactor),
                    maxNodes: UInt32(jb.maxNodes)
                )

                let nodeCount = treeMask.nodeCount
                if nodeCount <= 1 { return nil }

                // 3. Multi-Node Parallel Tree Forward Pass with real MoE router pre-pass and expert budget enforcement
                var nodeScores: [Float] = [1.0]
                for s in topDraftScores {
                    nodeScores.append(s)
                }
                while nodeScores.count < Int(treeMask.nodeCount) {
                    nodeScores.append(0.5)
                }
                let ok = runJetSpecTreeForward(treeMask: treeMask, nodeScores: nodeScores, step: step)
                if !ok { return nil }

                // 5. Verify acceptance using Rust acceptance oracle (Greedy fast-path or Stochastic Sampling)
                // Gemma final logit soft-capping must reach EVERY logit row the tree
                // produces, exactly as the ordinary sampler does, or JetSpec samples from
                // softmax(logits) instead of the model's softmax(tanh(logits/cap)*cap).
                // Applied in place so verification, the winning-row copy, and the root
                // fallback all see the capped distribution.
                if let cap = modelConfig?.effectiveFinalLogitSoftcapping, cap > 0 {
                    let treeLogitsCount = Int(treeMask.nodeCount) * Int(vocabSize)
                    let treeSoftcapPtr = jb.targetLogitsBuffer.contents().bindMemory(to: Float.self, capacity: treeLogitsCount)
                    for i in 0..<treeLogitsCount {
                        treeSoftcapPtr[i] = tanhf(treeSoftcapPtr[i] / cap) * cap
                    }
                }
                let targetLogitsPtr = jb.targetLogitsBuffer.contents().bindMemory(to: Float.self, capacity: Int(treeMask.nodeCount) * Int(vocabSize))
                let treeTargetLogits = Array(UnsafeBufferPointer(start: targetLogitsPtr, count: Int(treeMask.nodeCount) * Int(vocabSize)))

                let acceptedResult: JetSpecAcceptedResult
                if temperature <= 0.01 {
                    acceptedResult = verifyJetspecTreeGreedy(
                        treeTokens: treeMask.tokenIds,
                        parentIndices: treeMask.parentIndices,
                        targetLogits: treeTargetLogits,
                        vocabSize: vocabSize
                    )
                } else {
                    let rngSeed = UInt64.random(in: 1...UInt64.max)
                    acceptedResult = verifyJetspecTreeSampling(
                        treeTokens: treeMask.tokenIds,
                        parentIndices: treeMask.parentIndices,
                        draftProbs: [],
                        targetLogits: treeTargetLogits,
                        vocabSize: vocabSize,
                        temperature: temperature,
                        rngSeed: rngSeed
                    )
                }

                // 6. Speculative KV Cache Compaction for accepted branch
                if acceptedResult.acceptedCount > 0,
                   let kCache = KVCacheManager.shared.kCacheBuffer,
                   let vCache = KVCacheManager.shared.vCacheBuffer {
                    let kvMgr = KVCacheManager.shared
                    let prec = kvMgr.activePrecision
                    let slotByteSize = Int(kvStride) * prec.bytesPerElement

                    var needsCompaction = false
                    for (i, nodeIdx) in acceptedResult.acceptedNodeIndices.enumerated() {
                        let destSlot = UInt32(i + 1)
                        if nodeIdx != destSlot {
                            needsCompaction = true
                            break
                        }
                    }

                    if needsCompaction,
                       let copyCmd = commandQueue.makeCommandBuffer(),
                       let blit = copyCmd.makeBlitCommandEncoder() {
                        for loopIdx in 0..<totalLoops {
                            for l in 0..<actualLayers {
                                let layer = cachedLayers[l]
                                if layer.attentionType == .fullAttention {
                                    let slot = (loopIdx * actualLayers) + layer.fullAttnIndex
                                    let layerByteOffset = kvMgr.kvSlotByteOffset(slot: slot, stride: Int(kvStride), elementBytes: prec.bytesPerElement)
                                    for (i, nodeIdx) in acceptedResult.acceptedNodeIndices.enumerated() {
                                        let destSlot = UInt32(i + 1)
                                        if nodeIdx != destSlot {
                                            // Ring slots (Gemma 4 sliding layers) compact inside
                                            // the ring: node k's slot is (step + k) % ringLen.
                                            let srcRow = kvMgr.ringSlotForRow(Int(step) + Int(nodeIdx), slot: slot)
                                            let dstRow = kvMgr.ringSlotForRow(Int(step) + Int(destSlot), slot: slot)
                                            let srcOffset = layerByteOffset + srcRow * slotByteSize
                                            let dstOffset = layerByteOffset + dstRow * slotByteSize
                                            blit.copy(from: kCache, sourceOffset: srcOffset, to: kCache, destinationOffset: dstOffset, size: slotByteSize)
                                            blit.copy(from: vCache, sourceOffset: srcOffset, to: vCache, destinationOffset: dstOffset, size: slotByteSize)
                                        }
                                    }
                                }
                            }
                        }
                        blit.endEncoding()
                        copyCmd.commit()
                        copyCmd.waitUntilCompleted()
                        if let cpErr = copyCmd.error {
                            print("❌ [METAL ERROR] tree step copyCmd aborted: \(cpErr)")
                            return nil
                        }
                    }
                    // Accepted positions step+1..step+m now live at their committed
                    // rows (linear or ring); mirror that in the ring bookkeeping.
                    kvMgr.noteRingStores(range: Int(step) + 1 ..< Int(step) + 1 + Int(acceptedResult.acceptedCount))
                }

                // 6.5 Commit winning node linear attention recurrent states to linearStateBuffer
                let winningNodeIdx = acceptedResult.acceptedNodeIndices.last ?? 0
                let linLayerCount = UInt32(cachedLayers.filter { $0.attentionType == .linearAttention }.count)
                if linLayerCount > 0,
                   let sBuf = KVCacheManager.shared.linearStateBuffer,
                   let commitPipe = InferenceEngine.shared.commitGdnTreeWinningStatePipeline,
                   let commitCmd = commandQueue.makeCommandBuffer(),
                   let commitEnc = commitCmd.makeComputeCommandEncoder() {
                    var winNode = UInt32(winningNodeIdx)
                    var maxN = UInt32(jb.maxNodes)
                    var numLin = linLayerCount
                    var stateSizeFloats = UInt32(jb.linValHeads * 128 * 128)

                    commitEnc.setComputePipelineState(commitPipe)
                    commitEnc.setBuffer(jb.treeGdnOutStateBuffer, offset: 0, index: 0)
                    commitEnc.setBuffer(sBuf, offset: 0, index: 1)
                    commitEnc.setBytes(&winNode, length: MemoryLayout<UInt32>.stride, index: 2)
                    commitEnc.setBytes(&maxN, length: MemoryLayout<UInt32>.stride, index: 3)
                    commitEnc.setBytes(&numLin, length: MemoryLayout<UInt32>.stride, index: 4)
                    commitEnc.setBytes(&stateSizeFloats, length: MemoryLayout<UInt32>.stride, index: 5)

                    let vec4Count = Int(stateSizeFloats / 4)
                    commitEnc.dispatchThreads(MTLSize(width: vec4Count, height: Int(numLin), depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, commitPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    commitEnc.endEncoding()
                    commitCmd.commit()
                    commitCmd.waitUntilCompleted()
                    if let cmErr = commitCmd.error {
                        print("❌ [METAL ERROR] tree step commitCmd aborted: \(cmErr)")
                        return nil
                    }
                }

                // 7. Copy winning node logits into logitsBuffer for next token sampling
                let lastNodeIdx = acceptedResult.acceptedNodeIndices.last ?? 0
                let srcOffset = Int(lastNodeIdx) * Int(vocabSize) * MemoryLayout<Float>.stride
                if let copyCmd = commandQueue.makeCommandBuffer(),
                   let blit = copyCmd.makeBlitCommandEncoder() {
                    blit.copy(from: jb.targetLogitsBuffer, sourceOffset: srcOffset, to: logitsBuffer, destinationOffset: 0, size: Int(vocabSize) * MemoryLayout<Float>.stride)
                    blit.endEncoding()
                    copyCmd.commit()
                    copyCmd.waitUntilCompleted()
                    if let cpErr = copyCmd.error {
                        print("❌ [METAL ERROR] tree step logits copy aborted: \(cpErr)")
                        return nil
                    }
                }

                // 8. Emit accepted tokens + bonus token
                var emittedTokens = acceptedResult.acceptedTokens
                if let bonus = acceptedResult.bonusToken {
                    emittedTokens.append(bonus)
                }

                if emittedTokens.isEmpty {
                    // Fallback: Sample directly from root node (Node 0) target logits
                    // avoiding redundant target forward pass for root token
                    let rootLogitsPtr = jb.targetLogitsBuffer.contents().bindMemory(to: Float.self, capacity: Int(vocabSize))
                    let rootLogits = Array(UnsafeBufferPointer(start: rootLogitsPtr, count: Int(vocabSize)))
                    let sampledTok: UInt32
                    if temperature <= 0.01 {
                        var bestVal = -Float.greatestFiniteMagnitude
                        var bestTok: UInt32 = 0
                        for (v, val) in rootLogits.enumerated() {
                            if val > bestVal {
                                bestVal = val
                                bestTok = UInt32(v)
                            }
                        }
                        sampledTok = bestTok
                    } else {
                        var maxL = -Float.greatestFiniteMagnitude
                        for l in rootLogits { if l > maxL { maxL = l } }
                        let effT = max(0.01, temperature)
                        let probs = rootLogits.map { exp(($0 - maxL) / effT) }
                        let s = probs.reduce(0.0, +)
                        let r = Float.random(in: 0...1) * s
                        var cum: Float = 0.0
                        var chosen: UInt32 = 0
                        for (idx, p) in probs.enumerated() {
                            cum += p
                            if cum >= r {
                                chosen = UInt32(idx)
                                break
                            }
                        }
                        sampledTok = chosen
                    }
                    emittedTokens = [sampledTok]
                }

                return (acceptedTokens: emittedTokens, newStep: step + UInt32(emittedTokens.count))
            }
            var jetSpecTotalDraftAccepted = 0
            var jetSpecTotalDraftProposed = 0
            var jetSpecMeanTau = 1.0

            let runAgentTools = request.agentToolsEnabled && !request.forceSynthesis
            var generatedTokenIds: [UInt32] = []
            var accumulatedDecodedText = ""
            var lastUIUpdateTime = CFAbsoluteTimeGetCurrent()


            if isFullRAM && WorkingSetManager.shared.residentExpertsCount < WorkingSetManager.shared.totalExpertKeysCount {
                        emit(.priming)
                WorkingSetManager.shared.preFaultAll(shardBuffers: buffers, summary: summary)
            }

            // Ingest prompt tokens into KV-cache and recurrent states (skipping pinned prefix)
            let promptCount = promptTokenIds.count - 1
            if promptCount > 0 {
                let startPos = min(prefixTokensReused, promptCount)
                if startPos < promptCount {
                    let prefillTokens = Array(promptTokenIds[startPos..<promptCount])
                    let ok: Bool
                    if modelConfig?.isLingModel == true || modelConfig?.isSparkModel == true {
                        var prefillSuccess = true
                        let prefillStartTime = CFAbsoluteTimeGetCurrent()
                        var lastPrefillUIUpdateTime = prefillStartTime
                        let totalTokens = prefillTokens.count

                        let initialPrefillStr = "Ingesting prompt: Token 1/\(totalTokens) (0%) • Initializing..."
                        emit(.prefillProgress(status: initialPrefillStr, speed: nil, rssGB: nil))

                        for (idx, pTok) in prefillTokens.enumerated() {
                            if Task.isCancelled { prefillSuccess = false; break }
                            let stepOk = autoreleasepool {
                                runTokenForward(tokenId: pTok, step: UInt32(startPos + idx), computeLogits: false)
                            }
                            if !stepOk {
                                if Task.isCancelled {
                                    print("🛑 [PREFILL] Token \(idx + 1)/\(totalTokens) stopped by user at step \(startPos + idx) — not a forward-pass error.")
                                } else {
                                    print("❌ [PREFILL] Token \(idx + 1)/\(totalTokens) (id=\(pTok), step=\(startPos + idx)) forward pass failed — see prior diagnostics for the failing layer/buffer.")
                                }
                                prefillSuccess = false
                                break
                            }

                            let now = CFAbsoluteTimeGetCurrent()
                            if now - lastPrefillUIUpdateTime >= 0.08 || idx == totalTokens - 1 {
                                lastPrefillUIUpdateTime = now
                                let elapsed = max(0.001, now - prefillStartTime)
                                let promptSpeed = Double(idx + 1) / elapsed
                                let pct = Int((Double(idx + 1) / Double(totalTokens)) * 100)
                                let speedStr = promptSpeed >= 10 ? String(format: "%.0f", promptSpeed) : String(format: "%.1f", promptSpeed)
                                let prefillStr = "Ingesting prompt: Token \(idx + 1)/\(totalTokens) (\(pct)%) • \(speedStr) tok/s"
                                let currentRss = WorkingSetManager.shared.effectiveResidentMemoryGB

                                emit(.prefillProgress(status: prefillStr, speed: promptSpeed, rssGB: currentRss))
                            }
                        }
                        ok = prefillSuccess
                    } else if modelConfig?.isGemma4Model == true {
                        // Chunked batched prefill: each call's batch must fit the
                        // sliding ring's slack (window + chunk) so a store never
                        // overwrites a row a later row in the same pass still
                        // reads, and the transient per-call buffers stay bounded
                        // instead of scaling with the whole delta prompt.
                        var chunkedOk = true
                        var chunkStart = 0
                        var chunkIndex = 0
                        let totalPrefillChunks = max(1, (prefillTokens.count + gemmaPrefillChunkCap - 1) / gemmaPrefillChunkCap)
                        while chunkStart < prefillTokens.count {
                            if Task.isCancelled { chunkedOk = false; break }
                            let chunkEnd = min(chunkStart + gemmaPrefillChunkCap, prefillTokens.count)
                            let chunk = Array(prefillTokens[chunkStart..<chunkEnd])
                        if !runLayerWisePrefill(promptTokens: chunk, startPos: UInt32(startPos) + UInt32(chunkStart), chunkIndex: chunkIndex, chunkCount: totalPrefillChunks, prefillProgress: { s, sp, rs in emit(.prefillProgress(status: s, speed: sp, rssGB: rs)) }) {
                                print("❌ [PREFILL] Gemma chunked prefill failed at tokens \(chunkStart)..<\(chunkEnd) — see prior diagnostics for the failing layer/buffer.")
                                chunkedOk = false
                                break
                            }
                            chunkStart = chunkEnd
                            chunkIndex += 1
                        }
                        ok = chunkedOk
                    } else {
                    ok = runLayerWisePrefill(promptTokens: prefillTokens, startPos: UInt32(startPos), prefillProgress: { s, sp, rs in emit(.prefillProgress(status: s, speed: sp, rssGB: rs)) })
                    }
                    if !ok {
                        PrefillTransientPool.shared.drain()
                        let prefillCancelled = Task.isCancelled
                        emit(.failedTurn(TurnFailure(kind: .prefillFailed, message: prefillCancelled ? "⏹ Generation stopped by user." : "❌ Ingestion failed during prefill.", cancelled: prefillCancelled, tokensGenerated: 0, isTechnicalError: !prefillCancelled, clearStreamingPlaceholders: true)))
                        return
                    }
                }
                currentStep = UInt32(promptCount)
                WorkingSetManager.shared.trimAfterPrefill(shardBuffers: buffers, mode: budgetMode)
                // The prefill transients (multi-GB expert staging) are no longer
                // needed once ingestion completes — release them before decode.
                PrefillTransientPool.shared.drain()

                // Clear prefill status once prefill completes
                emit(.prefillProgress(status: nil, speed: nil, rssGB: nil))
            }


            // Autoregressive generation loop
            var generationStartTime = CFAbsoluteTimeGetCurrent()

            var breakReasonText = ""
            var matchedStopSequence: String?
            for _ in 0..<maxTokens {
                if Task.isCancelled {
                    let tailText = (try? tokenizer.decode(ids: Array(generatedTokenIds.suffix(48)))) ?? ""
                    print("🛑 [GEN] ended at \(tokensGenerated) tokens: reason=cancelled tail='\(tailText.prefix(320))'")
                    break
                }
                let currentTokenId = contextTokens.last!

                var acceptedBatch: [UInt32] = []
                if effectiveJetSpec && tokensGenerated > 0,
                   let treeStepResult = runJetSpecTreeStep(rootToken: currentTokenId, step: currentStep, temperature: temp),
                   !treeStepResult.acceptedTokens.isEmpty {
                    acceptedBatch = treeStepResult.acceptedTokens
                    currentStep = treeStepResult.newStep
                    jetSpecTotalDraftAccepted += acceptedBatch.count
                    jetSpecTotalDraftProposed += 1
                    jetSpecMeanTau = Double(jetSpecTotalDraftAccepted) / Double(max(jetSpecTotalDraftProposed, 1))
                } else {
                    let ok = autoreleasepool {
                        runTokenForward(tokenId: currentTokenId, step: currentStep, computeLogits: true, wait: true)
                    }
                    if !ok {
                        let fwdCancelled = Task.isCancelled
                        emit(.failedTurn(TurnFailure(kind: .forwardPassFailed, message: fwdCancelled ? "⏹ Generation stopped by user." : "❌ Inference forward pass failed.", cancelled: fwdCancelled, tokensGenerated: tokensGenerated, isTechnicalError: !fwdCancelled, clearStreamingPlaceholders: true)))
                        return
                    }
                    currentStep += 1

                    // 4. Sample Next Token
                    let logitsPtr = logitsBuffer.contents().bindMemory(to: Float.self, capacity: Int(vocabSize))
                    if let cap = modelConfig?.effectiveFinalLogitSoftcapping, cap > 0 {
                        // Gemma final logit soft-capping: logits = tanh(logits / cap) * cap
                        for i in 0..<Int(vocabSize) {
                            logitsPtr[i] = tanhf(logitsPtr[i] / cap) * cap
                        }
                    }
                    let isGrammarActive = runAgentTools && (UserDefaults.standard.object(forKey: "dynamoe_agent_grammar_masking") == nil ? true : UserDefaults.standard.bool(forKey: "dynamoe_agent_grammar_masking"))
                    // Cancel can land after the loop-head check, and a superseded
                    // generation has its grammar mask dropped silently. Without this
                    // the task would draw a token from unmasked logits and commit it.
                    if Task.isCancelled || !GrammarConstrainedSampler.shared.isCurrent(grammarGenerationToken) {
                        print("⏹ [GEN] superseded at \(tokensGenerated) tokens — stopping before sampling")
                        break
                    }
                    let nextToken = InferenceEngine.sampleNextToken(
                        logits: logitsPtr,
                        vocabSize: Int(vocabSize),
                        contextTokens: contextTokens,
                        temperature: temp,
                        topP: topPVal,
                        minP: minPVal,
                        topK: topKVal,
                        repetitionPenalty: repPen,
                        presencePenalty: presPen,
                        eosTokenIds: modelConfig?.effectiveEosTokenIds.map { UInt32($0) } ?? [eosTokenId],
                        grammarMask: isGrammarActive && !UserDefaults.standard.bool(forKey: "dynamoe_disable_grammar") ? { maskLogits, maskVocab in
                            GrammarConstrainedSampler.shared.updateStateAndApplyLogitMask(
                                emittedText: accumulatedDecodedText,
                                logits: maskLogits,
                                vocabSize: maskVocab,
                                tokenDecoder: { try? tokenizer.decode(ids: [$0]) },
                                enforceStructuralTagContinuation: !(modelConfig?.isLingModel == true),
                                token: grammarGenerationToken
                            )
                        } : nil
                    )
                    if tokensGenerated < 10 && DebugFlags.tokenStepLogging {
                        let tokText = (try? tokenizer.decode(ids: [nextToken])) ?? ""
                        print("[Autoregressive] Step \(tokensGenerated): nextToken=\(nextToken) ('\(tokText)') logits[nextToken]=\(logitsPtr[Int(nextToken)])")
                    }
                    acceptedBatch = [nextToken]
                }

                var shouldBreak = false
                    breakReasonText = ""
                var lastCandidateToken: UInt32 = 0
                for nextToken in acceptedBatch {
                    // 5. Check EOS
                    let isLingEos = (modelConfig?.isLingModel == true) && (nextToken == 156895 || nextToken == 156892)
                    let isSparkEos = (modelConfig?.isSparkModel == true) && (nextToken == 1 || nextToken == 2 || nextToken == 7)
                    let isGemma4Eos = (modelConfig?.isGemma4Model == true) && (nextToken == 1 || nextToken == 106 || nextToken == 50)
                    if nextToken == eosTokenId || nextToken == 248044 || nextToken == 248046 || nextToken == 166101 || nextToken == 166102 || isLingEos || isSparkEos || isGemma4Eos {
                        shouldBreak = true
                        breakReasonText = "eos-token(\(nextToken))"
                        break
                    }

                    generatedTokenIds.append(nextToken)
                    contextTokens.append(nextToken)
                    tokensGenerated += 1

                    // Degenerate-loop guard: models occasionally fall into verbatim
                    // token cycles (the same 1-8 tokens repeating forever, EOS never
                    // sampled). Stop cleanly once the tail is an exact repeating
                    // cycle. Period-1 runs need more repeats so legitimate whitespace
                    // runs in code are never caught, and only a cycle whose unit
                    // carries a letter or digit counts (a run of blank lines or
                    // closing braces is formatting, not a loop).
                    if let period = TurnTextSplitter.detectDegenerateCycle(
                        tokenIds: generatedTokenIds,
                        decodeUnit: { ids in (try? tokenizer.decode(ids: ids)) ?? "" }
                    ) {
                        shouldBreak = true
                        breakReasonText = "degenerate-cycle(p\(period))"
                        break
                    }

                    if tokensGenerated == 1 {
                        firstTokenTimestamp = CFAbsoluteTimeGetCurrent()
                    }

                    // 6. Incremental Stream Token Decoding (UTF-8 safe)
                    // Re-decoding the full id sequence each step guarantees multi-byte
                    // characters (e.g. 4-byte astral emoji like 🚀) split across BPE token
                    // boundaries are assembled from complete bytes. A partial sequence
                    // lossily decodes to U+FFFD; when the completing token arrives the
                    // full decode changes that character, so we adopt the new text
                    // wholesale (the UI re-renders the full string each frame) instead of
                    // freezing on a stale prefix. A trailing U+FFFD is held back one frame
                    // so the placeholder never flashes mid-stream.
                    let fullDecoded = (try? tokenizer.decode(ids: generatedTokenIds)) ?? ""
                    var emittable = fullDecoded
                    if emittable.hasSuffix("\u{FFFD}") {
                        emittable = String(emittable.dropLast())
                    }
                    let deltaText: String
                    if emittable.hasPrefix(accumulatedDecodedText) {
                        deltaText = String(emittable.dropFirst(accumulatedDecodedText.count))
                    } else {
                        // Earlier partial bytes completed into a different character;
                        // the full-text re-render repairs it retroactively.
                        deltaText = ""
                    }
                    accumulatedDecodedText = emittable
                    if deltaText.contains("<｜end▁of▁text｜>") || deltaText.contains("<|im_end|>") || deltaText.contains("<|endoftext|>") || deltaText.contains("<|role_end|>") {
                        shouldBreak = true
                        breakReasonText = "end-tag-in-text"
                        break
                    }

                    // Pre-Execution Catching: Freeze decoding immediately when </tool_call> closes
                    // API stop sequences: matched incrementally against the full
                    // generated text so a sequence split across token boundaries is
                    // caught. Output truncates at the match start (OpenAI semantics —
                    // the stop sequence itself never reaches the client).
                    if !stopSequences.isEmpty {
                        var stopMatch: (offset: String.Index, sequence: String)?
                        for stop in stopSequences {
                            if let r = emittable.range(of: stop), stopMatch == nil || r.lowerBound < stopMatch!.offset {
                                stopMatch = (r.lowerBound, stop)
                            }
                        }
                        if let match = stopMatch {
                            accumulatedDecodedText = String(emittable[..<match.offset])
                            matchedStopSequence = match.sequence
                            shouldBreak = true
                            breakReasonText = "stop-sequence"
                            break
                        }
                    }

                    if runAgentTools && StreamingToolParser.shared.shouldFreezeGeneration(accumulatedText: accumulatedDecodedText, deltaText: deltaText, format: ToolCallFormat.forModel(isGemma4: modelConfig?.isGemma4Model == true)) {
                        shouldBreak = true
                        breakReasonText = "tool-parser-freeze"
                        break
                    }
                }

                if shouldBreak {
                    let tailText = (try? tokenizer.decode(ids: Array(generatedTokenIds.suffix(48)))) ?? ""
                    print("🛑 [GEN] ended at \(tokensGenerated) tokens: reason=\(breakReasonText) tail='\(tailText.prefix(320))'")
                    break
                }

                // Token boundary working set pruning: keep resident set strictly within budget
                WorkingSetManager.shared.trimToBudget(mode: budgetMode, shardBuffers: buffers)

                let genElapsedSec = CFAbsoluteTimeGetCurrent() - generationStartTime
                let tokPerSec = Double(tokensGenerated) / max(genElapsedSec, 0.001)
                let elapsedMs = genElapsedSec * 1000.0
                let currentRss = WorkingSetManager.shared.effectiveResidentMemoryGB
                let resCount = WorkingSetManager.shared.residentExpertsCount
                let totalExp = WorkingSetManager.shared.totalExpertKeysCount
                let hitRate = WorkingSetManager.shared.cacheHitRatePercent
                let prefetchEff = WorkingSetManager.shared.prefetchEfficiencyPercent
                let pageLat = WorkingSetManager.shared.currentPagingLatencyMs

                var updatedRaw = accumulatedDecodedText
                var thinkPart = ""
                var respPart = ""
                var activeThink = false

                let promptTrimmed = formattedPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
                let promptRequestsThinking = TurnTextSplitter.promptRequestsThinking(promptTrimmed)

                let thinkSplit = TurnTextSplitter.splitThinkingAndResponse(raw: updatedRaw, promptRequestsThinking: promptRequestsThinking)

                if thinkSplit.thinkClose {
                    if thinkingEndTimestamp == nil {
                        thinkingEndTimestamp = CFAbsoluteTimeGetCurrent()
                    }
                    thinkPart = thinkSplit.think
                    respPart = thinkSplit.resp
                    activeThink = false
                } else if thinkSplit.thinkOpen {
                    // Inside the thinking block before the response boundary arrives
                    thinkPart = thinkSplit.think
                    activeThink = true
                    respPart = ""
                } else {
                    // Normal direct response without think tags
                    thinkPart = ""
                    activeThink = false
                    respPart = thinkSplit.resp
                }
                let liveTtft = firstTokenTimestamp.map { $0 - startTime }
                let liveThinkDuration = thinkingEndTimestamp.map { $0 - generationStartTime }

                // Throttle MainActor UI updates to 60fps ProMotion frame cadence (16ms) or token interval
                let now = CFAbsoluteTimeGetCurrent()
                let shouldUpdateUI = (tokensGenerated == 1) || (now - lastUIUpdateTime >= 0.016) || (tokensGenerated % 4 == 0)

                if shouldUpdateUI {
                    lastUIUpdateTime = now
                    emit(.frame(CompletionFrame(rawText: updatedRaw, thinkingText: thinkPart, responseText: respPart, isActivelyThinking: activeThink, tokensGenerated: tokensGenerated, elapsedMs: elapsedMs, tokensPerSecond: tokPerSec, timeToFirstTokenSeconds: liveTtft, thinkingDurationSeconds: liveThinkDuration, jetSpecActive: effectiveJetSpec, jetSpecMeanTau: jetSpecMeanTau, jetSpecDraftAccepted: jetSpecTotalDraftAccepted, jetSpecDraftProposed: jetSpecTotalDraftProposed, residentMemoryGB: currentRss, residentExpertCount: resCount, totalExpertCount: totalExp, cacheHitRatePercent: hitRate, prefetchEfficiencyPercent: prefetchEff, pagingLatencyMs: pageLat)))
                }
            }

            let finalGenElapsedSec = CFAbsoluteTimeGetCurrent() - generationStartTime
            let finalTokPerSec = Double(tokensGenerated) / max(finalGenElapsedSec, 0.001)
            let finalElapsedMs = finalGenElapsedSec * 1000.0
            let finalRss = WorkingSetManager.shared.effectiveResidentMemoryGB
            let finalResCount = WorkingSetManager.shared.residentExpertsCount
            let finalHitRate = WorkingSetManager.shared.cacheHitRatePercent
            let finalPrefetchEff = WorkingSetManager.shared.prefetchEfficiencyPercent
            let finalPageLat = WorkingSetManager.shared.currentPagingLatencyMs
            let finalDecoded = accumulatedDecodedText.isEmpty ? ((try? tokenizer.decode(ids: generatedTokenIds)) ?? "") : accumulatedDecodedText

            var finalThink = ""
            var finalResp = ""
            let promptTrimmedFinal = formattedPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
            let promptRequestsThinkingFinal = TurnTextSplitter.promptRequestsThinking(promptTrimmedFinal)

            let finalSplit = TurnTextSplitter.splitThinkingAndResponse(raw: finalDecoded, promptRequestsThinking: promptRequestsThinkingFinal)
            if finalSplit.thinkClose {
                if thinkingEndTimestamp == nil {
                    thinkingEndTimestamp = CFAbsoluteTimeGetCurrent()
                }
                finalThink = finalSplit.think
                finalResp = finalSplit.resp
            } else if finalSplit.thinkOpen {
                finalThink = finalSplit.think
                finalResp = ""
            } else {
                finalThink = ""
                finalResp = finalSplit.resp
            }
            let finalTtft = firstTokenTimestamp.map { $0 - startTime }
            let finalThinkDuration = thinkingEndTimestamp.map { $0 - generationStartTime }

            // A cancelled turn must not finalize. The Stop/interrupt path already
            // cleared isGeneratingText/generationTask and set the status line, and a
            // replacement generation may be starting: finalizing here would nil out
            // the replacement's task handle, flip its UI flags, record this
            // truncated turn into the prefix cache while the replacement is using
            // the same KV cache, and can schedule another agent turn on top of the
            // replacement. The partial reply is already committed to the message
            // from the streaming updates above, so nothing is lost by returning.
            if Task.isCancelled {
                print("⏹ [GEN] cancelled at \(tokensGenerated) tokens — skipping finalization")
                emit(.failedTurn(TurnFailure(kind: .cancelled, message: "⏹ Generation stopped by user.", cancelled: true, tokensGenerated: tokensGenerated, isTechnicalError: false, clearStreamingPlaceholders: false)))
                return
            }

            // Prefix-pin backfill: each decode iteration forwards the PREVIOUS
            // token before sampling the next, so the final appended token never
            // gets a forward pass. recordTurn then pins prompt+generated as fully
            // cached, but that last token's KV slot (and the GDN recurrent state,
            // snapshotted below at pin time) holds stale data — and the next
            // turn's delta prefill resumes past the gap, leaving one poisoned
            // slot in the pinned prefix per agent turn (observed as progressive
            // stutter/degeneration across multi-tool runs at ~3900+ tokens).
            // Forward the appended-but-uncomputed tail so the pin is real model
            // state. Under JetSpec the range may re-forward one already-written
            // accepted token; re-writing the same token to the same slot is benign.
            // Invariant: currentStep tracks the last (unforwarded) index, i.e.
            // currentStep == totalContextTokens - 1, because the decode loop
            // advances the step BEFORE appending its sampled token (and JetSpec
            // advances by its batch size). So this loop forwards exactly one token.
            if !generatedTokenIds.isEmpty {
                let totalContextTokens = promptTokenIds.count + tokensGenerated
                var backfillPos = Int(currentStep)
                var backfillFailed = false
                while backfillPos < totalContextTokens {
                    if Task.isCancelled { break }
                    let ok = autoreleasepool {
                        runTokenForward(tokenId: contextTokens[backfillPos], step: UInt32(backfillPos), computeLogits: false, wait: true)
                    }
                    if !ok {
                        backfillFailed = true
                        break
                    }
                    backfillPos += 1
                }
                // A cancel that lands mid-backfill must abort the whole turn, not
                // fall through: the code below pins the prefix and can schedule the
                // next agent step, which is exactly what the interrupt is trying to
                // stop.
                if Task.isCancelled {
                    print("⏹ [GEN] cancelled during prefix backfill — skipping finalization")
                    emit(.failedTurn(TurnFailure(kind: .cancelled, message: "⏹ Generation stopped by user.", cancelled: true, tokensGenerated: tokensGenerated, isTechnicalError: false, clearStreamingPlaceholders: false)))
                    return
                }
                // A failed forward means the tail is short of the tokens the pin
                // would claim. Drop the pinned prefix instead of recording a pin
                // whose KV slots were never written; the next turn then re-prefills
                // from scratch rather than trusting stale state.
                if backfillFailed {
                    print("⚠️ [GEN] prefix backfill forward failed at \(backfillPos)/\(totalContextTokens) — dropping the pinned prefix")
                    PrefixCacheManager.shared.invalidate(sessionId: sessionId)
                    emit(.failedTurn(TurnFailure(kind: .backfillFailed, message: "❌ Prefix backfill failed; generation stopped.", cancelled: false, tokensGenerated: tokensGenerated, isTechnicalError: true, clearStreamingPlaceholders: true)))
                    return
                }
            }

            // Agent Harness Multi-Step Tool Check
            let parsedResult = runAgentTools ? StreamingToolParser.shared.parseStreamingToolCalls(from: finalDecoded, format: ToolCallFormat.forModel(isGemma4: modelConfig?.isGemma4Model == true)) : (calls: [], brokenFragments: [])
            // No-op "gesture" calls (shell_run echo/true/:) are the model's way of
            // signalling it is finished. Executing them restarts the loop and
            // produces a duplicate answer bubble, so treat them as turn end.
            // Gesture calls are skipped but still occupy their slot in the
            // transcript, so every call the assistant emitted gets exactly one
            // result (see `splitGestureCalls`).
            let gestureSplit = AgentHarness.splitGestureCalls(parsedResult.calls)
            let actionableCallIndices = gestureSplit.actionableIndices
            let actionableCalls = actionableCallIndices.map { parsedResult.calls[$0] }
            var responseSlots = [String?](repeating: nil, count: parsedResult.calls.count)
            for (idx, notice) in gestureSplit.skipNotices { responseSlots[idx] = notice }
            if !gestureSplit.skipNotices.isEmpty {
                print("🧹 [AGENT] dropped \(gestureSplit.skipNotices.count) no-op gesture call(s) — treating turn as finished")
            }

            // Liveness: a superseded/cancelled turn must not pin, meter or
            // schedule anything — this mirrors the old `ownsGeneration` guard;
            // the engine is the single-flight arbiter.
            guard !Task.isCancelled, isCurrentGeneration(myGenerationId) else { return }

            KVCacheManager.shared.captureLinearStates()
            PrefixCacheManager.shared.recordTurn(
                promptTokenIds: promptTokenIds,
                generatedTokenIds: generatedTokenIds,
                sessionId: sessionId
            )
            print("📌 [PREFIX] pinned \(promptTokenIds.count + generatedTokenIds.count) tokens (prompt=\(promptTokenIds.count) gen=\(generatedTokenIds.count))")

            let finishReason: TurnFinishReason
            if breakReasonText.hasPrefix("eos-token") {
                finishReason = .eos
            } else if breakReasonText == "end-tag-in-text" {
                finishReason = .endTag
            } else if breakReasonText == "tool-parser-freeze" {
                finishReason = .toolFreeze
            } else if breakReasonText.hasPrefix("degenerate-cycle") {
                finishReason = .degenerateCycle
            } else if breakReasonText == "stop-sequence" {
                finishReason = .stoppedBySequence
            } else {
                finishReason = (tokensGenerated >= maxTokens) ? .lengthCapped : .stopped
            }
            emit(.finished(TurnResult(
                finishReason: finishReason,
                formattedPrompt: formattedPrompt,
                promptTokenIds: promptTokenIds,
                generatedTokenIds: generatedTokenIds,
                rawDecodedText: finalDecoded,
                thinkingText: finalThink,
                responseText: finalResp,
                tokensGenerated: tokensGenerated,
                elapsedMs: finalElapsedMs,
                tokensPerSecond: finalTokPerSec,
                timeToFirstTokenSeconds: finalTtft,
                thinkingDurationSeconds: finalThinkDuration,
                jetSpecActive: effectiveJetSpec,
                jetSpecMeanTau: jetSpecMeanTau,
                jetSpecDraftAccepted: jetSpecTotalDraftAccepted,
                jetSpecDraftProposed: jetSpecTotalDraftProposed,
                residentMemoryGB: finalRss,
                residentExpertCount: finalResCount,
                totalExpertCount: WorkingSetManager.shared.totalExpertKeysCount,
                cacheHitRatePercent: finalHitRate,
                prefetchEfficiencyPercent: finalPrefetchEff,
                pagingLatencyMs: finalPageLat,
                prefixTokensReused: prefixTokensReused,
                parsedToolCalls: parsedResult.calls,
                actionableCallIndices: gestureSplit.actionableIndices,
                gestureSkipNotices: gestureSplit.skipNotices,
                brokenFragments: parsedResult.brokenFragments,
                matchedStopSequence: matchedStopSequence
            )))
        }
        stateLock.lock()
        if currentGenerationId == myGenerationId {
            currentGenerationTask = handleTask
        }
        stateLock.unlock()
        return CompletionLaunch(
            prep: prep,
            handle: CompletionTurnHandle(generationId: myGenerationId, task: handleTask),
            stream: stream
        )
    }

    /// Cancels the in-flight turn (cooperative at layer/step boundaries).
    public func cancelActiveTurn() {
        stateLock.lock()
        currentGenerationTask?.cancel()
        stateLock.unlock()
    }

    /// Suspends until no turn is active (model switch/load safety).
    public func awaitIdle() async {
        while true {
            stateLock.lock()
            let pending = currentGenerationTask
            stateLock.unlock()
            guard let task = pending else { break }
            _ = await task.value
        }
    }

    private func turnDidBegin(_ id: UInt64) {
        // Slot reserved at spawn time.
    }

    private func turnDidEnd(id: UInt64) {
        stateLock.lock()
        defer { stateLock.unlock() }
        if currentGenerationId == id {
            currentGenerationTask = nil
        }
    }

    /// True while the given turn still owns the engine slot.
    fileprivate func isCurrentGenerationUnchecked(_ id: UInt64) -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return id == currentGenerationId && currentGenerationTask != nil
    }

    /// Installs the loaded model snapshot; called from model load/switch.
    public func configure(
        summary: ModelSummary?,
        modelConfig: ModelConfig?,
        tokenizer: DynaMoeTokenizer?,
        shardBuffers: [UInt32: MTLBuffer],
        activeLoadedModelPath: String?
    ) {
        stateLock.lock()
        defer { stateLock.unlock() }
        self.summary = summary
        self.modelConfig = modelConfig
        self.tokenizer = tokenizer
        self.shardBuffers = shardBuffers
        self.activeLoadedModelPath = activeLoadedModelPath
    }
}
