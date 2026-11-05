//
//  Nanbeige43LogicTests.swift
//  DynaMoETests
//
//  Nanbeige 4.3 is NOT a new architecture: the checkpoint declares
//  `architectures: ["LlamaForCausalLM"]` / `model_type: "llama"` and ships
//  textbook Llama tensors, so the generic dense path already runs it. What this
//  file pins is everything that is NOT the forward pass, because the family is
//  otherwise indistinguishable from Llama:
//
//    * family identity, which can only come from the checkpoint's own shape,
//    * the chat template's deltas (a `<tool_response>` tool turn riding a USER
//      turn, and a reasoning block that is always present),
//    * the `qwen3_coder` XML tool dialect it documents,
//    * its EOS pair, which lives INSIDE a 166,912-row vocabulary,
//    * the KV geometry (8 KV heads x 128 = exactly the engine's 1024 stride
//      floor) and the long-context row ceiling.
//
//  All pure logic: no model weights, no Metal device, no snapshot.
//

import XCTest
@testable import DynaMoE

// MARK: - Config

final class Nanbeige43ConfigTests: XCTestCase {

    /// The published checkpoint's `config.json`, verbatim (snapshot
    /// `df919e1e…`). Embedded so the derived plan is asserted against the bytes
    /// the model actually ships rather than against a fixture invented here.
    private static let publishedConfigJSON = """
    {
      "architectures": ["LlamaForCausalLM"],
      "attention_bias": false,
      "attention_dropout": 0.0,
      "bos_token_id": 166100,
      "embd_pdrop": 0.0,
      "eos_token_id": 166101,
      "ffn_hidden_size": 22016,
      "head_dim": 128,
      "hidden_act": "silu",
      "hidden_size": 7168,
      "initializer_range": 0.02,
      "intermediate_size": 22016,
      "kv_channels": 128,
      "max_position_embeddings": 524288,
      "mlp_bias": false,
      "model_type": "llama",
      "num_attention_heads": 56,
      "num_hidden_layers": 52,
      "num_key_value_heads": 8,
      "num_query_groups": 8,
      "pad_token_id": 0,
      "pretraining_tp": 1,
      "resid_pdrop": 0.0,
      "rms_norm_eps": 1e-05,
      "rope_scaling": null,
      "rope_theta": 120000000,
      "rotary_base": 120000000,
      "seq_length": 524288,
      "tie_word_embeddings": false,
      "torch_dtype": "bfloat16",
      "transformers_version": "4.51.1",
      "use_cache": false,
      "vocab_size": 166912
    }
    """

    private func publishedConfig() throws -> ModelConfig {
        try JSONDecoder().decode(ModelConfig.self, from: Data(Self.publishedConfigJSON.utf8))
    }

    func testThePublishedConfigMatchesItsDeclaredGeometry() throws {
        let config = try publishedConfig()
        XCTAssertEqual(config.effectiveHiddenSize, 7168)
        XCTAssertEqual(config.effectiveNumHiddenLayers, 52)
        XCTAssertEqual(config.effectiveNumAttentionHeads, 56)
        XCTAssertEqual(config.effectiveNumKeyValueHeads, 8)
        XCTAssertEqual(config.effectiveHeadDim, 128)
        XCTAssertEqual(config.effectiveIntermediateSize, 22016)
        XCTAssertEqual(config.effectiveVocabSize, 166912)
        XCTAssertEqual(config.effectiveRopeTheta, 120000000)
        XCTAssertEqual(config.effectiveRmsNormEps, 1e-5)
        XCTAssertEqual(config.effectiveNumLoops, 1)
        XCTAssertEqual(config.effectiveMaxPositionEmbeddings, 524288)
    }

    func testTheFamilyRunsOnTheGenericDensePath() throws {
        // The whole point of the port: no new architecture case, no new branch
        // in the forward pass. If this ever stops holding, the family needs a
        // real engine path rather than the identity flag below.
        let config = try publishedConfig()
        XCTAssertEqual(config.resolveArchitectureType(summary: nil), .denseTransformer)
        XCTAssertFalse(config.isMoE)
        XCTAssertFalse(config.isRMSNormUnitOffset)
        XCTAssertEqual(config.effectiveRotaryDim, 128, "no partial_rotary_factor, so the full head rotates")
        XCTAssertNil(config.effectiveFinalLogitSoftcapping)
    }

    func testIdentityComesFromShapeNotFromTheName() throws {
        let config = try publishedConfig()
        XCTAssertTrue(config.isNanbeige43Model)

        // A Llama checkpoint with a different vocabulary and stack is NOT this
        // family: the fingerprint needs one of the checkpoint's two markers.
        let plainLlama = try JSONDecoder().decode(
            ModelConfig.self,
            from: Data("{\"model_type\":\"llama\",\"vocab_size\":128256,\"hidden_size\":4096,\"num_hidden_layers\":32}".utf8)
        )
        XCTAssertFalse(plainLlama.isNanbeige43Model)

        // Either marker alone is enough, so a checkpoint that renames its
        // architecture string is still recognized...
        XCTAssertTrue(ModelConfig.isNanbeige43Shape(
            modelType: "llama", architectures: ["LlamaForCausalLM"],
            vocabSize: 166912, hiddenSize: nil, layerCount: nil
        ))
        XCTAssertTrue(ModelConfig.isNanbeige43Shape(
            modelType: "llama", architectures: nil,
            vocabSize: nil, hiddenSize: 7168, layerCount: 52
        ))
        // ...but Llama typing is required: 166,912 rows alone is a wide
        // vocabulary, not an identity.
        XCTAssertFalse(ModelConfig.isNanbeige43Shape(
            modelType: "mistral", architectures: ["MistralForCausalLM"],
            vocabSize: 166912, hiddenSize: 7168, layerCount: 52
        ))
    }

    func testNoMandatoryPersonaIsInjected() throws {
        // The checkpoint's template has a Chinese default system line, but it is
        // the fallback for a conversation with NO system message; this app always
        // sends one. The Nanbeige-2 persona must not be claimed by 4.3.
        let config = try publishedConfig()
        let required = ModelConfig.resolveRequiredSystemPrompt(
            config: config,
            summary: nil,
            modelName: "Nanbeige4.3-33B",
            modelPath: "/tmp/models--Nanbeige--Nanbeige4.3-33B/snapshots/df919e1e"
        )
        XCTAssertEqual(required, "")
    }

    func testTheDeclaredEosIsAStopToken() throws {
        // `eos_token_id` is one number (166101). Answering the DEFAULT family's
        // pair instead meant the stop token was never injected as a sampler
        // candidate, so a turn could only end when it happened to rank in top-K.
        let config = try publishedConfig()
        XCTAssertEqual(config.effectiveEosTokenId, 166101)
        XCTAssertEqual(config.effectiveEosTokenIds, [166101])
    }

    func testThinkingIsDetectedFromShapeNotFromTheDirectoryName() throws {
        let config = try publishedConfig()
        XCTAssertTrue(ModelConfig.supportsThinking(config: config, modelName: "renamed-copy", modelPath: nil))
        XCTAssertFalse(ModelConfig.supportsThinkingLevels(modelPath: nil), "the family exposes no graded levels")
    }

    func testDtypeLabelsAreNormalised() {
        XCTAssertEqual(LocalModelManager.dtypeLabel("bfloat16"), "BF16")
        XCTAssertEqual(LocalModelManager.dtypeLabel("float16"), "FP16")
        XCTAssertEqual(LocalModelManager.dtypeLabel("float32"), "FP32")
        XCTAssertEqual(LocalModelManager.dtypeLabel("float8_e4m3fn"), "FP8 (E4M3)")
        XCTAssertEqual(LocalModelManager.dtypeLabel("Q4_0"), "Q4_0", "an unknown spelling passes through")
    }
}

// MARK: - KV geometry and the long-context ceiling

final class Nanbeige43KVGeometryTests: XCTestCase {

    private func geometry(_ precision: KVCachePrecision) throws -> KVCacheProvisioning.Geometry {
        let config = try JSONDecoder().decode(
            ModelConfig.self,
            from: Data((
                "{\"model_type\":\"llama\",\"hidden_size\":7168,\"num_hidden_layers\":52,"
                + "\"num_attention_heads\":56,\"num_key_value_heads\":8,\"head_dim\":128,"
                + "\"intermediate_size\":22016,\"vocab_size\":166912,\"max_position_embeddings\":524288}"
            ).utf8)
        )
        return KVCacheProvisioning.geometry(
            config: config,
            actualLayers: 52,
            totalLoops: 1,
            numKvHeads: 8,
            headDim: 128,
            precision: precision,
            slidingSlotFlags: nil,
            slidingRingLen: 0
        )
    }

    func testTheKvStrideLandsExactlyOnTheEngineFloor() throws {
        // 8 KV heads x 128 = 1024, which is the `max(kvStride, 1024)` floor's own
        // value. Allocation and the per-slot offset arithmetic therefore agree,
        // and this case is what fails loudly if either side changes.
        let geo = try geometry(.fp16)
        XCTAssertEqual(geo.kStride, 1024)
        XCTAssertEqual(geo.vStride, 1024)
        XCTAssertEqual(geo.totalSlots, 52)
        XCTAssertEqual(geo.kvHeadCapacity, 8)
        XCTAssertEqual(KVCacheProvisioning.ringSlotCount(geo), 0, "every layer is full attention")
    }

    func testTheCacheCostsWhatTheLongContextPlanSays() throws {
        let fp16 = try geometry(.fp16)
        let fp8 = try geometry(.fp8)
        XCTAssertEqual(KVCacheProvisioning.requiredBytes(maxSeqLen: 32768, geometry: fp16), 6_979_321_856)
        XCTAssertEqual(
            KVCacheProvisioning.requiredBytes(maxSeqLen: 32768, geometry: fp8) * 2,
            KVCacheProvisioning.requiredBytes(maxSeqLen: 32768, geometry: fp16),
            "FP8 is exactly half the FP16 bytes, which is the whole long-context argument"
        )
    }

    func testTheRowCeilingFollowsTheCheckpointRatherThanAConstant() throws {
        let config = try JSONDecoder().decode(
            ModelConfig.self,
            from: Data("{\"model_type\":\"llama\",\"max_position_embeddings\":524288}".utf8)
        )
        // 524,288 declared, capped at 262,144: the budget still decides, but a
        // policy constant no longer refuses a prompt the machine can serve.
        XCTAssertEqual(KVCacheProvisioning.policyMaxRows(config: config), 262144)
        XCTAssertEqual(KVCacheProvisioning.policyMaxRows(config: nil), KVCacheProvisioning.maxProvisionedRows)

        // A model declaring a small window keeps the range every existing turn
        // already had.
        let small = try JSONDecoder().decode(
            ModelConfig.self,
            from: Data("{\"model_type\":\"llama\",\"max_position_embeddings\":8192}".utf8)
        )
        XCTAssertEqual(KVCacheProvisioning.policyMaxRows(config: small), KVCacheProvisioning.maxProvisionedRows)

        // The longer ceiling is what lets a long prompt through at all.
        let prompt = 60_000
        let old = KVCacheProvisioning.neededSeqLen(
            sessionFloor: 2048, promptTokens: prompt, requestedMaxTokens: 512,
            policyMaxRows: KVCacheProvisioning.maxProvisionedRows
        )
        let new = KVCacheProvisioning.neededSeqLen(
            sessionFloor: 2048, promptTokens: prompt, requestedMaxTokens: 512,
            policyMaxRows: KVCacheProvisioning.policyMaxRows(config: config)
        )
        XCTAssertEqual(old, KVCacheProvisioning.maxProvisionedRows)
        XCTAssertGreaterThan(new, old)
    }

    func testAPrefillIsChunkedForEveryFamilyNowNotOnlyGemma() throws {
        // The transient buffers scale with the chunk, so ingesting a long dense
        // prompt in one call allocates them for the whole prompt.
        let llama = try JSONDecoder().decode(ModelConfig.self, from: Data("{\"model_type\":\"llama\"}".utf8))
        let gemma4 = try JSONDecoder().decode(ModelConfig.self, from: Data("{\"model_type\":\"gemma4\"}".utf8))
        XCTAssertEqual(prefillChunkCap(for: llama), densePrefillChunkCap)
        XCTAssertEqual(prefillChunkCap(for: gemma4), gemmaPrefillChunkCap)
        XCTAssertGreaterThan(densePrefillChunkCap, 0)
    }
}

// MARK: - Chat template

final class Nanbeige43PromptTests: XCTestCase {

    private func config() throws -> ModelConfig {
        try JSONDecoder().decode(
            ModelConfig.self,
            from: Data("{\"model_type\":\"llama\",\"vocab_size\":166912,\"hidden_size\":7168,\"num_hidden_layers\":52}".utf8)
        )
    }

    private let system = "You are helpful."

    private func render(
        turns: [ChatPromptFormatter.ConversationTurn],
        thinking: Bool,
        supports: Bool = true
    ) throws -> String {
        ChatPromptFormatter.formatConversation(
            turns: turns,
            cleanSystem: system,
            thinkingEnabled: thinking,
            modelSupportsThinking: supports,
            modelConfig: try config()
        )
    }

    private func single(_ prompt: String, thinking: Bool, supports: Bool = true) throws -> String {
        ChatPromptFormatter.formatPrompt(
            prompt: prompt,
            cleanSystem: system,
            thinkingEnabled: thinking,
            modelSupportsThinking: supports,
            modelConfig: try config()
        )
    }

    /// The checkpoint's template renders a thinking-ON generation prompt as
    /// `<|im_start|>assistant\n` followed by the reasoning opener and one newline.
    func testThinkingOnGolden() throws {
        let open = TurnTextSplitter.canonicalReasoningOpenTag
        XCTAssertEqual(
            try single("hello", thinking: true),
            "<|im_start|>system\n\(system)<|im_end|>\n"
                + "<|im_start|>user\nhello<|im_end|>\n"
                + "<|im_start|>assistant\n\(open)\n"
        )
    }

    /// ...and a thinking-OFF prompt as one already-closed, empty reasoning block.
    /// The block is always present: the template never omits it.
    func testThinkingOffGolden() throws {
        let open = TurnTextSplitter.canonicalReasoningOpenTag
        let close = TurnTextSplitter.canonicalReasoningCloseTag
        XCTAssertEqual(
            try single("hello", thinking: false),
            "<|im_start|>system\n\(system)<|im_end|>\n"
                + "<|im_start|>user\nhello<|im_end|>\n"
                + "<|im_start|>assistant\n\(open)\n\n\(close)\n\n"
        )
    }

    func testSingleTurnRendersLikeTheMultiTurnTemplate() throws {
        for thinking in [true, false] {
            XCTAssertEqual(
                try render(turns: [.init(role: .user, text: "hello")], thinking: thinking),
                try single("hello", thinking: thinking),
                "thinking=\(thinking)"
            )
        }
    }

    /// A tool result is NOT a turn of its own in this family: the checkpoint has
    /// no tool role, so the payload rides a USER turn inside `<tool_response>`.
    func testToolResultRidesAUserTurnInTheToolResponseBlock() throws {
        let prompt = try render(
            turns: [
                .init(role: .user, text: "list the folder"),
                .init(role: .assistant, text: "on it"),
                .init(role: .user, text: "file_a.txt", kind: .toolResult(name: "shell_run")),
                .init(role: .user, text: "thanks"),
            ],
            thinking: false
        )
        XCTAssertTrue(
            prompt.contains("<|im_start|>user\n<tool_response>\nfile_a.txt\n</tool_response>\n<|im_end|>\n"),
            "the result must be wrapped: \(prompt)"
        )
        // The wrapper is not applied to an ordinary user turn.
        XCTAssertTrue(prompt.contains("<|im_start|>user\nthanks<|im_end|>\n"))
    }

    /// The live continuation and the rebuilt history must agree, or a post-tool
    /// turn can never extend the previous turn's pinned prefix.
    func testTheLiveToolContinuationMatchesTheHistoryRender() throws {
        let payload = "file_a.txt\nfile_b.txt"
        let live = ChatPromptFormatter.nanbeigeToolResponse(text: payload)
        let history = try render(
            turns: [.init(role: .user, text: payload, kind: .toolResult(name: "shell_run"))],
            thinking: false
        )
        XCTAssertTrue(history.contains("<|im_start|>user\n" + live + "<|im_end|>\n"))

        // The harness's own ChatML continuation wraps the payload the same way,
        // so the two paths cannot drift.
        let harnessTurn = AgentHarness.shared.formatToolResponseTurn(responses: [payload])
        XCTAssertTrue(harnessTurn.contains(live))
    }

    /// A rebuilt call uses the family's own XML markup, not DSML.
    func testARebuiltToolCallUsesTheQwenCoderXMLShape() {
        let block = ChatPromptFormatter.nanbeigeToolCallBlock(
            name: "file_read",
            arguments: ["path": "/tmp/a.txt"]
        )
        XCTAssertEqual(
            block,
            "<tool_call>\n<function=file_read>\n<parameter=path>\n/tmp/a.txt\n</parameter>\n</function>\n</tool_call>"
        )
    }

    /// A historical turn that carried reasoning keeps its block open even after
    /// thinking is switched off: re-rendering it closed would move the reasoning
    /// outside the block it was generated in (and lose the pin).
    func testHistoricalReasoningKeepsItsBlockOpen() throws {
        let open = TurnTextSplitter.canonicalReasoningOpenTag
        let close = TurnTextSplitter.canonicalReasoningCloseTag
        let prompt = try render(
            turns: [
                .init(role: .user, text: "hello"),
                .init(role: .assistant, text: "why" + close + "ANSWER"),
                .init(role: .user, text: "again"),
            ],
            thinking: false
        )
        XCTAssertTrue(prompt.contains("\(open)\nwhy" + close + "ANSWER"))
    }

    func testEveryTurnIsClosedWithTheChatMLEndMarker() throws {
        XCTAssertEqual(
            ChatPromptFormatter.assistantTurnCloseTag(modelConfig: try config()),
            "<|im_end|>"
        )
    }
}

// MARK: - Tool dialect

final class Nanbeige43ToolDialectTests: XCTestCase {

    /// Exactly what the checkpoint's `qwen3_coder` parser expects and what its
    /// template's own example shows: one `<function=…>` inside `<tool_call>`,
    /// each argument a `<parameter=…>` child whose value sits on its own lines.
    private let publishedCall = """
    I'll read that file.

    <tool_call>
    <function=file_read>
    <parameter=path>
    /tmp/notes.md
    </parameter>
    </function>
    </tool_call>
    """

    func testTheFamilyRoutesToTheQwenXMLDialect() {
        XCTAssertEqual(ToolCallFormat.forModel(isGemma4: false), .qwenXML)
    }

    func testTheCallParsesAndFreezesOnItsCloser() {
        let parser = StreamingToolParser.shared
        XCTAssertTrue(
            parser.shouldFreezeGeneration(
                accumulatedText: publishedCall,
                deltaText: lastLine(of: publishedCall),
                format: .qwenXML
            ),
            "generation must freeze the instant </tool_call> closes"
        )

        let parsed = parser.parseStreamingToolCalls(from: publishedCall, format: .qwenXML)
        XCTAssertEqual(parsed.calls.count, 1)
        XCTAssertEqual(parsed.calls.first?.name, "file_read")
        XCTAssertEqual(parsed.calls.first?.arguments["path"] as? String, "/tmp/notes.md")
        XCTAssertTrue(parsed.brokenFragments.isEmpty)
    }

    /// The freeze fires on the CLOSER, so what reaches the parser is usually a
    /// block with no `</tool_call>`: that shape still executes, and the caller is
    /// told the block needs closing before it is spliced into the next prompt
    /// (left open, the model's own history becomes a stack of broken examples).
    func testAFreezeRecoversTheBlockItCut() {
        let parser = StreamingToolParser.shared
        let noCloser = String(publishedCall.dropLast("</tool_call>".count))
        XCTAssertTrue(StreamingToolParser.hasUnclosedToolCallBlock(noCloser))

        let recovered = parser.parseStreamingToolCalls(from: noCloser, format: .qwenXML)
        XCTAssertEqual(recovered.calls.map(\.name), ["file_read"])
        XCTAssertEqual(recovered.calls.first?.arguments["path"] as? String, "/tmp/notes.md")
    }

    private func lastLine(of text: String) -> String {
        String(text.split(separator: "\n").last ?? "")
    }
}

// MARK: - Discovery

final class Nanbeige43DiscoveryTests: XCTestCase {

    /// Discovery used to name a family from `architectures[0] ?? model_type`,
    /// which for this checkpoint is "LlamaForCausalLM" and matched no branch, so
    /// it was labelled a plain "Dense Transformer".
    func testDiscoveryNamesTheFamilyFromShapeAndReadsItsPrecision() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("nb43-discovery-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let configJSON = """
        {"architectures":["LlamaForCausalLM"],"model_type":"llama",
         "hidden_size":7168,"num_hidden_layers":52,"num_attention_heads":56,
         "num_key_value_heads":8,"head_dim":128,"intermediate_size":22016,
         "vocab_size":166912,"torch_dtype":"bfloat16",
         "eos_token_id":166101,"bos_token_id":166100,"rope_theta":120000000}
        """
        try Data(configJSON.utf8).write(to: dir.appendingPathComponent("config.json"))
        try Data("{}".utf8).write(to: dir.appendingPathComponent("tokenizer.json"))
        // Discovery only requires a weights entry to exist.
        try Data().write(to: dir.appendingPathComponent("model.safetensors"))

        let model = try XCTUnwrap(
            LocalModelManager.shared.inspectModelDirectory(dirUrl: dir, customRepoId: "Nanbeige/Nanbeige4.3-33B"),
            "a directory with config + weights must be discovered"
        )
        XCTAssertEqual(model.architectureName, "Nanbeige Transformer (33B Dense)")
        XCTAssertTrue(model.supportsThinking)
        XCTAssertFalse(model.supportsThinkingLevels)
        XCTAssertFalse(model.isMoE)
        XCTAssertEqual(model.quantization, "BF16", "torch_dtype is normalised, not echoed raw")
        XCTAssertEqual(model.rawModelType, "llama")
    }
}
