//
//  ConversationCompactionTests.swift
//  DynaMoETests
//
//  Created by Derek Parris on 10/5/26.
//

import XCTest

@testable import DynaMoE

/// Pure-logic coverage for manual conversation compaction: slash-command
/// parsing, the evict/keep split, the digest prompt plumbing, and the
/// persistence round-trip of the new session fields. The on-device summary
/// calls themselves are availability-gated at runtime, like the service's
/// other hardware-dependent behavior.
@MainActor
final class ConversationCompactionTests: XCTestCase {

    private func makeMessage(_ role: MessageRole, _ content: String) -> ChatMessage {
        ChatMessage(role: role, content: content)
    }

    // MARK: - Slash Command Parsing

    func testCompactCommandParses() {
        XCTAssertEqual(ChatCommand.parse("/compact"), .compact(focus: nil))
        XCTAssertEqual(ChatCommand.parse("  /COMPACT  "), .compact(focus: nil))
        XCTAssertEqual(ChatCommand.parse("/Compact focus on the buffer bug"), .compact(focus: "focus on the buffer bug"))
        XCTAssertEqual(ChatCommand.parse("/compact   keep the pread commands "), .compact(focus: "keep the pread commands"))
        // Every whitespace kind delimits the command word, not just the
        // literal space: newline- or tab-separated composer drafts dispatch
        // exactly like space-separated ones.
        XCTAssertEqual(ChatCommand.parse("/compact\nkeep the errors"), .compact(focus: "keep the errors"))
        XCTAssertEqual(ChatCommand.parse("/compact\n\n  keep the errors"), .compact(focus: "keep the errors"))
        XCTAssertEqual(ChatCommand.parse("/compact\tkeep the errors"), .compact(focus: "keep the errors"))
    }

    func testUnknownOrOrdinaryTextIsNotACommand() {
        // Not slash-led, or an unrecognized slash word: both must pass through
        // as ordinary chat so pasted paths like "/etc/hosts" never trigger a
        // command by accident.
        XCTAssertNil(ChatCommand.parse("hello there"))
        XCTAssertNil(ChatCommand.parse("how about /compact mid-sentence"))
        XCTAssertNil(ChatCommand.parse("/etc/hosts needs a look"))
        XCTAssertNil(ChatCommand.parse("/compacto"))
        XCTAssertEqual(ChatCommand.names, ["/compact"])
    }

    // MARK: - Compaction Context Block

    func testCompactionContextBlockFormatsGeneralAndDetail() {
        var session = ChatSession(messages: [])
        XCTAssertNil(session.compactionContextBlock)

        session.rollingSummary = "User is building a Metal-based LLM engine."
        let generalOnly = session.compactionContextBlock ?? ""
        XCTAssertTrue(generalOnly.contains("earlier history"))
        XCTAssertTrue(generalOnly.contains("Metal-based LLM engine"))
        XCTAssertFalse(generalOnly.contains("recently compacted work"))

        session.recentWorkDigest = "Fixed the pread fallback in ExpertIOThreadPool."
        let both = session.compactionContextBlock ?? ""
        let generalRange = both.range(of: "Metal-based LLM engine")
        let detailRange = both.range(of: "pread fallback")
        XCTAssertNotNil(generalRange)
        XCTAssertNotNil(detailRange)
        XCTAssertTrue(generalRange!.lowerBound < detailRange!.lowerBound)

        session.rollingSummary = "   "
        session.recentWorkDigest = nil
        XCTAssertNil(session.compactionContextBlock)
    }

    // MARK: - Summarizer Prompts

    func testCompactionPromptIncludesPriorDigestsAndExcerpt() {
        let prompt = AppleFoundationModelService.compactionPrompt(
            previousSummary: "Standing context so far.",
            previousRecap: "Recent recap details.",
            evictedTranscript: "User: do the thing"
        )
        XCTAssertTrue(prompt.contains("Previous standing summary"))
        XCTAssertTrue(prompt.contains("Standing context so far."))
        XCTAssertTrue(prompt.contains("Previous detailed recap"))
        XCTAssertTrue(prompt.contains("Recent recap details."))
        XCTAssertTrue(prompt.contains("Conversation excerpt to fold in"))
        XCTAssertTrue(prompt.contains("User: do the thing"))

        let fresh = AppleFoundationModelService.compactionPrompt(
            previousSummary: nil,
            previousRecap: nil,
            evictedTranscript: "User: fresh start"
        )
        XCTAssertFalse(fresh.contains("Previous standing summary"))
        XCTAssertFalse(fresh.contains("Previous detailed recap"))
        XCTAssertTrue(fresh.contains("User: fresh start"))
    }

    func testCompactionInstructionsCarryPrioritiesAndUserFocus() {
        let summary = AppleFoundationModelService.compactionSummaryInstructions(focus: nil)
        for term in ["file paths", "commands", "decisions", "open threads", "Do not invent facts"] {
            XCTAssertTrue(summary.contains(term), "summary instructions lost: \(term)")
        }
        XCTAssertFalse(summary.contains("Extra emphasis"))

        let focused = AppleFoundationModelService.compactionSummaryInstructions(focus: "the JetSpec budget math")
        XCTAssertTrue(focused.contains("Extra emphasis from the user for this compaction: the JetSpec budget math"))

        let recap = AppleFoundationModelService.compactionRecapInstructions(focus: nil)
        XCTAssertTrue(recap.contains("Completeness beats brevity"))
        XCTAssertFalse(recap.contains("Extra emphasis"))

        let recapFocused = AppleFoundationModelService.compactionRecapInstructions(focus: "the KV cache bug")
        XCTAssertTrue(recapFocused.contains("Extra emphasis from the user for this recap: the KV cache bug"))
    }

    func testChatTurnInstructionsPlaceCompactedContextBeforeHistory() {
        let instructions = AppleFoundationModelService.buildChatTurnInstructions(
            systemPrompt: "You are DynaMoE.",
            historyTranscript: "User: latest message",
            compactionContext: "Digest of the older work."
        )
        let personaRange = instructions.range(of: "You are DynaMoE.")
        let contextRange = instructions.range(of: "Digest of the older work.")
        let historyRange = instructions.range(of: "latest message")
        XCTAssertNotNil(personaRange)
        XCTAssertNotNil(contextRange)
        XCTAssertNotNil(historyRange)
        XCTAssertTrue(personaRange!.lowerBound < contextRange!.lowerBound)
        XCTAssertTrue(contextRange!.lowerBound < historyRange!.lowerBound)
    }

    func testHistoryBudgetReservesRoomForCompactionContext() {
        let base = AppleFoundationModelService.historyCharBudget(
            systemPrompt: "p", prompt: "hi", maximumResponseTokens: 64
        )
        let withContext = AppleFoundationModelService.historyCharBudget(
            systemPrompt: "p", prompt: "hi", maximumResponseTokens: 64,
            compactionContext: String(repeating: "x", count: 2_000)
        )
        XCTAssertEqual(base - withContext, 2_000)
    }

    // MARK: - Compaction Transcript (destructive-safe)

    func testCompactionTranscriptIncludesToolCallsAndResults() {
        let calls = [
            ToolCallRecord(name: "shell_run", arguments: ["command": "cargo test"], output: "test result: ok. 16 passed"),
            ToolCallRecord(name: "file_read", arguments: ["path": "ContentView.swift"], error: "file not found"),
            ToolCallRecord(name: "no_op", isGesture: true),
            ToolCallRecord(name: "web_fetch", arguments: ["url": "example.invalid"], output: String(repeating: "x", count: 1_500))
        ]
        let turn = AppleFoundationModelService.compactionTranscriptTurn(
            for: ChatMessage(role: .assistant, content: "checking the build", toolCalls: calls)
        ) ?? ""
        XCTAssertTrue(turn.contains("Assistant: checking the build"))
        XCTAssertTrue(turn.contains("Tool call shell_run(command=\"cargo test\") result: test result: ok. 16 passed"))
        XCTAssertTrue(turn.contains("Tool call file_read(path=\"ContentView.swift\") error: file not found"))
        XCTAssertFalse(turn.contains("no_op"), "skipped gesture calls are noise, not context")
        XCTAssertTrue(turn.contains("[…result truncated, 500 characters omitted]"), "huge tool results must be capped, not copied whole")
        XCTAssertFalse(turn.contains(String(repeating: "x", count: 1_100)), "the cap must actually cut the result off")
    }

    func testCompactionTranscriptSkipsSystemNoticesAndEmptyTurns() {
        XCTAssertNil(AppleFoundationModelService.compactionTranscriptTurn(for: makeMessage(.system, "⚠️ gate notice")))
        XCTAssertNil(AppleFoundationModelService.compactionTranscriptTurn(for: makeMessage(.assistant, "   ")))
        XCTAssertEqual(
            AppleFoundationModelService.compactionTranscriptTurn(for: makeMessage(.user, "run the tests")),
            "User: run the tests"
        )
    }

    func testCompactionChunksCoverEveryTurnOldestFirst() {
        // Six ~430-character turns with a 1,200-character budget: two turns
        // per chunk, three chunks, and no turn silently dropped the way the
        // old single-pass builder dropped everything past its cap.
        let messages = (0..<6).map { index in
            makeMessage(.user, "turn \(index): " + String(repeating: "detail ", count: 60))
        }
        let chunks = AppleFoundationModelService.buildCompactionTranscriptChunks(
            from: messages,
            charBudget: 1_200
        )
        XCTAssertEqual(chunks.count, 3)
        for (index, chunk) in chunks.enumerated() {
            XCTAssertLessThanOrEqual(chunk.count, 1_200, "chunk \(index) exceeds the budget")
        }
        for index in 0..<6 {
            XCTAssertTrue(chunks.contains { $0.contains("turn \(index):") }, "turn \(index) landed in no chunk")
        }
        XCTAssertTrue(chunks[0].contains("turn 0:"), "chunks must run oldest first")
        XCTAssertTrue(chunks[0].contains("turn 1:"))
        XCTAssertTrue(chunks[2].contains("turn 5:"), "the newest work must land in the recap chunk")
    }

    func testCompactionChunksTruncateOversizedTurnInsteadOfDroppingIt() {
        let huge = makeMessage(.user, String(repeating: "a", count: 5_000))
        let small = makeMessage(.user, "final note")
        let chunks = AppleFoundationModelService.buildCompactionTranscriptChunks(
            from: [huge, small],
            charBudget: 1_200
        )
        // A turn too large for any chunk becomes its own truncated chunk with
        // a marker instead of vanishing, and the turn after it still gets
        // summarized.
        XCTAssertEqual(chunks.count, 2)
        XCTAssertTrue(chunks[0].contains("turn truncated"))
        XCTAssertLessThanOrEqual(chunks[0].count, 1_200)
        XCTAssertTrue(chunks[1].contains("final note"))
        // A system-only history has nothing worth keeping.
        XCTAssertTrue(AppleFoundationModelService.buildCompactionTranscriptChunks(
            from: [makeMessage(.system, "notice")],
            charBudget: 1_200
        ).isEmpty)
    }

    func testCompactionChunkBudgetShrinksForStoredDigests() {
        // Fresh session: the derived budget is just the ceiling.
        let fresh = AppleFoundationModelService.compactionChunkCharBudget(
            previousSummary: nil,
            previousRecap: nil,
            instructions: "short instructions",
            maximumResponseTokens: AppleFoundationModelService.compactionSummaryResponseTokens
        )
        XCTAssertEqual(fresh, AppleFoundationModelService.compactionTranscriptCharBudget)

        // Repeat compaction with full-size digests: the chunk shrinks below the
        // ceiling so digests + instructions + response + chunk stay inside the
        // session window instead of overflowing it.
        let summary = String(repeating: "s", count: 3_072)
        let recap = String(repeating: "r", count: 4_096)
        let instructions = String(repeating: "i", count: 1_300)
        let repeatPass = AppleFoundationModelService.compactionChunkCharBudget(
            previousSummary: summary,
            previousRecap: recap,
            instructions: instructions,
            maximumResponseTokens: AppleFoundationModelService.compactionSummaryResponseTokens
        )
        XCTAssertLessThan(repeatPass, AppleFoundationModelService.compactionTranscriptCharBudget)
        XCTAssertGreaterThan(repeatPass, 0)
        // The request it sizes must actually fit: fixed parts plus the chunk
        // cannot exceed the session window.
        let fixed = AppleFoundationModelService.compactionSummaryResponseTokens * 4
            + summary.count + recap.count + instructions.count + 192
        XCTAssertLessThanOrEqual(fixed + repeatPass, AppleFoundationModelService.sessionContextChars)

        // Fixed digest/instruction content that alone fills the window reports
        // infeasibility (0) so the caller refuses with a clear message, instead
        // of flooring to a chunk that would guarantee an overflow.
        let degenerate = AppleFoundationModelService.compactionChunkCharBudget(
            previousSummary: String(repeating: "s", count: 9_000),
            previousRecap: String(repeating: "r", count: 9_000),
            instructions: instructions,
            maximumResponseTokens: AppleFoundationModelService.compactionSummaryResponseTokens
        )
        XCTAssertEqual(degenerate, 0)
    }

    func testCompactionChunkBudgetStaysInsideWindowForAnyRemainder() {
        // A large focus pushes the instructions up; the derived budget must
        // still never let fixed parts plus the chunk exceed the window.
        let instructions = String(repeating: "i", count: 12_000)
        let budget = AppleFoundationModelService.compactionChunkCharBudget(
            previousSummary: nil,
            previousRecap: nil,
            instructions: instructions,
            maximumResponseTokens: AppleFoundationModelService.compactionSummaryResponseTokens
        )
        XCTAssertGreaterThanOrEqual(budget, 0)
        let reserved = AppleFoundationModelService.compactionSummaryResponseTokens * 4
            + instructions.count + 192
        XCTAssertLessThanOrEqual(reserved + budget, AppleFoundationModelService.sessionContextChars)
    }

    func testCompactionFocusIsCappedSoInstructionsCannotFillTheWindow() {
        let hugeFocus = String(repeating: "f", count: 20_000)
        let instructions = AppleFoundationModelService.compactionSummaryInstructions(focus: hugeFocus)
        // The focus rides in the instructions once, bounded by the cap.
        XCTAssertGreaterThan(instructions.count, AppleFoundationModelService.maxCompactionFocusChars)
        XCTAssertLessThan(instructions.count, 3_000)
        // With normal digests the capped focus cannot drive the budget to zero.
        let budget = AppleFoundationModelService.compactionChunkCharBudget(
            previousSummary: String(repeating: "s", count: 3_072),
            previousRecap: String(repeating: "r", count: 4_096),
            instructions: instructions,
            maximumResponseTokens: AppleFoundationModelService.compactionRecapResponseTokens
        )
        XCTAssertGreaterThan(budget, 0)
    }

    func testChatTurnResponseCeilingClampsWhenDigestsCrowdTheWindow() {
        // Roomy turn: the user's requested ceiling survives untouched.
        let roomy = AppleFoundationModelService.chatTurnResponseTokenCeiling(
            requested: 8_192,
            systemPrompt: "You are DynaMoE.",
            prompt: "hello",
            compactionContext: nil
        )
        XCTAssertEqual(roomy, AppleFoundationModelService.maxResponseTokenCeiling)

        // A digest context crowding the window shrinks the response ceiling
        // rather than letting the assembled turn overflow with history
        // already floored at zero.
        let persona = String(repeating: "p", count: 600)
        let prompt = String(repeating: "u", count: 500)
        let crowded = AppleFoundationModelService.chatTurnResponseTokenCeiling(
            requested: 8_192,
            systemPrompt: persona,
            prompt: prompt,
            compactionContext: String(repeating: "d", count: 7_270)
        )
        XCTAssertLessThan(crowded, AppleFoundationModelService.maxResponseTokenCeiling)
        XCTAssertGreaterThan(crowded, 64)

        // Absurd fixed parts degrade to the framework's minimum instead of a
        // zero or negative ceiling.
        let absurd = AppleFoundationModelService.chatTurnResponseTokenCeiling(
            requested: 8_192,
            systemPrompt: persona,
            prompt: prompt,
            compactionContext: String(repeating: "d", count: 15_000)
        )
        XCTAssertEqual(absurd, 64)
    }

    func testChatTurnCompactionContextFitsWindowAroundMinimumResponse() {
        let persona = String(repeating: "p", count: 600)
        let prompt = String(repeating: "u", count: 500)
        // The oversized digest is trimmed to leave room for the framework's
        // minimum response, and the assembled fixed parts plus that response fit
        // inside the session window.
        let oversized = String(repeating: "d", count: 15_000)
        let fitted = AppleFoundationModelService.chatTurnCompactionContext(
            oversized,
            systemPrompt: persona,
            prompt: prompt
        )
        XCTAssertNotNil(fitted)
        XCTAssertLessThan(fitted!.count, oversized.count)
        let fixed = persona.count + prompt.count + fitted!.count
            + AppleFoundationModelService.chatTurnRulesText.count
            + AppleFoundationModelService.historyBlockHeader.count + 64
        XCTAssertLessThanOrEqual(
            fixed + AppleFoundationModelService.minimumResponseTokens * 4,
            AppleFoundationModelService.sessionContextChars
        )
        // The ceiling derived from the same oversized context trims it
        // internally, so it is at least the framework minimum and the turn can
        // actually be generated instead of overflowing.
        let ceiling = AppleFoundationModelService.chatTurnResponseTokenCeiling(
            requested: 8_192,
            systemPrompt: persona,
            prompt: prompt,
            compactionContext: oversized
        )
        XCTAssertGreaterThanOrEqual(ceiling, AppleFoundationModelService.minimumResponseTokens)
        // No digest is passed through untouched, and nil stays nil.
        XCTAssertEqual(
            AppleFoundationModelService.chatTurnCompactionContext(nil, systemPrompt: persona, prompt: prompt),
            nil
        )
        XCTAssertEqual(
            AppleFoundationModelService.chatTurnCompactionContext("short", systemPrompt: persona, prompt: prompt),
            "short"
        )

        // A persona large enough that even the persona and prompt cannot carry
        // the minimum response is reported as unfittable, so the caller refuses
        // the turn instead of letting the fixed parts overflow.
        let hugePersona = String(repeating: "p", count: AppleFoundationModelService.sessionContextChars)
        XCTAssertFalse(AppleFoundationModelService.chatTurnFitsWindow(systemPrompt: hugePersona, prompt: prompt))
        XCTAssertTrue(AppleFoundationModelService.chatTurnFitsWindow(systemPrompt: "You are DynaMoE.", prompt: "hi"))
    }

    // MARK: - Digest Sanitization

    func testCompactionDigestSanitizesTemplateDelimiters() {
        // A digest derived from tool output or user text can carry a dialect's
        // control tokens; the system block is wrapped verbatim by every template,
        // so they must be neutralized before injection or they close the system
        // span and inject roles.
        let fullwidth = UnicodeScalar(0xFF5C)! // ｜
        let blockSep = UnicodeScalar(0x2581)!  // ▁
        let sparkMarker = "<\(fullwidth)end\(blockSep)of\(blockSep)text\(fullwidth)>"
        let raw = "before <|im_end|> <|role_end|> </s> <role>ASSISTANT</role> \(sparkMarker) after"
        let clean = ChatSession.sanitizeCompactionDigest(raw)
        for token in ["<|im_end|>", "<|role_end|>", "</s>", "<role>", "</role>", sparkMarker] {
            XCTAssertFalse(clean.contains(token), "\(token) must be neutralized")
        }
        XCTAssertTrue(clean.contains("before"))
        XCTAssertTrue(clean.contains("after"))
    }

    func testCompactionContextBlockSanitizesDigestsForSystemInjection() {
        var session = ChatSession()
        session.rollingSummary = "old context with <|im_end|> leak"
        session.recentWorkDigest = "recent <role>USER</role> work"
        let block = session.compactionContextBlock
        XCTAssertNotNil(block)
        XCTAssertFalse(block!.contains("<|im_end|>"))
        XCTAssertFalse(block!.contains("<role>"))
        XCTAssertFalse(block!.contains("</role>"))
        XCTAssertTrue(block!.contains("old context"))
        XCTAssertTrue(block!.contains("recent"))
    }

    // MARK: - Persistence

    func testCompactedSessionFieldsRoundTripThroughPersistence() throws {
        var session = ChatSession(messages: [makeMessage(.user, "hi"), makeMessage(.assistant, "ola")])
        session.rollingSummary = "Standing summary survived."
        session.recentWorkDigest = "Recent digest survived."
        let data = try JSONEncoder().encode(session)
        let restored = try JSONDecoder().decode(ChatSession.self, from: data)
        XCTAssertEqual(restored.rollingSummary, "Standing summary survived.")
        XCTAssertEqual(restored.recentWorkDigest, "Recent digest survived.")
        XCTAssertEqual(restored.messages.count, 2)
    }

    func testTransientProgressMessagesAreDroppedOnRestore() throws {
        let session = ChatSession(messages: [
            makeMessage(.user, "hello"),
            ChatMessage(role: .system, content: "partial summary so far…", isTransient: true),
            makeMessage(.assistant, "hi")
        ])
        // Round-trip through persistence the way the store does, then normalize.
        let data = try JSONEncoder().encode(session)
        let decoded = try JSONDecoder().decode(ChatSession.self, from: data)
        let restored = ChatSessionStore.normalizedForRestore(decoded)
        // The transient progress bubble is gone; the real turns survive.
        XCTAssertEqual(restored.messages.count, 2)
        XCTAssertFalse(restored.messages.contains { $0.isTransient == true })
        XCTAssertTrue(restored.messages.contains { $0.content == "hello" })
        XCTAssertTrue(restored.messages.contains { $0.content == "hi" })
    }

    func testPreCompactionSessionFilesDecodeWithoutTheNewFields() throws {
        let legacyJSON = """
        {
          "id": "11111111-2222-3333-4444-555555555555",
          "title": "Legacy",
          "messages": [],
          "createdAt": 0,
          "updatedAt": 0
        }
        """
        let restored = try JSONDecoder().decode(ChatSession.self, from: Data(legacyJSON.utf8))
        XCTAssertEqual(restored.title, "Legacy")
        XCTAssertNil(restored.rollingSummary)
        XCTAssertNil(restored.recentWorkDigest)
    }
}
