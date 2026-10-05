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

    // MARK: - Evict/Keep Split

    func testCompactionSplitKeepsRecentVerbatimTail() {
        let session = ChatSession(
            messages: (0..<10).map { makeMessage($0 % 2 == 0 ? .user : .assistant, "m\($0)") }
        )
        let split = session.compactionSplit()
        XCTAssertEqual(split.evicted.count, 4)
        XCTAssertEqual(split.kept.count, ChatSession.messagesKeptRecentOnCompact)
        XCTAssertEqual(split.evicted.first?.content, "m0")
        XCTAssertEqual(split.kept.first?.content, "m4")
        XCTAssertEqual(split.kept.last?.content, "m9")
    }

    func testCompactionSplitEvictsNothingShortOfTheTail() {
        let short = ChatSession(messages: [makeMessage(.user, "hi"), makeMessage(.assistant, "hello")])
        XCTAssertEqual(short.compactionSplit().evicted, [])
        XCTAssertEqual(short.compactionSplit().kept.count, 2)

        let exact = ChatSession(
            messages: (0..<ChatSession.messagesKeptRecentOnCompact).map { makeMessage(.assistant, "t\($0)") }
        )
        XCTAssertEqual(exact.compactionSplit().evicted, [])
        XCTAssertEqual(exact.compactionSplit().kept.count, ChatSession.messagesKeptRecentOnCompact)
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
