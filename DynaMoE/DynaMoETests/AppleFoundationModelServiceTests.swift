//
//  AppleFoundationModelServiceTests.swift
//  DynaMoETests
//
//  Created by Derek Parris on 10/5/26.
//

import XCTest

@testable import DynaMoE

/// Pure-logic coverage for the Apple Foundation Model backend's helpers.
/// Availability-dependent behavior is hardware-gated like the model snapshot
/// tests: it verifies consistency, not that any given Mac can serve requests.
final class AppleFoundationModelServiceTests: XCTestCase {

    // MARK: - Virtual Model Identity

    func testVirtualModelIdentityIsRecognized() {
        let model = AppleFoundationModelService.makeDiscoveredModel()
        XCTAssertEqual(model.id, AppleFoundationModelService.modelId)
        XCTAssertEqual(model.snapshotPath, AppleFoundationModelService.modelPath)
        XCTAssertEqual(model.displayName, AppleFoundationModelService.displayName)
        XCTAssertTrue(model.isAppleFoundationModel)
        XCTAssertFalse(model.isMoE)
        XCTAssertFalse(model.supportsThinking)
        XCTAssertFalse(model.hasTokenizer)
    }

    func testOrdinaryModelIsNotAppleFoundationModel() {
        let model = DiscoveredModel(
            id: "models--qwen--test",
            displayName: "Qwen Test",
            author: "qwen",
            repoId: "qwen/test",
            snapshotPath: "/tmp/qwen-snapshot",
            weightsEntryPath: "/tmp/qwen-snapshot",
            sizeBytes: 1,
            formattedSize: "1 B",
            architectureName: "Qwen3",
            isMoE: true
        )
        XCTAssertFalse(model.isAppleFoundationModel)
    }

    // MARK: - Transcript Building

    private func makeMessage(_ role: MessageRole, _ content: String) -> ChatMessage {
        ChatMessage(role: role, content: content)
    }

    func testTranscriptFormatsSpeakerTurnsInOrder() {
        let messages = [
            makeMessage(.user, "Hello"),
            makeMessage(.assistant, "Hi there!"),
            makeMessage(.user, "Write a haiku")
        ]
        let transcript = AppleFoundationModelService.buildConversationTranscript(
            from: messages,
            excludingMessageIds: [],
            charBudget: 10_000
        )
        XCTAssertEqual(transcript, "User: Hello\n\nAssistant: Hi there!\n\nUser: Write a haiku")
    }

    func testTranscriptSkipsPlaceholderAndEmptyTurns() {
        var placeholder = makeMessage(.assistant, "   ")
        placeholder.id = UUID()
        let messages = [
            makeMessage(.user, "Question"),
            placeholder,
            makeMessage(.assistant, "")
        ]
        let transcript = AppleFoundationModelService.buildConversationTranscript(
            from: messages,
            excludingMessageIds: [placeholder.id],
            charBudget: 10_000
        )
        XCTAssertEqual(transcript, "User: Question")
    }

    func testTranscriptDropsOldestTurnsFirstUnderBudget() {
        let messages = [
            makeMessage(.user, String(repeating: "a", count: 50)),
            makeMessage(.assistant, String(repeating: "b", count: 50)),
            makeMessage(.user, String(repeating: "c", count: 50))
        ]
        // "User: " + 50 chars = 55 chars per turn, plus a 2-char separator each.
        // A budget of 100 keeps the newest turn (55) but cannot fit the next (55 + 2 + 55).
        let transcript = AppleFoundationModelService.buildConversationTranscript(
            from: messages,
            excludingMessageIds: [],
            charBudget: 100
        )
        XCTAssertEqual(transcript, "User: " + String(repeating: "c", count: 50))
    }

    func testTranscriptAlwaysKeepsNewestTurnEvenOverBudget() {
        let messages = [makeMessage(.user, String(repeating: "x", count: 500))]
        let transcript = AppleFoundationModelService.buildConversationTranscript(
            from: messages,
            excludingMessageIds: [],
            charBudget: 10
        )
        XCTAssertEqual(transcript, "User: " + String(repeating: "x", count: 500))
    }

    func testTranscriptExcludesThinkingAndSystemTurns() {
        let messages = [
            makeMessage(.system, "You are a system prompt."),
            makeMessage(.user, "Real question")
        ]
        let transcript = AppleFoundationModelService.buildConversationTranscript(
            from: messages,
            excludingMessageIds: [],
            charBudget: 10_000
        )
        XCTAssertEqual(transcript, "User: Real question")
    }

    // MARK: - Approximations & Clamping

    func testApproximateTokenCount() {
        XCTAssertEqual(AppleFoundationModelService.approximateTokenCount(for: ""), 1)
        XCTAssertEqual(AppleFoundationModelService.approximateTokenCount(for: "abcd"), 1)
        XCTAssertEqual(AppleFoundationModelService.approximateTokenCount(for: "abcdefgh"), 2)
    }

    func testSendGateAllowsSystemModelSessionsWithoutEngine() {
        // Regression: the gate previously parsed as `tokenizer != nil && (summary != nil || AFM)`,
        // so system-model sessions — which release the tokenizer on activation —
        // could never send. The prompt stayed in the composer.
        XCTAssertTrue(AppleFoundationModelService.sendGateAllowsSend(hasEngine: false, sessionUsesSystemModel: true))
        XCTAssertTrue(AppleFoundationModelService.sendGateAllowsSend(hasEngine: true, sessionUsesSystemModel: true))
        XCTAssertTrue(AppleFoundationModelService.sendGateAllowsSend(hasEngine: true, sessionUsesSystemModel: false))
        XCTAssertFalse(AppleFoundationModelService.sendGateAllowsSend(hasEngine: false, sessionUsesSystemModel: false))
    }

    func testRevealStepPacesBacklogSmoothly() {
        // Small backlogs trickle at >= 1 char/tick; large ones cascade quickly
        // but never dump the full backlog in a single tick.
        XCTAssertEqual(AppleFoundationModelService.revealStep(forBacklog: 1), 1)
        XCTAssertEqual(AppleFoundationModelService.revealStep(forBacklog: 12), 2)
        XCTAssertEqual(AppleFoundationModelService.revealStep(forBacklog: 600), 100)
        XCTAssertEqual(AppleFoundationModelService.revealStep(forBacklog: 720), 120)
        XCTAssertEqual(
            AppleFoundationModelService.revealStep(forBacklog: 500_000),
            AppleFoundationModelService.maxRevealStepPerTick
        )
    }

    func testTemperatureIsClampedToValidRange() {
        XCTAssertEqual(AppleFoundationModelService.clampTemperature(nil), nil)
        XCTAssertEqual(AppleFoundationModelService.clampTemperature(0.3), 0.3)
        XCTAssertEqual(AppleFoundationModelService.clampTemperature(-0.5), 0.0)
        XCTAssertEqual(AppleFoundationModelService.clampTemperature(1.7), 1.0)
    }

    func testResponseTokenCeilingIsClamped() {
        XCTAssertEqual(AppleFoundationModelService.clampMaximumResponseTokens(nil), nil)
        XCTAssertEqual(AppleFoundationModelService.clampMaximumResponseTokens(10), 64)
        XCTAssertEqual(AppleFoundationModelService.clampMaximumResponseTokens(512), 512)
        XCTAssertEqual(
            AppleFoundationModelService.clampMaximumResponseTokens(8192),
            AppleFoundationModelService.maxResponseTokenCeiling
        )
    }

    func testTranscriptStripsLegacySpeakerLabelsFromHistory() {
        // Replies saved by the pre-fix completion-style prompting carry stray
        // speaker labels; the history block must not feed them back.
        let messages = [
            makeMessage(.user, "hi"),
            makeMessage(.assistant, "Assistant: Assistant: hello there!")
        ]
        let transcript = AppleFoundationModelService.buildConversationTranscript(
            from: messages,
            excludingMessageIds: [],
            charBudget: 10_000
        )
        XCTAssertEqual(transcript, "User: hi\n\nAssistant: hello there!")
    }

    func testChatTurnInstructionsFrameHistoryAsContext() {
        let instructions = AppleFoundationModelService.buildChatTurnInstructions(
            systemPrompt: "You are DynaMoE.",
            historyTranscript: "User: hi"
        )
        XCTAssertTrue(instructions.contains("You are DynaMoE."))
        XCTAssertTrue(instructions.contains("conversation has happened so far"))
        XCTAssertTrue(instructions.contains("User: hi"))
        XCTAssertTrue(instructions.contains("Reply with your next assistant message only"))
        XCTAssertTrue(instructions.contains("do not simulate further messages from either side"))
    }

    func testChatTurnInstructionsOmitEmptyBlocks() {
        let instructions = AppleFoundationModelService.buildChatTurnInstructions(
            systemPrompt: "   ",
            historyTranscript: ""
        )
        XCTAssertFalse(instructions.contains("conversation has happened so far"))
        XCTAssertTrue(instructions.contains("You are the assistant"))
    }

    func testStripLeadingSpeakerLabel() {
        XCTAssertEqual(AppleFoundationModelService.stripLeadingSpeakerLabel("Assistant: hello"), "hello")
        XCTAssertEqual(AppleFoundationModelService.stripLeadingSpeakerLabel("assistant:hello"), "hello")
        XCTAssertEqual(AppleFoundationModelService.stripLeadingSpeakerLabel("Assistant: Assistant: hi"), "hi")
        XCTAssertEqual(AppleFoundationModelService.stripLeadingSpeakerLabel("User: do this"), "do this")
        XCTAssertEqual(AppleFoundationModelService.stripLeadingSpeakerLabel("  \n Assistant: hi"), "hi")
        XCTAssertEqual(AppleFoundationModelService.stripLeadingSpeakerLabel("Plain reply"), "Plain reply")
        XCTAssertEqual(AppleFoundationModelService.stripLeadingSpeakerLabel("Userland is a game"), "Userland is a game")
        XCTAssertEqual(AppleFoundationModelService.stripLeadingSpeakerLabel("Assisting you today"), "Assisting you today")
        XCTAssertEqual(AppleFoundationModelService.stripLeadingSpeakerLabel("Assistant"), "Assistant")
    }

    // MARK: - Availability Consistency (hardware-gated, skips nothing but
    // verifies the status/isAvailable pair always agrees)

    func testAvailabilityStatusAndIsAvailableAgree() {
        let status = AppleFoundationModelService.checkAvailability()
        if status.isReady {
            XCTAssertTrue(AppleFoundationModelService.isAvailable)
        } else {
            XCTAssertFalse(AppleFoundationModelService.isAvailable)
        }
    }

    func testOSCompatibilityMatchesRuntimeOS() {
        // The build machine runs macOS 26.x, so the framework entry is offered;
        // on a macOS 15 machine this same check would report false.
        let os = ProcessInfo.processInfo.operatingSystemVersion
        let expected = os.majorVersion >= 26
        XCTAssertEqual(AppleFoundationModelService.isOSCompatible, expected)
    }
}
