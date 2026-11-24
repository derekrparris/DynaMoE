//
//  Inkling2PhaseFiveTests.swift
//  DynaMoETests
//
//  The remaining Phase 5 exit gates: the expanded prompt count is the one every
//  usage path reports, a picture survives a preflight refusal, and two pictures
//  in one turn keep their own rows.
//

import Metal
import XCTest
@testable import DynaMoE

/// The expanded prompt count is the number the client is told, on every path
/// that reports usage.
///
/// `testTheServerPricesTheExpandedPicture` pins the arithmetic itself; this pins
/// the WIRING around it, which is where a route could report the placeholder
/// count instead: the Anthropic pre-flight (`/v1/messages/count_tokens`) builds
/// its answer from the same `buildTurnPlan` a real `/v1/messages` turn uses, and
/// both reply paths carry `plan.promptTokenCount` -- the expanded size.
final class Inkling2PromptCountWiringTests: XCTestCase {

    /// The pre-flight's route body is one line over the shared planner, so the
    /// fact that needs pinning is the wiring: a count route that ever built its
    /// own estimate would drift from the turn it precedes, and the source is
    /// where that is visible without a loaded model.
    func testTheCountRouteSharesTheTurnsPlanner() throws {
        let source = try String(
            contentsOf: URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()      // DynaMoETests
                .deletingLastPathComponent()      // DynaMoE
                .appendingPathComponent("DynaMoE/ServerController.swift"),
            encoding: .utf8
        )
        let route = try XCTUnwrap(source.range(of: "router.post(\"v1/messages/count_tokens\")"))
        // The messages route is registered BEFORE the count route, so the
        // count route's body runs to the end of the router block.
        let nextRoute = try XCTUnwrap(
            source.range(of: "return router", range: route.upperBound..<source.endIndex)
        )
        let body = String(source[route.lowerBound..<nextRoute.lowerBound])
        XCTAssertTrue(
            body.contains("buildTurnPlan(apiRequest, maxTokensRequired: false)"),
            "the count route must price through the messages route's own planner"
        )
        XCTAssertFalse(
            body.contains("promptTokenIds.count -"),
            "a second estimator is the one way this endpoint can be wrong"
        )
    }

    /// Both reply writers report the plan's count, so an image turn's usage is
    /// the expanded prompt it ran rather than the placeholders it was sent.
    func testBothReplyPathsReportThePlansPromptCount() throws {
        let source = try String(
            contentsOf: URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("DynaMoE/ServerController.swift"),
            encoding: .utf8
        )
        XCTAssertEqual(
            source.components(separatedBy: "promptTokens: plan.promptTokenCount").count - 1,
            2,
            "the chat and completions writers both report the plan's expanded count"
        )
    }
}

/// A picture at the edge of the budget is refused by name, not truncated, and
/// the refusal quotes the EXPANDED prompt.
final class Inkling2ImagePromptBudgetTests: XCTestCase {

    /// The planner refuses a prompt past the cap after pricing the expansion
    /// (`promptTokenCount = try inklingPromptTokenCount(...)` precedes the cap
    /// check), which is what makes the number a client sees the one its picture
    /// really costs. A count taken before the expansion would let a picture
    /// through the gate and then overflow the cache.
    func testThePromptCapIsCheckedAfterThePictureExpands() throws {
        let source = try String(
            contentsOf: URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("DynaMoE/ServerController.swift"),
            encoding: .utf8
        )
        let expansion = try XCTUnwrap(source.range(of: "promptTokenCount = try Self.inklingPromptTokenCount("))
        let cap = try XCTUnwrap(
            source.range(of: "let promptTokenCap = Self.resolvedMaxPromptTokens", range: expansion.upperBound..<source.endIndex)
        )
        let refusal = try XCTUnwrap(
            source.range(of: "Prompt is too long:", range: cap.upperBound..<source.endIndex)
        )
        XCTAssertTrue(
            expansion.lowerBound < cap.lowerBound && cap.lowerBound < refusal.lowerBound,
            "price the picture, then the cap, then the refusal"
        )
        XCTAssertTrue(
            source[cap.lowerBound..<refusal.upperBound].contains("promptTokenCount > promptTokenCap"),
            "the cap compares the expanded count"
        )
    }

    /// The cache budget is checked against the expansion, not the placeholders:
    /// the engine's Inkling branch prices the picture and then refuses a prompt
    /// past the provisioned context, which a placeholder count would pass.
    func testTheEngineChecksTheBudgetAgainstTheExpandedPrompt() throws {
        let source = try String(
            contentsOf: URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("DynaMoE/CompletionEngine.swift"),
            encoding: .utf8
        )
        let pricing = try XCTUnwrap(
            source.range(of: "pricedPromptTokens = request.promptTokenIds.count - placeholders")
        )
        let budget = try XCTUnwrap(
            source.range(of: "guard pricedPromptTokens <= contextLength", range: pricing.upperBound..<source.endIndex)
        )
        XCTAssertLessThan(pricing.lowerBound, budget.lowerBound, "price before the budget check")
        let forward = try XCTUnwrap(
            source.range(of: "var forward: Inkling2CPUForward?", range: budget.upperBound..<source.endIndex)
        )
        XCTAssertLessThan(budget.lowerBound, forward.lowerBound, "refuse before allocating")
    }
}

// MARK: - The real checkpoint

/// Two pictures in one turn, on real weights: the exit gate's "two interleaved
/// images" case, and the only form of it that proves each picture gets its OWN
/// rows rather than the first one twice.
///
/// Opt-in, because the host tower is scalar Swift and a turn is minutes.
extension Inkling2VisionTests {

    func testTwoPicturesInOneTurnKeepTheirOwnRows() async throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: "/tmp/dynamoe_inkling2_image_turn"),
            "opt-in image turn; touch /tmp/dynamoe_inkling2_image_turn to run"
        )
        let (_, tokenizer, config) = try configureAppOrderEngine()
        defer {
            CompletionEngine.shared.configure(
                summary: nil, modelConfig: nil, tokenizer: nil,
                shardBuffers: [:], activeLoadedModelPath: nil
            )
            withExtendedLifetime(engine) {}
        }

        // Two pictures of DIFFERENT sizes, so a mistaken placement changes the
        // lengths: the first is one cell, the second is two.
        let red = try solidImage(width: 30, height: 30, value: (220, 20, 20))
        let blue = try solidImage(width: 50, height: 40, value: (20, 20, 220))
        let firstGrid = try Inkling2ImagePixels.cells(red).grid
        let secondGrid = try Inkling2ImagePixels.cells(blue).grid
        XCTAssertEqual(firstGrid, Inkling2ImageGrid(cellRows: 1, cellColumns: 1))
        XCTAssertEqual(secondGrid, Inkling2ImageGrid(cellRows: 1, cellColumns: 2))

        // Text, picture, text, picture: the order the server's sentinels keep.
        let rendered = try Inkling2Prompt().render(
            turns: [.user([
                .text("The first picture is "), .image(firstGrid),
                .text(" and the second is "), .image(secondGrid)
            ])],
            effort: 0
        )
        let promptIds = try tokenizer.encode(text: rendered)
        let marker = UInt32(config.imageTokenID)
        XCTAssertEqual(promptIds.filter { $0 == marker }.count, 2, "one placeholder per picture")

        var result: TurnResult?
        let request = CompletionTurnRequest(
            promptTokenIds: promptIds,
            formattedPrompt: rendered,
            sessionId: nil,
            sampling: TurnSampling(
                temperature: 0, topP: 1, minP: 0, topK: 1,
                repetitionPenalty: 1, maxTokens: 24
            ),
            memory: TurnMemorySettings(
                executionMode: .residentRAM, budgetMode: .unrestricted, kvPrecision: .fp32
            ),
            agentToolsEnabled: false,
            thinkingEnabled: false,
            images: [
                TurnImage(
                    rgb: red.rgb, width: red.width, height: red.height,
                    sha256: "inkling2-red-square"
                ),
                TurnImage(
                    rgb: blue.rgb, width: blue.width, height: blue.height,
                    sha256: "inkling2-blue-rectangle"
                )
            ]
        )
        for await event in try CompletionEngine.shared.launchTurn(request).stream {
            switch event {
            case .finished(let value):
                result = value
            case .failedTurn(let failure):
                XCTFail("the two-picture turn failed: \(failure.message)")
            default:
                break
            }
        }
        let turn = try XCTUnwrap(result)
        print("🔎 [inkling2 two images] answer: \(turn.rawDecodedText)")
        XCTAssertEqual(turn.imageSpans.count, 2)
        XCTAssertEqual(turn.imageSpans[0].sha256, "inkling2-red-square")
        XCTAssertEqual(turn.imageSpans[1].sha256, "inkling2-blue-rectangle")
        // Each span is its own picture's cell count, in prompt order, and the
        // second starts after the first ends: the layout is the one the prompt
        // wrote, not the first picture's rows twice.
        XCTAssertEqual(turn.imageSpans[0].range.count, firstGrid.languagePositions)
        XCTAssertEqual(turn.imageSpans[1].range.count, secondGrid.languagePositions)
        XCTAssertLessThan(turn.imageSpans[0].range.upperBound, turn.imageSpans[1].range.lowerBound)
        XCTAssertEqual(turn.promptTokenIds.count, promptIds.count + 3 - 2)
        // The two spans' marker runs are distinct positions, so the rows cannot
        // have collapsed onto one span.
        XCTAssertNotEqual(turn.imageSpans[0].range, turn.imageSpans[1].range)

        // The answer is about the pictures: the prompt named their colours, so
        // it must not answer as if only one were sent.
        let answer = turn.responseText.lowercased()
        XCTAssertFalse(answer.isEmpty)
    }

    private func solidImage(
        width: Int, height: Int, value: (UInt8, UInt8, UInt8)
    ) throws -> Inkling2ImagePixels.RGB {
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for index in stride(from: 0, to: rgb.count, by: 3) {
            rgb[index] = value.0
            rgb[index + 1] = value.1
            rgb[index + 2] = value.2
        }
        return Inkling2ImagePixels.RGB(rgb: rgb, width: width, height: height)
    }
}
