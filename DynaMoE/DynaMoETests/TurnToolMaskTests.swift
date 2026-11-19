//
//  TurnToolMaskTests.swift
//  DynaMoE
//
//  Coverage for the per-turn tool schemas installed into the grammar mask. A
//  client-defined API tool is not in the in-app agent catalog, so the server
//  installs the turn's own schemas around its generation; the mask must follow
//  those schemas (name and parameter keys) rather than the built-in catalog.
//

import XCTest
@testable import DynaMoE

final class TurnToolMaskTests: XCTestCase {

    private var savedRegistry: GrammarToolRegistry!

    override func setUp() {
        super.setUp()
        savedRegistry = GrammarConstrainedSampler.shared.snapshotRegistry()
    }

    override func tearDown() {
        GrammarConstrainedSampler.shared.restoreRegistry(savedRegistry)
        GrammarConstrainedSampler.shared.reset()
        super.tearDown()
    }

    private func masked(_ text: String, tokens: [String]) -> [Bool] {
        let sampler = GrammarConstrainedSampler.shared
        sampler.isEnabled = true
        sampler.updateState(emittedText: text)
        var logits = [Float](repeating: 0, count: tokens.count)
        logits.withUnsafeMutableBufferPointer { buf in
            sampler.applyLogitMask(
                logits: buf.baseAddress!,
                vocabSize: tokens.count,
                tokenDecoder: { tokens[Int($0)] }
            )
        }
        return logits.map { $0 == -.infinity }
    }

    func testSchemaExtractsParameterAndRequiredKeys() {
        let schema = GrammarToolSchema(
            name: "get_weather",
            schemaJSON: Data(#"{"type":"object","properties":{"city":{"type":"string"},"unit":{}},"required":["city"]}"#.utf8)
        )
        XCTAssertEqual(schema.parameterKeys, ["city", "unit"])
        XCTAssertEqual(schema.requiredKeys, ["city"])

        let empty = GrammarToolSchema(name: "noop", schemaJSON: nil)
        XCTAssertTrue(empty.parameterKeys.isEmpty)
        XCTAssertTrue(empty.requiredKeys.isEmpty)
    }

    func testClientToolNameSurvivesTheMask() {
        let sampler = GrammarConstrainedSampler.shared

        // The in-app catalog (here a single built-in) does not know the client
        // tool, so the mask blocks its name at the function tag and would steer
        // the model into the built-in one instead.
        sampler.registerToolSchemas([GrammarToolSchema(name: "shell_run")])
        sampler.reset()
        XCTAssertEqual(
            masked("<tool_call><function=", tokens: ["get_weather>", "shell_run>"]),
            [true, false],
            "a client tool absent from the installed registry must be blocked"
        )

        // Installing the turn's own schemas lets the client tool be emitted and
        // blocks the built-in instead.
        sampler.registerToolSchemas([GrammarToolSchema(name: "get_weather", parameterKeys: ["city"], requiredKeys: ["city"])])
        sampler.reset()
        XCTAssertEqual(
            masked("<tool_call><function=", tokens: ["get_weather>", "shell_run>"]),
            [false, true],
            "the turn's own tool name must survive the mask"
        )

        // Parameter keys come from the same installed schema.
        sampler.reset()
        XCTAssertEqual(
            masked("<tool_call><function=get_weather><parameter=", tokens: ["city>", "command"]),
            [false, true],
            "only the client's parameter keys may be written"
        )
    }

    /// A scoped mask owns the registry for its turn. The app catalog used to be
    /// able to rebuild it from tool registration, a Skills reload or MCP while
    /// that turn was generating, which left the model calling a vocabulary the
    /// mask no longer described. The sampler refuses the registration instead.
    func testAScopedMaskRefusesTheAppCatalogUntilItIsRestored() async {
        let sampler = GrammarConstrainedSampler.shared
        sampler.registerToolSchemas([GrammarToolSchema(name: "shell_run")])
        sampler.reset()

        let installed = await GrammarConstrainedSampler.withTurnToolMask(
            [GrammarToolSchema(name: "client_only")]
        ) {
            XCTAssertTrue(GrammarConstrainedSampler.isScopedMaskInstalled)
            return GrammarConstrainedSampler.registerAppTools([])
        }

        XCTAssertFalse(installed, "the app catalog must not replace a live scoped mask")
        XCTAssertFalse(GrammarConstrainedSampler.isScopedMaskInstalled)
        sampler.reset()
        XCTAssertEqual(
            masked("<tool_call><function=", tokens: ["client_only>", "shell_run>"]),
            [true, false],
            "the registry installed before the scoped turn must be restored"
        )
    }

    /// Two scopes can overlap rather than nest, and the one that ends FIRST
    /// must not restore its own snapshot over the other. That ordering is what
    /// the engine produces: a local run's mask outlives its generation task,
    /// and a server turn can install its mask in the freed-slot window. The
    /// old depth-count restore put the run's vocabulary back when the run
    /// unwound and left it installed after the server had restored the app
    /// catalog.
    func testTheScopeThatEndsFirstDoesNotRestoreOverTheOneStillInstalled() async {
        let sampler = GrammarConstrainedSampler.shared
        sampler.registerToolSchemas([GrammarToolSchema(name: "app_tool")])

        let (firstInstalled, firstInstalledSink) = AsyncStream<Void>.makeStream()
        let (firstRelease, firstReleaseSink) = AsyncStream<Void>.makeStream()
        let (secondInstalled, secondInstalledSink) = AsyncStream<Void>.makeStream()
        let (secondRelease, secondReleaseSink) = AsyncStream<Void>.makeStream()

        func scope(
            _ name: String,
            installed: AsyncStream<Void>.Continuation,
            release: AsyncStream<Void>
        ) async {
            await GrammarConstrainedSampler.withTurnToolMask([GrammarToolSchema(name: name)]) {
                installed.yield()
                var iterator = release.makeAsyncIterator()
                _ = await iterator.next()
            }
        }

        let first = Task { await scope("first_only", installed: firstInstalledSink, release: firstRelease) }
        for await _ in firstInstalled { break }
        let second = Task { await scope("second_only", installed: secondInstalledSink, release: secondRelease) }
        for await _ in secondInstalled { break }

        // While BOTH are live the registry is their COMPOSITION, and the app
        // catalog is still replaced. The rebuild used to install one scope's
        // maps at a time, so the second scope silently erased the first one's
        // vocabulary.
        sampler.reset()
        XCTAssertEqual(
            masked("<tool_call><function=", tokens: ["first_only>", "second_only>", "app_tool>"]),
            [false, false, true],
            "two live scopes compose: both are callable and the app catalog is not"
        )

        // The FIRST scope ends while the second is still live.
        firstReleaseSink.yield()
        await first.value

        XCTAssertTrue(GrammarConstrainedSampler.isScopedMaskInstalled)
        sampler.reset()
        XCTAssertEqual(
            masked("<tool_call><function=", tokens: ["first_only>", "second_only>", "app_tool>"]),
            [true, false, true],
            "ending the first scope must leave the second scope's vocabulary installed"
        )

        secondReleaseSink.yield()
        await second.value

        XCTAssertFalse(GrammarConstrainedSampler.isScopedMaskInstalled)
        sampler.reset()
        XCTAssertEqual(
            masked("<tool_call><function=", tokens: ["first_only>", "second_only>", "app_tool>"]),
            [true, true, false],
            "the app catalog must return once every scope has ended"
        )
    }

    func testSnapshotAndRestoreRoundTripsTheRegistry() {
        let sampler = GrammarConstrainedSampler.shared
        sampler.registerToolSchemas([GrammarToolSchema(name: "before")])
        let snapshot = sampler.snapshotRegistry()
        sampler.registerToolSchemas([GrammarToolSchema(name: "after")])
        sampler.restoreRegistry(snapshot)
        sampler.reset()
        XCTAssertEqual(
            masked("<tool_call><function=", tokens: ["before>", "after>"]),
            [false, true],
            "restoring a snapshot must reinstate the previous registry"
        )
    }

    /// A chat send asserts the conversation's mode before it asks for the engine
    /// slot, so the assertion can land while a server turn holds that slot with
    /// its own client schemas installed. Rebuilding the shared registry there
    /// would replace a live turn's mask mid-generation; the send itself is
    /// refused a moment later, so nothing needs the rebuild.
    func testAModeAssertionWhileTheSlotIsHeldLeavesTheClientMaskAlone() {
        let sampler = GrammarConstrainedSampler.shared
        let harness = AgentHarness.shared
        sampler.registerToolSchemas([GrammarToolSchema(name: "client_only")])
        sampler.reset()

        guard let claim = CompletionEngine.shared.reserveTurnSlot() else {
            return XCTFail("the engine slot must be free in a unit test")
        }
        harness.setActiveMode(.plan)
        claim.release()

        sampler.reset()
        XCTAssertEqual(
            masked("<tool_call><function=", tokens: ["client_only>", "shell_run>"]),
            [false, true],
            "a mode assertion made under a claim must not rebuild the registry"
        )

        // Once the slot is free the next assertion heals it, which is what the
        // re-sync exists for.
        harness.setActiveMode(.code)
        sampler.reset()
        XCTAssertEqual(
            masked("<tool_call><function=", tokens: ["client_only>", "shell_run>"]),
            [true, false],
            "an idle assertion restores the harness surface"
        )
        harness.resetLoadedToolsToCore()
    }
}
