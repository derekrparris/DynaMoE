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
}
