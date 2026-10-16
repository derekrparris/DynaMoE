//
//  ServerToolArgumentTests.swift
//  DynaMoE
//
//  Pure-logic coverage for how parsed tool calls are re-serialized into the
//  protocol `arguments` JSON string. The parser leaves artifacts in
//  `rawArguments` (a Swift dictionary description from the bare-JSON fallback,
//  or the whole hermetic call envelope), so the parsed argument dictionary
//  must win whenever it is available.
//

import XCTest
@testable import DynaMoE

final class ServerToolArgumentTests: XCTestCase {

    func testSerializesParsedArgumentsWhenRawIsSwiftDescription() throws {
        let call = ParsedToolCall(
            name: "get_weather",
            arguments: ["city": "Oslo", "days": 2],
            rawArguments: "[\"city\": \"Oslo\", \"days\": 2]",
            rawText: "raw",
        )
        let json = ServerController.toolArgumentsJSON(call)
        let parsed = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(json.utf8))) as? [String: Any]
        XCTAssertEqual(parsed?["city"] as? String, "Oslo")
        XCTAssertEqual(parsed?["days"] as? Int, 2)
    }

    func testSerializesParsedArgumentsWhenRawIsHermeticEnvelope() throws {
        let call = ParsedToolCall(
            name: "get_weather",
            arguments: ["city": "Oslo"],
            rawArguments: "{\"tool\":\"get_weather\",\"parameters\":{\"city\":\"Oslo\"}}",
            rawText: "raw",
        )
        let json = ServerController.toolArgumentsJSON(call)
        let parsed = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(json.utf8))) as? [String: Any]
        XCTAssertEqual(parsed?["city"] as? String, "Oslo", "the envelope keys must not leak into arguments")
        XCTAssertNil(parsed?["tool"])
        XCTAssertNil(parsed?["parameters"])
    }

    func testKeepsRawArgumentsWhenArgumentsEmptyAndRawIsJSONObject() throws {
        let call = ParsedToolCall(
            name: "ping",
            arguments: [:],
            rawArguments: "{\"region\":\"eu\"}",
            rawText: "raw",
        )
        let json = ServerController.toolArgumentsJSON(call)
        let parsed = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(json.utf8))) as? [String: Any]
        XCTAssertEqual(parsed?["region"] as? String, "eu")
    }

    func testFallsBackToEmptyObjectForUnusableArguments() {
        let call = ParsedToolCall(
            name: "ping",
            arguments: [:],
            rawArguments: "not json at all",
            rawText: "raw",
        )
        XCTAssertEqual(ServerController.toolArgumentsJSON(call), "{}")
    }
}
