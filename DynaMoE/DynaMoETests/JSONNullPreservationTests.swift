//
//  JSONNullPreservationTests.swift
//  DynaMoE
//
//  The shared JSON representation (`AnyCodable`) must preserve JSON null
//  recursively. Collapsing null to an empty string rewrote client tool inputs,
//  historical Anthropic tool-use inputs, and both adapters' tool schemas
//  (including `const: null` and nullable enums) on the round trip.
//

import XCTest
@testable import DynaMoE

final class JSONNullPreservationTests: XCTestCase {

    func testAnyCodableRoundTripsNullRecursively() throws {
        let json = #"{"a":null,"b":{"c":null,"d":1},"e":[null,"x"]}"#
        let decoded = try JSONDecoder().decode(AnyCodable.self, from: Data(json.utf8))
        let encoded = try JSONEncoder().encode(decoded)
        let obj = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])

        XCTAssertTrue(obj["a"] is NSNull, "a top-level null must survive, not become \"\"")
        let b = try XCTUnwrap(obj["b"] as? [String: Any])
        XCTAssertTrue(b["c"] is NSNull, "a nested null must survive")
        XCTAssertEqual(b["d"] as? Int, 1)
        let e = try XCTUnwrap(obj["e"] as? [Any])
        XCTAssertTrue(e[0] is NSNull, "a null inside an array must survive")
        XCTAssertEqual(e[1] as? String, "x")
    }

    func testNullIsNotEqualToEmptyString() {
        XCTAssertEqual(AnyCodable(NSNull()), AnyCodable(NSNull()))
        XCTAssertNotEqual(AnyCodable(NSNull()), AnyCodable(""))
    }
}
