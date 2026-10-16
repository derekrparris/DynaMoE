//
//  ServerHardeningLogicTests.swift
//  DynaMoE
//
//  Pure-logic coverage for the server's hardening configuration resolvers:
//  queue-wait timeout and max prompt token cap defaults and overrides.
//

import XCTest
import HTTPTypes
@testable import DynaMoE

final class ServerHardeningLogicTests: XCTestCase {

    private var savedQueueTimeout: Any?
    private var savedMaxPromptTokens: Any?

    override func setUp() {
        super.setUp()
        let defaults = UserDefaults.standard
        savedQueueTimeout = defaults.object(forKey: ServerController.queueTimeoutKey)
        savedMaxPromptTokens = defaults.object(forKey: ServerController.maxPromptTokensKey)
        defaults.removeObject(forKey: ServerController.queueTimeoutKey)
        defaults.removeObject(forKey: ServerController.maxPromptTokensKey)
    }

    override func tearDown() {
        let defaults = UserDefaults.standard
        if let saved = savedQueueTimeout {
            defaults.set(saved, forKey: ServerController.queueTimeoutKey)
        } else {
            defaults.removeObject(forKey: ServerController.queueTimeoutKey)
        }
        if let saved = savedMaxPromptTokens {
            defaults.set(saved, forKey: ServerController.maxPromptTokensKey)
        } else {
            defaults.removeObject(forKey: ServerController.maxPromptTokensKey)
        }
        super.tearDown()
    }

    func testQueueWaitTimeoutDefaultsAndOverrides() {
        // Absent (first run) → 120 seconds.
        XCTAssertEqual(ServerController.resolvedQueueWaitTimeout, 120)

        UserDefaults.standard.set(5.0, forKey: ServerController.queueTimeoutKey)
        XCTAssertEqual(ServerController.resolvedQueueWaitTimeout, 5)

        // Explicit zero disables the timeout (wait indefinitely).
        UserDefaults.standard.set(0.0, forKey: ServerController.queueTimeoutKey)
        XCTAssertNil(ServerController.resolvedQueueWaitTimeout)
    }

    func testMaxPromptTokensDefaultsAndOverrides() {
        // Absent (first run) → 262,144 tokens.
        XCTAssertEqual(ServerController.resolvedMaxPromptTokens, 262_144)

        UserDefaults.standard.set(4096, forKey: ServerController.maxPromptTokensKey)
        XCTAssertEqual(ServerController.resolvedMaxPromptTokens, 4096)

        // Explicit zero lifts the cap entirely.
        UserDefaults.standard.set(0, forKey: ServerController.maxPromptTokensKey)
        XCTAssertEqual(ServerController.resolvedMaxPromptTokens, Int.max)
    }

    func testServerAPIErrorFactoryStatuses() {
        XCTAssertEqual(ServerAPIError.badRequest("m").status, .badRequest)
        XCTAssertEqual(ServerAPIError.badRequest("m").wireType, .invalidRequest)
        XCTAssertEqual(ServerAPIError.unauthorized("m").status, .unauthorized)
        XCTAssertEqual(ServerAPIError.unauthorized("m").wireType, .authentication)
        XCTAssertEqual(ServerAPIError.requestTimeout("m").status, .requestTimeout)
        XCTAssertEqual(ServerAPIError.serviceUnavailable("m").status, .serviceUnavailable)
        XCTAssertEqual(ServerAPIError.serviceUnavailable("m").wireType, .overloaded)
        XCTAssertEqual(ServerAPIError.internalError("m").status, .internalServerError)
        XCTAssertEqual(ServerAPIError.internalError("m").wireType, .api)
    }
}
