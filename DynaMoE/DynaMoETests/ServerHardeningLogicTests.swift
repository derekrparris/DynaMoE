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
        XCTAssertNil(ServerAPIError.badRequest("m").code)
        XCTAssertEqual(ServerAPIError.unauthorized("m").status, .unauthorized)
        XCTAssertEqual(ServerAPIError.unauthorized("m").wireType, .authentication)
        XCTAssertEqual(ServerAPIError.requestTimeout("m").status, .requestTimeout)
        XCTAssertEqual(ServerAPIError.serviceUnavailable("m").status, .serviceUnavailable)
        XCTAssertEqual(ServerAPIError.serviceUnavailable("m").wireType, .overloaded)
        XCTAssertEqual(ServerAPIError.internalError("m").status, .internalServerError)
        XCTAssertEqual(ServerAPIError.internalError("m").wireType, .api)
    }

    func testKVCapacityRejectionCarriesContextLengthCode() {
        // A prompt that cannot fit the KV cache is a context-length error, not a
        // generic bad request: clients classify on the code. The message comes
        // from the real builder, so this also pins that a budget/reserve bind is
        // never reported as prompt length. 2,000 tokens against the 910 rows the
        // budget affords is that bind: the reserve was capped, not the prompt.
        let message = KVCacheProvisioning.capacityMessage(
            promptTokens: 2_000,
            provisioned: 910,
            affordableRows: 910,
            budgetGB: 0.25,
            precision: .fp16,
            isAgentSession: true,
            sessionFloor: 8_192,
            modelFootprintGB: 8.75,
            physicalGB: 16
        )
        let error = ServerAPIError.contextLengthExceeded(message)
        XCTAssertEqual(error.status, .badRequest)
        XCTAssertEqual(error.wireType, .invalidRequest)
        XCTAssertEqual(error.code, "context_length_exceeded")

        let envelope = try? JSONEncoder().encode(OpenAIErrorEnvelope(
            error: .init(message: error.message, type: error.wireType.rawValue, code: error.code)
        ))
        let json = envelope.flatMap { String(data: $0, encoding: .utf8) } ?? ""
        XCTAssertTrue(json.contains("\"code\":\"context_length_exceeded\""), "envelope lost the code: \(json)")
        XCTAssertFalse(
            error.message.contains("Prompt too long"),
            "the message must not blame prompt length for a budget/reserve bind"
        )
        XCTAssertTrue(error.message.contains("Prompt does not fit"), error.message)
        XCTAssertTrue(error.message.contains("capped by the budget to 910"), error.message)
    }

    func testLaunchErrorMappingClassifiesKVCapacityRejections() {
        let mapped = ServerController.wireLaunchError(CompletionLaunchError.promptTooLong("context does not fit"))
        XCTAssertEqual(mapped.status, .badRequest)
        XCTAssertEqual(mapped.wireType, .invalidRequest)
        XCTAssertEqual(mapped.code, "context_length_exceeded")

        // Engine-state failures stay 503: the client cannot fix them by editing
        // the request.
        let busy = ServerController.wireLaunchError(CompletionLaunchError.engineNotReady("busy"))
        XCTAssertEqual(busy.status, .serviceUnavailable)
        XCTAssertNil(busy.code)
    }
}
