//
//  KeychainBackendLogicTests.swift
//  DynaMoETests
//
//  The rules that keep the launch path from asking for keychain access, and the
//  import rule that carries an older credential forward exactly once.
//
//  Everything here except the round-trip cases is a pure function of the query
//  builder, the status classifier, the backend resolver or the migration
//  planner, so no dialog can appear and no real credential is touched.
//

import XCTest
import Security
@testable import DynaMoE

// MARK: - Query builders

final class KeychainQueryLogicTests: XCTestCase {

    private let dp = KeychainBackend.dataProtection(accessGroup: "RMY84Y8G7M.DRP.DynaMoE")
    private let legacy = KeychainBackend.legacy

    func testTheExistenceQueryNeverAsksForTheSecret() {
        for backend in [dp, legacy] {
            let query = SecretStoreQuery.existence(service: "svc", account: "acct", backend: backend)
            XCTAssertNil(
                query[kSecReturnData as String],
                "an existence check that asks for the secret data is what makes the launch path prompt"
            )
            XCTAssertEqual(query[kSecReturnAttributes as String] as? Bool, true)
            XCTAssertEqual(query[kSecMatchLimit as String] as? String, kSecMatchLimitOne as String)
        }
    }

    func testEveryQueryNamesItsBackendInBothDirections() {
        let dpRead = SecretStoreQuery.read(service: "svc", account: "acct", backend: dp)
        let legacyRead = SecretStoreQuery.read(service: "svc", account: "acct", backend: legacy)
        XCTAssertEqual(dpRead[kSecUseDataProtectionKeychain as String] as? Bool, true)
        XCTAssertEqual(legacyRead[kSecUseDataProtectionKeychain as String] as? Bool, false)
        XCTAssertEqual(dpRead[kSecAttrAccessGroup as String] as? String, "RMY84Y8G7M.DRP.DynaMoE")
        XCTAssertNil(
            legacyRead[kSecAttrAccessGroup as String],
            "an access group on a file-keychain query is the mismatch that makes a legacy item read as not found"
        )

        let legacyDelete = SecretStoreQuery.delete(service: "svc", account: "acct", backend: legacy)
        XCTAssertEqual(legacyDelete[kSecUseDataProtectionKeychain as String] as? Bool, false)
        XCTAssertEqual(legacyDelete[kSecAttrAccount as String] as? String, "acct")
    }

    func testTheListingQueryIsAttributesOnly() {
        let query = SecretStoreQuery.listing(service: "svc", backend: dp)
        XCTAssertNil(query[kSecReturnData as String])
        XCTAssertEqual(query[kSecReturnAttributes as String] as? Bool, true)
        XCTAssertEqual(query[kSecMatchLimit as String] as? String, kSecMatchLimitAll as String)
    }

    func testOnlyTheDataProtectionItemCarriesAnAccessibilityClass() {
        let data = Data("x".utf8)
        let dpAdd = SecretStoreQuery.add(
            service: "svc", account: "acct", data: data,
            label: "l", description: "d", backend: dp
        )
        let legacyAdd = SecretStoreQuery.add(
            service: "svc", account: "acct", data: data,
            label: "l", description: "d", backend: legacy
        )
        XCTAssertEqual(dpAdd[kSecAttrAccessible as String] as? String, kSecAttrAccessibleAfterFirstUnlock as String)
        XCTAssertNil(legacyAdd[kSecAttrAccessible as String])
        XCTAssertEqual((dpAdd[kSecValueData as String] as? Data), data)
    }
}

// MARK: - Status classification

final class SecretAccessClassificationTests: XCTestCase {

    func testMissingIsNotADenial() {
        XCTAssertEqual(SecretAccessClassifier.failure(errSecItemNotFound), .missing)
        XCTAssertTrue(SecretAccessClassifier.failure(errSecItemNotFound).isMissing)
        XCTAssertFalse(SecretAccessClassifier.failure(errSecItemNotFound).needsAuthorization)
    }

    func testEveryRefusalIsClassifiedAsANeedsAuthorizationState() {
        for status in [errSecAuthFailed, errSecUserCanceled, errSecInteractionNotAllowed, errSecNoAccessForItem] {
            let result = SecretAccessClassifier.failure(status)
            XCTAssertEqual(result, .accessDenied(status), "status \(status)")
            XCTAssertTrue(result.needsAuthorization)
        }
    }

    func testAMissingEntitlementIsAnUnavailableStateRatherThanAMissingVault() {
        let result = SecretAccessClassifier.failure(errSecMissingEntitlement)
        XCTAssertEqual(result, .unavailable(errSecMissingEntitlement))
        XCTAssertTrue(result.needsAuthorization)
        XCTAssertTrue(SecretAccessClassifier.indicatesMissingEntitlement(errSecMissingEntitlement))
        XCTAssertFalse(SecretAccessClassifier.indicatesMissingEntitlement(errSecSuccess))
    }

    func testOnlyFoundCarriesAValue() {
        XCTAssertEqual(SecretAccessResult.found("sk-1").value, "sk-1")
        XCTAssertNil(SecretAccessResult.missing.value)
        XCTAssertNil(SecretAccessResult.accessDenied(errSecAuthFailed).value)
    }
}

// MARK: - Backend resolution

final class SecretStoreBackendResolverTests: XCTestCase {

    private let resolver = SecretStoreBackendResolver.shared

    override func tearDown() {
        resolver.overrideForTesting(nil)
        resolver.entitlementGroupsProvider = nil
        resolver.capabilityProbe = nil
        super.tearDown()
    }

    private func resolve(groups: [String], probe: OSStatus) -> KeychainBackend {
        resolver.overrideForTesting(nil)
        resolver.entitlementGroupsProvider = { groups }
        resolver.capabilityProbe = { _ in probe }
        return resolver.backend()
    }

    func testABuildWithoutTheEntitlementStaysOnTheLegacyKeychain() {
        XCTAssertEqual(resolve(groups: [], probe: errSecSuccess), .legacy)
    }

    func testAnEntitledBuildThatCanWriteUsesTheDataProtectionKeychain() {
        XCTAssertEqual(
            resolve(groups: ["RMY84Y8G7M.DRP.DynaMoE"], probe: errSecSuccess),
            .dataProtection(accessGroup: "RMY84Y8G7M.DRP.DynaMoE")
        )
    }

    func testTheProbeFailingForMissingEntitlementFallsBackRatherThanFailing() {
        XCTAssertEqual(resolve(groups: ["RMY84Y8G7M.DRP.DynaMoE"], probe: errSecMissingEntitlement), .legacy)
        XCTAssertEqual(resolve(groups: ["RMY84Y8G7M.DRP.DynaMoE"], probe: errSecParam), .legacy)
    }

    func testAProbeThatFindsAnExistingSentinelStillCounts() {
        XCTAssertEqual(
            resolve(groups: ["RMY84Y8G7M.DRP.DynaMoE"], probe: errSecDuplicateItem),
            .dataProtection(accessGroup: "RMY84Y8G7M.DRP.DynaMoE")
        )
    }

    func testTheChosenGroupIsTheOneForThisBundle() {
        let resolved = resolve(
            groups: ["RMY84Y8G7M.some.other.app", "RMY84Y8G7M.DRP.DynaMoE"],
            probe: errSecSuccess
        )
        XCTAssertEqual(resolved.accessGroup, "RMY84Y8G7M.DRP.DynaMoE")
    }

    func testTheDecisionIsCachedPerProcess() {
        resolver.overrideForTesting(nil)
        resolver.entitlementGroupsProvider = { ["RMY84Y8G7M.DRP.DynaMoE"] }
        var probeCount = 0
        resolver.capabilityProbe = { _ in
            probeCount += 1
            return errSecSuccess
        }
        _ = resolver.backend()
        _ = resolver.backend()
        _ = resolver.backend()
        XCTAssertEqual(probeCount, 1, "the capability probe must not run per keychain call")
    }
}

// MARK: - Migration planning

final class KeychainMigrationPlanningTests: XCTestCase {

    func testACredentialAlreadyInTheCurrentKeychainIsLeftAlone() {
        XCTAssertEqual(
            KeychainMigration.decision(current: .found("new"), legacy: .found("old")),
            .alreadyPresent
        )
    }

    func testALegacyValueIsMigrated() {
        XCTAssertEqual(
            KeychainMigration.decision(current: .missing, legacy: .found("old")),
            .migrate("old")
        )
    }

    func testAnEmptyLegacyItemIsATombstoneRatherThanACredential() {
        XCTAssertEqual(
            KeychainMigration.decision(current: .missing, legacy: .found("")),
            .tombstone
        )
    }

    func testARefusedReadIsReportedAsAuthorizationRatherThanFailure() {
        XCTAssertEqual(
            KeychainMigration.decision(current: .missing, legacy: .accessDenied(errSecUserCanceled)),
            .needsAuthorization(errSecUserCanceled)
        )
    }

    func testAnUnreadableLegacyItemFailsWithoutCopyingAnything() {
        XCTAssertEqual(
            KeychainMigration.decision(current: .missing, legacy: .unavailable(errSecInteractionNotAllowed)),
            .failed(errSecInteractionNotAllowed)
        )
        XCTAssertEqual(
            KeychainMigration.decision(current: .missing, legacy: .missing),
            .failed(errSecItemNotFound)
        )
    }

    func testOutcomeSuccessCoversEverythingThatLeftTheCredentialUsable() {
        XCTAssertTrue(KeychainImportOutcome.imported(account: "a").succeeded)
        XCTAssertTrue(KeychainImportOutcome.tombstoneRemoved(account: "a").succeeded)
        XCTAssertTrue(KeychainImportOutcome.alreadyPresent(account: "a").succeeded)
        XCTAssertFalse(KeychainImportOutcome.needsAuthorization(account: "a", errSecUserCanceled).succeeded)
        XCTAssertFalse(KeychainImportOutcome.failed(account: "a", errSecDecode).succeeded)
        XCTAssertTrue(
            KeychainImportOutcome.needsAuthorization(account: "a", errSecUserCanceled)
                .message.contains("Always Allow"),
            "a refused read has to tell the user what to click"
        )
    }

    func testPendingLabelsAreDerivedFromTheAccountShape() {
        XCTAssertEqual(
            PendingCredential(service: CloudSecretStore.defaultService, account: "dynamoe.cloud.synthetic.apikey")
                .displayLabel,
            "Synthetic API key"
        )
        XCTAssertEqual(
            PendingCredential(service: MCPSecretStore.defaultService, account: "dynamoe.mcp.linear.Authorization")
                .displayLabel,
            "MCP · Linear · Authorization"
        )
    }

    func testNothingIsPendingWhenTheCurrentKeychainIsTheLegacyOne() {
        let resolver = SecretStoreBackendResolver.shared
        resolver.overrideForTesting(.legacy)
        defer {
            resolver.overrideForTesting(nil)
            resolver.entitlementGroupsProvider = nil
            resolver.capabilityProbe = nil
        }
        XCTAssertTrue(
            KeychainMigration.pending(services: ["DRP.DynaMoE.tests.import"]).isEmpty,
            "an unentitled build reads the legacy keychain directly, so there is nothing to import"
        )
    }
}

// MARK: - Round trip

final class KeychainSecretStoreTests: XCTestCase {

    private var service = ""

    override func setUp() {
        super.setUp()
        service = "DRP.DynaMoE.tests.keychain.\(UUID().uuidString)"
    }

    override func tearDown() {
        KeychainSecretStore(service: service).delete(account: "acct")
        super.tearDown()
    }

    func testReadDistinguishesAStoredSecretFromAMissingOne() throws {
        let store = KeychainSecretStore(service: service)
        XCTAssertEqual(store.read(account: "acct"), .missing)

        do {
            try store.set("sk-test", account: "acct", label: "l", description: "d")
        } catch {
            throw XCTSkip("Keychain is not reachable from this test host: \(error.localizedDescription)")
        }

        XCTAssertEqual(store.read(account: "acct"), .found("sk-test"))
        XCTAssertTrue(store.exists(account: "acct"))
        XCTAssertEqual(store.accounts(includingLegacy: true), ["acct"])

        store.delete(account: "acct")
        XCTAssertEqual(store.read(account: "acct"), .missing)
        XCTAssertFalse(store.exists(account: "acct"))
    }

    func testAnEmptySecretIsRefused() {
        XCTAssertThrowsError(
            try KeychainSecretStore(service: service).set("", account: "acct", label: "l", description: "d")
        ) { error in
            XCTAssertEqual(error as? SecretStoreFailure, .emptySecret)
        }
    }

    /// Exercises the attributes-only query against the real keychain: if the
    /// existence check ever needed the secret data, this call is the one that
    /// would put a dialog on screen during a launch-path query.
    func testExistenceOnAnAbsentAccountIsFalseAndReadsAsMissing() {
        let store = KeychainSecretStore(service: service)
        XCTAssertFalse(store.exists(account: "absent"))
        XCTAssertEqual(store.read(account: "absent"), .missing)
        XCTAssertTrue(store.accounts(includingLegacy: true).isEmpty)
    }

    func testTheBackendDecisionIsStableWithinOneProcess() {
        let resolver = SecretStoreBackendResolver.shared
        XCTAssertEqual(resolver.backend(), resolver.backend())
    }
}

// MARK: - Store facades

final class MCPSecretStoreFacadeTests: XCTestCase {

    func testTheInstanceUsesItsOwnService() throws {
        let service = "DRP.DynaMoE.tests.mcp.\(UUID().uuidString)"
        let store = MCPSecretStore(service: service)
        defer { KeychainSecretStore(service: service).delete(account: "dynamoe.mcp.tests.roundtrip") }

        do {
            try store.set("token", account: "dynamoe.mcp.tests.roundtrip")
        } catch {
            throw XCTSkip("Keychain is not reachable from this test host: \(error.localizedDescription)")
        }
        XCTAssertEqual(store.read(account: "dynamoe.mcp.tests.roundtrip").value, "token")

        store.deleteAll(serverKey: "tests")
        XCTAssertEqual(store.read(account: "dynamoe.mcp.tests.roundtrip"), .missing)
    }

    func testAnUnstoredTokenStaysUnresolvedAndIsNotReportedAsDenied() {
        let token = MCPTokenExpander.keychainToken(account: "dynamoe.mcp.tests.absent-\(UUID().uuidString)")
        let resolution = MCPTokenExpander.expand("Bearer \(token)", workspacePath: nil)
        XCTAssertEqual(resolution.unresolved, [token])
        XCTAssertTrue(resolution.denied.isEmpty)
        XCTAssertTrue(resolution.needsImport.isEmpty)
        XCTAssertEqual(resolution.value, "Bearer ")
    }

    func testAWorkspaceTokenStillExpandsWithoutAKeychainRead() {
        let resolution = MCPTokenExpander.expand("/repos/${workspace}/app", workspacePath: "/tmp/ws")
        XCTAssertEqual(resolution.value, "/repos//tmp/ws/app")
        XCTAssertTrue(resolution.unresolved.isEmpty)
        XCTAssertTrue(resolution.denied.isEmpty)
    }
}

// MARK: - Token outcome

final class MCPKeychainTokenOutcomeTests: XCTestCase {

    func testAReadableSecretExpands() {
        XCTAssertEqual(
            MCPTokenExpander.keychainTokenOutcome(read: .found("sk-1"), existsInLegacyKeychain: false),
            .value("sk-1")
        )
    }

    func testNothingAnywhereIsUnset() {
        XCTAssertEqual(
            MCPTokenExpander.keychainTokenOutcome(read: .missing, existsInLegacyKeychain: false),
            .unset
        )
    }

    func testACredentialWaitingInTheOlderKeychainAsksToBeImported() {
        XCTAssertEqual(
            MCPTokenExpander.keychainTokenOutcome(read: .missing, existsInLegacyKeychain: true),
            .needsImport,
            "after an update the item is in the keychain the app no longer reads, and 'add a key' would be wrong advice"
        )
    }

    func testARefusedReadIsNeverReportedAsAnImport() {
        XCTAssertEqual(
            MCPTokenExpander.keychainTokenOutcome(read: .accessDenied(errSecUserCanceled), existsInLegacyKeychain: true),
            .denied
        )
        XCTAssertEqual(
            MCPTokenExpander.keychainTokenOutcome(read: .unavailable(errSecMissingEntitlement), existsInLegacyKeychain: false),
            .denied
        )
    }

    func testAnEmptySecretFallsBackToTheKeychainItCameFrom() {
        XCTAssertEqual(
            MCPTokenExpander.keychainTokenOutcome(read: .found(""), existsInLegacyKeychain: true),
            .needsImport
        )
        XCTAssertEqual(
            MCPTokenExpander.keychainTokenOutcome(read: .found(""), existsInLegacyKeychain: false),
            .unset
        )
    }
}
