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

// MARK: - Test support

/// Statuses that mean "this test host has no usable keychain", as distinct from
/// "the keychain rejected this write". Only the first kind may skip a case:
/// making every write failure a skip is how a broken write path reports green.
let keychainUnreachableStatuses: Set<OSStatus> = [
    errSecMissingEntitlement,
    errSecInteractionNotAllowed,
    errSecNotAvailable,
    errSecUserCanceled,
]

func keychainStatus(of error: Error) -> OSStatus? {
    if let failure = error as? SecretStoreFailure, case .status(let status) = failure {
        return status
    }
    if let failure = error as? CloudKeychainError, case .status(let status) = failure {
        return status
    }
    return nil
}

/// Skips when the host has no keychain, and rethrows anything else so a genuine
/// write failure fails the test instead of hiding behind a skip.
func skipIfKeychainUnreachable(_ error: Error) throws {
    guard let status = keychainStatus(of: error),
          keychainUnreachableStatuses.contains(status)
    else { throw error }
    throw XCTSkip("Keychain is not reachable from this test host: \(KeychainSecretStore.describe(status))")
}

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
        resolver.probeAttempts = 2
        resolver.probeRetryDelayMicroseconds = 25_000
        super.tearDown()
    }

    private func resolve(groups: [String], probe: OSStatus) -> KeychainBackend {
        resolver.overrideForTesting(nil)
        resolver.entitlementGroupsProvider = { groups }
        resolver.capabilityProbe = { _ in probe }
        return resolver.backend()
    }

    /// A scripted probe that never makes a test wait for the retry delay.
    private func resolve(
        groups: [String] = ["RMY84Y8G7M.DRP.DynaMoE"],
        probe: @escaping () -> OSStatus
    ) -> KeychainBackend {
        resolver.overrideForTesting(nil)
        resolver.entitlementGroupsProvider = { groups }
        resolver.capabilityProbe = { _ in probe() }
        resolver.probeRetryDelayMicroseconds = 0
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

    /// A keychain that is still coming up at launch answers with an interaction
    /// status; the answer it gives a moment later is the real one.
    func testATransientProbeFailureIsRetriedBeforeFallingBack() {
        var statuses = [errSecInteractionNotAllowed, errSecSuccess]
        let resolved = resolve {
            statuses.isEmpty ? errSecInteractionNotAllowed : statuses.removeFirst()
        }
        XCTAssertEqual(
            resolved,
            .dataProtection(accessGroup: "RMY84Y8G7M.DRP.DynaMoE"),
            "a status that could clear is not a reason to spend the session on the legacy keychain"
        )
        XCTAssertTrue(statuses.isEmpty)
    }

    /// The defect this pins: a transient launch failure used to pin the whole
    /// process to the legacy store, so secrets saved that session landed where
    /// the next launch could not read them and looked like they needed importing.
    func testAnInconclusiveProbeIsNotRememberedForTheProcess() {
        resolver.overrideForTesting(nil)
        resolver.entitlementGroupsProvider = { ["RMY84Y8G7M.DRP.DynaMoE"] }
        resolver.probeRetryDelayMicroseconds = 0
        var probeCount = 0
        resolver.capabilityProbe = { _ in
            probeCount += 1
            return errSecInteractionNotAllowed
        }
        XCTAssertEqual(resolver.backend(), .legacy)
        XCTAssertEqual(probeCount, 2, "one resolution asks twice before calling the answer inconclusive")
        XCTAssertEqual(resolver.backend(), .legacy)
        XCTAssertEqual(probeCount, 4, "an inconclusive answer is asked again rather than cached")
    }

    func testAMissingEntitlementIsDefinitiveAndCached() {
        resolver.overrideForTesting(nil)
        resolver.entitlementGroupsProvider = { ["RMY84Y8G7M.DRP.DynaMoE"] }
        resolver.probeRetryDelayMicroseconds = 0
        var probeCount = 0
        resolver.capabilityProbe = { _ in
            probeCount += 1
            return errSecMissingEntitlement
        }
        XCTAssertEqual(resolver.backend(), .legacy)
        XCTAssertEqual(resolver.backend(), .legacy)
        XCTAssertEqual(probeCount, 1, "the signature will not change, so asking again cannot help")
    }
}

// MARK: - Migration planning

final class KeychainMigrationPlanningTests: XCTestCase {

    func testACredentialAlreadyInTheCurrentKeychainIsLeftAlone() {
        XCTAssertEqual(
            KeychainMigration.decision(current: .found("new"), legacy: .found("old")),
            .alreadyPresent,
            "differing values are a real conflict, not a leftover to delete"
        )
    }

    /// An import that wrote the new copy but was interrupted before the reap (or
    /// whose delete failed) leaves both keychains holding the same value. Nothing
    /// else would ever remove the legacy copy, which is the one thing this change
    /// exists to get rid of.
    func testMatchingCopiesAreALeftoverToReapRatherThanAConflict() {
        XCTAssertEqual(
            KeychainMigration.decision(current: .found("same"), legacy: .found("same")),
            .duplicate
        )
    }

    func testAnEmptyCurrentValueLetsTheStoredValueWin() {
        XCTAssertEqual(
            KeychainMigration.decision(current: .found(""), legacy: .found("old")),
            .migrate("old")
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
        XCTAssertTrue(KeychainImportOutcome.duplicateRemoved(account: "a").succeeded)
        XCTAssertTrue(KeychainImportOutcome.alreadyPresent(account: "a").succeeded)
        XCTAssertFalse(KeychainImportOutcome.needsAuthorization(account: "a", errSecUserCanceled).succeeded)
        XCTAssertFalse(KeychainImportOutcome.failed(account: "a", errSecDecode).succeeded)
        XCTAssertTrue(
            KeychainImportOutcome.needsAuthorization(account: "a", errSecUserCanceled)
                .message.contains("Always Allow"),
            "a refused read has to tell the user what to click"
        )
        XCTAssertTrue(
            KeychainImportOutcome.duplicateRemoved(account: "a").message.contains("already had it"),
            "removing a leftover copy has to say the credential was never at risk"
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

    /// Replaces a vacuous version of this test, which overrode the resolver to
    /// `.legacy` and then asserted emptiness: in a Debug build the resolver
    /// answers `.legacy` anyway, so the short-circuit being asserted was the
    /// only reason it passed. The rule is a pure function of the inventory now.
    func testNothingIsPendingForABuildWithoutTheEntitlement() {
        let inventory = KeychainMigrationInventory(
            usesDataProtection: false,
            legacyAccounts: ["dynamoe.cloud.synthetic.apikey"]
        )
        XCTAssertTrue(
            KeychainMigration.pendingCredentials(
                service: CloudSecretStore.defaultService,
                inventory: inventory
            ).isEmpty,
            "an unentitled build reads the legacy keychain directly, so there is nothing to import"
        )
    }

    func testEveryLegacyAccountIsOfferedIncludingOneTheCurrentKeychainHolds() {
        let inventory = KeychainMigrationInventory(
            usesDataProtection: true,
            legacyAccounts: ["dynamoe.cloud.b.apikey", "dynamoe.cloud.a.apikey"]
        )
        XCTAssertEqual(
            KeychainMigration.pendingCredentials(
                service: CloudSecretStore.defaultService,
                inventory: inventory
            ).map(\.id),
            [
                "DRP.DynaMoE.cloud|dynamoe.cloud.b.apikey",
                "DRP.DynaMoE.cloud|dynamoe.cloud.a.apikey"
            ],
            "every legacy copy is listed: one whose value matches is a leftover the import has to remove"
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
            try skipIfKeychainUnreachable(error)
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
            try skipIfKeychainUnreachable(error)
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

// MARK: - MCP credential issues

/// The two user-visible payoff paths of the keychain split: a server that could
/// not read its credential says why, and the three reasons stay distinct
/// because the fix differs for each.
final class MCPCredentialIssueTests: XCTestCase {

    func testAnUnimportedCredentialNamesTheImportRatherThanAMissingKey() {
        let issues = MCPConfigResolver.credentialIssues(
            key: "linear",
            unresolved: [],
            denied: [],
            needsImport: ["${keychain:dynamoe.mcp.linear.Authorization}"]
        )
        XCTAssertEqual(issues.count, 1)
        XCTAssertEqual(issues.first?.severity, .error)
        XCTAssertEqual(issues.first?.serverKey, "linear")
        let message = issues.first?.message ?? ""
        XCTAssertTrue(message.contains("not imported yet"))
        XCTAssertTrue(message.contains("Import stored credentials"))
        XCTAssertFalse(
            message.contains("Unresolved credential"),
            "'add a key' is the wrong advice for a credential that exists somewhere"
        )
    }

    func testARefusedCredentialNamesAuthorizationAndDeduplicatesItsTokens() {
        let issues = MCPConfigResolver.credentialIssues(
            key: "linear",
            unresolved: [],
            denied: ["${keychain:b}", "${keychain:a}", "${keychain:a}"],
            needsImport: []
        )
        XCTAssertEqual(issues.count, 1)
        let message = issues.first?.message ?? ""
        XCTAssertTrue(message.contains("Always Allow"))
        XCTAssertTrue(message.contains("${keychain:a}, ${keychain:b}"), "tokens are deduplicated and sorted")
    }

    func testAMissingCredentialAsksForAKey() {
        let issues = MCPConfigResolver.credentialIssues(
            key: "linear",
            unresolved: ["${env:LINEAR_TOKEN}"],
            denied: [],
            needsImport: []
        )
        XCTAssertEqual(issues.count, 1)
        XCTAssertTrue(issues.first?.message.contains("Unresolved credential token(s)") == true)
    }

    func testEveryOutcomeIsReportedSeparatelyOnTheSameServer() {
        let issues = MCPConfigResolver.credentialIssues(
            key: "linear",
            unresolved: ["${env:LINEAR_TOKEN}"],
            denied: ["${keychain:a}"],
            needsImport: ["${keychain:b}"]
        )
        XCTAssertEqual(issues.count, 3)
        XCTAssertTrue(issues.allSatisfy { $0.severity == .error && $0.serverKey == "linear" })
    }

    func testNoOutcomeMeansNoIssue() {
        XCTAssertTrue(
            MCPConfigResolver.credentialIssues(key: "linear", unresolved: [], denied: [], needsImport: []).isEmpty
        )
    }
}
