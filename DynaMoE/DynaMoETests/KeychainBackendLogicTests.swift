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

    /// `errSecParam` is what a malformed query returns, so it is not evidence
    /// about the signature. Caching it would report a query-construction bug as
    /// "this build cannot use the keychain" for the rest of the process.
    func testAMalformedQueryStatusIsNotAnEntitlementFact() {
        XCTAssertFalse(SecretAccessClassifier.indicatesMissingEntitlement(errSecParam))
        XCTAssertTrue(SecretAccessClassifier.indicatesMalformedQuery(errSecParam))
        XCTAssertFalse(SecretAccessClassifier.indicatesMalformedQuery(errSecMissingEntitlement))

        resolver.overrideForTesting(nil)
        resolver.entitlementGroupsProvider = { ["RMY84Y8G7M.DRP.DynaMoE"] }
        resolver.probeRetryDelayMicroseconds = 0
        var probeCount = 0
        resolver.capabilityProbe = { _ in
            probeCount += 1
            return errSecParam
        }
        XCTAssertEqual(resolver.backend(), .legacy)
        XCTAssertEqual(resolver.backend(), .legacy)
        XCTAssertEqual(
            probeCount, 4,
            "an answer that points at the query must stay provisional, not be remembered"
        )
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
        XCTAssertTrue(message.contains("Stored Credentials"), "names the card that fixes it")
        XCTAssertTrue(message.contains("press Import"))
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

// MARK: - The shipped keychain path

/// An in-memory stand-in for the four `SecItem` primitives, with each backend
/// as its own namespace.
///
/// The backend that ships can only be WRITTEN from a build carrying
/// `keychain-access-groups`, and the test host is an unentitled Debug app, so
/// without this seam every case below would have to skip the path users
/// actually run.
nonisolated final class FakeKeychainItemAccess: KeychainItemAccess, @unchecked Sendable {
    struct Entry: Equatable {
        var data: Data
        var label: String
        var description: String
        var accessible: String?
        var group: String?
    }

    private let lock = NSLock()
    private var entries: [String: Entry] = [:]
    /// Every attribute dictionary this fake was asked to add, so a test can
    /// assert what the write actually said.
    private(set) var written: [[String: Any]] = []

    private static func isDataProtection(_ query: [String: Any]) -> Bool {
        query[kSecUseDataProtectionKeychain as String] as? Bool ?? false
    }

    private static func key(_ query: [String: Any]) -> String? {
        guard let service = query[kSecAttrService as String] as? String else { return nil }
        let owner = isDataProtection(query) ? "dp" : "legacy"
        if let account = query[kSecAttrAccount as String] as? String {
            return "\(owner)|\(service)|\(account)"
        }
        return "\(owner)|\(service)|"
    }

    private func entry(_ query: [String: Any]) -> Entry? {
        lock.lock()
        defer { lock.unlock() }
        guard let key = Self.key(query) else { return nil }
        return entries[key]
    }

    func copyMatching(_ query: [String: Any]) -> KeychainItemLookup {
        guard let service = query[kSecAttrService as String] as? String else {
            // What the real keychain answers for a query missing its class or
            // service: a malformed request, not a missing item.
            return KeychainItemLookup(status: errSecParam)
        }
        let wantsData = query[kSecReturnData as String] as? Bool ?? false
        let returnsAll = (query[kSecMatchLimit as String] as? String) == (kSecMatchLimitAll as String)

        lock.lock()
        defer { lock.unlock() }
        let owner = Self.isDataProtection(query) ? "dp" : "legacy"
        let prefix = "\(owner)|\(service)|"
        let matching = entries.filter { $0.key.hasPrefix(prefix) }

        if returnsAll {
            let attributes = matching.compactMap { pair -> [String: Any]? in
                let account = String(pair.key.dropFirst(prefix.count))
                return account.isEmpty ? nil : [kSecAttrAccount as String: account]
            }
            return attributes.isEmpty
                ? KeychainItemLookup(status: errSecItemNotFound)
                : KeychainItemLookup(status: errSecSuccess, value: attributes as NSArray)
        }

        guard let account = query[kSecAttrAccount as String] as? String,
              let found = entries["\(prefix)\(account)"]
        else { return KeychainItemLookup(status: errSecItemNotFound) }

        if wantsData { return KeychainItemLookup(status: errSecSuccess, value: found.data as NSData) }
        return KeychainItemLookup(
            status: errSecSuccess,
            value: [kSecAttrAccount as String: account] as NSDictionary
        )
    }

    func add(_ attributes: [String: Any]) -> OSStatus {
        guard let key = Self.key(attributes),
              let data = attributes[kSecValueData as String] as? Data
        else { return errSecParam }
        lock.lock()
        defer { lock.unlock() }
        written.append(attributes)
        guard entries[key] == nil else { return errSecDuplicateItem }
        entries[key] = Entry(
            data: data,
            label: attributes[kSecAttrLabel as String] as? String ?? "",
            description: attributes[kSecAttrDescription as String] as? String ?? "",
            accessible: attributes[kSecAttrAccessible as String] as? String,
            group: attributes[kSecAttrAccessGroup as String] as? String
        )
        return errSecSuccess
    }

    func update(query: [String: Any], attributes: [String: Any]) -> OSStatus {
        guard let key = Self.key(query), let data = attributes[kSecValueData as String] as? Data
        else { return errSecParam }
        lock.lock()
        defer { lock.unlock() }
        guard var existing = entries[key] else { return errSecItemNotFound }
        existing.data = data
        entries[key] = existing
        return errSecSuccess
    }

    func delete(_ query: [String: Any]) -> OSStatus {
        guard let key = Self.key(query) else { return errSecParam }
        lock.lock()
        defer { lock.unlock() }
        guard entries.removeValue(forKey: key) != nil else { return errSecItemNotFound }
        return errSecSuccess
    }

    /// Seeds a copy the app did not write, e.g. one an earlier version left in
    /// the login keychain.
    func seed(service: String, account: String, secret: String, dataProtection: Bool) {
        lock.lock()
        defer { lock.unlock() }
        let owner = dataProtection ? "dp" : "legacy"
        entries["\(owner)|\(service)|\(account)"] = Entry(
            data: Data(secret.utf8),
            label: "seeded",
            description: "seeded",
            accessible: nil,
            group: nil
        )
    }

    func secret(service: String, account: String, dataProtection: Bool) -> String? {
        lock.lock()
        defer { lock.unlock() }
        let owner = dataProtection ? "dp" : "legacy"
        return entries["\(owner)|\(service)|\(account)"].flatMap { String(data: $0.data, encoding: .utf8) }
    }

    func accounts(service: String, dataProtection: Bool) -> [String] {
        lock.lock()
        defer { lock.unlock() }
        let prefix = "\(dataProtection ? "dp" : "legacy")|\(service)|"
        return entries.keys
            .filter { $0.hasPrefix(prefix) }
            .map { String($0.dropFirst(prefix.count)) }
            .sorted()
    }
}

/// The data protection path: the one that ships, and the one this test host
/// cannot be signed for. Pinned through the backend and item seams so the
/// branch decisions are exercised rather than skipped.
final class DataProtectionKeychainPathTests: XCTestCase {

    private let group = "RMY84Y8G7M.DRP.DynaMoE"
    private let resolver = SecretStoreBackendResolver.shared
    private var service = ""
    private var items: FakeKeychainItemAccess!

    override func setUp() {
        super.setUp()
        service = "DRP.DynaMoE.tests.dp.\(UUID().uuidString)"
        items = FakeKeychainItemAccess()
    }

    override func tearDown() {
        resolver.overrideForTesting(nil)
        resolver.entitlementGroupsProvider = nil
        resolver.capabilityProbe = nil
        resolver.probeAttempts = 2
        resolver.probeRetryDelayMicroseconds = 25_000
        items = nil
        super.tearDown()
    }

    private var entitled: KeychainSecretStore {
        KeychainSecretStore(
            service: service,
            items: items,
            backend: .dataProtection(accessGroup: group)
        )
    }

    private var unentitled: KeychainSecretStore {
        KeychainSecretStore(service: service, items: items, backend: .legacy)
    }

    func testADataProtectionStoreRoundTripsInItsOwnNamespace() throws {
        try entitled.set("sk-dp", account: "acct", label: "l", description: "d")

        XCTAssertEqual(entitled.read(account: "acct"), .found("sk-dp"))
        XCTAssertEqual(items.secret(service: service, account: "acct", dataProtection: true), "sk-dp")
        XCTAssertTrue(
            items.accounts(service: service, dataProtection: false).isEmpty,
            "a data protection write must not land in the login keychain"
        )
    }

    func testADataProtectionWriteNamesItsGroupAndAccessibilityClass() throws {
        try entitled.set("sk-dp", account: "acct", label: "l", description: "d")

        let written = try XCTUnwrap(items.written.last)
        XCTAssertEqual(written[kSecUseDataProtectionKeychain as String] as? Bool, true)
        XCTAssertEqual(written[kSecAttrAccessGroup as String] as? String, group)
        XCTAssertEqual(
            written[kSecAttrAccessible as String] as? String,
            kSecAttrAccessibleAfterFirstUnlock as String
        )
    }

    func testTheResolvedStoreNeverFallsBackToTheOlderKeychainForARead() {
        items.seed(service: service, account: "acct", secret: "sk-old", dataProtection: false)

        XCTAssertEqual(
            entitled.read(account: "acct"), .missing,
            "reading a legacy item is the one call that can show a dialog, so it belongs to the import"
        )
    }

    func testExistenceAsksEachKeychainOnlyWhereTheBuildCanReadIt() {
        // A credential only an entitled build can see is invisible to one that
        // reads the login keychain directly.
        items.seed(service: service, account: "own", secret: "sk-dp", dataProtection: true)
        XCTAssertTrue(entitled.exists(account: "own"))
        XCTAssertFalse(
            unentitled.exists(account: "own"),
            "an unentitled build must not claim a credential it cannot read"
        )

        // An older copy is visible to both: that is what tells an entitled
        // build the credential still needs importing.
        items.seed(service: service, account: "old", secret: "sk-old", dataProtection: false)
        XCTAssertTrue(
            entitled.exists(account: "old"),
            "an entitled build reads the new keychain, so the older copy is the one to report"
        )
        XCTAssertTrue(unentitled.exists(account: "old"))
    }

    func testDeletingFromAnEntitledBuildRemovesBothCopies() {
        items.seed(service: service, account: "acct", secret: "sk-dp", dataProtection: true)
        items.seed(service: service, account: "acct", secret: "sk-old", dataProtection: false)

        entitled.delete(account: "acct")

        XCTAssertTrue(items.accounts(service: service, dataProtection: true).isEmpty)
        XCTAssertTrue(
            items.accounts(service: service, dataProtection: false).isEmpty,
            "a deleted credential must not survive in the keychain this build does not read"
        )
    }

    func testAnImportCopiesForwardAndReapsTheOlderCopy() {
        resolver.overrideForTesting(.dataProtection(accessGroup: group))
        items.seed(service: service, account: "acct", secret: "sk-old", dataProtection: false)

        let outcome = KeychainMigration.adopt(
            PendingCredential(service: service, account: "acct"),
            items: items
        )

        XCTAssertEqual(outcome, .imported(account: "acct"))
        XCTAssertTrue(outcome.succeeded)
        XCTAssertEqual(items.secret(service: service, account: "acct", dataProtection: true), "sk-old")
        XCTAssertNil(
            items.secret(service: service, account: "acct", dataProtection: false),
            "the copy-then-reap order means the older item goes only after the new one reads back"
        )
        let written = items.written.last
        XCTAssertEqual(
            written?[kSecAttrAccessible as String] as? String,
            kSecAttrAccessibleAfterFirstUnlock as String,
            "an imported credential needs the same accessibility class a fresh write gets"
        )
    }

    func testAnImportReapsALeftoverWhoseCopiesAgree() {
        resolver.overrideForTesting(.dataProtection(accessGroup: group))
        items.seed(service: service, account: "acct", secret: "sk-same", dataProtection: true)
        items.seed(service: service, account: "acct", secret: "sk-same", dataProtection: false)

        XCTAssertEqual(
            KeychainMigration.adopt(PendingCredential(service: service, account: "acct"), items: items),
            .duplicateRemoved(account: "acct")
        )
        XCTAssertNil(items.secret(service: service, account: "acct", dataProtection: false))
        XCTAssertEqual(items.secret(service: service, account: "acct", dataProtection: true), "sk-same")
    }

    func testAnImportLeavesAConflictingOlderCopyAlone() {
        resolver.overrideForTesting(.dataProtection(accessGroup: group))
        items.seed(service: service, account: "acct", secret: "sk-new", dataProtection: true)
        items.seed(service: service, account: "acct", secret: "sk-old", dataProtection: false)

        XCTAssertEqual(
            KeychainMigration.adopt(PendingCredential(service: service, account: "acct"), items: items),
            .alreadyPresent(account: "acct")
        )
        XCTAssertEqual(items.secret(service: service, account: "acct", dataProtection: true), "sk-new")
        XCTAssertEqual(
            items.secret(service: service, account: "acct", dataProtection: false), "sk-old",
            "two different values are a conflict for a human, not a delete"
        )
    }

    /// The legacy backend reads and reaps the SAME item, so an import there
    /// would recognise its own copy as a leftover and delete the only one.
    func testAnImportOnALegacyBuildNeverTouchesTheOnlyCopy() {
        resolver.overrideForTesting(.legacy)
        items.seed(service: service, account: "acct", secret: "sk-only", dataProtection: false)

        let outcome = KeychainMigration.adopt(
            PendingCredential(service: service, account: "acct"),
            items: items
        )

        XCTAssertEqual(outcome, .noSecondKeychain(account: "acct"))
        XCTAssertEqual(
            items.secret(service: service, account: "acct", dataProtection: false), "sk-only",
            "an unentitled build has nothing to import and must not reap"
        )
        XCTAssertTrue(items.accounts(service: service, dataProtection: true).isEmpty)
    }
}
