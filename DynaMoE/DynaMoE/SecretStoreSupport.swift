//
//  SecretStoreSupport.swift
//  DynaMoE
//
//  One keychain access layer for MCP credentials and cloud provider keys.
//
//  Two things about the legacy (file-based) login keychain shaped this file:
//
//  1. Every item carries an ACL that authorizes a code signature. A grant is
//     recorded as the writer's designated requirement, so an ad-hoc or newly
//     signed build reads the same item as a different program and macOS asks
//     again. A Developer ID requirement is stable across versions, which is why
//     a shipped build can keep one "Always Allow" forever while a locally
//     signed one cannot.
//  2. Only a read of the secret *data* consults that ACL. An attributes-only
//     query returns names and service strings without a dialog, which is what
//     makes a prompt-free "is a credential stored?" check possible.
//
//  The data protection keychain (`kSecUseDataProtectionKeychain`) replaces the
//  ACL with an access group derived from the app's signature, so an entitled
//  build reads its own items silently on every future version. It requires a
//  `keychain-access-groups` entitlement backed by an embedded provisioning
//  profile; a build without one gets `errSecMissingEntitlement` on write. The
//  resolver below therefore decides per process and falls back to the legacy
//  keychain, which keeps unentitled local builds working exactly as before.
//

import Foundation
import Security

// MARK: - Backend

/// Which keychain the process should use. `dataProtection` carries the access
/// group read from this build's own signature; it is never hardcoded, because
/// an item written under the implicit default group can read back as
/// `errSecItemNotFound` when the group is later named explicitly.
public nonisolated enum KeychainBackend: Sendable, Equatable {
    case dataProtection(accessGroup: String)
    case legacy

    public var usesDataProtection: Bool {
        if case .dataProtection = self { return true }
        return false
    }

    public var accessGroup: String? {
        if case .dataProtection(let group) = self { return group }
        return nil
    }
}

// MARK: - Result shape

/// What a secret read actually established. The distinction between "nothing is
/// stored" and "the user did not authorize this read" is the difference between
/// telling somebody to add a key and telling them a dialog was dismissed.
public nonisolated enum SecretAccessResult: Sendable, Equatable {
    case found(String)
    case missing
    case accessDenied(OSStatus)
    case unavailable(OSStatus)

    public var value: String? {
        if case .found(let secret) = self { return secret }
        return nil
    }

    public var isMissing: Bool {
        if case .missing = self { return true }
        return false
    }

    /// True when a credential exists but this build may not read it, i.e. the
    /// actionable advice is "authorize or import", not "add a key".
    public var needsAuthorization: Bool {
        switch self {
        case .accessDenied, .unavailable: return true
        case .found, .missing: return false
        }
    }
}

public nonisolated enum SecretAccessClassifier {
    /// Statuses macOS reports when the ACL (or a locked keychain) refused the
    /// read, as opposed to the item not existing.
    public static let deniedStatuses: Set<OSStatus> = [
        errSecAuthFailed,
        errSecUserCanceled,
        errSecInteractionNotAllowed,
        errSecNoAccessForItem,
    ]

    public static func failure(_ status: OSStatus) -> SecretAccessResult {
        if status == errSecItemNotFound { return .missing }
        if deniedStatuses.contains(status) { return .accessDenied(status) }
        return .unavailable(status)
    }

    /// `errSecMissingEntitlement` is the documented answer for an app that asks
    /// the data protection keychain for a group its signature does not grant.
    ///
    /// `errSecParam` is deliberately NOT included: it is what a malformed query
    /// returns, so it says nothing about the signature. Treating it as proof of
    /// a missing capability would cache a query-construction bug for the life of
    /// the process as "this build cannot use the keychain".
    public static func indicatesMissingEntitlement(_ status: OSStatus) -> Bool {
        status == errSecMissingEntitlement
    }

    /// A status that points at the QUERY rather than at the build's signature.
    public static func indicatesMalformedQuery(_ status: OSStatus) -> Bool {
        status == errSecParam
    }
}

// MARK: - Queries

/// Every query this app issues is built here so the two properties that matter
/// are visible in one place: no query that must not prompt asks for
/// `kSecReturnData`, and every query names its backend explicitly.
public nonisolated enum SecretStoreQuery {
    public static func scoped(_ query: [String: Any], backend: KeychainBackend) -> [String: Any] {
        var scoped = query
        // Not optional, in either direction. On an entitled app, omitting the
        // flag is not a request for the file keychain: reads may resolve the
        // legacy item while deletes resolve the data protection one, which is
        // how a migration can delete the copy it just wrote.
        scoped[kSecUseDataProtectionKeychain as String] = backend.usesDataProtection
        if let group = backend.accessGroup {
            scoped[kSecAttrAccessGroup as String] = group
        }
        return scoped
    }

    public static func existence(
        service: String,
        account: String,
        backend: KeychainBackend
    ) -> [String: Any] {
        scoped([
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecMatchLimit as String: kSecMatchLimitOne,
            kSecReturnAttributes as String: true,
        ], backend: backend)
    }

    public static func read(
        service: String,
        account: String,
        backend: KeychainBackend
    ) -> [String: Any] {
        scoped([
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecMatchLimit as String: kSecMatchLimitOne,
            kSecReturnData as String: true,
        ], backend: backend)
    }

    public static func listing(service: String, backend: KeychainBackend) -> [String: Any] {
        scoped([
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecMatchLimit as String: kSecMatchLimitAll,
            kSecReturnAttributes as String: true,
        ], backend: backend)
    }

    public static func delete(
        service: String,
        account: String,
        backend: KeychainBackend
    ) -> [String: Any] {
        scoped([
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ], backend: backend)
    }

    public static func add(
        service: String,
        account: String,
        data: Data,
        label: String,
        description: String,
        backend: KeychainBackend
    ) -> [String: Any] {
        var attributes = scoped([
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecAttrLabel as String: label,
            kSecAttrDescription as String: description,
            kSecValueData as String: data,
        ], backend: backend)
        if backend.usesDataProtection {
            attributes[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        }
        return attributes
    }

}

// MARK: - Failure

public nonisolated enum SecretStoreFailure: LocalizedError, Equatable {
    case emptySecret
    case invalidValue
    case status(OSStatus)

    public var errorDescription: String? {
        switch self {
        case .emptySecret:
            return "Refusing to store an empty secret."
        case .invalidValue:
            return "The secret could not be encoded as UTF-8."
        case .status(let status):
            return "Keychain error \(status): \(KeychainSecretStore.describe(status))"
        }
    }
}

// MARK: - Backend resolution

/// Chooses the backend once per process: entitlement first, capability probe
/// second, legacy as the answer whenever either is inconclusive.
///
/// "Inconclusive" is not the same as "refused", and the difference matters.
/// `errSecMissingEntitlement` says the signature will never be granted the
/// group, so the legacy keychain is the answer for the life of the process.
/// Every other probe failure is a property of this instant (a keychain still
/// coming up at launch, an interaction that could not be shown yet, or a query
/// this code built wrongly) so it is retried immediately, and a retry that also
/// fails leaves the answer uncached: the next caller asks again rather than the
/// process spending its whole life writing secrets to a keychain the next
/// launch will not read.
public nonisolated final class SecretStoreBackendResolver: @unchecked Sendable {
    public static let shared = SecretStoreBackendResolver()

    /// Test seam: the `keychain-access-groups` values this build was signed
    /// with. Nil means "read them from the running signature".
    var entitlementGroupsProvider: (() -> [String])?

    /// Test seam: writes and removes a sentinel item, returning the status of
    /// the write. Nil means "use the real keychain".
    var capabilityProbe: ((KeychainBackend) -> OSStatus)?

    /// How many times one resolution asks the probe before calling the answer
    /// inconclusive. Two is enough for a keychain that was still coming up.
    var probeAttempts: Int = 2

    /// Delay between those attempts, in microseconds. Zero disables the wait,
    /// which is what tests want.
    var probeRetryDelayMicroseconds: UInt32 = 25_000

    private let lock = NSLock()
    private var cached: KeychainBackend?
    private var lastLogged: KeychainBackend?

    private init() {}

    public func backend() -> KeychainBackend {
        lock.lock()
        if let cached {
            lock.unlock()
            return cached
        }
        lock.unlock()

        let resolved = resolve()
        lock.lock()
        if let cached = self.cached {
            lock.unlock()
            return cached
        }
        if resolved.definitive {
            cached = resolved.backend
        }
        let announce = lastLogged != resolved.backend
        if announce { lastLogged = resolved.backend }
        lock.unlock()

        if announce {
            switch resolved.backend {
            case .dataProtection(let group):
                DiagLog.info("keychain", "backend=dataProtection group=\(group)")
            case .legacy:
                if resolved.definitive {
                    DiagLog.info("keychain", "backend=legacy (no keychain-access-groups capability in this build)")
                } else {
                    DiagLog.info("keychain", "backend=legacy (provisional: the data protection keychain did not answer, will ask again)")
                }
            }
        }
        return resolved.backend
    }

    /// Test seam so a decision can be exercised without a signed build.
    func overrideForTesting(_ backend: KeychainBackend?) {
        lock.lock()
        cached = backend
        lastLogged = backend
        lock.unlock()
    }

    private struct Resolution {
        let backend: KeychainBackend
        /// True when the answer is a property of this build rather than of this
        /// instant, i.e. worth remembering for the life of the process.
        let definitive: Bool
    }

    private func resolve() -> Resolution {
        let groups = entitlementGroupsProvider?() ?? Self.signatureAccessGroups()
        // No entitlement at all: nothing to probe, and nothing can grant it
        // later, so this is the legacy answer for good.
        guard !groups.isEmpty else { return Resolution(backend: .legacy, definitive: true) }

        let bundleId = Bundle.main.bundleIdentifier ?? "DRP.DynaMoE"
        let group = groups.first(where: { $0.hasSuffix(".\(bundleId)") || $0 == bundleId }) ?? groups.first
        guard let group, !group.isEmpty else { return Resolution(backend: .legacy, definitive: true) }

        let candidate = KeychainBackend.dataProtection(accessGroup: group)
        var lastStatus = errSecSuccess
        let attempts = max(1, probeAttempts)
        for attempt in 1...attempts {
            let status = capabilityProbe?(candidate) ?? Self.probeWrite(candidate)
            lastStatus = status
            switch status {
            case errSecSuccess, errSecDuplicateItem:
                return Resolution(backend: candidate, definitive: true)
            default:
                if SecretAccessClassifier.indicatesMissingEntitlement(status) {
                    // The signature does not grant the group. Retrying cannot
                    // change that, and a transient status must not be mistaken
                    // for it.
                    return Resolution(backend: .legacy, definitive: true)
                }
                if SecretAccessClassifier.indicatesMalformedQuery(status) {
                    // errSecParam is returned for a malformed query, not for an
                    // unentitled build. Name the status instead of reporting an
                    // entitlement fact, and leave the answer provisional so the
                    // next caller asks again rather than inheriting it.
                    DiagLog.warning(
                        "keychain",
                        "data protection keychain probe answered errSecParam (\(status)), which points at the query, not the entitlement; not caching this answer"
                    )
                }
                if attempt < attempts, probeRetryDelayMicroseconds > 0 {
                    Thread.sleep(forTimeInterval: TimeInterval(probeRetryDelayMicroseconds) / 1_000_000)
                }
            }
        }
        DiagLog.warning(
            "keychain",
            "data protection keychain did not answer (\(lastStatus)); using the legacy keychain until it does"
        )
        return Resolution(backend: .legacy, definitive: false)
    }

    /// A read is not enough to prove the capability: an unentitled app is
    /// answered `errSecItemNotFound` on read for backwards compatibility and
    /// only refuses on write. The sentinel item is removed immediately.
    private static func probeWrite(_ backend: KeychainBackend) -> OSStatus {
        let service = "DRP.DynaMoE.keychain-probe"
        let account = "capability"
        let query = SecretStoreQuery.add(
            service: service,
            account: account,
            data: Data("1".utf8),
            label: "DynaMoE keychain probe",
            description: "Written once per launch to detect keychain access capability",
            backend: backend
        )
        let status = SecItemAdd(query as CFDictionary, nil)
        if status == errSecSuccess || status == errSecDuplicateItem {
            SecItemDelete(SecretStoreQuery.delete(service: service, account: account, backend: backend) as CFDictionary)
        }
        return status
    }

    private static func signatureAccessGroups() -> [String] {
        guard let task = SecTaskCreateFromSelf(nil),
              let value = SecTaskCopyValueForEntitlement(
                task,
                "keychain-access-groups" as CFString,
                nil
              ) else { return [] }
        return (value as? [String]) ?? []
    }
}

// MARK: - Item access

/// One keychain lookup's outcome: the status, plus whatever the query was
/// allowed to return. Deliberately not `Sendable`: the value is a Core
/// Foundation object that the caller consumes immediately.
public nonisolated struct KeychainItemLookup {
    public let status: OSStatus
    public let value: CFTypeRef?

    public init(status: OSStatus, value: CFTypeRef? = nil) {
        self.status = status
        self.value = value
    }
}

/// The four keychain primitives this app issues, behind a protocol.
///
/// The default implementation is the real `SecItem*` family; the seam exists
/// because the data protection keychain can only be WRITTEN from a build that
/// carries `keychain-access-groups`, and the test host is an unentitled Debug
/// app. Without it, every test would have to skip the backend that ships.
public nonisolated protocol KeychainItemAccess: Sendable {
    func copyMatching(_ query: [String: Any]) -> KeychainItemLookup
    func add(_ attributes: [String: Any]) -> OSStatus
    func update(query: [String: Any], attributes: [String: Any]) -> OSStatus
    func delete(_ query: [String: Any]) -> OSStatus
}

public nonisolated struct SystemKeychainItemAccess: KeychainItemAccess {
    public init() {}

    public func copyMatching(_ query: [String: Any]) -> KeychainItemLookup {
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        return KeychainItemLookup(status: status, value: item)
    }

    public func add(_ attributes: [String: Any]) -> OSStatus {
        SecItemAdd(attributes as CFDictionary, nil)
    }

    public func update(query: [String: Any], attributes: [String: Any]) -> OSStatus {
        SecItemUpdate(query as CFDictionary, attributes as CFDictionary)
    }

    public func delete(_ query: [String: Any]) -> OSStatus {
        SecItemDelete(query as CFDictionary)
    }
}

// MARK: - Store

/// A keychain-backed secret store scoped to one service string. The MCP and
/// cloud stores are thin facades over this so both answer the same questions
/// the same way.
public nonisolated struct KeychainSecretStore: Sendable {
    public let service: String
    private let items: any KeychainItemAccess
    /// Test seam: pins this store to one backend instead of asking the
    /// process-wide resolver. Nil in production.
    private let pinnedBackend: KeychainBackend?

    public init(service: String) {
        self.service = service
        self.items = SystemKeychainItemAccess()
        self.pinnedBackend = nil
    }

    init(service: String, items: any KeychainItemAccess, backend: KeychainBackend?) {
        self.service = service
        self.items = items
        self.pinnedBackend = backend
    }

    public var backend: KeychainBackend {
        pinnedBackend ?? SecretStoreBackendResolver.shared.backend()
    }

    @discardableResult
    public func set(
        _ secret: String,
        account: String,
        label: String,
        description: String
    ) throws -> String {
        guard !secret.isEmpty else { throw SecretStoreFailure.emptySecret }
        guard let data = secret.data(using: .utf8) else { throw SecretStoreFailure.invalidValue }

        let backend = self.backend
        let updateStatus = items.update(
            query: SecretStoreQuery.scoped([
                kSecClass as String: kSecClassGenericPassword,
                kSecAttrService as String: service,
                kSecAttrAccount as String: account,
            ], backend: backend),
            attributes: [kSecValueData as String: data]
        )
        if updateStatus == errSecSuccess { return account }
        guard updateStatus == errSecItemNotFound else {
            throw SecretStoreFailure.status(updateStatus)
        }

        let addStatus = items.add(
            SecretStoreQuery.add(
                service: service,
                account: account,
                data: data,
                label: label,
                description: description,
                backend: backend
            )
        )
        guard addStatus == errSecSuccess else {
            throw SecretStoreFailure.status(addStatus)
        }
        return account
    }

    /// Reads the secret from the resolved backend. A legacy item is deliberately
    /// not consulted here: that read is the one call that can show an ACL
    /// dialog, so it belongs to the explicit import path instead.
    public func read(account: String) -> SecretAccessResult {
        read(account: account, backend: backend)
    }

    /// Reads a credential from the legacy keychain, which may prompt once.
    public func readLegacy(account: String) -> SecretAccessResult {
        read(account: account, backend: .legacy)
    }

    public func read(account: String, backend: KeychainBackend) -> SecretAccessResult {
        let lookup = items.copyMatching(
            SecretStoreQuery.read(service: service, account: account, backend: backend)
        )
        guard lookup.status == errSecSuccess else {
            return SecretAccessClassifier.failure(lookup.status)
        }
        guard let data = lookup.value as? Data, let secret = String(data: data, encoding: .utf8) else {
            return .unavailable(errSecDecode)
        }
        return .found(secret)
    }

    /// Answers "is this credential stored?" without touching the secret data,
    /// so it can run on the launch path. Both keychains are asked because a
    /// pre-import credential still lives in the legacy one.
    public func exists(account: String) -> Bool {
        if exists(account: account, backend: backend) { return true }
        guard backend.usesDataProtection else { return false }
        return exists(account: account, backend: .legacy)
    }

    public func exists(account: String, backend: KeychainBackend) -> Bool {
        items.copyMatching(
            SecretStoreQuery.existence(service: service, account: account, backend: backend)
        ).status == errSecSuccess
    }

    /// Removes the credential from both keychains, so a deleted integration
    /// cannot leave a copy behind in the one this build does not read.
    public func delete(account: String) {
        delete(account: account, backend: backend)
        if backend.usesDataProtection {
            delete(account: account, backend: .legacy)
        }
    }

    public func delete(account: String, backend: KeychainBackend) {
        _ = items.delete(SecretStoreQuery.delete(service: service, account: account, backend: backend))
    }

    /// Removes only the legacy copy, which is what a finished import reaps.
    public func deleteLegacy(account: String) {
        delete(account: account, backend: .legacy)
    }

    /// Account names stored under this service. Attributes only, so no dialog.
    public func accounts(includingLegacy: Bool) -> [String] {
        var names = Set(accountNames(backend: backend))
        if includingLegacy, backend.usesDataProtection {
            names.formUnion(accountNames(backend: .legacy))
        }
        return names.sorted()
    }

    public func legacyAccounts() -> [String] {
        accountNames(backend: .legacy)
    }

    private func accountNames(backend: KeychainBackend) -> [String] {
        let lookup = items.copyMatching(
            SecretStoreQuery.listing(service: service, backend: backend)
        )
        guard lookup.status == errSecSuccess else { return [] }
        let records: [[String: Any]]
        if let list = lookup.value as? [[String: Any]] {
            records = list
        } else if let single = lookup.value as? [String: Any] {
            records = [single]
        } else {
            return []
        }
        return records.compactMap { $0[kSecAttrAccount as String] as? String }
    }

    public static func describe(_ status: OSStatus) -> String {
        if let message = SecCopyErrorMessageString(status, nil) as String? {
            return message
        }
        return "unknown status"
    }
}
