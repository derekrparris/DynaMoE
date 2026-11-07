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
    public static func indicatesMissingEntitlement(_ status: OSStatus) -> Bool {
        status == errSecMissingEntitlement || status == errSecParam
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
public nonisolated final class SecretStoreBackendResolver: @unchecked Sendable {
    public static let shared = SecretStoreBackendResolver()

    /// Test seam: the `keychain-access-groups` values this build was signed
    /// with. Nil means "read them from the running signature".
    var entitlementGroupsProvider: (() -> [String])?

    /// Test seam: writes and removes a sentinel item, returning the status of
    /// the write. Nil means "use the real keychain".
    var capabilityProbe: ((KeychainBackend) -> OSStatus)?

    private let lock = NSLock()
    private var cached: KeychainBackend?
    private var didLog = false

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
        cached = resolved
        let announce = !didLog
        didLog = true
        lock.unlock()

        if announce {
            switch resolved {
            case .dataProtection(let group):
                DiagLog.info("keychain", "backend=dataProtection group=\(group)")
            case .legacy:
                DiagLog.info("keychain", "backend=legacy (no keychain-access-groups capability in this build)")
            }
        }
        return resolved
    }

    /// Test seam so a decision can be exercised without a signed build.
    func overrideForTesting(_ backend: KeychainBackend?) {
        lock.lock()
        cached = backend
        lock.unlock()
    }

    private func resolve() -> KeychainBackend {
        let groups = entitlementGroupsProvider?() ?? Self.signatureAccessGroups()
        guard !groups.isEmpty else { return .legacy }

        let bundleId = Bundle.main.bundleIdentifier ?? "DRP.DynaMoE"
        let group = groups.first(where: { $0.hasSuffix(".\(bundleId)") || $0 == bundleId }) ?? groups.first
        guard let group, !group.isEmpty else { return .legacy }

        let candidate = KeychainBackend.dataProtection(accessGroup: group)
        let status = capabilityProbe?(candidate) ?? Self.probeWrite(candidate)

        switch status {
        case errSecSuccess, errSecDuplicateItem:
            return candidate
        default:
            if SecretAccessClassifier.indicatesMissingEntitlement(status) {
                return .legacy
            }
            DiagLog.warning("keychain", "data protection probe failed (\(status)); using legacy keychain")
            return .legacy
        }
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

// MARK: - Store

/// A keychain-backed secret store scoped to one service string. The MCP and
/// cloud stores are thin facades over this so both answer the same questions
/// the same way.
public nonisolated struct KeychainSecretStore: Sendable {
    public let service: String

    public init(service: String) {
        self.service = service
    }

    public var backend: KeychainBackend {
        SecretStoreBackendResolver.shared.backend()
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
        let updateStatus = SecItemUpdate(
            SecretStoreQuery.scoped([
                kSecClass as String: kSecClassGenericPassword,
                kSecAttrService as String: service,
                kSecAttrAccount as String: account,
            ], backend: backend) as CFDictionary,
            [kSecValueData as String: data] as CFDictionary
        )
        if updateStatus == errSecSuccess { return account }
        guard updateStatus == errSecItemNotFound else {
            throw SecretStoreFailure.status(updateStatus)
        }

        let addStatus = SecItemAdd(
            SecretStoreQuery.add(
                service: service,
                account: account,
                data: data,
                label: label,
                description: description,
                backend: backend
            ) as CFDictionary,
            nil
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
        var item: CFTypeRef?
        let status = SecItemCopyMatching(
            SecretStoreQuery.read(service: service, account: account, backend: backend) as CFDictionary,
            &item
        )
        guard status == errSecSuccess else {
            return SecretAccessClassifier.failure(status)
        }
        guard let data = item as? Data, let secret = String(data: data, encoding: .utf8) else {
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
        SecItemCopyMatching(
            SecretStoreQuery.existence(service: service, account: account, backend: backend) as CFDictionary,
            nil
        ) == errSecSuccess
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
        SecItemDelete(SecretStoreQuery.delete(service: service, account: account, backend: backend) as CFDictionary)
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
        var result: CFTypeRef?
        let status = SecItemCopyMatching(
            SecretStoreQuery.listing(service: service, backend: backend) as CFDictionary,
            &result
        )
        guard status == errSecSuccess else { return [] }
        let items: [[String: Any]]
        if let list = result as? [[String: Any]] {
            items = list
        } else if let single = result as? [String: Any] {
            items = [single]
        } else {
            return []
        }
        return items.compactMap { $0[kSecAttrAccount as String] as? String }
    }

    public static func describe(_ status: OSStatus) -> String {
        if let message = SecCopyErrorMessageString(status, nil) as String? {
            return message
        }
        return "unknown status"
    }
}
