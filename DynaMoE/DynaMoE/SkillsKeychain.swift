//
//  SkillsKeychain.swift
//  DynaMoE
//
//  Phase 8 — Skills.
//
//  Credentials a skill's tool needs, kept in the Keychain so no secret ever
//  reaches `skills.json`, a prompt, or a log line. The facade mirrors
//  `MCPSecretStore`: the same `KeychainSecretStore` core, a service namespace
//  of its own, and a static one-liner surface for call sites.
//
//  Only `subq_decide` uses this today, under the account
//  `dynamoe.skill.subq-decisions.apikey`. An existing Cloud Provider whose
//  base URL points at api.subq.ai is preferred over this store, because that is
//  where a user adding SubQ chat would already have put the key.
//

import Foundation

public nonisolated enum SkillsKeychainError: LocalizedError {
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
            return "Keychain error \(status): \(SkillsSecretStore.describe(status))"
        }
    }

    init(_ failure: SecretStoreFailure) {
        switch failure {
        case .emptySecret: self = .emptySecret
        case .invalidValue: self = .invalidValue
        case .status(let status): self = .status(status)
        }
    }
}

public nonisolated struct SkillsSecretStore: Sendable {
    /// Namespaces every item this feature creates, apart from MCP and cloud
    /// credentials.
    public static let defaultService = "DRP.DynaMoE.skills"

    public static let shared = SkillsSecretStore()

    public let service: String

    public init(service: String = SkillsSecretStore.defaultService) {
        self.service = service
    }

    private var core: KeychainSecretStore {
        KeychainSecretStore(service: service)
    }

    /// `dynamoe.skill.<skill>.<field>`. Skill names are validated to
    /// `[A-Za-z0-9-]`, so the dots stay unambiguous.
    public static func account(skill: String, field: String) -> String {
        "dynamoe.skill.\(skill).\(field)"
    }

    @discardableResult
    public func set(_ secret: String, account: String) throws -> String {
        do {
            return try core.set(
                secret,
                account: account,
                label: "DynaMoE skill credential",
                description: "Credential used by an installed agent skill"
            )
        } catch let failure as SecretStoreFailure {
            throw SkillsKeychainError(failure)
        }
    }

    public func read(account: String) -> SecretAccessResult {
        core.read(account: account)
    }

    public func readLegacy(account: String) -> SecretAccessResult {
        core.readLegacy(account: account)
    }

    /// Fails closed: nil covers "not stored", "denied" and "unreadable", so a
    /// caller can never send a blank credential upstream.
    public func get(account: String) -> String? {
        core.read(account: account).value
    }

    /// Attributes only, so asking whether a key exists never prompts.
    public func exists(account: String) -> Bool {
        core.exists(account: account)
    }

    /// True only when the keychain this build actually READS holds the key,
    /// i.e. a call can succeed without importing anything first. `exists`
    /// covers the legacy keychain too, and those two questions had different
    /// answers: a legacy-only item reported "Ready" while `get` returned nil.
    public func hasCurrentCredential(account: String) -> Bool {
        core.exists(account: account, backend: core.backend)
    }

    public func delete(account: String) {
        core.delete(account: account)
    }

    public static func describe(_ status: OSStatus) -> String {
        KeychainSecretStore.describe(status)
    }
}

// MARK: - Static facade

public extension SkillsSecretStore {
    @discardableResult
    static func set(_ secret: String, account: String) throws -> String {
        try shared.set(secret, account: account)
    }

    static func read(account: String) -> SecretAccessResult {
        shared.read(account: account)
    }

    static func get(account: String) -> String? {
        shared.get(account: account)
    }

    static func exists(account: String) -> Bool {
        shared.exists(account: account)
    }

    static func hasCurrentCredential(account: String) -> Bool {
        shared.hasCurrentCredential(account: account)
    }

    static func delete(account: String) {
        shared.delete(account: account)
    }
}
