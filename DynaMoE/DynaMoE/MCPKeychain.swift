//
//  MCPKeychain.swift
//  DynaMoE
//
//  Phase 7: Model Context Protocol client.
//  Keychain-backed secret store for MCP credentials.
//
//  mcp.json never holds a secret. Secret fields carry a
//  "${keychain:<account>}" token instead, and the value lives in the keychain
//  keyed by that account. That keeps an API key out of a file the user may put
//  in a dotfiles repo or paste into an issue.
//
//  Which keychain, and why it matters for repeated authorization, is documented
//  in SecretStoreSupport.swift. In short: the data protection keychain is used
//  whenever this build is signed with a keychain-access-groups entitlement, and
//  the legacy login keychain otherwise. Reads here never fall back to the
//  legacy item on their own, because that read is the one call that can show an
//  ACL dialog; KeychainMigration performs it on the user's request instead.
//

import Foundation
import Security

public nonisolated enum MCPKeychainError: LocalizedError {
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
            return "Keychain error \(status): \(MCPSecretStore.describe(status))"
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

public nonisolated struct MCPSecretStore: Sendable {
    /// Namespaces every item this app creates.
    public static let defaultService = "DRP.DynaMoE.mcp"

    /// The process-wide store. Tests construct their own instance with a
    /// throwaway service name instead of writing into the real one.
    public static let shared = MCPSecretStore()

    public let service: String

    public init(service: String = MCPSecretStore.defaultService) {
        self.service = service
    }

    private var core: KeychainSecretStore {
        KeychainSecretStore(service: service)
    }

    /// Deterministic per-server, per-field account name, so the JSON token and
    /// the Keychain item cannot drift. `serverKey` is already constrained to
    /// `[A-Za-z0-9_-]` and may not contain `__`, so the dots are unambiguous.
    public static func account(serverKey: String, field: String) -> String {
        "dynamoe.mcp.\(serverKey).\(field)"
    }

    public static func token(serverKey: String, field: String) -> String {
        MCPTokenExpander.keychainToken(account: account(serverKey: serverKey, field: field))
    }

    @discardableResult
    public func set(_ secret: String, account: String) throws -> String {
        do {
            return try core.set(
                secret,
                account: account,
                label: "DynaMoE MCP secret",
                description: "MCP server credential"
            )
        } catch let failure as SecretStoreFailure {
            throw MCPKeychainError(failure)
        }
    }

    /// Answers "could this credential be read, and if not, why". A missing item
    /// and a refused read are different states: a server should fail with an
    /// actionable message either way, but only one of them is fixed by adding a
    /// key.
    public func read(account: String) -> SecretAccessResult {
        core.read(account: account)
    }

    public func readLegacy(account: String) -> SecretAccessResult {
        core.readLegacy(account: account)
    }

    /// Returns nil for "not found", "could not read right now" and "denied":
    /// an MCP server must fail closed with a clear unresolved-token error
    /// rather than connect with a blank value. Callers that need the
    /// distinction use `read(account:)`.
    public func get(account: String) -> String? {
        core.read(account: account).value
    }

    /// True when a credential is stored, whether in the current keychain or
    /// still in the legacy one. Attributes only, so never prompts.
    public func exists(account: String) -> Bool {
        core.exists(account: account)
    }

    /// True when the credential is only in the keychain an earlier version
    /// wrote, i.e. a token needs importing before it can be expanded. Attribute
    /// query only, so this is safe on the connect path.
    public func existsInLegacyKeychain(account: String) -> Bool {
        core.exists(account: account, backend: .legacy)
    }

    public func delete(account: String) {
        core.delete(account: account)
    }

    /// Removes every secret belonging to one server, from both keychains. Called
    /// when a server is deleted from the config so a removed integration leaves
    /// no credential behind, including one that was never imported.
    public func deleteAll(serverKey: String) {
        let prefix = "dynamoe.mcp.\(serverKey)."
        for account in core.accounts(includingLegacy: true) where account.hasPrefix(prefix) {
            core.delete(account: account)
        }
    }

    public static func describe(_ status: OSStatus) -> String {
        KeychainSecretStore.describe(status)
    }
}

// MARK: - Static facade

/// The app reads and writes through the shared store. Kept as statics so the
/// call sites stay one-liners; tests build an instance with their own service.
public extension MCPSecretStore {
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

    static func delete(account: String) {
        shared.delete(account: account)
    }

    static func deleteAll(serverKey: String) {
        shared.deleteAll(serverKey: serverKey)
    }
}
