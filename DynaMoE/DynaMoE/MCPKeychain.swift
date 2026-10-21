//
//  MCPKeychain.swift
//  DynaMoE
//
//  Phase 7: Model Context Protocol client.
//  Minimal Keychain-backed secret store for MCP credentials.
//
//  mcp.json never holds a secret. Secret fields carry a
//  "${keychain:<account>}" token instead, and the value lives in the login
//  Keychain keyed by that account. That keeps an API key out of a file the
//  user may put in a dotfiles repo or paste into an issue.
//
//  The legacy file-based keychain is used deliberately (no
//  kSecUseDataProtectionKeychain): the data-protection keychain requires an
//  application-identifier entitlement, and ad-hoc-signed local builds of this
//  app (the recovery path in AGENTS.md) do not have one.
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
}

public nonisolated enum MCPSecretStore {
    /// Namespaces every item this app creates.
    public static let service = "DRP.DynaMoE.mcp"

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
    public static func set(_ secret: String, account: String) throws -> String {
        guard !secret.isEmpty else { throw MCPKeychainError.emptySecret }
        guard let data = secret.data(using: .utf8) else { throw MCPKeychainError.invalidValue }

        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]

        let updateStatus = SecItemUpdate(
            query as CFDictionary,
            [kSecValueData as String: data] as CFDictionary
        )
        if updateStatus == errSecSuccess { return account }
        guard updateStatus == errSecItemNotFound else {
            throw MCPKeychainError.status(updateStatus)
        }

        var attributes = query
        attributes[kSecValueData as String] = data
        attributes[kSecAttrLabel as String] = "DynaMoE MCP secret"
        attributes[kSecAttrDescription as String] = "MCP server credential"
        let addStatus = SecItemAdd(attributes as CFDictionary, nil)
        guard addStatus == errSecSuccess else {
            throw MCPKeychainError.status(addStatus)
        }
        return account
    }

    /// Returns nil for "not found" *and* for "could not read right now"
    /// (locked keychain, denied access): an MCP server must fail closed with a
    /// clear unresolved-token error rather than connect with a blank value.
    public static func get(account: String) -> String? {
        var query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        query[kSecMatchLimit as String] = kSecMatchLimitOne

        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        guard status == errSecSuccess,
              let data = item as? Data,
              let secret = String(data: data, encoding: .utf8) else { return nil }
        return secret
    }

    public static func exists(account: String) -> Bool {
        get(account: account) != nil
    }

    public static func delete(account: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        SecItemDelete(query as CFDictionary)
    }

    /// Removes every secret belonging to one server. Called when a server is
    /// deleted from the config so a removed integration leaves no credential
    /// behind.
    public static func deleteAll(serverKey: String) {
        let prefix = "dynamoe.mcp.\(serverKey)."
        var query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecReturnAttributes as String: true,
            kSecMatchLimit as String: kSecMatchLimitAll,
        ]
        query[kSecReturnAttributes as String] = true

        var result: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let items = result as? [[String: Any]] else { return }

        for item in items {
            guard let account = item[kSecAttrAccount as String] as? String,
                  account.hasPrefix(prefix) else { continue }
            delete(account: account)
        }
    }

    public static func describe(_ status: OSStatus) -> String {
        if let message = SecCopyErrorMessageString(status, nil) as String? {
            return message
        }
        return "unknown status"
    }
}
