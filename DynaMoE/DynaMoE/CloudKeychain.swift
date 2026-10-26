//
//  CloudKeychain.swift
//  DynaMoE
//
//  Minimal Keychain-backed secret store for connected cloud providers.
//
//  Modeled on MCPKeychain.swift: a provider's API key lives only in the
//  login Keychain, so neither UserDefaults nor any file the user may sync
//  or paste ever holds a credential. The legacy file-based keychain is used
//  deliberately (no kSecUseDataProtectionKeychain): the data-protection
//  keychain requires an application-identifier entitlement, and
//  ad-hoc-signed local builds of this app do not have one.
//

import Foundation
import Security

public nonisolated enum CloudKeychainError: LocalizedError {
    case emptySecret
    case invalidValue
    case status(OSStatus)

    public var errorDescription: String? {
        switch self {
        case .emptySecret:
            return "Refusing to store an empty API key."
        case .invalidValue:
            return "The API key could not be encoded as UTF-8."
        case .status(let status):
            return "Keychain error \(status): \(CloudSecretStore.describe(status))"
        }
    }
}

public nonisolated struct CloudSecretStore: Sendable {
    public static let defaultService = "DRP.DynaMoE.cloud"
    /// The process-wide store. Tests construct their own instance with a
    /// throwaway service name instead of mutating shared state.
    public static let shared = CloudSecretStore()

    public let service: String

    public init(service: String = CloudSecretStore.defaultService) {
        self.service = service
    }

    /// Deterministic per-provider account name: one API key per provider
    /// ("dynamoe.cloud.synthetic.apikey").
    public static func account(providerId: String) -> String {
        "dynamoe.cloud.\(providerId).apikey"
    }

    @discardableResult
    public func set(_ secret: String, providerId: String) throws -> String {
        guard !secret.isEmpty else { throw CloudKeychainError.emptySecret }
        guard let data = secret.data(using: .utf8) else { throw CloudKeychainError.invalidValue }

        let account = Self.account(providerId: providerId)
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
            throw CloudKeychainError.status(updateStatus)
        }

        var attributes = query
        attributes[kSecValueData as String] = data
        attributes[kSecAttrLabel as String] = "DynaMoE cloud provider API key"
        attributes[kSecAttrDescription as String] = "Cloud inference provider credential"
        let addStatus = SecItemAdd(attributes as CFDictionary, nil)
        guard addStatus == errSecSuccess else {
            throw CloudKeychainError.status(addStatus)
        }
        return account
    }

    /// Returns nil for "not found" *and* for "could not read right now"
    /// (locked keychain, denied access): a turn must fail with a clear
    /// missing-key error rather than send a blank bearer token upstream.
    public func get(providerId: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: Self.account(providerId: providerId),
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]

        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        guard status == errSecSuccess,
              let data = item as? Data,
              let secret = String(data: data, encoding: .utf8) else { return nil }
        return secret
    }

    public func exists(providerId: String) -> Bool {
        get(providerId: providerId) != nil
    }

    public func delete(providerId: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: Self.account(providerId: providerId),
        ]
        SecItemDelete(query as CFDictionary)
    }

    public static func describe(_ status: OSStatus) -> String {
        if let message = SecCopyErrorMessageString(status, nil) as String? {
            return message
        }
        return "unknown status"
    }
}
