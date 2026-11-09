//
//  CloudKeychain.swift
//  DynaMoE
//
//  Keychain-backed secret store for connected cloud providers.
//
//  Modeled on MCPKeychain.swift: a provider's API key lives only in the
//  keychain, so neither UserDefaults nor any file the user may sync or paste
//  ever holds a credential. Backend selection (data protection keychain when
//  this build is entitled, legacy login keychain otherwise) and the reasoning
//  behind it live in SecretStoreSupport.swift.
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

    init(_ failure: SecretStoreFailure) {
        switch failure {
        case .emptySecret: self = .emptySecret
        case .invalidValue: self = .invalidValue
        case .status(let status): self = .status(status)
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

    private var core: KeychainSecretStore {
        KeychainSecretStore(service: service)
    }

    /// Deterministic per-provider account name: one API key per provider
    /// ("dynamoe.cloud.synthetic.apikey").
    public static func account(providerId: String) -> String {
        "dynamoe.cloud.\(providerId).apikey"
    }

    @discardableResult
    public func set(_ secret: String, providerId: String) throws -> String {
        let account = Self.account(providerId: providerId)
        do {
            return try core.set(
                secret,
                account: account,
                label: "DynaMoE cloud provider API key",
                description: "Cloud inference provider credential"
            )
        } catch let failure as SecretStoreFailure {
            throw CloudKeychainError(failure)
        }
    }

    /// A missing key and a refused read are different states: the first means
    /// "add one", the second means a dialog was dismissed. Callers that only
    /// need a value use `get(providerId:)`.
    public func read(providerId: String) -> SecretAccessResult {
        core.read(account: Self.account(providerId: providerId))
    }

    public func readLegacy(providerId: String) -> SecretAccessResult {
        core.readLegacy(account: Self.account(providerId: providerId))
    }

    /// Returns nil for "not found" *and* for "could not read right now"
    /// (locked keychain, denied access): a turn must fail with a clear
    /// missing-key error rather than send a blank bearer token upstream.
    public func get(providerId: String) -> String? {
        core.read(account: Self.account(providerId: providerId)).value
    }

    /// True when a key is stored, in either keychain. Attributes only, so this
    /// is safe on the launch path where it is asked once per provider.
    public func exists(providerId: String) -> Bool {
        core.exists(account: Self.account(providerId: providerId))
    }

    /// True only when the keychain this build actually reads holds the key, i.e.
    /// a turn can run without an import step first.
    public func hasCurrentCredential(providerId: String) -> Bool {
        core.exists(account: Self.account(providerId: providerId), backend: core.backend)
    }

    public func delete(providerId: String) {
        core.delete(account: Self.account(providerId: providerId))
    }

    public static func describe(_ status: OSStatus) -> String {
        KeychainSecretStore.describe(status)
    }
}
