//
//  KeychainMigration.swift
//  DynaMoE
//
//  Moves credentials out of the legacy login keychain and into whichever
//  keychain this build reads (see SecretStoreSupport.swift), at the user's
//  request.
//
//  Why an explicit step instead of an automatic one: reading a legacy item is
//  the only keychain call in this app that can put a system dialog on screen,
//  and the app already learns whether a credential exists without one. Doing it
//  behind a button keeps every dialog attached to something the user just asked
//  for, and keeps the launch path silent on the first run after an update.
//
//  The sequence is copy-then-reap, never move: the new copy is written and read
//  back before the old one is deleted, so an interrupted or refused import
//  leaves the credential exactly where it was.
//

import Foundation
import Security

// MARK: - Credentials

/// One credential still sitting in the legacy keychain.
public nonisolated struct PendingCredential: Sendable, Hashable, Identifiable {
    public let service: String
    public let account: String

    public init(service: String, account: String) {
        self.service = service
        self.account = account
    }

    public var id: String { "\(service)|\(account)" }

    /// A human name derived from the account's own shape, so a dialog can say
    /// which credential it is about without reading anything from the keychain.
    public var displayLabel: String {
        if service == CloudSecretStore.defaultService {
            let prefix = "dynamoe.cloud."
            let suffix = ".apikey"
            if account.hasPrefix(prefix), account.hasSuffix(suffix) {
                let provider = String(account.dropFirst(prefix.count).dropLast(suffix.count))
                return "\(Self.humanized(provider)) API key"
            }
            return "Cloud provider credential"
        }
        if service == MCPSecretStore.defaultService {
            let prefix = "dynamoe.mcp."
            guard account.hasPrefix(prefix) else { return "MCP credential" }
            let remainder = account.dropFirst(prefix.count)
            guard let separator = remainder.firstIndex(of: ".") else { return "MCP · \(remainder)" }
            let server = String(remainder[remainder.startIndex..<separator])
            let field = String(remainder[remainder.index(after: separator)...])
            return "MCP · \(Self.humanized(server)) · \(field)"
        }
        return account
    }

    private static func humanized(_ token: String) -> String {
        guard !token.isEmpty else { return token }
        return token.prefix(1).uppercased() + token.dropFirst()
    }
}

// MARK: - Inventory

/// What one service's keychains hold, so the planning rule is a pure function
/// of account names. Deliberately not "legacy minus current": see
/// `KeychainMigration.pending(services:)`.
public nonisolated struct KeychainMigrationInventory: Sendable, Equatable {
    public let usesDataProtection: Bool
    public let legacyAccounts: [String]

    public init(usesDataProtection: Bool, legacyAccounts: [String]) {
        self.usesDataProtection = usesDataProtection
        self.legacyAccounts = legacyAccounts
    }
}

// MARK: - Decision

/// What an import should do with one credential, decided from two reads so the
/// rule is testable without a keychain.
public nonisolated enum KeychainMigrationDecision: Sendable, Equatable {
    /// Copy this value forward, then reap the legacy item.
    case migrate(String)
    /// The legacy item exists but holds nothing; reap it and copy nothing.
    case tombstone
    /// Both keychains hold this account and the copies agree, so the legacy one
    /// is a leftover: reap it and copy nothing.
    case duplicate
    /// The current keychain has this account with a different value; leave the
    /// legacy copy for a conflict to be resolved by hand.
    case alreadyPresent
    /// The legacy read was refused, which is what a dismissed dialog returns.
    case needsAuthorization(OSStatus)
    case failed(OSStatus)
}

// MARK: - Outcome

public nonisolated enum KeychainImportOutcome: Sendable, Equatable {
    case imported(account: String)
    case tombstoneRemoved(account: String)
    case duplicateRemoved(account: String)
    case alreadyPresent(account: String)
    case needsAuthorization(account: String, OSStatus)
    case failed(account: String, OSStatus)

    public var account: String {
        switch self {
        case .imported(let account), .tombstoneRemoved(let account),
             .duplicateRemoved(let account), .alreadyPresent(let account),
             .failed(let account, _):
            return account
        case .needsAuthorization(let account, _):
            return account
        }
    }

    public var succeeded: Bool {
        switch self {
        case .imported, .tombstoneRemoved, .duplicateRemoved, .alreadyPresent: return true
        case .needsAuthorization, .failed: return false
        }
    }

    public var message: String {
        switch self {
        case .imported:
            return "Imported."
        case .tombstoneRemoved:
            return "Removed an empty entry."
        case .duplicateRemoved:
            return "Removed the older copy; the current keychain already had it."
        case .alreadyPresent:
            return "Both keychains hold this credential with different values; left the older copy alone."
        case .needsAuthorization(_, let status):
            return "Not authorized (\(KeychainSecretStore.describe(status))). Click Always Allow when macOS asks, then import again."
        case .failed(_, let status):
            return "Could not import: \(KeychainSecretStore.describe(status))."
        }
    }
}

// MARK: - Migration

public nonisolated enum KeychainMigration {
    public static var services: [String] {
        [MCPSecretStore.defaultService, CloudSecretStore.defaultService]
    }

    /// Credentials still sitting in the legacy keychain. Attribute-only queries,
    /// so this never prompts and is safe to call from Settings' appearance path.
    ///
    /// An account that the current keychain already holds is listed too, because
    /// the legacy copy is exactly what has to go: an import that was interrupted
    /// after the write but before the reap leaves both, and nothing else would
    /// ever remove the older one. What to do with it is decided by the values in
    /// `adopt`, which is where a real conflict is told apart from a leftover.
    public static func pending(services: [String] = services) -> [PendingCredential] {
        var found: [PendingCredential] = []
        for service in services {
            let store = KeychainSecretStore(service: service)
            let inventory = KeychainMigrationInventory(
                usesDataProtection: store.backend.usesDataProtection,
                legacyAccounts: store.legacyAccounts()
            )
            found.append(contentsOf: pendingCredentials(service: service, inventory: inventory))
        }
        return found.sorted { lhs, rhs in
            lhs.service == rhs.service ? lhs.account < rhs.account : lhs.service < rhs.service
        }
    }

    /// The planning rule as a pure function of account names, so the "a build
    /// without the entitlement has nothing to import" contract is testable
    /// without a keychain.
    public static func pendingCredentials(
        service: String,
        inventory: KeychainMigrationInventory
    ) -> [PendingCredential] {
        // An unentitled build reads the legacy keychain directly; there is no
        // second keychain to copy anything into.
        guard inventory.usesDataProtection else { return [] }
        return inventory.legacyAccounts.map { PendingCredential(service: service, account: $0) }
    }

    public static func decision(
        current: SecretAccessResult,
        legacy: SecretAccessResult
    ) -> KeychainMigrationDecision {
        switch legacy {
        case .accessDenied(let status):
            return .needsAuthorization(status)
        case .unavailable(let status):
            return .failed(status)
        case .missing:
            return .failed(errSecItemNotFound)
        case .found(let older):
            if older.isEmpty { return .tombstone }
            // An empty current value is not a credential: treat it as absent so
            // the real value wins rather than being reported as a conflict.
            guard let existing = current.value, !existing.isEmpty else { return .migrate(older) }
            return existing == older ? .duplicate : .alreadyPresent
        }
    }

    /// Copies one credential forward. `nil` means the legacy read prompted and
    /// the user dismissed it, which the caller reports rather than retries.
    public static func adopt(_ credential: PendingCredential) -> KeychainImportOutcome {
        let store = KeychainSecretStore(service: credential.service)
        let current = store.read(account: credential.account)
        let legacy = store.readLegacy(account: credential.account)

        switch decision(current: current, legacy: legacy) {
        case .duplicate:
            // The copies agree, so this credential is already usable from the
            // current keychain and the legacy one is a leftover. Removing it is
            // the whole point of the import.
            store.deleteLegacy(account: credential.account)
            return .duplicateRemoved(account: credential.account)

        case .alreadyPresent:
            return .alreadyPresent(account: credential.account)

        case .tombstone:
            store.deleteLegacy(account: credential.account)
            return .tombstoneRemoved(account: credential.account)

        case .needsAuthorization(let status):
            return .needsAuthorization(account: credential.account, status)
        case .failed(let status):
            return .failed(account: credential.account, status)

        case .migrate(let secret):
            do {
                try store.set(
                    secret,
                    account: credential.account,
                    label: label(for: credential.service),
                    description: description(for: credential.service)
                )
            } catch let failure as SecretStoreFailure {
                if case .status(let status) = failure {
                    return .failed(account: credential.account, status)
                }
                return .failed(account: credential.account, errSecParam)
            } catch {
                return .failed(account: credential.account, errSecInternalError)
            }

            // Read back before reaping: the legacy copy is the only copy until
            // this comparison succeeds.
            guard store.read(account: credential.account).value == secret else {
                return .failed(account: credential.account, errSecDecode)
            }
            store.deleteLegacy(account: credential.account)
            return .imported(account: credential.account)
        }
    }

    public static func adoptAll(
        services: [String] = services,
        progress: ((PendingCredential) -> Void)? = nil
    ) -> [KeychainImportOutcome] {
        pending(services: services).map { credential in
            progress?(credential)
            let outcome = adopt(credential)
            let result = outcome.succeeded ? "ok" : "not imported"
            DiagLog.info("keychain", "import \(credential.account) -> \(result)")
            return outcome
        }
    }

    private static func label(for service: String) -> String {
        service == CloudSecretStore.defaultService
            ? "DynaMoE cloud provider API key"
            : "DynaMoE MCP secret"
    }

    private static func description(for service: String) -> String {
        service == CloudSecretStore.defaultService
            ? "Cloud inference provider credential"
            : "MCP server credential"
    }
}
