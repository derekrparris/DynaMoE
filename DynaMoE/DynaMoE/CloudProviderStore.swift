//
//  CloudProviderStore.swift
//  DynaMoE
//
//  Settings-facing owner of the cloud provider list.
//
//  The catalog (`CloudProviderCatalog`) is the persistence rule and the
//  runtime (`CloudProviderRuntime`) is the lock-guarded snapshot every other
//  surface reads. This store is the write path between them: each mutation
//  validates, persists, refreshes the runtime, republishes for SwiftUI, and
//  asks the model scan to recompute the composer's cloud entries.
//

import Foundation
import SwiftUI
import Combine

// MARK: - Store Errors

/// The mutations that fail for a reason other than a bad draft.
nonisolated enum CloudProviderStoreError: LocalizedError {
    case builtInProviderIsImmutable(String)
    case providerNotFound(String)

    var errorDescription: String? {
        switch self {
        case .builtInProviderIsImmutable(let name):
            return "\(name) ships with DynaMoE and cannot be edited or removed."
        case .providerNotFound(let identifier):
            return "No provider with the identifier \"\(identifier)\" is configured."
        }
    }
}

// MARK: - Store

/// Main-actor store behind Settings ▸ Cloud Providers. Tests construct one
/// with a throwaway defaults suite and keychain service, and turn the model
/// rescan off so a unit test never touches the shared model list.
final class CloudProviderStore: ObservableObject {
    static let shared = CloudProviderStore()

    @Published private(set) var providers: [CloudProvider]

    private let defaults: UserDefaults
    private let secretStore: CloudSecretStore
    private let rescansModels: Bool
    private let updatesSharedRuntime: Bool

    init(
        defaults: UserDefaults = .standard,
        secretStore: CloudSecretStore = .shared,
        rescansModels: Bool = true,
        updatesSharedRuntime: Bool = true
    ) {
        self.defaults = defaults
        self.secretStore = secretStore
        self.rescansModels = rescansModels
        self.updatesSharedRuntime = updatesSharedRuntime
        self.providers = CloudProviderCatalog.all(defaults: defaults)
    }

    // MARK: Queries

    var builtInIDs: Set<String> {
        Set(CloudProviderCatalog.builtIns.map(\.id))
    }

    func isBuiltIn(_ provider: CloudProvider) -> Bool {
        builtInIDs.contains(provider.id)
    }

    // MARK: Mutations

    /// Registers a provider and, when a key was typed, stores it. The key is
    /// written first: a keychain refusal must not leave a provider behind that
    /// claims to be connected.
    @discardableResult
    func add(
        displayName: String,
        identifier: String,
        baseURL: String,
        docsURL: String = "",
        apiKey: String = ""
    ) throws -> CloudProvider {
        let provider = try CloudProviderCatalog.validate(
            displayName: displayName,
            identifier: identifier,
            baseURL: baseURL,
            docsURL: docsURL,
            existing: CloudProviderCatalog.all(defaults: defaults)
        ).get()

        let key = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        if !key.isEmpty {
            try secretStore.set(key, providerId: provider.id)
        }

        var custom = CloudProviderCatalog.customProviders(defaults: defaults)
        custom.append(provider)
        CloudProviderCatalog.saveCustomProviders(custom, defaults: defaults)
        publish()
        return provider
    }

    /// Changes the display name, base URL, or documentation link. The
    /// identifier is deliberately immutable: it is the Keychain account
    /// suffix and part of the `cloud://` address saved with every conversation
    /// that used the provider. A new base URL drops the per-model state, since
    /// every id and capability mark it held described the old endpoint.
    func update(
        _ provider: CloudProvider,
        displayName: String,
        baseURL: String,
        docsURL: String = ""
    ) throws {
        guard !isBuiltIn(provider) else {
            throw CloudProviderStoreError.builtInProviderIsImmutable(provider.displayName)
        }
        var custom = CloudProviderCatalog.customProviders(defaults: defaults)
        guard let index = custom.firstIndex(where: { $0.id == provider.id }) else {
            throw CloudProviderStoreError.providerNotFound(provider.id)
        }
        let existing = CloudProviderCatalog.builtIns
            + custom.filter { $0.id != provider.id }
        let updated = try CloudProviderCatalog.validate(
            displayName: displayName,
            identifier: provider.id,
            baseURL: baseURL,
            docsURL: docsURL,
            existing: existing
        ).get()

        let previousBaseURL = custom[index].openAIBaseURL
        custom[index] = updated
        CloudProviderCatalog.saveCustomProviders(custom, defaults: defaults)
        // The registered ids, the no-tools list, and the vision marks are all
        // facts about the server at the old base URL rather than about this
        // identifier, so keeping them would offer model ids the new endpoint
        // never listed and fail with that endpoint's own error on the send.
        if updated.openAIBaseURL != previousBaseURL {
            CloudModelRegistry.clearProviderState(for: updated, defaults: defaults)
        }
        publish()
    }

    /// Removes a custom provider together with everything keyed by its
    /// identifier. Saved conversations that pinned one of its models are
    /// handled by `CloudModelAddress.selectionIsUnresolved`: the send is
    /// refused with a reselect message rather than falling back to weights.
    /// A built-in throws instead of no-oping, matching `update`: the caller is
    /// asking for something the store cannot do.
    func remove(_ provider: CloudProvider) throws {
        guard !isBuiltIn(provider) else {
            throw CloudProviderStoreError.builtInProviderIsImmutable(provider.displayName)
        }

        secretStore.delete(providerId: provider.id)
        CloudModelRegistry.clearProviderState(for: provider, defaults: defaults)

        var custom = CloudProviderCatalog.customProviders(defaults: defaults)
        custom.removeAll { $0.id == provider.id }
        CloudProviderCatalog.saveCustomProviders(custom, defaults: defaults)
        publish()
    }

    // MARK: Change propagation

    /// Re-reads the persisted list, for a Settings reappearance after an
    /// external `defaults write`. An unchanged list does not republish, so
    /// opening the pane again does not fire `objectWillChange`.
    func reloadFromStorage() {
        if updatesSharedRuntime {
            CloudProviderRuntime.shared.refresh(defaults: defaults)
        }
        publish(providers: CloudProviderCatalog.all(defaults: defaults))
    }

    private func publish() {
        if updatesSharedRuntime {
            CloudProviderRuntime.shared.refresh(defaults: defaults)
        }
        publish(providers: CloudProviderCatalog.all(defaults: defaults))
        if rescansModels {
            LocalModelManager.shared.scanLocalModels()
        }
    }

    private func publish(providers fresh: [CloudProvider]) {
        guard fresh != providers else { return }
        providers = fresh
    }
}
