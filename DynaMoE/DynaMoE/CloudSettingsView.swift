//
//  CloudSettingsView.swift
//  DynaMoE
//
//  Settings ▸ Cloud Providers: one card per configured provider with a
//  Keychain-backed API key, a connection test, and a one-click model sync
//  that registers the provider's chat models (non-chat catalog entries such as
//  embeddings and image generators are filtered out). Providers are data:
//  "Add Provider…" registers any OpenAI-compatible endpoint at runtime.
//
//  Secrets are never pre-filled into the field: an existing key shows as a
//  "stored" placeholder and typing replaces it. "Test & Sync" is the only
//  network call made here: GET /models, which both validates the key and
//  returns the live catalog, written back through the app process so the
//  chat picker updates immediately.
//

import SwiftUI

struct CloudProvidersSettingsView: View {
    private enum TestState: Equatable {
        case untested
        case testing
        case success(modelCount: Int, registeredCount: Int, sample: [String])
        case failure(String)
    }

    /// One sheet serves both flows: `.add` creates a provider, `.edit` changes
    /// the display name, base URL, or docs link of an existing one.
    private enum ProviderEditorTarget: Identifiable {
        case add
        case edit(CloudProvider)

        var id: String {
            switch self {
            case .add: return "add"
            case .edit(let provider): return provider.id
            }
        }

        var provider: CloudProvider? {
            if case .edit(let provider) = self { return provider }
            return nil
        }
    }

    @ObservedObject private var localModelManager = LocalModelManager.shared
    @ObservedObject private var providerStore = CloudProviderStore.shared
    @State private var keyDrafts: [String: String] = [:]
    @State private var storedProviders: Set<String> = []
    @State private var testStates: [String: TestState] = [:]
    @State private var providerEditor: ProviderEditorTarget? = nil
    @State private var providerPendingRemoval: CloudProvider? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 22) {
            ForEach(providerStore.providers) { provider in
                providerCard(provider)
            }

            addProviderCard
            registeredModelsCard
        }
        .onAppear(perform: refreshState)
        .sheet(item: $providerEditor) { target in
            CloudAddProviderSheet(store: providerStore, provider: target.provider) { _ in
                refreshState()
            }
        }
        .alert(
            "Remove provider?",
            isPresented: Binding(
                get: { providerPendingRemoval != nil },
                set: { if !$0 { providerPendingRemoval = nil } }
            ),
            presenting: providerPendingRemoval
        ) { provider in
            Button("Remove", role: .destructive) {
                providerStore.remove(provider)
                keyDrafts[provider.id] = ""
                testStates[provider.id] = .untested
                providerPendingRemoval = nil
            }
            Button("Cancel", role: .cancel) { providerPendingRemoval = nil }
        } message: { provider in
            Text("\(provider.displayName) is removed together with its Keychain API key, its registered model list, and its vision marks. Conversations pinned to its models will ask you to reselect a model.")
        }
    }

    // MARK: - Cards

    private func providerCard(_ provider: CloudProvider) -> some View {
        SettingsCard(header: provider.displayName) {
            SettingsRow(
                title: "API Key",
                subtitle: "Created in the \(provider.displayName) dashboard; stored in the login Keychain and never written to disk. DynaMoE uses the OpenAI-compatible endpoint at \(provider.openAIBaseURL.absoluteString).",
                icon: "key.fill",
                iconTint: .cyan
            ) {
                HStack(spacing: 8) {
                    SecureField(
                        storedProviders.contains(provider.id) ? "Stored — enter to replace" : "sk-…",
                        text: draftBinding(for: provider.id)
                    )
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 200)

                    Button("Save") { saveKey(for: provider) }
                        .disabled(draft(for: provider.id).isEmpty)

                    if storedProviders.contains(provider.id) {
                        Button("Remove") { removeKey(for: provider) }
                    }
                }
            }

            SettingsRowDivider()

            SettingsRow(
                title: "Connection",
                subtitle: connectionStatusText(for: provider),
                icon: "bolt.horizontal.circle.fill",
                iconTint: .cyan
            ) {
                HStack(spacing: 8) {
                    if testStates[provider.id] == .testing {
                        ProgressView()
                            .controlSize(.small)
                    }
                    Button("Test & Sync Models") { testAndSync(provider) }
                        .disabled(!storedProviders.contains(provider.id) || testStates[provider.id] == .testing)
                }
            }

            SettingsRowDivider()

            SettingsRow(
                title: "Models",
                subtitle: modelsSubtitle(for: provider),
                icon: "square.stack.3d.up.fill",
                iconTint: .cyan
            ) {
                HStack(spacing: 8) {
                    Button("Sync Models") { testAndSync(provider) }
                        .disabled(!storedProviders.contains(provider.id) || testStates[provider.id] == .testing)
                    if !registeredSpecs(for: provider).isEmpty {
                        Button("Clear") { clearModels(for: provider) }
                    }
                }
            }

            SettingsRowDivider()

            SettingsRow(
                title: "Workspace Data",
                subtitle: "With Agent Tools on, this provider receives the conversation plus verbatim tool output (file contents, search results, command output). Only enable it for a provider you trust with the workspace you open.",
                icon: "exclamationmark.shield.fill",
                iconTint: .orange
            ) {
                EmptyView()
            }

            SettingsRowDivider()

            if let docsURL = provider.docsURL {
                SettingsRow(
                    title: "Documentation",
                    subtitle: "API reference, model catalog, and dashboard access.",
                    icon: "book.fill",
                    iconTint: .cyan
                ) {
                    Link("Open API Docs", destination: docsURL)
                }

                SettingsRowDivider()
            }

            SettingsRow(
                title: "Provider",
                subtitle: "Identifier \(provider.id) · OpenAI-compatible base \(provider.openAIBaseURL.absoluteString). Built-in providers stay available and cannot be edited.",
                icon: "gearshape.2.fill",
                iconTint: .cyan
            ) {
                HStack(spacing: 8) {
                    Button("Edit") { providerEditor = .edit(provider) }
                        .disabled(providerStore.isBuiltIn(provider))
                    Button("Remove") { providerPendingRemoval = provider }
                        .disabled(providerStore.isBuiltIn(provider))
                }
            }
        }
    }

    /// The generic seam: any endpoint that speaks OpenAI Chat Completions and
    /// lists models over GET /models can be added without a code change.
    private var addProviderCard: some View {
        SettingsCard(header: "Add a Provider") {
            SettingsRow(
                title: "OpenAI-Compatible Endpoint",
                subtitle: "DynaMoE drives an added provider through POST chat/completions with an Authorization: Bearer key and reads its catalog from GET models. Paste the API base including its version segment; OpenCode Zen, for example, is https://opencode.ai/zen/v1.",
                icon: "plus.circle.fill",
                iconTint: .cyan
            ) {
                Button("Add Provider…") { providerEditor = .add }
            }
        }
    }

    private var registeredModelsCard: some View {
        SettingsCard(header: "Registered Cloud Models") {
            ForEach(Array(providerStore.providers.enumerated()), id: \.element.id) { index, provider in
                let specs = registeredSpecs(for: provider)
                SettingsRow(
                    title: provider.displayName,
                    subtitle: specs.isEmpty
                        ? "Nothing registered yet. Press \"Test & Sync Models\" to register the provider's chat models (embedding, speech, rerank, and image models are skipped)."
                        : "Mark which models accept image input. No provider catalog publishes modalities, so this is opt-in; a model that rejects an image is un-marked automatically and the turn retries without it. Use the minus button to drop a model the provider does not serve on this wire.",
                    icon: "cloud.fill",
                    iconTint: .cyan
                ) {
                    VStack(alignment: .trailing, spacing: 6) {
                        Text(specs.isEmpty ? "None" : "\(specs.count) registered")
                            .font(.system(size: 11))
                            .foregroundStyle(specs.isEmpty ? .secondary : .primary)
                        // A provider can register dozens of models; without a bound
                        // the settings row grows past the window and the toggles at
                        // the bottom become unreachable.
                        ScrollView {
                            VStack(alignment: .trailing, spacing: 6) {
                                ForEach(specs, id: \.providerModelId) { spec in
                                    HStack(spacing: 6) {
                                        Toggle(isOn: visionBinding(for: spec, provider: provider)) {
                                            Text(spec.displayName)
                                                .font(.system(size: 10.5))
                                                .lineLimit(1)
                                                .truncationMode(.middle)
                                        }
                                        .toggleStyle(.switch)
                                        .controlSize(.mini)
                                        .help("Let this model receive images")

                                        Button {
                                            CloudModelRegistry.removeRegisteredModel(
                                                spec.providerModelId,
                                                for: provider
                                            )
                                            LocalModelManager.shared.scanLocalModels()
                                        } label: {
                                            Image(systemName: "minus.circle")
                                                .font(.system(size: 10.5))
                                        }
                                        .buttonStyle(.borderless)
                                        .help("Remove this model from the picker")
                                    }
                                }
                            }
                        }
                        .frame(maxHeight: 180)
                    }
                }
                if index < providerStore.providers.count - 1 {
                    SettingsRowDivider()
                }
            }
        }
    }

    /// Vision is a per-model registration fact: persistence and immediate
    /// composer refresh go through the same rescan the key actions use.
    private func visionBinding(for spec: CloudModelSpec, provider: CloudProvider) -> Binding<Bool> {
        Binding(
            get: {
                CloudModelRegistry.modelIdsWithVisionSupport(for: provider)
                    .contains(spec.providerModelId)
            },
            set: { supported in
                CloudModelRegistry.setVisionSupport(
                    supported,
                    forModel: spec.providerModelId,
                    provider: provider
                )
                LocalModelManager.shared.scanLocalModels()
            }
        )
    }

    // MARK: - Bindings & State

    private func draftBinding(for providerId: String) -> Binding<String> {
        Binding(
            get: { keyDrafts[providerId] ?? "" },
            set: { keyDrafts[providerId] = $0 }
        )
    }

    private func draft(for providerId: String) -> String {
        (keyDrafts[providerId] ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func registeredSpecs(for provider: CloudProvider) -> [CloudModelSpec] {
        CloudModelRegistry.registeredSpecs(for: provider)
    }

    private func refreshState() {
        // Re-read the persisted list so a `defaults write` made outside the
        // app is visible the next time this pane appears.
        providerStore.reloadFromStorage()
        storedProviders = Set(providerStore.providers.filter {
            CloudSecretStore.shared.exists(providerId: $0.id)
        }.map(\.id))
    }

    private func saveKey(for provider: CloudProvider) {
        let key = draft(for: provider.id)
        guard !key.isEmpty else { return }
        do {
            try CloudSecretStore.shared.set(key, providerId: provider.id)
            keyDrafts[provider.id] = ""
            storedProviders.insert(provider.id)
            testStates[provider.id] = .untested
            // Models only surface once a credential exists; re-scan so a key
            // saved after a sync makes the composer's list appear immediately.
            LocalModelManager.shared.scanLocalModels()
        } catch {
            testStates[provider.id] = .failure(error.localizedDescription)
        }
    }

    private func removeKey(for provider: CloudProvider) {
        CloudSecretStore.shared.delete(providerId: provider.id)
        storedProviders.remove(provider.id)
        keyDrafts[provider.id] = ""
        testStates[provider.id] = .untested
        // Drop this provider's entries from the composer immediately.
        LocalModelManager.shared.scanLocalModels()
    }

    /// Validates the key and registers the provider's live model catalog in
    /// one pass: GET /models both proves the credential and returns every
    /// model the provider serves, so there is nothing for the user to type.
    private func testAndSync(_ provider: CloudProvider) {
        let client: CloudChatClient
        switch CloudChatClient.makeStoredResult(provider: provider) {
        case .success(let ready):
            client = ready
        case .failure(let failure):
            testStates[provider.id] = .failure(failure.localizedDescription)
            return
        }
        testStates[provider.id] = .testing
        Task {
            switch await client.fetchModelIds() {
            case .success(let ids):
                guard !ids.isEmpty else {
                    testStates[provider.id] = .failure("The provider returned an empty model list.")
                    return
                }
                // The catalog also lists embeddings, rerankers, speech, and
                // image generators; registering those makes them pickable chat
                // models that fail on the first send.
                let chatIds = CloudModelRegistry.chatModelIds(from: ids)
                CloudModelRegistry.setRegisteredModelIds(chatIds, for: provider)
                // Re-scan so the composer's "Cloud Models" section reflects
                // the new list without a relaunch.
                LocalModelManager.shared.scanLocalModels()
                testStates[provider.id] = .success(
                    modelCount: ids.count,
                    registeredCount: chatIds.count,
                    sample: Array(chatIds.prefix(5))
                )
            case .failure(let failure):
                testStates[provider.id] = .failure(failure.localizedDescription)
            }
        }
    }

    private func clearModels(for provider: CloudProvider) {
        CloudModelRegistry.setRegisteredModelIds([], for: provider)
        testStates[provider.id] = .untested
        LocalModelManager.shared.scanLocalModels()
    }

    private func modelsSubtitle(for provider: CloudProvider) -> String {
        let specs = registeredSpecs(for: provider)
        if specs.isEmpty {
            return "Cloud models appear in the chat picker under \"Cloud Models\" once registered."
        }
        return "\(specs.count) model(s) registered and shown under \"Cloud Models\" in the composer."
    }

    private func connectionStatusText(for provider: CloudProvider) -> String {
        switch testStates[provider.id] ?? .untested {
        case .untested:
            return storedProviders.contains(provider.id)
                ? "API key stored. Press \"Test & Sync Models\" to validate it and register the live model catalog."
                : "No API key stored yet. Save a key to enable cloud models from this provider."
        case .testing:
            return "Checking \(provider.displayName)…"
        case .success(let modelCount, let registeredCount, let sample):
            let sampleText = sample.isEmpty ? "" : " — e.g. \(sample.joined(separator: ", "))"
            return "Connected ✓ \(modelCount) model(s) available, \(registeredCount) registered\(sampleText)"
        case .failure(let message):
            return "Connection failed: \(message)"
        }
    }
}
