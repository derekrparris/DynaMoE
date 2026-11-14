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

        /// `.add` uses a separator no identifier can contain: identifiers are
        /// lowercase letters, digits, and dashes, so a suffix of `.add` would
        /// collide with a provider whose id is literally `add`, while a colon
        /// cannot appear in one.
        var id: String {
            switch self {
            case .add: return "provider-editor:add"
            case .edit(let provider): return "provider-editor.\(provider.id)"
            }
        }

        var provider: CloudProvider? {
            if case .edit(let provider) = self { return provider }
            return nil
        }
    }

    /// One alert covers both destructive actions: SwiftUI only reliably hosts
    /// one alert on a view, so provider removal and key removal share state
    /// rather than racing two modifiers.
    private enum RemovalConfirmation: Identifiable {
        case provider(CloudProvider)
        case apiKey(CloudProvider)

        var id: String {
            switch self {
            case .provider(let provider): return "remove-provider.\(provider.id)"
            case .apiKey(let provider): return "remove-key.\(provider.id)"
            }
        }

        var provider: CloudProvider {
            switch self {
            case .provider(let provider), .apiKey(let provider):
                return provider
            }
        }

        var title: String {
            switch self {
            case .provider: return "Remove provider?"
            case .apiKey: return "Remove API key?"
            }
        }

        var message: String {
            switch self {
            case .provider(let provider):
                // Names every registration fact the removal deletes: prices and
                // context windows are lost with it, and they are typed by hand
                // from the provider's own price sheet with no other control to
                // retype them.
                return "\(provider.displayName) is removed together with its Keychain API key, its registered model list, its no-tools list, its vision marks, and every model price and context window recorded for it. Conversations pinned to its models will ask you to reselect a model."
            case .apiKey(let provider):
                return "The API key for \(provider.displayName) is deleted from the login Keychain. Its registered models stay in the picker but cannot answer until a new key is saved."
            }
        }
    }

    @ObservedObject private var localModelManager = LocalModelManager.shared
    @ObservedObject private var providerStore = CloudProviderStore.shared
    @State private var keyDrafts: [String: String] = [:]
    @State private var storedProviders: Set<String> = []
    @State private var testStates: [String: TestState] = [:]
    @State private var providerEditor: ProviderEditorTarget? = nil
    @State private var pendingConfirmation: RemovalConfirmation? = nil
    @AppStorage("dynamoe_cloud_resend_images") private var resendImages: Bool = true

    /// Per-provider in-progress context-window edits, keyed by provider id then
    /// by upstream model id. An entry is populated only while the user is
    /// editing a row, cleared in `refreshState()` and per-model on Save; when
    /// absent, the field and Save both fall back to `CloudContextPolicy`'s
    /// stored value so a row never shows or overwrites a stale number.
    @State private var windowDrafts: [String: [String: String]] = [:]

    /// Confirmation line per MODEL (`provider/model`), so a Save that stored
    /// nothing (an unparseable value) says so on the row it belongs to instead
    /// of appearing to work — and without repeating on every row of a provider.
    @State private var windowFeedback: [String: String] = [:]

    var body: some View {
        VStack(alignment: .leading, spacing: 22) {
            ForEach(providerStore.providers) { provider in
                providerCard(provider)
            }

            requestBehaviourCard
            addProviderCard
            registeredModelsCard
        }
        .onAppear(perform: refreshState)
        .sheet(item: $providerEditor) { target in
            CloudAddProviderSheet(store: providerStore, provider: target.provider) { saved in
                // The URL may have changed, so a "Connected ✓" line from the
                // previous endpoint would be a claim about a server this
                // provider no longer points at.
                testStates[saved.id] = .untested
                refreshState()
            }
        }
        .alert(
            pendingConfirmation?.title ?? "",
            isPresented: Binding(
                get: { pendingConfirmation != nil },
                set: { if !$0 { pendingConfirmation = nil } }
            ),
            presenting: pendingConfirmation
        ) { confirmation in
            Button("Remove", role: .destructive) { confirmRemoval(confirmation) }
            Button("Cancel", role: .cancel) { pendingConfirmation = nil }
        } message: { confirmation in
            Text(confirmation.message)
        }
    }

    // MARK: - Cards

    /// Provider-independent request behaviour: what every cloud turn asks for,
    /// and where its cost and context ceiling come from.
    private var requestBehaviourCard: some View {
        SettingsCard(header: "Request Behaviour") {
            SettingsRow(
                title: "Re-send Images",
                subtitle: "A stateless provider has to be re-told about a picture on every request that should still see it, and those tokens are billed every time. Off sends only the newest user turn's pictures and replaces pictures from earlier turns with a one-line notice, so a text follow-up does not keep paying for a picture from an earlier question.",
                icon: "photo.on.rectangle.angled",
                iconTint: .cyan
            ) {
                Toggle("", isOn: $resendImages)
                    .labelsHidden()
                    .toggleStyle(.switch)
                    .accessibilityLabel("Re-send images on every request")
            }

            SettingsRowDivider()

            SettingsRow(
                title: "Context Window",
                subtitle: "A model's window sets both when the turn compacts and how large a reply it may ask for (20% of the window). Record it per model below. A model with no recorded window is held to a \(CloudContextPolicy.unknownWindowReplyCeiling.formatted())-token reply budget and compacts only after the provider refuses a request.",
                icon: "ruler.fill",
                iconTint: .cyan
            ) { EmptyView() }

            // A card header followed by "record it per model below" and nothing
            // below it is a dead end: with no provider registered, say so and
            // point at the action that produces rows.
            let windowEntries = contextWindowEntries()
            if windowEntries.isEmpty {
                SettingsRow(
                    title: "No registered models",
                    subtitle: "Register a provider's models with \u{201C}Test & Sync Models\u{201D} on a provider above; each one then gets its own window field here.",
                    icon: "square.stack.3d.up.slash",
                    iconTint: .gray
                ) { EmptyView() }
            } else {
                ForEach(Array(windowEntries.enumerated()), id: \.element.id) { index, entry in
                    if index > 0 { SettingsRowDivider() }
                    contextWindowRow(entry)
                }
            }

            SettingsRowDivider()

            SettingsRow(
                title: "Prices",
                subtitle: "Cost reporting needs per-model prices in USD per million tokens: defaults write DRP.DynaMoE dynamoe_cloud_<provider>_pricing -string '{\"<model>\":{\"input\":0.27,\"cached_input\":0.07,\"output\":1.1}}'. A model with no price simply reports no cost.",
                icon: "dollarsign.circle.fill",
                iconTint: .cyan
            ) { EmptyView() }
        }
    }

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
                    .accessibilityLabel("\(provider.displayName) API key")

                    Button("Save") { saveKey(for: provider) }
                        .disabled(draft(for: provider.id).isEmpty)

                    if storedProviders.contains(provider.id) {
                        Button("Remove") { pendingConfirmation = .apiKey(provider) }
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
                            // Clear is not "remove the provider": it forgets the
                            // catalog, while the facts that describe the MODELS
                            // (hand-typed prices, recorded windows) survive, so a
                            // re-sync does not lose them.
                            .help("Forgets the registered model list. Recorded model prices and context windows are kept.")
                    }
                }
            }

            SettingsRowDivider()

            SettingsRow(
                title: "Prompt Caching",
                subtitle: "Requests carry a stable session-affinity header (unless this provider's setting is Off) so a router can keep this conversation on one replica. The setting beside this row can add OpenAI's prompt_cache_key where the endpoint accepts it; a provider that rejects the field is retried once without it and remembered for the rest of the process only when its refusal NAMES the field (a vague or transient 400 never latches it off). Cache reads are reported in the Request Log beside the input total.",
                icon: "internaldrive.fill",
                iconTint: .cyan
            ) {
                if providerStore.isBuiltIn(provider) {
                    HStack(spacing: 8) {
                        Text(provider.resolvedCacheHints.displayName)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(.secondary)
                        // A built-in's hint is fixed, so the picker that would
                        // forget a latch is unreachable. The latch is still a
                        // process-lifetime fact learned from a 400, and the
                        // operator may know it was wrong (the gateway had a bad
                        // minute); give them the one control the record needs.
                        Button("Forget Refusals") {
                            CloudCacheHintMemory.shared.allowCacheHints(
                                endpoint: provider.chatCompletionsURL.absoluteString
                            )
                        }
                        .controlSize(.small)
                        .disabled(isTesting(provider))
                        .help("Forgets this endpoint's remembered refusals of the prompt-cache key and session-affinity headers.")
                    }
                } else {
                    Picker("", selection: Binding(
                        get: { provider.resolvedCacheHints },
                        set: { try? providerStore.setCacheHints($0, for: provider) }
                    )) {
                        ForEach(CloudCacheHints.allCases, id: \.self) { hint in
                            Text(hint.displayName).tag(hint)
                        }
                    }
                    .labelsHidden()
                    .frame(width: 230)
                    .accessibilityLabel("Prompt caching support for \(provider.displayName)")
                    // A mid-sync change is now tolerated by the post-fetch
                    // guard, but the catalog still describes the server this
                    // request asked, so keep the record stable while it runs.
                    .disabled(isTesting(provider))
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
                        .disabled(providerStore.isBuiltIn(provider) || isTesting(provider))
                    Button("Remove") { pendingConfirmation = .provider(provider) }
                        .disabled(providerStore.isBuiltIn(provider) || isTesting(provider))
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
                        : "Mark which models accept image input. No provider catalog publishes modalities, so this is opt-in; a model that rejects an image is un-marked automatically and the turn retries without it. Use the minus button to drop a model the provider does not serve on this wire; its recorded context window and price are removed with it, so re-adding the same id starts clean.",
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
                                            clearWindowState(
                                                forProviderId: provider.id,
                                                modelId: spec.providerModelId
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

    /// One registered model's window facts, resolved ONCE per body pass so the
    /// row does not re-parse the provider's `model=window` string on every
    /// keystroke in any of the fields beside it.
    ///
    /// Identity is provider + upstream id, which is unique because a model id
    /// belongs to exactly one provider.
    private struct ContextWindowEntry: Identifiable {
        var provider: CloudProvider
        var spec: CloudModelSpec
        /// The window recorded for THIS model, or nil when it has none.
        var recorded: Int?
        /// The window the send path resolves: the model's own, else the single
        /// global fallback. The reply budget is derived from this one, because
        /// this is the one the turn is priced and clamped against.
        var resolvedWindow: Int?

        var id: String { "\(provider.id)/\(spec.providerModelId)" }
        /// What the field shows when the user has not typed into it: the stored
        /// value, so a window recorded out of band (or by a Sync that arrived
        /// after this pane opened) can never sit beside an empty field.
        var recordedText: String { recorded.map(String.init) ?? "" }
        var budget: Int {
            CloudContextPolicy.maximumOutputTokens(window: resolvedWindow)
        }
    }

    /// Every registered model across every provider, flattened so the window
    /// rows can be one `ForEach` (a nested `for` is control flow inside a
    /// `ViewBuilder` and does not compile).
    ///
    /// Each provider's `model=window` string is parsed ONCE here and the result
    /// is carried on the entry. Reading it per row meant N full splits and
    /// parses for N models on every evaluation, and a text field re-evaluates
    /// its row on every keystroke.
    private func contextWindowEntries() -> [ContextWindowEntry] {
        let globalWindow = CloudContextPolicy.globalWindow(defaults: .standard)
        return providerStore.providers.flatMap { provider -> [ContextWindowEntry] in
            let recordedWindows = CloudContextPolicy.overrides(for: provider.id)
            return registeredSpecs(for: provider).map { spec in
                // `overrides(for:)` already drops anything below
                // `minimumPlausibleWindow`, so a value it returned is capacity
                // and the fallback only applies when the model has none.
                let recorded = recordedWindows[spec.providerModelId]
                return ContextWindowEntry(
                    provider: provider,
                    spec: spec,
                    recorded: recorded,
                    resolvedWindow: recorded ?? globalWindow
                )
            }
        }
    }

    /// One model's context window, editable. This used to be a subtitle telling
    /// the user to run `defaults write`; that was acceptable while the window
    /// only drove compaction, but it now also sets the Max Output Tokens ceiling,
    /// so a knob nobody can find would leave every cloud model on the
    /// no-window budget.
    @ViewBuilder
    private func contextWindowRow(_ entry: ContextWindowEntry) -> some View {
        let provider = entry.provider
        let spec = entry.spec
        let budget = entry.budget
        // The per-model override, NOT the value this save would store: the row
        // reports what the turn will actually send with, and the global fallback
        // is named rather than implied, because "no window of its own" and "no
        // window at all" are two different budgets.
        SettingsRow(
            title: spec.displayName,
            subtitle: entry.recorded.map { "\($0.formatted()) tokens — reply budget \(budget.formatted())." }
                ?? (entry.resolvedWindow.map { "No window of its own — the global \($0.formatted()) applies, reply budget \(budget.formatted())." }
                    ?? "No window recorded — reply budget \(budget.formatted()).")
        ) {
            VStack(alignment: .trailing, spacing: 3) {
                HStack(spacing: 6) {
                    TextField(
                        "none",
                        text: Binding(
                            get: {
                                // A draft is an EDIT. With none, the field reads
                                // the same storage the subtitle and the ceiling
                                // read, so the two cannot disagree after a Sync
                                // or a `defaults write` lands mid-session.
                                windowDrafts[provider.id]?[spec.providerModelId] ?? entry.recordedText
                            },
                            set: { newValue in
                                windowDrafts[provider.id, default: [:]][spec.providerModelId] = newValue
                            }
                        )
                    )
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 12, design: .monospaced))
                    .multilineTextAlignment(.trailing)
                    .frame(width: 96)
                    .accessibilityLabel("Context window for \(spec.displayName)")

                    Button("Save") { saveWindow(for: provider, modelId: spec.providerModelId) }
                        .controlSize(.small)
                }

                if let feedback = windowFeedback[entry.id] {
                    Text(feedback)
                        .font(.system(size: 9))
                        .foregroundColor(.secondary)
                        .lineLimit(2)
                        .frame(maxWidth: 240, alignment: .trailing)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    private func refreshState() {
        // Re-read the persisted list so a `defaults write` made outside the
        // app is visible the next time this pane appears.
        providerStore.reloadFromStorage()
        storedProviders = Set(providerStore.providers.filter {
            CloudSecretStore.shared.exists(providerId: $0.id)
        }.map(\.id))
        // Drop every window draft rather than copying the stored values into
        // one: a draft is an EDIT, and a field that reads through to storage
        // when it has none cannot go stale against the subtitle beside it.
        // Seeding meant a row could report the recorded window while the field
        // held whatever this pane snapshotted on appear — which is exactly what
        // a Sync that registers a model whose window was recorded out of band
        // produced: the recorded number in the subtitle, an empty field.
        windowDrafts = [:]
        windowFeedback = [:]
    }

    /// Stores one model's context window, or removes it when the field is empty
    /// (empty means "no recorded window", which is a real state: the reply
    /// budget and the compaction pre-flight both treat it differently).
    ///
    /// Writes through `setOverride`, never through `overrides(for:)` +
    /// `setOverrides`: that pair drops every entry the reader cannot parse, so
    /// saving one row silently deleted a neighbour's hand-typed window (and an
    /// operator's typo, which is theirs to fix, not this button's to erase).
    ///
    /// Only the saved model's draft is replaced. Refreshing the whole provider's
    /// drafts would discard an in-progress edit on a sibling row.
    /// Resolves the string a Save click should persist: an explicit edit draft
    /// wins (including an empty draft that intentionally clears the stored
    /// value), whereas `nil` means the field was never edited and keeps the
    /// stored window it is displaying.
    static func effectiveWindowInput(draft: String?, stored: Int?) -> String {
        draft ?? stored.map(String.init) ?? ""
    }

    private func saveWindow(for provider: CloudProvider, modelId: String) {
        let rawInput = Self.effectiveWindowInput(
            draft: windowDrafts[provider.id]?[modelId],
            stored: CloudContextPolicy.overrides(for: provider.id)[modelId]
        )
        let typed = rawInput.trimmingCharacters(in: .whitespacesAndNewlines)
        let feedbackKey = "\(provider.id)/\(modelId)"

        var stored: Int?
        if typed.isEmpty {
            CloudContextPolicy.setOverride(
                model: modelId,
                window: nil,
                providerId: provider.id
            )
        } else if let window = Int(typed.replacingOccurrences(of: ",", with: "")
            .replacingOccurrences(of: "_", with: "")),
            window >= CloudContextPolicy.minimumPlausibleWindow {
            guard CloudContextPolicy.setOverride(
                model: modelId,
                window: window,
                providerId: provider.id
            ) else {
                // The parser would drop it, so storing it would look saved and
                // read back as absent.
                windowFeedback[feedbackKey] = "Not saved: enter a whole number of at least \(CloudContextPolicy.minimumPlausibleWindow.formatted()) tokens."
                return
            }
            stored = window
        } else {
            windowFeedback[feedbackKey] = "Not saved: enter a whole number of at least \(CloudContextPolicy.minimumPlausibleWindow.formatted()) tokens."
            return
        }

        // The budget this row reports must be the one the send path will use,
        // which resolves the per-model window AND the global fallback. Asking
        // the override alone advertised 16,384 for a model the turn would send
        // 40,000 to, and disagreed with the same model's row in Settings.
        let resolvedWindow = CloudContextPolicy.window(
            forModelId: modelId,
            providerId: provider.id
        )
        let budget = CloudContextPolicy.maximumOutputTokens(window: resolvedWindow)

        // Only this model's draft, and it is REMOVED rather than written back:
        // the field reads storage when it has no draft, so the row now shows
        // exactly what was stored and there is nothing left to drift. Assigning
        // the whole provider's map would also discard an in-progress edit on a
        // sibling row that shares the provider.
        clearWindowDraft(forProviderId: provider.id, modelId: modelId)

        // A save with no window of its own lands on the global fallback, and a
        // budget line that did not say so would read as if this row's clearing
        // chose the number.
        let bindsGlobal = stored == nil && resolvedWindow != nil
        if typed.isEmpty {
            windowFeedback[feedbackKey] = bindsGlobal
                ? "Cleared — the global \(resolvedWindow!.formatted()) applies, reply budget \(budget.formatted())."
                : "Cleared — reply budget now \(budget.formatted())."
        } else {
            windowFeedback[feedbackKey] = "Saved — reply budget up to \(budget.formatted())."
        }
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
            let result = await client.fetchModelIds()
            // The request can outlive the provider: the card is only disabled
            // while testing, but Settings can be closed, storage reloaded, or
            // the record replaced. Writing the catalog for a record that is
            // gone (or now points somewhere else) would stamp one server's
            // models onto another and leave a stale "Connected" row for a
            // future provider that reuses the identifier.
            //
            // Compared by identity and endpoints, not by the whole record: the
            // Prompt Caching picker is a fact about the same server, and
            // flipping it mid-sync used to discard the catalog this request
            // had just fetched.
            guard providerStore.providers.contains(where: {
                $0.id == provider.id
                    && $0.openAIBaseURL == provider.openAIBaseURL
                    && $0.anthropicBaseURL == provider.anthropicBaseURL
            }) else {
                testStates[provider.id] = .untested
                return
            }
            switch result {
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

    private func isTesting(_ provider: CloudProvider) -> Bool {
        testStates[provider.id] == .testing
    }

    private func confirmRemoval(_ confirmation: RemovalConfirmation) {
        let provider = confirmation.provider
        switch confirmation {
        case .provider:
            do {
                try providerStore.remove(provider)
                keyDrafts[provider.id] = ""
                storedProviders.remove(provider.id)
                testStates[provider.id] = .untested
                clearWindowState(forProviderId: provider.id)
            } catch {
                testStates[provider.id] = .failure(error.localizedDescription)
            }
        case .apiKey:
            removeKey(for: provider)
        }
        pendingConfirmation = nil
    }

    private func clearWindowDraft(forProviderId providerId: String, modelId: String) {
        windowDrafts[providerId]?[modelId] = nil
        if windowDrafts[providerId]?.isEmpty == true {
            windowDrafts[providerId] = nil
        }
    }

    private func clearWindowState(forProviderId providerId: String, modelId: String? = nil) {
        if let modelId {
            clearWindowDraft(forProviderId: providerId, modelId: modelId)
            windowFeedback["\(providerId)/\(modelId)"] = nil
        } else {
            windowDrafts[providerId] = nil
            let prefix = "\(providerId)/"
            windowFeedback = windowFeedback.filter { !$0.key.hasPrefix(prefix) }
        }
    }

    /// Forgets the registered catalog only.
    ///
    /// Deliberately NOT `CloudModelRegistry.clearProviderState`: that is the
    /// provider-removal path, and it also drops every recorded price and
    /// context window. Clear answers "the endpoint's catalog is stale", which
    /// says nothing about what its models cost or how much context they hold;
    /// those are hand-typed registration facts with no other control to retype
    /// one, so a re-sync must not delete them.
    private func clearModels(for provider: CloudProvider) {
        CloudModelRegistry.setRegisteredModelIds([], for: provider)
        testStates[provider.id] = .untested
        clearWindowState(forProviderId: provider.id)
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
