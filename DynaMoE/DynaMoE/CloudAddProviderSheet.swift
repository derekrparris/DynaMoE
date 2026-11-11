//
//  CloudAddProviderSheet.swift
//  DynaMoE
//
//  Settings ▸ Cloud Providers: the add/edit form for a runtime-configured
//  provider. Modeled on MCPAddServerSheet: identity, endpoint, and credential
//  are separate sections, validation is shown inline, and the identifier is
//  read-only once the provider exists.
//

import SwiftUI

struct CloudAddProviderSheet: View {
    @ObservedObject var store: CloudProviderStore
    let provider: CloudProvider?
    var onSaved: (CloudProvider) -> Void = { _ in }

    @Environment(\.dismiss) private var dismiss

    @State private var displayName: String = ""
    @State private var identifier: String = ""
    @State private var identifierWasEdited: Bool = false
    @State private var baseURL: String = ""
    @State private var docsURL: String = ""
    @State private var apiKey: String = ""
    @State private var errorText: String? = nil
    @State private var didLoad = false

    private var isEditing: Bool { provider != nil }

    /// Explicit because the private `@State` fields would otherwise make the
    /// synthesized memberwise initializer private to this file.
    init(
        store: CloudProviderStore,
        provider: CloudProvider?,
        onSaved: @escaping (CloudProvider) -> Void = { _ in }
    ) {
        self.store = store
        self.provider = provider
        self.onSaved = onSaved
    }

    /// Editing never renames: the identifier is the Keychain account suffix
    /// and part of the `cloud://` address saved with every conversation.
    private var editingIdentifier: String? { provider?.id }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    identitySection
                    endpointSection
                    if !isEditing {
                        credentialSection
                    }
                }
                .padding(18)
            }
            // Pinned above the footer rather than scrolled with the form: the
            // reason "Add Provider" is disabled must be visible without
            // discovering that the sheet scrolls.
            if let message = validationMessage ?? errorText {
                errorLabel(message)
                    .padding(.horizontal, 18)
                    .padding(.vertical, 10)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            Divider()
            footer
        }
        .frame(width: 520, height: 500)
        .onAppear(perform: load)
    }

    // MARK: Header & Footer

    private var header: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(isEditing ? "Edit Provider" : "Add Provider")
                .font(.system(size: 15, weight: .bold))
            Text("Any endpoint that speaks OpenAI Chat Completions and lists models over GET /models. The key is stored in the Keychain (DRP.DynaMoE.cloud); nothing secret is written to disk.")
                .font(.system(size: 10.5))
                .foregroundColor(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, 18)
        .padding(.top, 18)
        .padding(.bottom, 12)
    }

    private var footer: some View {
        HStack(spacing: 10) {
            if isEditing {
                Text("Rotate the API key on the provider card.")
                    .font(.system(size: 10))
                    .foregroundColor(.secondary)
            }
            Spacer()
            Button("Cancel") { dismiss() }
                .keyboardShortcut(.cancelAction)
            Button(isEditing ? "Save Changes" : "Add Provider") { save() }
                .keyboardShortcut(.defaultAction)
                .disabled(!isDraftValid)
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 14)
    }

    // MARK: Sections

    private var identitySection: some View {
        formSection("Identity") {
            labeledField("Display name", text: displayNameBinding, prompt: "OpenCode Zen")
            if let editingIdentifier {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Identifier")
                        .font(.system(size: 11, weight: .medium))
                    Text(editingIdentifier)
                        .font(.system(size: 12, design: .monospaced))
                        .textSelection(.enabled)
                }
                caption("Saved with every conversation that uses this provider, so it cannot change.")
            } else {
                labeledField(
                    "Identifier",
                    text: identifierBinding,
                    prompt: "opencode-zen",
                    isMonospaced: true
                )
                caption("Lowercase letters, digits, and dashes. Derived from the display name until you type one.")
            }
        }
    }

    private var endpointSection: some View {
        formSection("Endpoint") {
            labeledField(
                "API base URL",
                text: $baseURL,
                prompt: "https://opencode.ai/zen/v1",
                isMonospaced: true
            )
            caption("Include the version segment. Enter the base, not /chat/completions or /models. https:// is required except for loopback hosts. Put the key in the credential field, never in the URL.")
            labeledField(
                "Documentation (optional)",
                text: $docsURL,
                prompt: "https://opencode.ai/docs",
                isMonospaced: true
            )
        }
    }

    private var credentialSection: some View {
        formSection("Credential") {
            VStack(alignment: .leading, spacing: 4) {
                Text("API key (optional)")
                    .font(.system(size: 11, weight: .medium))
                SecureField("sk-…", text: $apiKey)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityLabel("API key")
            }
            caption("You can also save or replace it later on the provider card.")
        }
    }

    // MARK: Validation

    private var validation: Result<CloudProvider, CloudProviderValidationError> {
        let existing = store.providers.filter { $0.id != editingIdentifier }
        return CloudProviderCatalog.validate(
            displayName: displayName,
            identifier: editingIdentifier ?? identifier,
            baseURL: baseURL,
            docsURL: docsURL,
            existing: existing
        )
    }

    private var isDraftValid: Bool {
        if case .success = validation { return true }
        return false
    }

    /// A blank form should read as blank, not as an error, so the message
    /// only appears once something has been typed.
    private var validationMessage: String? {
        let touched = !displayName.isEmpty || !baseURL.isEmpty || !identifier.isEmpty
        guard touched, case .failure(let failure) = validation else { return nil }
        return failure.errorDescription
    }

    // MARK: Bindings

    /// Keeping the slug derived until the identifier is typed by hand avoids
    /// an onChange feedback loop between the two fields.
    private var displayNameBinding: Binding<String> {
        Binding(
            get: { displayName },
            set: { newValue in
                displayName = newValue
                guard !isEditing, !identifierWasEdited else { return }
                identifier = CloudProviderCatalog.slug(forDisplayName: newValue)
            }
        )
    }

    private var identifierBinding: Binding<String> {
        Binding(
            get: { identifier },
            set: { newValue in
                identifier = newValue.lowercased()
                identifierWasEdited = true
            }
        )
    }

    // MARK: Actions

    private func load() {
        guard !didLoad, let provider else {
            didLoad = true
            return
        }
        displayName = provider.displayName
        identifier = provider.id
        baseURL = provider.openAIBaseURL.absoluteString
        docsURL = provider.docsURL?.absoluteString ?? ""
        didLoad = true
    }

    private func save() {
        errorText = nil
        do {
            let saved: CloudProvider
            if let provider {
                try store.update(
                    provider,
                    displayName: displayName,
                    baseURL: baseURL,
                    docsURL: docsURL
                )
                saved = store.providers.first { $0.id == provider.id } ?? provider
            } else {
                saved = try store.add(
                    displayName: displayName,
                    identifier: identifier,
                    baseURL: baseURL,
                    docsURL: docsURL,
                    apiKey: apiKey
                )
            }
            onSaved(saved)
            dismiss()
        } catch {
            errorText = error.localizedDescription
        }
    }

    // MARK: Presentation helpers

    private func formSection<Content: View>(
        _ title: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.system(size: 10.5, weight: .semibold))
                .foregroundColor(.secondary)
            VStack(alignment: .leading, spacing: 10) {
                content()
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(Color.settingsCardFill)
            )
        }
    }

    private func labeledField(
        _ title: String,
        text: Binding<String>,
        prompt: String,
        isMonospaced: Bool = false
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.system(size: 11, weight: .medium))
            TextField(prompt, text: text)
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 12, design: isMonospaced ? .monospaced : .default))
        }
    }

    private func caption(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 9.5))
            .foregroundColor(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    private func errorLabel(_ message: String) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 11))
                .foregroundColor(.red)
            Text(message)
                .font(.system(size: 11))
                .foregroundColor(.red)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
