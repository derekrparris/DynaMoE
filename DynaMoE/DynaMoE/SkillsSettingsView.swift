//
//  SkillsSettingsView.swift
//  DynaMoE
//
//  Phase 8 — Skills: the Agent & Tools card.
//
//  Modeled on `MCPSettingsSection` because both are file-backed capability
//  registries: a master switch, the config path with Open/Reveal/Reload/Edit
//  as JSON, a list of what was found, the issues that stopped something from
//  loading, and a sheet for the raw file. The SubQ card is the one
//  feature-specific addition: a skill that needs a credential should not send
//  the user hunting through another tab to find where it goes.
//

import AppKit
import SwiftUI

struct SkillsSettingsSection: View {
    @ObservedObject private var registry = SkillsRegistry.shared

    @State private var showRawJSON = false
    @State private var previewSkill: Skill?
    @State private var installMessage: String?
    @State private var keyDraft = ""
    @State private var keyError: String?
    @State private var testState: SubQTestState = .idle
    @State private var keyRefresh = 0
    @State private var confirmRemoveKey = false
    // Read once per appearance / mutation instead of in the view body: these
    // are Keychain attribute queries, and a body evaluation runs on every
    // keystroke in the SecureField.
    @State private var keyConfigured = false
    @State private var skillKeyStored = false
    @State private var keyDescription = "Checking for a SubQ key…"

    @AppStorage(SubQDecisionSettings.endpointKey) private var endpoint = SubQDecisionSettings.defaultEndpoint
    @AppStorage(SubQDecisionSettings.modelKey) private var model = SubQDecisionSettings.defaultModel

    private enum SubQTestState: Equatable {
        case idle
        case running
        case success(String)
        case failure(String)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            masterCard
            skillsCard
            subQCard
        }
        .sheet(isPresented: $showRawJSON) {
            SkillsRawJSONSheet()
        }
        .sheet(item: $previewSkill) { skill in
            SkillsPreviewSheet(skill: skill)
        }
        .alert("Remove the SubQ API key?", isPresented: $confirmRemoveKey) {
            Button("Remove", role: .destructive) { removeKey() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("subq_decide will stop working until a key is stored here again or a SubQ Cloud Provider provides one.")
        }
        .onAppear { refreshKeyStatus() }
        .onChange(of: registry.generation) { _, _ in
            // An import from Stored Credentials, a Cloud Provider mutation or
            // an external edit can make a key resolvable without this card
            // appearing again.
            refreshKeyStatus()
        }
        .onChange(of: endpoint) { _, _ in
            // The test result belonged to the previous destination.
            testState = .idle
            keyError = nil
        }
        .onChange(of: model) { _, _ in
            testState = .idle
        }
    }

    private func refreshKeyStatus() {
        keyConfigured = SubQDecisionKeySource.hasAnyKey()
        skillKeyStored = SkillsSecretStore.exists(account: SubQDecisionKeySource.account)
        keyDescription = SubQDecisionKeySource.describeWithoutPrompt()
        // The key can become resolvable without a relaunch (a Cloud Provider
        // added for api.subq.ai, or an import from the legacy keychain), so
        // re-arm the tool whenever this card reads the state.
        AgentHarness.shared.syncSubQDecideAvailability()
    }

    // MARK: Master card

    private var masterCard: some View {
        SettingsCard(header: "Skills", spacing: 10) {
            SettingsRow(
                title: "Enable Skills",
                subtitle: "Advertises installed skills to the agent by name and one-line description. A skill's procedure is read on demand with the skill_load tool, so an unused skill costs a line of prompt rather than its whole body.",
                icon: "book.closed.fill",
                iconTint: .brown
            ) {
                Toggle("Enable Skills", isOn: Binding(
                    get: { registry.config.enabled },
                    set: { registry.setEnabled($0) }
                ))
                .toggleStyle(.switch)
                .labelsHidden()
            }

            SettingsRowDivider()

            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 6) {
                    Image(systemName: "doc.text")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(.secondary)
                    Text(registry.configPath)
                        .font(.system(size: 10.5, design: .monospaced))
                        .foregroundColor(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .textSelection(.enabled)
                    Spacer(minLength: 8)
                    SettingsStatusBadge(
                        text: registry.config.enabled ? "On" : "Off",
                        systemImage: registry.config.enabled ? "circle.fill" : "pause.circle",
                        tint: registry.config.enabled ? .green : .secondary
                    )
                }

                HStack(spacing: 8) {
                    Button {
                        registry.openConfigInEditor()
                    } label: {
                        Label("Open Config File", systemImage: "arrow.up.forward.app")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .buttonStyle(.bordered)

                    Button {
                        registry.revealConfigInFinder()
                    } label: {
                        Label("Reveal", systemImage: "folder")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .buttonStyle(.bordered)

                    Button {
                        registry.requestReload()
                    } label: {
                        Label("Reload", systemImage: "arrow.clockwise")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .buttonStyle(.bordered)

                    Button {
                        showRawJSON = true
                    } label: {
                        Label("Edit as JSON", systemImage: "curlybraces")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .buttonStyle(.bordered)

                    Spacer(minLength: 0)
                }

                if let summary = registry.lastReloadSummary, !summary.isEmpty {
                    Text(summary)
                        .font(.system(size: 10.5))
                        .foregroundColor(.secondary)
                }

                if let error = registry.lastError, !error.isEmpty {
                    Text(error)
                        .font(.system(size: 10.5))
                        .foregroundColor(.red)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    // MARK: Installed skills card

    private var skillsCard: some View {
        SettingsCard(
            header: "Installed Skills (\(registry.allSkills.count))",
            footer: "Skills execute nothing by themselves. Their scripts run through shell_run and inherit its approval rules, and Plan Mode can read a skill but cannot run its scripts because shell_run is withheld there."
        ) {
            if registry.allSkills.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("No skills found.")
                        .font(.system(size: 12.5, weight: .medium))
                    Text("Scanned folders:")
                        .font(.system(size: 11))
                        .foregroundColor(.secondary)
                    ForEach(registry.scannedRoots) { root in
                        HStack(spacing: 6) {
                            Text(root.label)
                                .font(.system(size: 10.5, design: .monospaced))
                                .foregroundColor(.secondary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                            Spacer(minLength: 0)
                        }
                    }
                    HStack(spacing: 8) {
                        Button {
                            installStarter()
                        } label: {
                            Label("Install subq-decisions Starter", systemImage: "arrow.down.doc")
                                .font(.system(size: 11, weight: .medium))
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.brown)

                        Button {
                            chooseRootFolder()
                        } label: {
                            Label("Add Root Folder…", systemImage: "folder.badge.plus")
                                .font(.system(size: 11, weight: .medium))
                        }
                        .buttonStyle(.bordered)
                        Spacer(minLength: 0)
                    }
                }
                .padding(.vertical, 4)
            } else {
                ForEach(Array(registry.allSkills.enumerated()), id: \.element.id) { index, skill in
                    if index > 0 { SettingsRowDivider() }
                    skillRow(skill)
                }

                SettingsRowDivider()

                HStack(spacing: 8) {
                    Button {
                        installStarter()
                    } label: {
                        Label("Install Starter", systemImage: "arrow.down.doc")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .buttonStyle(.bordered)

                    Button {
                        chooseRootFolder()
                    } label: {
                        Label("Add Root Folder…", systemImage: "folder.badge.plus")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .buttonStyle(.bordered)

                    Button {
                        registry.revealSkillsRoot()
                    } label: {
                        Label("App Skills Folder", systemImage: "folder")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .buttonStyle(.bordered)

                    Spacer(minLength: 0)
                }
            }

            if let installMessage {
                Text(installMessage)
                    .font(.system(size: 10.5))
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if !registry.scannedRoots.isEmpty && !registry.allSkills.isEmpty {
                SettingsRowDivider()
                VStack(alignment: .leading, spacing: 4) {
                    Text("Scanned folders (later wins a name collision)")
                        .font(.system(size: 10.5, weight: .medium))
                        .foregroundColor(.secondary)
                    ForEach(registry.scannedRoots) { root in
                        HStack(spacing: 6) {
                            Text("· \(root.label)")
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundColor(.secondary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                            if isExtraRoot(root) {
                                Button {
                                    registry.removeRoot(root.url.standardizedFileURL.path)
                                } label: {
                                    Image(systemName: "minus.circle")
                                        .font(.system(size: 10, weight: .medium))
                                }
                                .buttonStyle(.borderless)
                                .help("Remove \(root.label) from the scanned folders")
                            }
                            Spacer(minLength: 0)
                        }
                    }
                }
                .padding(.vertical, 2)
            }

            if !registry.issues.isEmpty {
                SettingsRowDivider()
                VStack(alignment: .leading, spacing: 5) {
                    ForEach(registry.issues.prefix(10)) { issue in
                        HStack(alignment: .top, spacing: 6) {
                            Image(systemName: issue.severity == .error ? "exclamationmark.triangle.fill" : "info.circle")
                                .font(.system(size: 9.5))
                                .foregroundColor(issue.severity == .error ? .red : .orange)
                                .padding(.top, 1)
                            Text(issue.displayMessage)
                                .font(.system(size: 10.5))
                                .foregroundColor(issue.severity == .error ? .red : .secondary)
                                .fixedSize(horizontal: false, vertical: true)
                            Spacer(minLength: 0)
                            if let path = issue.path {
                                Button {
                                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path, isDirectory: true)])
                                } label: {
                                    Image(systemName: "folder")
                                        .font(.system(size: 10, weight: .medium))
                                }
                                .buttonStyle(.borderless)
                                .help("Reveal \(path) in Finder")
                            }
                        }
                    }
                    if registry.issues.count > 10 {
                        Text("…and \(registry.issues.count - 10) more.")
                            .font(.system(size: 10.5))
                            .foregroundColor(.secondary)
                    }
                }
                .padding(.vertical, 2)
            }
        }
    }

    private func isExtraRoot(_ root: SkillRoot) -> Bool {
        let path = root.url.standardizedFileURL.path
        return registry.config.extraRoots.contains { raw in
            URL(fileURLWithPath: (raw as NSString).expandingTildeInPath, isDirectory: true)
                .standardizedFileURL.path == path
        }
    }

    private func skillRow(_ skill: Skill) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "book.closed.fill")
                .font(.system(size: 11, weight: .semibold))
                .foregroundColor(.brown)
                .padding(.top, 2)

            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(skill.name)
                        .font(.system(size: 12.5, weight: .medium))
                        .lineLimit(1)
                    if skill.always {
                        Text("ALWAYS")
                            .font(.system(size: 8.5, weight: .bold))
                            .foregroundColor(.orange)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1.5)
                            .background(Color.orange.opacity(0.14))
                            .clipShape(Capsule())
                    }
                    Text(skill.rootLabel)
                        .font(.system(size: 9.5, design: .monospaced))
                        .foregroundColor(.secondary)
                        .lineLimit(1)
                }
                Text(skill.description)
                    .font(.system(size: 10.5))
                    .foregroundColor(.secondary)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer(minLength: 6)

            HStack(spacing: 6) {
                Button {
                    previewSkill = skill
                } label: {
                    Image(systemName: "eye")
                        .font(.system(size: 10.5, weight: .medium))
                }
                .buttonStyle(.borderless)
                .help("Preview SKILL.md")

                Button {
                    registry.revealSkillInFinder(skill)
                } label: {
                    Image(systemName: "folder")
                        .font(.system(size: 10.5, weight: .medium))
                }
                .buttonStyle(.borderless)
                .help("Reveal the skill folder in Finder")

                Toggle("Enable \(skill.name)", isOn: Binding(
                    get: { registry.config.isEnabled(skill) },
                    set: { registry.setSkill(skill, enabled: $0) }
                ))
                .toggleStyle(.switch)
                .controlSize(.mini)
                .labelsHidden()
            }
        }
        .padding(.vertical, 3)
        .opacity(registry.config.isEnabled(skill) ? 1 : 0.55)
    }

    // MARK: SubQ card

    private var subQCard: some View {
        SettingsCard(
            header: "SubQ Decisions",
            footer: "subq_decide posts to https://api.subq.ai/v1/systemone and bills input tokens only ($0.05 per 1M; output is free). The key never enters a prompt or a log line. The subq-decisions skill teaches the model when to call it and how to act on the numbers."
        ) {
            SettingsRow(
                title: "Decisions Model",
                subtitle: "subq-snap-preview answers typed questions about a state with probabilities and confidence. It is not a chat model and cannot be registered in Cloud Providers as one.",
                icon: "arrow.triangle.branch",
                iconTint: .indigo
            ) {
                SettingsStatusBadge(
                    text: keyConfigured ? "Ready" : "No Key",
                    systemImage: keyConfigured ? "checkmark.circle.fill" : "key.slash",
                    tint: keyConfigured ? .green : .secondary
                )
                .id(keyRefresh)
            }

            SettingsRowDivider()

            VStack(alignment: .leading, spacing: 6) {
                Text(keyDescription)
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                HStack(spacing: 8) {
                    SecureField("Paste a SubQ API key (console.subq.ai)", text: $keyDraft)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(size: 11))

                    Button {
                        saveKey()
                    } label: {
                        Label("Save Key", systemImage: "key.fill")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.indigo)
                    .disabled(keyDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                    Button {
                        confirmRemoveKey = true
                    } label: {
                        Label("Remove", systemImage: "trash")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .buttonStyle(.bordered)
                    .disabled(!skillKeyStored)

                    Button {
                        runTest()
                    } label: {
                        Label("Test", systemImage: "play.fill")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .buttonStyle(.bordered)
                    .disabled(testState == .running || !keyConfigured)
                }

                if let keyError {
                    Text(keyError)
                        .font(.system(size: 10.5))
                        .foregroundColor(.red)
                }

                switch testState {
                case .idle:
                    EmptyView()
                case .running:
                    Text("Asking SubQ for a test decision…")
                        .font(.system(size: 10.5))
                        .foregroundColor(.secondary)
                case .success(let message):
                    Text(message)
                        .font(.system(size: 10.5))
                        .foregroundColor(.green)
                        .fixedSize(horizontal: false, vertical: true)
                case .failure(let message):
                    Text(message)
                        .font(.system(size: 10.5))
                        .foregroundColor(.red)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(.vertical, 2)

            SettingsRowDivider()

            HStack(spacing: 10) {
                Text("Endpoint")
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
                    .frame(width: 62, alignment: .leading)
                TextField(SubQDecisionSettings.defaultEndpoint, text: $endpoint)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 10.5, design: .monospaced))
            }

            if let warning = SubQDecisionSettings.customEndpointWarning() {
                Text(warning)
                    .font(.system(size: 10.5))
                    .foregroundColor(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let rejection = SubQDecisionSettings.endpointRejection() {
                Text(rejection)
                    .font(.system(size: 10.5))
                    .foregroundColor(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack(spacing: 10) {
                Text("Model")
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
                    .frame(width: 62, alignment: .leading)
                TextField(SubQDecisionSettings.defaultModel, text: $model)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 10.5, design: .monospaced))
            }
        }
    }

    // MARK: Actions

    private func saveKey() {
        let trimmed = keyDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        do {
            try SkillsSecretStore.set(trimmed, account: SubQDecisionKeySource.account)
            keyDraft = ""
            keyError = nil
            testState = .idle
            keyRefresh += 1
            // A configured key is the whole reason subq_decide is seeded into
            // the prompt. Arm it immediately so the next turn can use it
            // without a discovery step.
            _ = try? AgentHarness.shared.loadTool(named: "subq_decide", mode: .code)
            refreshKeyStatus()
        } catch {
            keyError = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        }
    }

    private func removeKey() {
        SkillsSecretStore.delete(account: SubQDecisionKeySource.account)
        keyRefresh += 1
        keyError = nil
        testState = .idle
        refreshKeyStatus()
        if !keyConfigured {
            _ = try? AgentHarness.shared.unloadTool(named: "subq_decide")
        }
    }

    private func runTest() {
        guard let key = SubQDecisionKeySource.resolve() else {
            testState = .failure("No SubQ API key is configured.")
            return
        }
        testState = .running
        let client = SubQDecisionClient(apiKey: key.key)
        Task {
            do {
                let response = try await client.decide(
                    state: AnyCodable("Ticket: 'The export button does nothing after I click it.' Submitted 1 minute ago. Customer tier: free."),
                    questions: [
                        "needs_human": AnyCodable([
                            "type": "noul",
                            "instructions": "Does this ticket need a human before any automated reply?",
                            "criteria": [
                                "true": "Account access or money is at stake",
                                "false": "Routine question a bot can answer"
                            ]
                        ])
                    ]
                )
                testState = .success("Connected (\(key.label)). \(response.readableSummary) · \(response.usageLine)")
            } catch {
                testState = .failure((error as? LocalizedError)?.errorDescription ?? error.localizedDescription)
            }
        }
    }

    private func installStarter() {
        switch registry.installSubQDecisionsStarter() {
        case .success(let url):
            let installedDirectory = url.deletingLastPathComponent().standardizedFileURL.path
            var message = "Installed at \(installedDirectory)."
            // The app folder is scanned LAST among the built-in roots, so an
            // installed starter wins an older ~/.agents or ~/.claude copy. Only
            // an extraRoot scans after it, and that is the single case where the
            // copy just installed is the shadowed one. Say which path matters
            // instead of guessing from a message.
            if let shadow = registry.shadows.first(where: {
                $0.winner.directory.standardizedFileURL.path == installedDirectory
            }) {
                message += " The copy at \(shadow.loser.directory.path) is shadowed by this one and will not be used."
            } else if let shadow = registry.shadows.first(where: {
                $0.loser.directory.standardizedFileURL.path == installedDirectory
            }) {
                message += " A copy at \(shadow.winner.directory.path) is scanned later (an extraRoot) and shadows this one. Remove or rename that copy to use this one."
            }
            installMessage = message
        case .failure(let error):
            installMessage = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        }
    }

    private func chooseRootFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = false
        panel.prompt = "Add Folder"
        panel.message = "Choose a folder whose immediate subfolders are skills (each with its own SKILL.md)."
        if panel.runModal() == .OK, let url = panel.url {
            registry.addRoot(url.path)
        }
    }
}

// MARK: - Preview Sheet

struct SkillsPreviewSheet: View {
    let skill: Skill
    @Environment(\.dismiss) private var dismiss

    private var content: String {
        do {
            return try SkillDiscovery.readText(skill.skillFile)
        } catch {
            return "Could not read \(skill.skillFile.path): \(error.localizedDescription)"
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(skill.name)
                        .font(.system(size: 15, weight: .semibold))
                    Text(skill.skillFile.path)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(.secondary)
                        .textSelection(.enabled)
                }
                Spacer()
                Button("Close") { dismiss() }
                    .keyboardShortcut(.cancelAction)
            }

            ScrollView {
                Text(content)
                    .font(.system(size: 11, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(10)
            }
            .background(Color.settingsCardFill)
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        }
        .padding(18)
        .frame(minWidth: 720, minHeight: 560)
    }
}

// MARK: - Raw JSON Sheet

struct SkillsRawJSONSheet: View {
    @Environment(\.dismiss) private var dismiss
    @State private var text: String = ""
    @State private var saveError: String?
    @State private var didLoad = false

    private var issues: [SkillIssue] {
        guard let data = text.data(using: .utf8),
              let config = try? JSONDecoder().decode(SkillsConfig.self, from: data) else {
            return [SkillIssue(severity: .error, message: "The buffer is not valid JSON for skills.json.")]
        }
        return SkillsConfigValidator.validate(config)
    }

    private var canSave: Bool {
        guard !issues.contains(where: { $0.severity == .error }) else { return false }
        guard let data = text.data(using: .utf8) else { return false }
        return (try? JSONDecoder().decode(SkillsConfig.self, from: data)) != nil
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("skills.json")
                    .font(.system(size: 15, weight: .semibold))
                Spacer()
                Button("Format") {
                    if let pretty = SkillsConfigStore.prettyPrinted(text) { text = pretty }
                }
                Button("Save") { save() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(!canSave)
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)
            }

            TextEditor(text: $text)
                .font(.system(size: 11, design: .monospaced))
                .frame(minHeight: 380)
                .overlay(
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .stroke(Color.secondary.opacity(0.25))
                )

            if let saveError {
                Text(saveError)
                    .font(.system(size: 11))
                    .foregroundColor(.red)
            } else if !issues.isEmpty {
                ForEach(issues.prefix(4)) { issue in
                    Text(issue.displayMessage)
                        .font(.system(size: 10.5))
                        .foregroundColor(issue.severity == .error ? .red : .orange)
                }
            }
        }
        .padding(18)
        .frame(minWidth: 720, minHeight: 560)
        .onAppear {
            guard !didLoad else { return }
            didLoad = true
            text = SkillsRegistry.shared.rawJSONText()
        }
    }

    private func save() {
        do {
            try SkillsRegistry.shared.saveRawJSON(text)
            dismiss()
        } catch {
            saveError = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        }
    }
}
