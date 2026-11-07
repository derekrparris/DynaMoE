//
//  KeychainMigrationSettingsView.swift
//  DynaMoE
//
//  Settings ▸ Stored Credentials: the one place a credential left in the legacy
//  login keychain by an earlier version is copied into the keychain this build
//  reads.
//
//  The card hides itself when there is nothing to import, so a fresh install and
//  a fully imported one look identical. Everything the card does before the
//  button is pressed is an attributes-only keychain query, which is why it can
//  appear on a settings tab without putting a dialog on screen.
//

import SwiftUI

struct KeychainMigrationSettingsSection: View {
    @State private var pending: [PendingCredential] = []
    @State private var outcomes: [KeychainImportOutcome] = []
    @State private var labels: [String: String] = [:]
    @State private var isImporting: Bool = false

    var body: some View {
        Group {
            if !pending.isEmpty || !outcomes.isEmpty {
                SettingsCard(
                    header: "Stored Credentials",
                    footer: footer,
                    spacing: 10
                ) {
                    SettingsRow(
                        title: "Import stored credentials",
                        subtitle: summary,
                        icon: "key.viewfinder",
                        iconTint: .orange
                    ) {
                        HStack(spacing: 8) {
                            if isImporting {
                                ProgressView()
                                    .controlSize(.small)
                            }
                            Button(isImporting ? "Importing…" : "Import \(pending.count)") {
                                runImport()
                            }
                            .buttonStyle(.borderedProminent)
                            .controlSize(.small)
                            .disabled(isImporting || pending.isEmpty)
                        }
                    }

                    ForEach(pending) { credential in
                        SettingsRowDivider()
                        HStack(spacing: 8) {
                            Text(credential.displayLabel)
                                .font(.system(size: 11.5))
                            Spacer()
                            Text(statusText(for: credential))
                                .font(.system(size: 11))
                                .foregroundStyle(outcome(for: credential.account)?.succeeded == true ? Color.secondary : Color.orange)
                        }
                        .padding(.vertical, 2)
                    }

                    if !outcomes.isEmpty {
                        SettingsRowDivider()
                        VStack(alignment: .leading, spacing: 4) {
                            ForEach(outcomes, id: \.account) { result in
                                Text("\(label(for: result.account)): \(result.message)")
                                    .font(.system(size: 10.5))
                                    .foregroundStyle(result.succeeded ? Color.secondary : Color.orange)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                        }
                        .padding(.vertical, 2)
                    }
                }
            }
        }
        .onAppear(perform: refresh)
    }

    private var summary: String {
        guard !pending.isEmpty else {
            return "Every stored credential is in the keychain this version reads."
        }
        return "\(pending.count) credential\(pending.count == 1 ? "" : "s") still in the older login keychain."
    }

    private var footer: String? {
        guard !pending.isEmpty else { return nil }
        return "macOS asks once per credential during the import. Choose Always Allow so no future update asks again. The older copy is removed only after the new one reads back correctly, so an interrupted import loses nothing."
    }

    private func outcome(for account: String) -> KeychainImportOutcome? {
        outcomes.first { $0.account == account }
    }

    private func label(for account: String) -> String {
        pending.first { $0.account == account }?.displayLabel ?? labels[account] ?? account
    }

    private func statusText(for credential: PendingCredential) -> String {
        guard let result = outcome(for: credential.account) else { return "Waiting" }
        return result.succeeded ? "Imported" : "Needs attention"
    }

    private func refresh() {
        pending = KeychainMigration.pending()
    }

    private func runImport() {
        guard !isImporting, !pending.isEmpty else { return }
        let credentials = pending
        for credential in credentials {
            labels[credential.account] = credential.displayLabel
        }
        isImporting = true
        Task {
            let imported = await Task.detached(priority: .userInitiated) {
                credentials.map { KeychainMigration.adopt($0) }
            }.value

            outcomes = imported
            isImporting = false
            refresh()
            // Imported secrets change what a server can expand and which cloud
            // models are complete, so both consumers re-read their state.
            MCPManager.shared.restartAll()
            LocalModelManager.shared.scanLocalModels()
        }
    }
}
