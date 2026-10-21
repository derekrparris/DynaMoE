//
//  MCPSettingsView.swift
//  DynaMoE
//
//  Phase 7: Model Context Protocol client.
//  Settings surface for MCP servers: status, enable toggles, discovery, the
//  add/edit form, and the raw JSON escape hatch.
//
//  The file on disk is the source of truth. Every control here writes through
//  `MCPManager`, which saves `mcp.json` and reconciles live connections, so the
//  form, the JSON editor, and a hand-edited file cannot disagree.
//

import SwiftUI
import AppKit

// MARK: - Section

struct MCPSettingsSection: View {
    @ObservedObject private var manager = MCPManager.shared
    @AppStorage(MCPManager.enabledKey) private var mcpEnabled: Bool = true

    @State private var showAddServer: Bool = false
    @State private var editingServerKey: String? = nil
    @State private var showRawJSON: Bool = false
    @State private var expandedServers: Set<String> = []
    @State private var confirmingRemoval: MCPServerState? = nil
    @State private var loadFeedback: String? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            masterCard
            serversCard
            toolsCard
        }
        .sheet(isPresented: $showAddServer) {
            MCPAddServerSheet(manager: manager, existingKey: editingServerKey) { savedKey in
                expandedServers.insert(savedKey)
            }
        }
        .sheet(isPresented: $showRawJSON) {
            MCPRawJSONSheet(manager: manager)
        }
        .alert(
            "Remove MCP server?",
            isPresented: Binding(
                get: { confirmingRemoval != nil },
                set: { if !$0 { confirmingRemoval = nil } }
            ),
            presenting: confirmingRemoval
        ) { server in
            Button("Remove", role: .destructive) {
                _ = manager.removeServer(key: server.key)
                expandedServers.remove(server.key)
                confirmingRemoval = nil
            }
            Button("Cancel", role: .cancel) { confirmingRemoval = nil }
        } message: { server in
            Text("\(server.key) will be stopped and its Keychain secrets deleted. The entry is removed from mcp.json.")
        }
    }

    // MARK: Master Card

    private var masterCard: some View {
        SettingsCard(header: "MCP Servers", spacing: 10) {
            SettingsRow(
                title: "Enable MCP Servers",
                subtitle: "Connects configured Model Context Protocol servers and exposes their tools to the agent. Tools stay in the catalog until the model loads them with tools_discover / tools_load.",
                icon: "point.3.connected.trianglepath.dotted",
                iconTint: .indigo
            ) {
                Toggle("Enable MCP Servers", isOn: $mcpEnabled)
                    .toggleStyle(.switch)
                    .labelsHidden()
                    .onChange(of: mcpEnabled) { _, _ in
                        Task { await manager.apply() }
                    }
            }

            SettingsRowDivider()

            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 6) {
                    Image(systemName: "doc.text")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(.secondary)
                    Text(manager.configPath)
                        .font(.system(size: 10.5, design: .monospaced))
                        .foregroundColor(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .textSelection(.enabled)
                    Spacer(minLength: 8)
                }

                HStack(spacing: 8) {
                    Button {
                        manager.openConfigInEditor()
                    } label: {
                        Label("Open Config File", systemImage: "arrow.up.forward.app")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .buttonStyle(.bordered)

                    Button {
                        manager.revealConfigInFinder()
                    } label: {
                        Label("Reveal", systemImage: "folder")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .buttonStyle(.bordered)

                    Button {
                        manager.reloadFromDisk(reason: "manual reload")
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

                    SettingsStatusBadge(
                        text: mcpEnabled ? (manager.isApplying ? "Applying" : "On") : "Off",
                        systemImage: mcpEnabled ? "circle.fill" : "pause.circle",
                        tint: mcpEnabled ? .green : .secondary
                    )
                }

                if let summary = manager.lastReloadSummary, !summary.isEmpty {
                    Text(summary)
                        .font(.system(size: 10.5))
                        .foregroundColor(.secondary)
                }
            }
            .padding(.vertical, 2)

            if let error = manager.lastError, !error.isEmpty {
                SettingsRowDivider()
                HStack(alignment: .top, spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 11))
                        .foregroundColor(.red)
                    Text(error)
                        .font(.system(size: 10.5))
                        .foregroundColor(.red)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                }
            }

            if !manager.configIssues.isEmpty {
                SettingsRowDivider()
                VStack(alignment: .leading, spacing: 5) {
                    Text("Configuration checks")
                        .font(.system(size: 11, weight: .semibold))
                    ForEach(manager.configIssues) { issue in
                        HStack(alignment: .top, spacing: 6) {
                            Image(systemName: issue.severity == .error ? "xmark.octagon.fill" : "exclamationmark.circle.fill")
                                .font(.system(size: 10))
                                .foregroundColor(issue.severity == .error ? .red : .orange)
                            Text(issue.displayMessage)
                                .font(.system(size: 10.5))
                                .foregroundColor(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
            }
        }
    }

    // MARK: Servers Card

    private var serversCard: some View {
        SettingsCard(header: "Configured Servers (\(manager.servers.count))", spacing: 10) {
            if manager.servers.isEmpty {
                emptyState
            } else {
                VStack(spacing: 0) {
                    ForEach(Array(manager.servers.enumerated()), id: \.element.id) { index, server in
                        serverRow(server)
                        if index < manager.servers.count - 1 {
                            SettingsRowDivider()
                        }
                    }
                }
            }

            if !manager.servers.isEmpty {
                SettingsRowDivider()
            }

            HStack(spacing: 8) {
                Button {
                    editingServerKey = nil
                    showAddServer = true
                } label: {
                    Label("Add Server…", systemImage: "plus.circle.fill")
                        .font(.system(size: 11, weight: .medium))
                }
                .buttonStyle(.borderedProminent)
                .tint(.indigo)

                if !manager.servers.isEmpty {
                    Button {
                        manager.restartAll()
                    } label: {
                        Label("Restart All", systemImage: "arrow.clockwise.circle")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .buttonStyle(.bordered)
                }

                if let loadFeedback {
                    Text(loadFeedback)
                        .font(.system(size: 10.5))
                        .foregroundColor(.secondary)
                }

                Spacer(minLength: 0)
            }
            .padding(.top, 2)
        }
    }

    private var emptyState: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("No MCP servers configured.")
                .font(.system(size: 12, weight: .semibold))
                .foregroundColor(.primary)
            Text("Add a server here, or edit mcp.json directly and press Reload. stdio servers are launched as child processes; HTTP servers are reached over Streamable HTTP.")
                .font(.system(size: 11))
                .foregroundColor(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.vertical, 2)
    }

    private func serverRow(_ server: MCPServerState) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Circle()
                    .fill(statusTint(server.status))
                    .frame(width: 8, height: 8)

                Text(server.key)
                    .font(.system(size: 12.5, weight: .semibold, design: .monospaced))

                SettingsStatusBadge(
                    text: server.status == .ready ? "Ready · \(server.toolCount)" : server.status.label,
                    systemImage: statusIcon(server.status),
                    tint: statusTint(server.status)
                )

                if let error = server.error, !error.isEmpty, server.status == .failed {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 10))
                        .foregroundColor(.orange)
                        .help(error)
                }

                Spacer(minLength: 8)

                Text(server.transportLabel)
                    .font(.system(size: 9.5, weight: .semibold, design: .monospaced))
                    .foregroundColor(.secondary)
                    .padding(.horizontal, 5)
                    .padding(.vertical, 1)
                    .background(Color.secondary.opacity(0.12))
                    .cornerRadius(4)

                Toggle("Enabled", isOn: Binding(
                    get: { server.enabled },
                    set: { newValue in _ = manager.setEnabled(newValue, for: server.key) }
                ))
                .toggleStyle(.switch)
                .labelsHidden()
                .controlSize(.mini)
            }

            Text(server.endpoint)
                .font(.system(size: 10, design: .monospaced))
                .foregroundColor(.secondary)
                .lineLimit(2)
                .truncationMode(.middle)
                .textSelection(.enabled)

            if let error = server.error, !error.isEmpty {
                Text(error)
                    .font(.system(size: 10))
                    .foregroundColor(server.status == .failed ? .red : .orange)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
            }

            HStack(spacing: 8) {
                Button {
                    manager.restart(key: server.key)
                } label: {
                    Label("Restart", systemImage: "arrow.clockwise")
                        .font(.system(size: 10.5, weight: .medium))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button {
                    editingServerKey = server.key
                    showAddServer = true
                } label: {
                    Label("Edit", systemImage: "slider.horizontal.3")
                        .font(.system(size: 10.5, weight: .medium))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button(role: .destructive) {
                    confirmingRemoval = server
                } label: {
                    Label("Remove", systemImage: "trash")
                        .font(.system(size: 10.5, weight: .medium))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Spacer(minLength: 0)
            }

            toolDisclosure(server)
        }
        .padding(.vertical, 6)
    }

    @ViewBuilder
    private func toolDisclosure(_ server: MCPServerState) -> some View {
        if !server.qualifiedToolNames.isEmpty {
            let isExpanded = expandedServers.contains(server.key)
            Button {
                if isExpanded {
                    expandedServers.remove(server.key)
                } else {
                    expandedServers.insert(server.key)
                }
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                        .font(.system(size: 9, weight: .bold))
                    Text("\(server.toolCount) tool\(server.toolCount == 1 ? "" : "s") discovered")
                        .font(.system(size: 10.5, weight: .medium))
                }
                .foregroundColor(.secondary)
            }
            .buttonStyle(.plain)

            if isExpanded {
                let loaded = Set(manager.loadedToolNames(for: server.key))
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(server.qualifiedToolNames, id: \.self) { name in
                        HStack(spacing: 6) {
                            Text(name)
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundColor(.primary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                            Spacer(minLength: 6)
                            if loaded.contains(name) {
                                Text("Loaded")
                                    .font(.system(size: 9.5, weight: .semibold))
                                    .foregroundColor(.green)
                            } else {
                                Button("Load") {
                                    if manager.loadTool(named: name) {
                                        loadFeedback = "Loaded \(name)"
                                    } else {
                                        loadFeedback = manager.lastError
                                    }
                                }
                                .buttonStyle(.borderless)
                                .controlSize(.mini)
                                .font(.system(size: 10, weight: .medium))
                            }
                        }
                    }
                }
                .padding(.leading, 14)
                .padding(.top, 2)
            }
        } else if server.status == .ready {
            Text("Connected, but the server advertises no tools.")
                .font(.system(size: 10.5))
                .foregroundColor(.secondary)
        }
    }

    // MARK: Tools Card

    private var toolsCard: some View {
        SettingsCard(header: "MCP Tool Exposure", spacing: 10) {
            SettingsRow(
                title: "Discovered MCP tools",
                subtitle: manager.discoveredToolCount == 0
                    ? "No MCP tools are registered. Connect a server to add them to the tool catalog."
                    : "\(manager.discoveredToolCount) MCP tool\(manager.discoveredToolCount == 1 ? " is" : "s are") registered in the catalog and discoverable with tools_discover (category \"mcp\"). They are not loaded into the prompt until the model requests them, so they cost no prefill tokens.",
                icon: "wrench.adjustable",
                iconTint: .teal
            ) {
                EmptyView()
            }

            SettingsRowDivider()

            HStack(alignment: .top, spacing: 6) {
                Image(systemName: "lock.shield.fill")
                    .font(.system(size: 11))
                    .foregroundColor(.orange)
                Text("Every MCP tool call asks for confirmation unless Turbo Mode is on. MCP servers run with the same privileges as shell_run: they are external programs and their tool annotations are not trusted. Credentials are stored in the Keychain, never in mcp.json.")
                    .font(.system(size: 10.5))
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    // MARK: Helpers

    private func statusTint(_ status: MCPServerStatus) -> Color {
        switch status {
        case .ready: return .green
        case .connecting: return .orange
        case .failed: return .red
        case .disabled, .stopped: return .secondary
        }
    }

    private func statusIcon(_ status: MCPServerStatus) -> String {
        switch status {
        case .ready: return "checkmark.circle.fill"
        case .connecting: return "arrow.triangle.2.circlepath"
        case .failed: return "xmark.circle.fill"
        case .disabled: return "pause.circle"
        case .stopped: return "circle"
        }
    }
}

// MARK: - Add / Edit Sheet

struct MCPAddServerSheet: View {
    @ObservedObject var manager: MCPManager
    let existingKey: String?
    let onSaved: (String) -> Void

    @Environment(\.dismiss) private var dismiss

    private enum Transport: String, CaseIterable, Identifiable {
        case stdio = "stdio"
        case http = "http"
        var id: String { rawValue }
    }

    @State private var serverKey: String = ""
    @State private var transport: Transport = .stdio
    @State private var enabled: Bool = true
    @State private var command: String = ""
    @State private var argsJSON: String = ""
    @State private var workingDirectory: String = ""
    @State private var url: String = ""
    @State private var startupTimeout: String = ""
    @State private var toolTimeout: String = ""
    @State private var envRows: [MCPKeyValueDraft] = []
    @State private var headerRows: [MCPKeyValueDraft] = []
    @State private var errorText: String? = nil
    @State private var didLoad = false

    private var isEditing: Bool { existingKey != nil }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(isEditing ? "Edit MCP Server" : "Add MCP Server")
                        .font(.system(size: 15, weight: .bold))
                    Text("Saved to \(MCPConfigStore.defaultPathDescription)")
                        .font(.system(size: 10.5))
                        .foregroundColor(.secondary)
                }
                Spacer()
                Toggle("Enabled", isOn: $enabled)
                    .toggleStyle(.switch)
                    .controlSize(.small)
            }
            .padding(.horizontal, 18)
            .padding(.top, 18)
            .padding(.bottom, 12)

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    formSection("Identity") {
                        labeledField("Server name", text: $serverKey, placeholder: "filesystem", isMonospaced: true)
                            .disabled(isEditing)
                        Text("Letters, digits, '-' and '_' only. Tools are exposed as mcp__<server>__<tool>.")
                            .font(.system(size: 10))
                            .foregroundColor(.secondary)
                    }

                    formSection("Transport") {
                        Picker("", selection: $transport) {
                            Text("stdio (local process)").tag(Transport.stdio)
                            Text("Streamable HTTP (remote)").tag(Transport.http)
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()

                        if transport == .stdio {
                            labeledField("Command", text: $command, placeholder: "npx", isMonospaced: true)
                            labeledMultiline(
                                "Arguments (JSON array)",
                                text: $argsJSON,
                                placeholder: "[\"-y\", \"@modelcontextprotocol/server-filesystem\", \"${workspace}\"]"
                            )
                            Text("A JSON array of strings, for example [\"-y\", \"server\"]. Each element is passed as one argument, so spaces, empty strings and multi-line values survive an edit.")
                                .font(.system(size: 9.5))
                                .foregroundColor(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                            labeledField("Working directory (optional)", text: $workingDirectory, placeholder: "${workspace}", isMonospaced: true)
                            keyValueEditor(
                                title: "Environment",
                                rows: $envRows,
                                serverKey: effectiveServerKey,
                                placeholderKey: "VAR_NAME",
                                hint: "One variable per row. Names may contain letters, digits and '_'."
                            )
                        } else {
                            labeledField("URL", text: $url, placeholder: "https://mcp.example.com/mcp", isMonospaced: true)
                            keyValueEditor(
                                title: "Headers",
                                rows: $headerRows,
                                serverKey: effectiveServerKey,
                                placeholderKey: "Authorization",
                                hint: "One header per row: name on the left, full value on the right (Authorization -> Bearer <token>). A name cannot contain spaces or commas."
                            )
                        }
                    }

                    formSection("Timeouts") {
                        HStack(spacing: 12) {
                            labeledField("Startup (seconds)", text: $startupTimeout, placeholder: "30", isMonospaced: true)
                            labeledField("Per tool call (seconds)", text: $toolTimeout, placeholder: "120", isMonospaced: true)
                        }
                    }

                    Text("Tokens: ${env:NAME} reads an environment variable, ${workspace} is the agent working directory, and secret fields are stored in the Keychain as ${keychain:account}.")
                        .font(.system(size: 10))
                        .foregroundColor(.secondary)
                        .fixedSize(horizontal: false, vertical: true)

                    if let errorText {
                        HStack(alignment: .top, spacing: 6) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .font(.system(size: 11))
                                .foregroundColor(.red)
                            Text(errorText)
                                .font(.system(size: 11))
                                .foregroundColor(.red)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
                .padding(18)
            }

            Divider()

            HStack(spacing: 10) {
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Spacer()
                Button(isEditing ? "Save Changes" : "Add Server") { save() }
                    .keyboardShortcut(.defaultAction)
                    .buttonStyle(.borderedProminent)
                    .tint(.indigo)
            }
            .padding(.horizontal, 18)
            .padding(.vertical, 12)
        }
        .frame(width: 620, height: 640)
        .onAppear(perform: loadExisting)
    }

    // MARK: Form pieces

    private var effectiveServerKey: String {
        existingKey ?? serverKey.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    @ViewBuilder
    private func formSection<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            Text(title.uppercased())
                .font(.system(size: 9.5, weight: .bold))
                .foregroundColor(.secondary)
                .tracking(0.6)
            content()
        }
    }

    private func labeledField(_ title: String, text: Binding<String>, placeholder: String, isMonospaced: Bool = false) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .font(.system(size: 10.5, weight: .medium))
                .foregroundColor(.secondary)
            TextField(placeholder, text: text)
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 11.5, design: isMonospaced ? .monospaced : .default))
        }
    }

    private func labeledMultiline(_ title: String, text: Binding<String>, placeholder: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .font(.system(size: 10.5, weight: .medium))
                .foregroundColor(.secondary)
            TextEditor(text: text)
                .font(.system(size: 11, design: .monospaced))
                .frame(height: 74)
                .padding(4)
                .background(Color.settingsCardFill)
                .cornerRadius(6)
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(Color.secondary.opacity(0.25), lineWidth: 1)
                )
        }
    }

    private func keyValueEditor(
        title: String,
        rows: Binding<[MCPKeyValueDraft]>,
        serverKey: String,
        placeholderKey: String,
        hint: String? = nil
    ) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text(title)
                    .font(.system(size: 10.5, weight: .medium))
                    .foregroundColor(.secondary)
                Spacer()
                Button {
                    rows.wrappedValue.append(MCPKeyValueDraft())
                } label: {
                    Label("Add", systemImage: "plus")
                        .font(.system(size: 10, weight: .medium))
                }
                .buttonStyle(.borderless)
            }

            if rows.wrappedValue.isEmpty {
                Text("None")
                    .font(.system(size: 10.5))
                    .foregroundColor(.secondary)
            }

            ForEach(rows) { $draft in
                MCPKeyValueRow(
                    placeholderKey: placeholderKey,
                    draft: $draft,
                    onDelete: {
                        let target = $draft.wrappedValue.id
                        rows.wrappedValue.removeAll { $0.id == target }
                    }
                )
            }

            if let hint {
                Text(hint)
                    .font(.system(size: 9.5))
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if rows.wrappedValue.contains(where: { $0.isSecret }) {
                Text("Keychain account: dynamoe.mcp.<server>.<field>. The server must be named before saving a secret.")
                    .font(.system(size: 9.5))
                    .foregroundColor(.secondary)
            }
        }
    }

    // MARK: Load / Save

    private func loadExisting() {
        guard !didLoad else { return }
        didLoad = true
        guard let key = existingKey, let config = manager.document.mcpServers[key] else {
            envRows = []
            headerRows = []
            return
        }

        serverKey = key
        enabled = config.enabled
        transport = config.transportKind == .http ? .http : .stdio
        command = config.command ?? ""
        argsJSON = Self.argumentsJSON(config.args ?? [])
        workingDirectory = config.cwd ?? ""
        url = config.url ?? ""
        startupTimeout = config.startupTimeoutSec.map { Self.format($0) } ?? ""
        toolTimeout = config.toolTimeoutSec.map { Self.format($0) } ?? ""
        envRows = Self.drafts(from: config.env ?? [:])
        headerRows = Self.drafts(from: config.headers ?? [:])
    }

    private static func drafts(from values: [String: String]) -> [MCPKeyValueDraft] {
        values.keys.sorted().map { key in
            let raw = values[key] ?? ""
            if let account = MCPTokenExpander.keychainAccount(in: raw) {
                return MCPKeyValueDraft(key: key, value: "", isSecret: true, hasStoredSecret: true, account: account)
            }
            return MCPKeyValueDraft(key: key, value: raw, isSecret: false, hasStoredSecret: false, account: nil)
        }
    }

    private static func format(_ value: Double) -> String {
        value == value.rounded() ? String(Int(value)) : String(value)
    }

    /// The editor's representation of `args`: JSON, so it is exactly
    /// reversible. A line-per-argument editor cannot round-trip an argument
    /// that is empty, padded with spaces, or contains a newline.
    static func argumentsJSON(_ args: [String]) -> String {
        guard !args.isEmpty else { return "" }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        guard let data = try? encoder.encode(args),
              let text = String(data: data, encoding: .utf8) else { return "" }
        return text
    }

    /// Reads the argument editor back. Nil means "not a JSON array of
    /// strings"; an empty editor and an empty array both mean no arguments.
    static func parseArguments(_ text: String) -> [String]? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        guard let data = trimmed.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode([String].self, from: data)
    }

    private func save() {
        errorText = nil
        let key = existingKey ?? serverKey.trimmingCharacters(in: .whitespacesAndNewlines)

        if key.isEmpty {
            errorText = "Give the server a name."
            return
        }
        if !isEditing, manager.document.mcpServers[key] != nil {
            errorText = "A server named '\(key)' already exists."
            return
        }

        var config = MCPServerConfig()
        config.enabled = enabled
        config.startupTimeoutSec = Double(startupTimeout.trimmingCharacters(in: .whitespacesAndNewlines))
        config.toolTimeoutSec = Double(toolTimeout.trimmingCharacters(in: .whitespacesAndNewlines))

        switch transport {
        case .stdio:
            let trimmedCommand = command.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmedCommand.isEmpty else {
                errorText = "A stdio server needs a command (for example npx)."
                return
            }
            config.command = trimmedCommand
            let rawArgs = argsJSON.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let parsedArgs = Self.parseArguments(rawArgs) else {
                errorText = "Arguments must be a JSON array of strings, for example [\"-y\", \"server\"]."
                return
            }
            config.args = parsedArgs.isEmpty ? nil : parsedArgs
            let cwd = workingDirectory.trimmingCharacters(in: .whitespacesAndNewlines)
            config.cwd = cwd.isEmpty ? nil : cwd
            let env = resolve(rows: envRows, serverKey: key, fieldLabel: "env")
            if let failure = env.error {
                errorText = failure
                return
            }
            config.env = env.values.isEmpty ? nil : env.values
        case .http:
            let trimmedURL = url.trimmingCharacters(in: .whitespacesAndNewlines)
            guard URL(string: trimmedURL)?.host != nil else {
                errorText = "Enter a valid absolute URL."
                return
            }
            config.url = trimmedURL
            let headers = resolve(rows: headerRows, serverKey: key, fieldLabel: "header")
            if let failure = headers.error {
                errorText = failure
                return
            }
            config.headers = headers.values.isEmpty ? nil : headers.values
        }

        let issues = MCPConfigValidator.validate(server: config, key: key)
            .filter { $0.severity == .error }
        if let first = issues.first {
            errorText = first.message
            return
        }

        guard manager.upsertServer(key: key, config: config) else {
            errorText = manager.lastError ?? "Could not save the server."
            return
        }
        onSaved(key)
        dismiss()
    }

    private func resolve(
        rows: [MCPKeyValueDraft],
        serverKey: String,
        fieldLabel: String
    ) -> (values: [String: String], error: String?) {
        var values: [String: String] = [:]
        for row in rows {
            let name = row.key.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !name.isEmpty else { continue }

            if row.isSecret {
                guard !serverKey.isEmpty else {
                    return (values, "Name the server before storing a \(fieldLabel) value in the Keychain.")
                }
                let account = row.account ?? MCPSecretStore.account(serverKey: serverKey, field: name)
                let newValue = row.value
                if !newValue.isEmpty {
                    do {
                        try MCPSecretStore.set(newValue, account: account)
                    } catch {
                        return (values, "Keychain error for \(name): \(error.localizedDescription)")
                    }
                } else if !MCPSecretStore.exists(account: account) {
                    return (values, "No Keychain secret stored for \(name). Type the value or untick Keychain.")
                }
                values[name] = MCPTokenExpander.keychainToken(account: account)
            } else {
                values[name] = row.value
            }
        }
        return (values, nil)
    }
}

private struct MCPKeyValueRow: View {
    let placeholderKey: String
    @Binding var draft: MCPKeyValueDraft
    let onDelete: () -> Void

    var body: some View {
        HStack(spacing: 6) {
            TextField(placeholderKey, text: $draft.key)
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 10.5, design: .monospaced))
                .frame(width: 190)

            if draft.isSecret {
                SecureField(
                    draft.hasStoredSecret && draft.value.isEmpty
                        ? "\u{2022}\u{2022}\u{2022}\u{2022} stored in Keychain"
                        : "Secret value",
                    text: $draft.value
                )
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 10.5))
            } else {
                TextField("Value", text: $draft.value)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 10.5, design: .monospaced))
            }

            Toggle("Keychain", isOn: $draft.isSecret)
                .toggleStyle(.checkbox)
                .font(.system(size: 10))
                .help("Store this value in the Keychain instead of mcp.json")

            Button(action: onDelete) {
                Image(systemName: "minus.circle")
                    .font(.system(size: 11))
            }
            .buttonStyle(.borderless)
        }
    }
}

struct MCPKeyValueDraft: Identifiable, Equatable {
    let id = UUID()
    var key: String = ""
    var value: String = ""
    var isSecret: Bool = false
    var hasStoredSecret: Bool = false
    var account: String? = nil
}

// MARK: - Raw JSON Sheet

struct MCPRawJSONSheet: View {
    @ObservedObject var manager: MCPManager
    @Environment(\.dismiss) private var dismiss

    @State private var text: String = ""
    @State private var issues: [MCPConfigIssue] = []
    @State private var saveError: String? = nil
    @State private var didLoad = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("mcp.json")
                        .font(.system(size: 15, weight: .bold))
                    Text(MCPConfigStore.defaultPathDescription)
                        .font(.system(size: 10.5, design: .monospaced))
                        .foregroundColor(.secondary)
                }
                Spacer()
                Button {
                    if let pretty = MCPConfigStore.prettyPrinted(text) {
                        text = pretty
                        revalidate()
                    }
                } label: {
                    Label("Format", systemImage: "text.alignleft")
                        .font(.system(size: 11, weight: .medium))
                }
                .buttonStyle(.bordered)
            }
            .padding(.horizontal, 18)
            .padding(.top, 18)
            .padding(.bottom, 12)

            Divider()

            TextEditor(text: $text)
                .font(.system(size: 11.5, design: .monospaced))
                .padding(10)
                .onChange(of: text) { _, _ in revalidate() }

            Divider()

            VStack(alignment: .leading, spacing: 6) {
                if let saveError {
                    issueLine(severity: .error, message: saveError)
                }
                if issues.isEmpty {
                    if saveError == nil {
                        Text("Valid configuration.")
                            .font(.system(size: 10.5))
                            .foregroundColor(.green)
                    }
                } else {
                    ForEach(issues.prefix(6)) { issue in
                        issueLine(severity: issue.severity, message: issue.displayMessage)
                    }
                }
            }
            .padding(.horizontal, 18)
            .padding(.top, 8)

            HStack {
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Spacer()
                Button("Save") { save() }
                    .keyboardShortcut(.defaultAction)
                    .buttonStyle(.borderedProminent)
                    .tint(.indigo)
                    .disabled(!canSave)
            }
            .padding(.horizontal, 18)
            .padding(.vertical, 12)
        }
        .frame(width: 760, height: 640)
        .onAppear {
            guard !didLoad else { return }
            didLoad = true
            text = manager.rawJSONText()
            revalidate()
        }
    }

    private var canSave: Bool {
        MCPConfigValidator.hasErrors(issues) == false && saveError == nil
    }

    private func issueLine(severity: MCPConfigIssueSeverity, message: String) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Image(systemName: severity == .error ? "xmark.octagon.fill" : "exclamationmark.circle.fill")
                .font(.system(size: 10))
                .foregroundColor(severity == .error ? .red : .orange)
            Text(message)
                .font(.system(size: 10.5))
                .foregroundColor(severity == .error ? .red : .secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func revalidate() {
        saveError = nil
        guard let data = text.data(using: .utf8) else {
            issues = [MCPConfigIssue(severity: .error, message: "Text is not valid UTF-8.")]
            return
        }
        do {
            let document = try JSONDecoder().decode(MCPConfigDocument.self, from: data)
            issues = MCPConfigValidator.validate(document)
        } catch {
            issues = [MCPConfigIssue(severity: .error, message: "Invalid JSON: \(error.localizedDescription)")]
        }
    }

    private func save() {
        if manager.saveRawJSON(text) {
            dismiss()
        } else {
            saveError = manager.lastError ?? "Could not save the configuration."
        }
    }
}
