//
//  MCPManager.swift
//  DynaMoE
//
//  Phase 7: Model Context Protocol client.
//  Owns the lifecycle of every configured MCP server: config load, connect,
//  tool discovery, registration into `AgentHarness`, and teardown.
//
//  The shape mirrors `ServerController`: one MainActor `ObservableObject`
//  singleton, `@Published` state for the settings UI, and a single
//  `applyRequestedState`-style entry point (`apply()`) that reconciles live
//  connections against the config file. Settings never connects a server
//  directly; it writes the file and asks the manager to reconcile.
//

import Foundation
import Combine
import AppKit

// MARK: - State

public enum MCPServerStatus: String, Sendable, Equatable {
    case disabled
    case connecting
    case ready
    case failed
    case stopped

    public var label: String {
        switch self {
        case .disabled: return "Disabled"
        case .connecting: return "Connecting"
        case .ready: return "Ready"
        case .failed: return "Failed"
        case .stopped: return "Stopped"
        }
    }

    public var isConnected: Bool {
        self == .ready
    }
}

public struct MCPServerState: Identifiable, Sendable, Equatable {
    public let key: String
    public let enabled: Bool
    public let status: MCPServerStatus
    public let error: String?
    public let endpoint: String
    public let transportLabel: String
    public let instructions: String?
    public let qualifiedToolNames: [String]
    public let remoteToolNames: [String]

    public var id: String { key }
    public var toolCount: Int { qualifiedToolNames.count }
    public var isConnected: Bool { status == .ready }
}

// MARK: - Manager

@MainActor
public final class MCPManager: ObservableObject {
    public static let shared = MCPManager()

    /// Master switch, mirroring `ServerController.enabledKey`. Absent means on.
    public nonisolated static let enabledKey = "dynamoe_mcp_enabled"
    /// Agent tools must be enabled for MCP tools to mean anything.
    public nonisolated static let agentToolsKey = "dynamoe_agent_tools_enabled"

    @Published public private(set) var servers: [MCPServerState] = []
    @Published public private(set) var configIssues: [MCPConfigIssue] = []
    @Published public private(set) var lastError: String?
    @Published public private(set) var lastReloadSummary: String?
    @Published public private(set) var configPath: String = MCPConfigStore.defaultPathDescription
    @Published public private(set) var isApplying = false
    @Published public private(set) var discoveredToolCount: Int = 0

    /// The file as last read from disk. Settings mutates this and saves through
    /// the manager so the in-memory copy and the file cannot diverge.
    public private(set) var document = MCPConfigDocument()

    private struct MCPConnection {
        let client: MCPClient
        let config: MCPServerConfig
        let bridges: [MCPToolBridge]
    }

    private struct ServerRuntime {
        var status: MCPServerStatus = .stopped
        var error: String?
        var instructions: String?
        var qualifiedToolNames: [String] = []
        var remoteToolNames: [String] = []
    }

    private var connections: [String: MCPConnection] = [:]
    private var runtime: [String: ServerRuntime] = [:]
    private var connecting: Set<String> = []
    private var connectTasks: [String: Task<Void, Never>] = [:]
    private var watcher: DispatchSourceFileSystemObject?
    private var watcherDescriptor: Int32 = -1
    private var configToken: String?
    private var applyInFlight = false
    private var applyRequested = false
    private var pendingReload = false
    private var deferredReloadTask: Task<Void, Never>?
    private var terminationObserver: NSObjectProtocol?

    private init() {}

    // MARK: Requested State

    public nonisolated static var isEnabledRequested: Bool {
        if UserDefaults.standard.object(forKey: enabledKey) == nil { return true }
        return UserDefaults.standard.bool(forKey: enabledKey)
    }

    public nonisolated static var agentToolsEnabled: Bool {
        if UserDefaults.standard.object(forKey: agentToolsKey) == nil { return true }
        return UserDefaults.standard.bool(forKey: agentToolsKey)
    }

    /// True when the master switch, the agent-tools switch, and at least one
    /// enabled server all agree that MCP should be live.
    public var isActive: Bool {
        Self.isEnabledRequested && Self.agentToolsEnabled
    }

    // MARK: Bootstrap

    /// Called once at app launch. Starts the config watcher and reconciles.
    public func bootstrap() {
        installTerminationObserver()
        startWatching()
        reloadFromDisk(reason: "launch")
    }

    private func installTerminationObserver() {
        guard terminationObserver == nil else { return }
        terminationObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.willTerminateNotification,
            object: nil,
            queue: .main
        ) { _ in
            // Runs synchronously on the main thread just before exit, so the
            // child processes are signalled while the app still exists.
            MainActor.assumeIsolated {
                MCPManager.shared.emergencyShutdown()
            }
        }
    }

    // MARK: Config IO

    public func reloadFromDisk(reason: String) {
        configPath = MCPConfigStore.defaultPathDescription
        configToken = MCPConfigStore.modificationToken()
        do {
            let loaded = try MCPConfigStore.loadOrSeed()
            document = loaded
            configIssues = MCPConfigValidator.validate(loaded)
            lastError = nil
            lastReloadSummary = Self.summary(for: loaded, reason: reason)
            DiagLog.info(
                "mcp",
                "config loaded (\(reason)) servers=\(loaded.mcpServers.count) enabled=\(loaded.enabledServerCount) issues=\(configIssues.count)"
            )
            scheduleApply()
        } catch {
            lastError = error.localizedDescription
            DiagLog.error("mcp", "config load failed: \(error.localizedDescription)")
            publishStates()
        }
    }

    private static func summary(for document: MCPConfigDocument, reason: String) -> String {
        let count = document.mcpServers.count
        if count == 0 {
            return "No MCP servers configured."
        }
        return "Loaded \(count) server\(count == 1 ? "" : "s") (\(document.enabledServerCount) enabled) from disk."
    }

    @discardableResult
    public func saveDocument(_ updated: MCPConfigDocument) -> Bool {
        let issues = MCPConfigValidator.validate(updated)
        guard !MCPConfigValidator.hasErrors(issues) else {
            configIssues = issues
            lastError = issues
                .filter { $0.severity == .error }
                .map(\.displayMessage)
                .joined(separator: "\n")
            return false
        }

        do {
            try MCPConfigStore.save(updated)
        } catch {
            lastError = error.localizedDescription
            DiagLog.error("mcp", "config save failed: \(error.localizedDescription)")
            return false
        }

        document = updated
        configIssues = issues
        configPath = MCPConfigStore.defaultPathDescription
        configToken = MCPConfigStore.modificationToken()
        lastError = nil
        scheduleApply()
        return true
    }

    @discardableResult
    public func saveRawJSON(_ text: String) -> Bool {
        do {
            let parsed = try MCPConfigStore.saveRawJSON(text)
            let issues = MCPConfigValidator.validate(parsed)
            document = parsed
            configIssues = issues
            configPath = MCPConfigStore.defaultPathDescription
            configToken = MCPConfigStore.modificationToken()
            lastError = nil
            lastReloadSummary = Self.summary(for: parsed, reason: "edited")
            scheduleApply()
            return true
        } catch {
            lastError = error.localizedDescription
            return false
        }
    }

    public func rawJSONText() -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        guard let data = try? encoder.encode(document),
              let string = String(data: data, encoding: .utf8) else {
            return MCPConfigStore.templateJSON
        }
        return string + "\n"
    }

    @discardableResult
    public func upsertServer(key: String, config: MCPServerConfig) -> Bool {
        var updated = document
        updated[key] = config
        return saveDocument(updated)
    }

    @discardableResult
    public func removeServer(key: String) -> Bool {
        guard document.mcpServers[key] != nil else { return true }
        var updated = document
        updated[key] = nil
        guard saveDocument(updated) else { return false }

        // An explicit removal also forgets the server's credentials; a
        // hand-edited file does not, so a transient edit cannot lose a key.
        MCPSecretStore.deleteAll(serverKey: key)
        runtime[key] = ServerRuntime(status: .disabled)
        disconnectSynchronously(key: key, reason: "removed")
        publishStates()
        return true
    }

    @discardableResult
    public func setEnabled(_ enabled: Bool, for key: String) -> Bool {
        guard var config = document.mcpServers[key] else { return false }
        guard config.enabled != enabled else { return true }
        config.enabled = enabled
        var updated = document
        updated[key] = config
        guard saveDocument(updated) else { return false }
        if !enabled {
            runtime[key] = ServerRuntime(status: .disabled)
            publishStates()
        }
        return true
    }

    public func openConfigInEditor() {
        do {
            _ = try MCPConfigStore.loadOrSeed()
        } catch {
            lastError = error.localizedDescription
            return
        }
        guard let url = MCPConfigStore.defaultURL else { return }
        NSWorkspace.shared.open(url)
    }

    public func revealConfigInFinder() {
        guard let url = MCPConfigStore.defaultURL else { return }
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    // MARK: Reconcile

    private func scheduleApply() {
        if applyInFlight {
            applyRequested = true
            return
        }
        Task { await apply() }
    }

    /// Reconciles live connections with `document`. This is the only place a
    /// server is connected or disconnected, so the file is always the truth.
    public func apply() async {
        applyInFlight = true
        isApplying = true
        defer {
            applyInFlight = false
            isApplying = false
            if applyRequested {
                applyRequested = false
                Task { await apply() }
            }
        }

        guard isActive else {
            for key in connections.keys { disconnectSynchronously(key: key, reason: "MCP disabled") }
            connecting.removeAll()
            for task in connectTasks.values { task.cancel() }
            connectTasks.removeAll()
            for key in document.sortedServerKeys {
                runtime[key] = ServerRuntime(status: document.mcpServers[key]?.enabled == true ? .stopped : .disabled)
            }
            publishStates()
            return
        }

        let workspacePath = AgentHarness.shared.defaultWorkingDirectory?.path

        // Tear down anything removed, disabled, or reconfigured.
        for (key, connection) in connections {
            guard let config = document.mcpServers[key], config.enabled else {
                disconnectSynchronously(key: key, reason: "removed or disabled")
                continue
            }
            if connection.config != config {
                disconnectSynchronously(key: key, reason: "configuration changed")
            }
        }

        // Connect everything enabled that is not already live.
        for key in document.sortedServerKeys {
            guard let config = document.mcpServers[key] else { continue }
            guard config.enabled else {
                runtime[key] = ServerRuntime(status: .disabled)
                continue
            }
            guard connections[key] == nil, !connecting.contains(key), connectTasks[key] == nil else { continue }
            connecting.insert(key)
            runtime[key] = ServerRuntime(status: .connecting)
            connectTasks[key] = Task { [weak self] in
                await self?.connect(key: key, config: config, workspacePath: workspacePath)
            }
        }

        publishStates()
    }

    public func restart(key: String) {
        disconnectSynchronously(key: key, reason: "manual restart")
        runtime[key] = ServerRuntime(status: .stopped)
        publishStates()
        scheduleApply()
    }

    public func restartAll() {
        for key in connections.keys {
            disconnectSynchronously(key: key, reason: "manual restart")
        }
        publishStates()
        scheduleApply()
    }

    // MARK: Connect

    private func connect(key: String, config: MCPServerConfig, workspacePath: String?) async {
        defer {
            connecting.remove(key)
            connectTasks[key] = nil
        }

        let (resolved, issues) = MCPConfigResolver.resolve(key: key, config: config, workspacePath: workspacePath)
        guard let resolved else {
            let message = issues
                .filter { $0.severity == .error }
                .map(\.message)
                .joined(separator: " ")
            runtime[key] = ServerRuntime(
                status: .failed,
                error: message.isEmpty ? "The server configuration is invalid." : message
            )
            DiagLog.error("mcp", "server=\(key) not started: \(message)")
            publishStates()
            return
        }

        do {
            // The callback runs on the client's actor, so it hops to the main
            // actor and reaches the singleton instead of capturing self (which
            // would form a retain cycle: manager -> connection -> client ->
            // closure -> manager).
            let client = try MCPClient(config: resolved) { [key] in
                Task { @MainActor in
                    MCPManager.shared.handleToolsListChanged(serverKey: key)
                }
            }

            try await client.start()
            let descriptors = try await client.listTools()

            // The config may have changed while we were handshaking.
            guard isActive,
                  document.mcpServers[key]?.enabled == true,
                  document.mcpServers[key] == config else {
                await client.stop()
                runtime[key] = ServerRuntime(status: .stopped)
                publishStates()
                return
            }

            let visible = Array(descriptors.prefix(MCPConfigValidator.maxToolsPerServer))
            let bridges = visible.map { MCPToolBridge(serverKey: key, descriptor: $0, client: client) }

            AgentHarness.shared.unregisterTools(prefix: Self.toolPrefix(for: key))
            AgentHarness.shared.registerTools(bridges)

            let instructions = await client.serverInstructions
            connections[key] = MCPConnection(client: client, config: config, bridges: bridges)
            runtime[key] = ServerRuntime(
                status: .ready,
                error: descriptors.count > visible.count
                    ? "Only the first \(visible.count) of \(descriptors.count) tools are exposed; the rest exceed the tool cap."
                    : nil,
                instructions: instructions,
                qualifiedToolNames: bridges.map { $0.definition.function.name },
                remoteToolNames: visible.map(\.name)
            )
            DiagLog.info("mcp", "server=\(key) ready with \(bridges.count) tools")
        } catch {
            AgentHarness.shared.unregisterTools(prefix: Self.toolPrefix(for: key))
            // A cancelled connect is a deliberate teardown (disable, remove,
            // config change, app quit), not a failure: reporting it as one
            // would leave the row showing "Failed" after the user turned it off.
            if Task.isCancelled || !isActive || document.mcpServers[key]?.enabled != true || document.mcpServers[key] != config {
                DiagLog.info("mcp", "server=\(key) connect abandoned: \(error.localizedDescription)")
            } else {
                runtime[key] = ServerRuntime(status: .failed, error: error.localizedDescription)
                DiagLog.error("mcp", "server=\(key) connect failed: \(error.localizedDescription)")
            }
        }

        publishStates()
    }

    private func handleToolsListChanged(serverKey: String) {
        guard let connection = connections[serverKey] else { return }
        Task { [weak self] in
            guard let self else { return }
            do {
                let descriptors = try await connection.client.listTools()
                guard self.connections[serverKey] != nil else { return }
                let visible = Array(descriptors.prefix(MCPConfigValidator.maxToolsPerServer))
                let bridges = visible.map {
                    MCPToolBridge(serverKey: serverKey, descriptor: $0, client: connection.client)
                }
                AgentHarness.shared.unregisterTools(prefix: Self.toolPrefix(for: serverKey))
                AgentHarness.shared.registerTools(bridges)
                self.connections[serverKey] = MCPConnection(
                    client: connection.client,
                    config: connection.config,
                    bridges: bridges
                )
                var state = self.runtime[serverKey] ?? ServerRuntime(status: .ready)
                state.status = .ready
                state.qualifiedToolNames = bridges.map { $0.definition.function.name }
                state.remoteToolNames = visible.map(\.name)
                self.runtime[serverKey] = state
                self.publishStates()
                DiagLog.info("mcp", "server=\(serverKey) tool list refreshed: \(bridges.count) tools")
            } catch {
                DiagLog.warning("mcp", "server=\(serverKey) tool refresh failed: \(error.localizedDescription)")
            }
        }
    }

    // MARK: Disconnect

    private static func toolPrefix(for key: String) -> String {
        MCPToolBridge.namePrefix + key + MCPToolBridge.separator
    }

    /// Unregisters the server's tools and closes the transport without awaiting
    /// the connection. Tools are removed from the harness *before* the process
    /// is signalled, so a generation that is mid-flight stops being offered a
    /// tool whose server is going away.
    private func disconnectSynchronously(key: String, reason: String) {
        connectTasks[key]?.cancel()
        connectTasks[key] = nil
        connecting.remove(key)

        AgentHarness.shared.unregisterTools(prefix: Self.toolPrefix(for: key))

        guard let connection = connections.removeValue(forKey: key) else {
            if runtime[key]?.status == .ready || runtime[key]?.status == .connecting {
                runtime[key] = ServerRuntime(status: document.mcpServers[key]?.enabled == true ? .stopped : .disabled)
            }
            return
        }
        DiagLog.info("mcp", "server=\(key) disconnected (\(reason))")
        // Nonisolated and synchronous: closes stdin, then SIGTERM/SIGKILL.
        connection.client.forceCloseTransport()
        runtime[key] = ServerRuntime(status: document.mcpServers[key]?.enabled == true ? .stopped : .disabled)
    }

    // MARK: State Publishing

    private func publishStates() {
        var states: [MCPServerState] = []
        for key in document.sortedServerKeys {
            guard let config = document.mcpServers[key] else { continue }
            var state = runtime[key] ?? ServerRuntime(status: config.enabled ? .stopped : .disabled)
            if !config.enabled {
                state.status = .disabled
                state.error = nil
            }
            states.append(MCPServerState(
                key: key,
                enabled: config.enabled,
                status: state.status,
                error: state.error,
                endpoint: config.endpointDescription,
                transportLabel: config.transportKind == .http ? "http" : "stdio",
                instructions: state.instructions,
                qualifiedToolNames: state.qualifiedToolNames,
                remoteToolNames: state.remoteToolNames
            ))
        }
        servers = states
        discoveredToolCount = AgentHarness.shared.tools.keys.filter { $0.hasPrefix(MCPToolBridge.namePrefix) }.count
    }

    // MARK: Tool Loading

    /// Loads one discovered tool into the prompt/grammar, exactly like
    /// `tools_load` would. Used by the settings disclosure rows.
    @discardableResult
    public func loadTool(named qualifiedName: String) -> Bool {
        do {
            _ = try AgentHarness.shared.loadTool(named: qualifiedName)
            return true
        } catch {
            lastError = error.localizedDescription
            return false
        }
    }

    public func loadedToolNames(for serverKey: String) -> [String] {
        AgentHarness.shared.loadedTools.keys
            .filter { $0.hasPrefix(Self.toolPrefix(for: serverKey)) }
            .sorted()
    }

    // MARK: File Watching

    private func startWatching() {
        guard watcher == nil, let directory = MCPConfigStore.directoryURL else { return }
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let descriptor = open(directory.path, O_EVTONLY)
        guard descriptor >= 0 else {
            DiagLog.warning("mcp", "could not watch the config directory; external edits need a manual reload")
            return
        }
        watcherDescriptor = descriptor

        // Watching the directory rather than the file catches atomic writes,
        // which replace the inode and would silently detach a file watcher.
        let source = DispatchSource.makeFileSystemObjectSource(
            fileDescriptor: descriptor,
            eventMask: [.write, .rename, .delete],
            queue: .main
        )
        source.setEventHandler { [weak self] in
            MainActor.assumeIsolated {
                self?.handleDirectoryChange()
            }
        }
        source.setCancelHandler {
            close(descriptor)
        }
        source.resume()
        watcher = source
    }

    private func stopWatching() {
        watcher?.cancel()
        watcher = nil
        watcherDescriptor = -1
    }

    private func handleDirectoryChange() {
        let token = MCPConfigStore.modificationToken()
        guard token != configToken else { return }

        // Restarting a server mid-generation would kill an in-flight tool call,
        // so an edit that lands during a turn waits for the turn to end.
        if CompletionEngine.shared.isActive {
            pendingReload = true
            scheduleDeferredReload()
            return
        }
        reloadFromDisk(reason: "external edit")
    }

    private func scheduleDeferredReload() {
        guard deferredReloadTask == nil else { return }
        deferredReloadTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 2_000_000_000)
                guard let self else { return }
                guard self.pendingReload else { break }
                if CompletionEngine.shared.isActive { continue }
                self.pendingReload = false
                self.reloadFromDisk(reason: "external edit (deferred)")
                break
            }
            self?.deferredReloadTask = nil
        }
    }

    // MARK: Shutdown

    /// Synchronous teardown for app termination: closes every transport so no
    /// MCP server outlives DynaMoE.
    public func emergencyShutdown() {
        stopWatching()
        deferredReloadTask?.cancel()
        deferredReloadTask = nil
        for task in connectTasks.values { task.cancel() }
        connectTasks.removeAll()
        for key in connections.keys {
            disconnectSynchronously(key: key, reason: "app terminating")
        }
        for key in document.sortedServerKeys {
            AgentHarness.shared.unregisterTools(prefix: Self.toolPrefix(for: key))
        }
        if let terminationObserver {
            NotificationCenter.default.removeObserver(terminationObserver)
            self.terminationObserver = nil
        }
        publishStates()
    }

    /// Async teardown used by tests and by the settings "Stop all" action.
    public func shutdown() async {
        stopWatching()
        deferredReloadTask?.cancel()
        deferredReloadTask = nil
        for task in connectTasks.values { task.cancel() }
        connectTasks.removeAll()
        let clients = connections.values.map(\.client)
        for key in connections.keys {
            AgentHarness.shared.unregisterTools(prefix: Self.toolPrefix(for: key))
        }
        connections.removeAll()
        runtime.removeAll()
        for client in clients {
            await client.stop()
        }
        publishStates()
    }
}
