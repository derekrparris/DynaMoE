//
//  MCPConfig.swift
//  DynaMoE
//
//  Phase 7: Model Context Protocol client.
//  Config-file model, on-disk store, validation, and ${...} token expansion.
//
//  The file at ~/Library/Application Support/DynaMoE/mcp.json is the single
//  source of truth for which MCP servers exist. Settings renders a projection
//  of it and writes back through MCPConfigStore, so a hand-edited file, a
//  version-controlled dotfile, and the in-app form can never disagree.
//

import Foundation

// MARK: - Server Configuration

/// One entry under `mcpServers`. Transport is inferred rather than declared:
/// a `command` means stdio, a `url` means Streamable HTTP. Exactly one of the
/// two must be present (see `MCPConfigValidator`).
public nonisolated struct MCPServerConfig: Codable, Equatable, Sendable {
    public var enabled: Bool
    public var command: String?
    public var args: [String]?
    public var env: [String: String]?
    public var cwd: String?
    public var url: String?
    public var headers: [String: String]?
    public var startupTimeoutSec: Double?
    public var toolTimeoutSec: Double?

    public static let defaultStartupTimeoutSec: Double = 30
    public static let defaultToolTimeoutSec: Double = 120

    public init(
        enabled: Bool = true,
        command: String? = nil,
        args: [String]? = nil,
        env: [String: String]? = nil,
        cwd: String? = nil,
        url: String? = nil,
        headers: [String: String]? = nil,
        startupTimeoutSec: Double? = nil,
        toolTimeoutSec: Double? = nil
    ) {
        self.enabled = enabled
        self.command = command
        self.args = args
        self.env = env
        self.cwd = cwd
        self.url = url
        self.headers = headers
        self.startupTimeoutSec = startupTimeoutSec
        self.toolTimeoutSec = toolTimeoutSec
    }

    // MARK: Transport

    public enum TransportKind: String, Sendable {
        case stdio
        case http
    }

    /// `nil` when the entry declares neither or both endpoints. Validation
    /// reports that; the runtime simply refuses to connect.
    public var transportKind: TransportKind? {
        let hasCommand = !(command ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        let hasURL = !(url ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        if hasCommand == hasURL { return nil }
        return hasCommand ? .stdio : .http
    }

    public var resolvedStartupTimeoutSec: Double {
        guard let value = startupTimeoutSec, value > 0 else { return Self.defaultStartupTimeoutSec }
        return value
    }

    public var resolvedToolTimeoutSec: Double {
        guard let value = toolTimeoutSec, value > 0 else { return Self.defaultToolTimeoutSec }
        return value
    }

    /// Human-readable endpoint for the settings row.
    public var endpointDescription: String {
        switch transportKind {
        case .stdio:
            let parts = ([command ?? ""] + (args ?? [])).filter { !$0.isEmpty }
            return parts.joined(separator: " ")
        case .http:
            return url ?? ""
        case nil:
            return "No endpoint configured"
        }
    }

    // MARK: Codable

    private enum CodingKeys: String, CodingKey {
        case enabled, command, args, env, cwd, url, headers
        case serverUrl
        case startupTimeoutSec, toolTimeoutSec
        case startupTimeout, toolTimeout
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        // Absent `enabled` means "on": a config pasted from another client,
        // which never carries the field, should just work.
        self.enabled = try container.decodeIfPresent(Bool.self, forKey: .enabled) ?? true
        self.command = try container.decodeIfPresent(String.self, forKey: .command)
        self.args = try container.decodeIfPresent([String].self, forKey: .args)
        self.env = try container.decodeIfPresent([String: String].self, forKey: .env)
        self.cwd = try container.decodeIfPresent(String.self, forKey: .cwd)
        // Both spellings are accepted so configs pasted from other clients
        // decode; `url` wins when both are present.
        self.url = try container.decodeIfPresent(String.self, forKey: .url)
            ?? container.decodeIfPresent(String.self, forKey: .serverUrl)
        self.headers = try container.decodeIfPresent([String: String].self, forKey: .headers)
        self.startupTimeoutSec = try container.decodeIfPresent(Double.self, forKey: .startupTimeoutSec)
            ?? container.decodeIfPresent(Double.self, forKey: .startupTimeout)
        self.toolTimeoutSec = try container.decodeIfPresent(Double.self, forKey: .toolTimeoutSec)
            ?? container.decodeIfPresent(Double.self, forKey: .toolTimeout)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(enabled, forKey: .enabled)
        try container.encodeIfPresent(command, forKey: .command)
        try container.encodeIfPresent(args, forKey: .args)
        try container.encodeIfPresent(env, forKey: .env)
        try container.encodeIfPresent(cwd, forKey: .cwd)
        try container.encodeIfPresent(url, forKey: .url)
        try container.encodeIfPresent(headers, forKey: .headers)
        try container.encodeIfPresent(startupTimeoutSec, forKey: .startupTimeoutSec)
        try container.encodeIfPresent(toolTimeoutSec, forKey: .toolTimeoutSec)
    }
}

// MARK: - Document

public nonisolated struct MCPConfigDocument: Codable, Equatable, Sendable {
    public static let currentVersion = 1

    public var version: Int
    public var mcpServers: [String: MCPServerConfig]

    public init(version: Int = MCPConfigDocument.currentVersion, mcpServers: [String: MCPServerConfig] = [:]) {
        self.version = version
        self.mcpServers = mcpServers
    }

    private enum CodingKeys: String, CodingKey {
        case version, mcpServers
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.version = try container.decodeIfPresent(Int.self, forKey: .version) ?? Self.currentVersion
        self.mcpServers = try container.decodeIfPresent([String: MCPServerConfig].self, forKey: .mcpServers) ?? [:]
    }

    public var sortedServerKeys: [String] {
        mcpServers.keys.sorted()
    }

    public var enabledServerCount: Int {
        mcpServers.values.filter { $0.enabled }.count
    }

    public subscript(serverKey: String) -> MCPServerConfig? {
        get { mcpServers[serverKey] }
        set {
            if let newValue {
                mcpServers[serverKey] = newValue
            } else {
                mcpServers.removeValue(forKey: serverKey)
            }
        }
    }
}

// MARK: - Errors

public nonisolated enum MCPConfigError: LocalizedError {
    case applicationSupportUnavailable(String)
    case unreadable(path: String, reason: String)
    case malformed(path: String, reason: String)
    case unwritable(path: String, reason: String)

    public var errorDescription: String? {
        switch self {
        case .applicationSupportUnavailable(let reason):
            return "Application Support is unavailable, so MCP configuration cannot be stored: \(reason)"
        case .unreadable(let path, let reason):
            return "Could not read \(path): \(reason)"
        case .malformed(let path, let reason):
            return "\(path) is not valid MCP configuration: \(reason)"
        case .unwritable(let path, let reason):
            return "Could not write \(path): \(reason)"
        }
    }
}

// MARK: - On-Disk Store

public nonisolated enum MCPConfigStore {
    public static let fileName = "mcp.json"

    /// `~/Library/Application Support/DynaMoE`, the same tree that holds
    /// `sessions/` and `logs/`. Returns nil only when Application Support
    /// itself cannot be resolved.
    public static var directoryURL: URL? {
        guard let base = try? FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        ) else { return nil }
        return base.appendingPathComponent("DynaMoE", isDirectory: true)
    }

    public static var defaultURL: URL? {
        directoryURL?.appendingPathComponent(fileName, isDirectory: false)
    }

    public static var defaultPathDescription: String {
        defaultURL?.path ?? "~/Library/Application Support/DynaMoE/\(fileName)"
    }

    /// Empty-but-valid template written on first launch so the file can be
    /// opened, edited, and version-controlled before any server exists.
    public static let templateJSON = """
    {
      "version": 1,
      "mcpServers": {}
    }

    """

    public static func load(from url: URL) throws -> MCPConfigDocument {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw MCPConfigError.unreadable(path: url.path, reason: error.localizedDescription)
        }
        do {
            return try JSONDecoder().decode(MCPConfigDocument.self, from: data)
        } catch {
            throw MCPConfigError.malformed(path: url.path, reason: error.localizedDescription)
        }
    }

    /// Reads the file, creating the parent directory and an empty template
    /// first if it does not exist yet.
    public static func loadOrSeed(at url: URL? = nil) throws -> MCPConfigDocument {
        guard let target = url ?? defaultURL else {
            throw MCPConfigError.applicationSupportUnavailable("the directory could not be resolved")
        }
        if !FileManager.default.fileExists(atPath: target.path) {
            try writeTemplate(to: target)
        }
        return try load(from: target)
    }

    public static func save(_ document: MCPConfigDocument, to url: URL? = nil) throws {
        guard let target = url ?? defaultURL else {
            throw MCPConfigError.applicationSupportUnavailable("the directory could not be resolved")
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data: Data
        do {
            data = try encoder.encode(document)
        } catch {
            throw MCPConfigError.unwritable(path: target.path, reason: error.localizedDescription)
        }
        var payload = data
        payload.append(0x0A)
        try writeAtomically(payload, to: target)
    }

    /// Saves the raw text the user typed in the JSON editor, after validating
    /// that it decodes. Returns the decoded document so the caller can apply
    /// it without re-reading the file.
    public static func saveRawJSON(_ text: String, to url: URL? = nil) throws -> MCPConfigDocument {
        guard let data = text.data(using: .utf8) else {
            throw MCPConfigError.malformed(path: url?.path ?? defaultPathDescription, reason: "text is not valid UTF-8")
        }
        let document: MCPConfigDocument
        do {
            document = try JSONDecoder().decode(MCPConfigDocument.self, from: data)
        } catch {
            throw MCPConfigError.malformed(path: url?.path ?? defaultPathDescription, reason: error.localizedDescription)
        }
        try save(document, to: url)
        return document
    }

    /// Pretty-prints arbitrary JSON text. Returns nil when the text does not
    /// parse, so the editor can leave the buffer untouched.
    public static func prettyPrinted(_ text: String) -> String? {
        guard let data = text.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data),
              let pretty = try? JSONSerialization.data(
                withJSONObject: object,
                options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
              ),
              let string = String(data: pretty, encoding: .utf8) else { return nil }
        return string + "\n"
    }

    /// Cheap change detector for the directory watcher: modification date plus
    /// size. Two saves inside the same timestamp tick still differ in size in
    /// every realistic edit, and the watcher only uses this to skip no-op work.
    public static func modificationToken(at url: URL? = nil) -> String? {
        guard let target = url ?? defaultURL,
              let attributes = try? FileManager.default.attributesOfItem(atPath: target.path) else { return nil }
        let modified = (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        let size = (attributes[.size] as? NSNumber)?.intValue ?? 0
        return "\(modified)-\(size)"
    }

    private static func writeTemplate(to target: URL) throws {
        try writeAtomically(Data(templateJSON.utf8), to: target)
    }

    private static func writeAtomically(_ data: Data, to target: URL) throws {
        let directory = target.deletingLastPathComponent()
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        } catch {
            throw MCPConfigError.unwritable(path: directory.path, reason: error.localizedDescription)
        }
        do {
            try data.write(to: target, options: .atomic)
        } catch {
            throw MCPConfigError.unwritable(path: target.path, reason: error.localizedDescription)
        }
    }
}

// MARK: - Validation

public nonisolated enum MCPConfigIssueSeverity: Int, Sendable, Comparable {
    case warning
    case error

    public static func < (lhs: MCPConfigIssueSeverity, rhs: MCPConfigIssueSeverity) -> Bool {
        lhs.rawValue < rhs.rawValue
    }

    public var label: String {
        switch self {
        case .warning: return "Warning"
        case .error: return "Error"
        }
    }
}

public nonisolated struct MCPConfigIssue: Identifiable, Sendable, Hashable {
    public let id: String
    public let severity: MCPConfigIssueSeverity
    public let serverKey: String?
    public let message: String

    public init(severity: MCPConfigIssueSeverity, serverKey: String? = nil, message: String) {
        self.id = "\(severity.rawValue)|\(serverKey ?? "")|\(message)"
        self.severity = severity
        self.serverKey = serverKey
        self.message = message
    }

    public var displayMessage: String {
        if let serverKey, !serverKey.isEmpty {
            return "\(serverKey): \(message)"
        }
        return message
    }
}

public nonisolated enum MCPConfigValidator {
    /// Hard ceilings. Each registered tool name costs a trie insert per
    /// `syncGrammar()` and a linear scan per name-emitting token, so the
    /// grammar sampler needs a bound rather than an open-ended tool count.
    public static let maxServers = 32
    public static let maxToolsPerServer = 120
    public static let maxTotalTools = 120

    /// Server keys become part of `mcp__<key>__<tool>`, so `__` inside a key
    /// would make that name ambiguous to split.
    public static let allowedKeyCharacters = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_"))

    public static func validate(_ document: MCPConfigDocument) -> [MCPConfigIssue] {
        var issues: [MCPConfigIssue] = []

        if document.mcpServers.count > maxServers {
            issues.append(MCPConfigIssue(
                severity: .error,
                message: "\(document.mcpServers.count) servers configured; the limit is \(maxServers)."
            ))
        }
        if document.version > MCPConfigDocument.currentVersion {
            issues.append(MCPConfigIssue(
                severity: .warning,
                message: "File declares version \(document.version); this build understands \(MCPConfigDocument.currentVersion)."
            ))
        }

        for key in document.sortedServerKeys {
            guard let server = document.mcpServers[key] else { continue }
            issues.append(contentsOf: validate(server: server, key: key))
        }
        return issues
    }

    public static func validate(server: MCPServerConfig, key: String) -> [MCPConfigIssue] {
        var issues: [MCPConfigIssue] = []

        func error(_ message: String) {
            issues.append(MCPConfigIssue(severity: .error, serverKey: key, message: message))
        }
        func warning(_ message: String) {
            issues.append(MCPConfigIssue(severity: .warning, serverKey: key, message: message))
        }

        if key.isEmpty {
            error("Server name is empty.")
        } else {
            if key.rangeOfCharacter(from: allowedKeyCharacters.inverted) != nil {
                error("Server name may only contain letters, digits, '-' and '_' (got '\(key)').")
            }
            if key.contains("__") {
                error("Server name cannot contain '__'; tool names are built as mcp__<server>__<tool>.")
            }
            if key.hasPrefix("mcp__") {
                error("Server name cannot start with 'mcp__'.")
            }
        }

        let hasCommand = !(server.command ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        let hasURL = !(server.url ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty

        if hasCommand && hasURL {
            error("Declares both 'command' (stdio) and 'url' (http); keep exactly one.")
        } else if !hasCommand && !hasURL {
            error("Needs either 'command' (stdio) or 'url' (http).")
        }

        if hasURL, let raw = server.url?.trimmingCharacters(in: .whitespacesAndNewlines) {
            if let parsed = URL(string: raw), let scheme = parsed.scheme?.lowercased(), parsed.host != nil {
                if scheme != "http" && scheme != "https" {
                    error("Unsupported URL scheme '\(scheme)'; use https (or http for a loopback server).")
                } else if scheme == "http" && !isLoopbackHost(parsed.host ?? "") {
                    warning("Plain http to a non-loopback host sends credentials in the clear; prefer https.")
                }
            } else {
                error("'url' is not a valid absolute URL.")
            }
        }

        if (server.startupTimeoutSec ?? 1) <= 0 {
            error("'startupTimeoutSec' must be greater than 0.")
        }
        if (server.toolTimeoutSec ?? 1) <= 0 {
            error("'toolTimeoutSec' must be greater than 0.")
        }

        // Names, not just values. A header name containing a space or comma
        // makes the whole HTTP request unparseable, and the server answers with
        // an opaque 400 before the token is ever looked at.
        for name in (server.env ?? [:]).keys {
            if !isValidEnvironmentName(name) {
                error("Environment variable name '\(name)' is invalid: use letters, digits and '_', and do not start with a digit. Add one row per variable.")
            }
        }
        for (name, value) in server.headers ?? [:] {
            if !isValidHeaderName(name) {
                error("Header name '\(name)' is invalid: HTTP header names cannot contain spaces, commas or colons. Add one row per header (for example 'Authorization').")
            }
            // `contains("\n")` is Character-based, and Swift treats "\r\n" as a
            // single grapheme cluster, so it would never match CRLF (the exact
            // header-injection shape). Scan characters for a newline instead.
            if value.contains(where: { $0.isNewline }) {
                error("Header '\(name)' contains a line break, which is not allowed.")
            }
        }

        let stringsToScan: [String] = {
            var values: [String] = []
            values.append(contentsOf: server.args ?? [])
            values.append(contentsOf: (server.env ?? [:]).values)
            values.append(contentsOf: (server.headers ?? [:]).values)
            if let cwd = server.cwd { values.append(cwd) }
            return values
        }()
        for value in stringsToScan {
            let resolution = MCPTokenExpander.expand(value, workspacePath: nil)
            for token in resolution.unresolved {
                warning("Unresolved token \(token).")
            }
        }

        return issues
    }

    /// `NAME` or `_NAME`: letters, digits and underscores, not starting with a digit.
    public static func isValidEnvironmentName(_ name: String) -> Bool {
        guard let first = name.unicodeScalars.first,
              CharacterSet.letters.union(CharacterSet(charactersIn: "_")).contains(first) else { return false }
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "_"))
        return name.unicodeScalars.allSatisfy { allowed.contains($0) }
    }

    /// RFC 7230 `token`: what an HTTP header field name may contain.
    public static func isValidHeaderName(_ name: String) -> Bool {
        guard !name.isEmpty else { return false }
        let allowed = CharacterSet(charactersIn: "!#$%&'*+-.^_`|~").union(.alphanumerics)
        return name.unicodeScalars.allSatisfy { allowed.contains($0) }
    }

    public static func isLoopbackHost(_ host: String) -> Bool {
        let lowered = host.lowercased()
        return lowered == "localhost" || lowered == "127.0.0.1" || lowered == "::1" || lowered.hasSuffix(".localhost")
    }

    public static func hasErrors(_ issues: [MCPConfigIssue]) -> Bool {
        issues.contains { $0.severity == .error }
    }
}

// MARK: - Token Expansion

/// Resolves `${env:NAME}`, `${keychain:ACCOUNT}`, and `${workspace}` at connect
/// time. Expansion is deliberately not applied when the file is read: the raw
/// file must round-trip byte-faithfully through the settings editor, and a
/// secret token must never be replaced by its plaintext on disk.
public nonisolated enum MCPTokenExpander {
    public struct Resolution: Sendable, Equatable {
        public let value: String
        public let unresolved: [String]
    }

    public static func keychainToken(account: String) -> String {
        "${keychain:\(account)}"
    }

    public static func envToken(name: String) -> String {
        "${env:\(name)}"
    }

    public static func containsToken(_ raw: String) -> Bool {
        raw.contains("${")
    }

    /// True when the value is nothing but a single keychain token, i.e. the
    /// field is a pure secret reference the settings form can render as a
    /// secure field instead of a plaintext editor.
    public static func isPureKeychainToken(_ raw: String) -> Bool {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("${keychain:"), trimmed.hasSuffix("}") else { return false }
        return !trimmed.dropFirst("${keychain:".count).dropLast().contains("}")
    }

    public static func keychainAccount(in raw: String) -> String? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard isPureKeychainToken(trimmed) else { return nil }
        let inner = trimmed.dropFirst("${keychain:".count).dropLast()
        return inner.isEmpty ? nil : String(inner)
    }

    public static func expand(_ raw: String, workspacePath: String?) -> Resolution {
        guard raw.contains("${") else { return Resolution(value: raw, unresolved: []) }

        var output = ""
        var unresolved: [String] = []
        var index = raw.startIndex

        while index < raw.endIndex {
            guard let open = raw.range(of: "${", range: index..<raw.endIndex) else {
                output += raw[index..<raw.endIndex]
                break
            }
            output += raw[index..<open.lowerBound]
            guard let close = raw.range(of: "}", range: open.upperBound..<raw.endIndex) else {
                output += raw[open.lowerBound..<raw.endIndex]
                break
            }
            let token = String(raw[open.lowerBound..<close.upperBound])
            let body = String(raw[open.upperBound..<close.lowerBound])

            if body == "workspace" {
                output += workspacePath ?? NSHomeDirectory()
            } else if body.hasPrefix("env:") {
                let name = String(body.dropFirst("env:".count))
                if !name.isEmpty, let value = ProcessInfo.processInfo.environment[name], !value.isEmpty {
                    output += value
                } else {
                    unresolved.append(token)
                }
            } else if body.hasPrefix("keychain:") {
                let account = String(body.dropFirst("keychain:".count))
                if let secret = MCPSecretStore.get(account: account), !secret.isEmpty {
                    output += secret
                } else {
                    unresolved.append(token)
                }
            } else {
                unresolved.append(token)
                output += token
            }
            index = close.upperBound
        }

        return Resolution(value: output, unresolved: unresolved)
    }

    public static func expand(
        _ values: [String: String],
        workspacePath: String?
    ) -> (values: [String: String], unresolved: [String]) {
        var expanded: [String: String] = [:]
        var unresolved: [String] = []
        for (key, raw) in values {
            let resolution = expand(raw, workspacePath: workspacePath)
            expanded[key] = resolution.value
            unresolved.append(contentsOf: resolution.unresolved)
        }
        return (expanded, unresolved)
    }

    public static func expand(
        _ values: [String],
        workspacePath: String?
    ) -> (values: [String], unresolved: [String]) {
        var expanded: [String] = []
        var unresolved: [String] = []
        for raw in values {
            let resolution = expand(raw, workspacePath: workspacePath)
            expanded.append(resolution.value)
            unresolved.append(contentsOf: resolution.unresolved)
        }
        return (expanded, unresolved)
    }
}
