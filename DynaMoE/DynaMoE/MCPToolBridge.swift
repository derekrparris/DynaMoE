//
//  MCPToolBridge.swift
//  DynaMoE
//
//  Phase 7: Model Context Protocol client.
//  Adapts one discovered MCP tool to the in-app `AgentTool` contract.
//
//  This is the whole integration seam: `AgentHarness.tools` is a dictionary of
//  protocol existentials dispatched by name, so an MCP tool becomes a first-class
//  agent tool by conforming here. The agent loop, HITL approval, tool timeline,
//  subagent tool whitelist, grammar sampler, and history rendering all keep
//  working untouched.
//

import Foundation

public final class MCPToolBridge: AgentTool {
    /// `mcp__<server>__<tool>`, the Claude Code convention. Server keys may not
    /// contain `__` or end with `_`, so the first separator always terminates
    /// the server key and no two servers can share a tool-name prefix.
    public static let namePrefix = "mcp__"
    public static let separator = "__"

    /// Servers sometimes ship multi-kilobyte descriptions. Every loaded schema
    /// is paid for on every turn, so descriptions are bounded.
    public static let maxDescriptionCharacters = 1200

    public let serverKey: String
    public let remoteToolName: String
    public let readOnlyHint: Bool
    public let destructiveHint: Bool
    public let definition: ToolDefinition

    private let client: MCPClient

    public static func qualifiedName(serverKey: String, toolName: String) -> String {
        namePrefix + serverKey + separator + toolName
    }

    public static func parse(qualifiedName name: String) -> (serverKey: String, toolName: String)? {
        guard name.hasPrefix(namePrefix) else { return nil }
        let remainder = name.dropFirst(namePrefix.count)
        guard let separatorRange = remainder.range(of: separator) else { return nil }
        let serverKey = String(remainder[..<separatorRange.lowerBound])
        let toolName = String(remainder[separatorRange.upperBound...])
        guard !serverKey.isEmpty, !toolName.isEmpty else { return nil }
        return (serverKey, toolName)
    }

    public init(serverKey: String, descriptor: MCPToolDescriptor, client: MCPClient) {
        self.serverKey = serverKey
        self.remoteToolName = descriptor.name
        self.readOnlyHint = descriptor.readOnlyHint
        self.destructiveHint = descriptor.destructiveHint
        self.client = client

        let qualified = Self.qualifiedName(serverKey: serverKey, toolName: descriptor.name)
        self.definition = ToolDefinition(
            name: qualified,
            description: Self.composeDescription(serverKey: serverKey, descriptor: descriptor),
            parameters: Self.parameters(from: descriptor.inputSchema)
        )
    }

    // MARK: Description & Schema

    private static func composeDescription(serverKey: String, descriptor: MCPToolDescriptor) -> String {
        var body = descriptor.description.trimmingCharacters(in: .whitespacesAndNewlines)
        if body.isEmpty {
            body = "Remote tool '\(descriptor.name)' provided by MCP server '\(serverKey)'."
        }
        let composed: String
        if descriptor.destructiveHint {
            composed = "[MCP: \(serverKey)] [may modify state] \(body)"
        } else if descriptor.readOnlyHint {
            composed = "[MCP: \(serverKey)] [read-only] \(body)"
        } else {
            composed = "[MCP: \(serverKey)] \(body)"
        }
        guard composed.count > maxDescriptionCharacters else { return composed }
        return String(composed.prefix(maxDescriptionCharacters)) + "..."
    }

    /// Converts an arbitrary JSON Schema from the server into the
    /// `[String: AnyCodable]` shape `ToolDefinition` carries. `type` and
    /// `properties` are defaulted because the grammar sampler reads those keys
    /// to decide which parameter names are legal.
    public static func parameters(from schema: MCPJSONValue) -> [String: AnyCodable] {
        var object: [String: MCPJSONValue]
        if case .object(let raw) = schema {
            object = raw
        } else {
            object = [:]
        }
        if object["type"] == nil {
            object["type"] = .string("object")
        }
        if object["properties"] == nil {
            object["properties"] = .object([:])
        }

        var converted: [String: AnyCodable] = [:]
        for (key, value) in object {
            converted[key] = anyCodable(value)
        }
        return converted
    }

    private static func anyCodable(_ value: MCPJSONValue) -> AnyCodable {
        switch value {
        case .null:
            return AnyCodable(NSNull())
        case .bool(let wrapped):
            return AnyCodable(wrapped)
        case .int(let wrapped):
            return AnyCodable(wrapped)
        case .double(let wrapped):
            return AnyCodable(wrapped)
        case .string(let wrapped):
            return AnyCodable(wrapped)
        case .array(let values):
            return AnyCodable(values.map { anyCodable($0).value })
        case .object(let values):
            return AnyCodable(values.mapValues { anyCodable($0).value })
        }
    }

    // MARK: Execution

    public func execute(arguments: [String: Any], workingDirectory: URL?, maxOutputLength: Int) async throws -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        let startTime = CFAbsoluteTimeGetCurrent()
        let result = try await client.callTool(name: remoteToolName, arguments: arguments)
        let duration = (CFAbsoluteTimeGetCurrent() - startTime)
        let qualified = definition.function.name

        var text = result.text
        if text.isEmpty {
            text = "(the MCP server returned no text content)"
        }
        let clipped = Self.clip(text, to: maxOutputLength)

        if result.isError {
            // A non-nil stderr is what marks the record `.error` for the loop
            // guard, exactly like a failing built-in tool.
            let json = AgentHarness.toolErrorJSON(
                tool: qualified,
                error: clipped.text,
                extra: [
                    "server": serverKey,
                    "tool": remoteToolName,
                    "duration_seconds": Self.rounded(duration),
                    "content_blocks": result.contentBlockCount
                ]
            )
            return (json, nil, clipped.text, false)
        }

        var payload: [String: Any] = [
            "server": serverKey,
            "tool": remoteToolName,
            "duration_seconds": Self.rounded(duration),
            "content_blocks": result.contentBlockCount,
            "text": clipped.text
        ]
        if result.imageBlockCount > 0 {
            payload["image_blocks"] = result.imageBlockCount
            payload["images_note"] = "Binary image content is summarized as a placeholder; this model cannot see images."
        }
        if clipped.didClip {
            payload["truncated"] = true
            payload["original_characters"] = clipped.originalCount
            payload["truncation_note"] = "Result truncated to \(maxOutputLength) characters. Narrow the request (one item, a smaller range) to see the rest."
        }
        if result.contentBlockCount == 0,
           let structured = result.structuredContent,
           !structured.isNull {
            payload["structured_content"] = structured.anyValue
        }

        let json = AgentHarness.toolSuccessJSON(tool: qualified, data: payload)
        return (json, clipped.text, nil, false)
    }

    struct ClippedText {
        let text: String
        let didClip: Bool
        let originalCount: Int
    }

    static func clip(_ text: String, to limit: Int) -> ClippedText {
        guard limit > 0, text.count > limit else {
            return ClippedText(text: text, didClip: false, originalCount: text.count)
        }
        return ClippedText(
            text: String(text.prefix(limit)),
            didClip: true,
            originalCount: text.count
        )
    }

    private static func rounded(_ seconds: Double) -> Double {
        (seconds * 1000).rounded() / 1000
    }
}
