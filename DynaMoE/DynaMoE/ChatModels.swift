//
//  ChatModels.swift
//  DynaMoE
//

import Foundation

public enum MessageRole: String, Codable, Equatable {
    case user
    case assistant
    case system
}

public enum ToolExecutionStatus: String, Codable, Equatable {
    case running
    case success
    case error
}

public struct ToolCallRecord: Identifiable, Codable, Equatable {
    public var id: UUID
    public var name: String
    public var arguments: [String: String]
    public var rawArguments: String
    public var status: ToolExecutionStatus
    public var output: String?
    public var error: String?
    public var executionDurationSeconds: Double?
    public var timestamp: Date

    public init(
        id: UUID = UUID(),
        name: String,
        arguments: [String: String] = [:],
        rawArguments: String = "",
        status: ToolExecutionStatus = .running,
        output: String? = nil,
        error: String? = nil,
        executionDurationSeconds: Double? = nil,
        timestamp: Date = Date()
    ) {
        self.id = id
        self.name = name
        self.arguments = arguments
        self.rawArguments = rawArguments
        self.status = status
        self.output = output
        self.error = error
        self.executionDurationSeconds = executionDurationSeconds
        self.timestamp = timestamp
    }
}

public struct ChatMessage: Identifiable, Codable, Equatable {
    public var id: UUID
    public var role: MessageRole
    public var content: String
    public var thinkingContent: String?
    public var isThinking: Bool
    public var timestamp: Date
    public var tokenCount: Int
    public var tokensPerSec: Double
    public var timeToFirstTokenSeconds: Double?
    public var thinkingTimeSeconds: Double?
    public var prefillStatus: String?
    public var toolCalls: [ToolCallRecord]?
    public var jetSpecTau: Double?
    public var jetSpecDraftAccepted: Int?

    public init(
        id: UUID = UUID(),
        role: MessageRole,
        content: String,
        thinkingContent: String? = nil,
        isThinking: Bool = false,
        timestamp: Date = Date(),
        tokenCount: Int = 0,
        tokensPerSec: Double = 0.0,
        timeToFirstTokenSeconds: Double? = nil,
        thinkingTimeSeconds: Double? = nil,
        prefillStatus: String? = nil,
        toolCalls: [ToolCallRecord]? = nil,
        jetSpecTau: Double? = nil,
        jetSpecDraftAccepted: Int? = nil
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.thinkingContent = thinkingContent
        self.isThinking = isThinking
        self.timestamp = timestamp
        self.tokenCount = tokenCount
        self.tokensPerSec = tokensPerSec
        self.timeToFirstTokenSeconds = timeToFirstTokenSeconds
        self.thinkingTimeSeconds = thinkingTimeSeconds
        self.prefillStatus = prefillStatus
        self.toolCalls = toolCalls
        self.jetSpecTau = jetSpecTau
        self.jetSpecDraftAccepted = jetSpecDraftAccepted
    }
}

public struct QueuedPrompt: Identifiable, Codable, Equatable {
    public var id: UUID
    public var text: String
    public var timestamp: Date

    public init(
        id: UUID = UUID(),
        text: String,
        timestamp: Date = Date()
    ) {
        self.id = id
        self.text = text
        self.timestamp = timestamp
    }
}

public struct ChatSession: Identifiable, Codable, Equatable {
    public var id: UUID
    public var title: String
    public var messages: [ChatMessage]
    public var createdAt: Date
    public var updatedAt: Date
    public var selectedModelId: String?
    public var selectedModelName: String?
    public var selectedModelPath: String?
    public var isThinkingEnabled: Bool?
    public var isAgentToolsEnabled: Bool?
    public var queuedPrompts: [QueuedPrompt]

    public init(
        id: UUID = UUID(),
        title: String = "New Conversation",
        messages: [ChatMessage] = [],
        createdAt: Date = Date(),
        updatedAt: Date = Date(),
        selectedModelId: String? = nil,
        selectedModelName: String? = nil,
        selectedModelPath: String? = nil,
        isThinkingEnabled: Bool? = nil,
        isAgentToolsEnabled: Bool? = nil,
        queuedPrompts: [QueuedPrompt] = []
    ) {
        self.id = id
        self.title = title
        self.messages = messages
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.selectedModelId = selectedModelId
        self.selectedModelName = selectedModelName
        self.selectedModelPath = selectedModelPath
        self.isThinkingEnabled = isThinkingEnabled
        self.isAgentToolsEnabled = isAgentToolsEnabled
        self.queuedPrompts = queuedPrompts
    }

    enum CodingKeys: String, CodingKey {
        case id, title, messages, createdAt, updatedAt
        case selectedModelId, selectedModelName, selectedModelPath
        case isThinkingEnabled, isAgentToolsEnabled
        case queuedPrompts
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try container.decode(UUID.self, forKey: .id)
        self.title = try container.decode(String.self, forKey: .title)
        self.messages = try container.decode([ChatMessage].self, forKey: .messages)
        self.createdAt = try container.decode(Date.self, forKey: .createdAt)
        self.updatedAt = try container.decode(Date.self, forKey: .updatedAt)
        self.selectedModelId = try container.decodeIfPresent(String.self, forKey: .selectedModelId)
        self.selectedModelName = try container.decodeIfPresent(String.self, forKey: .selectedModelName)
        self.selectedModelPath = try container.decodeIfPresent(String.self, forKey: .selectedModelPath)
        self.isThinkingEnabled = try container.decodeIfPresent(Bool.self, forKey: .isThinkingEnabled)
        self.isAgentToolsEnabled = try container.decodeIfPresent(Bool.self, forKey: .isAgentToolsEnabled)
        self.queuedPrompts = try container.decodeIfPresent([QueuedPrompt].self, forKey: .queuedPrompts) ?? []
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(title, forKey: .title)
        try container.encode(messages, forKey: .messages)
        try container.encode(createdAt, forKey: .createdAt)
        try container.encode(updatedAt, forKey: .updatedAt)
        try container.encodeIfPresent(selectedModelId, forKey: .selectedModelId)
        try container.encodeIfPresent(selectedModelName, forKey: .selectedModelName)
        try container.encodeIfPresent(selectedModelPath, forKey: .selectedModelPath)
        try container.encodeIfPresent(isThinkingEnabled, forKey: .isThinkingEnabled)
        try container.encodeIfPresent(isAgentToolsEnabled, forKey: .isAgentToolsEnabled)
        try container.encode(queuedPrompts, forKey: .queuedPrompts)
    }
}
