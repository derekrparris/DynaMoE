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

public struct ChatMessage: Identifiable, Codable, Equatable {
    public var id: UUID
    public var role: MessageRole
    public var content: String
    public var thinkingContent: String?
    public var isThinking: Bool
    public var timestamp: Date
    public var tokenCount: Int
    public var tokensPerSec: Double

    public init(
        id: UUID = UUID(),
        role: MessageRole,
        content: String,
        thinkingContent: String? = nil,
        isThinking: Bool = false,
        timestamp: Date = Date(),
        tokenCount: Int = 0,
        tokensPerSec: Double = 0.0
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.thinkingContent = thinkingContent
        self.isThinking = isThinking
        self.timestamp = timestamp
        self.tokenCount = tokenCount
        self.tokensPerSec = tokensPerSec
    }
}

public struct ChatSession: Identifiable, Codable, Equatable {
    public var id: UUID
    public var title: String
    public var messages: [ChatMessage]
    public var createdAt: Date
    public var updatedAt: Date

    public init(
        id: UUID = UUID(),
        title: String = "New Conversation",
        messages: [ChatMessage] = [],
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.title = title
        self.messages = messages
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }
}
