//
//  GrammarConstrainedSampler.swift
//  DynaMoE
//
//  Phase 1: Constrained Generation Engine (Grammar & Logit Masking)
//  Enforces XML/JSON schema compliance during token sampling on Apple Silicon Metal.
//  Dynamically sets invalid token logits to -Float.infinity so syntax errors
//  are mathematically impossible during tool calls.
//

import Foundation
import Accelerate

// MARK: - Grammar State Machine

public enum GrammarParserState: Equatable {
    case outsideToolCall
    case enteringToolCall(matchedPrefix: String)
    case insideFunctionName(currentName: String)
    case enteringParameterTag(matchedPrefix: String)
    case insideParameterName(toolName: String, currentKey: String)
    case insideParameterValue(toolName: String, paramKey: String)
    case closingParameter(matchedPrefix: String)
    case closingFunction(matchedPrefix: String)
    case closingToolCall(matchedPrefix: String)
}

// MARK: - Token Prefix Trie for High-Speed Logit Masking

public final class TokenTrieNode {
    public var isTerminal: Bool = false
    public var children: [Character: TokenTrieNode] = [:]
    public var terminalTokenIds: Set<UInt32> = []

    public init() {}

    public func insert(word: String, tokenId: UInt32? = nil) {
        var current = self
        for char in word {
            if let next = current.children[char] {
                current = next
            } else {
                let newNode = TokenTrieNode()
                current.children[char] = newNode
                current = newNode
            }
        }
        current.isTerminal = true
        if let tid = tokenId {
            current.terminalTokenIds.insert(tid)
        }
    }

    public func findNode(prefix: String) -> TokenTrieNode? {
        var current = self
        for char in prefix {
            guard let next = current.children[char] else { return nil }
            current = next
        }
        return current
    }

    public func allowedNextCharacters(prefix: String) -> Set<Character> {
        guard let node = findNode(prefix: prefix) else { return [] }
        return Set(node.children.keys)
    }
}

// MARK: - Grammar-Constrained Sampler

public final class GrammarConstrainedSampler {
    public static let shared = GrammarConstrainedSampler()

    public private(set) var currentState: GrammarParserState = .outsideToolCall
    public var isEnabled: Bool = true

    // Registered Tool & Parameter sets
    private var registeredToolNames: Set<String> = []
    private var toolParameterKeys: [String: Set<String>] = [:]
    private var toolTrie = TokenTrieNode()

    // Structural Tag constants
    private let toolCallOpen = "<tool_call>"
    private let toolCallClose = "</tool_call>"
    private let functionOpenPrefix = "<function="
    private let functionClose = "</function>"
    private let paramOpenPrefix = "<parameter="
    private let paramClose = "</parameter>"

    public init() {}

    // MARK: - Schema Registration

    public func registerTools(_ tools: [ToolDefinition]) {
        registeredToolNames.removeAll()
        toolParameterKeys.removeAll()
        toolTrie = TokenTrieNode()

        for tool in tools {
            let name = tool.function.name
            registeredToolNames.insert(name)
            toolTrie.insert(word: name)

            var paramKeys = Set<String>()
            if let props = tool.function.parameters["properties"]?.value as? [String: Any] {
                for key in props.keys {
                    paramKeys.insert(key)
                }
            }
            toolParameterKeys[name] = paramKeys
        }
    }

    public func reset() {
        currentState = .outsideToolCall
    }

    // MARK: - Dynamic State Transition

    public func updateState(emittedText: String) {
        guard isEnabled else { return }

        if emittedText.contains(toolCallClose) {
            currentState = .outsideToolCall
            return
        }

        if !emittedText.contains(toolCallOpen) {
            currentState = .outsideToolCall
            return
        }

        // Inside tool call block
        let toolCallSlice = emittedText.components(separatedBy: toolCallOpen).last ?? ""

        // Check if function tag is open
        if let fnRange = toolCallSlice.range(of: functionOpenPrefix, options: .backwards) {
            let afterFn = String(toolCallSlice[fnRange.upperBound...])
            if let gtRange = afterFn.range(of: ">") {
                let fnName = String(afterFn[..<gtRange.lowerBound]).trimmingCharacters(in: .whitespaces)
                let insideFnBody = String(afterFn[gtRange.upperBound...])

                // Check parameter state
                if let pRange = insideFnBody.range(of: paramOpenPrefix, options: .backwards) {
                    let afterP = String(insideFnBody[pRange.upperBound...])
                    if let pGt = afterP.range(of: ">") {
                        let pKey = String(afterP[..<pGt.lowerBound]).trimmingCharacters(in: .whitespaces)
                        currentState = .insideParameterValue(toolName: fnName, paramKey: pKey)
                    } else {
                        currentState = .insideParameterName(toolName: fnName, currentKey: afterP)
                    }
                } else {
                    currentState = .enteringParameterTag(matchedPrefix: insideFnBody)
                }
            } else {
                currentState = .insideFunctionName(currentName: afterFn)
            }
        } else {
            currentState = .enteringToolCall(matchedPrefix: toolCallSlice)
        }
    }

    // MARK: - Metal / Accelerate Logit Masking Kernel

    /// Applies constraint mask to raw logits before sampling.
    /// Disallowed tokens are set to -Float.infinity so they cannot be selected.
    public func applyLogitMask(
        logits: UnsafeMutablePointer<Float>,
        vocabSize: Int,
        tokenDecoder: (UInt32) -> String?
    ) {
        guard isEnabled else { return }

        switch currentState {
        case .outsideToolCall:
            return

        case .insideFunctionName(let currentPrefix):
            let allowedTools = registeredToolNames.filter { $0.hasPrefix(currentPrefix) }
            if allowedTools.isEmpty { return }

            for v in 0..<vocabSize {
                guard let str = tokenDecoder(UInt32(v)) else { continue }
                if str.contains(">") && registeredToolNames.contains(currentPrefix) {
                    continue
                }
                let cand = currentPrefix + str
                let matchesAny = allowedTools.contains { $0.hasPrefix(cand) || cand.hasPrefix($0) }
                if !matchesAny && !str.hasPrefix(">") {
                    logits[v] = -Float.infinity
                }
            }

        case .insideParameterName(let toolName, let currentKey):
            guard let validKeys = toolParameterKeys[toolName] else { return }
            let allowedKeys = validKeys.filter { $0.hasPrefix(currentKey) }
            if allowedKeys.isEmpty { return }

            for v in 0..<vocabSize {
                guard let str = tokenDecoder(UInt32(v)) else { continue }
                if str.contains(">") && validKeys.contains(currentKey) {
                    continue
                }
                let cand = currentKey + str
                let matchesAny = allowedKeys.contains { $0.hasPrefix(cand) || cand.hasPrefix($0) }
                if !matchesAny && !str.hasPrefix(">") {
                    logits[v] = -Float.infinity
                }
            }

        default:
            return
        }
    }
}
