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
    /// Ling/Bailing-native calls put the name right after the open tag with no
    /// `<function=` wrapper; the tag-choice masks below would fight that format,
    /// so they only engage for models that use the structural-tag shape.
    public var enforceStructuralTagContinuation: Bool = true

    // Registered Tool & Parameter sets
    private var registeredToolNames: Set<String> = []
    private var toolParameterKeys: [String: Set<String>] = [:]
    private var toolRequiredKeys: [String: Set<String>] = [:]
    private var toolTrie = TokenTrieNode()
    private let registrationLock = NSLock()

    // Live call context, refreshed on every updateState so the tag-choice masks
    // can withhold `</function>` until a tool's required parameters are present.
    private var currentToolName: String?
    private var seenParameterKeys: Set<String> = []

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
        registrationLock.lock()
        registeredToolNames.removeAll()
        toolParameterKeys.removeAll()
        toolRequiredKeys.removeAll()
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

            var requiredKeys = Set<String>()
            if let required = tool.function.parameters["required"]?.value as? [String] {
                requiredKeys.formUnion(required)
            } else if let requiredAny = tool.function.parameters["required"]?.value as? [Any] {
                requiredKeys.formUnion(requiredAny.compactMap { $0 as? String })
            }
            toolRequiredKeys[name] = requiredKeys
        }
        registrationLock.unlock()
    }

    public func reset() {
        currentState = .outsideToolCall
        currentToolName = nil
        seenParameterKeys.removeAll()
    }

    // MARK: - Dynamic State Transition

    public func updateState(emittedText: String) {
        guard isEnabled else { return }

        if emittedText.contains(toolCallClose) {
            currentState = .outsideToolCall
            currentToolName = nil
            seenParameterKeys.removeAll()
            return
        }

        if !emittedText.contains(toolCallOpen) {
            currentState = .outsideToolCall
            currentToolName = nil
            seenParameterKeys.removeAll()
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
                currentToolName = fnName
                seenParameterKeys = parameterKeysTyped(in: insideFnBody)

                // Check parameter state
                if let pRange = insideFnBody.range(of: paramOpenPrefix, options: .backwards) {
                    let afterP = String(insideFnBody[pRange.upperBound...])
                    if let pGt = afterP.range(of: ">") {
                        let pKey = String(afterP[..<pGt.lowerBound]).trimmingCharacters(in: .whitespaces)
                        let afterValue = String(afterP[pGt.upperBound...])
                        if let cRange = afterValue.range(of: paramClose) {
                            let afterClose = String(afterValue[cRange.upperBound...])
                            if let fRange = afterClose.range(of: functionClose, options: .backwards) {
                                currentState = .closingToolCall(matchedPrefix: String(afterClose[fRange.upperBound...]))
                            } else {
                                currentState = .closingFunction(matchedPrefix: afterClose)
                            }
                        } else {
                            currentState = .insideParameterValue(toolName: fnName, paramKey: pKey)
                        }
                    } else {
                        currentState = .insideParameterName(toolName: fnName, currentKey: afterP)
                    }
                } else {
                    currentState = .enteringParameterTag(matchedPrefix: insideFnBody)
                }
            } else {
                currentState = .insideFunctionName(currentName: afterFn)
                currentToolName = nil
                seenParameterKeys.removeAll()
            }
        } else {
            currentState = .enteringToolCall(matchedPrefix: toolCallSlice)
            currentToolName = nil
            seenParameterKeys.removeAll()
        }
    }

    /// Parameter keys actually OPENED in a function body, used to decide whether
    /// the call has satisfied its tool's required arguments yet. Walks the body
    /// and skips each value span, so a literal `<parameter=...>` printed inside
    /// a value (shell commands echoing tool markup, docs, cwd strings) is not
    /// mistaken for an opened required key. Unterminated values stop the walk,
    /// which keeps the required-parameter gate engaged (the safe direction).
    private func parameterKeysTyped(in body: String) -> Set<String> {
        var keys = Set<String>()
        var searchStart = body.startIndex
        while searchStart < body.endIndex,
              let open = body.range(of: paramOpenPrefix, range: searchStart..<body.endIndex),
              let gt = body.range(of: ">", range: open.upperBound..<body.endIndex) {
            let key = body[open.upperBound..<gt.lowerBound].trimmingCharacters(in: .whitespaces)
            if !key.isEmpty { keys.insert(key) }
            guard let close = body.range(of: paramClose, range: gt.upperBound..<body.endIndex) else { break }
            searchStart = close.upperBound
        }
        return keys
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

        case .enteringToolCall(let matchedPrefix):
            // Once the tool-call open tag has been emitted, the only legal
            // continuation is typing `<function=` (then a registered name via
            // insideFunctionName). Left unconstrained, the model can loop bare
            // open tags forever (observed: 21 consecutive empty tool calls).
            if enforceStructuralTagContinuation {
                applyTagChoiceMask(
                    logits: logits,
                    vocabSize: vocabSize,
                    currentPrefix: matchedPrefix,
                    tokenDecoder: tokenDecoder,
                    options: [functionOpenPrefix]
                )
            }

        case .enteringParameterTag(let matchedPrefix):
            // Reached with a function body that has no complete `<parameter=`:
            // a fresh body, a malformed attempt like `<parameter(names>`, or a
            // body that already closed with `</function>`. Constrain each case
            // to its legal continuations instead of leaving the vocab open.
            guard enforceStructuralTagContinuation else { return }
            if matchedPrefix.contains(functionClose) {
                applyTagChoiceMask(
                    logits: logits,
                    vocabSize: vocabSize,
                    currentPrefix: canonicalPrefix(matchedPrefix, after: functionClose),
                    tokenDecoder: tokenDecoder,
                    options: [toolCallClose]
                )
            } else {
                applyTagChoiceMask(
                    logits: logits,
                    vocabSize: vocabSize,
                    currentPrefix: canonicalPrefix(matchedPrefix, after: paramClose),
                    tokenDecoder: tokenDecoder,
                    options: optionsRequiringRequiredParams([paramOpenPrefix, functionClose])
                )
            }

        case .insideFunctionName(let currentPrefix):
            let allowedTools = registeredToolNames.filter { $0.hasPrefix(currentPrefix) }
            if allowedTools.isEmpty { return }
            // The '>' escape must only open once a COMPLETE tool name has been
            // typed; otherwise the model can legally emit '<function>' with an
            // empty name and then loop on empty '<parameter>' tags (observed as
            // degenerate-cycle breaks that truncate tool calls).
            let nameComplete = registeredToolNames.contains(currentPrefix)

            for v in 0..<vocabSize {
                guard let str = tokenDecoder(UInt32(v)) else { continue }
                if str.contains(">") && nameComplete {
                    continue
                }
                let cand = currentPrefix + str
                let matchesAny = allowedTools.contains { $0.hasPrefix(cand) || cand.hasPrefix($0) }
                if !matchesAny && !(str.hasPrefix(">") && nameComplete) {
                    logits[v] = -Float.infinity
                }
            }

        case .insideParameterName(let toolName, let currentKey):
            guard let validKeys = toolParameterKeys[toolName] else { return }
            let allowedKeys = validKeys.filter { $0.hasPrefix(currentKey) }
            if allowedKeys.isEmpty { return }
            let keyComplete = validKeys.contains(currentKey)

            for v in 0..<vocabSize {
                guard let str = tokenDecoder(UInt32(v)) else { continue }
                if str.contains(">") && keyComplete {
                    continue
                }
                let cand = currentKey + str
                let matchesAny = allowedKeys.contains { $0.hasPrefix(cand) || cand.hasPrefix($0) }
                if !matchesAny && !(str.hasPrefix(">") && keyComplete) {
                    logits[v] = -Float.infinity
                }
            }

        case .closingFunction(let matchedPrefix):
            guard enforceStructuralTagContinuation else { return }
            applyTagChoiceMask(
                logits: logits,
                vocabSize: vocabSize,
                // Re-anchor past any spurious extra `</parameter>`: without this
                // the second close dead-ends the prefix and unmasks the vocab,
                // resurrecting the close-tag loop.
                currentPrefix: canonicalPrefix(matchedPrefix, after: paramClose),
                tokenDecoder: tokenDecoder,
                options: optionsRequiringRequiredParams([paramOpenPrefix, functionClose])
            )

        case .closingToolCall(let matchedPrefix):
            guard enforceStructuralTagContinuation else { return }
            applyTagChoiceMask(
                logits: logits,
                vocabSize: vocabSize,
                currentPrefix: canonicalPrefix(matchedPrefix, after: functionClose),
                tokenDecoder: tokenDecoder,
                options: [toolCallClose]
            )

        default:
            return
        }
    }

    /// Withholds `</function>` while the current tool still has required
    /// parameters that were never opened, forcing the model to emit them.
    /// Prevents structurally-legal but semantically-empty calls (e.g. a
    /// shell_run with no `command`) that the harness must otherwise reject.
    private func optionsRequiringRequiredParams(_ base: [String]) -> [String] {
        guard let tool = currentToolName,
              let required = toolRequiredKeys[tool], !required.isEmpty,
              !required.isSubset(of: seenParameterKeys) else { return base }
        return base.filter { $0 != functionClose }
    }

    /// Drops everything through the last occurrence of `marker` so tag-choice
    /// masking re-anchors at the latest structural boundary instead of
    /// dead-ending on already-emitted tags.
    private func canonicalPrefix(_ prefix: String, after marker: String) -> String {
        guard let range = prefix.range(of: marker, options: .backwards) else { return prefix }
        return String(prefix[range.upperBound...])
    }

    /// After a parameter value has closed, the only legal continuations are
    /// another `<parameter=` tag or `</function>`, with whitespace between them.
    /// Without this mask the model can repeat `</parameter>` until the cycle
    /// guard truncates the call, which then poisons the pinned prefix.
    private func applyTagChoiceMask(
        logits: UnsafeMutablePointer<Float>,
        vocabSize: Int,
        currentPrefix: String,
        tokenDecoder: (UInt32) -> String?,
        options: [String]
    ) {
        func isWSChar(_ c: Character) -> Bool { c == " " || c == "\n" || c == "\t" || c == "\r" }
        func isWS(_ s: some StringProtocol) -> Bool { s.allSatisfy(isWSChar) }
        let trimmed = currentPrefix.drop(while: isWSChar)
        let atWhitespaceBoundary = trimmed.isEmpty
        // Dead-end valve: a prefix that can no longer reach any legal tag (e.g. a
        // half-typed second `</parameter>`) would mask the entire vocab; unmask
        // instead and let the next state transition recover.
        if !options.contains(where: { $0.hasPrefix(trimmed) || trimmed.hasPrefix($0) }) { return }

        func allowed(_ candidate: String) -> Bool {
            let t = candidate.drop(while: isWSChar)
            for option in options {
                if option.hasPrefix(t) { return true }
                if t.hasPrefix(option) {
                    let rest = t.dropFirst(option.count)
                    if isWS(rest) { return true }
                    if options.contains(where: { $0.hasPrefix(rest.drop(while: isWSChar)) }) { return true }
                }
            }
            return false
        }

        for v in 0..<vocabSize {
            guard let str = tokenDecoder(UInt32(v)) else { continue }
            if atWhitespaceBoundary && isWS(str) { continue }
            if allowed(currentPrefix + str) { continue }
            logits[v] = -Float.infinity
        }
    }
}
