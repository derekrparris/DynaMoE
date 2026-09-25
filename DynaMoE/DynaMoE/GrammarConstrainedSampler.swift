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

/// `nonisolated` for the same reason as the sampler: it is a pure data
/// structure owned and walked by the nonisolated grammar masking path.
nonisolated public final class TokenTrieNode {
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

/// Thread-safe by explicit locking, not by actor isolation: the live state
/// machine is driven from the detached generation task
/// (`ContentView.startAutoregressiveGeneration`), so `nonisolated` states the
/// real contract and `stateLock` provides the mutual exclusion.
nonisolated public final class GrammarConstrainedSampler {
    public static let shared = GrammarConstrainedSampler()

    private var _currentState: GrammarParserState = .outsideToolCall
    public var currentState: GrammarParserState {
        stateLock.lock()
        defer { stateLock.unlock() }
        return _currentState
    }

    private var _isEnabled: Bool = true
    public var isEnabled: Bool {
        get {
            stateLock.lock()
            defer { stateLock.unlock() }
            return _isEnabled
        }
        set {
            stateLock.lock()
            defer { stateLock.unlock() }
            _isEnabled = newValue
        }
    }
    /// Ling/Bailing-native calls put the name right after the open tag with no
    /// `<function=` wrapper; the tag-choice masks below would fight that format,
    /// so they only engage for models that use the structural-tag shape.
    private var _enforceStructuralTagContinuation: Bool = true
    public var enforceStructuralTagContinuation: Bool {
        get {
            stateLock.lock()
            defer { stateLock.unlock() }
            return _enforceStructuralTagContinuation
        }
        set {
            stateLock.lock()
            defer { stateLock.unlock() }
            _enforceStructuralTagContinuation = newValue
        }
    }

    // Registered Tool & Parameter sets
    private var registeredToolNames: Set<String> = []
    private var toolParameterKeys: [String: Set<String>] = [:]
    private var toolRequiredKeys: [String: Set<String>] = [:]
    private var toolTrie = TokenTrieNode()
    private let registrationLock = NSLock()

    /// Guards every piece of mutable grammar state. Recursive because the locked
    /// internals read the public accessors above.
    private let stateLock = NSRecursiveLock()
    /// Bumped by `beginGeneration`. A generation task carries the token it was
    /// issued; writes bearing a stale token are dropped, so a cancelled
    /// generation unwinding in parallel cannot clobber its successor's state.
    private var generationToken: UInt64 = 0

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

    /// Starts a generation: clears the live context and returns the token that
    /// `updateStateAndApplyLogitMask` requires. A cancelled generation unwinding
    /// in parallel still holds its old token, so its late `updateState` calls are
    /// dropped instead of corrupting the new generation's required-param gate.
    @discardableResult
    public func beginGeneration() -> UInt64 {
        stateLock.lock()
        defer { stateLock.unlock() }
        generationToken &+= 1
        _currentState = .outsideToolCall
        currentToolName = nil
        seenParameterKeys.removeAll()
        return generationToken
    }

    public func reset() {
        _ = beginGeneration()
    }

    /// True when `token` still identifies the live generation.
    ///
    /// The mask is silently skipped for a superseded generation, so the sampling
    /// loop uses this to notice that its mask was dropped and stop, rather than
    /// drawing a token from unmasked logits and committing it.
    public func isCurrent(_ token: UInt64) -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return token == generationToken
    }

    // MARK: - Dynamic State Transition

    public func updateState(emittedText: String) {
        stateLock.lock()
        defer { stateLock.unlock() }
        updateStateLocked(emittedText: emittedText)
    }

    /// Updates the shared state machine, refreshes the structural-tag setting and
    /// applies the mask for the resulting state under a single lock hold, so the
    /// three cannot be interleaved by another generation. Writes carrying a stale
    /// `token` are discarded.
    public func updateStateAndApplyLogitMask(
        emittedText: String,
        logits: UnsafeMutablePointer<Float>,
        vocabSize: Int,
        tokenDecoder: (UInt32) -> String?,
        enforceStructuralTagContinuation: Bool,
        token: UInt64
    ) {
        stateLock.lock()
        defer { stateLock.unlock() }
        guard token == generationToken else { return }
        _enforceStructuralTagContinuation = enforceStructuralTagContinuation
        updateStateLocked(emittedText: emittedText)
        applyLogitMaskLocked(logits: logits, vocabSize: vocabSize, tokenDecoder: tokenDecoder)
    }

    /// Caller must hold `stateLock`.
    private func updateStateLocked(emittedText: String) {
        guard isEnabled else { return }

        if emittedText.contains(toolCallClose) {
            _currentState = .outsideToolCall
            currentToolName = nil
            seenParameterKeys.removeAll()
            return
        }

        if !emittedText.contains(toolCallOpen) {
            _currentState = .outsideToolCall
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
                                _currentState = .closingToolCall(matchedPrefix: String(afterClose[fRange.upperBound...]))
                            } else {
                                _currentState = .closingFunction(matchedPrefix: afterClose)
                            }
                        } else {
                            _currentState = .insideParameterValue(toolName: fnName, paramKey: pKey)
                        }
                    } else {
                        _currentState = .insideParameterName(toolName: fnName, currentKey: afterP)
                    }
                } else {
                    _currentState = .enteringParameterTag(matchedPrefix: insideFnBody)
                }
            } else {
                _currentState = .insideFunctionName(currentName: afterFn)
                currentToolName = nil
                seenParameterKeys.removeAll()
            }
        } else {
            _currentState = .enteringToolCall(matchedPrefix: toolCallSlice)
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
        stateLock.lock()
        defer { stateLock.unlock() }
        applyLogitMaskLocked(logits: logits, vocabSize: vocabSize, tokenDecoder: tokenDecoder)
    }

    /// Caller must hold `stateLock`. Also takes `registrationLock` for the
    /// duration: `registerTools` rebuilds the schema in place, and a torn read
    /// mid-rebuild would let the tool-name mask silently drop out for a token.
    private func applyLogitMaskLocked(
        logits: UnsafeMutablePointer<Float>,
        vocabSize: Int,
        tokenDecoder: (UInt32) -> String?
    ) {
        guard isEnabled else { return }
        registrationLock.lock()
        defer { registrationLock.unlock() }

        switch _currentState {
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

            var allowedCount = 0
            var valveIndex: Int? = nil
            var valveLogit: Float = 0
            for v in 0..<vocabSize {
                guard let str = tokenDecoder(UInt32(v)) else { continue }
                let cand = currentPrefix + str
                // Still typing a name, or a token that spans name completion
                // and terminates it with '>' ("rch>"). Anything else that runs
                // past a complete name mints an unregistered tool (observed:
                // "web_search" + "_fetch" produced `web_search_fetch`, which
                // both broke the call and emptied the allowed set, silently
                // dropping the mask for the rest of the name).
                if allowedTools.contains(where: { $0.hasPrefix(cand) }) { allowedCount += 1; continue }
                if overshootTerminatesWord(cand, words: registeredToolNames) { allowedCount += 1; continue }
                // Dead-end valve: if a tokenizer has no bare '>' token, the strict
                // rule can mask the entire vocab, and sampling on all -inf logits
                // yields garbage rather than an error. Remember one token the loose
                // rule would accept so it can be released below.
                if valveIndex == nil, overshootStartsWordTerminator(cand, words: registeredToolNames) {
                    valveIndex = v
                    valveLogit = logits[v]
                }
                logits[v] = -Float.infinity
            }
            if allowedCount == 0, let valve = valveIndex { logits[valve] = valveLogit }

        case .insideParameterName(let toolName, let currentKey):
            guard let validKeys = toolParameterKeys[toolName] else { return }
            let allowedKeys = validKeys.filter { $0.hasPrefix(currentKey) }
            if allowedKeys.isEmpty { return }

            var allowedCount = 0
            var valveIndex: Int? = nil
            var valveLogit: Float = 0
            for v in 0..<vocabSize {
                guard let str = tokenDecoder(UInt32(v)) else { continue }
                let cand = currentKey + str
                if allowedKeys.contains(where: { $0.hasPrefix(cand) }) { allowedCount += 1; continue }
                if overshootTerminatesWord(cand, words: validKeys) { allowedCount += 1; continue }
                if valveIndex == nil, overshootStartsWordTerminator(cand, words: validKeys) {
                    valveIndex = v
                    valveLogit = logits[v]
                }
                logits[v] = -Float.infinity
            }
            if allowedCount == 0, let valve = valveIndex { logits[valve] = valveLogit }

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

    /// A candidate that runs past a complete word (tool name or parameter key)
    /// is only legal when the overshoot terminates the word: optional space/tab,
    /// then '>'. Trailing whitespace alone is not accepted — the state machine
    /// stays in the name state for it, where no registered word would match the
    /// padded prefix and the mask would silently drop out.
    private func overshootTerminatesWord(_ cand: String, words: Set<String>) -> Bool {
        for word in words where cand.hasPrefix(word) {
            var rest = cand.dropFirst(word.count)
            while rest.first == " " || rest.first == "\t" { rest = rest.dropFirst() }
            // '>' must END the overshoot. Accepting arbitrary text after it let one
            // merged token carry the whole tail past this state's mask, and the tail
            // is where the gates live: "name></function>" closed a call whose
            // required parameters had never been written (the required-param gate
            // only ever withholds the `</function>` tag choice, so it cannot see a
            // token that smuggles the tag in behind a '>'), and "name>junk" began
            // the body early. The tail must arrive as its own token to be gated.
            if rest == ">" { return true }
        }
        return false
    }

    /// Loose form of the overshoot test, used ONLY as a dead-end valve: it accepts
    /// '>' followed by anything. Not a legality rule — see `overshootTerminatesWord`.
    private func overshootStartsWordTerminator(_ cand: String, words: Set<String>) -> Bool {
        for word in words where cand.hasPrefix(word) {
            var rest = cand.dropFirst(word.count)
            while rest.first == " " || rest.first == "\t" { rest = rest.dropFirst() }
            if rest.first == ">" { return true }
        }
        return false
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
