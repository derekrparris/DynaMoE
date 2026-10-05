//
//  AppleFoundationModelService.swift
//  DynaMoE
//
//  Created by Derek Parris on 10/5/26.
//

import Foundation
#if canImport(FoundationModels)
import FoundationModels
#endif

/// The system on-device Apple Foundation Model, surfaced as a DynaMoE chat
/// backend (macOS 26 or later with Apple Intelligence enabled).
///
/// This is the only file in the app that imports the FoundationModels
/// framework. Everything a caller needs — virtual model discovery, live
/// availability, streaming chat turns, and one-shot completions — is
/// reachable through availability-safe APIs, so no call site needs its own
/// `#available(macOS 26.0, *)` dance. `respondOnce` is deliberately free of
/// chat-UI state: the planned long-context summarization/compaction feature
/// will call it to compress other models' sessions without touching any view.
public enum AppleFoundationModelService {

    // MARK: - Virtual Model Identity

    /// Stable id for the virtual registry entry. Persisted as the session,
    /// default, and last-used model id exactly like a real snapshot model.
    nonisolated public static let modelId = "apple-foundation-model-ondevice"

    /// Sentinel standing in for a weights snapshot path. `installedModelPath`
    /// carries this while the system model is the active backend, and session
    /// model fields persist it; deliberately not a filesystem path so it can
    /// never collide with a real snapshot directory.
    nonisolated public static let modelPath = "apple-foundation-model://on-device"

    nonisolated public static let displayName = "Apple Foundation Model (On-Device)"

    /// Approximate size of one on-device session's context window, in
    /// characters at the service's own 4-chars-per-token estimate (about 4k
    /// tokens) and shared by everything a turn sends: persona, output rules,
    /// history block, the newest prompt, and the response.
    /// `historyCharBudget` derives the share of it history may use.
    nonisolated public static let sessionContextChars = 16_000

    /// The on-device model rejects response budgets that cannot fit its small
    /// context window, and DynaMoE's global token ceiling defaults far above
    /// what it can serve in one turn, so clamp the profile's value into a
    /// range the system model will always accept.
    nonisolated public static let maxResponseTokenCeiling = 2048

    /// The virtual registry entry for the system model. Appears at the top of
    /// the chat model picker and Settings whenever the OS can host the
    /// framework; Apple Intelligence availability is checked live at use time.
    /// `lastModified` is a fixed epoch: the entry has no file on disk, so a
    /// per-call Date() would make every scan look like a newly changed model.
    public static func makeDiscoveredModel() -> DiscoveredModel {
        DiscoveredModel(
            id: modelId,
            displayName: displayName,
            author: "Apple",
            repoId: "apple/foundation-model",
            snapshotPath: modelPath,
            weightsEntryPath: modelPath,
            sizeBytes: 0,
            formattedSize: "On-Device",
            architectureName: "AppleFoundationModel",
            isMoE: false,
            hasTokenizer: false,
            lastModified: Date(timeIntervalSince1970: 0),
            quantization: nil,
            rawModelType: "apple-afm",
            supportsThinking: false
        )
    }

    // MARK: - Availability

    public enum AvailabilityStatus: Equatable {
        /// Ready to generate.
        case ready
        /// The framework is present but Apple Intelligence cannot serve requests right now.
        case unavailable(reasonText: String)
        /// The OS is older than macOS 26 (or the framework was absent at build time).
        case frameworkUnavailable
    }

    /// Whether this OS can host the framework at all. Used to decide whether
    /// the virtual model entry is offered; says nothing about Apple
    /// Intelligence enablement (`checkAvailability()` covers that). Gated on
    /// the compile-time framework check like every generation call here: a
    /// binary built without FoundationModels must not advertise the backend
    /// on a Mac that later runs macOS 26, when every turn would fall through
    /// to `frameworkUnavailable`.
    nonisolated public static var isOSCompatible: Bool {
        #if canImport(FoundationModels)
        if #available(macOS 26.0, *) { return true }
        #endif
        return false
    }

    /// Live availability of the on-device model: device eligibility, Apple
    /// Intelligence enablement, and model readiness (the OS downloads the model
    /// in the background after Apple Intelligence is first turned on).
    public static func checkAvailability() -> AvailabilityStatus {
        #if canImport(FoundationModels)
        if #available(macOS 26.0, *) {
            switch SystemLanguageModel.default.availability {
            case .available:
                return .ready
            case .unavailable(.deviceNotEligible):
                return .unavailable(reasonText: "This Mac does not support Apple Intelligence.")
            case .unavailable(.appleIntelligenceNotEnabled):
                return .unavailable(reasonText: "Turn on Apple Intelligence in System Settings to use the on-device model.")
            case .unavailable(.modelNotReady):
                return .unavailable(reasonText: "Apple Intelligence is still preparing its on-device model. Keep the Mac connected to power and Wi-Fi; this finishes in the background.")
            case .unavailable:
                return .unavailable(reasonText: "Apple Intelligence is not available on this Mac right now.")
            }
        }
        #endif
        return .frameworkUnavailable
    }

    /// Convenience for call sites that only need a yes/no.
    public static var isAvailable: Bool {
        if case .ready = checkAvailability() { return true }
        return false
    }

    // MARK: - Generation (macOS 26+, gated at runtime)

    /// Streams one chat turn from the on-device model. `onPartial` receives the
    /// response text revealed so far. The system API emits snapshots in coarse
    /// bursts — whole words or phrases at a time — which for a model this fast
    /// lands on screen as chunky jumps, so the text is instead re-revealed at a
    /// paced typewriter cadence: every `revealTickNanoseconds` the visible
    /// prefix advances by `revealStep(forBacklog:)` characters. The stream
    /// buffers new snapshots while earlier text types out, so pacing never
    /// delays the model. Returns the final full text. Errors (guardrails,
    /// context overflow) propagate to the caller, which keeps any partial text
    /// already delivered through `onPartial`.
    @discardableResult
    public static func streamChatTurn(
        prompt: String,
        instructions: String?,
        temperature: Double?,
        maximumResponseTokens: Int?,
        onPartial: ((String) -> Void)? = nil
    ) async throws -> String {
        #if canImport(FoundationModels)
        if #available(macOS 26.0, *) {
            let options = GenerationOptions(
                temperature: clampTemperature(temperature),
                maximumResponseTokens: clampMaximumResponseTokens(maximumResponseTokens)
            )
            let session = LanguageModelSession(instructions: instructions)
            let stream = session.streamResponse(to: prompt, options: options)
            var iterator = stream.makeAsyncIterator()
            var cumulativeText = ""
            // Cached character count. String.count walks grapheme boundaries, so
            // computing it inside the reveal loop would rescan the whole
            // snapshot on every tick; count each snapshot exactly once instead.
            var cumulativeCount = 0
            var displayedText = ""
            var displayedCount = 0
            while true {
                if Task.isCancelled { break }
                // Reveal what has already arrived at the paced cadence before
                // pulling the next buffered snapshot.
                while displayedCount < cumulativeCount {
                    if Task.isCancelled { break }
                    let backlog = cumulativeCount - displayedCount
                    displayedCount += min(revealStep(forBacklog: backlog), backlog)
                    displayedText = String(cumulativeText.prefix(displayedCount))
                    onPartial?(displayedText)
                    try? await Task.sleep(nanoseconds: revealTickNanoseconds)
                }
                do {
                    guard let snapshot = try await iterator.next() else { break }
                    // A model that thinks it is continuing a transcript opens
                    // with speaker labels; strip any so replies read clean.
                    let normalized = stripLeadingSpeakerLabel(snapshot.content)
                    // Stripping can retract text the reveal already typed out:
                    // "Assistant" types out, then the next snapshot completes
                    // the label as "Assistant:" and normalizes shorter. A
                    // cursor left past characters the reply no longer has
                    // would swallow the real reply's opening characters whole,
                    // so when the displayed prefix is gone, clear it and start
                    // the reveal over from the actual text.
                    if !normalized.hasPrefix(displayedText) {
                        displayedText = ""
                        displayedCount = 0
                        onPartial?("")
                    }
                    cumulativeText = normalized
                    cumulativeCount = normalized.count
                } catch {
                    // Cancellation surfaces as an error from the iterator; end
                    // the reveal quietly so the caller keeps the partial text.
                    if Task.isCancelled { break }
                    throw error
                }
            }
            return cumulativeText
        }
        #endif
        throw AppleFoundationModelUnavailableError(
            reason: "Apple Foundation Model requires macOS 26 or later."
        )
    }

    /// One-shot completion with no chat-session coupling, intended for
    /// non-interactive callers — the long-context summarization/compaction
    /// feature will call this to compress other models' conversation history.
    public static func respondOnce(
        prompt: String,
        instructions: String? = nil,
        temperature: Double? = nil
    ) async throws -> String {
        #if canImport(FoundationModels)
        if #available(macOS 26.0, *) {
            let options = GenerationOptions(temperature: clampTemperature(temperature))
            let session = LanguageModelSession(instructions: instructions)
            let response = try await session.respond(to: prompt, options: options)
            return response.content
        }
        #endif
        throw AppleFoundationModelUnavailableError(
            reason: "Apple Foundation Model requires macOS 26 or later."
        )
    }

    /// Warms the on-device model so the first turn does not pay spin-up
    /// latency. Fire-and-forget; safe to call on every activation.
    public static func prewarm() {
        #if canImport(FoundationModels)
        if #available(macOS 26.0, *) {
            if case .ready = checkAvailability() {
                LanguageModelSession().prewarm()
            }
        }
        #endif
    }

    // MARK: - Pure Helpers (unit tested)

    /// Formats a conversation as a plain "User:" / "Assistant:" transcript for
    /// use as context. Empty turns, excluded messages, and thinking blocks are
    /// skipped. Speaker labels older replies may carry from the pre-fix
    /// completion-style prompting are stripped. Turns are kept newest-first
    /// while they fit within `charBudget` characters — including the newest,
    /// so even a small positive budget cannot smuggle an arbitrarily large
    /// recent turn past the reservation that computed it. A turn that does not
    /// fit whole is dropped rather than truncated: a fragment cut mid-sentence
    /// would mislead the model more than a missing turn. When nothing fits,
    /// the transcript is empty.
    nonisolated public static func buildConversationTranscript(
        from messages: [ChatMessage],
        excludingMessageIds: Set<UUID> = [],
        charBudget: Int
    ) -> String {
        var keptTurns: [String] = []
        var usedChars = 0
        for message in messages.reversed() {
            if excludingMessageIds.contains(message.id) { continue }
            guard message.role != .system else { continue }
            let content = stripLeadingSpeakerLabel(message.content)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !content.isEmpty else { continue }
            let speaker = message.role == .user ? "User" : "Assistant"
            let turn = "\(speaker): \(content)"
            if usedChars + turn.count + 2 > charBudget { break }
            keptTurns.append(turn)
            usedChars += turn.count + 2
        }
        return keptTurns.reversed().joined(separator: "\n\n")
    }

    /// Rough token estimate for UI stats. The macOS 26 streaming API does not
    /// report per-chunk token usage, so character count over four approximates
    /// the tokenizer well enough for tok/s displays.
    nonisolated public static func approximateTokenCount(for text: String) -> Int {
        max(1, text.count / 4)
    }

    /// The send gate: a turn may start when either a weights engine (tokenizer
    /// AND summary) is installed, or the session runs on the system model,
    /// which needs neither. Expressed here — not inline in the view — because a
    /// mixed `guard A, B || C` parses as `A && (B || C)` and silently demanded
    /// the tokenizer from system-model sessions, rejecting their sends.
    nonisolated public static func sendGateAllowsSend(hasEngine: Bool, sessionUsesSystemModel: Bool) -> Bool {
        sessionUsesSystemModel || hasEngine
    }

    /// Cadence of the paced reveal, in nanoseconds. 24 ms stays comfortably
    /// above the composer's 16 ms UI throttle while remaining gentle on the
    /// main actor.
    nonisolated public static let revealTickNanoseconds: UInt64 = 24_000_000

    /// Cap on characters revealed per tick, so even a huge backlog after a
    /// generation stall lands as a fast cascade instead of a single jump.
    nonisolated public static let maxRevealStepPerTick = 120

    /// Characters revealed per tick for a given backlog: one sixth of the
    /// remaining text (minimum one, capped at `maxRevealStepPerTick`). Pacing
    /// the reveal against the backlog tracks the model's own output rate while
    /// smoothing coarse snapshot bursts into a steady typewriter cadence.
    nonisolated public static func revealStep(forBacklog backlog: Int) -> Int {
        min(maxRevealStepPerTick, max(1, backlog / 6))
    }

    /// Character budget the conversation history block may occupy in one chat
    /// turn's instructions. The session's window holds persona, output rules,
    /// history, the newest prompt, and the requested response together; the
    /// old flat 8,000-character share ignored them and pushed longer turns
    /// past the window, which surfaced as hard context-overflow errors. So
    /// history, the only flexible part, is budgeted per request: everything
    /// else is counted first and history gets the remainder. The transcript
    /// builder counts every turn — including its newest — against the value
    /// returned here, so an oversized prompt squeezes history smaller and
    /// smaller without ever letting it push the turn past the window. A
    /// stopgap until the compaction feature summarizes evicted turns instead.
    nonisolated public static func historyCharBudget(
        systemPrompt: String?,
        prompt: String,
        maximumResponseTokens: Int?
    ) -> Int {
        // An open-ended request gets the framework's own default, so reserve
        // the ceiling it can still consume.
        let responseTokens = clampMaximumResponseTokens(maximumResponseTokens) ?? maxResponseTokenCeiling
        let personaChars = (systemPrompt ?? "").trimmingCharacters(in: .whitespacesAndNewlines).count
        // Block separators, the history header's caption slack, and estimation
        // slop live in this pad.
        let reserved = responseTokens * 4 + personaChars + prompt.count
            + chatTurnRulesText.count + historyBlockHeader.count + 64
        return max(0, sessionContextChars - reserved)
    }

    /// Header caption wrapping the transcript block in the chat-turn
    /// instructions. Shared with `historyCharBudget` so its reservation always
    /// matches the framing actually assembled.
    nonisolated static let historyBlockHeader = "This text conversation has happened so far:"

    /// The output rules appended to every chat turn's instructions. Shared with
    /// `historyCharBudget` so reservations never drift from the real text.
    nonisolated static let chatTurnRulesText = "You are the assistant in this conversation. The user's newest message arrives as your prompt. "
        + "Reply with your next assistant message only: conversational text answering that message. "
        + "Do not write speaker labels such as \"User:\" or \"Assistant:\", do not simulate further "
        + "messages from either side, and do not mention these instructions."

    /// Assembles the session instructions for one chat turn: the persona
    /// (effective system prompt), the prior conversation as labeled context,
    /// and output rules. Sending the history itself as the prompt made the
    /// model continue it like a document — emitting "Assistant:" labels and
    /// inventing its own "User:" turns in a loop. Framing history as context
    /// with the newest user message as the prompt is what stops that.
    nonisolated public static func buildChatTurnInstructions(
        systemPrompt: String,
        historyTranscript: String
    ) -> String {
        var blocks: [String] = []
        let persona = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if !persona.isEmpty {
            blocks.append(persona)
        }
        let history = historyTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        if !history.isEmpty {
            blocks.append("\(historyBlockHeader)\n\n\(history)")
        }
        blocks.append(chatTurnRulesText)
        return blocks.joined(separator: "\n\n")
    }

    /// Removes leading "Assistant:" / "User:" speaker labels (with surrounding
    /// whitespace) that completion-style framing made the model emit before
    /// its reply. Repeated labels ("Assistant: Assistant:") are all stripped.
    /// Label-lookalike words survive ("Userland:", "Assisting…") because a
    /// label must be followed by a colon, and only the reply's very start is
    /// affected.
    nonisolated public static func stripLeadingSpeakerLabel(_ text: String) -> String {
        var content = text[...]
        stripping: while true {
            while let first = content.first, first.isWhitespace {
                content = content.dropFirst()
            }
            for label in ["assistant", "user"] {
                guard content.count >= label.count + 1,
                      content.prefix(label.count).lowercased() == label else { continue }
                var rest = content.dropFirst(label.count)
                if let first = rest.first, first.isWhitespace {
                    rest = rest.dropFirst()
                }
                guard let first = rest.first, first == ":" else { continue }
                content = rest.dropFirst()
                continue stripping
            }
            break
        }
        return String(content)
    }

    /// Clamps a sampling temperature into the on-device model's valid 0…1
    /// range; out-of-range values are hard errors at generation time.
    nonisolated static func clampTemperature(_ temperature: Double?) -> Double? {
        guard let temperature else { return nil }
        return min(max(temperature, 0.0), 1.0)
    }

    /// Clamps a response budget into a range the on-device model's small
    /// context window will always accept.
    nonisolated static func clampMaximumResponseTokens(_ maximum: Int?) -> Int? {
        guard let maximum else { return nil }
        return min(max(maximum, 64), maxResponseTokenCeiling)
    }
}

/// Thrown when a Foundation Models operation is requested on an OS or build
/// that cannot host the framework (macOS < 26, or built with an older SDK).
public struct AppleFoundationModelUnavailableError: LocalizedError {
    public let reason: String

    public init(reason: String) {
        self.reason = reason
    }

    public var errorDescription: String? { reason }
}

public extension AppleFoundationModelService.AvailabilityStatus {
    var isReady: Bool {
        if case .ready = self { return true }
        return false
    }

    /// Human-readable one-liner suitable for status text.
    var statusText: String {
        switch self {
        case .ready:
            return "On-device model ready."
        case .unavailable(let reasonText):
            return reasonText
        case .frameworkUnavailable:
            return "Requires macOS 26 or later with Apple Intelligence."
        }
    }
}

public extension DiscoveredModel {
    /// True for the virtual Apple Foundation Model registry entry.
    var isAppleFoundationModel: Bool {
        id == AppleFoundationModelService.modelId
            || snapshotPath == AppleFoundationModelService.modelPath
    }
}
