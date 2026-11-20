//
//  AgentReportBuilder.swift
//  DynaMoE
//
//  Derives the part of an `AgentReport` that comes from what the run actually
//  said, rather than from a fixed template. See SUBAGENTS.md §4 and §5.
//

import Foundation

/// Turns a run's transcript into the findings, open questions and partial
/// summary its report carries.
///
/// The loop used to hand the report literal `findings: []` and
/// `openQuestions: []`, so the inspector's sections could never fill, and a
/// run that ended on a budget kept only whatever its LAST text happened to be —
/// often a preamble like "Let me check the other file", with the analysis it
/// had already finished thrown away. Everything here is a pure function of the
/// transcript and invents nothing: a finding is a claim the run itself made,
/// and its evidence names the step it was made at and the tool results that
/// were in front of it.
nonisolated enum AgentReportBuilder {
    /// A report is a report, not a second transcript: a run may talk for
    /// dozens of steps, and the drawer shows the transcript separately.
    static let maximumFindings = 12
    static let maximumOpenQuestions = 8
    /// One-line narration ("Now let me read the file.") is not a finding.
    static let minimumClaimCharacters = 40
    static let maximumClaimCharacters = 400
    static let maximumQuestionCharacters = 240
    /// A finding is the model's own statement, not something verified against
    /// the tree, so every finding carries the same fixed confidence rather than
    /// a per-claim number nothing measured.
    static let statementConfidence = 0.5

    // MARK: Findings

    /// One finding per substantive claim in the run's own text.
    ///
    /// A claim is a bullet or a prose line outside a fenced code block. Headings
    /// are structure, a question is not an assertion, and a code sample is not a
    /// claim, so all three are skipped. Texts are visited most-substantial first
    /// (ties go to the later step), so a long run's finished analysis survives
    /// the cap even when it is followed by a dozen short preambles.
    static func findings(from transcript: [AgentTranscriptEntry]) -> [AgentFinding] {
        Array(allFindings(from: transcript).prefix(maximumFindings))
    }

    /// How many findings the run produced, before the report's cap. The
    /// report says what it dropped, and it can only do that by counting the
    /// whole list.
    static func findingCount(from transcript: [AgentTranscriptEntry]) -> Int {
        allFindings(from: transcript).count
    }

    private static func allFindings(from transcript: [AgentTranscriptEntry]) -> [AgentFinding] {
        var findings: [AgentFinding] = []
        var seen = Set<String>()
        for block in rankedTexts(in: transcript) {
            // Evidence must cite tools whose RESULTS the model had when it
            // wrote the text — the steps BEFORE this one. The step's own tools
            // run after the text is emitted, so citing them was evidence for
            // a claim that could not have seen it.
            let tools = toolNames(beforeStep: block.entry.stepIndex, transcript: transcript)
            for claim in claims(in: block.text) {
                let key = normalized(claim)
                guard seen.insert(key).inserted else { continue }
                findings.append(AgentFinding(
                    claim: claim,
                    evidence: evidence(step: block.entry.stepIndex, tools: tools),
                    confidence: statementConfidence
                ))
            }
        }
        return findings
    }

    // MARK: Open questions

    /// Questions the run raised in its own text and did not answer in the same
    /// line. Nothing is generated here either: a line that does not end with a
    /// question mark is not a question. Later steps win the cap, because the
    /// questions a run leaves open are the ones it raised after its reading.
    static func openQuestions(from transcript: [AgentTranscriptEntry]) -> [String] {
        var questions: [String] = []
        var seen = Set<String>()
        for block in rankedTexts(in: transcript, longestFirst: false) {
            for line in contentLines(in: block.text) {
                let trimmed = strippingBullet(line)
                guard trimmed.hasSuffix("?"), trimmed.count >= 8 else { continue }
                let sanitized = AgentHarness.sanitizeText(trimmed)
                guard seen.insert(normalized(sanitized)).inserted else { continue }
                questions.append(eliding(sanitized, to: maximumQuestionCharacters))
                if questions.count >= maximumOpenQuestions { return questions }
            }
        }
        return questions
    }

    // MARK: Partial summary

    /// The most substantive text the run produced, or nil when it produced
    /// none.
    ///
    /// A run that ends on a step, token, time or cost budget never reaches
    /// `complete`, and the loop's `finalText` is only the LAST non-empty turn's
    /// text. Keeping that alone discarded an analysis a run had already
    /// written, so a partial report picks the longest text instead (ties prefer
    /// the later one, which is the run's most considered version of it).
    static func bestEffortSummary(in transcript: [AgentTranscriptEntry]) -> String? {
        var best: String?
        for entry in transcript where entry.kind == .assistantText {
            let text = (entry.text ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { continue }
            if let current = best, text.count < current.count { continue }
            best = text
        }
        return best
    }

    // MARK: Parsing

    /// Cuts text to a limit and SAYS it was cut. A bare `prefix` ends a
    /// paragraph mid-sentence, which reads as a complete claim.
    static func eliding(_ text: String, to limit: Int) -> String {
        guard text.count > limit else { return text }
        return String(text.prefix(max(0, limit - 1))) + "…"
    }

    /// The run's assistant texts, most substantive first. `longestFirst` false
    /// ranks later steps first instead, which is what open questions want.
    private static func rankedTexts(
        in transcript: [AgentTranscriptEntry],
        longestFirst: Bool = true
    ) -> [(entry: AgentTranscriptEntry, text: String)] {
        transcript
            .filter { $0.kind == .assistantText }
            .compactMap { entry -> (entry: AgentTranscriptEntry, text: String)? in
                let text = entry.text ?? ""
                guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
                return (entry, text)
            }
            .sorted { lhs, rhs in
                if longestFirst, lhs.text.count != rhs.text.count {
                    return lhs.text.count > rhs.text.count
                }
                return lhs.entry.stepIndex > rhs.entry.stepIndex
            }
    }

    /// The claims in one assistant text block, in order.
    static func claims(in text: String) -> [String] {
        var claims: [String] = []
        for line in contentLines(in: text) {
            guard !line.hasPrefix("#") else { continue }
            let stripped = strippingBullet(line)
            // A question is not an assertion; it belongs in open questions.
            guard !stripped.hasSuffix("?") else { continue }
            guard stripped.count >= minimumClaimCharacters else { continue }
            let sanitized = AgentHarness.sanitizeText(stripped)
            claims.append(eliding(sanitized, to: maximumClaimCharacters))
        }
        return claims
    }

    /// The non-empty lines outside fenced code blocks, trimmed.
    private static func contentLines(in text: String) -> [String] {
        var lines: [String] = []
        var inFence = false
        for rawLine in text.components(separatedBy: .newlines) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.hasPrefix("```") || line.hasPrefix("~~~") {
                inFence.toggle()
                continue
            }
            guard !inFence, !line.isEmpty else { continue }
            lines.append(line)
        }
        return lines
    }

    /// Drops a list marker, so a bullet's text is the claim rather than the
    /// bullet itself. Only markers at the START of the line are removed.
    private static func strippingBullet(_ line: String) -> String {
        for marker in ["- ", "* ", "• "] where line.hasPrefix(marker) {
            return String(line.dropFirst(marker.count)).trimmingCharacters(in: .whitespaces)
        }
        // An ordered marker is one or more DIGITS then "." or ")": the old
        // form read exactly two scalars, so "10. claim" kept its marker and
        // read as prose while "1. claim" did not.
        let scalars = Array(line.prefix(6))
        var digitCount = 0
        while digitCount < scalars.count, scalars[digitCount].isNumber {
            digitCount += 1
        }
        if digitCount >= 1, digitCount + 1 < scalars.count,
           (scalars[digitCount] == "." || scalars[digitCount] == ")"),
           scalars[digitCount + 1].isWhitespace {
            return String(line.dropFirst(digitCount + 2)).trimmingCharacters(in: .whitespaces)
        }
        return line
    }

    /// Tools whose RESULTS the model had before the given step: a step's own
    /// tools run AFTER its text is emitted, so citing them was evidence for
    /// a claim that could not have seen it.
    private static func toolNames(beforeStep step: Int, transcript: [AgentTranscriptEntry]) -> [String] {
        var names: [String] = []
        var seen = Set<String>()
        for entry in transcript where entry.kind == .toolResult && entry.stepIndex < step {
            guard let name = entry.toolRecord?.name, seen.insert(name).inserted else { continue }
            names.append(name)
        }
        return names
    }

    private static func evidence(step: Int, tools: [String]) -> String {
        let position = "the run's own account, step \(step + 1)"
        guard !tools.isEmpty else { return position }
        return position + ", after " + tools.joined(separator: ", ")
    }

    private static func normalized(_ text: String) -> String {
        text.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ").lowercased()
    }
}
