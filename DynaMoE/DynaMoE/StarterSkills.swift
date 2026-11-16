//
//  StarterSkills.swift
//  DynaMoE
//
//  Phase 8 — Skills.
//
//  The one skill DynaMoE ships as content rather than code. It is embedded as a
//  string instead of an Xcode resource so installing it needs no project-file
//  plumbing, and `SkillsTests` asserts this constant matches the canonical copy
//  at `Skills/subq-decisions/SKILL.md` in the repository — edit that file, run
//  the test, and the drift is caught instead of shipping two versions.
//
//  Installing writes into the app-managed skills folder
//  (`~/Library/Application Support/DynaMoE/skills`). An existing copy is never
//  overwritten: a user edit is the truth once the file exists.
//

import Foundation

public nonisolated enum StarterSkills {
    public static let subqDecisionsName = "subq-decisions"
    public static let subqDecisionsFileName = "SKILL.md"

    /// Canonical text lives at `Skills/subq-decisions/SKILL.md`. Stored without
    /// the trailing newline; the installer appends one.
    public static let subqDecisionsMarkdown = #"""
---
name: subq-decisions
description: "Use when a judgment reduces to a structured question (a yes/no gate, routing, triage, classification, a guardrail, or ordered scoring) through the subq_decide tool, SubQ's decisions-only model. Also for 'subq', 'snap', or many cheap batch judgments. Not for drafting, summarizing or extraction."
always: true
user-invocable: true
metadata:
  version: 1.0.0
---

# SubQ Decisions (`subq_decide`)

`subq_decide` asks SubQ's decisions-only model, `subq-snap-preview`, one to
three typed questions about a `state` and returns probabilities and confidence.
It is not a chat model: it answers questions about text, objects, arrays and
(optionally) images, and it never writes prose.

- Tool: `subq_decide` (native; the API key never enters the transcript)
- Endpoint: `POST https://api.subq.ai/v1/systemone`
- Pricing: $0.05 per 1M input tokens; output is free
- Docs: https://docs.subq.ai/endpoints/decisions-api/

## When to use it

Use it for judgments, not for writing:

- **Yes/no gates** — is this a bug? is this change safe? is this a duplicate?
- **Routing** — which file, module, team or bucket does this belong to?
- **Triage** — severity, urgency, risk or quality as ordered levels.
- **Classification** — labeling items (bug vs feature vs question, sentiment, topic).
- **Guardrails** — does this output meet a bar or a policy?
- **Batch judgments** — the same structured question over many items.

Do not use it to draft, summarize, explain, extract fields, or answer questions
about the world. Those need a chat model.

## How to frame a decision

1. Name one to three questions. Each answer is keyed by its question name.
2. Compose `state` from the actual facts — paste the real text, values or
   objects. Never send your own summary or your own opinion; the judgment must
   be made from the same evidence a careful reviewer would see.
3. Choose the question type:
   - `noul` — binary gate. The answer is the probability of yes, 0 to 1. Give
     `criteria.true` and `criteria.false` so both outcomes are defined.
   - `choice` — one of 2-255 named options. `criteria` maps each option name to
     what it means. The answer is `choice`, per-option probabilities and
     confidence.
   - `score` — 2-10 ordered levels, low to high, as a `criteria` array. The
     answer is a probability-weighted score that may fall between levels, plus
     the legend and confidence.
4. Make the criteria non-overlapping. If two options could both be true, the
   answer is not a decision.

Example request:

```json
{
  "state": "Ticket: 'My card was charged twice for order #1234 and the app will not let me cancel.' Customer tier: free. Submitted 3 minutes ago.",
  "questions": {
    "needs_human": {
      "type": "noul",
      "instructions": "Does this ticket need a human agent before any automated reply?",
      "criteria": {
        "true": "Money movement, fraud, or account access is at stake",
        "false": "Routine question a bot can answer"
      }
    },
    "queue": {
      "type": "choice",
      "instructions": "Which team should own this ticket?",
      "criteria": {
        "billing": "Duplicate charges, refunds, payment failures",
        "tech_support": "App bugs, broken flows, error messages",
        "fraud": "Unauthorized charges or stolen credentials"
      }
    },
    "priority": {
      "type": "score",
      "instructions": "How fast must this be resolved?",
      "criteria": ["Within a week", "Within a day", "Within an hour"]
    }
  }
}
```

Call it as `subq_decide` with `state` and `questions` as above.

## How to act on the numbers

- **Decisive** — a probability of at least 0.8 (or at most 0.2 against) *and*
  confidence of at least 0.6: follow the answer without second-guessing it.
- **Mid-band** — a probability between 0.2 and 0.8, or confidence below 0.6:
  do not force it. Decide yourself with explicit reasoning, ask the user when
  the stakes are high, or send the item to a stronger model for review.
- **Say which way** — when the answer changes what you do, state the question,
  the number and the threshold you applied ("duplicate: 0.93, above 0.8, so
  ..."). Do not present a probability as certainty.

Do not re-ask the same question hoping for a different number. If the answer is
mid-band, the evidence is genuinely ambiguous — gather more evidence instead.

## Images

`images` accepts up to four PNG/JPEG/WebP files under 4 MiB each, by absolute
path. Use them for visual judgments (is this screenshot broken? does this
layout overflow?) and keep the question about the image concrete.

## When SubQ is unavailable

If `subq_decide` reports a missing key, a rejected key (401), insufficient
credits (402), a rate limit (429) or an outage (503):

1. Do not retry more than twice.
2. Decide yourself with explicit reasoning and say the decision was not
   SubQ-verified. Never invent numbers or quote a confidence you did not get.
3. Tell the user what to fix (Settings → Agent & Tools → Skills → SubQ
   Decisions, or a Cloud Provider for api.subq.ai).

## Cost

The API bills input tokens only; output tokens are always 0. Batch several
items into one call by making `state` an array and asking the questions about
the batch shape, rather than making one call per item where the judgment is the
same. Never send secrets or credentials in `state`.
"""#

    /// Writes the starter into `directory/subq-decisions/`, refusing to replace
    /// an existing SKILL.md. Returns the file it created.
    @discardableResult
    public static func installSubQDecisions(in directory: URL) throws -> URL {
        let skillDirectory = directory.appendingPathComponent(subqDecisionsName, isDirectory: true)
        let skillFile = skillDirectory.appendingPathComponent(subqDecisionsFileName, isDirectory: false)
        if FileManager.default.fileExists(atPath: skillFile.path) {
            throw StarterSkillError.alreadyInstalled(skillFile.path)
        }
        do {
            try FileManager.default.createDirectory(at: skillDirectory, withIntermediateDirectories: true)
        } catch {
            throw StarterSkillError.unwritable(skillDirectory.path, error.localizedDescription)
        }
        let payload = Data((subqDecisionsMarkdown + "\n").utf8)
        do {
            try payload.write(to: skillFile, options: .atomic)
        } catch {
            throw StarterSkillError.unwritable(skillFile.path, error.localizedDescription)
        }
        return skillFile
    }
}

public nonisolated enum StarterSkillError: LocalizedError {
    case alreadyInstalled(String)
    case unwritable(String, String)

    public var errorDescription: String? {
        switch self {
        case .alreadyInstalled(let path):
            return "A skill is already installed at \(path). Delete it first if you want the starter copy."
        case .unwritable(let path, let reason):
            return "Could not write \(path): \(reason)"
        }
    }
}
