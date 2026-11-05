//
//  ModelServingID.swift
//  DynaMoE
//
//  Phase: the server's model identity, shared by both protocols (slice #92)
//
//  Every route served the loaded checkpoint's snapshot directory name as the model
//  id, i.e. a commit hash, while the name a client could actually use (the
//  repository's own) was recovered for Anthropic's `display_name` and nowhere else.
//  The two are the wrong way round: a hub snapshot hash changes whenever the same
//  model is re-resolved, which is the opposite of what a client stores, and no
//  human recognises it in a log row. A display name is not a substitute either,
//  because it carries spaces.
//

import Foundation

/// The served identity of a loaded checkpoint, derived from its snapshot path.
///
/// The path is what every checkpoint has and what both facts come from:
/// `.../hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/<hash>` names
/// the repository, and the repository name is stable across a re-resolve where the
/// hash is not. A directory that is not a hub snapshot has no repository name to
/// recover, so its own name is used, which is what the app scanned in the first
/// place. Pure string logic, so it is pinned without a model.
nonisolated enum ModelServingID {

    /// The characters an id may carry. An id is a path segment on
    /// `GET /v1/models/:model_id`, so a space (which the display name has) or a
    /// slash (which the repository id has) would make the one route that reads an
    /// id by its own value unreachable without escaping.
    private static let allowed = Set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")

    /// `author/name` when the path names a hub repository, else the directory's own
    /// name. Nil when there is nothing to read. This is the human-facing spelling,
    /// not the id.
    static func repository(from snapshotPath: String?) -> String? {
        guard let snapshotPath, !snapshotPath.isEmpty else { return nil }
        let components = URL(fileURLWithPath: snapshotPath).pathComponents
        guard let repository = components.last(where: { $0.hasPrefix("models--") }) else {
            let directory = URL(fileURLWithPath: snapshotPath).lastPathComponent
            return directory.isEmpty ? nil : directory
        }
        let stripped = String(repository.dropFirst("models--".count))
        let parts = stripped.components(separatedBy: "--")
        guard let author = parts.first, parts.count > 1 else {
            return stripped.isEmpty ? nil : stripped
        }
        return author + "/" + parts.dropFirst().joined(separator: "--")
    }

    /// The repository name without its author, i.e. the part an id is made of. The
    /// author is dropped because the id has to survive as one path segment.
    static func shortName(from snapshotPath: String?) -> String? {
        guard let snapshotPath, !snapshotPath.isEmpty else { return nil }
        let components = URL(fileURLWithPath: snapshotPath).pathComponents
        guard let repository = components.last(where: { $0.hasPrefix("models--") }) else {
            let directory = URL(fileURLWithPath: snapshotPath).lastPathComponent
            return directory.isEmpty ? nil : directory
        }
        let stripped = String(repository.dropFirst("models--".count))
        let parts = stripped.components(separatedBy: "--")
        guard parts.count > 1 else { return stripped.isEmpty ? nil : stripped }
        let name = parts.dropFirst().joined(separator: "--")
        return name.isEmpty ? nil : name
    }

    /// A path-segment-safe id: anything outside the allowed set becomes a single
    /// dash, and no id starts or ends with a separator. `.` and `..` are trimmed
    /// away for the same reason they cannot be a path segment, so a checkpoint
    /// directory named `.` cannot address its parent.
    static func slug(_ raw: String) -> String {
        var out = ""
        var lastWasDash = false
        for scalar in raw.unicodeScalars {
            let character = Character(String(scalar))
            if allowed.contains(character) {
                out.append(character)
                lastWasDash = (character == "-")
            } else if !lastWasDash {
                out.append("-")
                lastWasDash = true
            }
        }
        return out.trimmingCharacters(in: CharacterSet(charactersIn: "-._"))
    }

    /// The one id both protocols serve for this checkpoint, and the identity a
    /// session key hashes. `fallback` is what the caller already holds, so a
    /// checkpoint with an unreadable path degrades to it rather than to nothing.
    static func servedID(from snapshotPath: String?, fallback: String) -> String {
        let raw = shortName(from: snapshotPath) ?? fallback
        let slugged = slug(raw)
        return slugged.isEmpty ? "model" : slugged
    }
}
