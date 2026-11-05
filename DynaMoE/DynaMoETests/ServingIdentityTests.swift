//
//  ServingIdentityTests.swift
//  DynaMoE
//
//  Pure-logic coverage for the id both protocols serve (#92). Every route served
//  the snapshot directory's own name, which for a hub checkpoint is a commit hash:
//  not a path segment, not something a human reads, and different every time the
//  same model is re-resolved. The id is now the repository's short name, and the
//  facts pinned here are the ones a client depends on: it contains no space and no
//  slash, it survives as `/v1/models/:model_id`, it degrades to the directory's own
//  name off the hub, and the previous id is still answered as an alias.
//

import XCTest
@testable import DynaMoE

final class ModelServingIDTests: XCTestCase {

    private let hubPath =
        "/Users/x/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017"
        + "/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"

    /// The reason the slice exists: the served id used to be the snapshot
    /// directory's own name, a commit hash, while the name a client could use was
    /// already being recovered for `display_name` and nowhere else.
    func testTheServedIdIsTheRepositoryNameAndNotTheSnapshotHash() {
        XCTAssertEqual(ModelServingID.servedID(from: hubPath, fallback: "hash"), "DeepSeek-V4.1-Flash-1017")
        XCTAssertEqual(
            URL(fileURLWithPath: hubPath).lastPathComponent, "1bae0e1088993b9b82de08f59de5be5a34e42170",
            "the path's own last component is the hash the old id served"
        )
    }

    /// An id is a path segment on the retrieve route, so the two characters that
    /// make a name unusable there are the two the checkpoint's names carry: the
    /// display name's spaces and the repository id's slash.
    func testTheIdIsUsableAsAPathSegment() {
        let id = ModelServingID.servedID(from: hubPath, fallback: "hash")
        XCTAssertFalse(id.contains(" "))
        XCTAssertFalse(id.contains("/"))
        XCTAssertEqual(id, id.trimmingCharacters(in: CharacterSet(charactersIn: "-._")))
    }

    /// The human spelling keeps the author, because this protocol has a field for
    /// it (`display_name`) while the id cannot carry a slash.
    func testTheRepositorySpellingKeepsTheAuthor() {
        XCTAssertEqual(
            ModelServingID.repository(from: hubPath), "deepseek-ai/DeepSeek-V4.1-Flash-1017"
        )
        XCTAssertEqual(ModelServingID.shortName(from: hubPath), "DeepSeek-V4.1-Flash-1017")
    }

    /// A checkpoint that is not a hub snapshot has no repository name to recover,
    /// so its own directory name is the id. Nothing may degrade to an empty id or
    /// to a path fragment.
    func testACheckpointOutsideTheHubUsesItsOwnDirectoryName() {
        XCTAssertEqual(
            ModelServingID.servedID(from: "/Models/my_own checkpoint", fallback: "hash"),
            "my_own-checkpoint"
        )
        XCTAssertEqual(
            ModelServingID.repository(from: "/Models/my_own checkpoint"), "my_own checkpoint",
            "the display name is the directory's name, undecorated"
        )
        XCTAssertEqual(ModelServingID.servedID(from: nil, fallback: "deepseek"), "deepseek")
        XCTAssertEqual(ModelServingID.servedID(from: "", fallback: "deepseek"), "deepseek")
    }

    /// `.` and `..` are not path segments, so a directory named either one cannot
    /// address itself or its parent, and an id that is entirely separators still
    /// has to be something.
    func testAnIdIsNeverAPathFragmentAndNeverEmpty() {
        XCTAssertEqual(ModelServingID.slug(".."), "")
        XCTAssertEqual(ModelServingID.slug("."), "")
        XCTAssertEqual(ModelServingID.slug("../../etc/passwd"), "etc-passwd")
        XCTAssertEqual(ModelServingID.slug("%2e%2e"), "2e-2e")
        XCTAssertEqual(ModelServingID.slug("!!!"), "")
        XCTAssertEqual(ModelServingID.servedID(from: "/Models/..", fallback: ""), "model")
    }

    /// Runs collapse, so a name with several unusable characters in a row produces
    /// one separator rather than a run of them.
    func testSeparatorsCollapseAndTheIdKeepsItsOwnPunctuation() {
        XCTAssertEqual(ModelServingID.slug("Qwen 2.5  Coder"), "Qwen-2.5-Coder")
        XCTAssertEqual(ModelServingID.slug("a//b"), "a-b")
        XCTAssertEqual(ModelServingID.slug("gpt-oss_20b"), "gpt-oss_20b")
        XCTAssertEqual(ModelServingID.slug(" DeepSeek-V4.1-Flash-1017 "), "DeepSeek-V4.1-Flash-1017")
    }

    /// The `models--author--name` split: an author is dropped from the id and kept
    /// in the repository spelling, and a repository directory with no author is
    /// still a name rather than nothing.
    func testTheHubDirectoryShapeIsParsedBothWays() {
        let noAuthor = "/hub/models--Llama-3.1-8B/snapshots/abc"
        XCTAssertEqual(ModelServingID.shortName(from: noAuthor), "Llama-3.1-8B")
        XCTAssertEqual(ModelServingID.repository(from: noAuthor), "Llama-3.1-8B")

        // The directory is an ancestor, not the last component: a nested layout
        // still reads it, which is why the scan looks for the prefix rather than
        // assuming a depth.
        let nested = "/hub/models--meta-llama--Llama-3.1-8B/snapshots/abc/extra"
        XCTAssertEqual(ModelServingID.shortName(from: nested), "Llama-3.1-8B")
    }
}

final class ServedModelAliasTests: XCTestCase {

    private func info(displayName: String, path: String?) -> CompletionEngine.LoadedModelInfo {
        CompletionEngine.LoadedModelInfo(
            displayName: displayName,
            path: path,
            supportsThinking: true,
            supportsThinkingLevels: true,
            supportsVision: false,
            visionCost: nil,
            weightsSizeGB: 475.25
        )
    }

    /// The retrieved id is the advertised id, the previous id is still answered,
    /// and a name this server never served stays a 404 rather than resolving to the
    /// loaded model.
    func testTheRetrieveRouteAnswersTheAdvertisedIdAndTheOneItReplaced() {
        let loaded = info(
            displayName: "1bae0e1088993b9b82de08f59de5be5a34e42170",
            path: "/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"
        )
        XCTAssertTrue(ServerController.serves(modelID: "DeepSeek-V4.1-Flash-1017", info: loaded))
        XCTAssertTrue(
            ServerController.serves(
                modelID: "1bae0e1088993b9b82de08f59de5be5a34e42170", info: loaded
            ),
            "the snapshot name was the served id before this slice, so a client holding it keeps working"
        )
        XCTAssertFalse(ServerController.serves(modelID: "not-a-served-model", info: loaded))
        XCTAssertFalse(ServerController.serves(modelID: "", info: loaded))
    }

    /// The OpenAI list entry advertises the id a client then addresses, and it is
    /// the served id rather than the display name the row is keyed on.
    func testTheListEntryAdvertisesTheServedId() {
        let loaded = info(
            displayName: "1bae0e1088993b9b82de08f59de5be5a34e42170",
            path: "/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"
        )
        let entry = OpenAIModelList.entry(
            for: loaded, contextLength: 163_840, maxPromptTokens: 262_144, created: 1_700_000_000
        )
        XCTAssertEqual(entry.id, "DeepSeek-V4.1-Flash-1017")
        XCTAssertEqual(entry.id, loaded.servedID)
    }

    /// The two protocols name the same checkpoint differently on purpose: the id is
    /// the path segment, the display name is the human spelling. They must not
    /// collapse back into one string.
    func testTheAnthropicIdAndDisplayNameAreDifferentSpellingsOfOneCheckpoint() {
        let loaded = info(
            displayName: "1bae0e1088993b9b82de08f59de5be5a34e42170",
            path: "/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"
        )
        let displayName = AnthropicAdapter.displayName(servedId: loaded.servedID, snapshotPath: loaded.path)
        XCTAssertEqual(displayName, "deepseek-ai/DeepSeek-V4.1-Flash-1017")
        XCTAssertNotEqual(displayName, loaded.servedID)
        XCTAssertFalse(loaded.servedID.contains("/"), "the id is one path segment; the display name is not")
    }
}
