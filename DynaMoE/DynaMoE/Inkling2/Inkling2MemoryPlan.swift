//
//  Inkling2MemoryPlan.swift
//  DynaMoE
//
//  Where an Inkling 2 turn's bytes go.
//
//  Unlike DeepSeek-V4.1, whose latent cache is nearly free and whose Engram
//  tables dominate, this model's KV is a real constraint and its weights are
//  small enough to be mostly resident. The three unusual terms:
//
//    * Executor state is per VISIT, not per physical layer. Eight visits own a
//      growing full-attention cache; the other forty are bounded rings. The
//      naive "48 full caches" figure at 1M context is 206 GB, which is why the
//      plan refuses to price it that way.
//    * Four convolution states per visit, FP32, fixed per sequence:
//      48 visits * (2304 + 1024 + 1024 + 2304) channels * 3 history values * 4
//      bytes = 3,833,856 bytes.
//    * Loop messages are same-position, so a bounded chunk carries them and
//      they are released when the chunk commits: nothing here grows with the
//      prompt.
//
//  Getting this arithmetic wrong is how a 19.5 GB checkpoint appears to "fit"
//  and then pages itself to death, so it is computed once from the config and
//  the tensor catalog before anything is allocated.
//

import Foundation

nonisolated public struct Inkling2MemoryPlan: Sendable, Equatable {
    public let contextLength: Int
    /// Bytes per stored KV element: 2 for FP16/BF16, 1 for DynaMoE's FP8
    /// (INT8) storage, 4 for FP32. The precision enum lives outside this type
    /// because it is internal to the app module.
    public let kvElementBytes: Int
    /// K+V bytes one token costs across every full-attention visit: the
    /// growing part of the cache, which is what a context budget is spent on.
    public let bytesPerPosition: Int
    /// The DEVICE's copy of one sequence's K and V, per position: the
    /// executor stages the rows an attention dispatch reads into two buffers of
    /// its own, so the staging is a second copy of the growing cache at the same
    /// element width. It is not per visit -- one buffer holds the sequence --
    /// and a plan that ignored it priced a context the workspace could not
    /// allocate: at the checkpoint's 1M declaration the two buffers are 15.8 GB
    /// on their own, which is what an app turn tried to allocate on 16 GB.
    public let kvStagingBytesPerPosition: Int
    /// Growing full-attention caches: eight visits x K+V x head_dim x kv heads.
    public let fullAttentionCacheBytes: Int
    /// Bounded rings: forty visits x K+V x head_dim x kv heads x window.
    public let localRingBytes: Int
    /// Four FP32 convolution states per visit.
    public let convolutionStateBytes: Int
    /// One chunk's cross-pass messages: 4 per core layer per position.
    public let loopMessageBytes: Int
    /// Dense (non-expert) weights, from the catalog.
    public let denseWeightBytes: Int
    /// Routed experts' packed bytes for one resident slot, per expert.
    public let bytesPerExpertSlot: Int
    /// Vision tower bytes, counted separately from the text budget.
    public let visionWeightBytes: Int
    /// DSpark sidecar bytes, counted separately again.
    public let sidecarWeightBytes: Int

    /// Bytes of executor state at this context length.
    public var cacheBytes: Int { fullAttentionCacheBytes + localRingBytes + convolutionStateBytes }

    /// Bytes needed with `expertSlots` resident (layer, expert) pairs, on a
    /// DEVICE turn: the host caches, the executor's staging copy of the growing
    /// K/V, the loop messages, the dense weights and the arena.
    public func requiredBytes(expertSlots: Int, chunkPositions: Int = 1) -> Int {
        let slots = max(1, expertSlots)
        return cacheBytes
            + contextLength * kvStagingBytesPerPosition
            + loopMessageBytesPerPosition * max(1, chunkPositions)
            + denseWeightBytes
            + slots * bytesPerExpertSlot
    }

    /// One position's cross-pass message storage: four messages per core
    /// layer, each `hidden` FP32 values.
    public var loopMessageBytesPerPosition: Int { loopMessageBytes }

    /// Expert slots this machine can hold resident, with the plan's own fixed
    /// terms taken out of `fraction` of RAM. Routed experts are streamed, so
    /// the arena is a cache: the number decides how often a token re-reads an
    /// expert it already had.
    public func affordableExpertSlots(physicalGB: Double, fraction: Double = 0.6) -> Int {
        guard bytesPerExpertSlot > 0, physicalGB > 0, fraction > 0 else { return 1 }
        let budget = physicalGB * fraction * 1_073_741_824
        let fixed = Double(cacheBytes + denseWeightBytes)
        let slots = (budget - fixed) / Double(bytesPerExpertSlot)
        guard slots.isFinite, slots >= 1 else { return 1 }
        return Int(min(slots, 65_536))
    }

    /// Positions the KV cache may hold under `budgetGB`, i.e. the cap the
    /// forward is built with so a long conversation refuses by name instead of
    /// growing until the process is killed.
    ///
    /// Only the growing part is priced against the budget: the rings are
    /// bounded by the window, so they are a fixed term, exactly as
    /// `KVCacheProvisioning` treats a sliding ring. One position costs one
    /// record per full-attention VISIT (from the schedule, since a full layer
    /// in the loop core appears once per pass).
    public func affordableContextLength(budgetGB: Double) -> Int {
        guard bytesPerPosition > 0, budgetGB.isFinite else { return contextLength }
        let budget = max(0.0, budgetGB * 1_073_741_824) - Double(localRingBytes)
        guard budget > 0 else { return 1 }
        // Both copies are paid for: the sequence's own rows and the executor's
        // staging of them. The rings came out above; the staging grows with the
        // context exactly as the rows do.
        let perPosition = max(1, bytesPerPosition + kvStagingBytesPerPosition)
        return max(1, min(contextLength, Int(budget / Double(perPosition))))
    }

    /// One-line load-log summary.
    public func describe(expertSlots: Int) -> String {
        func gb(_ bytes: Int) -> String {
            String(format: "%.2f GB", Double(bytes) / 1_073_741_824.0)
        }
        return [
            "dense \(gb(denseWeightBytes))",
            "experts \(expertSlots)×\(gb(bytesPerExpertSlot))",
            "kv \(gb(cacheBytes)) @\(contextLength) tok (\(kvElementBytes)-byte)",
            "conv \(gb(convolutionStateBytes))",
            "loop msgs \(gb(loopMessageBytes))/position",
            "total \(gb(requiredBytes(expertSlots: expertSlots)))"
        ].joined(separator: ", ")
    }

    // MARK: Building

    /// The published checkpoint's layout, as a testable constant rather than a
    /// number quoted in prose: 20 layers of NVFP4 experts and 4 of MXFP8.
    public static func expertLayout(forLayer layer: Int, config: Inkling2Config) -> Inkling2ExpertQuantLayout {
        // The checkpoint's own split (layers 0...19 NVFP4, 20...23 MXFP8) is
        // what `hf_quant_config.json` declares and what every stored shape
        // agrees with. A checkpoint that moved it would be caught by the
        // catalog's per-layer stride checks, so this only has to be the
        // default a plan is built with.
        layer < max(0, config.numHiddenLayers - 4) ? .nvfp4 : .mxfp8
    }

    public static func make(
        config: Inkling2Config,
        catalog: Inkling2WeightCatalog?,
        contextLength: Int,
        kvElementBytes: Int = 2,
        chunkPositions: Int = 1
    ) -> Inkling2MemoryPlan {
        let context = max(0, contextLength)
        // One attention visit's K+V per token: two tensors of `kvHeads * headDim`.
        // The FP8 mode is DynaMoE's own storage contract (symmetric INT8 plus an
        // FP16 scale), so its element is one byte and its scales are priced
        // separately, as the plan's table notes.
        let recordPerToken = 2 * config.numKeyValueHeads * config.headDim * max(1, kvElementBytes)
        // Full-attention visits: a FULL physical layer appears once per pass it
        // participates in, so the count comes from the schedule, not from
        // `fullAttentionLayers.count`. The published schedule has one full layer
        // per six and the core's three passes repeat layers 6...17: eight visits.
        let plan = (try? config.makeVisitPlan()) ?? []
        let fullVisitCount = plan.isEmpty
            ? config.fullAttentionLayers.count
            : plan.filter(\.isFullAttention).count
        let full = fullVisitCount * context * recordPerToken
        // Rings belong to the LOCAL visits: a full-attention visit's cache is
        // unbounded, so its window ring would be dead storage (the reference's
        // window cache on every layer is an implementation detail, not a byte
        // this plan needs to reserve). At 32,768 tokens this is what makes the
        // hybrid total 1.242 GB rather than 1.275.
        let localVisitCount = plan.isEmpty
            ? config.executedVisitCount - config.fullAttentionLayers.count
            : plan.filter { !$0.isFullAttention }.count
        let rings = localVisitCount * min(context, config.slidingWindowSize) * recordPerToken
        let visitCount = plan.isEmpty ? config.executedVisitCount : plan.count
        // Four states per visit, three history values each, FP32:
        //   K and V are `kvHeads * headDim` wide, the post-attention and post-MLP
        //   convolutions are `hidden` wide.
        let channels = 2 * (config.numKeyValueHeads * config.headDim) + 2 * config.hiddenSize
        let convolution = visitCount * channels * 3 * MemoryLayout<Float>.stride
        let messages = config.coreLayerCount * 4 * config.hiddenSize * MemoryLayout<Float>.stride

        var dense = 0
        var expertSlot = 0
        var vision = 0
        if let catalog {
            for slot in catalog.slots {
                let bytes = Inkling2WeightCatalog.byteCount(slot.ref)
                if slot.scope == .vision {
                    vision += bytes
                } else if slot.scope == .sidecar {
                    // The sidecar is its own budget; it is not part of a text turn.
                    continue
                } else if slot.role.isExpertBank {
                    expertSlot += bytes / max(1, slot.ref.shape.first ?? 1)
                } else if slot.role.isExpertCompanion {
                    continue
                } else {
                    dense += bytes
                }
            }
        }

        return Inkling2MemoryPlan(
            contextLength: context,
            kvElementBytes: max(1, kvElementBytes),
            bytesPerPosition: fullVisitCount * recordPerToken,
            kvStagingBytesPerPosition: 2 * config.numKeyValueHeads * config.headDim
                * max(1, kvElementBytes),
            fullAttentionCacheBytes: full,
            localRingBytes: rings,
            convolutionStateBytes: convolution,
            loopMessageBytes: messages,
            denseWeightBytes: dense,
            bytesPerExpertSlot: expertSlot,
            visionWeightBytes: vision,
            sidecarWeightBytes: 0
        )
    }
}

// MARK: - Arena capacity

/// How many expert slots one packed layout's arena gets.
///
/// The shipped answer is one POSITION's working set -- the experts a position's
/// schedule revisits -- because that is what measurement paid for: an arena
/// sized to the plan's fraction of RAM (6.53 GB across both layouts) ran a
/// position in 102-118 s with the mixture at 672-857 ms/visit, against
/// 0.43-1.35 s and 5.8-22.9 ms/visit at 1.71 GB. Wired arena memory displaces
/// the mmap's own pages, so over-sizing the arena is slower, not merely
/// wasteful.
///
/// `override` is the measurement seam (`dynamoe_inkling2_arena_slots`): a
/// positive slot count replaces the working-set cap without a rebuild, and is
/// still held under the device ceiling, because an arena past it cannot be
/// allocated at all. Zero means "no override", which is what ships.
///
/// `minimum` is the floor a working arena cannot go under: one visit routes
/// `topK` experts and fills their slots BEFORE it encodes, so an arena with
/// fewer slots than that hands two routed experts the SAME slot index, and the
/// dispatch then reads the later fill's bytes for both. That is a wrong number
/// with no error -- measured on the shipping path, a 1-slot MXFP8 arena left
/// layers 20-23 reading one expert's bytes eight ways, which is why the turn
/// answered a template token block instead of `Paris`'s continuation. The floor
/// is `numExpertsPerToken`; a budget that cannot afford it is a refusal, not a
/// smaller arena.
nonisolated public enum Inkling2ArenaCapacity {
    /// `UserDefaults` key for the override, read on the turn's own task.
    public static let overrideDefaultsKey = "dynamoe_inkling2_arena_slots"

    public static func slots(
        workingSet: Int, planned: Int, ceilingSlots: Int, override: Int = 0, minimum: Int = 1
    ) -> Int {
        let ceiling = max(1, ceilingSlots)
        let floor = max(1, min(minimum, ceiling))
        if override > 0 { return max(floor, min(override, ceiling)) }
        return max(floor, min(workingSet, planned, ceiling))
    }
}

// MARK: - Preflight

/// A prompt this runtime cannot provision, refused BEFORE anything is
/// allocated. The KV budget is the authority; the checkpoint's 1M declaration
/// is only an architectural ceiling.
nonisolated public enum Inkling2PreflightError: Error, Equatable, CustomStringConvertible {
    case promptTooLong(promptTokens: Int, availableTokens: Int)
    case checkpointIncomplete(String)
    case unsupportedVariant(String)

    public var description: String {
        switch self {
        case .promptTooLong(let prompt, let available):
            return "Inkling 2 cannot provision a \(prompt)-token prompt: only \(available) positions fit the memory budget"
        case .checkpointIncomplete(let detail):
            return "Inkling 2 checkpoint is incomplete: \(detail)"
        case .unsupportedVariant(let detail):
            return "Inkling 2 checkpoint is a variant this runtime does not implement: \(detail)"
        }
    }
}

nonisolated public struct Inkling2Preflight: Sendable, Equatable {
    /// Positions the plan can address at the current context length.
    public let availablePromptTokens: Int
    /// Positions left for the reply once the prompt is placed.
    public let replyBudget: Int

    /// Checks a prompt against the plan, leaving room for a reply. `contextLength`
    /// is what the cache was provisioned for; a prompt that does not fit it plus
    /// at least one reply position is refused by name rather than truncated.
    public static func check(
        promptTokens: Int,
        contextLength: Int,
        requestedReplyTokens: Int
    ) throws -> Inkling2Preflight {
        let available = max(0, contextLength)
        let reply = min(max(0, requestedReplyTokens), max(0, available - promptTokens))
        guard promptTokens >= 0, promptTokens + 1 <= available else {
            throw Inkling2PreflightError.promptTooLong(
                promptTokens: promptTokens,
                availableTokens: available
            )
        }
        return Inkling2Preflight(availablePromptTokens: available, replyBudget: reply)
    }

    /// Shard completeness for a snapshot, read from each shard's own header
    /// rather than the index: a truncated shard is present and has a plausible
    /// size, so only the offsets it declares can tell a finished file from a
    /// half-written one.
    ///
    /// Returns nil when there is no index to check (a packed repack, say).
    public static func shardIssues(snapshotDir: URL) -> [String]? {
        let indexURL = snapshotDir.appendingPathComponent("model.safetensors.index.json")
        guard let data = try? Data(contentsOf: indexURL),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let weightMap = object["weight_map"] as? [String: String] else {
            return nil
        }
        var issues: [String] = []
        for name in Set(weightMap.values).sorted() {
            let url = snapshotDir.appendingPathComponent(name)
            guard let actual = Self.fileSizeFollowingSymlinks(at: url) else {
                issues.append("\(name): missing")
                continue
            }
            guard let expected = Self.expectedShardBytes(at: url) else {
                issues.append("\(name): header unreadable, \(actual) bytes")
                continue
            }
            if Int64(expected) != actual {
                issues.append("\(name): \(actual) of \(expected) bytes")
            }
        }
        return issues
    }

    /// Whether the snapshot really ships the vision tower, decided by FILE
    /// presence rather than the config's promise: the composer's attachment
    /// button must stay closed for a half-finished download. Every tower tensor
    /// the catalog requires has to resolve to a shard that exists.
    public static func visionTowerIsPresent(snapshotDir: URL) -> Bool {
        let indexURL = snapshotDir.appendingPathComponent("model.safetensors.index.json")
        guard let data = try? Data(contentsOf: indexURL),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let weightMap = object["weight_map"] as? [String: String] else {
            return false
        }
        let names = Set(weightMap.keys)
        // The tower's anchors: one per stage the catalog cannot run without.
        let required: [String] = [
            "model.visual.backbone.embeddings.patch_embedding.weight",
            "model.visual.backbone.embeddings.class_embedding",
            "model.visual.backbone.embeddings.position_embedding",
            "model.visual.backbone.ln_pre.weight",
            "model.visual.backbone.layers.0.self_attn.q_proj.weight",
            "model.visual.backbone.layers.0.mlp.c_fc.weight",
            "model.visual.proj_in.weight",
            "model.visual.proj_out.weight"
        ]
        for name in required {
            guard let shard = weightMap[name] ?? firstMatch(of: name, in: names) else { return false }
            let shardURL = snapshotDir.appendingPathComponent(shard)
            guard FileManager.default.fileExists(atPath: shardURL.path) else { return false }
        }
        return true
    }

    static func firstMatch(of name: String, in names: Set<String>) -> String? {
        // A block's tensors are named `layers.N.`, so an anchor on block 0 is an
        // exact name; this only exists so a renamed block prefix fails closed.
        names.first { $0.hasPrefix(name.split(separator: ".").dropLast().joined(separator: ".")) }
    }

    /// Bytes of the file a shard path resolves to, following symlinks: a
    /// Hugging Face snapshot is a directory of links into `blobs/`, so an
    /// attribute read that does not resolve reports a link's length instead.
    public static func fileSizeFollowingSymlinks(at url: URL) -> Int64? {
        #if canImport(Darwin)
        var info = stat()
        guard stat(url.path, &info) == 0 else { return nil }
        return Int64(info.st_size)
        #else
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = attributes[.size] as? NSNumber else { return nil }
        return size.int64Value
        #endif
    }

    /// Bytes a shard must have according to the safetensors header it starts
    /// with: the 8-byte little-endian header length, the JSON header, then the
    /// largest `data_offsets` end any tensor declares. Nil when the header
    /// cannot be read, which is reported separately and must not veto a shard.
    public static func expectedShardBytes(at url: URL) -> Int? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        guard let prefix = try? handle.read(upToCount: 8), prefix.count == 8 else { return nil }
        let headerLength = prefix.withUnsafeBytes { raw -> UInt64 in
            UInt64(littleEndian: raw.loadUnaligned(as: UInt64.self))
        }
        // A header is a few MB at most; anything past this is not safetensors
        // and must not become an allocation.
        guard headerLength > 0, headerLength <= 64 * 1_024 * 1_024 else { return nil }
        guard let headerData = try? handle.read(upToCount: Int(headerLength)),
              headerData.count == Int(headerLength),
              let header = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            return nil
        }
        var dataEnd: Int64 = 0
        for (key, value) in header where key != "__metadata__" {
            guard let tensor = value as? [String: Any],
                  let offsets = tensor["data_offsets"] as? [NSNumber],
                  offsets.count == 2 else { continue }
            dataEnd = max(dataEnd, offsets[1].int64Value)
        }
        guard dataEnd > 0 else { return nil }
        return Int(clamping: 8 + Int64(headerLength) + dataEnd)
    }
}
