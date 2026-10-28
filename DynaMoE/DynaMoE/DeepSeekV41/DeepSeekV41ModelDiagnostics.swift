//
//  DeepSeekV41ModelDiagnostics.swift
//  DynaMoE
//
//  Bridges the FFI model summary into the DeepSeek V4.1 checks, and prints what
//  the load path learned.
//
//  This is the only DeepSeek V4.1 file that touches the UniFFI types, kept
//  separate so the rest of the port stays Foundation-only and unit-testable
//  without a loaded engine.
//

import Foundation

public extension DeepSeekV41WeightCatalog {

    /// Builds the catalog from a loaded summary. `engine`-less: the summary is
    /// everything the checks need (name, dtype, shape), so this runs before any
    /// Metal buffer exists.
    static func from(summary: ModelSummary, layerCount: Int) -> DeepSeekV41WeightCatalog {
        let refs = summary.tensors.map { tensor in
            DeepSeekV41TensorRef(
                name: tensor.name,
                dtype: tensor.dtype,
                shape: Self.shape(from: tensor.shapeDisplay),
                shardIndex: Int(tensor.shardIndex),
                offset: Int(tensor.offsetStart)
            )
        }
        return DeepSeekV41WeightCatalog(named: refs, layerCount: layerCount)
    }

    /// Safetensors shapes arrive as `Debug` output (`[512, 7168]`), which is what
    /// the summary carries; parse the integers back out.
    static func shape(from display: String) -> [Int] {
        var values: [Int] = []
        var current = ""
        for character in display {
            if character.isNumber {
                current.append(character)
            } else if !current.isEmpty {
                values.append(Int(current) ?? 0)
                current = ""
            }
        }
        if !current.isEmpty { values.append(Int(current) ?? 0) }
        return values
    }
}

public enum DeepSeekV41LoadDiagnostics {

    /// Runs every check that does not need a GPU, and returns the lines the load
    /// log should carry. Never throws: a diagnostic must not break a load.
    public static func report(
        snapshotDir: URL,
        config: DeepSeekV41Config,
        summary: ModelSummary,
        contextLength: Int
    ) -> [String] {
        var lines: [String] = []

        if let preflight = DeepSeekV41Preflight.inspect(snapshotDir: snapshotDir) {
            lines.append("checkpoint: \(preflight.humanSummary)")
            if !preflight.isComplete {
                let names = preflight.issues.prefix(4).map(\.name).joined(separator: ", ")
                lines.append("download incomplete: missing \(names)")
            }
        }

        let plan: [DeepSeekV41LayerPlan]
        do {
            plan = try config.makeLayerPlan()
        } catch {
            lines.append("layer plan failed: \(error)")
            return lines
        }

        let compressed = plan.filter { $0.flavor == .compressed }.count
        let engramLayers = plan.compactMap(\.engramTable)
        lines.append(
            "layers: \(plan.count) (\(compressed) compressed, \(engramLayers.count) with Engram), "
                + "caches at \(config.compressedKVOwnerLayerIndices)"
        )

        let catalog = DeepSeekV41WeightCatalog.from(summary: summary, layerCount: config.numHiddenLayers)
        let problems = catalog.validate(config: config, plan: plan)
        if problems.isEmpty {
            lines.append("tensor catalog: every required tensor present, nothing unread")
        } else {
            lines.append("tensor catalog: \(problems.count) problem(s)")
            lines.append(contentsOf: problems.prefix(12).map { "  \($0)" })
        }

        let memory = DeepSeekV41MemoryPlan.make(
            config: config,
            plan: plan,
            catalog: catalog,
            contextLength: max(1, contextLength)
        )
        lines.append("memory: \(memory.describe(expertSlots: 32))")

        // The prompt dialect: which marker set the checkpoint's own template asked
        // for. A repacked checkpoint silently falls back, and a wrong marker set
        // tokenizes every turn into plain text, so the source is printed.
        let dialect = DeepSeekV41PromptDialect.resolve(snapshotDir: snapshotDir)
        lines.append("prompt dialect: \(dialect.source.rawValue)")

        let dspark = DeepSeekV41DSparkPlan.make(config: config)
        lines.append(
            "dspark: " + (dspark.isEnabled
                ? "block \(dspark.blockSize) over layers \(dspark.targetLayerIDs)"
                : "disabled")
        )

        if let vision = config.visionTowerPlan {
            lines.append(vision.describe())
        } else {
            lines.append("vision: no tower (text-only checkpoint)")
        }

        for warning in config.warnings(layerPlan: plan) {
            lines.append("note: \(warning)")
        }
        let unverified = DeepSeekV41Conformance.unverifiedSummary
        if !unverified.isEmpty {
            lines.append(
                "UNVERIFIED rules: \(unverified.joined(separator: ", ")). "
                    + "Run `python3 Tools/dsv41_probe.py tensors|engram|encoding <snapshot>` "
                    + "against the checkpoint before trusting numeric output."
            )
        }
        return lines
    }
}
