import SwiftUI
import UniformTypeIdentifiers
import Metal
import Foundation
import AppKit
import Accelerate

/// Central gate for chat-tab diagnostic logging. Off by default; enable individual flags from
/// the Debug pane (or `defaults write`) to re-enable the corresponding console output.
enum DebugFlags {
    /// Per-token `[Autoregressive] Step N: nextToken=...` console lines during generation.
    static var tokenStepLogging: Bool {
        UserDefaults.standard.object(forKey: "dynamoe_debug_token_logging") != nil
            ? UserDefaults.standard.bool(forKey: "dynamoe_debug_token_logging")
            : false
    }
    static var tokenLogging: Bool { tokenStepLogging }
}

struct ExpertRoutingBadgeView: View {
    let rank: Int
    let expertId: Int
    let weight: Float

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("#\(rank + 1)")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                Text("Expert #\(expertId)")
                    .font(.system(.caption, design: .monospaced))
                    .fontWeight(.bold)
                    .foregroundColor(.purple)
                Spacer()
                Text(String(format: "%.1f%%", weight * 100.0))
                    .font(.system(.caption, design: .monospaced))
                    .fontWeight(.semibold)
            }
            
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(Color.purple.opacity(0.15))
                        .frame(height: 6)
                    Capsule()
                        .fill(LinearGradient(colors: [.purple, .blue], startPoint: .leading, endPoint: .trailing))
                        .frame(width: max(4, geo.size.width * CGFloat(weight)), height: 6)
                }
            }
            .frame(height: 6)
        }
        .padding(8)
        .background(Color(NSColor.controlBackgroundColor))
        .cornerRadius(8)
    }
}
#if canImport(Darwin)
import Darwin

func getProcessResidentMemoryGB() -> Double {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / 4)
    let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    if kerr == KERN_SUCCESS {
        return Double(info.resident_size) / (1024.0 * 1024.0 * 1024.0)
    }
    return 0.0
}

// phys_footprint: dirty + compressed + IOKit pages, EXCLUDING clean file-backed
// pages such as the zero-copy mmap'd weight shards. This is the number Activity
// Monitor's "Memory" column and Xcode's memory gauge actually display — it is
// NOT the resident size above, which counts every resident page the task maps.
func getActivityMonitorFootprintGB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / 4)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    if kr == KERN_SUCCESS {
        return Double(info.phys_footprint) / (1024.0 * 1024.0 * 1024.0)
    }
    return 0.0
}
#else
func getProcessResidentMemoryGB() -> Double {
    return 0.0
}

func getActivityMonitorFootprintGB() -> Double {
    return 0.0
}
#endif

/// Counts FlashMoE backbone pins so a *stacking* pin is attributable from the
/// console alone. Each FlashMoE load preads `model_weights.bin` into a
/// ~5.5 GB anonymous GPU buffer; a field report showed ~20 GB resident on a
/// 16 GB machine, i.e. several pins' worth, and the pin has no release log of
/// its own. Three things are logged per pin: its ordinal, how many *previous*
/// pins are still alive (a weak reference proves whether our code still holds
/// them or Metal kept the allocation after we dropped it), and the process
/// footprint before/after the pread.
enum BackbonePinTelemetry {
    final class WeakPin {
        weak var buffer: AnyObject?

        init(buffer: AnyObject) {
            self.buffer = buffer
        }

        var isAlive: Bool { buffer != nil }
    }

    private static let lock = NSLock()
    private static var pins: [WeakPin] = []

    /// Records a pin and reports its ordinal plus the *distinct* live pin
    /// buffers. Distinct matters because the pool reuses one buffer across
    /// loads, so a per-pin sum would count the same allocation once per model
    /// (it read `2 = 9.72 GB` for a single 5.53 GB buffer).
    static func record(buffer: AnyObject) -> (ordinal: Int, liveDistinct: Int, liveDistinctGB: Double) {
        lock.lock()
        defer { lock.unlock() }
        var seen = Set<ObjectIdentifier>()
        var liveDistinct = 0
        var liveDistinctGB = 0.0
        for pin in pins {
            guard let live = pin.buffer else { continue }
            let id = ObjectIdentifier(live)
            guard seen.insert(id).inserted else { continue }
            liveDistinct += 1
            liveDistinctGB += Double((live as? MTLBuffer)?.length ?? 0) / 1_073_741_824
        }
        pins.append(WeakPin(buffer: buffer))
        return (pins.count, liveDistinct, liveDistinctGB)
    }
}

/// Holds the single FlashMoE backbone pin buffer.
///
/// The pin is a multi-GB anonymous allocation pread once per load so per-token
/// backbone reads never re-fault the file-backed mmap. Allocating a fresh one
/// per load stacked several resident at once (measured live: 3 pins = 15.25 GB
/// on a 16 GB machine, one of them from an earlier model's different backbone
/// size — the ~20 GB footprint). The pool keeps one buffer, grows only when a
/// model's backbone needs more room, and is released when the model is torn
/// down, so pin memory is bounded by the largest backbone loaded in the session
/// rather than by the number of loads.
final class BackbonePinPool: @unchecked Sendable {
    static let shared = BackbonePinPool()

    private let lock = NSLock()
    private var buffer: MTLBuffer?

    /// A buffer of at least `length` bytes to pread the backbone into, reusing
    /// the retained one when it is already large enough.
    func acquire(device: MTLDevice, length: Int) -> (buffer: MTLBuffer, reused: Bool)? {
        lock.lock()
        defer { lock.unlock() }
        if let existing = buffer, existing.length >= length {
            return (existing, true)
        }
        guard let fresh = device.makeBuffer(length: length, options: .storageModeShared) else { return nil }
        buffer = fresh
        return (fresh, false)
    }

    func release() {
        lock.lock()
        buffer = nil
        lock.unlock()
    }
}


extension TensorMetadata: Identifiable {
    public var id: String { name }
}

struct ContentView: View {
    @Environment(\.scenePhase) private var scenePhase
    @State private var engine: DynaMoeEngine? = nil
    @State private var tokenizer: DynaMoeTokenizer? = nil
    @State private var summary: ModelSummary? = nil
    @State private var metalStatus: String = "GPU Status: Waiting for weights..."
    
    // Multi-Shard Metal Buffers
    @State private var shardBuffers: [UInt32: MTLBuffer] = [:]
    
    // File Importers
    @State private var isWeightImporterPresented: Bool = false
    @State private var isTokenizerImporterPresented: Bool = false
    
    // Tokenizer Playground State
    @State private var promptInput: String = "Hello DynaMoE, routing tokens to experts..."
    @State private var tokenIDsOutput: String = "Load a tokenizer.json file to tokenize text"
    
    // Autoregressive Text Generation State
    @State private var temperature: Float = 0.0
    @State private var topP: Float = 0.9
    @State private var minP: Float = 0.05
    @State private var topK: Int = 50
    @State private var repetitionPenalty: Float = 1.1
    @State private var presencePenalty: Float = 0.0
    @AppStorage("dynamoe_max_tokens") private var maxNewTokens: Int = 8192
    @State private var activeProfile: ModelProfileType = .coder
    @State private var isGeneratingText: Bool = false
    /// One in-flight manual `/compact`, keyed by the conversation it rewrites.
    ///
    /// A pass replaces its session's whole message array, so a send or a second
    /// pass in THAT conversation is refused until it settles — whatever
    /// interleaved would be dropped. Keying the state by conversation is what
    /// makes the refusal trustworthy: a single optional let a pass in chat B
    /// clear the state a pass in chat A still needed, after which a send could
    /// land in A's rewrite and A's Stop control cancelled nothing. `id`
    /// identifies the run, so only the pass that owns the entry clears it.
    private struct ManualCompaction {
        let id: UUID
        /// The bubble this pass streams its summary into. Only this message
        /// renders with the spinner/typewriter treatment; passing a bare
        /// "compacting" flag to the list would mark every historical bubble as
        /// live-streaming for the duration of the pass.
        let progressMessageId: UUID
        /// Held so Stop can interrupt a pass. Without a handle a multi-minute
        /// cloud compaction had no way out but waiting (or quitting).
        var task: Task<Void, Never>?
    }
    @State private var manualCompactions: [UUID: ManualCompaction] = [:]
    @State private var generatingSessionId: UUID? = nil
    /// Sessions pinned against retention while a generation unwinds or a
    /// replacement is being sent. `stopAutoregressiveGeneration` flips
    /// `isGeneratingText` false immediately, but the cancelled task can still
    /// append to its session for up to one layer, so retention must keep guarding
    /// it until the task exits. Counted rather than a plain set because the
    /// cancellation cleanup and an interrupt-and-send can both hold the same
    /// session at once, and either releasing first must not drop the guard the
    /// other still needs.
    @State private var retentionProtectionCounts: [UUID: Int] = [:]
    /// The task waiting for a cancelled generation to actually exit before its
    /// replacement may start. Held so a second interrupt can await the same
    /// teardown instead of launching a replacement while the old task still uses
    /// the shared KV buffers.
    @State private var generationTeardownTask: Task<Void, Never>? = nil
    /// The in-flight interrupt-and-send operation, chained so concurrent
    /// immediate sends run one after another rather than racing on cancellation.
    @State private var interruptSendTask: Task<Bool, Never>? = nil
    @State private var interruptSendToken: UUID? = nil
    @State private var generatedStreamText: String = ""
    @State private var thinkingText: String = ""
    @State private var responseText: String = ""
    @State private var isThinking: Bool = false
    @State private var isThinkingExpanded: Bool = true
    @State private var generationSpeedTokPerSec: Double = 0.0
    @State private var generationTotalTokens: Int = 0
    @State private var generationElapsedMs: Double = 0.0
    /// One transient status line for the whole window: refusals, failures and
    /// turn outcomes. It is drawn as an overlay like the zoom HUD, never as a
    /// row, so a message can never resize the layout it is reporting on.
    @State private var statusNotice: StatusNotice? = nil
    /// The conversation a window notice is about, or nil when it is about the app
    /// itself. Compaction and on-device generation describe work inside one chat,
    /// so their pills are withheld while another chat is on screen.
    @State private var statusNoticeSessionId: UUID? = nil
    /// Mints notice identities, so a repeat of the same text still restarts the
    /// dismissal timer.
    @State private var statusSerial: UInt64 = 0
    @State private var statusDismissTask: Task<Void, Never>? = nil
    @State private var generationTask: Task<Void, Never>? = nil
    /// Monotonic id for the live generation. A cancelled task that has already
    /// been superseded (interrupt + replacement, or a scheduled continuation)
    /// must not clear this generation's task handle/UI flags, record a prefix,
    /// or schedule another turn — see `ownsGeneration`.
    @State private var generationId: UInt64 = 0
    /// The agent mode this TURN started in. A continuation re-enters
    /// `startAutoregressiveGeneration` after the conversation may have been
    /// deleted (retention, a manual delete), and re-deriving the mode from a
    /// missing session silently resumed the coder in the middle of a plan turn.
    /// Resolved once at step 0 and carried by value for every later step.
    @State private var activeTurnMode: AgentMode? = nil

    // Working Set & Dynamic SSD Expert Paging State. Memory settings persist via
    // @AppStorage so the local server reads the same policy the UI configured.
    @AppStorage("dynamoe_memory_execution_mode") private var memoryExecutionModeRaw: String = MemoryExecutionMode.autoDetect.rawValue
    @AppStorage("dynamoe_memory_budget_mode") private var memoryBudgetModeRaw: String = MemoryBudgetMode.balanced16GB.rawValue
    @AppStorage("dynamoe_kv_cache_precision") private var kvCachePrecisionRaw: String = KVCachePrecision.fp16.rawValue
    @AppStorage("dynamoe_speculative_prefetch_enabled") private var speculativePrefetchEnabled: Bool = true
    @AppStorage("dynamoe_prefetch_lookahead_depth") private var prefetchLookaheadDepth: Int = 1
    @AppStorage("dynamoe_pin_backbone_weights") private var pinBackboneWeights: Bool = true
    @AppStorage("dynamoe_jetspec_enabled") private var jetSpecEnabled: Bool = false
    @AppStorage("dynamoe_jetspec_depth") private var jetSpecMaxDepth: Int = 3
    @AppStorage("dynamoe_jetspec_branching") private var jetSpecBranchingFactor: Int = 2
    @AppStorage("dynamoe_jetspec_expert_cap") private var jetSpecMaxExpertCap: Int = 8
    @State private var jetSpecMeanTau: Double = 1.0
    @State private var jetSpecTotalDraftAccepted: Int = 0
    @State private var jetSpecTotalDraftProposed: Int = 0

    var kvCachePrecisionBinding: Binding<KVCachePrecision> {
        Binding(
            get: { KVCachePrecision(rawValue: kvCachePrecisionRaw) ?? .fp16 },
            set: { kvCachePrecisionRaw = $0.rawValue }
        )
    }
    var memoryExecutionMode: MemoryExecutionMode {
        MemoryExecutionMode(rawValue: memoryExecutionModeRaw) ?? .autoDetect
    }
    var memoryExecutionModeBinding: Binding<MemoryExecutionMode> {
        Binding(
            get: { memoryExecutionMode },
            set: { memoryExecutionModeRaw = $0.rawValue }
        )
    }
    var memoryBudgetMode: MemoryBudgetMode {
        MemoryBudgetMode(rawValue: memoryBudgetModeRaw) ?? .balanced16GB
    }
    var memoryBudgetModeBinding: Binding<MemoryBudgetMode> {
        Binding(
            get: { memoryBudgetMode },
            set: { memoryBudgetModeRaw = $0.rawValue }
        )
    }
    var kvCachePrecision: KVCachePrecision {
        KVCachePrecision(rawValue: kvCachePrecisionRaw) ?? .fp16
    }

    @State private var modelConfig: ModelConfig? = nil
    @State private var detectedArchitecture: ModelArchitectureType = .hybridSsmMoe
    @State private var currentRssGB: Double = 0.0
    @State private var residentExpertCount: Int = 0
    @State private var totalExpertCount: Int = 0
    @State private var cacheHitRate: Double = 100.0
    @State private var prefetchEfficiency: Double = 100.0
    @State private var lastPagingLatencyMs: Double = 0.0
    @State private var pagingStatusMessage: String? = nil

    // Multi-Session Chat UI State (Antigravity Style)
    @ObservedObject private var zoomManager = AppZoomManager.shared
    @ObservedObject private var sessionStore = ChatSessionStore.shared
    @ObservedObject var localModelManager: LocalModelManager = LocalModelManager.shared
    @State private var activeLoadedModelPath: String? = nil
    /// The path whose engine, summary, and tokenizer are actually installed.
    /// `activeLoadedModelPath` is set optimistically before a load finishes (for
    /// UI) and survives failures, so it cannot prove the engine is usable; this is
    /// set only on a successful install and cleared on failure, which is what
    /// `ensureModelLoaded` requires before an async send runs.
    @State private var installedModelPath: String? = nil
    @State private var isLoadingModel: Bool = false
    /// The in-flight engine load, so an async send can wait for the specific
    /// model it needs instead of racing the detached load.
    @State private var modelLoadTask: Task<Void, Never>? = nil
    /// Monotonic id of the most recent engine load. A stale load consults it and
    /// refuses to install its engine or clear `isLoadingModel` once a newer
    /// request has superseded it.
    @State private var modelLoadToken: UInt64 = 0
    /// Monotonic id of the most recent `switchModel` request. Requests that wait
    /// out a generation teardown resume in an order Swift does not guarantee, so
    /// only the latest request is allowed to install its engine, keeping an older
    /// selection from superseding the user's newer one.
    @State private var modelSwitchRequestId: UInt64 = 0
    @State private var columnVisibility: NavigationSplitViewVisibility = .all
    /// Reference box holding the `sessions` array. SwiftUI's `@State` exposes only
    /// get/set for its value, so a computed proxy that yields `&storedSessions`
    /// during a nested write (for example `sessions[i].messages[j].content = ...`,
    /// which streaming runs every token) would copy-on-write the whole array on
    /// each statement. Keeping the array in a class and yielding into the class's
    /// stored property mutates it in place.
    private final class SessionStorage {
        var sessions: [ChatSession]
        init(_ sessions: [ChatSession]) { self.sessions = sessions }
    }
    /// Backing storage for `sessions`.
    @State private var sessionStorage = SessionStorage([
        ChatSession(title: "New Chat")
    ])
    /// Bumped on every `sessions` mutation, whole-value or in place. Watching this
    /// O(1) counter replaces observing the full `[ChatSession]` array (or a
    /// length-based fingerprint): the array's synthesized `Equatable` compared
    /// every message's text on each streaming token, and a fingerprint of string
    /// lengths still scanned the growing response, both of which scale with total
    /// history per token.
    @State private var sessionPersistRevision: UInt64 = 0
    private var sessions: [ChatSession] {
        get { sessionStorage.sessions }
        nonmutating set {
            sessionStorage.sessions = newValue
            sessionPersistRevision &+= 1
        }
        nonmutating _modify {
            yield &sessionStorage.sessions
            sessionPersistRevision &+= 1
        }
    }
    @State private var selectedSessionId: UUID? = nil
    /// Identity of the single empty conversation the view starts with, so a
    /// successful history load drops only that placeholder and not a user-created
    /// empty chat, which may already carry a chosen model or settings.
    @State private var launchPlaceholderId: UUID? = nil
    @State private var isSettingsPresented: Bool = false
    @State private var chatPromptText: String = ""
    @State private var systemPrompt: String = ModelConfig.getUserDefaultSystemPrompt()
    @AppStorage("dynamoe_thinking_enabled") private var defaultThinkingEnabled: Bool = true
    @AppStorage("dynamoe_thinking_effort") private var defaultThinkingEffort: String = ThinkingEffort.high.rawValue
    @AppStorage("dynamoe_agent_tools_enabled") private var defaultAgentToolsEnabled: Bool = false
    @AppStorage("dynamoe_agent_working_directory") private var agentWorkingDirectory: String = ""
    @AppStorage("dynamoe_max_tool_output_length") private var maxToolOutputLength: Int = 4000
    /// Whether a cloud turn re-sends every picture in the conversation on every
    /// request. On by default, because a model that forgets the picture mid-run
    /// answers about nothing; off sends only the newest user turn's pictures
    /// and replaces older ones with a one-line notice, so a twenty-step agent
    /// run stops paying for the same image twenty times.
    @AppStorage("dynamoe_cloud_resend_images") private var cloudResendImages: Bool = true
    /// The agent step budget lives in UserDefaults rather than an `@AppStorage`
    /// mirror here: the generation loops read it from inside `Task.detached`,
    /// and `@AppStorage`'s wrapper is main-actor isolated. Settings owns the
    /// controls; `agentStepCap` is how this view reads what they wrote.
    @AppStorage("dynamoe_chat_auto_delete_enabled") private var chatAutoDeleteEnabled: Bool = true
    @AppStorage("dynamoe_chat_retention_limit") private var chatRetentionLimit: Int = 10
    @State private var hasLoadedPersistedSessions: Bool = false
    /// True once the first history-load attempt has finished, whether it
    /// succeeded or failed. Gates in-memory chat readiness so a persistence
    /// outage does not leave the chat UI unable to send, while
    /// `hasLoadedPersistedSessions` still guards the merge and orphan cleanup.
    @State private var hasAttemptedInitialLoad: Bool = false
    @State private var isLoadingPersistedSessions: Bool = false
    @State private var sessionPersistTask: Task<Void, Never>? = nil
    @State private var hasPendingSessionPersist: Bool = false

    /// The step cap every agent loop in this view enforces. `nil` is "no
    /// limit". Resolved per call, so a change made in Settings applies to the
    /// next comparison of a run already in flight.
    nonisolated var agentStepCap: Int? {
        AgentStepBudget.resolvedFromDefaults()
    }

    var isAgentToolsEnabledForActiveSession: Bool {
        if let active = activeSessionBinding.wrappedValue, let enabled = active.isAgentToolsEnabled {
            return enabled
        }
        return defaultAgentToolsEnabled
    }

    var isStreamingOffDisk: Bool {
        guard let summary = summary else { return false }
        let isMoE = (modelConfig?.isMoE ?? (totalExpertCount > 0))
        if !isMoE {
            // Dense models always run purely in unified RAM -> Fast Bunny
            return false
        }
        let eff = memoryExecutionMode.resolveEffectiveMode(modelFootprintGB: summary.sizeGb)
        if eff == .residentRAM {
            return false
        }
        // In SSD streaming mode, if speed is actively fast (> 15 tok/s), show bunny; otherwise tortoise
        if generationSpeedTokPerSec >= 15.0 {
            return false
        }
        return true
    }

    var activeModelDisplayName: String? {
        if let session = activeSessionBinding.wrappedValue,
           let name = session.selectedModelName, !name.isEmpty {
            return name
        }
        if let activePath = activeLoadedModelPath {
            if let dm = localModelManager.discoveredModels.first(where: { $0.snapshotPath == activePath }) {
                return dm.displayName
            }
            if activePath.lowercased().contains("ornith") {
                return "Ornith 1.5 9B OptiQ-4bit"
            }
        }
        if let summary = summary {
            if let type = modelConfig?.modelType, !type.isEmpty {
                if type.lowercased().contains("ornith") {
                    return "Ornith 1.5 35B A3B FP8"
                }
                let isOrnithGDN = summary.tensors.contains(where: { $0.name.contains("linear_attn") }) &&
                                 !summary.tensors.contains(where: { $0.name.contains("hc_norm") || $0.name.contains("hyper_connection") })
                if isOrnithGDN {
                    return "Ornith 1.5 9B OptiQ-4bit"
                }
                return type
            }
            return detectedArchitecture.shortName
        }
        return nil
    }

    var activeModelSupportsThinking: Bool {
        // A cloud model's reasoning is the provider's own default: local name
        // heuristics say nothing about it, and the toggle is hidden on that
        // path. Streamed `reasoning_content` still renders when it arrives.
        if isCloudModelActive { return false }
        // 1. Check loaded model config / summary / detected architecture
        if ModelConfig.supportsThinking(
            config: modelConfig,
            summary: summary,
            modelName: activeModelDisplayName,
            modelPath: activeLoadedModelPath
        ) {
            return true
        }
        // 2. Check active session selected model
        if let session = activeSessionBinding.wrappedValue {
            if let model = localModelManager.discoveredModels.first(where: { $0.id == (session.selectedModelId ?? "") || $0.snapshotPath == (session.selectedModelPath ?? "") }) {
                if model.supportsThinking { return true }
            }
            if ModelConfig.supportsThinking(
                modelName: session.selectedModelName,
                modelPath: session.selectedModelPath
            ) {
                return true
            }
        }
        return false
    }

    var isThinkingEnabledForActiveSession: Bool {
        if let session = activeSessionBinding.wrappedValue, let enabled = session.isThinkingEnabled {
            return enabled
        }
        return defaultThinkingEnabled
    }

    /// Whether the active model's chat template exposes graded thinking levels
    /// (Spark X2.6's `reasoning_effort`), for rendering the level menu.
    var activeModelSupportsThinkingLevels: Bool {
        if let loadedPath = activeLoadedModelPath,
           supportsThinkingLevels(modelPath: loadedPath, modelId: nil) {
            return true
        }
        let session = activeSessionBinding.wrappedValue
        return supportsThinkingLevels(modelPath: session?.selectedModelPath, modelId: session?.selectedModelId)
    }

    /// Whether the active model can be sent images. Cloud providers opt in per
    /// model in Settings; local weights always report false today because the
    /// engine has no vision tower, and a false value makes the composer refuse
    /// an attachment instead of sending pixels that would be silently dropped.
    var activeModelSupportsVision: Bool {
        supportsVision(for: activeSessionBinding.wrappedValue)
    }

    /// Vision capability for one conversation's model, resolved from that
    /// conversation: an async send may resume after the user switched chats.
    private func supportsVision(for session: ChatSession?) -> Bool {
        guard let session else { return false }
        let selectedId = session.selectedModelId ?? ""
        let selectedPath = session.selectedModelPath ?? ""
        if let model = localModelManager.discoveredModels.first(where: {
            $0.id == selectedId || $0.snapshotPath == selectedPath
        }) {
            return model.supportsVision
        }
        // A restored cloud selection whose registry entry is absent still needs a
        // verdict; resolve it from the persisted list so a re-registered model
        // stays usable.
        if let resolved = cloudModelResolution(for: session) {
            return CloudModelRegistry.modelIdsWithVisionSupport(for: resolved.provider)
                .contains(resolved.providerModelId)
        }
        return false
    }

    /// Mirrors a provider's own image rejection into the in-memory model list so
    /// the composer's attachment gate flips immediately, without waiting for the
    /// next scan.
    private func applyVisionCapabilityChange(
        provider: CloudProvider,
        providerModelId: String,
        supported: Bool
    ) {
        let id = CloudModelAddress.modelId(providerId: provider.id, providerModelId: providerModelId)
        guard let idx = localModelManager.discoveredModels.firstIndex(where: { $0.id == id }) else { return }
        localModelManager.discoveredModels[idx].supportsVision = supported
    }

    /// Whether a cloud model accepts images. The discovered entry mirrors the
    /// registry, but a restored conversation can outlive a rescan, so the
    /// persisted list is the fallback.
    private func visionSupport(forProvider provider: CloudProvider, providerModelId: String) -> Bool {
        let id = CloudModelAddress.modelId(providerId: provider.id, providerModelId: providerModelId)
        if let model = localModelManager.discoveredModels.first(where: { $0.id == id }) {
            return model.supportsVision
        }
        return CloudModelRegistry.modelIdsWithVisionSupport(for: provider).contains(providerModelId)
    }

    /// Local weights and the Apple Foundation Model have no vision tower, so a
    /// historical user turn that carried images renders a metadata-only notice
    /// instead of losing the picture silently. Mirrors the cloud assembler's
    /// omission policy so switching a conversation to local weights is honest.
    private static func localAttachmentNotice(for message: ChatMessage) -> String {
        guard let count = message.attachments?.count, count > 0 else { return "" }
        return "\n\n" + CloudMessageAssembler.omittedImagesNotice(
            count: count,
            reason: "model has no vision support"
        )
    }

    /// Reads each attached image once and encodes it as an `image_url` data URL.
    /// Runs off the main actor: a 4 MB PNG becomes ~5.3 MB of base64, and the
    /// cloud loop re-assembles its message history on every agent step.
    private nonisolated static func buildImageDataURLs(for messages: [ChatMessage]) -> [UUID: [String]] {
        var payload: [UUID: [String]] = [:]
        let store = ImageAttachmentStore.shared
        for message in messages {
            guard let refs = message.attachments, !refs.isEmpty else { continue }
            let urls = refs.compactMap { ref -> String? in
                guard let data = store.data(for: ref) else { return nil }
                return PreparedImage.dataURL(mimeType: ref.mimeType, data: data)
            }
            if !urls.isEmpty {
                payload[message.id] = urls
            }
        }
        return payload
    }

    /// Graded thinking level shown selected in the chat menu: the conversation's
    /// choice or the clamped app default. Only meaningful for level-capable models.
    var thinkingEffortForActiveSession: ThinkingEffort {
        activeSessionBinding.wrappedValue?.thinkingEffort ?? defaultThinkingEffortLevel
    }

    /// The app-wide default level, clamped so a stale `dynamoe_thinking_effort`
    /// value can never reach the prompt or render an unchecked menu item.
    private var defaultThinkingEffortLevel: ThinkingEffort {
        ThinkingEffort.clamped(defaultThinkingEffort)
    }

    /// Display name for a specific conversation's model. An async send that
    /// resumed after the user switched chats must format with the model it is
    /// actually running on, not whatever is selected now.
    private func modelDisplayName(for session: ChatSession?) -> String? {
        if let name = session?.selectedModelName, !name.isEmpty { return name }
        return activeModelDisplayName
    }

    /// Whether a specific conversation's model supports thinking, derived from
    /// that conversation rather than the current selection.
    private func supportsThinking(for session: ChatSession?) -> Bool {
        // Cloud reasoning is provider-controlled; the local name heuristics below
        // would light the thinking spinner for names like "deepseek" even
        // though the cloud path has no on/off switch to honor.
        if let session, cloudModelResolution(for: session) != nil { return false }
        if ModelConfig.supportsThinking(
            config: modelConfig,
            summary: summary,
            modelName: modelDisplayName(for: session),
            modelPath: session?.selectedModelPath ?? activeLoadedModelPath
        ) {
            return true
        }
        if let session,
           let model = localModelManager.discoveredModels.first(where: {
               $0.id == (session.selectedModelId ?? "") || $0.snapshotPath == (session.selectedModelPath ?? "")
           }),
           model.supportsThinking {
            return true
        }
        return ModelConfig.supportsThinking(
            modelName: session?.selectedModelName,
            modelPath: session?.selectedModelPath
        )
    }

    /// Thinking toggle for a specific conversation, defaulting to the app default.
    private func isThinkingEnabled(for session: ChatSession?) -> Bool {
        session?.isThinkingEnabled ?? defaultThinkingEnabled
    }

    /// Whether a specific conversation's model exposes graded thinking levels,
    /// derived from that conversation rather than the current selection. The
    /// scan's result is checked first; the file probe is memoized by path.
    private func supportsThinkingLevels(for session: ChatSession?) -> Bool {
        supportsThinkingLevels(
            modelPath: session?.selectedModelPath ?? activeLoadedModelPath,
            modelId: session?.selectedModelId
        )
    }

    private func supportsThinkingLevels(modelPath: String?, modelId: String?) -> Bool {
        if let model = discoveredModel(modelPath: modelPath, modelId: modelId) {
            return model.supportsThinkingLevels
        }
        return ModelConfig.supportsThinkingLevels(modelPath: modelPath)
    }

    private func discoveredModel(modelPath: String?, modelId: String?) -> DiscoveredModel? {
        if let modelId, !modelId.isEmpty,
           let model = localModelManager.discoveredModels.first(where: { $0.id == modelId }) {
            return model
        }
        guard let modelPath, !modelPath.isEmpty else { return nil }
        return localModelManager.discoveredModels.first(where: { $0.snapshotPath == modelPath })
    }

    /// Graded thinking level for a specific conversation, defaulting to the app
    /// default. Only consumed when the model is level-capable and thinking is on.
    private func thinkingEffort(for session: ChatSession?) -> ThinkingEffort {
        session?.thinkingEffort ?? defaultThinkingEffortLevel
    }

    var activeSessionBinding: Binding<ChatSession?> {
        Binding<ChatSession?>(
            get: {
                if let id = selectedSessionId {
                    return sessions.first(where: { $0.id == id }) ?? sessions.first
                }
                return sessions.first
            },
            set: { updated in
                guard let updated = updated else { return }
                if let idx = sessions.firstIndex(where: { $0.id == updated.id }) {
                    sessions[idx] = updated
                }
            }
        )
    }

    private func applyProfile(_ profile: ModelProfileType, for modelIdentifier: String? = nil) {
        let modelKey = modelIdentifier ?? activeModelDisplayName ?? localModelManager.defaultModelId
        let settings = ModelProfileManager.shared.getProfile(for: modelKey, type: profile)
        self.activeProfile = profile
        ModelProfileManager.shared.setActiveProfile(for: modelKey, type: profile)

        self.temperature = settings.temperature
        self.topP = settings.topP
        self.minP = settings.minP
        self.topK = settings.topK
        self.repetitionPenalty = settings.repetitionPenalty
        self.presencePenalty = settings.presencePenalty
        self.maxNewTokens = settings.maxNewTokens
        self.systemPrompt = settings.systemPrompt
        self.jetSpecEnabled = settings.jetSpecEnabled
        self.jetSpecMaxDepth = settings.jetSpecMaxDepth
        self.jetSpecBranchingFactor = settings.jetSpecBranchingFactor
        self.jetSpecMaxExpertCap = settings.jetSpecMaxExpertCap
    }

    private func saveCurrentSettingsToProfile(_ profile: ModelProfileType, for modelIdentifier: String? = nil) {
        let modelKey = modelIdentifier ?? activeModelDisplayName ?? localModelManager.defaultModelId
        let settings = GenerationProfileSettings(
            temperature: self.temperature,
            topP: self.topP,
            minP: self.minP,
            topK: self.topK,
            repetitionPenalty: self.repetitionPenalty,
            presencePenalty: self.presencePenalty,
            maxNewTokens: self.maxNewTokens,
            systemPrompt: self.systemPrompt,
            jetSpecEnabled: self.jetSpecEnabled,
            jetSpecMaxDepth: self.jetSpecMaxDepth,
            jetSpecBranchingFactor: self.jetSpecBranchingFactor,
            jetSpecMaxExpertCap: self.jetSpecMaxExpertCap
        )
        ModelProfileManager.shared.saveProfile(for: modelKey, type: profile, settings: settings)
    }

    /// Starts a model switch. Returns the task that finishes once the requested
    /// model's engine and buffers are installed (or the load fails), so callers
    /// that must run against the new model can await it; UI callers ignore it.
    @discardableResult
    private func switchModel(to model: DiscoveredModel, sessionId: UUID? = nil) -> Task<Void, Never>? {
        // Capture the conversation this switch is for before any suspension, so
        // a deferred switch still records the model on the intended session even
        // if the user selects another conversation while it waits.
        let targetSessionId = sessionId ?? selectedSessionId ?? sessions.first?.id
        // Mint a request id before any suspension. Requests that wait out a
        // teardown resume in an order Swift does not guarantee, so without this an
        // older selection could resume last and install its engine over a newer
        // one; only the latest request is allowed to complete.
        modelSwitchRequestId &+= 1
        let requestId = modelSwitchRequestId
        // Never swap the engine/buffers out from under an in-flight generation:
        // its mmaps are unmapped when the old engine releases, which would fault
        // any GPU work still in progress.
        if isGeneratingText {
            stopAutoregressiveGeneration()
        }
        guard let teardown = generationTeardownTask else {
            return completeModelSwitch(to: model, sessionId: targetSessionId)
        }
        // A cancelled generation is still unwinding and may still be inside a
        // kernel dispatch. Loading a new engine now would unmap its mmaps and
        // reset the shared buffers out from under it, so wait for the teardown.
        return Task { @MainActor in
            await teardown.value
            // A newer switch owns the outcome now; dropping this stale request
            // keeps it from superseding the user's newer selection.
            guard requestId == self.modelSwitchRequestId else { return }
            // A generation may have started during the wait: stop it and wait for
            // that teardown too, since swapping resets the shared buffers it uses.
            if self.isGeneratingText {
                self.stopAutoregressiveGeneration()
            }
            if let next = self.generationTeardownTask {
                await next.value
                guard requestId == self.modelSwitchRequestId else { return }
            }
            await self.completeModelSwitch(to: model, sessionId: targetSessionId).value
        }
    }

    /// The non-suspending half of a model switch. Runs only once no cancelled
    /// generation is still using the shared engine and buffers, and returns the
    /// load task so a caller can wait for that specific model to be ready.
    @discardableResult
    private func completeModelSwitch(to model: DiscoveredModel, sessionId targetSessionId: UUID?) -> Task<Void, Never> {
        if model.isCloudModel {
            return coordinateCloudModelActivation(of: model, sessionId: targetSessionId)
        }
        if model.isAppleFoundationModel {
            return coordinateAppleFoundationModelActivation(of: model, sessionId: targetSessionId)
        }
        PrefixCacheManager.shared.invalidate()
        let loadTask = loadAndBridgeToMetal(filePath: model.snapshotPath)
        activeLoadedModelPath = model.snapshotPath
        localModelManager.setLastUsedModel(id: model.id)
        if let sid = targetSessionId,
           let idx = sessions.firstIndex(where: { $0.id == sid }) {
            sessions[idx].selectedModelId = model.id
            sessions[idx].selectedModelName = model.displayName
            sessions[idx].selectedModelPath = model.snapshotPath
        }
        let preferredProfile = ModelProfileManager.shared.getActiveProfile(for: model.id)
        applyProfile(preferredProfile, for: model.id)
        return loadTask
    }

    /// Makes the model a conversation expects the active one, and waits for it to
    /// finish installing, before an async send. A send that waited out a teardown
    /// can resume after the user selected another chat and ran `switchModel`,
    /// which would otherwise leave the replacement turn running against the newly
    /// loaded model while targeting the old conversation. The await matters
    /// because `loadAndBridgeToMetal` returns while the previous engine and
    /// buffers are still installed.
    ///
    /// Returns whether the requested path is the one whose engine/summary/tokenizer
    /// are actually installed. A switch can drop this request when a newer one
    /// supersedes it (see `modelSwitchRequestId`), and a load can fail, so the
    /// awaited work may complete without this model usable; callers must then
    /// refuse to send rather than run on the old engine.
    @discardableResult
    private func ensureModelLoaded(forSession sessionId: UUID?) async -> Bool {
        guard let sessionId,
              let session = sessions.first(where: { $0.id == sessionId }),
              let targetPath = session.selectedModelPath,
              !targetPath.isEmpty else { return true }
        // Never hand an unresolved cloud sentinel to the weights loader: it has
        // no safetensors to mmap, and the send path refuses this selection too.
        if CloudModelAddress.selectionIsUnresolved(
            modelId: session.selectedModelId,
            modelPath: targetPath
        ) {
            showStatus("⚠️ This conversation's cloud model could not be resolved. Reselect the model in the model picker.")
            return false
        }
        if installedModelPath == targetPath {
            // The engine for this path may still be installing; wait for that
            // specific load rather than sending against the previous model.
            if isLoadingModel { await modelLoadTask?.value }
            return isEngineUsable(forPath: targetPath)
        }
        if let model = resolvedDiscoveredModel(forPath: targetPath, session: session) {
            await switchModel(to: model, sessionId: sessionId)?.value
        } else {
            PrefixCacheManager.shared.invalidate()
            let loadTask = loadAndBridgeToMetal(filePath: targetPath)
            activeLoadedModelPath = targetPath
            if let idx = sessions.firstIndex(where: { $0.id == sessionId }) {
                sessions[idx].selectedModelPath = targetPath
            }
            let preferredProfile = ModelProfileManager.shared.getActiveProfile(for: targetPath)
            applyProfile(preferredProfile, for: targetPath)
            await loadTask.value
        }
        return isEngineUsable(forPath: targetPath)
    }

    /// Whether the engine, summary, and tokenizer for `path` are installed and
    /// settled. `activeLoadedModelPath` is not used here because it is set before
    /// a load finishes and left in place on failure, so it cannot prove usability.
    private func isEngineUsable(forPath path: String) -> Bool {
        // The system on-device model has no engine/tokenizer/summary triple: it
        // is installed as soon as the sentinel path is the active backend.
        if path == AppleFoundationModelService.modelPath {
            return installedModelPath == path && !isLoadingModel
        }
        // A cloud provider entry has no local weights at all; like the system
        // model it is usable the moment its sentinel path is installed.
        if CloudModelAddress.isCloudPath(path) {
            return installedModelPath == path && !isLoadingModel
        }
        return installedModelPath == path
            && engine != nil
            && summary != nil
            && tokenizer != nil
            && !isLoadingModel
    }

    /// Resolves a conversation's model entry, rebuilding a virtual cloud entry
    /// ad hoc when the session is pinned to a provider model that is no longer
    /// in the registry (Clear, or a retired model). Without this the fallback
    /// would hand a `cloud://` sentinel to the weights loader, which has no
    /// safetensors to mmap.
    private func resolvedDiscoveredModel(forPath path: String, session: ChatSession?) -> DiscoveredModel? {
        if let model = localModelManager.getModel(byId: path) { return model }
        if let id = session?.selectedModelId, let model = localModelManager.getModel(byId: id) { return model }
        if let resolved = CloudModelAddress.parsePath(path) {
            return CloudModelRegistry.adHocDiscoveredModel(
                provider: resolved.provider,
                providerModelId: resolved.providerModelId
            )
        }
        return nil
    }

    /// Brings the process-wide engine back to the selected conversation's model
    /// after a turn ends. A queued or interrupted send can switch the engine to a
    /// background conversation's model (see `ensureModelLoaded`), and nothing
    /// restores the visible one because `selectedSessionId` never changed, so the
    /// next ordinary send would otherwise run on the wrong model. A no-op when
    /// the engine already matches the selection.
    private func reconcileEngineWithSelection() {
        guard let selectionId = selectedSessionId ?? sessions.first?.id,
              let session = sessions.first(where: { $0.id == selectionId }),
              let targetPath = session.selectedModelPath, !targetPath.isEmpty,
              activeLoadedModelPath != targetPath else { return }
        if CloudModelAddress.selectionIsUnresolved(
            modelId: session.selectedModelId,
            modelPath: targetPath
        ) { return }
        if let model = resolvedDiscoveredModel(forPath: targetPath, session: session) {
            switchModel(to: model, sessionId: selectionId)
        } else {
            loadAndBridgeToMetal(filePath: targetPath)
            activeLoadedModelPath = targetPath
            let preferredProfile = ModelProfileManager.shared.getActiveProfile(for: targetPath)
            applyProfile(preferredProfile, for: targetPath)
        }
    }

    // MARK: - Apple Foundation Model (System On-Device) Backend

    /// True when the active chat backend is Apple's system on-device Foundation
    /// Model rather than a locally loaded weights engine. `installedModelPath`
    /// carries the sentinel path while it is active; no Rust engine, tokenizer,
    /// Metal buffers, or KV cache exist in that state.
    var isAppleFoundationModelActive: Bool {
        installedModelPath == AppleFoundationModelService.modelPath
    }

    /// Whether a conversation is (or would default to) the Apple Foundation
    /// Model. Like the weights path, a session without explicit model fields
    /// runs against whatever backend is currently active.
    private func sessionUsesAppleFoundationModel(_ session: ChatSession) -> Bool {
        if session.selectedModelId == AppleFoundationModelService.modelId { return true }
        if let path = session.selectedModelPath, !path.isEmpty {
            return path == AppleFoundationModelService.modelPath
        }
        return isAppleFoundationModelActive
    }

    /// Installs the system on-device model as the active backend. Nothing loads
    /// from disk: any resident weights engine is released so the working-set
    /// budget is not held by a backend that no longer needs it. Callers must go
    /// through `coordinateAppleFoundationModelActivation` so an in-flight
    /// weights load cannot install over this activation afterward, and its
    /// unguarded side effects cannot repopulate what this releases.
    private func activateAppleFoundationModel(_ model: DiscoveredModel, sessionId targetSessionId: UUID?) {
        PrefixCacheManager.shared.invalidate()
        ExpertIOThreadPool.shared.closeAllLayerFDs()
        WorkingSetManager.shared.releaseShardMappings()
        CompletionEngine.shared.configure(summary: nil, modelConfig: nil, tokenizer: nil, shardBuffers: [:], activeLoadedModelPath: nil)
        BackbonePinPool.shared.release()
        engine = nil
        tokenizer = nil
        summary = nil
        modelConfig = nil
        shardBuffers = [:]
        isLoadingModel = false
        installedModelPath = AppleFoundationModelService.modelPath
        activeLoadedModelPath = AppleFoundationModelService.modelPath
        localModelManager.setLastUsedModel(id: model.id)
        if let sid = targetSessionId,
           let idx = sessions.firstIndex(where: { $0.id == sid }) {
            sessions[idx].selectedModelId = model.id
            sessions[idx].selectedModelName = model.displayName
            sessions[idx].selectedModelPath = model.snapshotPath
        }
        let preferredProfile = ModelProfileManager.shared.getActiveProfile(for: model.id)
        applyProfile(preferredProfile, for: model.id)
        let availability = AppleFoundationModelService.checkAvailability()
        if availability.isReady {
            metalStatus = "✅ Apple Foundation Model ready (on-device)."
            showStatus("🍎 Apple Foundation Model active.")
        } else {
            metalStatus = "⚠️ Apple Foundation Model: \(availability.statusText)"
            showStatus(metalStatus)
        }
        // Warm the on-device model so the first turn does not pay spin-up latency.
        AppleFoundationModelService.prewarm()
        // The sentinel is installed by now, so this resolves every conversation
        // that will actually run on the system model.
        retirePlanModeUnsupportedByActiveBackend()
    }

    /// Serializes AFM activation against any in-flight weights load, mirroring
    /// the `modelLoadToken` protocol two weights loads follow. Bumping the token
    /// makes a stale load skip its guarded installs instead of installing its
    /// engine/path over the sentinel afterwards, and draining that load task
    /// fully before mutating keeps its unguarded side effects (ExpertIO pool
    /// setup, working-set initialization and pre-faulting) from repopulating
    /// resources after this activation released them. Generation teardown is
    /// awaited for the same reason the weights load awaits it: direct callers
    /// reach here without `switchModel`'s stop-and-wait, and releasing the shared
    /// engine, FDs, and shard mappings would fault a generation still running.
    /// The token is also rechecked after each suspension: a newer request made
    /// while this one waits has already recorded its own session, profile, and
    /// last-used choices at request time, so a superseded activation must stand
    /// down rather than flip them back to the system model. The returned task
    /// tracks readiness, just like `loadAndBridgeToMetal`'s.
    @discardableResult
    private func coordinateAppleFoundationModelActivation(of model: DiscoveredModel, sessionId targetSessionId: UUID?) -> Task<Void, Never> {
        modelLoadToken &+= 1
        let token = modelLoadToken
        let previousLoad = modelLoadTask
        let activationTask = Task { @MainActor in
            await previousLoad?.value
            guard self.modelLoadToken == token else { return }
            await self.awaitGenerationTeardown()
            guard self.modelLoadToken == token else { return }
            self.activateAppleFoundationModel(model, sessionId: targetSessionId)
        }
        modelLoadTask = activationTask
        return activationTask
    }

    // MARK: - Connected Cloud Provider Backend

    /// True when the active chat backend is a remote cloud provider rather
    /// than local weights or the system on-device model. `installedModelPath`
    /// carries the provider's `cloud://` sentinel while it is active; no Rust
    /// engine, tokenizer, Metal buffers, or KV cache exist in that state.
    var isCloudModelActive: Bool {
        CloudModelAddress.isCloudPath(installedModelPath)
    }

    /// Resolves a conversation's backend to a cloud provider when it runs on
    /// one. Like the weights and system-model paths, a session without
    /// explicit model fields runs against whatever backend is active.
    private func cloudModelResolution(for session: ChatSession) -> (provider: CloudProvider, providerModelId: String)? {
        // A persisted cloud sentinel that no longer resolves (unknown provider,
        // corrupted defaults) must NOT fall through to the installed backend:
        // that would silently run the turn on a different model than the
        // conversation names. `unresolvedCloudSelection` turns this into a
        // refusal on the send path.
        //
        // An explicit cloud path wins first: sessions persist both fields, and
        // path is the field the engine keyed on when the entry was installed.
        if let path = session.selectedModelPath, CloudModelAddress.isCloudPath(path) {
            return CloudModelAddress.parsePath(path)
        }
        if let id = session.selectedModelId, !id.isEmpty {
            if let resolved = CloudModelAddress.parseModelId(id) { return resolved }
            if CloudModelAddress.isCloudModelId(id) { return nil }
            // A local registry id pins this conversation to local weights even
            // when another backend is installed globally.
            return nil
        }
        if let path = session.selectedModelPath, !path.isEmpty {
            // Any remaining explicit path is a local snapshot.
            return nil
        }
        guard let installedModelPath else { return nil }
        return CloudModelAddress.parsePath(installedModelPath)
    }

    /// True when a conversation's persisted model fields claim a cloud provider
    /// but can no longer be resolved. The send path refuses this instead of
    /// running the turn on whatever weights happen to be installed.
    private func unresolvedCloudSelection(for session: ChatSession) -> Bool {
        CloudModelAddress.selectionIsUnresolved(
            modelId: session.selectedModelId,
            modelPath: session.selectedModelPath
        )
    }

    /// Installs a cloud provider model as the active backend. Nothing loads
    /// from disk, so any resident weights engine is released exactly like the
    /// system-model activation releases it: the working-set budget must not
    /// stay held by a backend that will never touch it.
    private func activateCloudModel(_ model: DiscoveredModel, sessionId targetSessionId: UUID?) {
        guard let resolved = model.cloudResolution else {
            // Never leave the previous backend installed behind a selection
            // that claims a cloud model: say why nothing happened.
            let message = "⚠️ Could not resolve \(model.displayName) to a cloud provider — reselect the model in the model picker."
            metalStatus = message
            showStatus(message, kind: .warning)
            return
        }
        PrefixCacheManager.shared.invalidate()
        ExpertIOThreadPool.shared.closeAllLayerFDs()
        WorkingSetManager.shared.releaseShardMappings()
        CompletionEngine.shared.configure(summary: nil, modelConfig: nil, tokenizer: nil, shardBuffers: [:], activeLoadedModelPath: nil)
        BackbonePinPool.shared.release()
        engine = nil
        tokenizer = nil
        summary = nil
        modelConfig = nil
        shardBuffers = [:]
        isLoadingModel = false
        installedModelPath = model.snapshotPath
        activeLoadedModelPath = model.snapshotPath
        localModelManager.setLastUsedModel(id: model.id)
        if let sid = targetSessionId,
           let idx = sessions.firstIndex(where: { $0.id == sid }) {
            sessions[idx].selectedModelId = model.id
            sessions[idx].selectedModelName = model.displayName
            sessions[idx].selectedModelPath = model.snapshotPath
        }
        let preferredProfile = ModelProfileManager.shared.getActiveProfile(for: model.id)
        applyProfile(preferredProfile, for: model.id)
        if CloudSecretStore.shared.hasCurrentCredential(providerId: resolved.provider.id) {
            metalStatus = "☁️ \(resolved.provider.displayName) ready — \(model.displayName)."
            showStatus("☁️ \(resolved.provider.displayName) active.")
        } else if CloudSecretStore.shared.exists(providerId: resolved.provider.id) {
            let message = "⚠️ \(resolved.provider.displayName): the API key is still in the older keychain. Import it in Settings ▸ Cloud Providers."
            metalStatus = message
            showStatus(message, kind: .warning)
        } else {
            let message = "⚠️ \(resolved.provider.displayName): no API key stored — add one in Settings ▸ Cloud Providers."
            metalStatus = message
            showStatus(message, kind: .warning)
        }
        // A plan conversation needs a tool-calling model, and this backend may
        // not have one. The cloud sentinel is installed by now, so conversations
        // without an explicit model resolve to it.
        retirePlanModeUnsupportedByActiveBackend()
    }

    /// Serializes cloud activation against in-flight weights loads and
    /// generations, mirroring `coordinateAppleFoundationModelActivation`: the
    /// token makes a stale load skip its guarded installs, and draining the
    /// previous load plus the generation teardown before releasing the engine
    /// keeps unguarded side effects (expert pool setup, shard mappings) from
    /// repopulating resources after this activation released them.
    @discardableResult
    private func coordinateCloudModelActivation(of model: DiscoveredModel, sessionId targetSessionId: UUID?) -> Task<Void, Never> {
        modelLoadToken &+= 1
        let token = modelLoadToken
        let previousLoad = modelLoadTask
        let activationTask = Task { @MainActor in
            await previousLoad?.value
            guard self.modelLoadToken == token else { return }
            await self.awaitGenerationTeardown()
            guard self.modelLoadToken == token else { return }
            self.activateCloudModel(model, sessionId: targetSessionId)
        }
        modelLoadTask = activationTask
        return activationTask
    }

    // MARK: - Chat Session Persistence & Retention

    /// `nil` means "never auto-delete".
    private var chatRetentionLimitOrNil: Int? {
        chatAutoDeleteEnabled ? max(1, chatRetentionLimit) : nil
    }

    private func loadPersistedSessionsIfNeeded() async {
        guard !hasLoadedPersistedSessions, !isLoadingPersistedSessions else { return }
        isLoadingPersistedSessions = true
        // Identify the launch placeholder before merging even if `onAppear` has not
        // run yet, so a user-created empty chat is not mistaken for it.
        if launchPlaceholderId == nil, sessions.count == 1, sessions[0].messages.isEmpty {
            launchPlaceholderId = sessions[0].id
        }

        guard let loaded = await ChatSessionStore.shared.loadSessionsAsync() else {
            // The store could not read its history, so an empty result here
            // cannot be told apart from a failed read. Leave the persistence load
            // incomplete (and thus retryable) instead of replacing the visible
            // conversations with nothing, but mark the initial attempt as done so
            // the user can still chat; the store's own availability banner tells
            // them persistence is down.
            hasAttemptedInitialLoad = true
            isLoadingPersistedSessions = false
            return
        }

        if loaded.isEmpty {
            // Nothing on disk yet: the in-memory session becomes the starting
            // point rather than being discarded.
            hasAttemptedInitialLoad = true
            hasLoadedPersistedSessions = true
            isLoadingPersistedSessions = false
            ChatSessionStore.shared.saveAll(sessions)
            applyChatRetention()
            return
        }

        // Preserve every in-memory conversation the user actually used while the
        // load was pending or after a failed attempt, so a retry that finally
        // succeeds does not discard their work. A conversation already in memory
        // is at least as fresh as the disk copy: if the user kept chatting while
        // this load decoded, a debounced save may have written an older snapshot
        // to disk. So keep the live value on overlap and append only history that
        // memory does not already have. The untouched launch placeholder is
        // dropped by identity rather than by being empty, so a New Chat created
        // before the load finished survives even with no messages. A rename marks
        // the placeholder as used, so its new title is kept too.
        let memoryIds = Set(sessions.map(\.id))
        let diskOnly = loaded.filter { !memoryIds.contains($0.id) }
        let keptFromMemory = sessions.filter { session in
            if session.id == launchPlaceholderId, session.isUnusedPlaceholder { return false }
            return true
        }

        sessions = keptFromMemory + diskOnly
        // Preserve the user's current selection across a retry: they may have
        // used and selected an in-memory chat while the first load failed or was
        // pending, and jumping to `sessions.first` would yank them out of it.
        // Only fall back when their selection is no longer present.
        let selectionSurvived = selectedSessionId.map { selected in
            sessions.contains { $0.id == selected }
        } ?? false
        if !selectionSurvived {
            selectedSessionId = sessions.first?.id
        }

        hasAttemptedInitialLoad = true
        hasLoadedPersistedSessions = true
        isLoadingPersistedSessions = false

        // Conversations kept from memory were never written; persist now.
        if !keptFromMemory.isEmpty {
            ChatSessionStore.shared.saveAll(sessions)
        }
        applyChatRetention()
    }

    /// Drops conversations beyond the retention window and deletes their files.
    /// The active conversation and any session with a generation in flight are
    /// always preserved.
    private func applyChatRetention() {
        let plan = ChatSessionStore.retentionPlan(
            sessions: sessions,
            limit: chatRetentionLimitOrNil,
            protectedIds: chatRetentionProtectedIds
        )
        if !plan.removed.isEmpty {
            let removedIds = plan.removed.map(\.id)
            sessions = plan.kept
            if let selected = selectedSessionId, removedIds.contains(selected) {
                selectedSessionId = sessions.first?.id
            }
            ChatSessionStore.shared.delete(sessionIds: removedIds)
            ImageAttachmentStore.shared.delete(sessionIds: removedIds)
        }
        // Run the orphan pass on every retention sweep, not only when a
        // conversation was evicted: deleting a message or a queued prompt by hand
        // also orphans assets, and gating this on `plan.removed` let those bytes
        // survive indefinitely. The pass reads only the live refs, so an empty
        // plan still removes exactly what nothing references.
        ImageAttachmentStore.shared.prune(
            liveSessionIds: Set(sessions.map(\.id)),
            liveRelativePaths: Self.attachmentRelativePaths(in: sessions)
        )
    }

    /// Every asset path a live conversation still references, for the attachment
    /// store's orphan pass. Queued prompts count: their images are already on
    /// disk waiting for the turn that will send them.
    private static func attachmentRelativePaths(in sessions: [ChatSession]) -> Set<String> {
        var paths: Set<String> = []
        for session in sessions {
            for message in session.messages {
                for ref in message.attachments ?? [] {
                    paths.insert(ref.relativePath)
                }
            }
            for prompt in session.queuedPrompts {
                for ref in prompt.attachments ?? [] {
                    paths.insert(ref.relativePath)
                }
            }
        }
        return paths
    }

    /// Sessions retention must never evict: the one the user is viewing and the
    /// one actively generating (which may differ after switching chats mid-turn,
    /// since the generation loop still appends to its own session).
    private var chatRetentionProtectedIds: Set<UUID> {
        var ids: Set<UUID> = []
        if let selected = selectedSessionId {
            ids.insert(selected)
        } else if let first = sessions.first?.id {
            ids.insert(first)
        }
        if isGeneratingText, let generating = generatingSessionId {
            ids.insert(generating)
        }
        ids.formUnion(retentionProtectionCounts.keys)
        return ids
    }

    /// Pins a session against retention until a matching `release` runs.
    private func retainRetentionProtection(_ id: UUID) {
        retentionProtectionCounts[id, default: 0] += 1
    }

    /// Drops one retention pin; the session becomes prunable again once its last
    /// holder releases.
    private func releaseRetentionProtection(_ id: UUID) {
        guard let count = retentionProtectionCounts[id] else { return }
        retentionProtectionCounts[id] = count > 1 ? count - 1 : nil
    }

    /// Coalesces the rapid session mutations that happen during streaming into a
    /// single disk write once the conversation settles.
    private func scheduleSessionPersist() {
        // Wait for the first load attempt so a save never races the load, but do
        // not require it to have succeeded: with persistence down the user can
        // still chat, and those conversations must still reach disk when it
        // recovers.
        guard hasAttemptedInitialLoad else { return }
        hasPendingSessionPersist = true
        sessionPersistTask?.cancel()
        sessionPersistTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 800_000_000)
            guard !Task.isCancelled else { return }
            self.hasPendingSessionPersist = false
            // Orphan cleanup is only safe once the load has populated
            // `managedIds`; before that it would delete not-yet-loaded files.
            ChatSessionStore.shared.saveAll(self.sessions, cleanOrphans: self.hasLoadedPersistedSessions)
        }
    }

    /// Writes immediately, used when the app leaves the foreground or quits so
    /// an in-flight conversation is not lost. Always enqueues the current
    /// snapshot: the debounced save clears `hasPendingSessionPersist` before its
    /// I/O result is known, so gating on that flag here would skip the final save
    /// after a *failed* write and lose the latest turn. `saveAll` skips
    /// conversations whose bytes are already durable, so re-enqueuing is cheap.
    private func flushSessionPersist() {
        sessionPersistTask?.cancel()
        hasPendingSessionPersist = false
        // Before the load has populated `managedIds`, skip orphan cleanup: it
        // would treat the not-yet-merged files on disk as deleted conversations
        // and remove them. Writing without cleanup is safe and is what preserves
        // conversations created while the load was still running (or failed).
        ChatSessionStore.shared.saveAll(sessions, cleanOrphans: hasLoadedPersistedSessions)
    }

    // MARK: - Transient Status

    /// Says something without moving anything. Refusals (a queued prompt that
    /// cannot be sent, a model that cannot see a staged image), failures, and
    /// turn outcomes all arrive here; before this they were written to state that
    /// nothing rendered, so a refused send looked like a dead button.
    ///
    /// Severity comes from the message's own leading glyph, so a writer cannot
    /// disagree with the tint, and every notice clears itself unless it describes
    /// unfinished work: a status line that outlives its event is worse than none.
    ///
    /// Returns the identity the notice was given, so work that reports progress
    /// in place can take down exactly its own pill when it finishes. A notice
    /// that replaced it keeps its own identity and is left alone.
    @discardableResult
    private func showStatus(
        _ text: String,
        kind: StatusNotice.Kind? = nil,
        sessionId: UUID? = nil
    ) -> StatusNotice.Identity {
        let resolved = kind ?? StatusNoticePolicy.kind(for: text)
        assert(
            !StatusNoticePolicy.statesContradictorySeverity(text: text, kind: resolved),
            "\(text) leads with a glyph of another severity: drop the explicit kind, or the glyph"
        )
        statusDismissTask?.cancel()
        statusDismissTask = nil
        // A repeat of what is already on screen keeps its identity, so it does
        // not re-run the fade, but it does restart the dismissal timer.
        let isRepeat = StatusNoticePolicy.isRepeat(
            text: text, kind: resolved, source: .window, current: statusNotice
        )
        let identity: StatusNotice.Identity
        if isRepeat, let current = statusNotice {
            identity = current.id
        } else {
            statusSerial &+= 1
            identity = StatusNotice.Identity(source: .window, serial: statusSerial)
        }
        statusNotice = StatusNotice(id: identity, kind: resolved, text: text)
        // Every notice states its scope: a work pill belongs to one conversation
        // and must not float over another, while the rest are about the app.
        statusNoticeSessionId = sessionId
        // An overlay is not reachable by tabbing, so announce it instead — once
        // per identity, because a repeat is the same sentence continued.
        if !isRepeat {
            AccessibilityNotification.Announcement(text).post()
        }
        guard resolved.autoDismisses else { return identity }
        statusDismissTask = Task {
            try? await Task.sleep(nanoseconds: StatusNoticePolicy.autoDismissNanoseconds)
            guard !Task.isCancelled, statusNotice?.id == identity else { return }
            withAnimation(.easeOut(duration: 0.15)) {
                statusNotice = nil
            }
            statusDismissTask = nil
        }
        return identity
    }

    private func clearStatus() {
        statusDismissTask?.cancel()
        statusDismissTask = nil
        statusNotice = nil
        statusNoticeSessionId = nil
    }

    /// Clears the notice only if it is still the one `id` names. Progress work can
    /// finish after another message has taken the pill — a late finisher must not
    /// retract somebody else's notice.
    private func clearStatus(ifIdentity id: StatusNotice.Identity) {
        guard statusNotice?.id == id else { return }
        clearStatus()
    }

    /// The window notice the visible chat column may draw. A pill about one
    /// conversation is withheld while another is on screen — the transcript's own
    /// progress row is keyed to the session for the same reason — while a notice
    /// about the app stays visible everywhere.
    private var visibleStatusNotice: StatusNotice? {
        guard let statusNotice else { return nil }
        guard let statusNoticeSessionId else { return statusNotice }
        return statusNoticeSessionId == (selectedSessionId ?? sessions.first?.id) ? statusNotice : nil
    }

    /// A load failure is the one status line whose cause is long and whose effect
    /// outlives the six seconds the notice lives: "missing shard" or an allocation
    /// refusal is several lines in a three-line pill, and a chat that cannot send
    /// is a permanent state. The transcript row is the durable copy, the same way
    /// a provider's image downgrade is recorded there, so the reason stays
    /// readable after the pill has gone.
    private func recordLoadFailureInTranscript(_ reason: String) {
        let text = "❌ Engine load failed: \(reason)"
        guard let sid = selectedSessionId ?? sessions.first?.id,
              let idx = sessions.firstIndex(where: { $0.id == sid }) else { return }
        // Retrying the same load fails the same way; the row is a record, not a log.
        guard sessions[idx].messages.last?.content != text else { return }
        sessions[idx].messages.append(ChatMessage(role: .system, content: text))
    }

    var body: some View {
        let levelCapable = activeModelSupportsThinkingLevels
        return ZStack {
            NavigationSplitView(columnVisibility: $columnVisibility) {
                SidebarView(
                    // Route the binding through the computed `sessions` so sidebar
                    // edits also bump the persistence revision.
                    sessions: Binding(
                        get: { sessions },
                        set: { sessions = $0 }
                    ),
                    selectedSessionId: $selectedSessionId,
                    isSettingsPresented: $isSettingsPresented,
                    modelName: activeModelDisplayName,
                    metalStatus: metalStatus,
                    currentRssGB: currentRssGB,
                    isGenerating: isGeneratingText,
                    onNewChat: {
                        let defModel = localModelManager.getDefaultOrFirstModel()
                        let newSession = ChatSession(
                            title: "New Chat",
                            selectedModelId: defModel?.id,
                            selectedModelName: defModel?.displayName,
                            selectedModelPath: defModel?.snapshotPath
                        )
                        sessions.insert(newSession, at: 0)
                        selectedSessionId = newSession.id
                        applyChatRetention()
                        if let model = defModel, activeLoadedModelPath != model.snapshotPath {
                            switchModel(to: model)
                        }
                    },
                    onDeleteSession: { id in
                        PrefixCacheManager.shared.invalidate(sessionId: id)
                        sessions.removeAll(where: { $0.id == id })
                        // The conversation JSON is cleaned by the next save pass's
                        // orphan sweep, but its asset directory is a separate tree.
                        // Delete it here so a manual delete removes the image bytes
                        // too, as the README promises.
                        ImageAttachmentStore.shared.delete(sessionIds: [id])
                        if sessions.isEmpty {
                            let defModel = localModelManager.getDefaultOrFirstModel()
                            let newSession = ChatSession(
                                title: "New Chat",
                                selectedModelId: defModel?.id,
                                selectedModelName: defModel?.displayName,
                                selectedModelPath: defModel?.snapshotPath
                            )
                            sessions.append(newSession)
                            selectedSessionId = newSession.id
                        } else if selectedSessionId == id {
                            selectedSessionId = sessions.first?.id
                        }
                    }
                )
                .navigationSplitViewColumnWidth(min: max(180, 220 * zoomManager.zoomScale), ideal: max(210, 260 * zoomManager.zoomScale), max: max(260, 320 * zoomManager.zoomScale))
            } detail: {
                ChatDetailView(
                    session: activeSessionBinding,
                    promptText: $chatPromptText,
                    isGenerating: isGeneratingText && (generatingSessionId == (selectedSessionId ?? sessions.first?.id)),
                    isModeChangeLocked: isModeChangeLockedByVisibleTurn,
                    compactionProgressMessageId: (selectedSessionId ?? sessions.first?.id)
                        .flatMap { manualCompactions[$0]?.progressMessageId },
                    isStreamingOffDisk: isStreamingOffDisk,
                    generationSpeed: generationSpeedTokPerSec,
                    generationTokens: generationTotalTokens,
                    jetSpecEnabled: jetSpecEnabled,
                    jetSpecMeanTau: jetSpecMeanTau,
                    jetSpecDraftAccepted: jetSpecTotalDraftAccepted,
                    modelName: activeModelDisplayName,
                    isGemmaDialect: modelConfig?.isGemma4Model == true,
                    tokenizer: tokenizer,
                    activeProfile: activeProfile,
                    onSelectProfile: { profile in
                        applyProfile(profile, for: activeLoadedModelPath ?? localModelManager.defaultModelId)
                    },
                    supportsThinking: activeModelSupportsThinking,
                    isThinkingEnabled: isThinkingEnabledForActiveSession,
                    thinkingEffort: levelCapable ? thinkingEffortForActiveSession : nil,
                    supportsThinkingLevels: levelCapable,
                    supportsVision: activeModelSupportsVision,
                    windowNotice: visibleStatusNotice,
                    isHistoryUnavailable: !sessionStore.isPersistenceAvailable,
                    visionCostLabel: { image in
                        guard let estimate = ModelConfig.visionCostEstimate(
                            modelPath: activeLoadedModelPath,
                            width: image.pixelWidth,
                            height: image.pixelHeight
                        ) else { return nil }
                        // An estimate, not a promise: the model behind it was
                        // measured on this machine class, and the chip says so.
                        return estimate.seconds < 10
                            ? String(format: "about %.1f s", estimate.seconds)
                            : "about \(Int(estimate.seconds.rounded())) s"
                    },
                    isAgentToolsEnabled: isAgentToolsEnabledForActiveSession,
                    isModelLoaded: (tokenizer != nil && summary != nil) || isAppleFoundationModelActive || isCloudModelActive,
                    isAppleFoundationModelActive: isAppleFoundationModelActive,
                    isCloudModelActive: isCloudModelActive,
                    onSendMessage: { prompt, images in
                        handleSendMessage(prompt, preparedImages: images)
                    },
                    onStopGeneration: {
                        stopAutoregressiveGeneration()
                    },
                    onQueuePrompt: { text, images in
                        handleQueuePrompt(text, preparedImages: images)
                    },
                    onSendImmediate: { text, images in
                        await interruptAndSendMessage(text, preparedImages: images)
                    },
                    onRemoveQueuedPrompt: { id in
                        handleRemoveQueuedPrompt(id: id)
                    },
                    onSelectDiscoveredModel: { dm in
                        switchModel(to: dm)
                    },
                    onOpenSettings: {
                        openSettingsWindow()
                    },
                    onSetThinking: { enabled, effort in
                        let resolvedEffort = enabled ? effort : nil
                        defaultThinkingEnabled = enabled
                        if let effort {
                            defaultThinkingEffort = effort.rawValue
                        }
                        if let sid = selectedSessionId ?? sessions.first?.id,
                           let idx = sessions.firstIndex(where: { $0.id == sid }) {
                            let changed = sessions[idx].isThinkingEnabled != enabled
                                || sessions[idx].thinkingEffort != resolvedEffort
                            sessions[idx].isThinkingEnabled = enabled
                            sessions[idx].thinkingEffort = resolvedEffort
                            if changed {
                                // The system line and the assistant prefix both
                                // change from the next turn; drop the pin so the
                                // engine never splices into a stale prefix.
                                PrefixCacheManager.shared.invalidate(sessionId: sid)
                            }
                        }
                    },
                    onToggleAgentTools: { enabled in
                        // Plan mode is an agent mode, not a prompt preference:
                        // with tools off it has no directive, no read tools and
                        // no handoff, so the pill would claim a mode the turn
                        // never runs. Turning the tools off therefore leaves the
                        // mode with them, and a live turn refuses the change for
                        // the same reason it refuses a mode flip.
                        if !enabled,
                           let sid = selectedSessionId ?? sessions.first?.id,
                           let idx = sessions.firstIndex(where: { $0.id == sid }),
                           (sessions[idx].agentMode ?? .code) == .plan {
                            guard !isModeChangeLocked, applyPlanMode(false, sessionId: sid) else {
                                showStatus("⏳ Plan Mode is running — finish the turn before turning agent tools off.", kind: .warning)
                                return
                            }
                            showStatus("⌨️ Tools off — Plan Mode needs the agent's tools, so this conversation is back in Code Mode.", kind: .warning)
                        }
                        defaultAgentToolsEnabled = enabled
                        if let sid = selectedSessionId ?? sessions.first?.id,
                           let idx = sessions.firstIndex(where: { $0.id == sid }) {
                            sessions[idx].isAgentToolsEnabled = enabled
                        }
                    },
                    onSetPlanMode: { enabled in
                        if let sid = selectedSessionId ?? sessions.first?.id {
                            _ = setPlanMode(enabled, sessionId: sid)
                        }
                    },
                    onApprovePlan: { withTurbo in
                        if let sid = selectedSessionId ?? sessions.first?.id {
                            approvePlanAndImplement(sessionId: sid, withTurbo: withTurbo)
                        }
                    },
                    onRenameSession: { newTitle in
                        let trimmed = newTitle.trimmingCharacters(in: .whitespacesAndNewlines)
                        guard !trimmed.isEmpty,
                              let sid = selectedSessionId ?? sessions.first?.id,
                              let idx = sessions.firstIndex(where: { $0.id == sid }) else { return }
                        sessions[idx].title = trimmed
                        sessions[idx].titleOrigin = .user
                    }
                )
            }
            .toolbar(removing: .title)
            .toolbar {
                ToolbarItem(placement: .principal) {
                    HStack(spacing: 6) {
                        Image(systemName: "circle.hexagongrid.circle.fill")
                            .font(.system(size: max(11, 15 * zoomManager.zoomScale)))
                            .foregroundStyle(
                                LinearGradient(
                                    colors: [.purple, .indigo, .blue],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                )
                            )
                        Text("DynaMoE")
                            .font(.system(size: max(11, 14 * zoomManager.zoomScale), weight: .bold))
                    }
                    .padding(.horizontal, 10)
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel("DynaMoE")
                }
            }

            if let hud = zoomManager.hudText {
                VStack {
                    Spacer()
                    HStack {
                        Spacer()
                        Text("Zoom: \(hud)")
                            .font(.system(size: 13, weight: .semibold, design: .rounded))
                            .foregroundColor(.white)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 7)
                            .background(Color.black.opacity(0.8))
                            .clipShape(Capsule())
                            .shadow(color: Color.black.opacity(0.2), radius: 6, x: 0, y: 3)
                            .padding(20)
                            .transition(.opacity.combined(with: .scale(scale: 0.9)))
                    }
                }
                .allowsHitTesting(false)
                .animation(.easeInOut(duration: 0.2), value: zoomManager.hudText)
            }

            // The window's status notice is not drawn here: it is handed to the
            // chat column, which owns the composer it belongs above. Anchored to
            // the window it landed on the composer's own controls, and the
            // composer already reports refusals — two surfaces for one fact.
        }
        .onChange(of: isSettingsPresented) { isPresented in
            if isPresented {
                openSettingsWindow()
                isSettingsPresented = false
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: .openDynaMoESettings)) { _ in
            openSettingsWindow()
        }
        .onChange(of: currentRssGB) { _ in
            syncSettingsWindowIfNeeded()
        }
        .onChange(of: residentExpertCount) { _ in
            syncSettingsWindowIfNeeded()
        }
        .onChange(of: cacheHitRate) { _ in
            syncSettingsWindowIfNeeded()
        }
        .onChange(of: activeLoadedModelPath) { _ in
            syncSettingsWindowIfNeeded()
        }
        .onChange(of: memoryExecutionMode) { newMode in
            applyMemoryExecutionMode(newMode)
            syncSettingsWindowIfNeeded()
        }
        .onChange(of: memoryBudgetMode) { newBudget in
            applyMemoryBudgetMode(newBudget)
            syncSettingsWindowIfNeeded()
        }
        .onAppear {
            if launchPlaceholderId == nil, sessions.count == 1, sessions[0].messages.isEmpty {
                launchPlaceholderId = sessions[0].id
            }
            if selectedSessionId == nil {
                selectedSessionId = sessions.first?.id
            }
            if summary == nil {
                let isTesting = ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil || NSClassFromString("XCTestCase") != nil
                if !isTesting, let initialModel = localModelManager.getDefaultOrFirstModel() {
                    switchModel(to: initialModel)
                }
            } else {
                let initialProfile = ModelProfileManager.shared.getActiveProfile(for: activeLoadedModelPath ?? localModelManager.defaultModelId)
                applyProfile(initialProfile, for: activeLoadedModelPath ?? localModelManager.defaultModelId)
            }
            updatePagingStats()

            // AppKit hands the window's automatic key-view focus to the first
            // control as the window appears, lighting up a focus ring the
            // moment the app opens. Clearing the first responder rests focus on
            // the window instead, so no control is highlighted until the user
            // navigates with the keyboard, where the standard ring returns.
            DispatchQueue.main.async {
                NSApp.keyWindow?.makeFirstResponder(nil)
            }
        }
        .onChange(of: selectedSessionId) { newId in
            // The session that was protected from pruning may have just changed,
            // so re-run retention before switching models.
            applyChatRetention()
            guard let newId = newId, let session = sessions.first(where: { $0.id == newId }) else { return }
            if let targetPath = session.selectedModelPath, !targetPath.isEmpty, activeLoadedModelPath != targetPath {
                if let model = localModelManager.getModel(byId: targetPath) ?? localModelManager.getModel(byId: session.selectedModelId ?? "") {
                    switchModel(to: model)
                } else {
                    loadAndBridgeToMetal(filePath: targetPath)
                    activeLoadedModelPath = targetPath
                    let preferredProfile = ModelProfileManager.shared.getActiveProfile(for: targetPath)
                    applyProfile(preferredProfile, for: targetPath)
                }
            }
        }
        .onChange(of: sessionPersistRevision) { _ in
            // A revision counter rather than the whole `sessions` array: streaming
            // mutates the array every token, and deep-comparing all message text
            // on the main actor each time scales with total history.
            scheduleSessionPersist()
        }
        .onChange(of: chatRetentionLimit) { _ in
            applyChatRetention()
        }
        .onChange(of: chatAutoDeleteEnabled) { _ in
            applyChatRetention()
        }
        .onChange(of: isGeneratingText) { generating in
            // A session kept alive only because it was generating becomes
            // prunable the moment generation stops, so recheck retention then.
            if !generating {
                applyChatRetention()
                // A background/interrupted send may have switched the engine to
                // another conversation's model; restore the selected one so the
                // next ordinary send does not run on the wrong model. Skip while an
                // interrupt replacement or teardown is still mid-flight: those
                // paths intentionally target another conversation's model, and
                // restoring the selection now would undo that switch. The
                // replacement's own end-of-turn change re-runs this later.
                if interruptSendTask == nil && generationTeardownTask == nil {
                    reconcileEngineWithSelection()
                }
            }
        }
        .onChange(of: scenePhase) { phase in
            if phase == .active {
                // Retry a history load that failed or never ran, so a transient
                // read failure does not hide every conversation for the whole run.
                Task { await loadPersistedSessionsIfNeeded() }
            } else {
                flushSessionPersist()
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)) { _ in
            flushSessionPersist()
            // flushSessionPersist only enqueues the write; block until the
            // utility queue drains so the process cannot exit first.
            ChatSessionStore.shared.flushPendingIO()
        }
        .task {
            await loadPersistedSessionsIfNeeded()
        }
        .task {
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                updatePagingStats()
            }
        }
    }

    // MARK: - Independent Settings Window Management
    @ViewBuilder
    private func buildSettingsSheetView() -> some View {
        SettingsSheetView(
            summary: summary,
            modelConfig: modelConfig,
            tokenizer: tokenizer,
            activeModelPath: activeLoadedModelPath,
            metalStatus: metalStatus,
            detectedArchitecture: detectedArchitecture,
            onSelectModel: {
                #if os(macOS)
                selectModelWithOpenPanel()
                #else
                isWeightImporterPresented = true
                #endif
            },
            onSelectTokenizer: {
                #if os(macOS)
                selectTokenizerWithOpenPanel()
                #else
                isTokenizerImporterPresented = true
                #endif
            },
            onLoadDiscoveredModel: { dm in
                switchModel(to: dm)
            },
            temperature: $temperature,
            topP: $topP,
            minP: $minP,
            topK: $topK,
            repetitionPenalty: $repetitionPenalty,
            presencePenalty: $presencePenalty,
            maxNewTokens: $maxNewTokens,
            systemPrompt: $systemPrompt,
            activeProfile: $activeProfile,
            memoryExecutionMode: memoryExecutionModeBinding,
            memoryBudgetMode: memoryBudgetModeBinding,
            kvCachePrecision: kvCachePrecisionBinding,
            speculativePrefetchEnabled: $speculativePrefetchEnabled,
            prefetchLookaheadDepth: $prefetchLookaheadDepth,
            currentRssGB: currentRssGB,
            residentExpertCount: residentExpertCount,
            totalExpertCount: totalExpertCount,
            cacheHitRate: cacheHitRate,
            prefetchEfficiency: prefetchEfficiency,
            lastPagingLatencyMs: lastPagingLatencyMs,
            pagingStatusMessage: pagingStatusMessage,
            onFlushCache: { flushExpertCache() },
            onPreFaultAll: { preFaultAllWeights() },
        )
    }

    private func openSettingsWindow() {
        SettingsWindowManager.shared.show(title: "Settings") {
            buildSettingsSheetView()
        }
    }

    private func syncSettingsWindowIfNeeded() {
        if SettingsWindowManager.shared.isWindowOpen {
            SettingsWindowManager.shared.update {
                buildSettingsSheetView()
            }
        }
    }

    private func handleQueuePrompt(_ text: String, preparedImages: [PreparedImage] = []) -> Bool {
        guard hasAttemptedInitialLoad else { return false }
        guard let currentSessionId = selectedSessionId ?? sessions.first?.id else { return false }
        guard let sessionIdx = sessions.firstIndex(where: { $0.id == currentSessionId }) else { return false }
        // Images are persisted with the queue entry so it stays self-contained
        // across a model switch, and are refused when the current model cannot
        // see them rather than silently queued as text only.
        guard preparedImages.count <= ImageAttachmentPreparer.maxImagesPerMessage else {
            let err = ImageAttachmentError.tooManyImages(limit: ImageAttachmentPreparer.maxImagesPerMessage)
                .errorDescription ?? "Too many images."
            showStatus(err, kind: .warning)
            return false
        }
        if !preparedImages.isEmpty, !supportsVision(for: sessions[sessionIdx]) {
            let err = "⚠️ " + ImageAttachmentMessages.cannotSeeImagesSend(
                modelName: modelDisplayName(for: sessions[sessionIdx])
            )
            showStatus(err, kind: .warning)
            return false
        }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        // An image-only prompt is a real prompt: a queued screenshot must not be
        // rejected for having no caption.
        guard !trimmed.isEmpty || !preparedImages.isEmpty else { return false }
        if !preparedImages.isEmpty, ChatCommand.parse(trimmed) != nil {
            // A slash command is an app action with no message of its own, so its
            // images would be persisted and then silently dropped when the command
            // returns. Refuse before writing and leave the draft intact.
            let err = Self.slashCommandWithImagesMessage
            showStatus(err, kind: .warning)
            return false
        }
        let refs = preparedImages.compactMap { ImageAttachmentStore.shared.write($0, sessionId: currentSessionId) }
        guard refs.count == preparedImages.count else {
            ImageAttachmentStore.shared.remove(refs)
            let err = "⚠️ Could not save the attached image(s) — the prompt was kept in the composer."
            showStatus(err, kind: .warning)
            return false
        }
        sessions[sessionIdx].queuedPrompts.append(QueuedPrompt(
            text: trimmed,
            attachments: refs.isEmpty ? nil : refs
        ))
        return true
    }

    private func handleRemoveQueuedPrompt(id: UUID) {
        // Resolve by identity across every conversation, not by the current
        // selection: an async "Send Now" removes the item after an await, during
        // which the user may have switched chats.
        for idx in sessions.indices {
            guard let prompt = sessions[idx].queuedPrompts.first(where: { $0.id == id }) else { continue }
            sessions[idx].queuedPrompts.removeAll(where: { $0.id == id })
            // The queue entry owned the only reference to these bytes. A "Send
            // Now"/Edit flow has already re-persisted its copy, so dropping the
            // originals cannot break the message that was just sent.
            if let attachments = prompt.attachments, !attachments.isEmpty {
                ImageAttachmentStore.shared.remove(attachments)
            }
        }
    }

    /// Returns whether the prompt was accepted. When a turn is already running
    /// this waits for the cancelled generation to stop and reports the result of
    /// the replacement send, so a caller only clears/removes the prompt once the
    /// new turn has actually started.
    ///
    /// Interrupts are serialized: a second immediate send that arrives while the
    /// first is still waiting for its cancelled task chains onto the in-flight
    /// operation instead of starting its own replacement, which would reset the
    /// shared KV buffers the cancelled task may still be using.
    private func interruptAndSendMessage(_ text: String, preparedImages: [PreparedImage] = []) async -> Bool {
        guard hasAttemptedInitialLoad else { return false }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty || !preparedImages.isEmpty else { return false }
        // Capture the conversation before any suspension, so a send that waits
        // behind an earlier interrupt still lands in the chat that was active
        // when the user pressed send, not whatever is selected once it runs.
        let targetSessionId = selectedSessionId ?? sessions.first?.id
        // Refuse a command that was staged with images *before* dispatching it:
        // the command action has no message to carry them, so running it first
        // would start the action and then leave the images behind.
        if !preparedImages.isEmpty, ChatCommand.parse(trimmed) != nil {
            showStatus(Self.slashCommandWithImagesMessage)
            return false
        }
        // Slash commands are app actions with their own backend, so they dispatch
        // before the interrupt/model-load lifecycle: Cmd-Return and a queued
        // item's "Send Now" reach /compact without loading weights or being
        // refused when an unrelated load fails.
        if let handled = dispatchChatCommand(trimmed, sessionId: targetSessionId) {
            return handled
        }

        let prior = interruptSendTask
        let token = UUID()
        // Pin the target for the whole operation: it can suspend behind an earlier
        // interrupt or a model load, and a selection change during either can
        // trigger retention that prunes this now-unselected session, after which
        // the send fails and its draft or queued prompt is lost. `performInterruptSend`
        // pins again for the generation branch; the count keeps the session guarded
        // past that inner release until this operation finishes.
        if let targetSessionId {
            retainRetentionProtection(targetSessionId)
        }
        let operation = Task { @MainActor in
            defer {
                if let targetSessionId {
                    self.releaseRetentionProtection(targetSessionId)
                    self.applyChatRetention()
                }
            }
            _ = await prior?.value
            let result = await self.performInterruptSend(trimmed, preparedImages: preparedImages, sessionId: targetSessionId)
            if self.interruptSendToken == token {
                self.interruptSendTask = nil
                self.interruptSendToken = nil
            }
            return result
        }
        interruptSendTask = operation
        interruptSendToken = token
        return await operation.value
    }

    /// The body of an interrupt send, run only after any earlier one completes.
    private func performInterruptSend(
        _ trimmed: String,
        preparedImages: [PreparedImage] = [],
        sessionId targetSessionId: UUID?
    ) async -> Bool {
        guard isGeneratingText || generationTeardownTask != nil else {
            // Nothing to interrupt, but the target's model must still be the
            // loaded one (the user may have switched chats since the send was
            // requested), and that load must finish before the turn starts. Refuse
            // the send if the switch was superseded, so it does not run on the old
            // engine.
            guard await ensureModelLoaded(forSession: targetSessionId) else { return false }
            guard !isGeneratingText else { return false }
            return handleSendMessage(trimmed, preparedImages: preparedImages, sessionId: targetSessionId)
        }
        print("⏹ [INT] interrupt requested — cancelling current generation and sending a fresh turn")
        let previous = generationTask
        stopAutoregressiveGeneration()
        let teardown = generationTeardownTask
        // Hold our own pin on the target across the wait: `stopAutoregressiveGeneration`
        // schedules a cleanup that releases its pin once the cancelled task exits,
        // which can run before this replacement send and leave the target prunable.
        // The count keeps the session guarded until the replacement has started.
        if let targetSessionId { retainRetentionProtection(targetSessionId) }
        // Wait for the cancelled generation to actually stop instead of guessing
        // with a fixed delay. Its replacement resets the shared KV cache here in
        // startAutoregressiveGeneration, and with no prefix to preserve that reset
        // drops the buffers and allocates fresh ones, so starting the replacement
        // while the old task is still inside a kernel dispatch would let that task
        // write into the replacement's buffers. Both the decode and prefill layer
        // loops check cancellation at every layer boundary, so this bounds the
        // wait to one layer (one token during decode).
        await previous?.value
        await teardown?.value
        let accepted: Bool
        if isGeneratingText {
            // A different turn started while we were waiting (the user sent
            // another prompt), so starting ours too would run two generations
            // against the shared KV buffers. Report rejection so the caller keeps
            // the draft or queued item.
            accepted = false
        } else {
            // The user may have switched conversations (and models) during the
            // wait; make the target's model active and loaded so the replacement
            // turn runs against the one the conversation expects.
            let modelReady = await ensureModelLoaded(forSession: targetSessionId)
            // The load suspends, so another turn may have started; starting ours
            // too would run two generations against the shared KV buffers. If the
            // switch was superseded the model is not the target's, so also refuse.
            if isGeneratingText || !modelReady {
                accepted = false
            } else {
                accepted = handleSendMessage(trimmed, preparedImages: preparedImages, sessionId: targetSessionId)
            }
        }
        if let targetSessionId {
            releaseRetentionProtection(targetSessionId)
            applyChatRetention()
        }
        return accepted
    }

    private func dequeueAndRunNextPromptIfNeeded(sessionId: UUID?) {
        guard let currentSessionId = sessionId ?? selectedSessionId ?? sessions.first?.id else { return }
        guard let sessionIdx = sessions.firstIndex(where: { $0.id == currentSessionId }) else { return }
        guard !sessions[sessionIdx].queuedPrompts.isEmpty else { return }
        let next = sessions[sessionIdx].queuedPrompts.removeFirst()
        // Pin the conversation across the delay: once generation has ended the
        // retention observer can prune a background conversation during the
        // sleep, which would make the send fail and drop the dequeued prompt.
        retainRetentionProtection(currentSessionId)
        Task { @MainActor in
            // Smooth transition to next turn. The conversation is pinned and the
            // id captured, so a selection change cannot misroute the send.
            try? await Task.sleep(nanoseconds: 80_000_000)
            // A stopped generation may still be winding down; wait for it so the
            // send is not rejected and the already-dequeued prompt is not lost.
            await self.generationTeardownTask?.value
            // A user send can slip in during that wait and start a generation,
            // and the send can still be refused (no model loaded). Neither may
            // start a second generation against the shared KV buffers, and the
            // already-dequeued prompt must not be lost, so put it back at the
            // front of the queue instead.
            if self.isGeneratingText {
                self.requeuePromptFront(next, sessionId: currentSessionId)
            } else {
                let modelReady = await self.ensureModelLoaded(forSession: currentSessionId)
                // The load suspends, so a turn may have started; requeue instead
                // of running two generations against the shared KV buffers. A
                // superseded switch means the model is not the target's, so requeue
                // rather than run on the wrong engine.
                let queuedAttachments = next.attachments ?? []
                let targetSession = self.sessions.first(where: { $0.id == currentSessionId })
                let attachmentsStillVisible = queuedAttachments.isEmpty || self.supportsVision(for: targetSession)
                if !attachmentsStillVisible {
                    // The queued prompt's pictures cannot ride this send, and the
                    // queue entry is about to disappear; drop the bytes with it so
                    // they are not orphaned on disk.
                    ImageAttachmentStore.shared.remove(queuedAttachments)
                }
                // A prompt queued with images can outlive a model switch. Sending
                // its text with a metadata-only notice is honest and terminates;
                // requeueing would retry the same impossible send forever.
                let text = attachmentsStillVisible
                    ? next.text
                    : ([next.text, CloudMessageAssembler.omittedImagesNotice(
                        count: queuedAttachments.count,
                        reason: "model has no vision support"
                    )].filter { !$0.isEmpty }.joined(separator: "\n\n"))
                let accepted = !self.isGeneratingText && modelReady
                    && self.handleSendMessage(
                        text,
                        attachmentRefs: attachmentsStillVisible ? queuedAttachments : [],
                        sessionId: currentSessionId
                    )
                if !accepted {
                    self.requeuePromptFront(next, sessionId: currentSessionId)
                }
            }
            self.releaseRetentionProtection(currentSessionId)
            self.applyChatRetention()
        }
    }

    /// Returns a dequeued prompt to the front of its conversation's queue when
    /// it could not be sent, so an interrupted turn never drops it. Guarded by
    /// identity so a queue edit during the wait cannot duplicate it.
    private func requeuePromptFront(_ prompt: QueuedPrompt, sessionId: UUID) {
        guard let idx = sessions.firstIndex(where: { $0.id == sessionId }) else { return }
        guard !sessions[idx].queuedPrompts.contains(where: { $0.id == prompt.id }) else { return }
        sessions[idx].queuedPrompts.insert(prompt, at: 0)
    }

    // MARK: - Subagent Result Auto-Relay

    /// Collects final reports from this session's completed subagents that have not yet been
    /// surfaced into the conversation. The harness injects these as `[SUBAGENT_RESULT]` blocks
    /// after tool turns so the coordinator model can present them to the user — subagents run
    /// in isolation and the user has no other way to receive their results.
    private func gatherCompletedSubagentRelayResponses(forSession sessionId: UUID?) -> [String] {
        guard let sessionId else { return [] }
        let pending = SubagentManager.shared.allSubagents.filter {
            $0.parentSessionId == sessionId
                && ($0.status == .completed || $0.status == .failed || $0.status == .cancelled)
                && !$0.isSummaryRelayed
                && !$0.finalSummary.isEmpty
        }
        guard !pending.isEmpty else { return [] }
        return pending.map { s in
            let clean = AgentHarness.truncateText(AgentHarness.sanitizeText(s.finalSummary), limit: AgentHarness.charBudget(forTokenBudget: self.maxToolOutputLength))
            return "[SUBAGENT_RESULT] role=\(s.role) id=\(s.id.uuidString) status=\(s.status.rawValue)\n\(clean)"
        }
    }

    /// Marks the given subagent summaries as delivered so the auto-relay does not inject
    /// them into the conversation twice.
    private func markSubagentSummariesRelayed(_ responses: [String]) {
        guard !responses.isEmpty else { return }
        for s in SubagentManager.shared.allSubagents {
            if responses.contains(where: { $0.contains(s.id.uuidString) }) {
                s.isSummaryRelayed = true
            }
        }
    }

    /// Waits until every running/pending subagent for the session reaches a terminal state
    /// or the timeout elapses. Awaited at agent-run teardown so the coordinator's run can
    /// pick up results it would otherwise drop.
    ///
    /// Suspends between polls instead of sleeping: the harness state it reads is main-actor
    /// state, and the weights and cloud teardowns both reach this from a background turn.
    private func awaitSessionSubagents(sessionId: UUID?, timeout: TimeInterval) async {
        guard let sessionId else { return }
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            let stillRunning = SubagentManager.shared.allSubagents.contains {
                $0.parentSessionId == sessionId && ($0.status == .running || $0.status == .pending)
            }
            if !stillRunning { return }
            try? await Task.sleep(nanoseconds: 250_000_000)
        }
    }

    // MARK: - Slash Commands (manual "/compact")

    /// Dispatches a parsed slash command. Commands are app actions rather than
    /// chat turns: nothing is appended as a user message, and true consumes the
    /// composer's text the moment the command is accepted for processing;
    /// outcomes land in the conversation as system notices and the status line.
    private func handleChatCommand(_ command: ChatCommand, sessionIndex sessionIdx: Int, sessionId: UUID) -> Bool {
        switch command {
        case .compact(let focus):
            return runManualCompact(sessionIndex: sessionIdx, sessionId: sessionId, focus: focus)
        case .plan(let enabled):
            // A bare /plan enters plan mode; it does not toggle, so a user who
            // types it twice is still in plan mode rather than silently back in
            // the coder.
            let entering = enabled ?? true
            return setPlanMode(entering, sessionId: sessionId)
        }
    }

    /// True while this view can SEE a turn reading the process-global agent
    /// surface: its own generation, or a stopped turn still unwinding. This is
    /// what the composer's pill reflects, because it is the half SwiftUI
    /// observes.
    private var isModeChangeLockedByVisibleTurn: Bool {
        isGeneratingText || generationTask != nil || generationTeardownTask != nil
    }

    /// True while ANY turn could be reading the process-global agent surface,
    /// including a local-API turn, which shares `AgentHarness.activeMode` and the
    /// grammar registry. The mode belongs to the TURN as well as to the
    /// conversation, so it cannot be changed underneath one.
    ///
    /// The engine's own single-flight state is the one term the view cannot
    /// observe (`CompletionEngine` is not an `ObservableObject`), so it is checked
    /// by the ACTIONS rather than by the pill's disabled state: a server turn
    /// starting under an on-screen pill leaves the pill enabled, and the click
    /// then refuses with a status that says why — rather than a control that
    /// claims a lock it cannot see and stays disabled after the turn is over.
    private var isModeChangeLocked: Bool {
        isModeChangeLockedByVisibleTurn || CompletionEngine.shared.isActive
    }

    /// Applies the conversation's agent surface. Shared by the composer's mode
    /// pill, `/plan` and the plan handoff, because all three must do the same
    /// three things: record the mode, arm the tools plan mode needs, and drop
    /// the KV pin — the system block's plan section and tool list change with
    /// the mode, so a prefix recorded in the other mode is stale.
    ///
    /// Refused while a turn is running. `AgentHarness.activeMode` is
    /// process-global and the live turn's prompt, grammar mask and tool
    /// executions were all built against it, so a flip mid-turn would be a
    /// change to a promise already made.
    @discardableResult
    private func setPlanMode(_ enabled: Bool, sessionId: UUID) -> Bool {
        guard !isModeChangeLocked else {
            showStatus("⏳ Finishing the current turn first — the agent's mode cannot change while it runs.", kind: .warning)
            return false
        }
        // The UI hides the pill on a plain-chat backend, but `/plan` is typed
        // text and would otherwise flip a mode that turn cannot run: the Apple
        // Foundation Model has no tool loop, so the directive, the read tools and
        // the handoff all belong to a surface it is not on.
        if enabled,
           let idx = sessions.firstIndex(where: { $0.id == sessionId }),
           sessionUsesAppleFoundationModel(sessions[idx]) {
            showStatus("🧭 Plan Mode needs a model with tool calling — Apple Foundation Model runs plain chat, so there is nothing to plan with.", kind: .warning)
            return false
        }
        // The same rule for a cloud provider registered without tool support:
        // plan mode's deliverable is a tool-driven investigation ending in a
        // marker-framed plan, and a turn sent as plain chat produces neither. The
        // gate mirrors the send path's `supportsTools ?? true` resolution, so a
        // model this refuses is exactly a model whose plan turn would have no
        // card and no approval bar.
        if enabled,
           let idx = sessions.firstIndex(where: { $0.id == sessionId }),
           cloudModelLacksToolSupport(sessions[idx]) {
            let name = modelDisplayName(for: sessions[idx]) ?? "This cloud model"
            showStatus("🧭 Plan Mode needs a model that can call tools — \(name) is registered without them, so there is nothing to plan with.", kind: .warning)
            return false
        }
        return applyPlanMode(enabled, sessionId: sessionId)
    }

    /// True when the conversation's cloud model is registered without tool
    /// support, i.e. its turns are sent as plain chat.
    private func cloudModelLacksToolSupport(_ session: ChatSession) -> Bool {
        guard let cloud = cloudModelResolution(for: session) else { return false }
        let spec = CloudModelRegistry.registeredSpecs(for: cloud.provider)
            .first { $0.providerModelId == cloud.providerModelId }
        return spec?.supportsTools == false
    }

    /// Ends Plan Mode on every conversation the backend that was just activated
    /// cannot run it for.
    ///
    /// Plan mode is a promise about a TOOL surface, and a switch to the
    /// on-device model (plain chat) or to a cloud model registered without tool
    /// support leaves a conversation in a mode whose pill is hidden, whose turn
    /// runs as plain chat, and whose ready-plan bar would send plain chat when
    /// the user clicked "Start coding". Clearing the mode is the honest repair;
    /// keeping it would be a promise the next turn cannot keep.
    private func retirePlanModeUnsupportedByActiveBackend() {
        var retiredCount = 0
        for idx in sessions.indices where (sessions[idx].agentMode ?? .code) == .plan {
            let unsupported = sessionUsesAppleFoundationModel(sessions[idx])
                || (cloudModelResolution(for: sessions[idx]) != nil
                    && cloudModelLacksToolSupport(sessions[idx]))
            guard unsupported else { continue }
            sessions[idx].agentMode = .code
            PrefixCacheManager.shared.invalidate(sessionId: sessions[idx].id)
            retiredCount += 1
        }
        guard retiredCount > 0 else { return }
        // The harness's surface is process-global and the conversation that
        // asserted it may be one of the ones just retired below the pill, so a
        // mode left at `.plan` would keep the shared grammar on the read-only
        // surface (and its armed read tools loaded) until the next send healed
        // it. Every send re-asserts its own mode, so clearing here costs the
        // turn that follows nothing.
        if AgentHarness.shared.activeMode == .plan {
            AgentHarness.shared.setActiveMode(.code)
        }
        showStatus(
            "🧭 Plan Mode ended — the selected model cannot call tools, so \(retiredCount == 1 ? "this conversation is" : "those conversations are") back in Code Mode.",
            kind: .warning
        )
    }

    /// The write half of `setPlanMode`, without the lock: a caller that has
    /// already established the change is allowed (or is undoing a change of its
    /// own) uses this so a freshly started turn cannot wedge the undo.
    @discardableResult
    private func applyPlanMode(_ enabled: Bool, sessionId: UUID) -> Bool {
        guard let idx = sessions.firstIndex(where: { $0.id == sessionId }) else { return false }
        let target: AgentMode = enabled ? .plan : .code
        if (sessions[idx].agentMode ?? .code) == target { return true }
        sessions[idx].agentMode = target
        if enabled {
            // Plan mode is an AGENT mode: with tools off there is nothing to
            // plan WITH, and the turn would run as plain chat against a prompt
            // that claims a tool surface it was never given.
            sessions[idx].isAgentToolsEnabled = true
        }
        PrefixCacheManager.shared.invalidate(sessionId: sessionId)
        AgentHarness.shared.setActiveMode(target)
        showStatus(enabled
            ? "🧭 Plan Mode: reads and searches only — approve the plan to start coding."
            : "⌨️ Code Mode: the agent can write and run commands again.")
        return true
    }

    /// The plan handoff's confirm action: switch the conversation back to the
    /// coder, optionally arm Turbo, then send the continuation that starts the
    /// implementation. The continuation is an ordinary user turn rather than a
    /// hidden one, so nothing happens in the conversation that the transcript
    /// does not show.
    ///
    /// A refused send UNDOES the mode change. The bar's whole promise is that
    /// approving starts the implementation turn; leaving the conversation in
    /// code mode with nothing sent would break it silently. It also unwinds the
    /// messages the refused send had already appended, because "Try again"
    /// re-sends the continuation: leaving the first copy in the transcript
    /// would duplicate the user turn on the retry.
    private func approvePlanAndImplement(sessionId: UUID, withTurbo: Bool) {
        guard !isModeChangeLocked else {
            showStatus("⏳ The current turn is still running — approve the plan once it finishes.", kind: .warning)
            return
        }
        guard let idx = sessions.firstIndex(where: { $0.id == sessionId }) else { return }
        let previousMode = sessions[idx].agentMode ?? .code
        let turboWasEnabled = UserDefaults.standard.bool(forKey: "dynamoe_agent_turbo_mode")
        // The send appends its user turn and assistant placeholder BEFORE the
        // launch can be refused, so remember where the transcript ended. A
        // refusal that happened earlier (another turn running, a model load in
        // flight) appended nothing and must not cost the conversation two real
        // messages when the unwind below runs.
        let messageCountBeforeSend = sessions[idx].messages.count
        guard applyPlanMode(false, sessionId: sessionId) else { return }
        if withTurbo {
            UserDefaults.standard.set(true, forKey: "dynamoe_agent_turbo_mode")
        }
        guard handleSendMessage("Implement the plan.", sessionId: sessionId) else {
            if sessions[idx].messages.count > messageCountBeforeSend {
                sessions[idx].messages.removeLast(sessions[idx].messages.count - messageCountBeforeSend)
            }
            UserDefaults.standard.set(turboWasEnabled, forKey: "dynamoe_agent_turbo_mode")
            _ = applyPlanMode(previousMode == .plan, sessionId: sessionId)
            showStatus("⚠️ Could not start the implementation turn — the conversation is still in Plan Mode. Try again.", kind: .warning)
            return
        }
    }

    /// Manual compaction via /compact. The Apple Foundation Model folds the
    /// entire conversation into two digests stored on the session: a rolling
    /// general summary that merges forward across compactions, and a detailed
    /// recap of the most recently evicted work, replaced wholesale each pass.
    /// The committed history becomes the visible digest card followed by the
    /// compaction marker: the card keeps the summary readable at the top of the
    /// chat, while both digests are injected into every later prompt on either
    /// backend, so the conversation keeps moving with the essential detail
    /// intact. Because the pass deletes what it summarizes,
    /// the transcript accounts for every evicted turn including tool calls:
    /// a history longer than one summarizer request is split into chunks,
    /// each older chunk folded into the rolling summary by its own pass, and
    /// the newest chunk additionally producing the recap. The general summary
    /// streams live into a progress bubble so the command is visibly working
    /// from the first moment, and the session is rewritten only after all
    /// passes succeed — an unavailable or failed summarizer leaves the
    /// history untouched. The conversation is pinned against chat retention
    /// for the pass's whole lifetime, so a retention sweep cannot delete it
    /// while the summarizer is suspended and the commit silently no-ops
    /// afterwards.
    private func runManualCompact(sessionIndex sessionIdx: Int, sessionId: UUID, focus: String?) -> Bool {
        // Compaction rewrites the message array mid-history: a turn still
        // generating into it, or one still unwinding from a stop, would write
        // into entries the pass is about to drop. Both states are refused
        // loudly — commands report themselves, unlike normal sends, which
        // silently keep the draft. A weights model being loaded does not
        // block: the summarizer is the system model, and a load touches
        // engine state, not the conversation.
        // `generatingSessionId` is not checked here: it keeps the last turn's
        // session after that turn ends, so it would refuse compaction in a
        // conversation forever. The global flag plus the teardown task are the
        // live-turn states.
        if isGeneratingText {
            let note = "⚠️ Stop the running turn before compacting — /compact rewrites the conversation while it works."
            sessions[sessionIdx].messages.append(ChatMessage(role: .system, content: note))
            return true
        }
        if generationTeardownTask != nil {
            let note = "⚠️ The previous turn is still winding down — run /compact again in a moment."
            sessions[sessionIdx].messages.append(ChatMessage(role: .system, content: note))
            return true
        }
        // The composer refuses a second `/compact` for the same conversation
        // before dispatching it; this is the same rule one layer down, so no
        // caller can start a pass over a history another pass is rewriting.
        if manualCompactions[sessionId] != nil {
            let note = "⚠️ A compaction is already running — try /compact again once it finishes."
            sessions[sessionIdx].messages.append(ChatMessage(role: .system, content: note))
            return true
        }
        // The whole array is taken, notices and an earlier compaction's card
        // plus marker included, so the transcript builder's own role filter is
        // the single place that decides what is summarizable: system rows
        // contribute nothing. If nothing has been said, the transcript comes
        // back empty and that is what "nothing to compact" keys on.
        let evicted = sessions[sessionIdx].messages
        guard !evicted.isEmpty else {
            let note = "ℹ️ Nothing to compact — this conversation is still empty."
            sessions[sessionIdx].messages.append(ChatMessage(role: .system, content: note))
            return true
        }
        // A cloud conversation compacts with the model that is already
        // answering it. Requiring the on-device system model here made
        // /compact unavailable on every Mac without Apple Intelligence, even
        // when the conversation's own provider could summarize it perfectly.
        if let cloud = cloudModelResolution(for: sessions[sessionIdx]) {
            return runManualCloudCompact(
                sessionId: sessionId,
                provider: cloud.provider,
                providerModelId: cloud.providerModelId,
                focus: focus
            )
        }
        let availability = AppleFoundationModelService.checkAvailability()
        guard case .ready = availability else {
            let note = "⚠️ Cannot compact — Apple Foundation Model unavailable (\(availability.statusText))."
            sessions[sessionIdx].messages.append(ChatMessage(role: .system, content: note))
            return true
        }
        // Read before any async work: both digests describe pre-pass state,
        // and the live session is re-resolved by identity at commit.
        let previousSummary = sessions[sessionIdx].rollingSummary
        let previousRecap = sessions[sessionIdx].recentWorkDigest
        let summaryInstructions = AppleFoundationModelService.compactionSummaryInstructions(focus: focus)
        let recapInstructions = AppleFoundationModelService.compactionRecapInstructions(focus: focus)
        // Compaction deletes what it summarizes, so the transcript must
        // account for every evicted turn: the chunker includes tool calls
        // with their results and splits long histories into budget-sized
        // pieces instead of silently dropping the oldest past a flat cap.
        // The chunk budget is derived per request — prior digests, pass
        // instructions, and the response reservation are counted first — so
        // a repeat compaction cannot overflow the system model's window the
        // way a fixed chunk size could. Each chunk gets its own summarizer
        // pass below; the newest chunk additionally feeds the detailed
        // recap.
        let chunkBudget = AppleFoundationModelService.compactionChunkCharBudget(
            previousSummary: previousSummary,
            previousRecap: previousRecap,
            instructions: summaryInstructions,
            // Reserve for the larger of the two passes: the recap's response is
            // bigger than the summary's, and the same chunk feeds both.
            maximumResponseTokens: max(
                AppleFoundationModelService.compactionSummaryResponseTokens,
                AppleFoundationModelService.compactionRecapResponseTokens
            )
        )
        // A non-positive budget means the prior digests and the pass
        // instructions already fill the summarizer's window; no chunk can fit,
        // and adding one would only push every pass further past the limit.
        // Refuse with a clear message instead of letting each pass overflow.
        guard chunkBudget > 0 else {
            let note = "⚠️ Cannot compact — the existing summary plus your /compact focus already fill the on-device model's window. Shorten the focus and try again."
            sessions[sessionIdx].messages.append(ChatMessage(role: .system, content: note))
            return true
        }
        let transcriptChunks = AppleFoundationModelService.buildCompactionTranscriptChunks(
            from: evicted,
            excludingMessageIds: [],
            charBudget: chunkBudget
        )
        guard !transcriptChunks.isEmpty else {
            let note = "ℹ️ Nothing to compact — no user or assistant content to summarize yet."
            sessions[sessionIdx].messages.append(ChatMessage(role: .system, content: note))
            return true
        }
        // Only conversation rows count toward the figure the marker and the
        // progress pill show: prior notices, and the card plus marker a previous
        // compaction left behind, are UI state rather than turns.
        let evictedCount = ChatSession.compactableMessageCount(in: evicted)
        // The progress bubble makes the command visible from the instant it is
        // accepted: before any summary text flows it carries the same spinning
        // composing pill a chat turn shows during prefill, then the general
        // summary streams into it at the AFM chat cadence. The commit rewrite
        // below drops it; on failure it becomes the error notice.
        let progressId = UUID()
        sessions[sessionIdx].messages.append(ChatMessage(
            id: progressId,
            role: .system,
            content: "",
            prefillStatus: "🍎 Apple Foundation Model — summarizing \(evictedCount) messages…",
            isTransient: true
        ))
        let compactionRunId = beginManualCompaction(
            sessionId: sessionId,
            progressMessageId: progressId
        )
        // The summarizer suspends for seconds at a time while the user stays
        // free to create or switch chats; an unpinned conversation sitting at
        // the retention limit could be deleted mid-pass, and the commit guard
        // would return silently after the store destroyed the original. Same
        // pin the other long-running session operations hold.
        retainRetentionProtection(sessionId)
        // Progress, not an outcome: a multi-pass compaction runs for minutes and
        // a pill that cleared itself after six seconds would leave the window
        // silent for the rest of it. The task's own defer retracts it.
        let compactionStatusId = showStatus(
            StatusNoticePolicy.compactionText(messageCount: evictedCount),
            kind: .progress,
            sessionId: sessionId
        )
        let compactionTask = Task { @MainActor in
            defer {
                self.finishManualCompaction(sessionId: sessionId, runId: compactionRunId)
                self.releaseRetentionProtection(sessionId)
                self.clearStatus(ifIdentity: compactionStatusId)
            }
            func postToProgress(_ text: String) {
                guard let liveIdx = self.sessions.firstIndex(where: { $0.id == sessionId }),
                      let msgIdx = self.sessions[liveIdx].messages.firstIndex(where: { $0.id == progressId }) else { return }
                // First visible characters retire the composing pill, exactly
                // like a chat turn's transition from ingestion to streaming.
                self.sessions[liveIdx].messages[msgIdx].prefillStatus = nil
                self.sessions[liveIdx].messages[msgIdx].content = text
            }
            // Keeps the composing pill but retitles it: the intermediate
            // passes of a long conversation stay visible as progress without
            // streaming their intermediate digests into the bubble.
            func updateProgressStatus(_ text: String) {
                guard let liveIdx = self.sessions.firstIndex(where: { $0.id == sessionId }),
                      let msgIdx = self.sessions[liveIdx].messages.firstIndex(where: { $0.id == progressId }) else { return }
                self.sessions[liveIdx].messages[msgIdx].prefillStatus = text
            }
            // Appends after the streamed summary instead of replacing it, so the
            // finished general summary stays visible through the recap pass.
            func postToProgressLatest(_ text: String) {
                guard let liveIdx = self.sessions.firstIndex(where: { $0.id == sessionId }),
                      let msgIdx = self.sessions[liveIdx].messages.firstIndex(where: { $0.id == progressId }) else { return }
                self.sessions[liveIdx].messages[msgIdx].content += text
            }
            do {
                // Older chunks merge into the rolling summary one pass each.
                // The previous recap rides along on the first pass only: it
                // describes work older than this transcript, after the first
                // pass it is already folded in, and the recap written at the
                // end replaces it wholesale.
                var mergedSummary = previousSummary
                var mergedRecap = previousRecap
                let olderChunks = transcriptChunks.dropLast()
                for (offset, chunk) in olderChunks.enumerated() {
                    updateProgressStatus(
                        "🍎 Apple Foundation Model — folding older history into the summary (part \(offset + 1) of \(olderChunks.count))…"
                    )
                    let merged = try await AppleFoundationModelService.respondOnce(
                        prompt: AppleFoundationModelService.compactionPrompt(
                            previousSummary: mergedSummary,
                            previousRecap: mergedRecap,
                            evictedTranscript: chunk
                        ),
                        instructions: summaryInstructions,
                        temperature: 0.0,
                        maximumResponseTokens: AppleFoundationModelService.compactionSummaryResponseTokens
                    )
                    // A Stop during the pass must not be read as a finished
                    // digest: the summarizer can return the text it had
                    // produced before the cancellation, and folding that
                    // fragment in would commit a summary whose omissions are
                    // permanent once the rewrite below deletes the messages.
                    try Task.checkCancellation()
                    let mergedText = merged.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !mergedText.isEmpty else {
                        throw AppleFoundationModelUnavailableError(reason: "An older-history digest came back empty.")
                    }
                    mergedSummary = mergedText
                    mergedRecap = nil
                }
                let recentTranscript = transcriptChunks.last!
                let summaryPrompt = AppleFoundationModelService.compactionPrompt(
                    previousSummary: mergedSummary,
                    previousRecap: mergedRecap,
                    evictedTranscript: recentTranscript
                )
                let recapPrompt = "Conversation excerpt, oldest messages first — the most recent work last:\n\n\(recentTranscript)"
                var lastUIWrite = 0.0
                let newSummary = try await AppleFoundationModelService.streamChatTurn(
                    prompt: summaryPrompt,
                    instructions: summaryInstructions,
                    temperature: 0.0,
                    maximumResponseTokens: AppleFoundationModelService.compactionSummaryResponseTokens
                ) { partial in
                    let now = CFAbsoluteTimeGetCurrent()
                    guard now - lastUIWrite >= 0.016 else { return }
                    lastUIWrite = now
                    postToProgress("🗂 Compacting \(evictedCount) messages — general summary so far:\n\n\(partial)")
                }
                // `streamChatTurn` RETURNS the partial text when the task is
                // cancelled (it breaks out of the reveal loop rather than
                // throwing), so without this check a Stop after the first
                // token committed a truncated summary and deleted every
                // message it failed to cover. Checked before the emptiness
                // guard below, because a Stop before the first token would
                // otherwise be reported as "the summary came back empty".
                try Task.checkCancellation()
                let newSummaryText = newSummary.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !newSummaryText.isEmpty else {
                    throw AppleFoundationModelUnavailableError(reason: "The summary came back empty.")
                }
                // Append rather than overwrite: the streamed summary stays
                // visible while the recap pass runs.
                postToProgressLatest("\n\n🗂 Summary written — capturing the detailed recap of the recent work…")
                let newRecap = try await AppleFoundationModelService.respondOnce(
                    prompt: recapPrompt,
                    instructions: recapInstructions,
                    temperature: 0.0,
                    maximumResponseTokens: AppleFoundationModelService.compactionRecapResponseTokens
                )
                // Same rule as the pass above: a cancelled one-shot can still
                // return what it had produced.
                try Task.checkCancellation()
                let newRecapText = newRecap.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !newRecapText.isEmpty else {
                    throw AppleFoundationModelUnavailableError(reason: "The recap came back empty.")
                }
                // Cancellation is checked LAST as well, immediately before the
                // rewrite: every await above suspended long enough for Stop to
                // arrive, and the array below is the destructive step.
                try Task.checkCancellation()
                // Re-resolve by identity: the user may have switched chats while
                // the on-device model worked, and a pruning retention pass may
                // have removed this conversation in the meantime.
                guard let liveIdx = self.sessions.firstIndex(where: { $0.id == sessionId }) else { return }
                // Rows appended while the summarizer ran (a `/plan` notice, a
                // command report) are newer than anything the pass summarized
                // and are not in the snapshot it rewrites from; carry them
                // forward rather than deleting them. The progress bubble is
                // excluded: the rewrite replaces it with the digest card.
                let carried = ChatSession.messagesAdded(
                    to: self.sessions[liveIdx].messages,
                    since: evicted,
                    excluding: [progressId]
                )
                self.sessions[liveIdx].rollingSummary = newSummaryText
                self.sessions[liveIdx].recentWorkDigest = newRecapText
                // The whole conversation (evicted messages and the progress
                // bubble) is replaced in one write by the digest card and the
                // outcome marker. The card keeps the summary readable at the
                // top of the chat; both rows are system-role, so neither
                // reaches a later prompt and the digests carry the context
                // instead.
                self.sessions[liveIdx].messages = ChatSession.compactedHistory(
                    summary: newSummaryText,
                    recap: newRecapText,
                    evictedMessageCount: evictedCount
                ) + carried
                // The context the odometer was counting is gone: reset it to
                // unmetered so the next send seeds a fresh total from the
                // compacted prompt (the digest block in the system prompt plus
                // the new message), which is the conversation's new context
                // size. The card and marker on screen are system rows and are
                // never part of that prompt.
                self.sessions[liveIdx].cumulativeTokenCount = nil
                // The token prefix changed: any pinned KV state for this
                // conversation no longer matches, and reusing it would splice
                // summaries into stale slots. One full re-prefill is the cost
                // of compaction; it is paid here, deliberately.
                PrefixCacheManager.shared.invalidate(sessionId: sessionId)
                self.showStatus(
                    "🗂 Compacted \(evictedCount) messages. The summary stays at the top of this chat.",
                    sessionId: sessionId
                )
            } catch is CancellationError {
                // Stop is an outcome the user asked for, not a fault: nothing
                // was rewritten, so say the conversation is untouched instead
                // of reporting a compaction failure.
                self.showStatus(
                    "⏹ Compaction stopped — the conversation was left unchanged.",
                    sessionId: sessionId
                )
                postToProgress("⏹ Compaction stopped — the conversation was left unchanged.")
            } catch {
                // Scoped to the conversation it happened in: the pill says
                // "the conversation", and an unscoped notice drew it over
                // whichever chat was on screen.
                self.showStatus(
                    "⚠️ Compaction failed: \(error.localizedDescription)",
                    sessionId: sessionId
                )
                postToProgress("⚠️ Compaction failed: \(error.localizedDescription) — the conversation was left unchanged.")
            }
        }
        attachManualCompaction(compactionTask, sessionId: sessionId, runId: compactionRunId)
        return true
    }

    // MARK: - Cloud compaction

    /// What one cloud compaction did: whether the history was rewritten, what
    /// the summarizer requests cost, the tokens they billed, and why nothing
    /// happened when nothing did. Spend and tokens are reported even when the
    /// pass fails, because the provider billed every request that did run; the
    /// run's summary line adds them so its cost and its token totals describe
    /// the same set of requests.
    private struct CloudCompactionOutcome {
        let didCompact: Bool
        let cost: Double
        let promptTokens: Int
        let cachedTokens: Int
        let completionTokens: Int
        let wasEstimated: Bool
        let failure: CloudCompactionFailure?

        static func failed(_ reason: CloudCompactionFailure) -> CloudCompactionOutcome {
            CloudCompactionOutcome(
                didCompact: false,
                cost: 0,
                promptTokens: 0,
                cachedTokens: 0,
                completionTokens: 0,
                wasEstimated: false,
                failure: reason
            )
        }
    }

    /// Summarizes a cloud conversation's older half with the ACTIVE provider
    /// model and rewrites the session in place.
    ///
    /// This is the cloud counterpart of `runManualCompact`, and it exists for
    /// two reasons the on-device summarizer cannot serve: a cloud-only Mac has
    /// no Apple Intelligence to summarize with, and a long agent turn needs a
    /// compaction it can run BETWEEN steps without destroying the assistant
    /// placeholder the turn is streaming into.
    ///
    /// `keepRecentMessages` is where those two callers differ. Auto-compaction
    /// keeps a tail (moved back to a user turn, so a tool exchange is never
    /// split across the boundary); `/compact` keeps nothing, matching the
    /// on-device command's "fold the whole conversation" contract. `focus` is
    /// the `/compact <focus>` steer and applies only to the manual command.
    ///
    /// Returns a failed outcome and leaves the conversation untouched on any
    /// failure: a half-rewritten history is worse than no compaction.
    @discardableResult
    private func compactCloudHistory(
        sessionId: UUID,
        provider: CloudProvider,
        providerModelId: String,
        window: Int?,
        keepRecentMessages: Int,
        focus: String? = nil,
        onProgress: ((String) -> Void)? = nil
    ) async -> CloudCompactionOutcome {
        guard let startIdx = sessions.firstIndex(where: { $0.id == sessionId }) else {
            return .failed(.conversationUnavailable)
        }
        let all = sessions[startIdx].messages
        guard all.count > 2 else { return .failed(.conversationTooShort) }

        // A message boundary is a safe cut: a tool exchange is emitted from one
        // assistant message, so a call can never be separated from its result.
        // The tail starts at a user turn when that cut evicts conversation, at
        // the requested cut when the conversation is a single agent run, and
        // keeps that run's question when the cut would otherwise leave the
        // request opening with an assistant turn.
        guard let plan = CloudCompactionBoundary.plan(
            messages: all,
            keepRecentMessages: keepRecentMessages
        ) else { return .failed(.nothingToEvict) }
        let evicted = plan.evicted
        let kept = plan.kept
        let evictedCount = ChatSession.compactableMessageCount(in: evicted)
        guard evictedCount >= 2 else { return .failed(.nothingToEvict) }

        let client: CloudChatClient
        switch CloudChatClient.makeStoredResult(provider: provider) {
        case .success(let ready):
            client = ready
        case .failure(let missing):
            return .failed(.noCredential(missing.localizedDescription))
        }

        let chunks = AppleFoundationModelService.buildCompactionTranscriptChunks(
            from: evicted,
            excludingMessageIds: [],
            charBudget: CloudContextPolicy.chunkCharacterBudget(window: window)
        )
        guard let recentChunk = chunks.last else { return .failed(.noTranscript) }

        let summaryInstructions = AppleFoundationModelService.compactionSummaryInstructions(focus: focus)
        let recapInstructions = AppleFoundationModelService.compactionRecapInstructions(focus: focus)
        let sessionKey = CloudSessionKey.hashed(sessionId)
        // Scales with the chunk: a fixed cap is too small for a third-of-window
        // chunk, and a summary cut off at the limit is billed and then refused.
        let responseTokens = CloudContextPolicy.summaryResponseTokens(window: window)
        let pricing = CloudPricingStore.pricing(
            forModel: providerModelId,
            providerId: provider.id
        )
        var spent: Double = 0
        var summarizerPromptTokens = 0
        var summarizerCachedTokens = 0
        var summarizerCompletionTokens = 0
        var summarizerWasEstimated = false
        // Summarizer requests are billed whether or not the pass completes, so
        // every exit settles them against the live session before returning.
        func finish(_ failure: CloudCompactionFailure?) -> CloudCompactionOutcome {
            if spent > 0, let liveIdx = sessions.firstIndex(where: { $0.id == sessionId }) {
                sessions[liveIdx].cumulativeCost = (sessions[liveIdx].cumulativeCost ?? 0) + spent
            }
            return CloudCompactionOutcome(
                didCompact: failure == nil,
                cost: spent,
                promptTokens: summarizerPromptTokens,
                cachedTokens: summarizerCachedTokens,
                completionTokens: summarizerCompletionTokens,
                wasEstimated: summarizerWasEstimated,
                failure: failure
            )
        }
        func charge(_ completion: CloudTextCompletion, estimatedPromptTokens: Int) {
            guard let usage = completion.usage else {
                // A pass that reported no usage still ran and still billed: the
                // app's own estimate keeps the run's token figures covering the
                // same requests as its cost, and `(est.)` says so.
                summarizerWasEstimated = true
                summarizerPromptTokens += max(0, estimatedPromptTokens)
                summarizerCompletionTokens += AppleFoundationModelService.recordedOutputTokenCount(
                    for: completion.text
                )
                return
            }
            summarizerPromptTokens += max(0, usage.promptTokens ?? 0)
            summarizerCachedTokens += max(0, usage.cachedTokens ?? 0)
            summarizerCompletionTokens += max(0, usage.completionTokens ?? 0)
            if usage.promptTokens == nil || usage.completionTokens == nil {
                summarizerWasEstimated = true
            }
            guard let pricing, let cost = pricing.cost(for: usage) else { return }
            spent += cost
        }
        // A summarizer request that was sent and did not come back — cancelled
        // mid-stream, or failed on the wire — was still billed for its prompt.
        // Its token counts are unknown, so the app's own estimate stands in and
        // `(est.)` marks the run; charging nothing here is what made a
        // cancelled pass report a cost its own status line called billed.
        func chargeUnreturnedRequest(estimatedPromptTokens: Int) {
            summarizerWasEstimated = true
            summarizerPromptTokens += max(0, estimatedPromptTokens)
            guard let pricing,
                  let cost = pricing.cost(for: CloudUsage(
                      promptTokens: estimatedPromptTokens,
                      completionTokens: 0
                  )) else { return }
            spent += cost
        }
        // One summarizer pass that may replace messages. A summary cut off at
        // the output limit is refused rather than committed: the messages it
        // was going to replace are deleted permanently, and the text that
        // would take their place never covered the end of the chunk.
        func summarize(system: String, prompt: String) async throws -> String {
            let estimatedPromptTokens = CloudMessageAssembler.approximatePromptTokens(
                system: system,
                messages: [CloudChatMessage(role: "user", content: prompt)]
            )
            let completion: CloudTextCompletion
            do {
                completion = try await client.completeText(
                    model: providerModelId,
                    systemPrompt: system,
                    userPrompt: prompt,
                    maxTokens: responseTokens,
                    sessionKey: sessionKey
                )
            } catch {
                // The request was SENT before it failed or was cancelled, so
                // the provider billed its prompt even though nothing came
                // back. `CloudCompactionFailure.spentRequest` says so in the
                // status line; the run's own figures have to agree with it.
                chargeUnreturnedRequest(estimatedPromptTokens: estimatedPromptTokens)
                throw error
            }
            charge(completion, estimatedPromptTokens: estimatedPromptTokens)
            guard !completion.text.isEmpty else { throw CloudCompactionFailure.emptySummary }
            guard !completion.stoppedAtOutputLimit else {
                throw CloudCompactionFailure.truncatedSummary
            }
            return completion.text
        }
        var mergedSummary = sessions[startIdx].rollingSummary
        var mergedRecap = sessions[startIdx].recentWorkDigest

        do {
            let olderChunkCount = max(0, chunks.count - 1)
            for (offset, chunk) in chunks.dropLast().enumerated() {
                onProgress?("🗂 \(provider.displayName) — folding older history into the summary (part \(offset + 1) of \(olderChunkCount))…")
                mergedSummary = try await summarize(
                    system: summaryInstructions,
                    prompt: AppleFoundationModelService.compactionPrompt(
                        previousSummary: mergedSummary,
                        previousRecap: mergedRecap,
                        evictedTranscript: chunk
                    )
                )
                // The recap describes work older than the newest chunk, and it
                // has just been folded into the summary; the pass at the end
                // writes a fresh one.
                mergedRecap = nil
            }

            onProgress?("🗂 \(provider.displayName) — writing the summary…")
            let summaryText = try await summarize(
                system: summaryInstructions,
                prompt: AppleFoundationModelService.compactionPrompt(
                    previousSummary: mergedSummary,
                    previousRecap: mergedRecap,
                    evictedTranscript: recentChunk
                )
            )

            let recapPrompt = "Conversation excerpt, oldest messages first — the most recent work last:\n\n\(recentChunk)"
            onProgress?("🗂 \(provider.displayName) — capturing the detailed recap…")
            let recapEstimatedPromptTokens = CloudMessageAssembler.approximatePromptTokens(
                system: recapInstructions,
                messages: [CloudChatMessage(role: "user", content: recapPrompt)]
            )
            let recap: CloudTextCompletion
            do {
                recap = try await client.completeText(
                    model: providerModelId,
                    systemPrompt: recapInstructions,
                    userPrompt: recapPrompt,
                    maxTokens: responseTokens,
                    sessionKey: sessionKey
                )
            } catch {
                // Same rule as `summarize`: a sent request is billed whether or
                // not its answer arrived.
                chargeUnreturnedRequest(estimatedPromptTokens: recapEstimatedPromptTokens)
                throw error
            }
            charge(recap, estimatedPromptTokens: recapEstimatedPromptTokens)
            // The recap supplements a complete summary rather than replacing
            // anything, so a truncated one costs detail and nothing else; the
            // next compaction asks for a fresh one.

            // Cancellation is checked LAST, immediately before the rewrite: a
            // Stop during any pass must leave the history it was about to
            // replace exactly as it was.
            try Task.checkCancellation()
            // Re-resolve by identity: the user may have switched chats, and a
            // retention sweep may have removed this conversation, while the
            // provider was summarizing.
            guard let liveIdx = sessions.firstIndex(where: { $0.id == sessionId }) else {
                return finish(.conversationUnavailable)
            }
            // A multi-pass compaction runs for minutes, and the transcript is
            // rewritten from the snapshot taken before the first pass. Rows
            // appended since then (a `/plan` notice, a command's report) are
            // newer than everything summarized, so they follow the kept tail
            // rather than being deleted by the rewrite.
            let carried = ChatSession.messagesAdded(
                to: sessions[liveIdx].messages,
                since: all
            )
            sessions[liveIdx].rollingSummary = summaryText
            sessions[liveIdx].recentWorkDigest = recap.text.isEmpty ? nil : recap.text
            sessions[liveIdx].messages = ChatSession.compactedHistory(
                summary: summaryText,
                recap: recap.text.isEmpty ? nil : recap.text,
                evictedMessageCount: evictedCount,
                modelLabel: provider.displayName
            ) + kept + carried
            // The context the odometer counted is gone; the next send seeds a
            // fresh total from the compacted prompt.
            sessions[liveIdx].cumulativeTokenCount = nil
            // The token prefix changed: a pinned KV state for this conversation
            // no longer matches. One full re-prefill is the cost of compaction.
            PrefixCacheManager.shared.invalidate(sessionId: sessionId)
            return finish(nil)
        } catch let reason as CloudCompactionFailure {
            return finish(reason)
        } catch is CancellationError {
            return finish(.cancelled)
        } catch {
            print("☁️ [COMPACT] cloud compaction failed: \(error.localizedDescription)")
            return finish(.provider(error.localizedDescription))
        }
    }

    /// `/compact` for a cloud conversation. The provider model that is already
    /// answering the chat does the summarizing, so the command works with no
    /// on-device system model present.
    private func runManualCloudCompact(
        sessionId: UUID,
        provider: CloudProvider,
        providerModelId: String,
        focus: String?
    ) -> Bool {
        // A multi-pass compaction runs for minutes, and a status pill alone
        // leaves the conversation looking frozen. Mirror the on-device path:
        // a transient bubble where the rewrite will land, retitled as each
        // pass starts, and taken down by the defer when the pass ends.
        let progressId = UUID()
        if let idx = sessions.firstIndex(where: { $0.id == sessionId }) {
            sessions[idx].messages.append(ChatMessage(
                id: progressId,
                role: .system,
                content: "",
                prefillStatus: "🗂 \(provider.displayName) — summarizing the conversation…",
                isTransient: true
            ))
        }
        let compactionRunId = beginManualCompaction(
            sessionId: sessionId,
            progressMessageId: progressId
        )
        retainRetentionProtection(sessionId)
        let compactionStatusId = showStatus(
            "🗂 \(provider.displayName) — compacting this conversation…",
            kind: .progress,
            sessionId: sessionId
        )
        let compactionTask = Task { @MainActor in
            defer {
                self.finishManualCompaction(sessionId: sessionId, runId: compactionRunId)
                // The history rewrite drops the bubble on success; every other
                // outcome leaves it behind, so take it down before the next
                // body evaluation can paint an empty progress row.
                if let liveIdx = self.sessions.firstIndex(where: { $0.id == sessionId }) {
                    self.sessions[liveIdx].messages.removeAll { $0.id == progressId }
                }
                self.releaseRetentionProtection(sessionId)
                self.clearStatus(ifIdentity: compactionStatusId)
            }
            @MainActor func updateProgressStatus(_ text: String) {
                guard let liveIdx = self.sessions.firstIndex(where: { $0.id == sessionId }),
                      let msgIdx = self.sessions[liveIdx].messages.firstIndex(where: { $0.id == progressId })
                else { return }
                self.sessions[liveIdx].messages[msgIdx].prefillStatus = text
            }
            // The same id-based lookup the chat header uses: resolving through
            // `registeredSpecs` falls back to the global window when the
            // conversation's model was never registered, which made /compact
            // size its chunks for a different ceiling than the header showed.
            let window = CloudContextPolicy.window(
                forModelId: providerModelId,
                providerId: provider.id
            )
            let outcome = await self.compactCloudHistory(
                sessionId: sessionId,
                provider: provider,
                providerModelId: providerModelId,
                window: window,
                keepRecentMessages: 0,
                focus: focus,
                onProgress: updateProgressStatus
            )
            if outcome.didCompact {
                let spend = outcome.cost > 0
                    ? " Summarizing cost \(CloudPricingStore.formatted(outcome.cost))."
                    : ""
                self.showStatus(
                    "🗂 Compacted this conversation with \(provider.displayName). The summary stays at the top of the chat.\(spend)",
                    sessionId: sessionId
                )
            } else if outcome.failure == .cancelled {
                // Stop is an outcome the user asked for, not a fault, and the
                // history it interrupted was never touched.
                self.showStatus(
                    "⏹ Compaction stopped — the conversation was left unchanged.",
                    sessionId: sessionId
                )
            } else {
                // Every refusal names its own cause: a missing key, a chat with
                // nothing evictable and an empty summary are not the provider
                // failing to summarize, and each has a different next step.
                let notice = outcome.failure?.notice(providerName: provider.displayName)
                    ?? "⚠️ Cloud compaction failed — the conversation was left unchanged."
                if let liveIdx = self.sessions.firstIndex(where: { $0.id == sessionId }) {
                    self.sessions[liveIdx].messages.append(ChatMessage(
                        role: .system,
                        content: notice
                    ))
                }
                self.showStatus(notice, sessionId: sessionId)
            }
        }
        attachManualCompaction(compactionTask, sessionId: sessionId, runId: compactionRunId)
        return true
    }

    // MARK: - Manual compaction state

    /// Registers a pass before its first await, so the composer's per-session
    /// gate and Stop control can see it from the runloop turn that accepted the
    /// command. Returns the run identity the pass uses to clear it.
    private func beginManualCompaction(sessionId: UUID, progressMessageId: UUID) -> UUID {
        let runId = UUID()
        manualCompactions[sessionId] = ManualCompaction(
            id: runId,
            progressMessageId: progressMessageId
        )
        return runId
    }

    private func attachManualCompaction(_ task: Task<Void, Never>, sessionId: UUID, runId: UUID) {
        guard var run = manualCompactions[sessionId], run.id == runId else { return }
        run.task = task
        manualCompactions[sessionId] = run
    }

    /// Clears the entry only when it still belongs to this run: a superseded
    /// pass must not reopen a gate a newer pass holds.
    private func finishManualCompaction(sessionId: UUID, runId: UUID) {
        guard manualCompactions[sessionId]?.id == runId else { return }
        manualCompactions.removeValue(forKey: sessionId)
    }

    /// Dispatches a composer slash command as an app action, independent of any
    /// model-load or generation gate. Returns nil when `text` is not a
    /// recognized command, so the caller falls through to an ordinary chat
    /// turn. Shared by the plain send path and the immediate-send path so
    /// Cmd-Return and queued "Send Now" reach `/compact` without loading weights
    /// or being refused by an unrelated load.
    private func dispatchChatCommand(_ text: String, sessionId explicitSessionId: UUID?) -> Bool? {
        guard let command = ChatCommand.parse(text) else { return nil }
        guard let cmdSessionId = explicitSessionId ?? selectedSessionId ?? sessions.first?.id,
              let cmdSessionIdx = sessions.firstIndex(where: { $0.id == cmdSessionId }) else { return false }
        // Only `/compact` collides with a running compaction. A mode change is
        // local conversation state — no turn is running — so `/plan` was being
        // refused with a message about a different command.
        if manualCompactions[cmdSessionId] != nil, case .compact = command {
            let note = "⚠️ A compaction is already running — try /compact again once it finishes."
            sessions[cmdSessionIdx].messages.append(ChatMessage(role: .system, content: note))
            return true
        }
        return handleChatCommand(command, sessionIndex: cmdSessionIdx, sessionId: cmdSessionId)
    }

    /// Shared refusal for a slash command that was staged with images.
    private static let slashCommandWithImagesMessage = "⚠️ Slash commands cannot carry images. Remove the attachment, or send the image as a normal message."

    /// Composer entry point for image input: persists the prepared images and
    /// hands the send its refs. Kept separate from the core send so a failed
    /// store write keeps the draft instead of sending text whose picture the
    /// user believes travelled with it.
    private func handleSendMessage(
        _ text: String,
        preparedImages: [PreparedImage],
        sessionId explicitSessionId: UUID? = nil
    ) -> Bool {
        guard !preparedImages.isEmpty else {
            return handleSendMessage(text, sessionId: explicitSessionId)
        }
        guard preparedImages.count <= ImageAttachmentPreparer.maxImagesPerMessage else {
            let err = ImageAttachmentError.tooManyImages(limit: ImageAttachmentPreparer.maxImagesPerMessage)
                .errorDescription ?? "Too many images."
            showStatus(err, kind: .warning)
            return false
        }
        guard let targetId = explicitSessionId ?? selectedSessionId ?? sessions.first?.id,
              let targetIdx = sessions.firstIndex(where: { $0.id == targetId }) else { return false }
        // Checked here as well as in the interrupt/queue entry points because this
        // wrapper is what actually writes bytes: a command must be refused before
        // anything lands on disk, or the composer clears its chips on `accepted`
        // and the picture disappears without a message.
        if ChatCommand.parse(text) != nil {
            let err = Self.slashCommandWithImagesMessage
            showStatus(err, kind: .warning)
            return false
        }
        guard supportsVision(for: sessions[targetIdx]) else {
            let err = "⚠️ " + ImageAttachmentMessages.cannotSeeImagesSend(
                modelName: modelDisplayName(for: sessions[targetIdx])
            )
            showStatus(err, kind: .warning)
            return false
        }
        let refs = preparedImages.compactMap { ImageAttachmentStore.shared.write($0, sessionId: targetId) }
        guard refs.count == preparedImages.count else {
            ImageAttachmentStore.shared.remove(refs)
            let err = "⚠️ Could not save the attached image(s) — the draft was kept so nothing is lost."
            showStatus(err, kind: .warning)
            return false
        }
        let accepted = handleSendMessage(text, attachmentRefs: refs, sessionId: explicitSessionId)
        if !accepted {
            // A refused send (another turn running, model load in flight) wrote
            // bytes that no message references; drop them rather than leave them
            // for the next prune pass.
            ImageAttachmentStore.shared.remove(refs)
        }
        return accepted
    }

    /// Returns whether the prompt was accepted. A `false` result means nothing
    /// was sent, so the composer must keep the text instead of clearing it.
    ///
    /// `sessionId` lets an async caller pin the conversation it resolved before
    /// suspending, so a send started in chat A cannot land in chat B if the user
    /// switches selection while the caller is awaiting.
    ///
    /// `attachmentRefs` are images already persisted to the asset store
    /// (`ImageAttachmentStore`). The dequeued-prompt path uses this directly so a
    /// queued prompt keeps its pictures across the queue round-trip.
    private func handleSendMessage(
        _ text: String,
        attachmentRefs: [ImageAttachmentRef] = [],
        sessionId explicitSessionId: UUID? = nil
    ) -> Bool {
        // Wait for the first load attempt before touching chat state, or the turn
        // could land in the launch placeholder the merge then replaces. The load
        // need not have succeeded: with persistence down the user can still chat,
        // and the merge preserves any in-memory conversation with real messages.
        guard hasAttemptedInitialLoad else { return false }
        // Slash commands are app actions, not chat turns, so they dispatch
        // before every send lifecycle gate: /compact needs no loaded weights
        // model — the system on-device model doing the summarizing is its own
        // backend — and states that would reject it tell the user why instead
        // of silently keeping the draft.
        if let handled = dispatchChatCommand(text, sessionId: explicitSessionId) { return handled }
        // A stopped generation may still be unwinding. Starting now would reset the
        // shared KV buffers the cancelled task can still be using, so reject the
        // send (leaving the draft intact) until its teardown finishes. Async callers
        // that need to send anyway await `generationTeardownTask` first.
        guard generationTeardownTask == nil else { return false }
        // The send paths that must run while another turn is active (interrupt,
        // dequeued prompts) stop it and resolve the model load first, then call
        // here. A plain send reaching this point with a turn running elsewhere
        // (the composer's `isGenerating` is session-scoped, so switching chats
        // exposes the normal send path) would overwrite `generationTask` and race
        // the shared KV buffers, so reject it.
        guard !isGeneratingText else { return false }
        // During a model load the outgoing tokenizer/summary are still non-nil,
        // so a send could run on the old engine; async callers await the specific
        // load before calling here.
        guard !isLoadingModel else { return false }
        guard let currentSessionId = explicitSessionId ?? selectedSessionId ?? sessions.first?.id else { return false }
        guard let sessionIdx = sessions.firstIndex(where: { $0.id == currentSessionId }) else { return false }
        // Attachments are refused, never dropped. A model without vision that
        // silently receives a turn whose picture vanished answers about nothing;
        // returning false keeps the composer's draft and chips intact.
        if !attachmentRefs.isEmpty, !supportsVision(for: sessions[sessionIdx]) {
            let err = "⚠️ " + ImageAttachmentMessages.cannotSeeImagesSend(
                modelName: modelDisplayName(for: sessions[sessionIdx])
            )
            showStatus(err, kind: .warning)
            return false
        }
        // A cloud selection that no longer resolves must refuse here, exactly
        // like `activateCloudModel` does: falling through would run the turn on
        // the installed weights while the conversation still claims a remote
        // model, which is both surprising and a privacy problem.
        if unresolvedCloudSelection(for: sessions[sessionIdx]) {
            let err = "⚠️ This conversation's cloud model could not be resolved. Reselect the model in the model picker."
            showStatus(err, kind: .warning)
            return false
        }
        // A running compaction rewrites its own session's messages when its
        // summaries land; an interleaved send would be part of the rewrite.
        // Other conversations are unaffected and must stay sendable.
        if manualCompactions[currentSessionId] != nil {
            showStatus("⚠️ Conversation is being compacted — try again in a moment.")
            return false
        }
        // Backstop for the send button's isModelLoaded gate: sending before the
        // engine is ready must not touch chat or harness state — a pre-load
        // prompt was observed to poison later sessions (garbled output even in
        // fresh chats) until the app relaunched. The Apple Foundation Model
        // backend has no tokenizer/summary at all, so its sessions must pass
        // this gate on the system-model exemption alone.
        let sendSessionUsesAFM = sessionUsesAppleFoundationModel(sessions[sessionIdx])
        // A connected cloud provider is the third non-weights backend, so it
        // passes the same gate exemption the system model does.
        let sendSessionUsesCloud = cloudModelResolution(for: sessions[sessionIdx]) != nil
        let weightsEngineReady = tokenizer != nil && summary != nil
        guard AppleFoundationModelService.sendGateAllowsSend(hasEngine: weightsEngineReady, sessionUsesSystemModel: sendSessionUsesAFM || sendSessionUsesCloud) else {
            let err = "⚠️ No model loaded — select a model in Settings (bottom left) before sending."
            showStatus(err, kind: .warning)
            return false
        }
        
        let userMsg = ChatMessage(
            role: .user,
            content: text,
            attachments: attachmentRefs.isEmpty ? nil : attachmentRefs
        )
        sessions[sessionIdx].messages.append(userMsg)
        
        // Auto-title session if it's the first user message, unless the user
        // already named this conversation before sending (the rename controls
        // are available on an empty chat), in which case their name wins. An
        // image-only turn has no text to title from, so it seeds "Image".
        if sessions[sessionIdx].messages.filter({ $0.role == .user }).count == 1 {
            if sessions[sessionIdx].titleOrigin != .user {
                let titleSeed = text.isEmpty && !attachmentRefs.isEmpty ? "Image" : text
                sessions[sessionIdx].title = ChatSession.autoTitle(forFirstMessage: titleSeed)
                sessions[sessionIdx].titleOrigin = ChatSession.autoTitleOrigin(forFirstMessage: titleSeed)
            }
            sessions[sessionIdx].createdAt = Date()
        }
        
        // Derive thinking/model from this conversation, not the current
        // selection: an async send may have resumed after the user switched chats.
        let sendSession = sessions[sessionIdx]
        // The conversation's agent surface, resolved once for the whole send.
        // The prompt's tool block, the grammar mask, the cloud `tools` array
        // and every executeTool call read this, so it is set after the send is
        // accepted and before any prompt is assembled.
        let sessionMode = sendSession.agentMode ?? .code
        AgentHarness.shared.setActiveMode(sessionMode)
        let sessionModelName = modelDisplayName(for: sendSession)
        let modelSupportsThinking = supportsThinking(for: sendSession)
        let thinkingEnabled = isThinkingEnabled(for: sendSession) && modelSupportsThinking
        // Graded thinking level (Spark X2.6's `reasoning_effort`): only when the
        // conversation's model is level-capable and thinking is actually on.
        let thinkingEffortLevel = (thinkingEnabled && supportsThinkingLevels(for: sendSession)) ? thinkingEffort(for: sendSession) : nil

        let assistantMsgId = UUID()
        let assistantMsg = ChatMessage(id: assistantMsgId, role: .assistant, content: "", thinkingContent: nil, isThinking: thinkingEnabled)
        sessions[sessionIdx].messages.append(assistantMsg)

        // === Apple Foundation Model backend branch ===
        // The system on-device model is a complete second backend: no Rust engine,
        // tokenizer, Metal buffers, KV/prefix cache, or tool harness. Route the
        // turn to its own generation path before any chat-template formatting,
        // grammar, or agent machinery engages. Plain chat only in this phase.
        if sendSessionUsesAFM {
            let conversationDate = sessions[sessionIdx].createdAt
            let afmSystemPrompt = ModelConfig.buildEffectiveSystemPrompt(
                userPrompt: systemPrompt,
                config: nil,
                summary: nil,
                modelName: AppleFoundationModelService.displayName,
                modelPath: nil,
                currentDate: conversationDate
            ).trimmingCharacters(in: .whitespacesAndNewlines)
            // The persona and newest prompt are not summarizable, so when even
            // they plus the framework's minimum response cannot fit the window
            // the turn is refused with a clear size error instead of being
            // allowed to overflow. Roll back the user/assistant pair appended
            // above so the composer keeps the text and the chat is unchanged.
            guard AppleFoundationModelService.chatTurnFitsWindow(
                systemPrompt: afmSystemPrompt,
                prompt: text
            ) else {
                sessions[sessionIdx].messages.removeLast(2)
                let err = "⚠️ This message is too long for the on-device model's context window. Shorten it and try again."
                showStatus(err, kind: .warning)
                return false
            }
            // History travels in the session instructions as context; only the
            // newest user message is the prompt. Feeding the speaker-labeled
            // transcript as the prompt made the model continue it like a
            // document — emitting "Assistant:" labels and inventing its own
            // "User:" turns in a loop. The slice is whatever the session's
            // context window can spare after the persona, rules, this prompt,
            // and the response ceiling; the transcript builder counts every
            // turn against that budget — including the newest — so history
            // can never be what pushes the turn past the window.
            // The digest is trimmed to whatever the window can spare once the
            // persona, this prompt, and the framework's minimum response are
            // reserved, so the assembled turn can never overflow through the
            // compacted context. The same trimmed value feeds the response
            // ceiling and the instruction builder below.
            let afmCompactionContext = AppleFoundationModelService.chatTurnCompactionContext(
                sessions[sessionIdx].compactionContextBlock,
                systemPrompt: afmSystemPrompt,
                prompt: text
            )
            // The response ceiling is derived from the same window accounting
            // as the history budget: with a large persona, prompt, or compacted
            // digest context riding along, the requested cap could overflow
            // the system model's window with history already floored at zero,
            // so the turn generates a little shorter instead of failing
            // mid-generation.
            let afmResponseTokens = AppleFoundationModelService.chatTurnResponseTokenCeiling(
                requested: maxNewTokens,
                systemPrompt: afmSystemPrompt,
                prompt: text,
                compactionContext: afmCompactionContext
            )
            let afmHistoryBudget = AppleFoundationModelService.historyCharBudget(
                systemPrompt: afmSystemPrompt,
                prompt: text,
                maximumResponseTokens: afmResponseTokens,
                compactionContext: afmCompactionContext
            )
            let historyTranscript = AppleFoundationModelService.buildConversationTranscript(
                from: sessions[sessionIdx].messages,
                excludingMessageIds: [assistantMsgId, userMsg.id],
                charBudget: afmHistoryBudget
            )
            let afmInstructions = AppleFoundationModelService.buildChatTurnInstructions(
                systemPrompt: afmSystemPrompt,
                historyTranscript: historyTranscript,
                compactionContext: afmCompactionContext
            )
            startAppleFoundationModelGeneration(
                sessionId: currentSessionId,
                messageId: assistantMsgId,
                userPrompt: text,
                instructionsText: afmInstructions,
                responseTokenCeiling: afmResponseTokens
            )
            return true
        }

        // === Connected cloud provider backend branch ===
        // A remote provider is a complete third backend: no Rust engine,
        // tokenizer, Metal buffers, or KV/prefix cache. The provider owns chat
        // templating, so the turn is assembled as a messages array. Agent mode
        // rides the provider's NATIVE function calling: schemas travel in the
        // request's `tools`, so the local XML tool dialect prompt must not be
        // injected — only the procedural guidance is.
        if let cloud = cloudModelResolution(for: sendSession) {
            // The compaction digest is deliberately NOT folded in here. An
            // auto-compaction can run between this turn's steps, and a digest
            // captured at send time would be the pre-compaction one for every
            // step after it; `startCloudGeneration` appends the session's live
            // digest to each step's system message instead.
            var cloudSystemPrompt = ModelConfig.buildEffectiveSystemPrompt(
                userPrompt: systemPrompt,
                config: nil,
                summary: nil,
                modelName: sessionModelName,
                modelPath: nil,
                currentDate: sessions[sessionIdx].createdAt
            ).trimmingCharacters(in: .whitespacesAndNewlines)
            let agentToolsRequested = sessions[sessionIdx].isAgentToolsEnabled ?? defaultAgentToolsEnabled
            let cloudSpec = CloudModelRegistry.registeredSpecs(for: cloud.provider)
                .first { $0.providerModelId == cloud.providerModelId }
            let cloudSupportsTools = cloudSpec?.supportsTools ?? true
            let cloudAgentMode = agentToolsRequested && cloudSupportsTools
            if agentToolsRequested && !cloudSupportsTools {
                // Park the notice in the transcript rather than the status
                // line: "this model cannot call tools" is a property of the
                // model that applies to every turn, so it belongs in the
                // conversation where it can be read back, not in a pill that
                // fades after six seconds.
                let notice = "⚠️ \(sessionModelName ?? cloud.providerModelId) is registered without tool support. Sending this turn as plain chat."
                let alreadyNoted = sessions[sessionIdx].messages.contains {
                    $0.role == .system && $0.content == notice
                }
                if !alreadyNoted,
                   let assistantIndex = sessions[sessionIdx].messages.firstIndex(where: { $0.id == assistantMsgId }) {
                    sessions[sessionIdx].messages.insert(
                        ChatMessage(role: .system, content: notice),
                        at: assistantIndex
                    )
                }
            } else if cloudAgentMode {
                AgentHarness.shared.beginAgentSearchGuard()
                cloudSystemPrompt += "\n\n" + AgentHarness.shared.buildCloudToolGuidance(mode: sessionMode)
            }
            return startCloudGeneration(
                sessionId: currentSessionId,
                messageId: assistantMsgId,
                provider: cloud.provider,
                providerModelId: cloud.providerModelId,
                modelDisplayName: sessionModelName
                    ?? CloudModelRegistry.defaultDisplayName(provider: cloud.provider, providerModelId: cloud.providerModelId),
                systemPrompt: cloudSystemPrompt,
                mode: sessionMode,
                isAgentTurn: cloudAgentMode
            )
        }

        // Runs BEFORE the system prompt is built: beginAgentSearchGuard (re)loads
        // `web_search` into the registry, and buildSystemPrompt renders <tools> from
        // the live registry. Calling it after the build left the first prompt of a
        // fresh process advertising only the fundamental tools while the grammar
        // accepted web_search — the model could call a tool the prompt denied having.
        AgentHarness.shared.beginAgentSearchGuard()

        // Build prompt formatted with chat template
        var promptString = ""
        let conversationDate = sessions[sessionIdx].createdAt
        var effectiveSystem = ModelConfig.buildEffectiveSystemPrompt(
            userPrompt: systemPrompt,
            config: modelConfig,
            summary: summary,
            modelName: sessionModelName,
            modelPath: activeLoadedModelPath,
            currentDate: conversationDate
        ).trimmingCharacters(in: .whitespacesAndNewlines)
        // The post-compaction digest travels inside the system block for every
        // chat-template dialect at once: the templates below each wrap
        // `effectiveSystem` verbatim, so appending here is the single injection
        // point for weights sessions.
        if let compactedContext = sessions[sessionIdx].compactionContextBlock {
            effectiveSystem += "\n\n" + compactedContext
        }

        let agentToolsEnabled = (sessions[sessionIdx].isAgentToolsEnabled ?? defaultAgentToolsEnabled)
        if agentToolsEnabled {
            effectiveSystem = AgentHarness.shared.buildSystemPrompt(
                baseSystem: effectiveSystem,
                modelName: sessionModelName,
                currentDate: conversationDate,
                isLingModel: modelConfig?.isLingModel == true,
                isGemma4Model: modelConfig?.isGemma4Model == true,
                isDeepSeekV41Model: modelConfig?.isDeepSeekV41Model == true,
                mode: sessionMode
            )
        }

        let isSpark = (modelConfig?.isSparkModel == true)
        let isLing = (modelConfig?.isLingModel == true)
        let isGemma4 = (modelConfig?.isGemma4Model == true)
        let isDeepSeekV41 = (modelConfig?.isDeepSeekV41Model == true)

        // Pictures in the prompt. Only the family with a vision tower can read
        // pixels this turn; every other local family keeps the metadata notice,
        // because a marker there would expand to rows the model never receives.
        // `turnImages` is what the engine encodes, in prompt order, so it must be
        // filled by the same walk that writes the markers.
        let visionMarkers: DeepSeekV41VisionMarkers? = {
            guard modelConfig?.isDeepSeekV41Model == true,
                  ModelConfig.supportsVision(modelPath: activeLoadedModelPath) else { return nil }
            return DeepSeekV41VisionMarkers.familyDefault
        }()
        var turnImages: [TurnImage] = []
        func attachmentFragment(for message: ChatMessage) -> String {
            guard let refs = message.attachments, !refs.isEmpty else { return "" }
            guard let visionMarkers else {
                return Self.localAttachmentNotice(for: message)
            }
            var fragment = ""
            for ref in refs {
                guard let data = ImageAttachmentStore.shared.data(for: ref),
                      let pixels = try? DeepSeekV41ImagePixels.decode(data) else { continue }
                // The picture goes FIRST inside the turn, the way the reference's
                // prompts place an image before the question about it.
                fragment += visionMarkers.imagePlaceholder + "\n"
                turnImages.append(
                    TurnImage(
                        rgb: pixels.rgb,
                        width: pixels.width,
                        height: pixels.height,
                        sha256: ref.sha256
                    )
                )
            }
            // A decode failure leaves the notice rather than an empty turn: the
            // turn still answers about something, and the user's picture is not
            // silently claimed to have been seen.
            return fragment.isEmpty ? Self.localAttachmentNotice(for: message) : fragment
        }
        if isDeepSeekV41 {
            // The checkpoint's own turn structure, rendered by the same function
            // the API server uses. Building a chatml transcript here and letting
            // `formatPrompt` wrap it (what the default branch does) hands the
            // model ONE user turn holding its whole history as prose: it reads
            // role markers that are not its own, and a tool result arrives as
            // text instead of the output pair it was trained on.
            let dialect = CompletionEngine.shared.promptDialect
            let markers = dialect.markers
            // Markers a stored turn may still carry from an older render. The
            // dialect's own role and sentence markers are stripped for the same
            // reason every other branch strips its own: the renderer re-emits
            // them, and a stored copy would put a turn boundary inside a turn.
            func storedTurnText(_ raw: String, isAssistant: Bool) -> String {
                var text = raw
                    .replacingOccurrences(of: "<|im_start|>", with: "")
                    .replacingOccurrences(of: "<|im_end|>", with: "")
                    .replacingOccurrences(of: "<|endoftext|>", with: "")
                    .replacingOccurrences(of: markers.endOfSentence, with: "")
                if isAssistant {
                    text = text.replacingOccurrences(of: markers.assistant, with: "")
                }
                return text.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            var turns: [ChatPromptFormatter.ConversationTurn] = []
            for msg in sessions[sessionIdx].messages.dropLast() {
                let cleanMsg = storedTurnText(msg.content, isAssistant: msg.role == .assistant)
                let hasTools = (msg.toolCalls != nil && !msg.toolCalls!.isEmpty)
                let hasReasoning = !(msg.thinkingContent ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                let attachmentNotice = Self.localAttachmentNotice(for: msg)
                guard !cleanMsg.isEmpty || hasReasoning || hasTools || !attachmentNotice.isEmpty else { continue }

                if msg.role == .user {
                    turns.append(.init(role: .user, text: attachmentFragment(for: msg) + cleanMsg))
                } else if msg.role == .assistant {
                    // The stored turn keeps its answer and its reasoning apart; the
                    // renderer splits them back out of the flattened canonical form
                    // the API path records, so a turn that carried reasoning keeps
                    // the channel it was generated in.
                    var body = cleanMsg
                    if hasReasoning {
                        let think = msg.thinkingContent!.trimmingCharacters(in: .whitespacesAndNewlines)
                        body = think + TurnTextSplitter.canonicalReasoningCloseTag + body
                    }
                    // A call the stored content no longer carries is rebuilt from
                    // its record, in the dialect the model actually emits. The raw
                    // call survives in `content` for every turn this app produced,
                    // so this only fires for a turn that came from somewhere else.
                    // Only the V4.1 dialect reaches this branch (`if isDeepSeekV41`
                    // above), so the call is always DSML: a Nanbeige test here
                    // could never be true, and the Nanbeige history rebuild lives
                    // in the ChatML branch with the other ChatML families.
                    if let calls = msg.toolCalls, !calls.isEmpty,
                       !cleanMsg.contains(DeepSeekV41ToolDialect.dsmlTag),
                       !cleanMsg.contains("<tool_call") {
                        for call in calls {
                            body += "\n" + DeepSeekV41ToolDialect.callBlock(name: call.name, arguments: call.arguments)
                        }
                    }
                    turns.append(.init(role: .assistant, text: body))
                    // One output pair per executed call, in transcript order. A
                    // recorded output is already model-facing text, so it is used
                    // as-is; the live loop applies the same rendering once.
                    for call in msg.toolCalls ?? [] {
                        guard let output = call.output ?? call.error, !output.isEmpty else { continue }
                        turns.append(.init(
                            role: .user,
                            text: output,
                            kind: .toolResult(name: call.name)
                        ))
                    }
                }
            }
            promptString = ChatPromptFormatter.formatConversation(
                turns: turns,
                cleanSystem: effectiveSystem,
                thinkingEnabled: thinkingEnabled,
                modelSupportsThinking: modelSupportsThinking,
                modelConfig: modelConfig,
                thinkingEffort: thinkingEffortLevel,
                promptDialect: dialect
            )
        } else if isSpark {
            // Spark 2.5 DeepSeek-style dialect:
            // <｜start▁of▁sentence｜><|System|>\n...<｜end▁of▁sentence｜>
            // <｜start▁of▁sentence｜><|User|>...<｜end▁of▁sentence｜>
            // <｜start▁of▁sentence｜><|Bot|></think> / <think>
            // Spark X2.6 graded thinking: the effort instruction line rides the
            // System block (never when thinking is off, per the chat template).
            let sparkSystem = ChatPromptFormatter.sparkSystemBody(
                cleanSystem: effectiveSystem,
                thinkingEnabled: thinkingEnabled,
                thinkingEffort: thinkingEffortLevel
            )
            promptString += "<｜start▁of▁sentence｜><|System|>\n" + sparkSystem + "<｜end▁of▁sentence｜>"
            for msg in sessions[sessionIdx].messages.dropLast() {
                let cleanMsg = msg.content
                    .replacingOccurrences(of: "<｜end▁of▁sentence｜>", with: "")
                    .replacingOccurrences(of: "<｜start▁of▁sentence｜>", with: "")
                    .replacingOccurrences(of: "<|im_end|>", with: "")
                    .replacingOccurrences(of: "<|im_start|>", with: "")
                    .replacingOccurrences(of: "<think>", with: "")
                    .replacingOccurrences(of: "</think>", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                let hasTools = (msg.toolCalls != nil && !msg.toolCalls!.isEmpty)
                let attachmentNotice = Self.localAttachmentNotice(for: msg)
                guard !cleanMsg.isEmpty || (msg.thinkingContent != nil && !msg.thinkingContent!.isEmpty) || hasTools || !attachmentNotice.isEmpty else { continue }

                if msg.role == .user {
                    promptString += "<｜start▁of▁sentence｜><|User|>\(cleanMsg)\(attachmentNotice)<｜end▁of▁sentence｜>"
                } else if msg.role == .assistant {
                    var assistantBody = ""
                    if let think = msg.thinkingContent, !think.isEmpty {
                        let cleanThink = think
                            .replacingOccurrences(of: "<think>", with: "")
                            .replacingOccurrences(of: "</think>", with: "")
                            .trimmingCharacters(in: .whitespacesAndNewlines)
                        assistantBody += "<think>\(cleanThink)</think>"
                    } else {
                        assistantBody += "</think>"
                    }
                    if !cleanMsg.isEmpty {
                        assistantBody += cleanMsg
                    }
                    // History reconstruction in Spark's native function-call dialect
                    if let calls = msg.toolCalls, !calls.isEmpty, !cleanMsg.contains("tool_call>") {
                        for call in calls {
                            assistantBody += "\n<tool_call>\(call.name)"
                            for (k, v) in call.arguments {
                                assistantBody += "<arg_key>\(k)</arg_key><arg_value>\(v)</arg_value>"
                                }
                            assistantBody += "\n</tool_call>"
                        }
                    }
                    promptString += "<｜start▁of▁sentence｜><|Bot|>\(assistantBody)<｜end▁of▁sentence｜>"

                    if let calls = msg.toolCalls, !calls.isEmpty {
                        var outputs: [String] = []
                        for call in calls {
                            if let out = call.output ?? call.error {
                                outputs.append(out)
                            }
                        }
                        if !outputs.isEmpty {
                            promptString += AgentHarness.shared.formatSparkToolResponseTurn(responses: outputs)
                        }
                    }
                }
            }
            if thinkingEnabled {
                promptString += "<｜start▁of▁sentence｜><|Bot|><think>"
            } else {
                promptString += "<｜start▁of▁sentence｜><|Bot|></think>"
            }
        } else if isGemma4 {
            // Gemma 4 canonical chat template:
            //   <bos><|turn>system\n[<|think|>\n]SYSTEM<turn|>\n
            //   <|turn>user\nUSER<turn|>\n
            //   <|turn>model\n[<|channel>thought\n<channel|>]
            promptString += "<bos>"
            if thinkingEnabled {
                promptString += "<|turn>system\n<|think|>\n\(effectiveSystem)<turn|>\n"
            } else if !effectiveSystem.isEmpty {
                promptString += "<|turn>system\n\(effectiveSystem)<turn|>\n"
            }
            for msg in sessions[sessionIdx].messages.dropLast() {
                let cleanMsg = msg.content
                    .replacingOccurrences(of: "<bos>", with: "")
                    .replacingOccurrences(of: "<turn|>", with: "")
                    .replacingOccurrences(of: "<|turn>", with: "")
                    .replacingOccurrences(of: "<|channel>", with: "")
                    .replacingOccurrences(of: "<channel|>", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                let hasTools = (msg.toolCalls != nil && !msg.toolCalls!.isEmpty)
                let attachmentNotice = Self.localAttachmentNotice(for: msg)
                guard !cleanMsg.isEmpty || (msg.thinkingContent != nil && !msg.thinkingContent!.isEmpty) || hasTools || !attachmentNotice.isEmpty else { continue }

                if msg.role == .user {
                    promptString += "<|turn>user\n\(cleanMsg)\(attachmentNotice)<turn|>\n"
                } else if msg.role == .assistant {
                    var assistantBody = ""
                    if let think = msg.thinkingContent, !think.isEmpty {
                        let cleanThink = think
                            .replacingOccurrences(of: "<|channel>", with: "")
                            .replacingOccurrences(of: "<channel|>", with: "")
                            .trimmingCharacters(in: .whitespacesAndNewlines)
                        assistantBody += "<|channel>thought\n\(cleanThink)\n<channel|>"
                    }
                    if !cleanMsg.isEmpty {
                        assistantBody += cleanMsg
                    }
                    // History reconstruction: rebuild the native call when the stored
                    // content no longer carries the raw text, and always replay the tool
                    // results, since Gemma keeps the call and its response in one model
                    // turn (no <turn|> between them).
                    if let calls = msg.toolCalls, !calls.isEmpty {
                        if !cleanMsg.contains("<|tool_call>") {
                            for call in calls {
                                assistantBody += AgentHarness.formatGemmaToolCall(name: call.name, arguments: call.arguments, rawArguments: call.rawArguments)
                            }
                        }
                        // The stored outputs are already-rendered text, not the original JSON,
                        // so the formatter cannot recover the tool name from them; pass the
                        // recorded names alongside so call/response pairs stay valid after a reload.
                        let pairs = calls.compactMap { call -> (name: String, response: String)? in
                            guard let response = call.output ?? call.error else { return nil }
                            return (call.name, response)
                        }
                        if !pairs.isEmpty {
                            assistantBody += AgentHarness.shared.formatGemmaToolResponseTurn(
                                responses: pairs.map { $0.response },
                                toolNames: pairs.map { $0.name },
                                thinkingEnabled: false
                            )
                        }
                    }
                    promptString += "<|turn>model\n\(assistantBody)<turn|>\n"
                }
            }
            if thinkingEnabled {
                promptString += "<|turn>model\n"
            } else {
                promptString += "<|turn>model\n<|channel>thought\n<channel|>"
            }
        } else if isLing {
            let thinkingOption = thinkingEnabled ? "on" : "off"
            if !effectiveSystem.isEmpty {
                promptString += "<role>SYSTEM</role>\(effectiveSystem)\ndetailed thinking \(thinkingOption)<|role_end|>"
            } else {
                promptString += "<role>SYSTEM</role>detailed thinking \(thinkingOption)<|role_end|>"
            }
            for msg in sessions[sessionIdx].messages.dropLast() {
                let cleanMsg = msg.content
                    .replacingOccurrences(of: "<|role_end|>", with: "")
                    .replacingOccurrences(of: "<role>", with: "")
                    .replacingOccurrences(of: "</role>", with: "")
                    .replacingOccurrences(of: "<|im_end|>", with: "")
                    .replacingOccurrences(of: "<|im_start|>", with: "")
                    .replacingOccurrences(of: "<|endoftext|>", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                let hasTools = (msg.toolCalls != nil && !msg.toolCalls!.isEmpty)
                let attachmentNotice = Self.localAttachmentNotice(for: msg)
                guard !cleanMsg.isEmpty || (msg.thinkingContent != nil && !msg.thinkingContent!.isEmpty) || hasTools || !attachmentNotice.isEmpty else { continue }

                if msg.role == .user {
                    promptString += "<role>HUMAN</role>\(cleanMsg)\(attachmentNotice)<|role_end|>"
                } else if msg.role == .assistant {
                    var assistantBody = ""
                    if let think = msg.thinkingContent, !think.isEmpty {
                        let cleanThink = think
                            .replacingOccurrences(of: "<think>", with: "")
                            .replacingOccurrences(of: "</think>", with: "")
                            .trimmingCharacters(in: .whitespacesAndNewlines)
                        assistantBody += "\n<think>\(cleanThink)</think>"
                    } else {
                        assistantBody += "\n<think></think>"
                    }
                    if !cleanMsg.isEmpty {
                        assistantBody += cleanMsg
                    }
                    // History reconstruction: if the stored content no longer carries the raw
                    // Ling-native call text, rebuild it from the structured ToolCallRecords.
                    if let calls = msg.toolCalls, !calls.isEmpty, !cleanMsg.contains("<tool_call>") {
                        for call in calls {
                            assistantBody += "\n<tool_call>\(call.name)"
                            for (k, v) in call.arguments {
                                assistantBody += "\n<arg_key>\(k)</arg_key>\n<arg_value>\(v)</arg_value>"
                            }
                            assistantBody += "\n</tool_call>"
                        }
                    }
                    promptString += "<role>ASSISTANT</role>\(assistantBody)<|role_end|>"

                    if let calls = msg.toolCalls, !calls.isEmpty {
                        var outputs: [String] = []
                        for call in calls {
                            if let out = call.output ?? call.error {
                                outputs.append(out)
                            }
                        }
                        if !outputs.isEmpty {
                            promptString += AgentHarness.shared.formatLingToolResponseTurn(
                                responses: outputs,
                                thinkingEnabled: thinkingEnabled,
                                includeAssistantPrefix: false
                            )
                        }
                    }
                }
            }
            if thinkingEnabled {
                promptString += "<role>ASSISTANT</role>\n<think>"
            } else {
                promptString += "<role>ASSISTANT</role>\n<think></think>"
            }
        } else {
            if !effectiveSystem.isEmpty {
                promptString += "<|im_start|>system\n\(effectiveSystem)<|im_end|>\n"
            }
            for msg in sessions[sessionIdx].messages.dropLast() {
                let cleanMsg = msg.content
                    .replacingOccurrences(of: "<|im_end|>", with: "")
                    .replacingOccurrences(of: "<|im_start|>", with: "")
                    .replacingOccurrences(of: "<|endoftext|>", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                let hasTools = (msg.toolCalls != nil && !msg.toolCalls!.isEmpty)
                let attachmentNotice = Self.localAttachmentNotice(for: msg)
                guard !cleanMsg.isEmpty || (msg.thinkingContent != nil && !msg.thinkingContent!.isEmpty) || hasTools || !attachmentNotice.isEmpty else { continue }

                if msg.role == .user {
                    promptString += "<|im_start|>user\n\(attachmentFragment(for: msg))\(cleanMsg)<|im_end|>\n"
                } else if msg.role == .assistant {
                    var assistantBody = ""
                    if modelSupportsThinking {
                        if let think = msg.thinkingContent, !think.isEmpty {
                            let cleanThink = think
                                .replacingOccurrences(of: "<think>", with: "")
                                .replacingOccurrences(of: "</think>", with: "")
                                .trimmingCharacters(in: .whitespacesAndNewlines)
                            assistantBody += "<think>\n\(cleanThink)\n</think>\n\n"
                        } else {
                            assistantBody += "<think>\n\n</think>\n\n"
                        }
                    }
                    if !cleanMsg.isEmpty {
                        assistantBody += cleanMsg + "\n"
                    }
                    // Rebuild the call only when the stored content no longer carries
                    // it (mirrors the Spark/Ling branches); otherwise the raw call is
                    // already in `cleanMsg` and rebuilding would print it twice.
                    if let calls = msg.toolCalls, !calls.isEmpty, !cleanMsg.contains("tool_call>") {
                        let isNanbeige = (modelConfig?.isNanbeige43Model == true)
                        for call in calls {
                            if isNanbeige {
                                // The family's own formatter owns the tag literals
                                // (AGENTS.md: never retype tag-bearing strings).
                                assistantBody += "\n" + ChatPromptFormatter.nanbeigeToolCallBlock(
                                    name: call.name,
                                    arguments: call.arguments
                                ) + "\n"
                                continue
                            }
                            assistantBody += "<tool_call>\n<function=\(call.name)>\n"
                            for (k, v) in call.arguments {
                                assistantBody += "<parameter=\(k)>\n\(v)\n</parameter>\n"
                            }
                            assistantBody += "</function>\n</tool_call>\n"
                        }
                    }
                    promptString += "<|im_start|>assistant\n\(assistantBody.trimmingCharacters(in: .whitespacesAndNewlines))<|im_end|>\n"

                    if let calls = msg.toolCalls, !calls.isEmpty {
                        var outputs: [String] = []
                        for call in calls {
                            if let out = call.output ?? call.error {
                                outputs.append(out)
                            }
                        }
                        if !outputs.isEmpty {
                            // No assistant prefix from this turn: the loop opens
                            // one per assistant message above, and asking for a
                            // second here put `<|im_start|>assistant\n` in the
                            // rebuilt prompt twice, so a post-tool turn could
                            // never extend the prefix the live continuation had
                            // pinned.
                            promptString += AgentHarness.shared.formatToolResponseTurn(
                                responses: outputs,
                                includeThinkSuffix: false,
                                includeAssistantPrefix: false
                            )
                        }
                    }
                }
            }
            if thinkingEnabled {
                promptString += "<|im_start|>assistant\n<think>\n"
            } else if modelSupportsThinking {
                // Prefill an empty closed <think>\n\n</think>\n\n block to guarantee reasoning models
                // (Nanbeige, DeepSeek-R1, QwQ) bypass reasoning entirely and output the direct answer!
                promptString += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            } else {
                promptString += "<|im_start|>assistant\n"
            }
        }
        // Propagated, not swallowed: a launch refused here has already appended
        // the turn's messages, so a caller that treats the send as accepted
        // would be left in the new mode with nothing running. The plan handoff
        // unwinds those messages and rolls its mode change back on `false`.
        return startAutoregressiveGeneration(
            customPrompt: promptString,
            images: turnImages,
            sessionId: currentSessionId,
            messageId: assistantMsgId
        )
    }

    // MARK: - Tokenizer Execution
    private func loadTokenizer(filePath: String) {
        do {
            let tok = try DynaMoeTokenizer(tokenizerPath: filePath)
            self.tokenizer = tok
            // New tokenizer: drop the grammar sampler's vocab-size-keyed token caches.
            GrammarConstrainedSampler.shared.invalidateTokenizerCaches()
            runTokenization(text: promptInput)
        } catch {
            tokenIDsOutput = "❌ Failed to load tokenizer.json: \(error.localizedDescription)"
        }
    }
    
    private func runTokenization(text: String) {
        guard let tokenizer = tokenizer else {
            tokenIDsOutput = "Load a tokenizer.json file to tokenize text"
            return
        }
        do {
            let ids = try tokenizer.encode(text: text)
            let decoded = try tokenizer.decode(ids: ids)
            tokenIDsOutput = "Tokens (\(ids.count)): \(ids) ➔ Decoded: \"\(decoded)\""
        } catch {
            tokenIDsOutput = "❌ Encoding Error: \(error.localizedDescription)"
        }
    }

    // MARK: - Autoregressive Generation & Sampling Engine

    private func buildCachedLayers(summary: ModelSummary) -> [CachedLayer] {
        return InferenceEngine.shared.buildCachedLayers(summary: summary, config: modelConfig)
    }

    private func sampleNextToken(
        logits: UnsafeMutablePointer<Float>,
        vocabSize: Int,
        contextTokens: [UInt32],
        temperature: Float,
        topP: Float,
        minP: Float,
        topK: Int,
        repetitionPenalty: Float,
        presencePenalty: Float = 0.0,
        eosTokenIds: [UInt32] = [],
        grammarMask: ((UnsafeMutablePointer<Float>, Int) -> Void)? = nil
    ) -> UInt32 {
        return InferenceEngine.sampleNextToken(
            logits: logits,
            vocabSize: vocabSize,
            contextTokens: contextTokens,
            temperature: temperature,
            topP: topP,
            minP: minP,
            topK: topK,
            repetitionPenalty: repetitionPenalty,
            presencePenalty: presencePenalty,
            eosTokenIds: eosTokenIds,
            grammarMask: grammarMask
        )
    }

    private func stopAutoregressiveGeneration() {
        // A manual /compact has no generation attached, so the stop control is
        // the only way to interrupt it. Scoped to the conversation the button
        // was pressed for: compacting chat A and pressing Stop on chat B must
        // not cancel A's pass.
        if let visibleSessionId = selectedSessionId ?? sessions.first?.id,
           let run = manualCompactions[visibleSessionId] {
            // A conversation being compacted is never the one generating —
            // `/compact` refuses to start while a turn runs — so any live
            // generation belongs to a DIFFERENT chat, and falling through to
            // the global teardown below would cancel it and announce it as a
            // stop the user asked for here. The pass reports its own
            // `.cancelled` outcome when the summarizer request settles.
            run.task?.cancel()
            showStatus(
                "⏹ Stopping the compaction — the conversation is left unchanged.",
                sessionId: visibleSessionId
            )
            return
        }
        // Shield the generating session from retention *before* flipping
        // `isGeneratingText` false, since that flip can trigger a retention pass
        // while the cancelled task is still unwinding and appending to it. The
        // session stays protected until the task actually exits, then retention
        // runs again to prune it if it truly fell out of the window.
        if let cancelled = generationTask, let sessionId = generatingSessionId {
            retainRetentionProtection(sessionId)
            let teardown = Task { @MainActor in
                await cancelled.value
                self.releaseRetentionProtection(sessionId)
                self.applyChatRetention()
                self.generationTeardownTask = nil
                // The isGeneratingText observer deliberately skips reconciliation
                // while a teardown is pending, and this clearing produces no later
                // state change, so restore the selected conversation's model here
                // unless an interrupt replacement is already taking over. Otherwise
                // a stopped background turn would leave the engine on its model and
                // the next ordinary send would run on it.
                if self.interruptSendTask == nil {
                    self.reconcileEngineWithSelection()
                }
            }
            generationTeardownTask = teardown
        }
        isGeneratingText = false
        // Cancel the engine's compute task too: the consumer task only relays
        // events, the Metal/KV work lives in CompletionEngine's detached turn.
        CompletionEngine.shared.cancelActiveTurn()
        generationTask?.cancel()
        generationTask = nil
        // The turn is over, so a continuation that somehow arrives afterwards
        // must not inherit its mode. The next send resolves a fresh one at step 0.
        activeTurnMode = nil
        ActionApprovalManager.shared.cancelAll()
        showStatus("⏹ Generation stopped by user.")
    }

    /// True when the given generation still owns the shared generation state.
    /// Cancellation alone is not enough to mutate that state: `interruptAndSendMessage`
    /// starts a replacement shortly after cancelling, so a cancelled task that
    /// clears `generationTask`/`isGeneratingText` would silently disarm the
    /// replacement (Stop would no longer cancel it). Call from the main actor.
    private func ownsGeneration(_ id: UInt64) -> Bool {
        id == generationId
    }

    // MARK: - Apple Foundation Model (System On-Device) Generation

    /// One chat turn on Apple's system on-device Foundation Model (macOS 26+,
    /// Apple Intelligence enabled). Mirrors the UI contract of
    /// `startAutoregressiveGeneration` — generation identity
    /// (`ownsGeneration`), streaming message updates, stop-button cancellation,
    /// and queued-prompt draining — without any of the weights/Metal/KV-cache
    /// machinery. Plain chat only: no tool parsing, grammar masking, JetSpec, or
    /// prefix caching. Token counts are character-based approximations because
    /// the macOS 26 streaming API does not report per-chunk token usage.
    /// Returns true once the turn is started (or its failure was recorded into
    /// the placeholder message), matching `handleSendMessage`'s contract.
    @discardableResult
    private func startAppleFoundationModelGeneration(
        sessionId: UUID,
        messageId: UUID,
        userPrompt: String,
        instructionsText: String,
        responseTokenCeiling: Int? = nil
    ) -> Bool {
        let availability = AppleFoundationModelService.checkAvailability()
        guard case .ready = availability else {
            let err = "⚠️ Apple Foundation Model unavailable — \(availability.statusText)"
            if let sIdx = sessions.firstIndex(where: { $0.id == sessionId }),
               let mIdx = sessions[sIdx].messages.firstIndex(where: { $0.id == messageId }) {
                sessions[sIdx].messages[mIdx].content = err
                sessions[sIdx].messages[mIdx].isThinking = false
            }
            showStatus(err, kind: .warning)
            return true
        }

        generationId &+= 1
        let myGenerationId = generationId
        isGeneratingText = true
        generatingSessionId = sessionId
        generatedStreamText = ""
        thinkingText = ""
        responseText = ""
        isThinking = false
        generationTotalTokens = 0
        generationElapsedMs = 0.0
        generationSpeedTokPerSec = 0.0
        // Progress, not an outcome: the on-device model spools up its session and
        // ingests the prompt before any content flows, which is longer than the
        // six seconds an informational pill lives, so the window would go silent
        // while the work it named was still running.
        let afmStatusId = showStatus(
            StatusNoticePolicy.appleFoundationModelGeneratingText,
            kind: .progress,
            sessionId: sessionId
        )
        if let sIdx = sessions.firstIndex(where: { $0.id == sessionId }),
           let mIdx = sessions[sIdx].messages.firstIndex(where: { $0.id == messageId }) {
            // Same plumbing the weights path uses during prefill: keeps the
            // header pill and the in-bubble spinner lit through the on-device
            // model's session spin-up and prompt ingestion before content flows.
            sessions[sIdx].messages[mIdx].prefillStatus = "🍎 Apple Foundation Model — composing…"
        }

        // The active profile drives temperature (clamped to the model's 0…1
        // range) and the response ceiling (clamped to its small context);
        // penalties, Min-P, and Top-P have no on-device equivalent and are ignored.
        // A caller that derived a window-fitting ceiling passes it; nil keeps
        // the user's setting.
        let profileTemperature = Double(temperature)
        let responseCeiling = responseTokenCeiling ?? maxNewTokens
        let generationStartTime = CFAbsoluteTimeGetCurrent()

        // This Task is deliberately non-detached: it inherits the enclosing
        // @MainActor (this target defaults actor isolation to MainActor), so
        // every state write below — including inside the onPartial callback,
        // which runs synchronously on this same task — is statically
        // main-actor-isolated and needs no MainActor.run hops. The weights
        // loop uses Task.detached because its per-token Metal work must not
        // occupy the main actor; this path only suspends on the on-device
        // model's snapshots between paced reveals, so nothing blocks here.
        // The framework re-sends the whole context (persona + history
        // transcript) every turn and reports no token accounting, so the
        // session odometer's input side uses the same character-based
        // approximation. The stored figure is the CONTEXT SIZE, like every
        // other backend (see `ChatSession.contextTokenBase`): this turn's
        // assembled input, not the previous total plus it, which counted the
        // same history again on every turn and jumped by a whole conversation
        // when a chat moved here from a cloud model. Streaming and
        // finalization add the generated tokens on top of this base.
        let afmTurnInputTokens = AppleFoundationModelService.approximateTokenCount(for: instructionsText)
            + AppleFoundationModelService.approximateTokenCount(for: userPrompt)
        let afmCumulativeBase = ChatSession.contextTokenBase(promptTokenCount: afmTurnInputTokens)
        if let sIdx = sessions.firstIndex(where: { $0.id == sessionId }) {
            sessions[sIdx].cumulativeTokenCount = afmCumulativeBase
        }

        // In-app turn log for the on-device path — the same row contract as
        // a weights turn (see `startAutoregressiveGeneration`): success,
        // failure, and stop each record exactly one row, bodies follow the
        // Store Request Content toggle, and the streamed partials reach the
        // row on a stop or failure. The availability pre-flight above stays
        // unlogged like an engine launch failure: the chat surface owns that
        // error display. The request-side analog of the weights path's
        // rendered prompt is what the framework actually sees this turn:
        // the assembled instructions plus the user message.
        let turnLog = ServerRequestLog.chatTurnLog(
            model: AppleFoundationModelService.displayName,
            inputTokens: afmTurnInputTokens,
            isAgentTurn: false,
            prompt: instructionsText.isEmpty ? userPrompt : instructionsText + "\n\n" + userPrompt
        )

        generationTask = Task(priority: .userInitiated) {
            // The progress pill lives exactly as long as this task. The success
            // path replaces it with the outcome status, which has its own
            // identity, so retracting here cannot take that down — it only stops
            // a stopped or failed turn leaving a spinner behind.
            defer { self.clearStatus(ifIdentity: afmStatusId) }
            var latestPartial = ""
            var sawFirstPartial = false
            var firstPartialAt = 0.0
            var lastUIUpdateTime = 0.0

            // Cancellation and error exits below bypass the success finalizer,
            // so the odometer would otherwise be left at the last throttled
            // streaming write and drop the tail of the final snapshot.
            // Committing here — only while this task still owns the generation,
            // which holds because a replacement cannot start until this task
            // returns — keeps stopped and failed turns fully counted.
            func commitOdometerTail() {
                guard !latestPartial.isEmpty, self.ownsGeneration(myGenerationId),
                      let sIdx = self.sessions.firstIndex(where: { $0.id == sessionId }) else { return }
                self.sessions[sIdx].cumulativeTokenCount = afmCumulativeBase
                    + AppleFoundationModelService.approximateTokenCount(for: latestPartial)
            }

            do {
                latestPartial = try await AppleFoundationModelService.streamChatTurn(
                    prompt: userPrompt,
                    instructions: instructionsText.isEmpty ? nil : instructionsText,
                    temperature: profileTemperature,
                    maximumResponseTokens: responseCeiling
                ) { partialText in
                    latestPartial = partialText
                    // The row's first-token stamp, mirroring the weights loop's
                    // first frame (first caller wins; an empty snapshot still
                    // proves the model began answering).
                    turnLog.noteFirstToken()
                    // 60fps-throttled streaming UI updates, mirroring the weights path.
                    let now = CFAbsoluteTimeGetCurrent()
                    if now - lastUIUpdateTime < 0.016 { return }
                    lastUIUpdateTime = now
                    if !sawFirstPartial {
                        sawFirstPartial = true
                        firstPartialAt = now
                    }
                    guard self.ownsGeneration(myGenerationId) else { return }
                    let elapsed = max(now - generationStartTime, 0.001)
                    let approxTokens = AppleFoundationModelService.approximateTokenCount(for: partialText)
                    let tokPerSec = Double(approxTokens) / elapsed
                    self.generatedStreamText = partialText
                    self.responseText = partialText
                    self.generationTotalTokens = approxTokens
                    self.generationElapsedMs = elapsed * 1000.0
                    self.generationSpeedTokPerSec = tokPerSec
                    if let sIdx = self.sessions.firstIndex(where: { $0.id == sessionId }),
                       let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == messageId }) {
                        if self.sessions[sIdx].messages[mIdx].prefillStatus != nil {
                            // First visible characters: retire the composing pill.
                            self.sessions[sIdx].messages[mIdx].prefillStatus = nil
                        }
                        self.sessions[sIdx].messages[mIdx].content = partialText
                        // Keep activity fresh so retention sees the streaming turn.
                        self.sessions[sIdx].messages[mIdx].timestamp = Date()
                        self.sessions[sIdx].messages[mIdx].tokenCount = approxTokens
                        self.sessions[sIdx].messages[mIdx].tokensPerSec = tokPerSec
                        self.sessions[sIdx].messages[mIdx].timeToFirstTokenSeconds = max(firstPartialAt - generationStartTime, 0.0)
                        // Keep the odometer ticking with the streamed tokens so
                        // the header counter grows live like the weights path.
                        // Empty snapshots (the speaker-label retraction reset)
                        // carry the UI-stat floor of one token, so they must
                        // contribute zero here.
                        self.sessions[sIdx].cumulativeTokenCount = afmCumulativeBase + (partialText.isEmpty ? 0 : approxTokens)
                    }
                }
            } catch {
                // The turn ends here instead of in the success finalizer below —
                // stopped, superseded, or errored — so the odometer needs the
                // tail of the final snapshot either way.
                commitOdometerTail()
                // Stop-button cancellation throws out of the stream iteration;
                // whatever partial text already streamed stays in the message.
                if Task.isCancelled || !self.ownsGeneration(myGenerationId) {
                    // Same cancelled-turn contract as the weights path: a user
                    // stop is informational (200 with the stop message), never
                    // an error; a stream closing some other way before a result
                    // is exceptional, and still owes the row its partial text.
                    turnLog.recordChatTurn(
                        status: Task.isCancelled ? 200 : 500,
                        errorMessage: Task.isCancelled ? "⏹ Generation stopped by user." : "Generation stream closed without a result",
                        outputTokens: AppleFoundationModelService.recordedOutputTokenCount(for: latestPartial),
                        finishReason: nil,
                        thinking: "",
                        response: latestPartial,
                        toolCalls: []
                    )
                    return
                }
                let errText = "⚠️ Apple Foundation Model error: \(error.localizedDescription)"
                self.isGeneratingText = false
                self.generationTask = nil
                self.showStatus(errText, kind: .warning)
                if let sIdx = self.sessions.firstIndex(where: { $0.id == sessionId }),
                   let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == messageId }) {
                    let partialPrefix = latestPartial.isEmpty ? "" : latestPartial + "\n\n"
                    self.sessions[sIdx].messages[mIdx].prefillStatus = nil
                    self.sessions[sIdx].messages[mIdx].content = partialPrefix + errText
                    self.sessions[sIdx].messages[mIdx].isThinking = false
                }
                // The row keeps the failure shape the weights path uses: the
                // error text the chat surface displays, plus the partial text
                // generated before the failure (its approximate token count
                // stands in for the engine's exact count — the macOS 26
                // streaming API reports no token usage).
                turnLog.recordChatTurn(
                    status: 500,
                    errorMessage: errText,
                    outputTokens: AppleFoundationModelService.recordedOutputTokenCount(for: latestPartial),
                    finishReason: nil,
                    thinking: "",
                    response: latestPartial,
                    toolCalls: []
                )
                // The turn failed, but queued prompts must not wait on a
                // generation that will never complete: drain exactly like the
                // success path does.
                self.dequeueAndRunNextPromptIfNeeded(sessionId: sessionId)
                return
            }

            // A cancelled or superseded task must not finalize: the replacement
            // generation owns the UI flags and task handle now (see
            // `ownsGeneration`). The streamed partial text already landed in the
            // message and is intentionally preserved.
            if Task.isCancelled || !self.ownsGeneration(myGenerationId) {
                commitOdometerTail()
                turnLog.recordChatTurn(
                    status: Task.isCancelled ? 200 : 500,
                    errorMessage: Task.isCancelled ? "⏹ Generation stopped by user." : "Generation stream closed without a result",
                    outputTokens: AppleFoundationModelService.recordedOutputTokenCount(for: latestPartial),
                    finishReason: nil,
                    thinking: "",
                    response: latestPartial,
                    toolCalls: []
                )
                return
            }

            let finalElapsed = max(CFAbsoluteTimeGetCurrent() - generationStartTime, 0.001)
            let finalText = latestPartial
            let approxTokens = AppleFoundationModelService.approximateTokenCount(for: finalText)
            let finalTokPerSec = Double(approxTokens) / finalElapsed

            self.isGeneratingText = false
            self.generationTask = nil
            self.generatedStreamText = finalText
            self.responseText = finalText
            self.generationTotalTokens = approxTokens
            self.generationElapsedMs = finalElapsed * 1000.0
            self.generationSpeedTokPerSec = finalTokPerSec
            self.showStatus("🍎 Generated ~\(approxTokens) tokens in \(String(format: "%.2f", finalElapsed * 1000.0)) ms (\(String(format: "%.1f", finalTokPerSec)) tok/s)")

            if let sIdx = self.sessions.firstIndex(where: { $0.id == sessionId }),
               let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == messageId }) {
                self.sessions[sIdx].messages[mIdx].prefillStatus = nil
                self.sessions[sIdx].messages[mIdx].content = finalText
                self.sessions[sIdx].messages[mIdx].timestamp = Date()
                self.sessions[sIdx].messages[mIdx].isThinking = false
                self.sessions[sIdx].messages[mIdx].tokenCount = approxTokens
                self.sessions[sIdx].messages[mIdx].tokensPerSec = finalTokPerSec
                // An empty response floors at one token for the UI stats, but
                // the odometer must not count a token that was never generated.
                self.sessions[sIdx].cumulativeTokenCount = afmCumulativeBase + (finalText.isEmpty ? 0 : approxTokens)
                if sawFirstPartial {
                    self.sessions[sIdx].messages[mIdx].timeToFirstTokenSeconds = max(firstPartialAt - generationStartTime, 0.0)
                }
            }
            // Success: one 200 row with the turn's final text and its
            // approximate token count. No finish reason and no thinking —
            // plain chat is the only on-device mode, and the framework
            // reports neither a stop reason nor token usage.
            turnLog.recordChatTurn(
                status: 200,
                errorMessage: nil,
                outputTokens: AppleFoundationModelService.recordedOutputTokenCount(for: finalText),
                finishReason: nil,
                thinking: "",
                response: finalText,
                toolCalls: []
            )
            self.dequeueAndRunNextPromptIfNeeded(sessionId: sessionId)
        }
        return true
    }

    // MARK: - Connected Cloud Provider Generation

    /// One chat turn on a connected cloud provider.
    ///
    /// Mirrors the UI contract of `startAppleFoundationModelGeneration` —
    /// generation identity (`ownsGeneration`), streaming message updates,
    /// stop-button cancellation, and queued-prompt draining — without any
    /// weights, Metal, or KV-cache machinery. The provider owns templating, so
    /// history travels as a messages array and is re-assembled from the session
    /// on every step, which is also what replays the tool exchange.
    ///
    /// With `isAgentTurn`, the turn runs the agent loop through the provider's
    /// NATIVE function calling instead of a parsed XML dialect: loaded tool
    /// schemas ride the request, streamed `tool_calls` are committed to the
    /// assistant message, each call gates on the same HITL approval and
    /// executes through `AgentHarness`, and the next step continues until the
    /// model answers, calls `complete`, a loop guard trips, or it exhausts the
    /// step budget (`agentStepCap`; no cap at all when "no limit" is on). The whole
    /// loop lives in ONE task, so Stop aborts streaming, approvals, and tool
    /// execution together.
    @discardableResult
    private func startCloudGeneration(
        sessionId: UUID,
        messageId: UUID,
        provider: CloudProvider,
        providerModelId: String,
        modelDisplayName: String,
        systemPrompt: String,
        mode: AgentMode,
        isAgentTurn: Bool = false
    ) -> Bool {
        let displayName = modelDisplayName.isEmpty ? providerModelId : modelDisplayName
        let client: CloudChatClient
        switch CloudChatClient.makeStoredResult(provider: provider) {
        case .success(let ready):
            client = ready
        case .failure(let failure):
            let err = "⚠️ \(failure.localizedDescription)"
            if let sIdx = sessions.firstIndex(where: { $0.id == sessionId }),
               let mIdx = sessions[sIdx].messages.firstIndex(where: { $0.id == messageId }) {
                sessions[sIdx].messages[mIdx].content = err
                sessions[sIdx].messages[mIdx].isThinking = false
            }
            showStatus(err, kind: .warning)
            // A provider with no usable client is a send that never happened,
            // so the caller must be able to undo what it changed (the plan
            // handoff's mode switch) rather than read it as accepted work.
            return false
        }

        generationId &+= 1
        let myGenerationId = generationId
        isGeneratingText = true
        generatingSessionId = sessionId
        // Ownership stamps the harness like the weights path does: a
        // `spawn_subagent` issued by this turn is parented to THIS session.
        // Without it the subagent inherits whatever local conversation ran
        // last, so its report is relayed into the wrong transcript — or never
        // relayed at all, because the relay filters on `parentSessionId`.
        AgentHarness.shared.currentSessionId = sessionId
        generatedStreamText = ""
        thinkingText = ""
        responseText = ""
        // Cloud reasoning is the provider's own default; the turn starts
        // without a thinking spinner and only lights one when real
        // `reasoning_content` deltas arrive.
        isThinking = false
        generationTotalTokens = 0
        generationElapsedMs = 0.0
        generationSpeedTokPerSec = 0.0
        showStatus("☁️ \(provider.displayName) — generating…")
        if let sIdx = sessions.firstIndex(where: { $0.id == sessionId }),
           let mIdx = sessions[sIdx].messages.firstIndex(where: { $0.id == messageId }) {
            sessions[sIdx].messages[mIdx].prefillStatus = "☁️ \(provider.displayName) — contacting the model…"
        }

        let params = CloudChatGenerationParams(
            temperature: Double(temperature),
            topP: Double(topP),
            presencePenalty: Double(presencePenalty),
            maxTokens: maxNewTokens > 0 ? maxNewTokens : nil,
            // The affinity/cache-key value is a hash, never the UUID: a
            // provider only needs a stable opaque token, and an app identifier
            // in a third party's logs buys nothing.
            sessionKey: CloudSessionKey.hashed(sessionId)
        )
        // The persona rides as a leading system message; the provider's own
        // templating consumes it verbatim. The compaction digest is composed
        // per request rather than captured here: an auto-compaction can run
        // between this turn's steps, and a digest frozen at send time would
        // leave every step after it describing a history that is already gone.
        let personaBase = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        func currentPersona() -> String {
            guard let digest = sessions.first(where: { $0.id == sessionId })?.compactionContextBlock,
                  !digest.isEmpty else { return personaBase }
            return personaBase.isEmpty ? digest : personaBase + "\n\n" + digest
        }
        // Loaded tools only, read fresh on each step so a `tools_load` issued
        // inside the loop is visible to the next request. The full catalog stays
        // discoverable through `tools_discover`; every schema sent costs prompt
        // tokens on every request of the loop.
        let toolsEnabled = isAgentTurn

        let openingMessages = CloudMessageAssembler.assemble(
            from: sessions.first(where: { $0.id == sessionId })?.messages ?? [],
            excludingMessageIds: [messageId]
        )
        let sessionMessagesForImages = sessions.first(where: { $0.id == sessionId })?.messages ?? []
        // Images are priced flat in the pre-turn estimate: the assembler is not
        // given their bytes here (that happens off the main actor inside the
        // turn), so count the refs directly.
        let attachmentImageCount = sessionMessagesForImages.reduce(0) {
            $0 + ($1.attachments?.count ?? 0)
        }
        let turnInputTokens = CloudMessageAssembler.approximatePromptTokens(
            system: currentPersona(),
            messages: openingMessages
        ) + attachmentImageCount * ImageAttachmentPreparer.estimatedTokensPerImage
        // The odometer measures the context in flight, not the invoice: a cloud
        // request re-sends the whole conversation, so the figure is the last
        // request's prompt plus the tokens generated after it. Seeding it with
        // the assembled prompt's size keeps the header right before the first
        // provider usage report arrives.
        var turnContextTokens = turnInputTokens
        if let sIdx = sessions.firstIndex(where: { $0.id == sessionId }) {
            sessions[sIdx].cumulativeTokenCount = turnInputTokens
        }

        let visionCapable = visionSupport(forProvider: provider, providerModelId: providerModelId)
        // nil when nobody recorded a window for this model: an unknown window
        // never triggers a compaction, because guessing one would spend money
        // to lose context. Resolved by model id — the same lookup the chat
        // header uses — so the ceiling the send compacts against and the
        // ceiling the header reports are one answer even for a model that was
        // never added to the provider's registered list.
        let contextWindow = CloudContextPolicy.window(
            forModelId: providerModelId,
            providerId: provider.id
        )
        // Resolved on the main actor before the task: an AppStorage read from
        // inside the loop would be a main-actor hop per step.
        let imageDeliveryPolicy: CloudImageDelivery = cloudResendImages ? .all : .newestUserTurnOnly
        generationTask = Task(priority: .userInitiated) {
            var currentMessageId = messageId
            var step = 0
            // Run-level totals for the final status line. Every agent step is
            // its own billed request, so the summary must report the sum of
            // the steps: reporting the last iteration alone named a fraction
            // of a long run while the header pill and `cumulativeCost`
            // already accumulated the whole thing.
            var runPromptTokens = 0
            var runCachedTokens = 0
            var runOutputTokens = 0
            var runCost: Double? = nil
            var runWasEstimated = false
            func odometerTotal(including generated: Int) -> Int {
                CloudContextOdometer.total(
                    promptTokens: turnContextTokens,
                    generatedTokens: generated
                )
            }
            // A long await can outlive the conversation: deleting a chat does
            // not stop a turn, so the next lookup finds nothing. Returning is
            // only half the fix — leaving `isGeneratingText` set would refuse
            // every later send until relaunch — and the flags belong to the
            // replacement if one has already been minted.
            @MainActor func endTurnBecauseSessionVanished(_ generation: UInt64) {
                guard self.ownsGeneration(generation) else { return }
                self.isGeneratingText = false
                self.generationTask = nil
                self.showStatus("⚠️ This conversation was closed — the turn ended.", kind: .warning)
            }
            // One pre-flight pass and one overflow-recovery pass per run.
            // The pre-flight flag is set when the pass STARTS, not when it
            // succeeds: a failed pass has already spent its request, and
            // leaving the gate open retried it at the top of every remaining
            // step — re-resolving the keychain, re-chunking the history and
            // re-billing the summarizer with no new chance of success.
            var compactionAttempts = CloudCompactionAttempts()
            // Why the last attempt left the history alone, so the error at the
            // context wall names what happened rather than blaming the provider
            // for a pass that summarised nothing.
            var contextCompactionFailure: CloudCompactionFailure?
            // Image parts are built once per turn, off the main actor. History is
            // re-assembled on every step, but the turn's pictures never change.
            // Skipped entirely for a text-only model: encoding megabytes of base64
            // only for the assembler to replace it with a notice is pure waste.
            let imageDataURLs: [UUID: [String]]
            if visionCapable {
                imageDataURLs = await Task.detached(priority: .userInitiated) {
                    Self.buildImageDataURLs(for: sessionMessagesForImages)
                }.value
            } else {
                imageDataURLs = [:]
            }
            // Cleared for the rest of the run if the provider rejects an image:
            // every later step then carries the metadata-only notice instead.
            var imagesAllowed = visionCapable
            var didRetryWithoutImages = false
            // What the rewritten history costs as a prompt. Used only right
            // after a compaction, which is the one moment the pre-turn estimate
            // stops describing what is being sent.
            //
            // `@MainActor` because it reads the session array and the persona:
            // a local function does not inherit the enclosing task's isolation
            // under the project's approachable-concurrency mode, and without it
            // both reads are concurrency warnings.
            @MainActor func rewrittenPromptEstimate() -> Int {
                guard let live = self.sessions.first(where: { $0.id == sessionId }) else { return 0 }
                return CloudMessageAssembler.approximatePromptTokens(
                    system: currentPersona(),
                    messages: CloudMessageAssembler.assemble(
                        from: live.messages,
                        excludingMessageIds: [currentMessageId],
                        imageDataURLs: imagesAllowed ? imageDataURLs : [:],
                        imagesSupported: imagesAllowed,
                        imageDelivery: imageDeliveryPolicy
                    )
                )
            }
            // A compaction rewrites the conversation the odometer was counting,
            // so its figure describes a prompt that no longer exists. Reseed
            // from the rewritten prompt (which now includes the digest the
            // compaction just installed): the next step's reported prompt
            // replaces this exactly, and a provider that reports none still
            // leaves a number describing what was sent. Returns the figure so
            // the caller's own copy is written in one hop rather than mutated
            // from an isolated nested function.
            @MainActor func odometerAfterCompaction() -> Int {
                let rewritten = rewrittenPromptEstimate()
                if let sIdx = self.sessions.firstIndex(where: { $0.id == sessionId }) {
                    self.sessions[sIdx].cumulativeTokenCount = rewritten
                }
                return rewritten
            }
            // Prompt-only coordinator directive carrying subagent reports that
            // landed mid-run. Cloud history is re-assembled from the session on
            // every step, so the reports ride the leading system message of the
            // remaining steps instead of a stored message (which would render in
            // the transcript as if the user sent them).
            var relayDirective = ""
            // At most one tools-disabled synthesis turn per run, mirroring the
            // weights path's forced-synthesis pass. It also bounds the subagent
            // result-delivery relay below.
            var ranSynthesisTurn = false

            @MainActor func takeSubagentRelayDirective() -> Bool {
                let pending = self.gatherCompletedSubagentRelayResponses(forSession: sessionId)
                guard !pending.isEmpty else { return false }
                self.markSubagentSummariesRelayed(pending)
                relayDirective = CloudSubagentRelay.directive(reports: pending)
                return true
            }

            while true {
                // A superseded or stopped turn must not spend one more provider
                // request: the loop issues a fresh call every iteration. Stop
                // cancels the task without minting a new generation id, so both
                // checks matter.
                guard !Task.isCancelled, self.ownsGeneration(myGenerationId) else { return }
                guard var sIdx = self.sessions.firstIndex(where: { $0.id == sessionId }) else {
                    endTurnBecauseSessionVanished(myGenerationId)
                    return
                }
                // Auto-compaction is priced before the request is sent, so the
                // turn never meets a context wall it could have avoided. It
                // runs between steps, never mid-request, and the in-flight
                // placeholder survives it because the rewrite keeps the newest
                // messages.
                if compactionAttempts.shouldAttemptPreflight, let window = contextWindow {
                    let projected = CloudMessageAssembler.approximatePromptTokens(
                        system: currentPersona(),
                        messages: CloudMessageAssembler.assemble(
                            from: self.sessions[sIdx].messages,
                            excludingMessageIds: [currentMessageId],
                            imageDataURLs: imagesAllowed ? imageDataURLs : [:],
                            imagesSupported: imagesAllowed,
                            imageDelivery: imageDeliveryPolicy
                        )
                    )
                    if CloudContextPolicy.shouldCompact(promptTokens: projected, window: window) {
                        compactionAttempts.beginPreflight()
                        let outcome = await self.compactCloudHistory(
                            sessionId: sessionId,
                            provider: provider,
                            providerModelId: providerModelId,
                            window: window,
                            keepRecentMessages: 8
                        )
                        // The summarizer is billed like any other request, so
                        // its spend AND its tokens join the run totals: a cost
                        // that covers requests its token counts omit cannot be
                        // reconciled with the figures beside it.
                        runPromptTokens += outcome.promptTokens
                        runCachedTokens += outcome.cachedTokens
                        runOutputTokens += outcome.completionTokens
                        if outcome.wasEstimated { runWasEstimated = true }
                        if outcome.cost > 0 {
                            runCost = (runCost ?? 0) + outcome.cost
                        }
                        if outcome.didCompact {
                            compactionAttempts.recordRewrite()
                            contextCompactionFailure = nil
                            turnContextTokens = odometerAfterCompaction()
                            self.showStatus(
                                "🗂 Context was near this model's limit — compacted older history before continuing.",
                                sessionId: sessionId
                            )
                            continue
                        }
                        contextCompactionFailure = outcome.failure
                        if outcome.failure?.isRetryableOnNextStep == true {
                            // Refused before any request was sent: nothing was
                            // billed, and the reason depends on the history's
                            // shape, which the next step changes.
                            compactionAttempts.abandonPreflight()
                        }
                        // A Stop during the summary cancelled this task; the
                        // loop must not spend a request on a turn that is over.
                        guard !Task.isCancelled, self.ownsGeneration(myGenerationId) else { return }
                        // The pass awaited for minutes. New Chat inserts at
                        // index 0 and deleting a chat removes one, so `sIdx`
                        // may now address a different conversation, or nothing
                        // at all. Re-resolve by identity before anything reads
                        // the messages again — a stale index either traps or
                        // sends another conversation's history to the provider.
                        guard let liveIdx = self.sessions.firstIndex(where: { $0.id == sessionId }) else {
                            endTurnBecauseSessionVanished(myGenerationId)
                            return
                        }
                        sIdx = liveIdx
                        if outcome.failure?.spentRequest == true,
                           let notice = outcome.failure?.notice(providerName: provider.displayName) {
                            // The pass was billed and left the history alone.
                            // Say so: without a row, the paid summary is
                            // invisible and the next step looks identical to
                            // one that never needed compacting.
                            self.sessions[sIdx].messages.append(ChatMessage(
                                role: .system,
                                content: "Auto-compaction. \(notice)"
                            ))
                        }
                    }
                }
                // The digest is read here, per step, so a compaction that ran
                // above is already in this request's system message.
                let persona = currentPersona()
                // A directive applies to exactly one step. Keeping it set while
                // tool rounds continue told every later step to "write the final
                // answer now" even though that answer was not due yet.
                let stepDirective = relayDirective
                relayDirective = ""
                // A coordinator directive is APPENDED, never merged into the
                // system message: that is the prefix the provider cached.
                let outbound = CloudStepPrompt.assemble(
                    persona: persona,
                    history: CloudMessageAssembler.assemble(
                        from: self.sessions[sIdx].messages,
                        excludingMessageIds: [currentMessageId],
                        imageDataURLs: imagesAllowed ? imageDataURLs : [:],
                        imagesSupported: imagesAllowed,
                        imageDelivery: imageDeliveryPolicy
                    ),
                    directive: stepDirective
                )
                let stepCarriesImages = outbound.contains { ($0.imageParts?.count ?? 0) > 0 }
                // What this step's request carries. The odometer reports this
                // prompt (not a sum of billable input): it already contains
                // every earlier step's output, so adding steps together would
                // count the same history once per step.
                let stepEstimatedPrompt = CloudMessageAssembler.approximatePromptTokens(
                    system: nil,
                    messages: outbound
                )
                turnContextTokens = stepEstimatedPrompt
                // The relay/synthesis turn is a final-answer pass: the model must
                // present the reports, not start another tool round.
                let tools = CloudAgentStepPlan.offersTools(
                    agentTurn: toolsEnabled,
                    isSynthesisTurn: ranSynthesisTurn
                ) ? AgentHarness.shared.availableToolDefinitions(mode: mode) : []
                let stepMessageId = currentMessageId
                let generationStartTime = CFAbsoluteTimeGetCurrent()

                // One row per agent step: each step is its own request and its own
                // assistant message, and the box is one-shot.
                let turnLog = ServerRequestLog.chatTurnLog(
                    model: displayName,
                    inputTokens: stepEstimatedPrompt,
                    isAgentTurn: isAgentTurn,
                    prompt: CloudMessageAssembler.renderedPromptForLog(system: nil, messages: outbound)
                )

                var latestText = ""
                var latestThinking = ""
                var sawFirstDelta = false
                var firstDeltaAt = 0.0
                var lastUIUpdateTime = 0.0
                var thinkingStartedAt: Double? = nil
                var thinkingDuration: Double? = nil
                var reportedUsage: CloudUsage? = nil
                var finishReason: String? = nil
                var serviceFailure: CloudTurnFailure? = nil
                var stepToolCalls: [CloudToolCall] = []

                func generatedTokenCount() -> Int {
                    AppleFoundationModelService.generatedTokenCount(
                        thinkingCharacters: latestThinking.count,
                        contentCharacters: latestText.count,
                        reportedCompletionTokens: reportedUsage?.completionTokens
                    )
                }

                @MainActor func commitOdometerTail() {
                    guard self.ownsGeneration(myGenerationId),
                          let sIdx = self.sessions.firstIndex(where: { $0.id == sessionId }) else { return }
                    self.sessions[sIdx].cumulativeTokenCount = odometerTotal(including: generatedTokenCount())
                }

                // 60fps-throttled streaming UI updates, mirroring the weights and
                // on-device paths.
                @MainActor func pushPartial() {
                    let now = CFAbsoluteTimeGetCurrent()
                    if now - lastUIUpdateTime < 0.016 { return }
                    lastUIUpdateTime = now
                    if !sawFirstDelta {
                        sawFirstDelta = true
                        firstDeltaAt = now
                    }
                    guard self.ownsGeneration(myGenerationId) else { return }
                    let elapsed = max(now - generationStartTime, 0.001)
                    // Count reasoning + content, matching the finalizer, so the
                    // header speed converges to the footer's final number and
                    // reads real tok/s during a thinking block instead of 0.
                    let generatedSoFar = generatedTokenCount()
                    let tokPerSec = Double(generatedSoFar) / elapsed
                    let activelyThinking = !latestThinking.isEmpty && latestText.isEmpty
                    self.generatedStreamText = latestText
                    self.thinkingText = latestThinking
                    self.responseText = latestText
                    self.isThinking = activelyThinking
                    self.generationTotalTokens = generatedSoFar
                    self.generationElapsedMs = elapsed * 1000.0
                    self.generationSpeedTokPerSec = tokPerSec
                    if let sIdx = self.sessions.firstIndex(where: { $0.id == sessionId }),
                       let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == stepMessageId }) {
                        self.sessions[sIdx].messages[mIdx].prefillStatus = nil
                        self.sessions[sIdx].messages[mIdx].thinkingContent = latestThinking.isEmpty ? nil : latestThinking
                        self.sessions[sIdx].messages[mIdx].content = latestText
                        self.sessions[sIdx].messages[mIdx].timestamp = Date()
                        self.sessions[sIdx].messages[mIdx].isThinking = activelyThinking
                        self.sessions[sIdx].messages[mIdx].tokenCount = generatedSoFar
                        self.sessions[sIdx].messages[mIdx].tokensPerSec = tokPerSec
                        self.sessions[sIdx].messages[mIdx].timeToFirstTokenSeconds = max(firstDeltaAt - generationStartTime, 0.0)
                        self.sessions[sIdx].cumulativeTokenCount = odometerTotal(including: generatedSoFar)
                    }
                }

                for await event in client.streamTurn(
                    model: providerModelId,
                    messages: outbound,
                    params: params,
                    tools: tools
                ) {
                    if Task.isCancelled { break }
                    switch event {
                    case .started:
                        break
                    case .thinkingDelta(let delta):
                        if thinkingStartedAt == nil { thinkingStartedAt = CFAbsoluteTimeGetCurrent() }
                        latestThinking += delta
                        turnLog.noteFirstToken()
                        pushPartial()
                    case .contentDelta(let delta):
                        if let started = thinkingStartedAt, thinkingDuration == nil {
                            thinkingDuration = max(CFAbsoluteTimeGetCurrent() - started, 0.0)
                        }
                        latestText += delta
                        turnLog.noteFirstToken()
                        pushPartial()
                    case .finished(let result):
                        latestText = result.text
                        latestThinking = result.thinking
                        reportedUsage = result.usage
                        finishReason = result.finishReason
                        stepToolCalls = result.toolCalls
                    case .failed(let failure):
                        serviceFailure = failure
                    }
                }

                // Stop-button cancellation and superseded turns bypass the
                // success finalizer: the replacement generation owns the UI
                // flags. Partials already streamed stay in the message, and the
                // row keeps the stopped/aborted contract the weights path uses.
                if Task.isCancelled || !self.ownsGeneration(myGenerationId) {
                    commitOdometerTail()
                    turnLog.recordChatTurn(
                        status: Task.isCancelled ? 200 : 500,
                        errorMessage: Task.isCancelled ? "⏹ Generation stopped by user." : "Generation stream closed without a result",
                        outputTokens: AppleFoundationModelService.recordedOutputTokenCount(for: latestText),
                        finishReason: nil,
                        thinking: latestThinking,
                        response: latestText,
                        toolCalls: []
                    )
                    return
                }

                // A model registered as vision-capable can still refuse an image
                // part: no provider catalog publishes modalities, so that
                // registration is an editorial guess. Un-mark it, say so once, and
                // spend this same step again without pixels instead of failing a
                // conversation that was working until the screenshot.
                if imagesAllowed,
                   stepCarriesImages,
                   !didRetryWithoutImages,
                   let failure = serviceFailure,
                   case .httpStatus(let code, let summary) = failure,
                   code == 400 || code == 422,
                   CloudTurnFailure.indicatesImageRejection(summary) {
                    didRetryWithoutImages = true
                    imagesAllowed = false
                    CloudModelRegistry.setVisionSupport(false, forModel: providerModelId, provider: provider)
                    self.applyVisionCapabilityChange(
                        provider: provider,
                        providerModelId: providerModelId,
                        supported: false
                    )
                    self.showStatus("⚠️ \(provider.displayName) rejected the image for this model — marked it text-only and retrying without it.")
                    // The status line is overwritten by the next streamed partial,
                    // so record the downgrade in the transcript too. Otherwise the
                    // model silently stops accepting images with no later clue why.
                    self.sessions[sIdx].messages.append(ChatMessage(
                        role: .system,
                        content: "⚠️ \(provider.displayName) rejected an image for \(CloudModelRegistry.defaultDisplayName(provider: provider, providerModelId: providerModelId)). The model is now marked text-only; re-enable it under Settings → Cloud Providers if that was wrong. Retrying this turn without the image."
                    ))
                    turnLog.recordChatTurn(
                        status: 400,
                        errorMessage: "Image content rejected by the provider; retrying without images.",
                        outputTokens: 0,
                        finishReason: nil,
                        thinking: latestThinking,
                        response: latestText,
                        toolCalls: []
                    )
                    continue
                }

                // A context-length rejection is recoverable: compact what is
                // behind this step and spend the SAME request again, once. The
                // alternative — what the app did before — was to end the turn
                // and tell the user to run a command that needs a different
                // model to be available on their Mac.
                if compactionAttempts.shouldAttemptOverflowRecovery,
                   let failure = serviceFailure,
                   case .httpStatus(let code, let summary) = failure,
                   code == 400 || code == 422,
                   CloudTurnFailure.indicatesContextOverflow(summary) {
                    compactionAttempts.beginOverflowRecovery()
                    let outcome = await self.compactCloudHistory(
                        sessionId: sessionId,
                        provider: provider,
                        providerModelId: providerModelId,
                        window: contextWindow,
                        keepRecentMessages: 4
                    )
                    runPromptTokens += outcome.promptTokens
                    runCachedTokens += outcome.cachedTokens
                    runOutputTokens += outcome.completionTokens
                    if outcome.wasEstimated { runWasEstimated = true }
                    if outcome.cost > 0 {
                        runCost = (runCost ?? 0) + outcome.cost
                    }
                    if outcome.didCompact {
                        compactionAttempts.recordRewrite()
                        contextCompactionFailure = nil
                        turnContextTokens = odometerAfterCompaction()
                        turnLog.recordChatTurn(
                            status: 400,
                            errorMessage: "Context overflow; compacted older history and retrying this step.",
                            outputTokens: AppleFoundationModelService.recordedOutputTokenCount(for: latestText),
                            finishReason: nil,
                            thinking: latestThinking,
                            response: latestText,
                            toolCalls: []
                        )
                        self.showStatus(
                            "🗂 The conversation exceeded this model's context window — compacted older history and retrying the step.",
                            sessionId: sessionId
                        )
                        continue
                    }
                    contextCompactionFailure = outcome.failure
                    guard !Task.isCancelled, self.ownsGeneration(myGenerationId) else { return }
                    // Same reason as the pre-flight: the pass suspended long
                    // enough for the sessions array to be reordered or pruned,
                    // and every later read below must use a fresh index.
                    guard let liveIdx = self.sessions.firstIndex(where: { $0.id == sessionId }) else {
                        endTurnBecauseSessionVanished(myGenerationId)
                        return
                    }
                    sIdx = liveIdx
                }

                if let failure = serviceFailure {
                    var errText = "⚠️ \(provider.displayName) error: \(failure.localizedDescription)"
                    if isAgentTurn, case .httpStatus(let code, let summary) = failure, code == 400 || code == 422 {
                        if CloudTurnFailure.indicatesContextOverflow(summary) {
                            // A long agent run overflows the provider's context
                            // window too; blaming tool schemas there sends the
                            // user after the wrong setting. The advice names
                            // which history this is: one that already compacted,
                            // one whose compaction failed, or one with no
                            // attempt at all.
                            errText += CloudOverflowAdvice.text(
                                alreadyCompacted: compactionAttempts.historyWasRewritten,
                                failure: contextCompactionFailure,
                                providerName: provider.displayName
                            )
                        } else {
                            errText += " Tool schemas may be unsupported for this model — turn Agent Tools off, or register the model under the provider's no-tools key (`dynamoe_cloud_\(provider.id)_no_tools`)."
                        }
                    }
                    self.isGeneratingText = false
                    self.generationTask = nil
                    // Earlier steps (and any summarizer pass) were billed even
                    // though this one failed. The status line is the only place
                    // that total is reported on this path; the transcript row
                    // keeps the error itself.
                    var statusText = errText
                    if runPromptTokens > 0 || runOutputTokens > 0 || runCost != nil {
                        let spent = CloudTurnReport(
                            promptTokens: runPromptTokens,
                            cachedTokens: runCachedTokens,
                            outputTokens: runOutputTokens,
                            cost: runCost,
                            wasEstimated: runWasEstimated
                        )
                        statusText += " · ☁️ run total so far: \(spent.statusSuffix)"
                    }
                    self.showStatus(statusText, kind: .warning)
                    if let sIdx = self.sessions.firstIndex(where: { $0.id == sessionId }),
                       let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == stepMessageId }) {
                        let partialPrefix = latestText.isEmpty ? "" : latestText + "\n\n"
                        self.sessions[sIdx].messages[mIdx].prefillStatus = nil
                        self.sessions[sIdx].messages[mIdx].content = partialPrefix + errText
                        self.sessions[sIdx].messages[mIdx].isThinking = false
                    }
                    turnLog.recordChatTurn(
                        status: 500,
                        errorMessage: errText,
                        outputTokens: AppleFoundationModelService.recordedOutputTokenCount(for: latestText),
                        finishReason: nil,
                        thinking: latestThinking,
                        response: latestText,
                        toolCalls: []
                    )
                    self.dequeueAndRunNextPromptIfNeeded(sessionId: sessionId)
                    return
                }

                // The provider's own prompt count replaces the step's estimate.
                // It is the FULL prompt (cache reads included), because the
                // header reports the context the model was given, not the part
                // of it that was billed at the fresh rate.
                if let usage = reportedUsage {
                    turnLog.noteProviderUsage(
                        promptTokens: usage.promptTokens,
                        cachedTokens: usage.cachedTokens,
                        cacheWriteTokens: usage.cacheWriteTokens
                    )
                    if let promptTokens = usage.promptTokens {
                        turnContextTokens = promptTokens
                    }
                    // Spend accumulates per step: every step is its own billed
                    // request, and a run that never reaches its final status
                    // line still cost what it cost. The same number feeds the
                    // summary line below.
                    if let cost = CloudPricingStore.pricing(
                        forModel: providerModelId,
                        providerId: provider.id
                    )?.cost(for: usage) {
                        if let sIdx = self.sessions.firstIndex(where: { $0.id == sessionId }) {
                            self.sessions[sIdx].cumulativeCost = (self.sessions[sIdx].cumulativeCost ?? 0) + cost
                        }
                        runCost = (runCost ?? 0) + cost
                    }
                }
                runPromptTokens += reportedUsage?.promptTokens ?? stepEstimatedPrompt
                runCachedTokens += reportedUsage?.cachedTokens ?? 0
                if reportedUsage?.promptTokens == nil {
                    runWasEstimated = true
                }

                let finalElapsed = max(CFAbsoluteTimeGetCurrent() - generationStartTime, 0.001)
                let generatedTokens = generatedTokenCount()
                runOutputTokens += generatedTokens
                if reportedUsage?.completionTokens == nil {
                    // `generatedTokenCount` falls back to a character estimate
                    // when the provider reported no completion count.
                    runWasEstimated = true
                }
                let finalTokPerSec = Double(generatedTokens) / finalElapsed

                if let sIdx = self.sessions.firstIndex(where: { $0.id == sessionId }),
                   let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == stepMessageId }) {
                    self.sessions[sIdx].messages[mIdx].prefillStatus = nil
                    self.sessions[sIdx].messages[mIdx].content = latestText
                    self.sessions[sIdx].messages[mIdx].thinkingContent = latestThinking.isEmpty ? nil : latestThinking
                    self.sessions[sIdx].messages[mIdx].isThinking = false
                    self.sessions[sIdx].messages[mIdx].timestamp = Date()
                    self.sessions[sIdx].messages[mIdx].tokenCount = generatedTokens
                    self.sessions[sIdx].messages[mIdx].tokensPerSec = finalTokPerSec
                    self.sessions[sIdx].messages[mIdx].thinkingTimeSeconds = thinkingDuration
                    if sawFirstDelta {
                        self.sessions[sIdx].messages[mIdx].timeToFirstTokenSeconds = max(firstDeltaAt - generationStartTime, 0.0)
                    }
                    self.sessions[sIdx].cumulativeTokenCount = odometerTotal(including: generatedTokens)
                }
                turnLog.recordChatTurn(
                    status: 200,
                    errorMessage: nil,
                    outputTokens: generatedTokens,
                    finishReason: finishReason,
                    thinking: latestThinking,
                    response: latestText,
                    toolCalls: stepToolCalls.map { call in
                        ParsedToolCall(
                            name: call.name,
                            arguments: call.decodedArguments ?? [:],
                            rawArguments: call.argumentsJSON,
                            rawText: call.argumentsJSON
                        )
                    }
                )

                // === Native function-calling agent step ===
                var ranComplete = false
                var failureLoop = false
                if CloudAgentStepPlan.shouldExecuteToolCalls(
                    offeredToolCount: tools.count,
                    callCount: stepToolCalls.count
                ) {
                    self.showStatus("⚙️ Executing \(stepToolCalls.count) tool call(s)...")
                    let outcome = await self.executeCloudToolCalls(
                        calls: stepToolCalls,
                        mode: mode,
                        sessionId: sessionId,
                        messageId: stepMessageId,
                        myGenerationId: myGenerationId
                    )
                    if outcome.cancelled {
                        if let sIdx = self.sessions.firstIndex(where: { $0.id == sessionId }),
                           let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == stepMessageId }),
                           var current = self.sessions[sIdx].messages[mIdx].toolCalls {
                            for i in current.indices where current[i].status == .running {
                                current[i].status = .error
                                current[i].error = "Cancelled by user."
                            }
                            self.sessions[sIdx].messages[mIdx].toolCalls = current
                        }
                        if self.ownsGeneration(myGenerationId) {
                            self.isGeneratingText = false
                            self.generationTask = nil
                            self.showStatus("⏹ Tool execution stopped by user.")
                        }
                        return
                    }
                    ranComplete = outcome.ranComplete
                    failureLoop = outcome.failureLoop

                    if CloudAgentLoopPolicy.shouldContinue(
                        stepCallCount: stepToolCalls.count,
                        step: step,
                        maxSteps: AgentStepBudget.compareCap(self.agentStepCap),
                        ranComplete: outcome.ranComplete,
                        failureLoop: outcome.failureLoop,
                        cancelled: outcome.cancelled
                    ) {
                        // Auto-relay: completed background subagents whose final
                        // reports have not reached the conversation ride the next
                        // step's system message so the coordinator can present
                        // them instead of the run ending with the report dropped.
                        _ = takeSubagentRelayDirective()
                        let nextMessageId = UUID()
                        if let sIdx = self.sessions.firstIndex(where: { $0.id == sessionId }) {
                            self.sessions[sIdx].messages.append(ChatMessage(
                                id: nextMessageId,
                                role: .assistant,
                                content: "",
                                thinkingContent: nil,
                                isThinking: false
                            ))
                        }
                        currentMessageId = nextMessageId
                        step += 1
                        continue
                    }
                    if failureLoop {
                        self.showStatus("🛡️ Tool loop guard hit — ending the agent run.")
                    } else if ranComplete {
                        self.showStatus("✅ Task complete — \(provider.displayName).")
                    }
                }

                // Forced-synthesis escalation: a guard-tripped or
                // budget-exhausted run otherwise ends on an assistant turn whose
                // content is empty beside its tool records, so the user gets no
                // answer. Spend ONE tools-disabled turn, mirroring the weights
                // path, and let the model present what it already learned.
                let wantedAnotherToolRound = stepToolCalls.count > 0 && !ranComplete && !failureLoop
                let budgetExhausted = wantedAnotherToolRound
                    && AgentStepBudget.isExhausted(step: step, cap: self.agentStepCap)
                if toolsEnabled,
                   CloudAgentLoopPolicy.shouldForceSynthesis(
                       failureLoop: failureLoop,
                       budgetExhausted: budgetExhausted,
                       ranComplete: ranComplete,
                       alreadyForced: ranSynthesisTurn
                   ) {
                    ranSynthesisTurn = true
                    relayDirective = CloudAgentSynthesis.directive
                    let nextMessageId = UUID()
                    if let sIdx = self.sessions.firstIndex(where: { $0.id == sessionId }) {
                        self.sessions[sIdx].messages.append(ChatMessage(
                            id: nextMessageId,
                            role: .assistant,
                            content: "",
                            thinkingContent: nil,
                            isThinking: false
                        ))
                    }
                    currentMessageId = nextMessageId
                    step += 1
                    self.showStatus(
                        failureLoop
                            ? "🛡️ Tool loop guard hit; forcing final answer..."
                            : "🛡️ Step budget exhausted; forcing final answer...",
                        kind: .warning
                    )
                    continue
                }

                // Result-delivery relay: the run is ending, but this session
                // still has completed subagents whose reports never reached the
                // conversation (the model spawned one and stopped without
                // polling). Spend ONE tools-disabled turn so the coordinator
                // presents them instead of dropping them.
                //
                // In-flight subagents land their reports later, so wait for them
                // first — the weights path waits at the same point, and skipping
                // it here would end the run while a report is still being
                // written.
                if toolsEnabled, !ranComplete, !ranSynthesisTurn,
                   AgentStepBudget.allowsAnotherStep(step: step, cap: self.agentStepCap) {
                    await self.awaitSessionSubagents(sessionId: sessionId, timeout: 120)
                }
                if toolsEnabled, !ranComplete, !ranSynthesisTurn,
                   AgentStepBudget.allowsAnotherStep(step: step, cap: self.agentStepCap),
                   takeSubagentRelayDirective() {
                    ranSynthesisTurn = true
                    let nextMessageId = UUID()
                    if let sIdx = self.sessions.firstIndex(where: { $0.id == sessionId }) {
                        self.sessions[sIdx].messages.append(ChatMessage(
                            id: nextMessageId,
                            role: .assistant,
                            content: "",
                            thinkingContent: nil,
                            isThinking: false
                        ))
                    }
                    currentMessageId = nextMessageId
                    step += 1
                    self.showStatus("📨 Subagent report ready — relaying results to user...")
                    continue
                }

                // A max-token cut ends the turn looking complete: the provider
                // reports `finish_reason: "length"` (or `"max_tokens"`) and just
                // stops. Park the notice in the transcript, because the status
                // line is transient and `.system` rows are never replayed into
                // the provider's history.
                let truncatedAtLimit = finishReason == "length" || finishReason == "max_tokens"
                if truncatedAtLimit {
                    let notice = "⚠️ This answer stopped at the Max Output Tokens limit and may be truncated. Raise it in Settings ▸ Sampling for a longer reply."
                    if let sIdx = self.sessions.firstIndex(where: { $0.id == sessionId }) {
                        let insertAt = self.sessions[sIdx].messages.firstIndex(where: { $0.id == stepMessageId })
                            .map { $0 + 1 } ?? self.sessions[sIdx].messages.count
                        self.sessions[sIdx].messages.insert(
                            ChatMessage(role: .system, content: notice),
                            at: insertAt
                        )
                    }
                }

                self.isGeneratingText = false
                self.generationTask = nil
                self.generatedStreamText = latestText
                self.thinkingText = latestThinking
                self.responseText = latestText
                self.isThinking = false
                self.generationTotalTokens = generatedTokens
                self.generationElapsedMs = finalElapsed * 1000.0
                self.generationSpeedTokPerSec = finalTokPerSec
                let turnCost = CloudTurnReport(
                    promptTokens: runPromptTokens,
                    cachedTokens: runCachedTokens,
                    outputTokens: runOutputTokens,
                    cost: runCost,
                    wasEstimated: runWasEstimated
                )
                // `4.20 s` rather than "in 4200.00 ms": the old phrasing put the
                // duration straight after the token counts, so "… in 4200.00 ms"
                // read as another input figure.
                let durationText = finalElapsed >= 1
                    ? String(format: "%.2f s", finalElapsed)
                    : String(format: "%.0f ms", finalElapsed * 1000.0)
                self.showStatus(
                    "☁️ \(provider.displayName) — \(turnCost.statusSuffix) · \(durationText) · \(String(format: "%.1f", finalTokPerSec)) tok/s"
                        + (truncatedAtLimit ? " — truncated at the Max Output Tokens limit." : "")
                )
                self.dequeueAndRunNextPromptIfNeeded(sessionId: sessionId)
                return
            }
        }
        return true
    }

    /// Executes one step's provider-issued calls through the shared harness.
    ///
    /// A record per call is committed to the assistant message BEFORE any
    /// execution, so the tool timeline and the reconstructed request agree even
    /// when the run is interrupted. Each call then passes the same guards the
    /// weights path applies: malformed or empty arguments are rejected without
    /// spending an approval, state-changing tools gate on HITL unless Turbo
    /// Mode is on, and outcomes feed the shared failure/repeat counters that
    /// decide whether the loop must stop.
    ///
    /// `mode` is the TURN's mode, carried by value from the step that started
    /// it. `executeTool` defaults it to `AgentHarness.shared.activeMode`, which
    /// is process-global and therefore mutable mid-turn; a plan turn that read
    /// the global could execute a write if the user flipped the pill while it
    /// ran.
    private func executeCloudToolCalls(
        calls: [CloudToolCall],
        mode: AgentMode,
        sessionId: UUID,
        messageId: UUID,
        myGenerationId: UInt64
    ) async -> (cancelled: Bool, ranComplete: Bool, failureLoop: Bool) {
        let agent = AgentHarness.shared
        var parsed: [ParsedToolCall?] = []
        var records: [ToolCallRecord] = []

        for call in calls {
            if let arguments = call.decodedArguments {
                parsed.append(ParsedToolCall(
                    name: call.name,
                    arguments: arguments,
                    rawArguments: call.argumentsJSON,
                    rawText: call.argumentsJSON
                ))
                records.append(ToolCallRecord(
                    name: call.name,
                    arguments: arguments.mapValues { "\($0)" },
                    rawArguments: call.argumentsJSON,
                    status: .running
                ))
            } else {
                let error = "Malformed arguments JSON — this call was not executed."
                parsed.append(nil)
                records.append(ToolCallRecord(
                    name: call.name,
                    arguments: [:],
                    rawArguments: call.argumentsJSON,
                    status: .error,
                    output: AgentHarness.renderToolResultForModel(
                        AgentHarness.toolErrorJSON(tool: call.name, error: error)
                    ),
                    error: error
                ))
                agent.recordToolCallValidity(false)
                agent.recordToolCallOutcome(toolName: call.name, outcome: .failed)
            }
        }

        if let sIdx = sessions.firstIndex(where: { $0.id == sessionId }),
           let mIdx = sessions[sIdx].messages.firstIndex(where: { $0.id == messageId }) {
            var current = sessions[sIdx].messages[mIdx].toolCalls ?? []
            current.append(contentsOf: records)
            sessions[sIdx].messages[mIdx].toolCalls = current
        }

        func updateRecord(_ id: UUID, _ mutate: (inout ToolCallRecord) -> Void) {
            guard let sIdx = sessions.firstIndex(where: { $0.id == sessionId }),
                  let mIdx = sessions[sIdx].messages.firstIndex(where: { $0.id == messageId }),
                  var current = sessions[sIdx].messages[mIdx].toolCalls,
                  let idx = current.firstIndex(where: { $0.id == id }) else { return }
            mutate(&current[idx])
            sessions[sIdx].messages[mIdx].toolCalls = current
        }

        let baseWorkingDirectory = agentWorkingDirectory.isEmpty
            ? nil
            : URL(fileURLWithPath: agentWorkingDirectory)
        var ranComplete = false

        for (index, call) in calls.enumerated() {
            AgentActivity.shared.begin(.toolCall)
            defer { AgentActivity.shared.end(.toolCall) }

            if Task.isCancelled || !ownsGeneration(myGenerationId) {
                return (true, ranComplete, false)
            }
            let record = records[index]
            guard let parsedCall = parsed[index] else { continue }

            showStatus("⚙️ Executing [\(index + 1)/\(calls.count)]: \(call.name)...")

            // A call whose required arguments are all empty is a broken
            // scaffold: reject it without executing and without a HITL prompt.
            if AgentHarness.hasEmptyRequiredArguments(toolName: call.name, arguments: parsedCall.arguments) {
                agent.recordToolCallValidity(false)
                agent.recordToolCallOutcome(toolName: call.name, outcome: .failed)
                let emptyArgJSON = AgentHarness.toolErrorJSON(
                    tool: call.name,
                    error: "Empty arguments — this call carried no usable parameters and was not executed. Either provide real arguments, or stop calling tools and answer the user directly from what you already know."
                )
                updateRecord(record.id) { rec in
                    rec.status = .error
                    rec.error = "Rejected: required arguments were empty."
                    rec.output = AgentHarness.renderToolResultForModel(emptyArgJSON)
                }
                continue
            }
            agent.recordToolCallValidity(true)

            let isTurbo = UserDefaults.standard.bool(forKey: "dynamoe_agent_turbo_mode")
            // A tool plan mode withholds is refused by `executeTool` without
            // running, so asking the user first offered them a write the mode had
            // already denied.
            if AgentHarness.requiresApproval(toolName: call.name, mode: mode), !isTurbo {
                updateRecord(record.id) { rec in rec.status = .awaitingApproval }
                showStatus("🛡️ Action requires confirmation: \(call.name)")
                let approved = await ActionApprovalManager.shared.waitForApproval(id: record.id)
                if !approved {
                    let rejectJSON = AgentHarness.toolErrorJSON(tool: call.name, error: "Action rejected by user.")
                    agent.recordToolCallOutcome(toolName: call.name, outcome: .failed)
                    updateRecord(record.id) { rec in
                        rec.status = .rejected
                        rec.error = "Action rejected by user."
                        rec.output = AgentHarness.renderToolResultForModel(rejectJSON)
                    }
                    continue
                }
            }

            if Task.isCancelled || !ownsGeneration(myGenerationId) {
                return (true, ranComplete, false)
            }

            let execResult = await agent.executeTool(
                call: parsedCall,
                workingDirectory: baseWorkingDirectory,
                maxOutputLength: maxToolOutputLength,
                mode: mode
            )
            agent.recordToolCallOutcome(
                toolName: call.name,
                outcome: execResult.record.status.toolCallOutcome
            )
            // A withheld call is not fingerprinted: `recordToolCallFingerprint`
            // CLEARS every count for the mutating tools, so counting a refusal of
            // `file_write` would reset the repeat guard on each attempt and leave
            // only the step budget bounding the loop.
            if execResult.record.status != .withheld {
                agent.recordToolCallFingerprint(toolName: call.name, arguments: parsedCall.arguments)
            }
            // `isCompleted` covers every tool that ends the run (the harness's
            // loop guards set it too), matching the weights path. The rule lives
            // on the status so both turn loops read it the same way — a withheld
            // `complete` must not end the run, because plan mode took it away so
            // the turn could finish with a plan.
            if execResult.record.status.endsAgentRun(
                toolName: call.name,
                isCompleted: execResult.isCompleted
            ) {
                ranComplete = true
            }

            updateRecord(record.id) { rec in
                rec.status = execResult.record.status
                rec.output = execResult.record.output
                rec.error = execResult.record.error
                rec.executionDurationSeconds = execResult.record.executionDurationSeconds
            }
        }

        let failureLoop = agent.consecutiveFailedToolCalls >= agent.failedToolCallLimit
            || agent.totalFailedToolCalls >= agent.failedToolCallTotalLimit
            || agent.maxRepeatedToolCallCount >= agent.repeatedToolCallLimit

        return (Task.isCancelled || !ownsGeneration(myGenerationId), ranComplete, failureLoop)
    }

    /// Builds the raw-token prompt for an agent continuation turn: the previous
    /// turn's actual prompt+generated token ids plus the freshly tokenized turn
    /// suffix (end tag + tool-response/continuation text). Re-tokenizing the
    /// accumulated text can diverge by one token when generation was cut
    /// mid-tag, which breaks exact prefix reuse on hybrid models and forces a
    /// full re-prefill; splicing the real ids makes the pin match byte-exact.
    /// Returns nil when the suffix cannot be tokenized (caller falls back to
    /// the string path).
    private func buildSplicedContinuationTokens(
        basePromptTokens: [UInt32],
        generatedTokenIds: [UInt32],
        turnText: String,
        endTag: String,
        suffix: String,
        encode: (String) throws -> [UInt32]
    ) -> [UInt32]? {
        var tokens = basePromptTokens
        // A turn whose CONTENT had to be rewritten — a parameter closer split across
        // lines (`</parameter=\n>`), or a heredoc the model over-ran — cannot be
        // spliced from the raw ids without teaching the model its own broken shape.
        // Re-encode the prepared turn instead. That costs one full re-prefill for the
        // turn (the pin no longer matches byte-for-byte), which is cheaper than the
        // loop. Closers/end-tag appended only for structure do NOT count as a content
        // change, so the common freeze case keeps the cheap exact splice.
        let prepared = prepareAssistantTurnText(turnText, endTag: endTag)
        if prepared.contentChanged {
            guard let preparedIds = try? encode(prepared.text), !preparedIds.isEmpty else { return nil }
            tokens.append(contentsOf: preparedIds)
        } else {
            tokens.append(contentsOf: generatedTokenIds)
            // `turnText` MUST be the raw decoded turn, never a normalized copy: the
            // closure decision has to be made against what the model actually emitted.
            guard let closure = StreamingToolParser.turnClosureSuffix(
                forRawDecodedTurn: turnText,
                endTag: endTag,
                encode: encode
            ) else { return nil }
            tokens.append(contentsOf: closure)
        }
        guard let suffixIds = try? encode(suffix), !suffixIds.isEmpty else { return nil }
        tokens.append(contentsOf: suffixIds)
        return tokens
    }

    /// The assistant turn text as it should be committed, plus whether the CONTENT
    /// (not just appended closers/end tag) changed. Repairs a split parameter closer,
    /// restores structural closers the freeze or token cap cut, and trims a heredoc
    /// the model over-ran. `contentChanged == false` means the raw generated ids plus
    /// the closure tokens reproduce it, so the caller can splice cheaply.
    private func prepareAssistantTurnText(_ decoded: String, endTag: String) -> (text: String, contentChanged: Bool) {
        var text = StreamingToolParser.repairSplitParameterClosers(decoded)
        var contentChanged = text != decoded
        for tag in StreamingToolParser.structuralClosureTags(forRawDecodedTurn: text) {
            text += tag
        }
        // Truncation needs the value closed to find it, so it runs after the closers.
        let beforeTruncate = text
        text = StreamingToolParser.truncateHeredocOverruns(inTurnText: text)
        if text != beforeTruncate { contentChanged = true }
        // A collapsed command argument (a repetition degeneration) must not be pinned
        // into history: the model would keep attending to — and imitating — tens of
        // thousands of repeated characters. Replace it with a short placeholder.
        let beforeCompress = text
        text = StreamingToolParser.compressDegenerateCommandArguments(inTurnText: text)
        if text != beforeCompress { contentChanged = true }
        if !text.contains(endTag) {
            text += endTag
        }
        return (text, contentChanged)
    }

    /// The pre-execution freeze fires the moment `</function>` closes, so a frozen
    /// tool call is routinely committed to context WITHOUT its `</tool_call>`
    /// closer. Spliced into the next prompt verbatim, the model's own history
    /// becomes a stack of structurally invalid call examples and it starts
    /// inventing shapes (observed: `<parameter=url>` re-opened inside a
    /// `<parameter=url>` value, then repeating a malformed call verbatim). Close
    /// the block and terminate the turn so the model always reads back valid
    /// examples of its own format. Paired with `buildSplicedContinuationTokens`,
    /// which performs the same normalization on the token stream.
    private func closedAssistantTurnText(_ decoded: String, endTag: String) -> String {
        prepareAssistantTurnText(decoded, endTag: endTag).text
    }

    // MARK: - Autoregressive Generation (CompletionEngine orchestrator)

    /// Thin orchestrator: builds and tokenizes the prompt, launches the
    /// headless CompletionEngine turn, consumes its event stream to drive the
    /// historical UI writes, then runs the agent-orchestration phase (tool
    /// execution, HITL approvals, multi-step recursion) verbatim.
    /// `images` are the turn's attachments, already decoded, in prompt order and
    /// matching the image placeholders the caller wrote into `customPrompt`. Only a
    /// family with a vision tower consumes them; the engine refuses a turn whose
    /// placeholder count disagrees with this list rather than dropping pixels.
    /// Returns whether this step actually launched. Most callers are continuations
    /// inside the turn loop and can ignore it; the composer's send path propagates
    /// it, because a launch refused AFTER the messages were appended (lost slot,
    /// engine not ready) is otherwise reported to the caller as accepted work.
    @discardableResult
    private func startAutoregressiveGeneration(customPrompt: String? = nil, promptTokens: [UInt32]? = nil, images: [TurnImage] = [], carriedImageSpans: [TurnImageSpan] = [], sessionId: UUID? = nil, messageId: UUID? = nil, agentStep: Int = 0, isInThinkingContinuation: Bool = false, forceSynthesis: Bool = false) -> Bool {
        guard let summary = summary,
              let tokenizer = tokenizer else {
            let err = "❌ Metal or Tokenizer not ready for text generation. Please load model and tokenizer in Settings."
            isGeneratingText = false
            generationTask = nil
            if let sId = sessionId, let mId = messageId,
               let sIdx = sessions.firstIndex(where: { $0.id == sId }),
               let mIdx = sessions[sIdx].messages.firstIndex(where: { $0.id == mId }) {
                sessions[sIdx].messages[mIdx].content = err
                sessions[sIdx].messages[mIdx].isThinking = false
            }
            showStatus(err, kind: .warning)
            return false
        }

        let prompt: String
        if let custom = customPrompt {
            prompt = custom
        } else {
            prompt = promptInput.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
        }
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            let err = "⚠️ Please enter a prompt to generate text."
            isGeneratingText = false
            generationTask = nil
            showStatus(err, kind: .warning)
            return false
        }

        // Derive model/thinking from the session this generation belongs to, not
        // the current selection, so an async send that resumed after a selection
        // change still formats with the conversation's own model.
        let generationSession = sessionId.flatMap { id in sessions.first(where: { $0.id == id }) }
        // The turn's agent surface, resolved ONCE and carried by value for the
        // rest of the turn. `AgentHarness.activeMode` is process-global, so
        // reading it at each tool call meant a mode flip while the turn ran
        // could hand a plan turn a write tool. Every tool execution and every
        // tool-block refresh below takes `turnMode` instead.
        //
        // Step 0 resolves it from the conversation; a continuation reuses the
        // value the turn started with, because the conversation can be deleted
        // (retention, a manual delete) while the turn is still running, and
        // re-deriving from a missing session would silently resume the coder.
        let resolvedMode: AgentMode = generationSession?.agentMode ?? .code
        let isContinuation = agentStep > 0 || isInThinkingContinuation
        let turnMode: AgentMode = isContinuation ? (activeTurnMode ?? resolvedMode) : resolvedMode
        activeTurnMode = turnMode
        // Asserted BEFORE the slot claim below, deliberately. `setActiveMode`
        // skips its grammar re-sync while `CompletionEngine.isActive`, and the
        // claim is exactly that flag, so re-asserting after the claim meant the
        // heal this call exists for never ran. With the slot free it now does;
        // with a server turn holding it, the sync is still refused (that turn's
        // own mask is live) and the claim below fails, so nothing runs on the
        // stale registry.
        AgentHarness.shared.setActiveMode(turnMode)

        // The single-flight slot is taken HERE, before the preparation below, not
        // left to the launch at the end. A server turn can claim the slot while
        // this turn is still rendering and tokenizing the prompt, and the launch
        // would then lose that race and report it as a refusal — the same turn
        // lost, one step earlier. Holding the claim makes the server's own claim
        // fail instead, which its queue already handles by going back to waiting.
        // Nil means somebody already holds the slot; the launch below then reports
        // it exactly the way it always has. See `EngineTurnSlot.swift` and
        // `Benchmarks/PERF_FINDINGS.md`, "DSV4.1 #85".
        let turnSlotClaim = CompletionEngine.shared.reserveTurnSlot()
        defer { turnSlotClaim?.release() }

        let generationDate: Date
        if let sId = sessionId, let sess = sessions.first(where: { $0.id == sId }) {
            generationDate = sess.createdAt
        } else {
            generationDate = Date()
        }

        let generationModelName = modelDisplayName(for: generationSession)
        let cleanSystem = ModelConfig.buildEffectiveSystemPrompt(
            userPrompt: systemPrompt,
            config: modelConfig,
            summary: summary,
            modelName: generationModelName,
            modelPath: activeLoadedModelPath,
            currentDate: generationDate
        ).trimmingCharacters(in: .whitespacesAndNewlines)

        let modelSupportsThinking = supportsThinking(for: generationSession)
        let thinkingEnabled = isThinkingEnabled(for: generationSession) && modelSupportsThinking
        // Agent follow-up turns rebuild the system block here; they must carry
        // the same effort line as the first turn, or X2.6 renders two prompts.
        let thinkingEffortLevel = (thinkingEnabled && supportsThinkingLevels(for: generationSession))
            ? thinkingEffort(for: generationSession)
            : nil

        let formattedPrompt = ChatPromptFormatter.formatPrompt(
            prompt: prompt,
            cleanSystem: cleanSystem,
            thinkingEnabled: thinkingEnabled,
            modelSupportsThinking: modelSupportsThinking,
            modelConfig: modelConfig,
            thinkingEffort: thinkingEffortLevel,
            promptDialect: CompletionEngine.shared.promptDialect
        )

        // Expose the exact prompt text to the harness: the path-repair hint uses it
        // to echo back the absolute path the user named when a tool call used a bare
        // relative filename that does not exist under the working directory.
        AgentHarness.shared.lastPromptText = formattedPrompt

        let promptTokenIds: [UInt32]
        do {
            if let spliced = promptTokens, !spliced.isEmpty {
                // Agent continuations arrive as raw tokens spliced from the
                // previous turn's actual prompt+generated ids, so the pinned
                // prefix matches exactly even when generation was cut mid-tag.
                promptTokenIds = spliced
                print("🧩 [SPLICE] prompt assembled from \(spliced.count) raw tokens — skipping re-encode")
            } else {
                promptTokenIds = try tokenizer.encode(text: formattedPrompt)
            }
        } catch {
            let err = "❌ Tokenizer failed to encode prompt: \(error.localizedDescription)"
            isGeneratingText = false
            generationTask = nil
            showStatus(err, kind: .warning)
            return false
        }

        guard !promptTokenIds.isEmpty else {
            let err = "⚠️ Prompt tokenization produced 0 tokens."
            isGeneratingText = false
            generationTask = nil
            showStatus(err, kind: .warning)
            return false
        }

        // One flag drives both the engine's mid-stream tool-freeze/parse ranges
        // and the UI's agent-orchestration phase (historical semantics: the
        // session's toggle, defaulting to the global setting).
        let agentToolsEnabled = ((sessionId != nil) ? (sessions.first(where: { $0.id == sessionId })?.isAgentToolsEnabled ?? defaultAgentToolsEnabled) : defaultAgentToolsEnabled)
        let runAgentTools = agentToolsEnabled && !forceSynthesis

        let request = CompletionTurnRequest(
            promptTokenIds: promptTokenIds,
            formattedPrompt: formattedPrompt,
            sessionId: sessionId,
            sampling: TurnSampling(
                temperature: temperature,
                topP: topP,
                minP: minP,
                topK: topK,
                repetitionPenalty: repetitionPenalty,
                presencePenalty: presencePenalty,
                maxTokens: maxNewTokens
            ),
            memory: TurnMemorySettings(
                executionMode: memoryExecutionMode,
                budgetMode: memoryBudgetMode,
                kvPrecision: kvCachePrecision
            ),
            agentToolsEnabled: agentToolsEnabled,
            thinkingEnabled: thinkingEnabled,
            agentStep: agentStep,
            forceSynthesis: forceSynthesis,
            isSplicedContinuation: (promptTokens?.isEmpty == false),
            asOfDate: generationDate,
            images: images,
            // Only a spliced continuation reuses ids that already carry an expanded
            // span. A prompt that had to be RE-ENCODED (the tool section changed, a
            // synthesis directive was appended) starts from one marker per image
            // again, so a span carried from the old layout would describe positions
            // the new prompt does not have.
            imageSpans: (promptTokens?.isEmpty == false) ? carriedImageSpans : []
        )

        let launch: CompletionLaunch
        do {
            launch = try CompletionEngine.shared.launchTurn(request, claim: turnSlotClaim)
        } catch {
            let err = (error as? CompletionLaunchError)?.errorDescription ?? "❌ Failed to launch generation: \(error.localizedDescription)"
            isGeneratingText = false
            generationTask = nil
            if let sId = sessionId, let mId = messageId,
               let sIdx = sessions.firstIndex(where: { $0.id == sId }),
               let mIdx = sessions[sIdx].messages.firstIndex(where: { $0.id == mId }) {
                sessions[sIdx].messages[mIdx].content = err
                sessions[sIdx].messages[mIdx].isThinking = false
            }
            showStatus(err, kind: .warning)
            return false
        }

        // In-app turn log: one request-log row per chat/agent turn (source
        // chatUI; bodies follow the Store Request Content toggle, the rendered
        // prompt included). Begun after a successful launch so launch failures
        // stay unlogged (the chat surface owns those errors).
        let turnLog = ServerRequestLog.chatTurnLog(
            model: generationModelName,
            inputTokens: promptTokenIds.count,
            isAgentTurn: runAgentTools || agentStep > 0 || forceSynthesis,
            prompt: formattedPrompt
        )

        // Session token odometer seeding: resolved from turn-start state (it is
        // deterministic — the seeded base is this turn's prompt, the
        // conversation's context size) so a failed or cancelled prefill never
        // writes a phantom base.
        var cumulativeStepBase: Int = 0
        var cumulativeStepTracked = false
        if let sId = sessionId, sessions.firstIndex(where: { $0.id == sId }) != nil {
            cumulativeStepBase = ChatSession.contextTokenBase(
                promptTokenCount: promptTokenIds.count
            )
            cumulativeStepTracked = true
        }

        generationId &+= 1
        let myGenerationId = generationId

        isGeneratingText = true
        generatingSessionId = sessionId ?? selectedSessionId ?? sessions.first?.id
        AgentHarness.shared.currentSessionId = generatingSessionId
        generatedStreamText = ""
        thinkingText = ""
        responseText = ""
        isThinking = thinkingEnabled
        isThinkingExpanded = true
        generationTotalTokens = 0
        generationSpeedTokPerSec = 0.0
        generationElapsedMs = 0.0
        showStatus(thinkingEnabled ? "🧠 Reasoning..." : "⚡ Initializing Autoregressive Generation...")

        var priorThinking: String? = nil
        var priorContent: String? = nil
        if let sId = sessionId, let mId = messageId {
            if let sIdx = sessions.firstIndex(where: { $0.id == sId }),
               let mIdx = sessions[sIdx].messages.firstIndex(where: { $0.id == mId }) {
                priorThinking = sessions[sIdx].messages[mIdx].thinkingContent
                priorContent = sessions[sIdx].messages[mIdx].content
                sessions[sIdx].messages[mIdx].isThinking = thinkingEnabled
            }
        }

        generationTask = Task.detached(priority: .userInitiated) {
            var finishedResult: TurnResult? = nil
            // Last streamed frame's facts: what a failed or cancelled turn
            // still owes its log row. Frames are consumed in this detached
            // loop, so the accumulators stay local rather than hopping to the
            // MainActor.
            var streamedThinking = ""
            var streamedResponse = ""
            var streamedTokens = 0
            for await event in launch.stream {
                switch event {
                case .started:
                    break
                case .priming:
                    await MainActor.run {
                        self.showStatus("⚡ Priming resident weights into RAM...")
                    }
                case let .prefillProgress(status, speed, rssGB):
                    await MainActor.run {
                        if let speedValue = speed { self.generationSpeedTokPerSec = speedValue }
                        // Ingesting a prompt reports at ~10 Hz, and the status it
                        // carries is already on the transcript row and in the
                        // header pill. Repeating it here re-minted the notice's
                        // identity every tick, so the pill re-faded and VoiceOver
                        // read the same sentence out ten times a second.
                        if let rssValue = rssGB { self.currentRssGB = rssValue }
                        if let sId = sessionId, let mId = messageId,
                           let sIdx = self.sessions.firstIndex(where: { $0.id == sId }),
                           let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }) {
                            self.sessions[sIdx].messages[mIdx].prefillStatus = status
                        }
                    }
                case let .frame(frame):
                    turnLog.noteFirstToken()
                    streamedThinking = frame.thinkingText
                    streamedResponse = frame.responseText
                    streamedTokens = frame.tokensGenerated
                    await MainActor.run {
                        self.generatedStreamText = frame.rawText
                        self.thinkingText = frame.thinkingText
                        self.responseText = frame.responseText
                        self.isThinking = frame.isActivelyThinking
                        self.generationTotalTokens = frame.tokensGenerated
                        self.generationElapsedMs = frame.elapsedMs
                        self.generationSpeedTokPerSec = frame.tokensPerSecond
                        self.jetSpecMeanTau = frame.jetSpecMeanTau
                        self.jetSpecTotalDraftAccepted = frame.jetSpecDraftAccepted
                        self.jetSpecTotalDraftProposed = frame.jetSpecDraftProposed
                        let jetSpecBadge = frame.jetSpecActive ? " | 🚀 JetSpec τ=\(String(format: "%.1f", frame.jetSpecMeanTau))" : ""
                        self.currentRssGB = frame.residentMemoryGB
                        self.residentExpertCount = frame.residentExpertCount
                        self.totalExpertCount = frame.totalExpertCount
                        self.cacheHitRate = frame.cacheHitRatePercent
                        self.prefetchEfficiency = frame.prefetchEfficiencyPercent
                        self.lastPagingLatencyMs = frame.pagingLatencyMs

                        if let sId = sessionId, let mId = messageId {
                            if let sIdx = self.sessions.firstIndex(where: { $0.id == sId }),
                               let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }) {
                                let combinedThinking: String?
                                if let prior = priorThinking, !prior.isEmpty {
                                    if !frame.thinkingText.isEmpty {
                                        combinedThinking = prior + "\n\n---\n\n" + frame.thinkingText
                                    } else {
                                        combinedThinking = prior
                                    }
                                } else {
                                    combinedThinking = frame.thinkingText.isEmpty ? nil : frame.thinkingText
                                }

                                let combinedContent: String
                                if let prior = priorContent, !prior.isEmpty {
                                    if !frame.responseText.isEmpty {
                                        combinedContent = prior + "\n\n" + frame.responseText
                                    } else {
                                        combinedContent = prior
                                    }
                                } else {
                                    combinedContent = frame.responseText
                                }

                                self.sessions[sIdx].messages[mIdx].thinkingContent = combinedThinking
                                self.sessions[sIdx].messages[mIdx].content = combinedContent
                                // Keep activity fresh while a turn streams: the
                                // message timestamp is otherwise frozen at
                                // placeholder creation, so a long response would
                                // look stale to retention as soon as it finishes.
                                self.sessions[sIdx].messages[mIdx].timestamp = Date()
                                // No frame can exist before prefill produced a token,
                                // so the pill retires here even if the backend's own
                                // clear never arrived — the same belt the Apple
                                // Foundation path wears on its first visible text.
                                self.sessions[sIdx].messages[mIdx].prefillStatus = nil
                                self.sessions[sIdx].messages[mIdx].isThinking = frame.isActivelyThinking
                                self.sessions[sIdx].messages[mIdx].tokenCount = frame.tokensGenerated
                                self.sessions[sIdx].messages[mIdx].tokensPerSec = frame.tokensPerSecond
                                self.sessions[sIdx].messages[mIdx].timeToFirstTokenSeconds = frame.timeToFirstTokenSeconds
                                self.sessions[sIdx].messages[mIdx].thinkingTimeSeconds = frame.thinkingDurationSeconds
                                self.sessions[sIdx].messages[mIdx].jetSpecTau = (frame.jetSpecActive && frame.jetSpecDraftProposed > 0) ? frame.jetSpecMeanTau : nil
                                self.sessions[sIdx].messages[mIdx].jetSpecDraftAccepted = (frame.jetSpecActive && frame.jetSpecDraftAccepted > 0) ? frame.jetSpecDraftAccepted : nil
                                if cumulativeStepTracked {
                                    self.sessions[sIdx].cumulativeTokenCount = cumulativeStepBase + frame.tokensGenerated
                                }
                            }
                        }
                    }
                case let .failedTurn(failure):
                    turnLog.recordChatTurn(
                        status: failure.cancelled ? 200 : 500,
                        errorMessage: failure.message,
                        outputTokens: failure.tokensGenerated,
                        finishReason: nil,
                        thinking: streamedThinking,
                        response: streamedResponse,
                        toolCalls: []
                    )
                    if failure.cancelled {
                        // Cancellation: exact odometer tail only — the Stop /
                        // interrupt path already owns the flags and status line,
                        // and a replacement generation may be starting.
                        if cumulativeStepTracked, let sId = sessionId {
                            await MainActor.run {
                                if let sIdx = self.sessions.firstIndex(where: { $0.id == sId }) {
                                    self.sessions[sIdx].cumulativeTokenCount = cumulativeStepBase + failure.tokensGenerated
                                }
                            }
                        }
                        return
                    }
                    switch failure.kind {
                    case .cancelled:
                        return
                    case .prefillFailed:
                        await MainActor.run {
                            guard self.ownsGeneration(myGenerationId) else { return }
                            self.isGeneratingText = false
                            self.generationTask = nil
                            self.showStatus(failure.message)
                            if let sId = sessionId, let mId = messageId,
                               let sIdx = self.sessions.firstIndex(where: { $0.id == sId }),
                               let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }) {
                                self.sessions[sIdx].messages[mIdx].prefillStatus = nil
                                self.sessions[sIdx].messages[mIdx].isThinking = false
                            }
                        }
                    case .forwardPassFailed:
                        await MainActor.run {
                            self.isGeneratingText = false
                            self.generationTask = nil
                            self.showStatus(failure.message)
                            if let sId = sessionId, let mId = messageId,
                               let sIdx = self.sessions.firstIndex(where: { $0.id == sId }),
                               let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }) {
                                self.sessions[sIdx].messages[mIdx].isThinking = false
                            }
                            if cumulativeStepTracked, let sId = sessionId,
                               let sIdx = self.sessions.firstIndex(where: { $0.id == sId }) {
                                self.sessions[sIdx].cumulativeTokenCount = cumulativeStepBase + failure.tokensGenerated
                            }
                        }
                    case .backfillFailed:
                        await MainActor.run {
                            guard self.ownsGeneration(myGenerationId) else { return }
                            self.isGeneratingText = false
                            self.generationTask = nil
                            self.showStatus(failure.message)
                            if let sId = sessionId, let mId = messageId,
                               let sIdx = self.sessions.firstIndex(where: { $0.id == sId }),
                               let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }) {
                                self.sessions[sIdx].messages[mIdx].prefillStatus = nil
                                self.sessions[sIdx].messages[mIdx].isThinking = false
                            }
                            if cumulativeStepTracked, let sId = sessionId,
                               let sIdx = self.sessions.firstIndex(where: { $0.id == sId }) {
                                self.sessions[sIdx].cumulativeTokenCount = cumulativeStepBase + failure.tokensGenerated
                            }
                        }
                    }
                    return
                case let .finished(result):
                    finishedResult = result
                }
                if finishedResult != nil { break }
            }
            guard let result = finishedResult else {
                // The engine delivers failedTurn for cancellations, but Stop
                // cancels this consumer task too, so the stream can close
                // before that event arrives. A cancelled close is the user's
                // own stop: record it as informational (200, engine stop
                // message), not as a server error. Only an unclosed stream
                // with no result is exceptional — still owed its row.
                let cancelled = Task.isCancelled
                turnLog.recordChatTurn(
                    status: cancelled ? 200 : 500,
                    errorMessage: cancelled ? "⏹ Generation stopped by user." : "Generation stream closed without a result",
                    outputTokens: streamedTokens,
                    finishReason: nil,
                    thinking: streamedThinking,
                    response: streamedResponse,
                    toolCalls: []
                )
                return
            }

            // Engine turn complete: backfill + prefix pin already done. Re-bind
            // the historical local names so the agent-orchestration phase below
            // runs verbatim from the old implementation.
            let finalDecoded = result.rawDecodedText
            let finalThink = result.thinkingText
            let finalResp = result.responseText
            let formattedPrompt = result.formattedPrompt
            let promptTokenIds = result.promptTokenIds
            // `promptTokenIds` is the EXPANDED prompt: a picture is a run of image
            // tokens, not one marker. Every continuation below splices those ids
            // forward, so this layout has to travel with them — re-deriving it from
            // a marker count reads one picture as hundreds of placeholders and
            // refuses the turn.
            let promptImageSpans = result.imageSpans
            let generatedTokenIds = result.generatedTokenIds
            let finalElapsedMs = result.elapsedMs
            let finalTokPerSec = result.tokensPerSecond
            let finalTtft = result.timeToFirstTokenSeconds
            let finalThinkDuration = result.thinkingDurationSeconds
            let effectiveJetSpec = result.jetSpecActive
            let parsedResult = (calls: result.parsedToolCalls, brokenFragments: result.brokenFragments)
            let tokensGenerated = result.tokensGenerated
            let finalRss = result.residentMemoryGB
            let finalResCount = result.residentExpertCount
            let finalHitRate = result.cacheHitRatePercent
            let finalPrefetchEff = result.prefetchEfficiencyPercent
            let finalPageLat = result.pagingLatencyMs

            turnLog.recordChatTurn(
                status: 200,
                errorMessage: nil,
                outputTokens: tokensGenerated,
                finishReason: result.finishReason.rawValue,
                thinking: finalThink,
                response: finalResp,
                toolCalls: parsedResult.calls
            )

            // No-op "gesture" calls (shell_run echo/true/:) are the model's way of
            // signalling it is finished. Executing them restarts the loop and
            // produces a duplicate answer bubble, so treat them as turn end.
            // Gesture calls are skipped but still occupy their slot in the
            // transcript, so every call the assistant emitted gets exactly one
            // result (see `splitGestureCalls`).
            let gestureSplit = AgentHarness.splitGestureCalls(parsedResult.calls)
            let actionableCallIndices = gestureSplit.actionableIndices
            let actionableCalls = actionableCallIndices.map { parsedResult.calls[$0] }
            var responseSlots = [String?](repeating: nil, count: parsedResult.calls.count)
            for (idx, notice) in gestureSplit.skipNotices { responseSlots[idx] = notice }
            if !gestureSplit.skipNotices.isEmpty {
                print("🧹 [AGENT] dropped \(gestureSplit.skipNotices.count) no-op gesture call(s) — treating turn as finished")
            }
            let hasUncalledIntent = runAgentTools && actionableCalls.isEmpty && agentStep == 0
                && AgentStepBudget.allowsAnotherStep(step: agentStep, cap: self.agentStepCap)
                && AgentHarness.shared.detectUncalledActionIntent(content: finalResp, thinking: finalThink)
            let willContinueAgent = (!actionableCalls.isEmpty || hasUncalledIntent)
            // Collapse guard: the agent run ended with no tool calls and no usable answer,
            // but this run saw repeated degenerate (empty-argument) tool calls — force one
            // final synthesis turn (tools disabled) so the model answers instead of ending
            // in tag noise.
            //
            // Gesture guard: a no-op gesture call (bare `echo`/`true`/`:`) is the model's
            // "task finished" signal, so it ends the turn (QA #31). But when the model
            // gestures WITHOUT having written any answer — e.g. mid-task, right after a
            // confusing tool result — ending the run leaves the user with nothing. Force
            // the same synthesis turn in that case. If the model already wrote an answer
            // this turn (or earlier), the gesture stands as a normal end.
            let gestureOnlyTurn = runAgentTools && actionableCalls.isEmpty && !gestureSplit.skipNotices.isEmpty
            let responseHasAnswer = !AgentHarness.shared.responseTextWithoutToolCalls(finalResp)
                .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            let shouldForceSynthesis = runAgentTools && !willContinueAgent && !Task.isCancelled
                && (
                    (gestureOnlyTurn && !responseHasAnswer)
                    || (AgentHarness.shared.consecutiveEmptyToolCalls >= 2
                        && finalResp.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                )
                && (priorContent ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty

            await MainActor.run {
                // Superseded by a newer generation: it owns the UI flags, the task
                // handle and the prefix cache now, so this task must not touch any
                // of them (nor schedule another turn).
                guard self.ownsGeneration(myGenerationId) else { return }
                if !willContinueAgent {
                    self.isGeneratingText = false
                    self.generationTask = nil
                }
                self.generatedStreamText = finalDecoded
                self.thinkingText = finalThink
                self.responseText = finalResp
                self.isThinking = false
                self.generationTotalTokens = tokensGenerated
                self.generationElapsedMs = finalElapsedMs
                self.generationSpeedTokPerSec = finalTokPerSec
                let finalJetSpecBadge = effectiveJetSpec && result.jetSpecDraftProposed > 0 ? " (JetSpec τ=\(String(format: "%.1f", result.jetSpecMeanTau)), \(result.jetSpecDraftAccepted) draft tokens accepted)" : ""
                if !willContinueAgent {
                    self.showStatus("✨ Generated \(tokensGenerated) tokens in \(String(format: "%.2f", finalElapsedMs)) ms (\(String(format: "%.1f", finalTokPerSec)) tok/s)\(finalJetSpecBadge)")
                } else if !actionableCalls.isEmpty {
                    self.showStatus("⚙️ Executing \(actionableCalls.count) tool call(s)...")
                } else {
                    self.showStatus("🔄 Continuing agent multi-turn action...")
                }
                self.currentRssGB = finalRss
                self.residentExpertCount = finalResCount
                self.cacheHitRate = finalHitRate
                self.prefetchEfficiency = finalPrefetchEff
                self.lastPagingLatencyMs = finalPageLat

                if let sId = sessionId, let mId = messageId {
                    if let sIdx = self.sessions.firstIndex(where: { $0.id == sId }),
                       let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }) {
                        let combinedFinalThinking: String?
                        if let prior = priorThinking, !prior.isEmpty {
                            if !finalThink.isEmpty {
                                combinedFinalThinking = prior + "\n\n---\n\n" + finalThink
                            } else {
                                combinedFinalThinking = prior
                            }
                        } else {
                            combinedFinalThinking = finalThink.isEmpty ? nil : finalThink
                        }

                        let combinedFinalContent: String
                        if let prior = priorContent, !prior.isEmpty {
                            if !finalResp.isEmpty {
                                combinedFinalContent = prior + "\n\n" + finalResp
                            } else {
                                combinedFinalContent = prior
                            }
                        } else {
                            combinedFinalContent = finalResp
                        }

                        self.sessions[sIdx].messages[mIdx].thinkingContent = combinedFinalThinking
                        self.sessions[sIdx].messages[mIdx].content = combinedFinalContent
                        self.sessions[sIdx].messages[mIdx].timestamp = Date()
                        self.sessions[sIdx].messages[mIdx].isThinking = false
                        self.sessions[sIdx].messages[mIdx].tokenCount = tokensGenerated
                        self.sessions[sIdx].messages[mIdx].tokensPerSec = finalTokPerSec
                        self.sessions[sIdx].messages[mIdx].timeToFirstTokenSeconds = finalTtft
                        self.sessions[sIdx].messages[mIdx].thinkingTimeSeconds = finalThinkDuration
                        self.sessions[sIdx].messages[mIdx].jetSpecTau = (effectiveJetSpec && result.jetSpecDraftProposed > 0) ? result.jetSpecMeanTau : nil
                        self.sessions[sIdx].messages[mIdx].jetSpecDraftAccepted = (effectiveJetSpec && result.jetSpecDraftAccepted > 0) ? result.jetSpecDraftAccepted : nil
                        if cumulativeStepTracked {
                            self.sessions[sIdx].cumulativeTokenCount = cumulativeStepBase + tokensGenerated
                        }
                    }
                }
                if shouldForceSynthesis {
                    let endTag = (modelConfig?.isGemma4Model == true) ? "<turn|>" : ((modelConfig?.isSparkModel == true) ? "<｜end▁of▁sentence｜>" : ((modelConfig?.isLingModel == true) ? "<|role_end|>" : ((modelConfig?.isDeepSeekV41Model == true) ? CompletionEngine.shared.promptDialect.markers.endOfSentence : "<|im_end|>")) )
                    let assistantTurnText = self.closedAssistantTurnText(finalDecoded, endTag: endTag)
                    let synthesisDirective = """
                    \n\n<system>
                    IMPORTANT: All tool use is now DISABLED for this task. Your turn ended without a usable tool call — your recent calls carried empty arguments and could not be executed, or you emitted a no-op call that cannot advance the task — so the run was ended to protect the conversation from looping.
                    Using ONLY the tool outputs already shown above in this conversation — plus your own knowledge — now write your complete, self-contained final answer to the user's original question.
                    Do not emit any tool calls. Do not search again. Just answer.
                    </system>
                    """
                    let staleSynthesisPrompt = formattedPrompt + assistantTurnText + synthesisDirective
                    let synthesisPrompt = AgentHarness.shared.refreshingLoadedToolsSection(
                        inPrompt: staleSynthesisPrompt,
                        mode: turnMode
                    )
                    let synthesisToolsChanged = (synthesisPrompt != staleSynthesisPrompt)
                    let splicedTokens = self.buildSplicedContinuationTokens(
                        basePromptTokens: promptTokenIds,
                        generatedTokenIds: generatedTokenIds,
                        turnText: finalDecoded,
                        endTag: endTag,
                        suffix: synthesisDirective,
                        encode: { try tokenizer.encode(text: $0) }
                    )
                    let synthesisMsgId = UUID()
                    if let sId = sessionId, let sIdx = self.sessions.firstIndex(where: { $0.id == sId }) {
                        if let mId = messageId, let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }) {
                            self.sessions[sIdx].messages[mIdx].isThinking = false
                            self.sessions[sIdx].messages[mIdx].prefillStatus = nil
                        }
                        let synthMsg = ChatMessage(
                            id: synthesisMsgId,
                            role: .assistant,
                            content: "",
                            thinkingContent: nil,
                            isThinking: thinkingEnabled
                        )
                        self.sessions[sIdx].messages.append(synthMsg)
                    }
                    self.showStatus("🛡️ Tool loop guard hit — forcing final answer...")
                    self.startAutoregressiveGeneration(
                        customPrompt: synthesisPrompt,
                        promptTokens: synthesisToolsChanged ? nil : splicedTokens,
                        images: images,
                        carriedImageSpans: promptImageSpans,
                        sessionId: sessionId,
                        messageId: synthesisMsgId,
                        agentStep: agentStep + 1,
                        forceSynthesis: true
                    )
                } else if !willContinueAgent {
                    self.dequeueAndRunNextPromptIfNeeded(sessionId: sessionId)
                }
            }

            // Recovery: the model emitted only its reasoning block and stopped (empty reply),
            // or produced nothing at all (empty think block + immediate turn end). Regenerate
            // the whole turn from scratch so it can answer cleanly — grafting onto a
            // reconstructed " response" prefix produced duplicate reply text. Never applies when
            // there are parsed tool calls (or uncalled action intent): those turns must run the
            // agent harness, otherwise an identical deterministic replay is triggered.
            let emptyReplyAfterThinking = !finalThink.isEmpty && finalResp.isEmpty
            let fullyEmptyTurn = finalThink.isEmpty && finalResp.isEmpty && tokensGenerated <= 24
            if !isInThinkingContinuation && !willContinueAgent && !shouldForceSynthesis && (emptyReplyAfterThinking || fullyEmptyTurn) && !Task.isCancelled {
                await MainActor.run {
                    if let sId = sessionId,
                       let sIdx = self.sessions.firstIndex(where: { $0.id == sId }),
                       let mId = messageId,
                       let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }) {
                        self.sessions[sIdx].messages[mIdx].thinkingContent = nil
                        self.sessions[sIdx].messages[mIdx].content = ""
                        self.sessions[sIdx].messages[mIdx].tokenCount = 0
                        self.sessions[sIdx].messages[mIdx].isThinking = thinkingEnabled
                    }
                    self.startAutoregressiveGeneration(
                        customPrompt: formattedPrompt,
                        promptTokens: promptTokenIds,
                        images: images,
                        carriedImageSpans: promptImageSpans,
                        sessionId: sessionId,
                        messageId: messageId,
                        agentStep: agentStep,
                        isInThinkingContinuation: true
                    )
                }
                return
            }

            // Agent Harness Multi-Step Tool Execution
            if runAgentTools {
                // One record per emitted call, in transcript order. Gesture calls are
                // skipped (not executed) but still get a record whose output is the
                // skip notice, marked `isGesture` and hidden from the UI: the raw
                // gesture call stays in `msg.content`, and reconstructed for a later
                // user turn it would otherwise read as a call with no result and
                // invite a repeat. Built before the execute loop so it can address
                // records by id.
                var recordsByCallIndex: [ToolCallRecord?] = Array(repeating: nil, count: parsedResult.calls.count)
                for idx in actionableCallIndices {
                    let call = parsedResult.calls[idx]
                    var stringArgs: [String: String] = [:]
                    for (k, v) in call.arguments {
                        // Show the command that actually ran, not a heredoc over-run
                        // the tool trimmed before executing.
                        if call.name == "shell_run", k == "command", let s = v as? String {
                            stringArgs[k] = StreamingToolParser.truncateHeredocOverrun(s)
                        } else {
                            stringArgs[k] = "\(v)"
                        }
                    }
                    recordsByCallIndex[idx] = ToolCallRecord(
                        name: call.name,
                        arguments: stringArgs,
                        rawArguments: call.rawArguments,
                        status: .running
                    )
                }
                for (idx, _) in gestureSplit.skipNotices {
                    recordsByCallIndex[idx] = AgentHarness.gestureSkipRecord(for: parsedResult.calls[idx])
                }
                let orderedRecords = recordsByCallIndex.compactMap { $0 }
                if !orderedRecords.isEmpty {
                    await MainActor.run {
                        if let sId = sessionId, let mId = messageId,
                           let sIdx = self.sessions.firstIndex(where: { $0.id == sId }),
                           let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }) {
                            var currentCalls = self.sessions[sIdx].messages[mIdx].toolCalls ?? []
                            currentCalls.append(contentsOf: orderedRecords)
                            self.sessions[sIdx].messages[mIdx].toolCalls = currentCalls
                            if !actionableCalls.isEmpty {
                                self.showStatus("⚙️ Executing \(actionableCalls.count) tool call(s)...")
                            }
                        }
                    }
                }
                let initialRecords = actionableCallIndices.map { recordsByCallIndex[$0]! }

                if !actionableCalls.isEmpty {
                    let baseWdURL = self.agentWorkingDirectory.isEmpty ? nil : URL(fileURLWithPath: self.agentWorkingDirectory)
                    var anyCompleted = false
                    var ranCompleteTool = false

                    // A call and the approval that authorizes it are one
                    // operation. `executeTool` raises the same counter, but the
                    // approval wait ends before it starts, and a deferred MCP
                    // config reload landing in that gap would close the
                    // transport the approved call is about to use.
                    for (idx, call) in actionableCalls.enumerated() {
                        AgentActivity.shared.begin(.toolCall)
                        defer { AgentActivity.shared.end(.toolCall) }

                        if Task.isCancelled {
                            await MainActor.run {
                                guard self.ownsGeneration(myGenerationId) else { return }
                                self.isGeneratingText = false
                                self.generationTask = nil
                                self.showStatus("⏹ Tool execution stopped by user.")
                                if let sId = sessionId, let mId = messageId,
                                   let sIdx = self.sessions.firstIndex(where: { $0.id == sId }),
                                   let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }),
                                   var currentCalls = self.sessions[sIdx].messages[mIdx].toolCalls {
                                    for i in idx..<currentCalls.count {
                                        if currentCalls[i].status == .running {
                                            currentCalls[i].status = .error
                                            currentCalls[i].error = "Cancelled by user."
                                        }
                                    }
                                    self.sessions[sIdx].messages[mIdx].toolCalls = currentCalls
                                }
                            }
                            return
                        }

                        let recordId = initialRecords[idx].id
                        await MainActor.run {
                            self.showStatus("⚙️ Executing [\(idx + 1)/\(actionableCalls.count)]: \(call.name)...")
                        }

                        // Degenerate-call guard: reject tool calls whose required arguments
                        // are all empty WITHOUT executing (and without a HITL prompt), so a
                        // broken scaffold never burns an execution step or an approval.
                        if AgentHarness.hasEmptyRequiredArguments(toolName: call.name, arguments: call.arguments) {
                            AgentHarness.shared.recordToolCallValidity(false)
                            AgentHarness.shared.recordToolCallOutcome(toolName: call.name, outcome: .failed)
                            await MainActor.run {
                                if let sId = sessionId, let mId = messageId,
                                   let sIdx = self.sessions.firstIndex(where: { $0.id == sId }),
                                   let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }),
                                   var currentCalls = self.sessions[sIdx].messages[mIdx].toolCalls,
                                   let callIdx = currentCalls.firstIndex(where: { $0.id == recordId }) {
                                    currentCalls[callIdx].status = .error
                                    currentCalls[callIdx].error = "Rejected: required arguments were empty."
                                    self.sessions[sIdx].messages[mIdx].toolCalls = currentCalls
                                }
                            }
                            let emptyArgJSON = AgentHarness.toolErrorJSON(
                                tool: call.name,
                                error: "Empty arguments — this call carried no usable parameters and was not executed. Either provide real arguments, or stop calling tools and answer the user directly from what you already know."
                            )
                            responseSlots[actionableCallIndices[idx]] = emptyArgJSON
                            if !emptyArgJSON.isEmpty {
                                // Keep record output identical to what the model sees in the
                                // live observation turn so history reconstruction agrees.
                                if let sId = sessionId, let mId = messageId,
                                   let sIdx = self.sessions.firstIndex(where: { $0.id == sId }),
                                   let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }),
                                   var currentCalls = self.sessions[sIdx].messages[mIdx].toolCalls,
                                   let callIdx = currentCalls.firstIndex(where: { $0.id == recordId }) {
                                    currentCalls[callIdx].output = AgentHarness.renderToolResultForModel(emptyArgJSON)
                                    self.sessions[sIdx].messages[mIdx].toolCalls = currentCalls
                                }
                            }
                            continue
                        }
                        AgentHarness.shared.recordToolCallValidity(true)

                        // Human-In-The-Loop (HITL) Safety & Turbo Mode
                        let isTurbo = UserDefaults.standard.bool(forKey: "dynamoe_agent_turbo_mode")
                        // See the cloud path: a call plan mode withholds is
                        // refused by `executeTool` without running, so it must not
                        // raise an approval prompt first.
                        let isDestructive = AgentHarness.requiresApproval(toolName: call.name, mode: turnMode)

                        if isDestructive && !isTurbo {
                            await MainActor.run {
                                if let sId = sessionId, let mId = messageId,
                                   let sIdx = self.sessions.firstIndex(where: { $0.id == sId }),
                                   let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }),
                                   var currentCalls = self.sessions[sIdx].messages[mIdx].toolCalls,
                                   let callIdx = currentCalls.firstIndex(where: { $0.id == recordId }) {
                                    currentCalls[callIdx].status = .awaitingApproval
                                    self.sessions[sIdx].messages[mIdx].toolCalls = currentCalls
                                    self.showStatus("🛡️ Action requires confirmation: \(call.name)")
                                }
                            }

                            let approved = await ActionApprovalManager.shared.waitForApproval(id: recordId)
                            if !approved {
                                await MainActor.run {
                                    if let sId = sessionId, let mId = messageId,
                                       let sIdx = self.sessions.firstIndex(where: { $0.id == sId }),
                                       let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }),
                                       var currentCalls = self.sessions[sIdx].messages[mIdx].toolCalls,
                                       let callIdx = currentCalls.firstIndex(where: { $0.id == recordId }) {
                                        currentCalls[callIdx].status = .rejected
                                        currentCalls[callIdx].error = "Action rejected by user."
                                        self.sessions[sIdx].messages[mIdx].toolCalls = currentCalls
                                    }
                                }
                                let rejectJSON = AgentHarness.toolErrorJSON(tool: call.name, error: "Action rejected by user.")
                                responseSlots[actionableCallIndices[idx]] = rejectJSON
                                AgentHarness.shared.recordToolCallOutcome(toolName: call.name, outcome: .failed)
                                if let sId = sessionId, let mId = messageId,
                                   let sIdx = self.sessions.firstIndex(where: { $0.id == sId }),
                                   let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }),
                                   var currentCalls = self.sessions[sIdx].messages[mIdx].toolCalls,
                                   let callIdx = currentCalls.firstIndex(where: { $0.id == recordId }) {
                                    currentCalls[callIdx].output = AgentHarness.renderToolResultForModel(rejectJSON)
                                    self.sessions[sIdx].messages[mIdx].toolCalls = currentCalls
                                }
                                continue
                            }
                        }

                        let execResult = await AgentHarness.shared.executeTool(
                            call: call,
                            workingDirectory: baseWdURL,
                            maxOutputLength: self.maxToolOutputLength,
                            mode: turnMode
                        )
                        responseSlots[actionableCallIndices[idx]] = execResult.resultJSON
                        AgentHarness.shared.recordToolCallOutcome(
                            toolName: call.name,
                            outcome: execResult.record.status.toolCallOutcome
                        )
                        // See the cloud path: a withheld call must not clear the
                        // repeat guard on the mutating tools, and it ends nothing
                        // either — plan mode withholds `complete` so the turn can
                        // finish with a plan, and reading the refusal as
                        // completion would end the run with no plan and no
                        // handoff.
                        if execResult.record.status != .withheld {
                            AgentHarness.shared.recordToolCallFingerprint(
                                toolName: call.name,
                                arguments: call.arguments
                            )
                            if execResult.isCompleted {
                                anyCompleted = true
                            }
                        }
                        if execResult.record.status.endsAgentRun(
                            toolName: call.name,
                            isCompleted: execResult.isCompleted
                        ) {
                            ranCompleteTool = true
                        }

                        // Update ToolCallRecord in ChatMessage
                        await MainActor.run {
                            if let sId = sessionId, let mId = messageId,
                               let sIdx = self.sessions.firstIndex(where: { $0.id == sId }),
                               let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }),
                               var currentCalls = self.sessions[sIdx].messages[mIdx].toolCalls,
                               let callIdx = currentCalls.firstIndex(where: { $0.id == recordId }) {
                                currentCalls[callIdx].status = execResult.record.status
                                currentCalls[callIdx].output = execResult.record.output
                                currentCalls[callIdx].error = execResult.record.error
                                currentCalls[callIdx].executionDurationSeconds = execResult.record.executionDurationSeconds
                                self.sessions[sIdx].messages[mIdx].toolCalls = currentCalls
                            }
                        }
                    }

                    if Task.isCancelled {
                        await MainActor.run {
                            guard self.ownsGeneration(myGenerationId) else { return }
                            self.isGeneratingText = false
                            self.generationTask = nil
                            self.showStatus("⏹ Generation stopped by user.")
                        }
                        return
                    }

                    // Transcript order: one result per call the assistant emitted,
                    // gestures included.
                    let toolResponses = responseSlots.compactMap { $0 }

                    // A same-tool failure streak, too many failed calls overall, or one
                    // identical call re-issued past its limit means the model is stuck;
                    // stop spending steps and force a synthesis turn instead.
                    let agent = AgentHarness.shared
                    let toolFailureLoop = agent.consecutiveFailedToolCalls >= agent.failedToolCallLimit
                        || agent.totalFailedToolCalls >= agent.failedToolCallTotalLimit
                        || agent.maxRepeatedToolCallCount >= agent.repeatedToolCallLimit

                    // Resolve the cap once for this post-turn decision, so the
                    // status below names the budget that actually applied even
                    // if Settings changed while the turn was generating.
                    let stepCap = self.agentStepCap

                    // If not finished and steps remaining, invoke next step
                    if !anyCompleted && !toolFailureLoop,
                       AgentStepBudget.allowsAnotherStep(step: agentStep, cap: stepCap) {
                        // Auto-relay: completed background subagents whose final reports have
                        // not yet reached the conversation get injected as SUBAGENT_RESULT
                        // blocks here, so the coordinator can present them to the user.
                        let relayResponses = await MainActor.run {
                            let r = self.gatherCompletedSubagentRelayResponses(forSession: sessionId)
                            self.markSubagentSummariesRelayed(r)
                            return r
                        }
                        let allResponses = toolResponses + relayResponses

                        let isGemma4Continuation = (modelConfig?.isGemma4Model == true)
                        let toolResponseTurn: String
                        if isGemma4Continuation {
                            toolResponseTurn = AgentHarness.shared.formatGemmaToolResponseTurn(
                                responses: allResponses,
                                thinkingEnabled: thinkingEnabled
                            )
                        } else if modelConfig?.isLingModel == true {
                            toolResponseTurn = AgentHarness.shared.formatLingToolResponseTurn(
                                responses: allResponses,
                                thinkingEnabled: thinkingEnabled
                            )
                        } else if modelConfig?.isSparkModel == true {
                            toolResponseTurn = AgentHarness.shared.formatSparkToolResponseTurn(
                                responses: allResponses,
                                includeAssistantPrefix: true,
                                thinkingEnabled: thinkingEnabled
                            )
                        } else if modelConfig?.isDeepSeekV41Model == true {
                            // The payload inside the checkpoint's own `tool▁output` pair. The
                            // model answered the same from every container measured in `#27`,
                            // so the family's own spelling is used. The turn it returns ends
                            // by opening the next assistant turn: `formatPrompt` passes an
                            // already-templated prompt through verbatim, so nothing
                            // downstream would add that marker.
                            toolResponseTurn = AgentHarness.shared.formatDeepSeekV41ToolResponseTurn(
                                responses: allResponses,
                                thinkingEnabled: thinkingEnabled,
                                promptDialect: CompletionEngine.shared.promptDialect
                            )
                        } else {
                            toolResponseTurn = AgentHarness.shared.formatToolResponseTurn(
                                responses: allResponses,
                                includeThinkSuffix: thinkingEnabled
                            )
                        }
                        let endTag = (modelConfig?.isGemma4Model == true) ? "<turn|>" : ((modelConfig?.isSparkModel == true) ? "<｜end▁of▁sentence｜>" : ((modelConfig?.isLingModel == true) ? "<|role_end|>" : ((modelConfig?.isDeepSeekV41Model == true) ? CompletionEngine.shared.promptDialect.markers.endOfSentence : "<|im_end|>")) )
                        // Gemma keeps the tool call and its response in one model turn (no
                        // <turn|> between them), so the assistant text is not closed here and
                        // the response turn is appended with no separator. DeepSeek V4.1
                        // closes the call turn with its end-of-sentence marker and the result
                        // follows it directly, which is the shape the published probe used.
                        let assistantEndTag = isGemma4Continuation ? "" : endTag
                        let continuationSeparator = (isGemma4Continuation || modelConfig?.isDeepSeekV41Model == true) ? "" : "\n"
                        let assistantTurnText = self.closedAssistantTurnText(finalDecoded, endTag: assistantEndTag)
                        let staleNextPrompt = formattedPrompt + assistantTurnText + continuationSeparator + toolResponseTurn
                        // tools_load/tools_unload may have changed the registry THIS step;
                        // the spliced ids carry the prompt's OLD tools block, so when the
                        // section changes the continuation must re-encode (full re-prefill)
                        // rather than splice from the stale tokens.
                        let nextPrompt = AgentHarness.shared.refreshingLoadedToolsSection(
                            inPrompt: staleNextPrompt,
                            mode: turnMode
                        )
                        let toolsSectionChanged = (nextPrompt != staleNextPrompt)
                        let splicedTokens = self.buildSplicedContinuationTokens(
                            basePromptTokens: promptTokenIds,
                            generatedTokenIds: generatedTokenIds,
                            turnText: finalDecoded,
                            endTag: assistantEndTag,
                            suffix: continuationSeparator + toolResponseTurn,
                            encode: { try tokenizer.encode(text: $0) }
                        )
                        await MainActor.run {
                            let nextAssistantMsgId = UUID()
                            if let sId = sessionId, let sIdx = self.sessions.firstIndex(where: { $0.id == sId }) {
                                if let mId = messageId, let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }) {
                                    self.sessions[sIdx].messages[mIdx].isThinking = false
                                    self.sessions[sIdx].messages[mIdx].prefillStatus = nil
                                }
                                let nextMsg = ChatMessage(
                                    id: nextAssistantMsgId,
                                    role: .assistant,
                                    content: "",
                                    thinkingContent: nil,
                                    isThinking: thinkingEnabled
                                )
                                self.sessions[sIdx].messages.append(nextMsg)
                            }
                            self.startAutoregressiveGeneration(
                                customPrompt: nextPrompt,
                                promptTokens: toolsSectionChanged ? nil : splicedTokens,
                                images: images,
                                carriedImageSpans: promptImageSpans,
                                sessionId: sessionId,
                                messageId: nextAssistantMsgId,
                                agentStep: agentStep + 1
                            )
                        }
                        return
                    } else {
                        // All steps finished or complete tool called. If the search loop guard
                        // hard-stopped the run — or the model kept emitting tool calls with
                        // empty arguments — give the model one final synthesis turn with
                        // tool calling disabled so it actually answers instead of spinning.
                        let escalateToSynthesis = AgentHarness.shared.consecutiveEmptyToolCalls >= 2
                            || AgentHarness.shared.consecutiveFailedToolCalls >= AgentHarness.shared.failedToolCallLimit
                            || AgentHarness.shared.totalFailedToolCalls >= AgentHarness.shared.failedToolCallTotalLimit
                            || AgentHarness.shared.maxRepeatedToolCallCount >= AgentHarness.shared.repeatedToolCallLimit
                        if (AgentHarness.shared.lastSearchGuardAction == .forceSynthesis || escalateToSynthesis) && !ranCompleteTool {
                            let endTag = (modelConfig?.isGemma4Model == true) ? "<turn|>" : ((modelConfig?.isSparkModel == true) ? "<｜end▁of▁sentence｜>" : ((modelConfig?.isLingModel == true) ? "<|role_end|>" : "<|im_end|>") )
                            let assistantTurnText = self.closedAssistantTurnText(finalDecoded, endTag: endTag)
                            let toolResponseContext = toolResponses.map { AgentHarness.renderToolResultForModel($0) }.joined(separator: "\n")
                            let synthesisDirective = """
                            \n\n<system>
                            IMPORTANT: All tool use is now DISABLED for this task. You consumed your search budget by repeatedly searching, or your recent tool calls carried empty arguments or failed outright (a bad path, a missing binary) and could not advance the task. The run was forcibly ended to protect the conversation from looping.
                            Using ONLY the search results and tool outputs already shown above in this conversation — plus your own knowledge — now write your complete, self-contained final answer to the user's original question.
                            Do not emit any tool calls. Do not search again. Just answer.
                            </system>
                            """
                            let staleSynthesisPrompt = formattedPrompt + assistantTurnText + "\n" + toolResponseContext + synthesisDirective
                            let synthesisPrompt = AgentHarness.shared.refreshingLoadedToolsSection(
                                inPrompt: staleSynthesisPrompt,
                                mode: turnMode
                            )
                            let synthesisToolsChanged = (synthesisPrompt != staleSynthesisPrompt)
                            let splicedTokens = self.buildSplicedContinuationTokens(
                                basePromptTokens: promptTokenIds,
                                generatedTokenIds: generatedTokenIds,
                                turnText: finalDecoded,
                                endTag: endTag,
                                suffix: "\n" + toolResponseContext + synthesisDirective,
                                encode: { try tokenizer.encode(text: $0) }
                            )
                            await MainActor.run {
                                let synthesisMsgId = UUID()
                                if let sId = sessionId, let sIdx = self.sessions.firstIndex(where: { $0.id == sId }) {
                                    if let mId = messageId, let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }) {
                                        self.sessions[sIdx].messages[mIdx].isThinking = false
                                        self.sessions[sIdx].messages[mIdx].prefillStatus = nil
                                    }
                                    let synthMsg = ChatMessage(
                                        id: synthesisMsgId,
                                        role: .assistant,
                                        content: "",
                                        thinkingContent: nil,
                                        isThinking: thinkingEnabled
                                    )
                                    self.sessions[sIdx].messages.append(synthMsg)
                                }
                                self.showStatus("🛡️ Tool loop guard hit — forcing final answer...")
                                self.startAutoregressiveGeneration(
                                    customPrompt: synthesisPrompt,
                                    promptTokens: synthesisToolsChanged ? nil : splicedTokens,
                                    images: images,
                                    carriedImageSpans: promptImageSpans,
                                    sessionId: sessionId,
                                    messageId: synthesisMsgId,
                                    agentStep: agentStep + 1,
                                    forceSynthesis: true
                                )
                            }
                            return
                        }

                        // Result-delivery relay: the run is ending, but this session still has
                        // completed subagents whose final reports never reached the conversation
                        // (e.g. the model spawned a background subagent and then stopped without
                        // polling). Wait briefly for in-flight subagents, then inject their
                        // reports and run ONE tools-disabled turn so the coordinator presents
                        // them to the user instead of dropping them.
                        if !ranCompleteTool {
                            await self.awaitSessionSubagents(sessionId: sessionId, timeout: 120)
                            let pendingRelay = await MainActor.run {
                                let r = self.gatherCompletedSubagentRelayResponses(forSession: sessionId)
                                self.markSubagentSummariesRelayed(r)
                                return r
                            }
                            if !pendingRelay.isEmpty {
                                let endTag = (modelConfig?.isGemma4Model == true) ? "<turn|>" : ((modelConfig?.isSparkModel == true) ? "<｜end▁of▁sentence｜>" : ((modelConfig?.isLingModel == true) ? "<|role_end|>" : "<|im_end|>") )
                                let assistantTurnText = self.closedAssistantTurnText(finalDecoded, endTag: endTag)
                                let relayContext = pendingRelay.joined(separator: "\n\n")
                                let relayDirective = """
                                \n\n<system>
                                A delegated subagent has finished its work. Its final report is below. You are the coordinator: the user only talks to you, and you are responsible for presenting subagent results.
                                Using the report below, write your complete final answer to the user's original question NOW. Do not emit any tool calls.

                                \(relayContext)
                                </system>
                                """
                                let relayPrompt = formattedPrompt + assistantTurnText + relayDirective
                                let splicedTokens = self.buildSplicedContinuationTokens(
                                    basePromptTokens: promptTokenIds,
                                    generatedTokenIds: generatedTokenIds,
                                    turnText: finalDecoded,
                                    endTag: endTag,
                                    suffix: relayDirective,
                                    encode: { try tokenizer.encode(text: $0) }
                                )
                                await MainActor.run {
                                    let relayMsgId = UUID()
                                    if let sId = sessionId, let sIdx = self.sessions.firstIndex(where: { $0.id == sId }) {
                                        let relayMsg = ChatMessage(
                                            id: relayMsgId,
                                            role: .assistant,
                                            content: "",
                                            thinkingContent: nil,
                                            isThinking: thinkingEnabled
                                        )
                                        self.sessions[sIdx].messages.append(relayMsg)
                                    }
                                    self.showStatus("📨 Subagent report ready — relaying results to user...")
                                    self.startAutoregressiveGeneration(
                                        customPrompt: relayPrompt,
                                        promptTokens: splicedTokens,
                                        images: images,
                                        carriedImageSpans: promptImageSpans,
                                        sessionId: sessionId,
                                        messageId: relayMsgId,
                                        agentStep: agentStep + 1,
                                        forceSynthesis: true
                                    )
                                }
                                return
                            }
                        }

                        await MainActor.run {
                            self.isGeneratingText = false
                            self.generationTask = nil
                            if let notice = AgentStepBudget.runEndNotice(
                                anyCompleted: anyCompleted,
                                loopGuardTripped: toolFailureLoop,
                                cap: stepCap
                            ) {
                                self.showStatus(notice)
                            }
                            self.dequeueAndRunNextPromptIfNeeded(sessionId: sessionId)
                        }
                        return
                    }
                } else if AgentStepBudget.allowsAnotherStep(step: agentStep, cap: self.agentStepCap)
                    && gestureSplit.skipNotices.isEmpty
                    && AgentHarness.shared.hasToolCallMarkup(in: finalDecoded) {
                    // The model attempted a tool call that the parser could not turn into
                    // a usable call: a call abandoned mid-stream (opener without a closer),
                    // a wrong dialect, or a body with no usable parameter. The parser drops
                    // such fragments silently, so without this branch the turn simply ends
                    // and the intended action never runs. This must apply at EVERY agent
                    // step, not just the first: observed a run that stopped dead at step 3
                    // after the model emitted another malformed call, because the recovery
                    // was gated to `agentStep == 0`. Gesture calls (intentional end signals)
                    // are parsed, so they never reach this branch.
                    if Task.isCancelled {
                        await MainActor.run {
                            guard self.ownsGeneration(myGenerationId) else { return }
                            self.isGeneratingText = false
                            self.generationTask = nil
                            self.showStatus("⏹ Generation stopped by user.")
                        }
                        return
                    }

                    let continuationTurn: String
                    if modelConfig?.isLingModel == true {
                        continuationTurn = AgentHarness.shared.formatLingTruncatedToolCallTurn(
                            thinkingEnabled: thinkingEnabled
                        )
                    } else {
                        continuationTurn = AgentHarness.shared.formatTruncatedToolCallTurn(
                            includeThinkSuffix: thinkingEnabled
                        )
                    }
                    let endTag = (modelConfig?.isGemma4Model == true) ? "<turn|>" : ((modelConfig?.isSparkModel == true) ? "<｜end▁of▁sentence｜>" : ((modelConfig?.isLingModel == true) ? "<|role_end|>" : "<|im_end|>") )
                    let assistantTurnText = self.closedAssistantTurnText(finalDecoded, endTag: endTag)
                    let nextPrompt = formattedPrompt + assistantTurnText + "\n" + continuationTurn
                    let splicedTokens = self.buildSplicedContinuationTokens(
                        basePromptTokens: promptTokenIds,
                        generatedTokenIds: generatedTokenIds,
                        turnText: finalDecoded,
                        endTag: endTag,
                        suffix: "\n" + continuationTurn,
                        encode: { try tokenizer.encode(text: $0) }
                    )
                    await MainActor.run {
                        let nextAssistantMsgId = UUID()
                        if let sId = sessionId, let sIdx = self.sessions.firstIndex(where: { $0.id == sId }) {
                            if let mId = messageId, let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }) {
                                self.sessions[sIdx].messages[mIdx].isThinking = false
                                self.sessions[sIdx].messages[mIdx].prefillStatus = nil
                            }
                            let nextMsg = ChatMessage(
                                id: nextAssistantMsgId,
                                role: .assistant,
                                content: "",
                                thinkingContent: nil,
                                isThinking: thinkingEnabled
                            )
                            self.sessions[sIdx].messages.append(nextMsg)
                        }
                        self.showStatus("⚠️ Tool call was cut off — asking the model to re-emit it...")
                        self.startAutoregressiveGeneration(
                            customPrompt: nextPrompt,
                            promptTokens: splicedTokens,
                            images: images,
                            carriedImageSpans: promptImageSpans,
                            sessionId: sessionId,
                            messageId: nextAssistantMsgId,
                            agentStep: agentStep + 1
                        )
                    }
                    return
                } else if agentStep == 0
                    && AgentStepBudget.allowsAnotherStep(step: agentStep, cap: self.agentStepCap)
                    && AgentHarness.shared.detectUncalledActionIntent(content: finalResp, thinking: finalThink) {
                    if Task.isCancelled {
                        await MainActor.run {
                            guard self.ownsGeneration(myGenerationId) else { return }
                            self.isGeneratingText = false
                            self.generationTask = nil
                            self.showStatus("⏹ Generation stopped by user.")
                        }
                        return
                    }

                    let continuationTurn: String
                    if modelConfig?.isLingModel == true {
                        continuationTurn = AgentHarness.shared.formatLingActionContinuationTurn(
                            thinkingEnabled: thinkingEnabled
                        )
                    } else {
                        continuationTurn = AgentHarness.shared.formatActionContinuationTurn(
                            includeThinkSuffix: thinkingEnabled
                        )
                    }
                    let endTag = (modelConfig?.isGemma4Model == true) ? "<turn|>" : ((modelConfig?.isSparkModel == true) ? "<｜end▁of▁sentence｜>" : ((modelConfig?.isLingModel == true) ? "<|role_end|>" : "<|im_end|>") )
                    let assistantTurnText = self.closedAssistantTurnText(finalDecoded, endTag: endTag)
                    let nextPrompt = formattedPrompt + assistantTurnText + "\n" + continuationTurn
                    let splicedTokens = self.buildSplicedContinuationTokens(
                        basePromptTokens: promptTokenIds,
                        generatedTokenIds: generatedTokenIds,
                        turnText: finalDecoded,
                        endTag: endTag,
                        suffix: "\n" + continuationTurn,
                        encode: { try tokenizer.encode(text: $0) }
                    )
                    await MainActor.run {
                        let nextAssistantMsgId = UUID()
                        if let sId = sessionId, let sIdx = self.sessions.firstIndex(where: { $0.id == sId }) {
                            if let mId = messageId, let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }) {
                                self.sessions[sIdx].messages[mIdx].isThinking = false
                                self.sessions[sIdx].messages[mIdx].prefillStatus = nil
                            }
                            let nextMsg = ChatMessage(
                                id: nextAssistantMsgId,
                                role: .assistant,
                                content: "",
                                thinkingContent: nil,
                                isThinking: thinkingEnabled
                            )
                            self.sessions[sIdx].messages.append(nextMsg)
                        }
                        self.startAutoregressiveGeneration(
                            customPrompt: nextPrompt,
                            promptTokens: splicedTokens,
                            images: images,
                            carriedImageSpans: promptImageSpans,
                            sessionId: sessionId,
                            messageId: nextAssistantMsgId,
                            agentStep: agentStep + 1
                        )
                    }
                    return
                }
            }

        }
        return true
    }

    private func updatePagingStats() {
        self.currentRssGB = WorkingSetManager.shared.effectiveResidentMemoryGB
        self.residentExpertCount = WorkingSetManager.shared.residentExpertsCount
        self.totalExpertCount = WorkingSetManager.shared.totalExpertKeysCount
        self.cacheHitRate = WorkingSetManager.shared.cacheHitRatePercent
        self.lastPagingLatencyMs = WorkingSetManager.shared.currentPagingLatencyMs
    }

    private func applyMemoryBudget(_ mode: MemoryBudgetMode) {
        guard let summary = summary else { return }
        WorkingSetManager.shared.setBudgetMode(mode: mode, shardBuffers: shardBuffers, summary: summary)
        updatePagingStats()
        pagingStatusMessage = "⚡ Budget updated: \(mode.rawValue)"
    }

    private func flushExpertCache() {
        WorkingSetManager.shared.flushAllExperts(shardBuffers: shardBuffers)
        updatePagingStats()
        pagingStatusMessage = "🧹 Expert cache flushed (MADV_DONTNEED)"
    }

    private func preFaultAllWeights() {
        guard let summary = summary else { return }
        WorkingSetManager.shared.preFaultAll(shardBuffers: shardBuffers, summary: summary)
        updatePagingStats()
        pagingStatusMessage = "🚀 All weights pre-faulted into RAM"
    }

    private func applyMemoryExecutionMode(_ mode: MemoryExecutionMode) {
        guard let summary = summary else { return }
        let mappedGB = summary.sizeGb
        let eff = mode.resolveEffectiveMode(modelFootprintGB: mappedGB)
        if eff == .residentRAM {
            pagingStatusMessage = "⚡ Priming resident weights into RAM..."
            DispatchQueue.global(qos: .userInitiated).async {
                WorkingSetManager.shared.preFaultAll(shardBuffers: self.shardBuffers, summary: summary)
                Task { @MainActor in
                    self.pagingStatusMessage = "⚡ Operating in Full RAM Resident Mode (Zero Disk Paging)"
                    self.updatePagingStats()
                    self.syncSettingsWindowIfNeeded()
                }
            }
        } else {
            let dirUrl: URL? = activeLoadedModelPath != nil ? {
                let p = activeLoadedModelPath!
                var isD: ObjCBool = false
                FileManager.default.fileExists(atPath: p, isDirectory: &isD)
                return isD.boolValue ? URL(fileURLWithPath: p) : URL(fileURLWithPath: p).deletingLastPathComponent()
            }() : nil
            WorkingSetManager.shared.initialize(summary: summary, shardBuffers: shardBuffers, mode: memoryBudgetMode)
            pagingStatusMessage = "🌊 Operating in Dynamic SSD Streaming Mode"
            updatePagingStats()
            syncSettingsWindowIfNeeded()
        }
    }

    private func applyMemoryBudgetMode(_ budget: MemoryBudgetMode) {
        applyMemoryBudget(budget)
        syncSettingsWindowIfNeeded()
    }

    /// Stops any running generation and waits for its teardown before shared model
    /// state is touched. A cancelled generation may still be inside a kernel
    /// dispatch whose mmaps are unmapped when the old engine releases, and a load
    /// closes the shared expert file descriptors and re-initializes the working
    /// set. `switchModel` performs this dance for its callers, but direct loads
    /// (the selected-session fallback, the open-panel loader) reach the load task
    /// without it, so the coordination lives here too. A no-op when idle.
    private func awaitGenerationTeardown() async {
        if isGeneratingText {
            stopAutoregressiveGeneration()
        }
        // The engine's compute task can outlive the consumer relay task while
        // it unwinds cooperatively at a layer boundary; loads must wait it out.
        await CompletionEngine.shared.awaitIdle()
        if let teardown = generationTeardownTask {
            await teardown.value
            // A generation may have started while we waited: stop it and wait for
            // that teardown too, since the load resets the buffers it uses.
            if isGeneratingText {
                stopAutoregressiveGeneration()
            }
            if let next = generationTeardownTask {
                await next.value
            }
        }
    }

    /// Starts loading a model's engine, summary, and Metal buffers. Returns the
    /// load task so callers that must run against the new model can await the
    /// specific load instead of racing it; UI callers ignore the result.
    @discardableResult
    private func loadAndBridgeToMetal(filePath: String) -> Task<Void, Never> {
        // A load request for the system-model sentinel (a session restore racing
        // the registry scan, or a raw path hand-off) must not attempt to mmap a
        // fake path: activate the on-device backend directly instead.
        if filePath == AppleFoundationModelService.modelPath {
            return coordinateAppleFoundationModelActivation(of: AppleFoundationModelService.makeDiscoveredModel(), sessionId: nil)
        }
        // Same rule for a cloud sentinel: a session restore racing the
        // registry scan must activate the provider backend, never mmap a fake
        // path. The entry is rebuilt ad hoc from the path when the registered
        // list no longer holds it.
        if let resolved = CloudModelAddress.parsePath(filePath) {
            let entry = CloudModelRegistry.adHocDiscoveredModel(
                provider: resolved.provider,
                providerModelId: resolved.providerModelId
            )
            return coordinateCloudModelActivation(of: entry, sessionId: nil)
        }
        self.isLoadingModel = true
        self.metalStatus = "⏳ Loading model engine & zero-copy weights..."

        // Tag this request and chain it onto the previous load: a stale load must
        // never install its engine over a newer one, and two loads must never
        // mutate the shared engine/buffer state concurrently.
        modelLoadToken &+= 1
        let token = modelLoadToken
        let previousLoad = modelLoadTask

        let memoryExecutionMode = self.memoryExecutionMode
        let memoryBudgetMode = self.memoryBudgetMode
        let shouldPinBackbone = pinBackboneWeights

        let loadTask = Task.detached(priority: .userInitiated) {
            await previousLoad?.value
            // If a newer load was requested while we waited, skip this stale
            // request entirely instead of mutating shared state under it.
            let isStale = await MainActor.run { self.modelLoadToken != token }
            if isStale { return }
            // A generation may still be using the shared engine, buffers, and
            // expert descriptors that this load is about to replace. SwitchModel
            // waits the generation out before loading; direct callers reach this
            // task without it, so stop and await any generation here too.
            await self.awaitGenerationTeardown()
            // ...and then HOLD the generation slot for the rest of the load. Waiting
            // a turn out only covers the turn that was running: `configure` below
            // replaces the shard buffers and retires the DeepSeek runtime, so a turn
            // that started in between would be computing on mappings that no longer
            // exist. The slot is the one thing both callers check, so the load takes
            // it and gives it back once the new engine is installed. Nil means
            // another caller got there first, which is no worse than the behaviour
            // before the claim existed.
            let swapClaim = CompletionEngine.shared.reserveTurnSlot()
            defer { swapClaim?.release() }
            // The wait can be long: a newer load may have superseded this one, in
            // which case it must not touch the shared state the newer one owns.
            let superseded = await MainActor.run { self.modelLoadToken != token }
            if superseded { return }
            // Drop file descriptors cached from a previously loaded model; they are
            // keyed by layerIndex only, so a new model with the same layer numbering
            // would otherwise pread the old model's packed expert files.
            ExpertIOThreadPool.shared.closeAllLayerFDs()
            do {
                guard let device = MTLCreateSystemDefaultDevice() else {
                    await MainActor.run {
                        guard self.modelLoadToken == token else { return }
                        self.metalStatus = "❌ Failed to initialize Metal GPU."
                        self.isLoadingModel = false
                    }
                    return
                }

                // Attempt to load and parse HuggingFace config.json if present
                let fileUrl = URL(fileURLWithPath: filePath)
                var isDir: ObjCBool = false
                FileManager.default.fileExists(atPath: filePath, isDirectory: &isDir)
                let dirUrl = isDir.boolValue ? fileUrl : fileUrl.deletingLastPathComponent()

                // A dense repack is one contiguous blob, and its weights are only
                // reachable if they can be mapped as MTLBuffers on THIS GPU. The
                // device has to be known before the engine is built for that check
                // to mean anything: the engine prefers the repack, and discovering
                // the limit afterwards leaves a model that "loads" and then throws
                // "weight tensor not found" the first time a packed projection runs.
                if DenseRepacker.setAsideRepackIfUnusable(dir: dirUrl, maxBufferLength: UInt64(device.maxBufferLength)) {
                    print("⚠️ [Dense] the quantized blob cannot be mapped on this GPU. Falling back to the checkpoint's safetensors shards.")
                }

                let loadedEngine = try DynaMoeEngine(filePath: filePath)
                var loadedSummary = try loadedEngine.getSummary()

                let cfg = ModelConfig.load(from: dirUrl)
                let arch = cfg?.resolveArchitectureType(summary: loadedSummary) ?? (loadedSummary.maxExpertId > 0 ? .hybridSsmMoe : .denseTransformer)

                // Auto-load tokenizer.json from model directory if present
                let tokUrl = dirUrl.appendingPathComponent("tokenizer.json")
                let tok: DynaMoeTokenizer?
                if FileManager.default.fileExists(atPath: tokUrl.path) {
                    tok = try? DynaMoeTokenizer(tokenizerPath: tokUrl.path)
                } else {
                    tok = nil
                }
                
                // Map every shard into Metal zero-copy space
                // Large models (e.g. Gemma 4's 46 GB shard) can exceed the device's
                // maxBufferLength, which is far below the shard size. Split any such
                // shard into per-buffer segments, remapping tensor offsets so the
                // rest of the engine keeps addressing tensors by (shardIndex, offset).
                loadedSummary = try InferenceEngine.segmentShardsForMetal(loadedSummary, maxBufferLength: UInt64(device.maxBufferLength))
                var buffers: [UInt32: MTLBuffer] = [:]
                var mappedGB: Double = 0.0
                
                for shard in loadedSummary.shards {
                    let address = UInt(shard.baseAddress)
                    guard let pointer = UnsafeMutableRawPointer(bitPattern: address) else { continue }
                    let length = Int(shard.length)
                    guard length > 0 else {
                        print("ℹ️ [Metal] Skipping zero-length shard #\(shard.index) (\(shard.filename)).")
                        continue
                    }
                    guard length <= Int(device.maxBufferLength) else {
                        print("⚠️ [Metal Warning] Shard #\(shard.index) (\(shard.filename)) size \(Double(length) / (1024*1024*1024)) GB exceeds device maxBufferLength (\(Double(device.maxBufferLength) / (1024*1024*1024)) GB). Skipping single-buffer mapping.")
                        continue
                    }
                    
                    if let buffer = device.makeBuffer(bytesNoCopy: pointer, length: length, options: .storageModeShared, deallocator: nil) {
                        buffers[shard.index] = buffer
                        mappedGB += Double(length) / (1024.0 * 1024.0 * 1024.0)
                    } else {
                        print("⚠️ [Metal Warning] Failed to map shard #\(shard.index) (\(shard.filename), \(String(format: "%.2f", Double(length) / 1073741824.0)) GB) as a zero-copy buffer. Tensors in this segment will be unavailable.")
                    }
                }

                // A segment with no buffer leaves its tensors unreadable, and that
                // only surfaces later as "weight tensor not found" inside whatever
                // dispatch touches one first, with no mention of the segment. Say
                // so here, where the shard is still known. Shards with no address
                // are not attempted (the loop skips them the same way) so they are
                // not counted.
                let mappable = loadedSummary.shards.filter { $0.length > 0 && $0.baseAddress != 0 }
                if buffers.count < mappable.count {
                    let missing = mappable.filter { buffers[$0.index] == nil }.map { $0.filename }
                    throw NSError(
                        domain: "DynaMoE.ModelLoad",
                        code: 2,
                        userInfo: [NSLocalizedDescriptionKey:
                            "\(missing.count) weight segment(s) could not be mapped on this GPU (\(missing.prefix(3).joined(separator: ", "))). "
                                + "A zero-copy buffer is limited to \(String(format: "%.1f", Double(device.maxBufferLength) / 1073741824.0)) GB per segment."]
                    )
                }

                // Hoisted so the guarded install below can exclude the pinned backbone
                // when it adopts the shard registry.
                var pinnedShardIndex: UInt32? = nil
                let isFlashMoE = ExpertRepacker.isPackedFormat(dir: dirUrl)
                if isFlashMoE {
                    ExpertIOThreadPool.shared.initialize(numThreads: 8)

                    // FlashMoE backbone pinning: the invariant weights (model_weights.bin —
                    // every layer's q/k/v/o/GDN projections, norms, lm_head, embeddings) are
                    // read by the GPU on EVERY token. Serving them from the file-backed mmap
                    // re-faults them at ~0.065-2.5 GB/s whenever GPU work on other buffers
                    // unwires the mappings (measured collapse to ~0.3 tok/s on 16 GB). Pread
                    // the file once into an anonymous .storageModeShared buffer so all
                    // backbone reads become resident DRAM traffic (~68 GB/s).
                    if shouldPinBackbone,
                       let shard0 = loadedSummary.shards.first(where: { $0.filename == "model_weights.bin" }) {
                        let length = Int(shard0.length)
                        if length > 0, length <= Int(device.maxBufferLength),
                           let (pinned, reusedPin) = BackbonePinPool.shared.acquire(device: device, length: length) {
                            let pinPath = dirUrl.appendingPathComponent("model_weights.bin")
                            let fd = open(pinPath.path, O_RDONLY | O_CLOEXEC)
                            if fd >= 0 {
                                await MainActor.run {
                                    guard self.modelLoadToken == token else { return }
                                    self.metalStatus = "⚡ Pinning backbone weights (\(String(format: "%.1f", Double(length) / 1073741824.0)) GB) into RAM..."
                                }
                                let footprintBefore = getActivityMonitorFootprintGB()
                                if let elapsed = ExpertIOThreadPool.preadFileIntoBuffer(fd: fd, dst: pinned.contents(), length: length, threads: 8) {
                                    buffers[shard0.index] = pinned
                                    pinnedShardIndex = shard0.index
                                    let pinGB = Double(length) / 1073741824.0
                                    let telemetry = BackbonePinTelemetry.record(buffer: pinned)
                                    let footprintAfter = getActivityMonitorFootprintGB()
                                    print("⚡ [FlashMoE] Pinned backbone (\(String(format: "%.2f", pinGB)) GB) into anonymous GPU memory in \(String(format: "%.2f", elapsed))s\(reusedPin ? " (reused retained pin)" : " (new allocation)") — per-token mmap re-faults eliminated (pin #\(telemetry.ordinal); live pin buffers: \(telemetry.liveDistinct) = \(String(format: "%.2f", telemetry.liveDistinctGB)) GB; footprint \(String(format: "%.2f", footprintBefore)) -> \(String(format: "%.2f", footprintAfter)) GB, delta +\(String(format: "%.2f", footprintAfter - footprintBefore)))")
                                } else {
                                    print("⚠️ [FlashMoE] Backbone pinning pread incomplete — falling back to mmap path")
                                }
                                close(fd)
                            } else {
                                print("⚠️ [FlashMoE] Cannot open \(pinPath.path) for pinning (errno=\(errno)) — falling back to mmap path")
                            }
                        } else {
                            print("⚠️ [FlashMoE] Backbone pinning skipped: length \(length) invalid or exceeds maxBufferLength (\(Double(device.maxBufferLength) / 1073741824.0) GB)")
                        }
                    }
                } else {
                    WorkingSetManager.shared.initialize(summary: loadedSummary, shardBuffers: buffers, mode: memoryBudgetMode)
                    let effMode = memoryExecutionMode.resolveEffectiveMode(modelFootprintGB: mappedGB)
                    if effMode == .residentRAM {
                        WorkingSetManager.shared.preFaultAll(shardBuffers: buffers, summary: loadedSummary)
                    }
                }
                
                // Preflight opens shards and walks the tensor catalog, so run it
                // here, off the main actor: doing it inside `MainActor.run` stalls
                // the UI for every load of a multi-thousand-tensor checkpoint.
                var deepSeekReport: [String]?
                if arch == .deepSeekV41Flash,
                   let dsConfig = try? DeepSeekV41Config.decode(
                       contentsOf: dirUrl.appendingPathComponent("config.json")
                   ) {
                    deepSeekReport = DeepSeekV41LoadDiagnostics.report(
                        snapshotDir: dirUrl,
                        config: dsConfig,
                        summary: loadedSummary,
                        contextLength: min(
                            dsConfig.maxPositionEmbeddings,
                            UserDefaults.standard.object(forKey: "dynamoe_max_tokens") as? Int ?? 8_192
                        )
                    )
                }

                await MainActor.run {
                    // A newer load superseded this one; drop the result rather than
                    // replacing the requested engine with this stale one.
                    guard self.modelLoadToken == token else { return }
                    self.engine = loadedEngine
                    self.summary = loadedSummary
                    self.shardBuffers = buffers
                    // Adopt the shard registry only now that this load is the accepted
                    // one: a superseded task is dropped at the guard above and its
                    // engine's mmaps can go away behind it, so registering any earlier
                    // would leave the residency loop walking a rejected model's pointers.
                    // A pinned backbone (FlashMoE) is excluded — its anonymous pages are
                    // dirty heap that the footprint line already counts, not clean
                    // unified-cache pages.
                    var registryBuffers = buffers
                    if let pinnedIdx = pinnedShardIndex {
                        registryBuffers.removeValue(forKey: pinnedIdx)
                    }
                    WorkingSetManager.shared.registerShardMappings(registryBuffers)
                    self.modelConfig = cfg
                    self.detectedArchitecture = arch
                    // DeepSeek V4.1 is served by its own engine and cannot run on the
                    // generic decode loop, so report what the checkpoint actually looks
                    // like at load time: download completeness, tensor coverage, and
                    // where the memory goes. A half-finished download is the normal
                    // state for hours, and this is the only place a user sees it. The
                    // report itself was computed off the main actor above.
                    for line in deepSeekReport ?? [] {
                        print("🔎 [DeepSeekV41] \(line)")
                    }
                    // Do NOT restore a system prompt captured before the load here.
                    // This completion runs asynchronously AFTER switchModel has already
                    // called applyProfile, so writing back the pre-load value clobbered
                    // the active profile's prompt (UI showed Assistant while the run used
                    // the stale coder prompt). Only fill the user default if empty.
                    if self.systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        self.systemPrompt = ModelConfig.getUserDefaultSystemPrompt()
                    }
                    // Assign even when nil: a model with no tokenizer.json must
                    // clear any previous model's tokenizer rather than tokenize new
                    // prompts with the wrong vocabulary. Sends are gated on
                    // `tokenizer != nil`, so this also disables chat for such a
                    // model instead of silently mis-tokenizing.
                    self.tokenizer = tok
                    if tok != nil {
                        // New model's tokenizer: drop the grammar sampler's
                        // vocab-size-keyed token caches, which do not encode identity.
                        GrammarConstrainedSampler.shared.invalidateTokenizerCaches()
                    }
                    // A previous load's failure notice is no longer true.
                    self.clearStatus()
                    self.activeLoadedModelPath = filePath
                    // Publish the installed model to the headless engine (chat UI
                    // and API server share this one runtime snapshot).
                    CompletionEngine.shared.configure(summary: loadedSummary, modelConfig: cfg, tokenizer: tok, shardBuffers: buffers, activeLoadedModelPath: filePath)
                    // The engine is actually installed now; this is what
                    // `ensureModelLoaded` verifies before an async send.
                    self.installedModelPath = filePath
                    self.isLoadingModel = false

                    if isFlashMoE {
                        self.pagingStatusMessage = "⚡ Ultra-Fast Parallel POSIX Pread Streaming Active (12+ tok/s)"
                    } else {
                        let effMode = memoryExecutionMode.resolveEffectiveMode(modelFootprintGB: mappedGB)
                        if effMode == .residentRAM {
                            self.pagingStatusMessage = "⚡ Operating in Full RAM Resident Mode (Zero Disk Paging)"
                        } else {
                            self.pagingStatusMessage = "🌊 Operating in Dynamic SSD Streaming Mode"
                        }
                    }
                    self.updatePagingStats()
                    self.metalStatus = "✅ Zero-Copy Active! \(loadedSummary.shards.count) Shards Mapped (\(String(format: "%.2f", mappedGB)) GB)"
                }
            } catch {
                await MainActor.run {
                    // Only the current load may report its failure; a stale one
                    // must not clobber the engine a newer load installed.
                    guard self.modelLoadToken == token else { return }
                    // The reason belongs where the user can read it: the
                    // sidebar keeps only "Engine Load Error", which is a
                    // symptom without a cause.
                    let reason = error.localizedDescription
                    self.showStatus("❌ Engine load failed: \(reason)")
                    self.recordLoadFailureInTranscript(reason)
                    self.summary = nil
                    self.engine = nil
                    self.shardBuffers.removeAll()
                    // Give the backbone pin back too: no model is installed, so a
                    // retained multi-GB pin would be pure waste until the next load.
                    BackbonePinPool.shared.release()
                    CompletionEngine.shared.configure(summary: nil, modelConfig: nil, tokenizer: nil, shardBuffers: [:], activeLoadedModelPath: nil)
                    // The installed model is being torn down, so drop its shard registry
                    // as well: the residency cadence would otherwise keep probing these
                    // addresses every 2 s after their mmaps are gone — and if the next
                    // load reuses the virtual addresses, mincore would report unrelated
                    // memory as shard residency.
                    WorkingSetManager.shared.releaseShardMappings()
                    // No usable engine is installed; clear the identity so an async
                    // send cannot treat a stale `activeLoadedModelPath` as ready.
                    self.installedModelPath = nil
                    self.isLoadingModel = false
                    self.metalStatus = "❌ Engine Load Error"
                }
            }
        }
        modelLoadTask = loadTask
        return loadTask
    }

    #if os(macOS)
    private func selectTokenizerWithOpenPanel() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.json, .data]
        panel.message = "Select tokenizer.json"
        panel.prompt = "Load Tokenizer"
        
        if panel.runModal() == .OK, let url = panel.url {
            loadTokenizer(filePath: url.path)
        }
    }

    private func selectModelWithOpenPanel() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.json, .data, .folder]
        panel.message = "Select Model Snapshot Folder, model_weights.bin, or model.safetensors.index.json"
        panel.prompt = "Load Model"
        
        if panel.runModal() == .OK, let url = panel.url {
            var targetPath = url.path
            if url.hasDirectoryPath {
                let indexPath = url.appendingPathComponent("model.safetensors.index.json").path
                let flashMoeJson = url.appendingPathComponent("model_weights.json").path
                let flashMoeBin = url.appendingPathComponent("model_weights.bin").path
                
                if FileManager.default.fileExists(atPath: indexPath) {
                    targetPath = indexPath
                } else if FileManager.default.fileExists(atPath: flashMoeJson) {
                    targetPath = flashMoeJson
                } else if FileManager.default.fileExists(atPath: flashMoeBin) {
                    targetPath = flashMoeBin
                }
            }
            loadAndBridgeToMetal(filePath: targetPath)
        }
    }
    #endif
}
