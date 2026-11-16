//
//  SkillsRegistry.swift
//  DynaMoE
//
//  Phase 8 — Skills.
//
//  The live registry: loads `skills.json`, scans the skill folders, watches
//  them for changes, and publishes what the prompt and the `skill_load` tool
//  read. Split from the pure parsing/discovery code in `AgentSkills.swift` so
//  the format can be tested without a filesystem watcher.
//
//  Watching deliberately mirrors the MCP subsystem: a change that lands while
//  a turn is running is DEFERRED instead of applied, because the system prompt
//  is fixed for the turn and swapping skills underneath it would silently
//  rewrite what the model was told it could do. When the prompt section
//  actually changes, the pinned KV prefix is invalidated — on a hybrid/SSM
//  checkpoint a changed system block means a full re-prefill, so the check is
//  "did the rendered section change", not "did any file change".
//

import AppKit
import Combine
import Darwin
import Foundation

// MARK: - File Watcher

/// One `DispatchSource` on one path, re-armable. Internal so a test can prove
/// `stop()` closes the descriptor; the registry owns the policy (when to
/// reload, when to defer) and this only reports "something happened here".
final class SkillPathWatcher {
    private let path: String
    private let onChange: () -> Void
    private var source: DispatchSourceFileSystemObject?
    /// The descriptor this watcher opened last, kept after `stop()` so a test
    /// can prove the cancel handler closed it (the fd must report EBADF). The
    /// close itself captures the number BY VALUE: it used to live on `self` and
    /// be closed from a `[weak self]` handler, which had no owner by the time
    /// the cancel ran, so every stopped watcher leaked its descriptor —
    /// re-arming `skills.json` on each external edit leaked one fd per edit.
    private(set) var descriptorForTesting: Int32 = -1

    init(path: String, onChange: @escaping () -> Void) {
        self.path = path
        self.onChange = onChange
    }

    /// Opens the descriptor this watcher needs, WITHOUT arming it.
    ///
    /// `open(O_EVTONLY)` blocks, so the registry calls this from a detached
    /// task and hands the result to `start(descriptor:)` on the MainActor. A
    /// caller that does not care can use `start()`, which does both.
    nonisolated static func openDescriptor(path: String) -> Int32 {
        open(path, O_EVTONLY)
    }

    @discardableResult
    func start() -> Bool {
        start(descriptor: Self.openDescriptor(path: path))
    }

    /// Arms the source against an already-open descriptor. Takes ownership of
    /// it: the cancel handler closes it.
    @discardableResult
    func start(descriptor: Int32) -> Bool {
        guard source == nil else {
            // Already armed; the caller's descriptor has no owner.
            close(descriptor)
            return true
        }
        guard descriptor >= 0 else { return false }
        let source = DispatchSource.makeFileSystemObjectSource(
            fileDescriptor: descriptor,
            eventMask: [.write, .extend, .attrib, .delete, .rename, .revoke, .link],
            queue: .main
        )
        source.setEventHandler { [weak self] in
            self?.onChange()
        }
        source.setCancelHandler {
            close(descriptor)
        }
        self.source = source
        self.descriptorForTesting = descriptor
        source.resume()
        return true
    }

    func stop() {
        source?.cancel()
        source = nil
    }
}

// MARK: - Registry

/// Process-wide skills state. Read from the prompt builder and from
/// `skill_load`; mutated from Settings and by the filesystem watchers.
public final class SkillsRegistry: ObservableObject {
    public static let shared = SkillsRegistry()

    // MARK: Published state

    @Published public private(set) var config = SkillsConfig()
    /// Every discovered skill, including ones the config disables.
    @Published public private(set) var allSkills: [Skill] = []
    @Published public private(set) var issues: [SkillIssue] = []
    /// Name collisions as data, in addition to the readable issues: the
    /// starter installer has to know which copy WON before it can tell the user
    /// which one to remove.
    @Published public private(set) var shadows: [SkillShadow] = []
    @Published public private(set) var scannedRoots: [SkillRoot] = []
    @Published public private(set) var configPath = SkillsConfigStore.defaultPathDescription
    @Published public private(set) var lastReloadSummary: String?
    @Published public private(set) var lastError: String?
    /// Bumped on every successful reload; Settings can show it and a future
    /// caller can tell "the registry moved" from "it did not".
    @Published public private(set) var generation = 0

    // MARK: Derived state

    /// What the prompt and the loader are allowed to see right now. A skill
    /// that opts out of model invocation is not in here: this build has no
    /// other invocation surface, so advertising it would produce exactly the
    /// auto-invocation the author forbade.
    public var enabledSkills: [Skill] {
        guard config.enabled else { return [] }
        return allSkills.filter { config.isEnabled($0) && $0.modelInvocable }
    }

    public var alwaysOnSkills: [Skill] {
        Array(enabledSkills.filter(\.always).prefix(SkillValidation.maximumAlwaysOnSkills))
    }

    public func skill(named name: String) -> Skill? {
        let normalized = SkillValidation.normalizedName(name)
        guard !normalized.isEmpty else { return nil }
        if let direct = allSkills.first(where: { $0.id == normalized }) { return direct }
        // A model may pass the folder name when it differs only by case or a
        // trailing slash; keep the lookup forgiving without making the address
        // ambiguous (dedupe already collapsed case).
        let stripped = normalized.trimmingCharacters(in: CharacterSet(charactersIn: "/ "))
        return allSkills.first { $0.id == stripped || SkillValidation.normalizedName($0.directory.lastPathComponent) == stripped }
    }

    // MARK: Private state

    private var hasBootstrapped = false
    private var configWatcher: SkillPathWatcher?
    private var rootWatchers: [String: SkillPathWatcher] = [:]
    /// One watcher per skill MANIFEST. Watching only the root directories sees
    /// folders appear and disappear but not a `description:` or `always:`
    /// edited in place, which is the most common change of all.
    private var manifestWatchers: [String: SkillPathWatcher] = [:]
    private var pendingReload = false
    private var deferredTask: Task<Void, Never>?
    private var reloadDebounceTask: Task<Void, Never>?
    /// Monotonic ticket so an in-flight background scan cannot publish over a
    /// newer synchronous reload (a settings save can land while a debounced
    /// watcher scan is still running).
    private var reloadTicket = 0
    private var lastRenderedSection: String?
    /// Whether a config has ever been read successfully. It decides what a
    /// FAILED read falls back to: the last good config once there is one, and
    /// a closed switch before there is.
    private var hasLoadedConfig = false

    private var isBusy: Bool {
        #if DEBUG
        if let override = busyOverrideForTesting { return override }
        #endif
        return CompletionEngine.shared.isActive || AgentActivity.shared.isBusy
    }

    private var configURL: URL? {
        #if DEBUG
        configURLForTesting ?? SkillsConfigStore.defaultURL
        #else
        SkillsConfigStore.defaultURL
        #endif
    }

    private init() {}

    // MARK: Lifecycle

    public func bootstrap() {
        guard !hasBootstrapped else { return }
        hasBootstrapped = true
        ensureSkillsDirectory()
        startConfigWatcher()
        // The initial scan reads up to 512 KB per manifest plus a stat and a
        // realpath per entry; doing it in `App.init` blocked launch. The first
        // turn cannot start before a model is loaded, so an immediate
        // background scan is ready well before the prompt is built.
        scheduleExternalReload(reason: "launch", delayNanoseconds: 0)
    }

    /// Creates the app-managed skills folder so a user has an obvious place to
    /// drop a skill, and so a first launch with no skills still has a valid
    /// root to watch.
    private func ensureSkillsDirectory() {
        guard let directory = SkillsConfigStore.skillsDirectoryURL else { return }
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    // MARK: Reload

    /// One load of the config, the roots and the scan, as a value.
    /// `nonisolated` so the watcher path can build it on a background task;
    /// it reads no registry state.
    private nonisolated struct ReloadSnapshot: Sendable {
        var configURL: URL?
        /// The decoded file, before clamping. Nil when the load failed.
        var loadedConfig: SkillsConfig?
        /// Issues from the RAW file, so an out-of-range budget is reported
        /// instead of being silently repaired by `clamped()`.
        var configIssues: [SkillIssue] = []
        var configError: String?
        var roots: [SkillRoot] = []
        var scan: SkillDiscovery.ScanResult = .init()
    }

    private nonisolated static func makeSnapshot(
        configURL: URL?,
        rootsOverride: [SkillRoot]?,
        fallbackConfig: SkillsConfig
    ) -> ReloadSnapshot {
        var snapshot = ReloadSnapshot(configURL: configURL)
        do {
            let raw = try SkillsConfigStore.loadOrSeed(at: configURL)
            snapshot.loadedConfig = raw
            snapshot.configIssues = SkillsConfigValidator.validate(raw)
        } catch {
            snapshot.configError = error.localizedDescription
        }
        let rootsConfig = snapshot.loadedConfig ?? fallbackConfig
        snapshot.roots = rootsOverride ?? SkillsRegistry.resolveRoots(rootsConfig)
        snapshot.scan = SkillDiscovery.scan(roots: snapshot.roots)
        return snapshot
    }

    private var rootOverride: [SkillRoot]? {
        #if DEBUG
        rootOverrideForTesting
        #else
        nil
        #endif
    }

    /// A settings save, the Reload button, a starter install and the deferred
    /// poll all arrive here. The scan reads every root and up to 512 KB per
    /// manifest, so it runs off the MainActor with no debounce delay instead of
    /// freezing the UI for the duration of a large `extraRoot`.
    public func reload(reason: String) {
        scheduleExternalReload(reason: reason, delayNanoseconds: 0)
    }

    /// The synchronous form. Only the deferred poll (which has already waited
    /// for the turn to end) and the test seam use it; every user-visible path
    /// goes through `reload` so the scan stays off the MainActor.
    private func reloadNow(reason: String) {
        // A synchronous reload supersedes any debounced scan: a settings save
        // must not be overwritten by a watcher event that started earlier.
        reloadDebounceTask?.cancel()
        reloadDebounceTask = nil
        reloadTicket += 1
        let snapshot = Self.makeSnapshot(
            configURL: configURL,
            rootsOverride: rootOverride,
            fallbackConfig: config
        )
        apply(snapshot: snapshot, reason: reason, armWatchersSynchronously: true)
    }

    /// Coalesces a burst of filesystem events (a `git checkout` touching N
    /// manifests) into one scan and runs it off the main actor. The section is
    /// still applied on the MainActor, and only when no turn owns the prompt.
    private func scheduleExternalReload(reason: String, delayNanoseconds: UInt64 = 350_000_000) {
        reloadDebounceTask?.cancel()
        reloadTicket += 1
        let ticket = reloadTicket
        let configURL = configURL
        let rootsOverride = rootOverride
        let fallbackConfig = config
        reloadDebounceTask = Task { [weak self] in
            if delayNanoseconds > 0 {
                try? await Task.sleep(nanoseconds: delayNanoseconds)
            }
            guard !Task.isCancelled else { return }
            let snapshot = await Task.detached(priority: .utility) {
                SkillsRegistry.makeSnapshot(
                    configURL: configURL,
                    rootsOverride: rootsOverride,
                    fallbackConfig: fallbackConfig
                )
            }.value
            guard let self, !Task.isCancelled, ticket == self.reloadTicket else { return }
            self.reloadDebounceTask = nil
            self.apply(snapshot: snapshot, reason: reason)
        }
    }

    /// Scans off the MainActor and applies the result before returning. Used by
    /// the one caller whose next statement reads the new state.
    private func reloadAwaitingScan(reason: String) async {
        reloadDebounceTask?.cancel()
        reloadDebounceTask = nil
        reloadTicket += 1
        let ticket = reloadTicket
        let configURL = configURL
        let rootsOverride = rootOverride
        let fallbackConfig = config
        let snapshot = await Task.detached(priority: .utility) {
            SkillsRegistry.makeSnapshot(
                configURL: configURL,
                rootsOverride: rootsOverride,
                fallbackConfig: fallbackConfig
            )
        }.value
        guard ticket == reloadTicket else { return }
        apply(snapshot: snapshot, reason: reason)
    }

    /// `armWatchersSynchronously` exists for the test seams only: production
    /// arms off the MainActor (see `restartWatchers`), and a test that asserts
    /// `manifestWatcherCountForTesting` immediately after a synchronous reload
    /// would otherwise race the detached task.
    private func apply(
        snapshot: ReloadSnapshot,
        reason: String,
        armWatchersSynchronously: Bool = false
    ) {
        guard !isBusy else {
            deferPendingReload(summary: "Change detected; it applies when the current turn finishes.")
            return
        }
        configPath = snapshot.configURL?.path ?? SkillsConfigStore.defaultPathDescription

        var configIssues: [SkillIssue] = []
        if let loaded = snapshot.loadedConfig {
            config = loaded.clamped()
            hasLoadedConfig = true
            lastError = nil
            // Validate the file as written. The previous order clamped first,
            // which made the validator's range errors unreachable on reload.
            configIssues = snapshot.configIssues
        } else if hasLoadedConfig {
            // Keep the last good config: a malformed external edit must not
            // hide every skill while the user fixes the file.
            lastError = snapshot.configError
            DiagLog.error("skills", "config load failed: \(snapshot.configError ?? "unknown error")")
            configIssues = SkillsConfigValidator.validate(config)
        } else {
            // Nothing has ever loaded, so there is no last good config to keep
            // and the default is `enabled: true` with no disabled skills. Fail
            // CLOSED instead: a typo'd file must not silently re-enable the
            // skills the user turned off and restore default budgets. The
            // error is reported as an issue, not only in the Settings banner,
            // because this is the state a user has to be told about.
            lastError = snapshot.configError
            DiagLog.error("skills", "config load failed at launch: \(snapshot.configError ?? "unknown error")")
            config = SkillsConfig(enabled: false)
            configIssues = [SkillIssue(
                severity: .error,
                path: snapshot.configURL?.path,
                message: "could not be read, so skills are switched off until it is fixed. Nothing is advertised or loadable meanwhile."
            )]
        }

        let roots = snapshot.roots
        let scan = snapshot.scan
        scannedRoots = roots
        allSkills = scan.skills
        shadows = scan.shadowed.map { SkillShadow(loser: $0.skill, winner: $0.winner) }

        var issues = configIssues + scan.issues
        for pair in scan.shadowed {
            // The path points at the copy that WINS, because that is the one a
            // reader has to remove or rename for their edit to take effect.
            issues.append(SkillIssue(
                severity: .warning,
                skillName: pair.skill.name,
                path: pair.winner.directory.path,
                message: "is shadowed by \(pair.winner.rootLabel): \(pair.winner.directory.path). The later-scanned folder wins; remove or rename that copy to use \(pair.skill.directory.path)."
            ))
        }
        // Count only what could actually render: a disabled skill's `always`
        // costs nothing, so warning about it pointed at the wrong file.
        let alwaysSkills = enabledSkills.filter(\.always)
        if alwaysSkills.count > SkillValidation.maximumAlwaysOnSkills {
            issues.append(SkillIssue(
                severity: .warning,
                message: "\(alwaysSkills.count) enabled skills declare always: true, but only the first \(SkillValidation.maximumAlwaysOnSkills) by name render an always-invoke rule. Remove 'always' from the ones that should not be mandatory."
            ))
        }
        issues.sort {
            if $0.severity != $1.severity { return $0.severity > $1.severity }
            return $0.displayMessage < $1.displayMessage
        }
        self.issues = issues

        let enabledCount = enabledSkills.count
        lastReloadSummary = "\(roots.count) folder(s) scanned · \(allSkills.count) skill(s) found · \(enabledCount) enabled"

        let section = SkillsPromptRenderer.section(
            skills: enabledSkills,
            maxAdvertised: config.maxAdvertised,
            advertisedCharacters: config.advertisedCharacters
        )
        if section != lastRenderedSection {
            lastRenderedSection = section
            invalidatePrefixIfIdle()
        }

        generation += 1
        DiagLog.info("skills", "reload (\(reason)): \(lastReloadSummary ?? "")")
        restartWatchers(roots: roots, skills: scan.skills, synchronously: armWatchersSynchronously)
        // The launch scan seeds a missing skills.json, so the config watcher
        // can only be armed once that file exists. Every later reload leaves
        // an armed watcher alone.
        if configWatcher == nil {
            startConfigWatcher()
        }
        // The tool's schema is paid on every turn, so it should not be in the
        // prompt while the feature is off. The harness owns the loaded set.
        AgentHarness.shared.applySkillsAvailability(enabled: config.enabled)
    }

    private func deferPendingReload(summary: String) {
        pendingReload = true
        lastReloadSummary = summary
        scheduleDeferredReload()
    }

    /// Invalidates the pinned prefix only when no turn owns the engine. While
    /// one runs, the change is applied to the NEXT turn instead; the prompt is
    /// snapshot at turn start either way.
    private func invalidatePrefixIfIdle() {
        guard !isBusy else { return }
        #if DEBUG
        prefixInvalidationCount += 1
        #endif
        PrefixCacheManager.shared.invalidate()
    }

    public func reloadAfterExternalChange() {
        guard !isBusy else {
            deferPendingReload(summary: "Change detected; it applies when the current turn finishes.")
            return
        }
        scheduleExternalReload(reason: "external edit")
    }

    /// The Settings Reload button. Deferred for the same reason an external
    /// edit is: applying a new skill list while a turn's prompt is frozen
    /// desynchronises `skill_load` from what the model was told.
    public func requestReload() {
        guard !isBusy else {
            deferPendingReload(summary: "Reload queued; it applies when the current turn finishes.")
            return
        }
        reload(reason: "manual reload")
    }

    private func scheduleDeferredReload() {
        guard deferredTask == nil else { return }
        deferredTask = Task { [weak self] in
            while true {
                try? await Task.sleep(nanoseconds: 2_000_000_000)
                guard let self else { return }
                if !self.isBusy {
                    if self.pendingReload {
                        self.pendingReload = false
                        self.reload(reason: "external edit")
                    }

                    self.deferredTask = nil
                    return
                }
            }
        }
    }

    // MARK: Roots

    /// The built-in locations in precedence order, then the user's extra
    /// folders. A later root wins a name collision.
    ///
    /// The app-managed folder deliberately comes AFTER the two shared ones: an
    /// installed starter is an explicit choice by the operator, so it should
    /// take effect even when an older copy of the same skill exists in a shared
    /// folder (which is exactly the migration case for `subq-decisions`).
    /// `extraRoots` stay last so a project can still override the app copy.
    public func resolveRoots(_ config: SkillsConfig) -> [SkillRoot] {
        if let override = rootOverride { return override }
        return Self.resolveRoots(config)
    }

    /// The root list as a pure function of the config, so the background scan
    /// can resolve it without touching registry state.
    public nonisolated static func resolveRoots(_ config: SkillsConfig) -> [SkillRoot] {
        var roots: [SkillRoot] = []
        let home = FileManager.default.homeDirectoryForCurrentUser
        roots.append(SkillRoot(
            url: home.appendingPathComponent(".agents/skills", isDirectory: true),
            label: "~/.agents/skills"
        ))
        roots.append(SkillRoot(
            url: home.appendingPathComponent(".claude/skills", isDirectory: true),
            label: "~/.claude/skills"
        ))
        if let appDirectory = SkillsConfigStore.skillsDirectoryURL {
            roots.append(SkillRoot(url: appDirectory, label: "Application Support"))
        }
        for raw in config.extraRoots {
            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            let expanded = (trimmed as NSString).expandingTildeInPath
            // A relative path would resolve against the process's working
            // directory, which is `/` for the app. Skipping it is what the
            // validator warns about, so config and scanner agree.
            guard expanded.hasPrefix("/") else { continue }
            let url = URL(fileURLWithPath: expanded, isDirectory: true).standardizedFileURL
            guard !roots.contains(where: { $0.url.standardizedFileURL.path == url.path }) else { continue }
            roots.append(SkillRoot(url: url, label: url.path))
        }
        return roots
    }

    // MARK: Settings operations

    public func setEnabled(_ enabled: Bool) {
        mutate(reason: "master toggle") { updated in
            updated.enabled = enabled
            return true
        }
    }

    public func setSkill(_ skill: Skill, enabled: Bool) {
        mutate(reason: enabled ? "enabled \(skill.name)" : "disabled \(skill.name)") { updated in
            var disabled = Set(updated.disabledSkills.map(SkillValidation.normalizedName))
            if enabled {
                disabled.remove(skill.id)
            } else {
                disabled.insert(skill.id)
            }
            updated.disabledSkills = disabled.sorted()
            return true
        }
    }

    public func addRoot(_ path: String) {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let expanded = (trimmed as NSString).expandingTildeInPath
        guard expanded.hasPrefix("/") else {
            lastError = "\(expanded) is not an absolute path. Choose the folder again; a relative path has no working directory to resolve against."
            return
        }
        let standardized = URL(fileURLWithPath: expanded, isDirectory: true).standardizedFileURL.path
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: standardized, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            lastError = "\(standardized) is not a folder."
            return
        }
        mutate(reason: "added root \(standardized)") { updated in
            guard !updated.extraRoots.contains(where: { ($0 as NSString).expandingTildeInPath == expanded }) else {
                return false
            }
            updated.extraRoots.append(standardized)
            return true
        }
    }

    public func removeRoot(_ path: String) {
        mutate(reason: "removed root \(path)") { updated in
            let before = updated.extraRoots.count
            updated.extraRoots.removeAll { raw in
                URL(fileURLWithPath: (raw as NSString).expandingTildeInPath, isDirectory: true)
                    .standardizedFileURL.path == path
            }
            return updated.extraRoots.count != before
        }
    }

    /// Replaces the whole document. This is the Settings FORM, which shows
    /// every field, so it is a deliberate overwrite rather than a partial
    /// edit; `mutate` is for the controls that change one field.
    public func save(_ updated: SkillsConfig) {
        persist(updated, reason: "settings")
    }

    /// Applies a one-field change to the file as it is ON DISK, then saves.
    ///
    /// `config` deliberately lags: a toggle during a turn is written
    /// immediately but not applied until the turn ends, so building the new
    /// document from `config` reverted any external edit made in the meantime
    /// (a hand-edited `extraRoots`, a skill disabled in another window). The
    /// base here is the current file, and `config` is only the fallback when it
    /// cannot be read.
    /// The closure returns `false` to abandon the write (a duplicate root, say),
    /// so a no-op control does not rewrite the file and trigger a reload.
    private func mutate(reason: String, _ change: (inout SkillsConfig) -> Bool) {
        var base = config
        if let url = configURL, let onDisk = try? SkillsConfigStore.load(from: url) {
            base = onDisk
        }
        guard change(&base) else { return }
        persist(base, reason: reason)
    }

    public func rawJSONText() -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        guard let data = try? encoder.encode(config),
              let text = String(data: data, encoding: .utf8) else {
            return SkillsConfigStore.templateJSON
        }
        return text + "\n"
    }

    public func saveRawJSON(_ text: String) throws {
        _ = try SkillsConfigStore.saveRawJSON(text, to: configURL)
        applySaved(reason: "raw JSON edit")
    }

    private func persist(_ updated: SkillsConfig, reason: String) {
        do {
            try SkillsConfigStore.save(updated, to: configURL)
            applySaved(reason: reason)
        } catch {
            lastError = error.localizedDescription
            DiagLog.error("skills", "config save failed: \(error.localizedDescription)")
        }
    }

    /// A settings write lands on disk immediately but is APPLIED through the
    /// same deferral the external watcher uses. Applying it mid-turn while the
    /// prompt is frozen desynchronised `skill_load` from what the model had
    /// been told: a skill could disappear from the loader while its description
    /// was still in the system block. `config` therefore keeps its previous
    /// value until the reload actually runs.
    private func applySaved(reason: String) {
        lastError = nil
        guard !isBusy else {
            deferPendingReload(summary: "Change saved; it applies when the current turn finishes.")
            return
        }
        reload(reason: reason)
    }

    // MARK: Starter skill

    /// Writes the bundled `subq-decisions` starter into the app skills folder.
    ///
    /// Async because the scan runs off the MainActor, and the caller reports
    /// which copy won a name collision from that scan's result; it therefore
    /// awaits the scan instead of returning a state the reload has not reached
    /// yet.
    @discardableResult
    public func installSubQDecisionsStarter() async -> Result<URL, Error> {
        guard let directory = SkillsConfigStore.skillsDirectoryURL else {
            return .failure(SkillsConfigError.applicationSupportUnavailable("the skills folder could not be resolved"))
        }
        do {
            let url = try StarterSkills.installSubQDecisions(in: directory)
            if isBusy {
                deferPendingReload(summary: "Starter installed; it appears when the current turn finishes.")
            } else {
                await reloadAwaitingScan(reason: "installed starter skill")
            }
            return .success(url)
        } catch {
            // "Already installed" is a benign no-op: it is reported by the
            // caller as an install message, but it must not paint the master
            // card red as though the feature were broken.
            if let starterError = error as? StarterSkillError,
               case .alreadyInstalled = starterError {
                lastError = nil
            } else {
                lastError = error.localizedDescription
            }
            return .failure(error)
        }
    }

    // MARK: Config file affordances (Settings)

    public func openConfigInEditor() {
        guard let url = configURL else { return }
        if !FileManager.default.fileExists(atPath: url.path) {
            do {
                try SkillsConfigStore.save(config, to: url)
            } catch {
                lastError = error.localizedDescription
                return
            }
        }
        NSWorkspace.shared.open(url)
    }

    public func revealConfigInFinder() {
        guard let url = configURL else { return }
        if FileManager.default.fileExists(atPath: url.path) {
            NSWorkspace.shared.activateFileViewerSelecting([url])
        } else {
            let directory = url.deletingLastPathComponent()
            try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            NSWorkspace.shared.open(directory)
        }
    }

    public func revealSkillInFinder(_ skill: Skill) {
        NSWorkspace.shared.activateFileViewerSelecting([skill.directory])
    }

    public func revealSkillsRoot() {
        guard let directory = SkillsConfigStore.skillsDirectoryURL else { return }
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        NSWorkspace.shared.open(directory)
    }

    // MARK: Watchers

    private func startConfigWatcher() {
        configWatcher?.stop()
        // A failed re-arm must leave `configWatcher` NIL. Keeping the stopped
        // object meant the `configWatcher == nil` recovery in `apply` could
        // never fire, so one transient `open` failure disabled external-edit
        // detection until the next launch.
        configWatcher = nil
        guard let url = configURL else { return }
        let path = url.path
        // The `open` happens off the MainActor for the same reason the root and
        // manifest watchers do: it is a filesystem call that blocks.
        Task { [weak self] in
            let descriptor = await Task.detached(priority: .utility) {
                SkillPathWatcher.openDescriptor(path: path)
            }.value
            guard let self else {
                if descriptor >= 0 { close(descriptor) }
                return
            }
            guard self.configWatcher == nil else {
                // A reload armed one while this was in flight.
                if descriptor >= 0 { close(descriptor) }
                return
            }
            let watcher = SkillPathWatcher(path: path) { [weak self] in
                guard let self else { return }
                self.reloadAfterExternalChange()
                // An atomic save replaces the inode; re-arm on the new file.
                self.startConfigWatcher()
            }
            if watcher.start(descriptor: descriptor) {
                self.configWatcher = watcher
            } else {
                DiagLog.warning("skills", "could not watch skills.json; external edits need a manual reload")
            }
        }
    }

    /// More watched manifests than this crowd the process's descriptor soft
    /// limit (a `DispatchSource` holds one fd each) and the failure lands on
    /// some unrelated later `open()`. Past the cap, root-directory watching
    /// still notices skills appearing and disappearing; edits inside existing
    /// manifests then need the Reload button.
    private static let maximumManifestWatchers = 128

    /// One watcher to arm, resolved off the MainActor.
    private nonisolated struct WatcherPlan: Sendable {
        enum Kind: Sendable { case root, manifest }
        var path: String
        var kind: Kind
        /// Opened off the MainActor; `start(descriptor:)` takes ownership.
        var descriptor: Int32
    }

    /// Arms the root and manifest watchers for a completed scan.
    ///
    /// `open(O_EVTONLY)` blocks, and this ran on the MainActor for every root
    /// plus up to `maximumManifestWatchers` manifests on each reload: one
    /// `extraRoot` on an unresponsive SMB mount froze the UI for the duration
    /// of the mount timeout. The `stat` and the `open` therefore happen on a
    /// detached task, and only the finished watchers are installed here.
    ///
    /// A watcher that is already armed for an unchanged path is left alone, so
    /// a reload does not churn descriptors.
    private func restartWatchers(
        roots: [SkillRoot],
        skills: [Skill],
        synchronously: Bool = false
    ) {
        var rootPaths: [String] = []
        var seenRoots = Set<String>()
        for root in roots.sorted(by: { $0.url.path < $1.url.path }) where seenRoots.insert(root.url.path).inserted {
            rootPaths.append(root.url.path)
        }
        var manifestPaths: [String] = []
        var seenManifests = Set<String>()
        for skill in skills.sorted(by: { $0.skillFile.path < $1.skillFile.path }) {
            let path = skill.skillFile.path
            guard seenManifests.insert(path).inserted else { continue }
            manifestPaths.append(path)
        }
        if manifestPaths.count > Self.maximumManifestWatchers {
            DiagLog.warning(
                "skills",
                "not watching \(manifestPaths.count) manifests individually (cap \(Self.maximumManifestWatchers)); edits inside them need a manual reload"
            )
            manifestPaths = Array(manifestPaths.prefix(Self.maximumManifestWatchers))
        }

        let wantedRoots = Set(rootPaths)
        let wantedManifests = Set(manifestPaths)
        // Stop what is no longer wanted FIRST, on the MainActor: the cancel
        // handler closes the descriptor and must not race the arm below.
        for (path, watcher) in rootWatchers where !wantedRoots.contains(path) {
            watcher.stop()
            rootWatchers.removeValue(forKey: path)
        }
        for (path, watcher) in manifestWatchers where !wantedManifests.contains(path) {
            watcher.stop()
            manifestWatchers.removeValue(forKey: path)
        }
        let armedRoots = Set(rootWatchers.keys)
        let armedManifests = Set(manifestWatchers.keys)
        let rootsToArm = rootPaths.filter { !armedRoots.contains($0) }
        let manifestsToArm = manifestPaths.filter { !armedManifests.contains($0) }
        guard !rootsToArm.isEmpty || !manifestsToArm.isEmpty else { return }

        if synchronously {
            let plans = Self.resolveWatcherPlans(roots: rootsToArm, manifests: manifestsToArm)
            install(plans: plans)
            return
        }
        Task { [weak self] in
            let plans = await Task.detached(priority: .utility) {
                Self.resolveWatcherPlans(roots: rootsToArm, manifests: manifestsToArm)
            }.value
            guard let self else {
                for plan in plans where plan.descriptor >= 0 { close(plan.descriptor) }
                return
            }
            self.install(plans: plans)
        }
    }

    private func install(plans: [WatcherPlan]) {
        for plan in plans {
            switch plan.kind {
            case .root: armRootWatcher(path: plan.path, descriptor: plan.descriptor)
            case .manifest: armManifestWatcher(path: plan.path, descriptor: plan.descriptor)
            }
        }
    }

    /// `stat` + `open` for each candidate, off the MainActor. A path that does
    /// not exist yet is dropped: the parent's watcher reports it appearing.
    private nonisolated static func resolveWatcherPlans(
        roots: [String],
        manifests: [String]
    ) -> [WatcherPlan] {
        var plans: [WatcherPlan] = []
        for path in roots {
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory),
                  isDirectory.boolValue else { continue }
            plans.append(WatcherPlan(
                path: path,
                kind: .root,
                descriptor: SkillPathWatcher.openDescriptor(path: path)
            ))
        }
        for path in manifests {
            guard FileManager.default.fileExists(atPath: path) else { continue }
            plans.append(WatcherPlan(
                path: path,
                kind: .manifest,
                descriptor: SkillPathWatcher.openDescriptor(path: path)
            ))
        }
        return plans
    }

    private func armRootWatcher(path: String, descriptor: Int32) {
        guard rootWatchers[path] == nil else {
            if descriptor >= 0 { close(descriptor) }
            return
        }
        let watcher = SkillPathWatcher(path: path) { [weak self] in
            guard let self else { return }
            // A directory replace (a `git checkout`, an installer replacing the
            // folder) kills this source under an unchanged path. Forget it
            // FIRST so the reload below can create one against the new inode,
            // the same stop-and-recreate the manifest watcher does.
            if let existing = self.rootWatchers.removeValue(forKey: path) {
                existing.stop()
            }
            self.reloadAfterExternalChange()
        }
        if watcher.start(descriptor: descriptor) {
            rootWatchers[path] = watcher
        } else {
            DiagLog.warning("skills", "could not watch skill root \(path); additions there need a manual reload")
        }
    }

    private func armManifestWatcher(path: String, descriptor: Int32) {
        guard manifestWatchers[path] == nil else {
            if descriptor >= 0 { close(descriptor) }
            return
        }
        let watcher = SkillPathWatcher(path: path) { [weak self] in
            guard let self else { return }
            // An atomic save (or an editor's write-and-rename) replaces the
            // inode: this source is dead after the `.delete`/`.rename` it
            // reports, and the path string is unchanged, so a plain
            // "create if absent" could never replace it. Forget this watcher
            // FIRST; the reload below re-creates it against the new file.
            if let existing = self.manifestWatchers.removeValue(forKey: path) {
                existing.stop()
            }
            self.reloadAfterExternalChange()
        }
        if watcher.start(descriptor: descriptor) {
            manifestWatchers[path] = watcher
        } else {
            DiagLog.warning("skills", "could not watch \(path); in-place edits to it need a manual reload")
        }
    }

    // MARK: Testing seams

    #if DEBUG
    /// Replaces the real `skills.json` location and root list so a test can
    /// exercise reload/watchers without touching the developer's home folder.
    var configURLForTesting: URL?
    var rootOverrideForTesting: [SkillRoot]?
    /// Overrides the engine/agent "a turn is running" answer so deferral is
    /// testable with no model and no server.
    var busyOverrideForTesting: Bool?
    /// Counts prefix invalidations, i.e. the "the rendered block changed"
    /// decision, without touching KV state.
    public private(set) var prefixInvalidationCount = 0
    public var isReloadPendingForTesting: Bool { pendingReload }
    public var manifestWatcherCountForTesting: Int { manifestWatchers.count }

    public func setTestingOverrides(configURL: URL?, roots: [SkillRoot]?) {
        configURLForTesting = configURL
        rootOverrideForTesting = roots
    }

    public func setBusyOverrideForTesting(_ busy: Bool?) {
        busyOverrideForTesting = busy
    }

    /// Applies a deferred reload now instead of after the poll interval.
    public func flushDeferredReloadForTesting() {
        deferredTask?.cancel()
        deferredTask = nil
        reloadDebounceTask?.cancel()
        reloadDebounceTask = nil
        guard pendingReload else { return }
        pendingReload = false
        reloadNow(reason: "external edit")
    }

    /// Applies a reload synchronously, for a test that asserts immediately
    /// afterwards. Production never calls this: a settings save, the Reload
    /// button, a starter install and a watcher event all scan off the
    /// MainActor through `reload`.
    public func reloadForTesting(reason: String) {
        reloadNow(reason: reason)
    }

    /// Installs a registry state without touching the user's real folders.
    /// Registry watchers are not started, so a test cannot trip over the
    /// developer's `~/.claude/skills` mid-run.
    public func installForTesting(skills: [Skill], config: SkillsConfig) {
        hasBootstrapped = true
        self.config = config
        self.allSkills = skills
        self.shadows = []
        self.scannedRoots = []
        self.issues = SkillsConfigValidator.validate(config)
        self.lastError = nil
        self.lastReloadSummary = "installed for testing"
        self.generation += 1
    }

    public func resetForTesting() {
        hasBootstrapped = false
        config = SkillsConfig()
        allSkills = []
        shadows = []
        issues = []
        scannedRoots = []
        lastError = nil
        lastReloadSummary = nil
        lastRenderedSection = nil
        generation = 0
        pendingReload = false
        deferredTask?.cancel()
        deferredTask = nil
        reloadDebounceTask?.cancel()
        reloadDebounceTask = nil
        reloadTicket = 0
        configURLForTesting = nil
        rootOverrideForTesting = nil
        busyOverrideForTesting = nil
        prefixInvalidationCount = 0
        hasLoadedConfig = false
        for (_, watcher) in rootWatchers { watcher.stop() }
        rootWatchers.removeAll()
        for (_, watcher) in manifestWatchers { watcher.stop() }
        manifestWatchers.removeAll()
        configWatcher?.stop()
        configWatcher = nil
    }
    #endif
}
