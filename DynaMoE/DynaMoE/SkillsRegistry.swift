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

/// One `DispatchSource` on one path, re-armable. Kept private: the registry
/// owns the policy (when to reload, when to defer) and this only reports
/// "something happened at this path".
private final class SkillPathWatcher {
    private let path: String
    private let onChange: () -> Void
    private var source: DispatchSourceFileSystemObject?
    private var descriptor: Int32 = -1

    init(path: String, onChange: @escaping () -> Void) {
        self.path = path
        self.onChange = onChange
    }

    @discardableResult
    func start() -> Bool {
        guard source == nil else { return true }
        let descriptor = open(path, O_EVTONLY)
        guard descriptor >= 0 else { return false }
        let source = DispatchSource.makeFileSystemObjectSource(
            fileDescriptor: descriptor,
            eventMask: [.write, .extend, .attrib, .delete, .rename, .revoke, .link],
            queue: .main
        )
        source.setEventHandler { [weak self] in
            self?.onChange()
        }
        source.setCancelHandler { [weak self] in
            guard let self else { return }
            if self.descriptor >= 0 {
                close(self.descriptor)
                self.descriptor = -1
            }
        }
        self.source = source
        self.descriptor = descriptor
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
    @Published public private(set) var scannedRoots: [SkillRoot] = []
    @Published public private(set) var configPath = SkillsConfigStore.defaultPathDescription
    @Published public private(set) var lastReloadSummary: String?
    @Published public private(set) var lastError: String?
    /// Bumped on every successful reload; Settings can show it and a future
    /// caller can tell "the registry moved" from "it did not".
    @Published public private(set) var generation = 0

    // MARK: Derived state

    /// What the prompt and the loader are allowed to see right now.
    public var enabledSkills: [Skill] {
        guard config.enabled else { return [] }
        return allSkills.filter { config.isEnabled($0) }
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
    private var pendingReload = false
    private var deferredTask: Task<Void, Never>?
    private var lastRenderedSection: String?

    private var isBusy: Bool {
        CompletionEngine.shared.isActive || AgentActivity.shared.isBusy
    }

    private init() {}

    // MARK: Lifecycle

    public func bootstrap() {
        guard !hasBootstrapped else { return }
        hasBootstrapped = true
        ensureSkillsDirectory()
        reload(reason: "launch")
        startConfigWatcher()
    }

    /// Creates the app-managed skills folder so a user has an obvious place to
    /// drop a skill, and so a first launch with no skills still has a valid
    /// root to watch.
    private func ensureSkillsDirectory() {
        guard let directory = SkillsConfigStore.skillsDirectoryURL else { return }
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    // MARK: Reload

    public func reload(reason: String) {
        configPath = SkillsConfigStore.defaultPathDescription

        var configIssues: [SkillIssue] = []
        do {
            config = try SkillsConfigStore.loadOrSeed()
            lastError = nil
        } catch {
            // Keep the last good config: a malformed external edit must not
            // hide every skill while the user fixes the file.
            lastError = error.localizedDescription
            DiagLog.error("skills", "config load failed: \(error.localizedDescription)")
        }
        configIssues = SkillsConfigValidator.validate(config)

        let roots = resolveRoots(config)
        let scan = SkillDiscovery.scan(roots: roots)
        scannedRoots = roots
        allSkills = scan.skills

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
        let alwaysSkills = allSkills.filter(\.always)
        if alwaysSkills.count > SkillValidation.maximumAlwaysOnSkills {
            issues.append(SkillIssue(
                severity: .warning,
                message: "\(alwaysSkills.count) skills declare always: true, but only the first \(SkillValidation.maximumAlwaysOnSkills) by name render an always-invoke rule. Remove 'always' from the ones that should not be mandatory."
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
        restartRootWatchers(roots: roots)
    }

    /// Invalidates the pinned prefix only when no turn owns the engine. While
    /// one runs, the change is applied to the NEXT turn instead; the prompt is
    /// snapshot at turn start either way.
    private func invalidatePrefixIfIdle() {
        guard !isBusy else { return }
        PrefixCacheManager.shared.invalidate()
    }

    public func reloadAfterExternalChange() {
        guard !isBusy else {
            pendingReload = true
            scheduleDeferredReload()
            return
        }
        reload(reason: "external edit")
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
            let url = URL(fileURLWithPath: expanded, isDirectory: true).standardizedFileURL
            guard !roots.contains(where: { $0.url.standardizedFileURL.path == url.path }) else { continue }
            roots.append(SkillRoot(url: url, label: url.path))
        }
        return roots
    }

    // MARK: Settings operations

    public func setEnabled(_ enabled: Bool) {
        var updated = config
        updated.enabled = enabled
        persist(updated, reason: "master toggle")
    }

    public func setSkill(_ skill: Skill, enabled: Bool) {
        var updated = config
        var disabled = Set(updated.disabledSkills.map(SkillValidation.normalizedName))
        if enabled {
            disabled.remove(skill.id)
        } else {
            disabled.insert(skill.id)
        }
        updated.disabledSkills = disabled.sorted()
        persist(updated, reason: enabled ? "enabled \(skill.name)" : "disabled \(skill.name)")
    }

    public func addRoot(_ path: String) {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let expanded = (trimmed as NSString).expandingTildeInPath
        let standardized = URL(fileURLWithPath: expanded, isDirectory: true).standardizedFileURL.path
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: standardized, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            lastError = "\(standardized) is not a folder."
            return
        }
        var updated = config
        guard !updated.extraRoots.contains(where: { ($0 as NSString).expandingTildeInPath == expanded }) else { return }
        updated.extraRoots.append(standardized)
        persist(updated, reason: "added root \(standardized)")
    }

    public func removeRoot(_ path: String) {
        var updated = config
        updated.extraRoots.removeAll { raw in
            URL(fileURLWithPath: (raw as NSString).expandingTildeInPath, isDirectory: true)
                .standardizedFileURL.path == path
        }
        persist(updated, reason: "removed root \(path)")
    }

    public func save(_ updated: SkillsConfig) {
        persist(updated, reason: "settings")
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
        let updated = try SkillsConfigStore.saveRawJSON(text)
        config = updated
        reload(reason: "raw JSON edit")
    }

    private func persist(_ updated: SkillsConfig, reason: String) {
        do {
            try SkillsConfigStore.save(updated)
            config = updated
            lastError = nil
            reload(reason: reason)
        } catch {
            lastError = error.localizedDescription
            DiagLog.error("skills", "config save failed: \(error.localizedDescription)")
        }
    }

    // MARK: Starter skill

    /// Writes the bundled `subq-decisions` starter into the app skills folder.
    @discardableResult
    public func installSubQDecisionsStarter() -> Result<URL, Error> {
        guard let directory = SkillsConfigStore.skillsDirectoryURL else {
            return .failure(SkillsConfigError.applicationSupportUnavailable("the skills folder could not be resolved"))
        }
        do {
            let url = try StarterSkills.installSubQDecisions(in: directory)
            reload(reason: "installed starter skill")
            return .success(url)
        } catch {
            lastError = error.localizedDescription
            return .failure(error)
        }
    }

    // MARK: Config file affordances (Settings)

    public func openConfigInEditor() {
        guard let url = SkillsConfigStore.defaultURL else { return }
        if !FileManager.default.fileExists(atPath: url.path) {
            do {
                try SkillsConfigStore.save(config)
            } catch {
                lastError = error.localizedDescription
                return
            }
        }
        NSWorkspace.shared.open(url)
    }

    public func revealConfigInFinder() {
        guard let url = SkillsConfigStore.defaultURL else { return }
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
        guard let url = SkillsConfigStore.defaultURL else { return }
        let watcher = SkillPathWatcher(path: url.path) { [weak self] in
            guard let self else { return }
            self.reloadAfterExternalChange()
            // An atomic save replaces the inode; re-arm on the new file.
            self.startConfigWatcher()
        }
        if watcher.start() {
            configWatcher = watcher
        } else {
            DiagLog.warning("skills", "could not watch skills.json; external edits need a manual reload")
        }
    }

    private func restartRootWatchers(roots: [SkillRoot]) {
        var wanted: [String: SkillRoot] = [:]
        for root in roots { wanted[root.url.path] = root }
        for (path, watcher) in rootWatchers where wanted[path] == nil {
            watcher.stop()
            rootWatchers.removeValue(forKey: path)
        }
        for (path, _) in wanted where rootWatchers[path] == nil {
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory),
                  isDirectory.boolValue else { continue }
            let watcher = SkillPathWatcher(path: path) { [weak self] in
                self?.reloadAfterExternalChange()
            }
            if watcher.start() {
                rootWatchers[path] = watcher
            }
        }
    }

    // MARK: Testing seams

    #if DEBUG
    /// Installs a registry state without touching the user's real folders.
    /// Registry watchers are not started, so a test cannot trip over the
    /// developer's `~/.claude/skills` mid-run.
    public func installForTesting(skills: [Skill], config: SkillsConfig) {
        hasBootstrapped = true
        self.config = config
        self.allSkills = skills
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
        issues = []
        scannedRoots = []
        lastError = nil
        lastReloadSummary = nil
        lastRenderedSection = nil
    }
    #endif
}
