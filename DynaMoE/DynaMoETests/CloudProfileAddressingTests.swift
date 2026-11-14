//
//  CloudProfileAddressingTests.swift
//  DynaMoETests
//
//  Which generation profile a CLOUD model resolves to, and which key a cloud
//  save lands on.
//
//  Every test here builds its own `ModelProfileManager` over a scratch
//  `UserDefaults` suite. That is not stylistic: `ModelProfileManager.shared`
//  persists to the app's own `DRP.DynaMoE` domain, which is shared with this
//  test host, so a probe that wrote through `shared` rewrote the developer's
//  saved profiles with no way to tell those writes apart afterwards. A suite per
//  test means a write path can be exercised at all.
//
//  The provider used throughout is the BUILT-IN `synthetic`, never a custom one
//  registered into the scratch store: `CloudModelAddress` resolves against
//  `CloudProviderRegistry`, which reads the process-wide runtime snapshot. A
//  custom provider saved into a scratch suite would be invisible to it, the
//  address would fail to parse, and these assertions would then pass through the
//  unresolvable-address fallback instead of the gate they exist to pin.
//

import XCTest
@testable import DynaMoE

@MainActor
final class CloudProfileAddressingTests: XCTestCase {

    /// A manager over its own domain, plus the suite name to tear down.
    ///
    /// Seeded, because `init` runs `seedOfficialProfiles`: the seeded LOCAL
    /// qwen/ornith/nanbeige keys are exactly what a cloud address must not
    /// inherit, so a store without them would pass for the wrong reason.
    private func scratchManager() -> (ModelProfileManager, UserDefaults, String) {
        let suite = "CloudProfileAddressingTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defaults.removePersistentDomain(forName: suite)
        return (ModelProfileManager(defaults: defaults), defaults, suite)
    }

    // MARK: - Blocking: the family gate must cover BOTH address forms

    /// The blocking finding this file exists for.
    ///
    /// The gate skipped a cloud address only for the `cloud-model-…` registry id
    /// form. The app's ACTUAL form is the `cloud://<provider>/<model>` sentinel:
    /// `ContentView.applyProfile` is handed `activeLoadedModelPath`, which is the
    /// sentinel, after every cloud activation. `normalizeKey` collapsed that
    /// sentinel to its last path component BEFORE the gate was consulted, so
    /// `cloud://synthetic/qwen` became `qwen` — the key of the seeded local Qwen
    /// profile — and step 1 answered with the local checkpoint's 8,192-token
    /// budget and its `presencePenalty 1.50`.
    ///
    /// A 262,144-token budget or a 1.50 presence penalty on a remote id is the
    /// exact defect the change exists to remove, so each address is checked for
    /// the generic sampling pair as well as the generic budget.
    func testEveryCloudAddressFormResolvesToTheGenericProfile() {
        let (manager, defaults, suite) = scratchManager()
        defer { defaults.removePersistentDomain(forName: suite) }

        let genericCoder = GenerationProfileSettings.defaultFor(type: .coder)
        let genericAssistant = GenerationProfileSettings.defaultFor(type: .assistant)

        // Each pair is (address, why it is interesting).
        let addresses: [(String, String)] = [
            ("cloud-model-synthetic-qwen3-coder", "registry id, family-named upstream"),
            ("cloud://synthetic/qwen3-coder", "sentinel form, family-named upstream"),
            ("cloud://synthetic/qwen", "sentinel form ending in a raw family name"),
            ("cloud://synthetic/ornith-1.5-9b-optiq-4bit", "sentinel form matching a SEEDED key"),
            ("cloud://synthetic/nanbeige4.3-33b", "the 262,144-token local profile"),
            ("cloud://synthetic/hf:deepseek-ai/DeepSeek-V4.1-Flash-1017", "provider-suffixed id"),
            ("cloud://synthetic/hf:Qwen/Qwen3.8-Flash-Next-FP8", "slash-separated upstream id"),
            ("cloud-model-synthetic-ornith-1.5-9b", "registry id, family-named upstream")
        ]

        for (address, why) in addresses {
            let assistant = manager.getProfile(for: address, type: .assistant)
            XCTAssertEqual(assistant.maxNewTokens, genericAssistant.maxNewTokens,
                           "\(address) (\(why)) must not inherit a local budget")
            XCTAssertEqual(assistant.presencePenalty, genericAssistant.presencePenalty, accuracy: 0.0001,
                           "\(address) (\(why)) must not inherit a local sampling pair")
            XCTAssertEqual(assistant.temperature, genericAssistant.temperature, accuracy: 0.0001,
                           "\(address) (\(why)) must not inherit a local sampling pair")

            let coder = manager.getProfile(for: address, type: .coder)
            XCTAssertEqual(coder.maxNewTokens, genericCoder.maxNewTokens,
                           "\(address) (\(why)) must not inherit a local coder budget")
            XCTAssertEqual(coder.temperature, genericCoder.temperature, accuracy: 0.0001,
                           "\(address) (\(why)) must not inherit a local coder sampling pair")
        }
    }

    /// A cloud address whose provider no longer resolves must not degrade to the
    /// local family name it happens to contain. That was the same collapse by a
    /// different route: `normalizeKey` keyed it by its last path component, so a
    /// retired provider's model inherited whatever local profile shared its name.
    func testAnUnresolvableCloudAddressStillSkipsTheFamilyGate() {
        let (manager, defaults, suite) = scratchManager()
        defer { defaults.removePersistentDomain(forName: suite) }

        let generic = GenerationProfileSettings.defaultFor(type: .assistant)
        let orphan = manager.getProfile(for: "cloud://retired-provider/qwen", type: .assistant)
        XCTAssertEqual(orphan.maxNewTokens, generic.maxNewTokens)
        XCTAssertEqual(orphan.presencePenalty, generic.presencePenalty, accuracy: 0.0001)

        // It also keeps its own key rather than collapsing onto `qwen`.
        XCTAssertNotEqual(manager.normalizeKey("cloud://retired-provider/qwen"), "qwen")
        XCTAssertEqual(manager.normalizeKey("cloud://retired-provider/qwen"), "cloud://retired-provider/qwen")
    }

    /// The two forms of ONE model must share a key, or a save through one and a
    /// read through the other disagree — which is how a cloud read was answered
    /// by a local profile in the first place.
    func testBothAddressFormsCollapseToOneProfileKey() {
        let (manager, defaults, suite) = scratchManager()
        defer { defaults.removePersistentDomain(forName: suite) }

        let byId = manager.normalizeKey("cloud-model-synthetic-qwen3-coder")
        let byPath = manager.normalizeKey("cloud://synthetic/qwen3-coder")
        XCTAssertEqual(byId, byPath, "one model cannot have two profile keys")
        XCTAssertEqual(byId, "cloud-model-synthetic-qwen3-coder")
        // Case folding is the rest of the manager's convention, so a catalog that
        // spells an id differently in two places still resolves to one profile.
        XCTAssertEqual(manager.normalizeKey("cloud://synthetic/Qwen3-Coder"), byId)
    }

    /// The second symptom of the same root cause, and the one that damaged a
    /// developer's saved profile: Settings ▸ Generation ▸ Save while a cloud
    /// model is active writes `activeModelPath` — the sentinel — so the write
    /// landed on the LOCAL model's key.
    ///
    /// A cloud save must leave every local profile byte-identical.
    func testACloudSaveNeverTouchesALocalProfile() {
        let (manager, defaults, suite) = scratchManager()
        defer { defaults.removePersistentDomain(forName: suite) }

        // A local checkpoint that shares its short name with the upstream id, so
        // the collision is real rather than hypothetical.
        let localKey = "qwen/qwen3.8-flash-next-fp8"
        let localBefore = manager.getProfile(for: localKey, type: .assistant)

        var cloudDraft = GenerationProfileSettings.defaultFor(type: .assistant)
        cloudDraft.maxNewTokens = 12_345
        cloudDraft.temperature = 0.11
        cloudDraft.presencePenalty = 0.00
        manager.saveProfile(for: "cloud://synthetic/qwen3.8-flash-next-fp8", type: .assistant, settings: cloudDraft)

        // The local profile is untouched — including its budget and its
        // presencePenalty 1.50, the two values the collision corrupted.
        let localAfter = manager.getProfile(for: localKey, type: .assistant)
        XCTAssertEqual(localAfter.maxNewTokens, 4_096, "a cloud save must not rewrite a local profile")
        XCTAssertEqual(localAfter.presencePenalty, 1.50, accuracy: 0.0001)
        XCTAssertEqual(localAfter, localBefore)

        // And the cloud profile round-trips through the OTHER form, which is the
        // one `getProfile` is called with at activation time.
        let roundTripped = manager.getProfile(for: "cloud-model-synthetic-qwen3.8-flash-next-fp8", type: .assistant)
        XCTAssertEqual(roundTripped.maxNewTokens, 12_345, "a cloud save must be readable by both address forms")
        XCTAssertEqual(roundTripped.temperature, 0.11, accuracy: 0.0001)
    }

    /// A local checkpoint still gets its own family profile: the skip is keyed
    /// on the ADDRESS, so a shorter test that only checked cloud addresses could
    /// pass by breaking this.
    func testLocalCheckpointsKeepTheirOfficialProfiles() {
        let (manager, defaults, suite) = scratchManager()
        defer { defaults.removePersistentDomain(forName: suite) }

        let localQwen = manager.getProfile(for: "qwen/qwen3.8-flash-next-fp8", type: .assistant)
        XCTAssertEqual(localQwen.maxNewTokens, 4_096)
        XCTAssertEqual(localQwen.presencePenalty, 1.50, accuracy: 0.0001)

        let localNanbeige = manager.getProfile(for: "nanbeige/nanbeige4.3-33b", type: .assistant)
        XCTAssertEqual(localNanbeige.maxNewTokens, 262_144,
                       "the local Nanbeige profile is real capacity, not a leaked ceiling")

        let localOrnith = manager.getProfile(for: "ornith-ai/Ornith-1.5-35B-A3B-FP8", type: .assistant)
        XCTAssertEqual(localOrnith.maxNewTokens, 4_096, "an unseeded local sibling still family-matches")
    }

    /// Resetting a cloud profile removes it rather than storing the generic
    /// values under a cloud key, so a later default change is visible.
    func testResettingACloudProfileFallsBackToTheGenericDefault() {
        let (manager, defaults, suite) = scratchManager()
        defer { defaults.removePersistentDomain(forName: suite) }

        let address = "cloud-model-synthetic-qwen3-coder"
        var custom = GenerationProfileSettings.defaultFor(type: .coder)
        custom.maxNewTokens = 2_048
        manager.saveProfile(for: address, type: .coder, settings: custom)
        XCTAssertEqual(manager.getProfile(for: address, type: .coder).maxNewTokens, 2_048)

        manager.resetProfile(for: address, type: .coder)
        let reset = manager.getProfile(for: address, type: .coder)
        XCTAssertEqual(reset.maxNewTokens, GenerationProfileSettings.defaultFor(type: .coder).maxNewTokens)
        XCTAssertNil(manager.profilesByModel[manager.normalizeKey(address)],
                     "a reset cloud profile holds no stored settings at all")
    }

    // MARK: - The generic profile and the ceiling it must fit under

    /// Pinned together with `unknownWindowReplyCeiling`, and read FROM it: an
    /// unrecorded-window turn is clamped to that ceiling, so a generic default
    /// above it would be silently lowered on every send and the setting would
    /// look ignored.
    ///
    /// The two numbers are not restated here — the point of the pairing is that
    /// the profile reads the constant, so a test that hardcoded both would keep
    /// passing if someone unlinked them.
    func testTheGenericProfileFitsUnderTheUnsetWindowCeiling() {
        let ceiling = CloudContextPolicy.unknownWindowReplyCeiling
        let coder = GenerationProfileSettings.defaultFor(type: .coder)
        let assistant = GenerationProfileSettings.defaultFor(type: .assistant)

        XCTAssertEqual(coder.maxNewTokens, ceiling,
                       "the cloud coder budget IS the unrecorded-window ceiling")
        XCTAssertEqual(assistant.maxNewTokens, ceiling / 2,
                       "the 2:1 coder:assistant ratio the shipped profiles keep")
        XCTAssertEqual(
            CloudContextPolicy.replyBudget(requested: coder.maxNewTokens, window: nil),
            coder.maxNewTokens,
            "the default must not be lowered by the unset-window ceiling"
        )
        XCTAssertEqual(
            CloudContextPolicy.maximumOutputTokens(window: nil),
            coder.maxNewTokens,
            "the slider and the cloud default are the same number"
        )
    }

    // MARK: - The window writer is a true no-op when nothing changes

    /// `setOverride`'s return answers "does the store hold the requested value",
    /// so a clear of a model that was never there reports false and writes
    /// nothing — the previous `true` claimed a save that had not happened.
    func testClearingAnAbsentWindowReportsNoChangeAndWritesNothing() {
        let suite = "CloudProfileAddressingTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }

        XCTAssertFalse(
            CloudContextPolicy.setOverride(model: "absent", window: nil, providerId: "p", defaults: defaults)
        )
        XCTAssertNil(defaults.string(forKey: CloudContextPolicy.defaultsKey(for: "p")),
                     "a no-op clear must not create an empty key")

        // A save of the value already stored is also a no-op, reported as the
        // success it is: the store does hold the requested value.
        XCTAssertTrue(
            CloudContextPolicy.setOverride(model: "m", window: 128_000, providerId: "p", defaults: defaults)
        )
        let stored = defaults.string(forKey: CloudContextPolicy.defaultsKey(for: "p"))
        XCTAssertTrue(
            CloudContextPolicy.setOverride(model: "m", window: 128_000, providerId: "p", defaults: defaults)
        )
        XCTAssertEqual(defaults.string(forKey: CloudContextPolicy.defaultsKey(for: "p")), stored)

        // And a real clear afterwards does change the store.
        XCTAssertTrue(
            CloudContextPolicy.setOverride(model: "m", window: nil, providerId: "p", defaults: defaults)
        )
        XCTAssertNil(defaults.string(forKey: CloudContextPolicy.defaultsKey(for: "p")))
    }

    /// The presets feed `ForEach(_:id: \.self)`, so they must stay unique, never
    /// exceed the ceiling, and never offer a button below the slider's own floor
    /// — which would set a value the control immediately clamps back up.
    func testMaxTokenPresetsRespectTheSliderFloor() {
        for ceiling in [16_384, 32_768, 40_000, 262_144] {
            let presets = maxTokenPresets(upTo: ceiling)
            XCTAssertEqual(presets, Array(Set(presets)).sorted(), "unique and ascending at \(ceiling)")
            XCTAssertTrue(presets.allSatisfy { $0 <= ceiling }, "no preset above the ceiling at \(ceiling)")
            XCTAssertEqual(presets.filter { $0 == ceiling }.count, 1, "the ceiling is offered once at \(ceiling)")
            XCTAssertTrue(presets.allSatisfy { $0 >= maxTokenSliderFloor }, "no preset below the floor")
        }

        // A ceiling under the floor is a backend that does not exist today. It
        // must still produce a usable control rather than a button below the
        // slider's own lower bound or a crashing range.
        let tiny = maxTokenPresets(upTo: 20)
        XCTAssertEqual(tiny, [maxTokenSliderFloor])
        let range = maxTokenSliderRange(upTo: 20)
        XCTAssertEqual(range.lowerBound, Double(maxTokenSliderFloor))
        XCTAssertGreaterThanOrEqual(range.upperBound, range.lowerBound)
    }

    // MARK: - Profile labels and fallback ceilings

    func testCloudAddressesDoNotUseLocalFamilyDisplayNamesOrDescriptions() {
        for cloudAddress in [
            "cloud://synthetic/qwen3-coder",
            "cloud-model-synthetic-qwen3-coder",
            "cloud://synthetic/ornith-1.5-9b",
            "cloud-model-synthetic-ornith-1.5-9b"
        ] {
            for profile in ModelProfileType.allCases {
                XCTAssertEqual(
                    profile.profileDisplayName(for: cloudAddress),
                    profile.rawValue,
                    "\(cloudAddress) must show the generic profile title"
                )
                XCTAssertEqual(
                    profile.description(for: cloudAddress),
                    profile.description,
                    "\(cloudAddress) must show the generic profile description"
                )
            }
        }

        // Local checkpoints keep their family labels and descriptions.
        XCTAssertEqual(
            ModelProfileType.coder.profileDisplayName(for: "qwen/qwen3.8-flash-next-fp8"),
            "Coding & Agentic (Thinking Mode)"
        )
        XCTAssertEqual(
            ModelProfileType.coder.profileDisplayName(for: "ornith-ai/Ornith-1.5-35B-A3B-FP8"),
            "Precise Coding & Tool Calling"
        )
    }

    func testUneditedContextWindowSavePreservesStoredValue() {
        XCTAssertEqual(CloudProvidersSettingsView.effectiveWindowInput(draft: nil, stored: 128_000), "128000")
        XCTAssertEqual(CloudProvidersSettingsView.effectiveWindowInput(draft: nil, stored: nil), "")
        XCTAssertEqual(CloudProvidersSettingsView.effectiveWindowInput(draft: "", stored: 128_000), "")
        XCTAssertEqual(CloudProvidersSettingsView.effectiveWindowInput(draft: "200000", stored: 128_000), "200000")
    }

    func testFallbackAddressResolvesCloudAndOnDeviceCeilingsWhenModelIsNil() {
        let suite = "CloudProfileAddressingTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }

        // Unregistered/ad-hoc cloud model with no recorded window -> 16,384 + cloud explanation.
        let adHocCloud = "cloud://synthetic/unregistered-model"
        XCTAssertEqual(
            resolveNonLocalMaxOutputCeiling(model: nil, fallbackAddress: adHocCloud, defaults: defaults),
            CloudContextPolicy.unknownWindowReplyCeiling
        )
        XCTAssertTrue(
            maxOutputCeilingExplanation(for: nil, fallbackAddress: adHocCloud, defaults: defaults)
                .contains("No context window is recorded for this model")
        )

        // String-formatted global window (e.g. "200,000") applies to fallback address too.
        defaults.set("200,000", forKey: CloudContextPolicy.globalWindowKey)
        XCTAssertEqual(
            resolveNonLocalMaxOutputCeiling(model: nil, fallbackAddress: adHocCloud, defaults: defaults),
            40_000
        )
        XCTAssertTrue(
            maxOutputCeilingExplanation(for: nil, fallbackAddress: adHocCloud, defaults: defaults)
                .contains("200,000-token context window")
        )

        // Apple Foundation Model fallback before discovery scan finishes.
        for afmAddress in [AppleFoundationModelService.modelPath, AppleFoundationModelService.modelId] {
            XCTAssertEqual(
                resolveNonLocalMaxOutputCeiling(model: nil, fallbackAddress: afmAddress, defaults: defaults),
                AppleFoundationModelService.maxResponseTokenCeiling
            )
            XCTAssertTrue(
                maxOutputCeilingExplanation(for: nil, fallbackAddress: afmAddress, defaults: defaults)
                    .contains("on-device model's own response limit")
            )
        }

        // Local fallback defers to KVCacheProvisioning and local checkpoint explanation.
        XCTAssertNil(
            resolveNonLocalMaxOutputCeiling(model: nil, fallbackAddress: "/tmp/local-model", defaults: defaults)
        )
        XCTAssertTrue(
            maxOutputCeilingExplanation(for: nil, fallbackAddress: "/tmp/local-model", defaults: defaults)
                .contains("checkpoint's own position limit")
        )
    }
}
