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

    /// A write that changes nothing must not rewrite the key: churning defaults
    /// notifies every observer of the domain for a value nobody changed. Both
    /// no-op shapes are covered — a save of the value already stored, and a clear
    /// of a model that was never there — and both report success, because the
    /// store does hold what was asked for.
    ///
    /// The RETURN meaning is pinned separately by
    /// `testSetOverrideReportsWhetherTheRequestedStateHolds`.
    func testAWindowWriteThatChangesNothingDoesNotTouchTheStore() {
        let suite = "CloudProfileAddressingTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }

        XCTAssertTrue(
            CloudContextPolicy.setOverride(model: "absent", window: nil, providerId: "p", defaults: defaults)
        )
        XCTAssertNil(defaults.string(forKey: CloudContextPolicy.defaultsKey(for: "p")),
                     "a no-op clear must not create an empty key")

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

    /// The presets feed `ForEach(_:id: \.self)`, so they must never offer a
    /// button below the slider's own floor — which would set a value the control
    /// immediately clamps back up — and the ceiling must still be the top rung.
    ///
    /// Uniqueness and the ceiling bound are pinned by
    /// `CloudProviderLogicTests.testMaxTokenPresetsRespectTheCeilingAndStayUnique`,
    /// which iterates the same ceilings; this covers only the part that is about
    /// the FLOOR, which that test does not touch. The ceiling loop here was a
    /// duplicate of it and has been dropped.
    func testMaxTokenPresetsRespectTheSliderFloor() {
        XCTAssertTrue(maxTokenPresets(upTo: 40_000).allSatisfy { $0 >= maxTokenSliderFloor })
        XCTAssertTrue(maxTokenPresets(upTo: 262_144).allSatisfy { $0 >= maxTokenSliderFloor })

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
        XCTAssertEqual(CloudProvidersSettingsView.effectiveWindowInput(draft: nil, stored: "128000"), "128000")
        XCTAssertEqual(CloudProvidersSettingsView.effectiveWindowInput(draft: nil, stored: nil), "")
        XCTAssertEqual(CloudProvidersSettingsView.effectiveWindowInput(draft: "", stored: "128000"), "")
        XCTAssertEqual(CloudProvidersSettingsView.effectiveWindowInput(draft: "200000", stored: "128000"), "200000")

        // An entry this build cannot parse is still the stored value. Handing
        // this the PARSE view meant a stored `12k` arrived as nil, so an
        // unedited Save resolved to "" and took the clear branch — deleting the
        // operator's typo through a button that had shown an empty field for it.
        XCTAssertEqual(
            CloudProvidersSettingsView.effectiveWindowInput(draft: nil, stored: "12k"),
            "12k",
            "an unreadable stored entry must be re-submitted, not resolved to a clear"
        )
    }

    /// Every family key `seedOfficialProfiles` writes must hold that FAMILY's
    /// profile, not the generic branch's.
    ///
    /// The seed names its families as literals while `defaultFor` decides which
    /// profile answers — and the seed takes every value FROM `defaultFor`, so
    /// the two agreeing today is what keeps the stored numbers the official
    /// ones. If the matching drifted, a fresh install would silently store the
    /// generic (cloud/ungated-local) budget and sampling pair under the Ornith,
    /// Qwen and Nanbeige keys, with no error and no visible difference until a
    /// turn misbehaved. Iterating `ModelProfileType.allCases` covers both
    /// profiles each key is seeded with.
    func testEverySeededFamilyKeyHoldsItsFamilyProfileNotTheGenericOne() {
        let (manager, defaults, suite) = scratchManager()
        defer { defaults.removePersistentDomain(forName: suite) }

        let seededKeys: [(key: String, familyName: String)] = [
            ("mlx-community/ornith-1.5-9b-optiq-4bit", "ornith"),
            ("ornith-1.5-9b-optiq-4bit", "ornith"),
            ("ornith", "ornith"),
            ("qwen/qwen3.8-flash-next-fp8", "qwen"),
            ("qwen3.8-flash-next-fp8", "qwen"),
            ("qwen", "qwen"),
            ("nanbeige/nanbeige4.3-33b", "nanbeige"),
            ("nanbeige4.3-33b", "nanbeige"),
            ("nanbeige", "nanbeige")
        ]

        for (key, familyName) in seededKeys {
            for type in ModelProfileType.allCases {
                let stored = manager.getProfile(for: key, type: type)
                let family = GenerationProfileSettings.defaultFor(
                    type: type,
                    modelName: familyName
                )
                let generic = GenerationProfileSettings.defaultFor(type: type)
                XCTAssertEqual(
                    stored,
                    family,
                    "\(key) must be seeded with the \(familyName) \(type.rawValue) profile"
                )
                XCTAssertNotEqual(
                    stored,
                    generic,
                    "\(key) must not be seeded with the generic \(type.rawValue) profile"
                )
            }
        }
    }

    /// The raw view is what the editor shows and what a Save re-submits; the
    /// parse view is what the send path uses. They disagree about what they
    /// cannot read — which is the point — and agree on everything else.
    func testRawOverridesKeepUnreadableEntriesTheParseViewDrops() {
        let suite = "CloudProfileAddressingTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }

        defaults.set("good=128000; typo=12k; tiny=512", forKey: CloudContextPolicy.defaultsKey(for: "p"))

        XCTAssertEqual(
            CloudContextPolicy.overrides(for: "p", defaults: defaults),
            ["good": 128_000],
            "the parse view reports only what it can read"
        )
        XCTAssertEqual(
            CloudContextPolicy.rawOverrides(for: "p", defaults: defaults),
            ["good": "128000", "typo": "12k", "tiny": "512"],
            "the raw view reports what is stored, including what it cannot read"
        )

        // Last entry wins in both views when both are readable.
        defaults.set("m=1000; m=2000", forKey: CloudContextPolicy.defaultsKey(for: "q"))
        XCTAssertEqual(CloudContextPolicy.rawOverrides(for: "q", defaults: defaults)["m"], "2000")
        XCTAssertEqual(CloudContextPolicy.overrides(for: "q", defaults: defaults)["m"], 2_000)
    }

    /// A duplicated model id whose LAST entry is unreadable must not take the
    /// whole model down with it.
    ///
    /// The parse view used to be derived from the raw view, so a top-level
    /// last-one-wins merge handed it `12k`, its parse dropped the entry, and the
    /// model vanished from the parse view entirely — on the SEND path, where
    /// `window(forModelId:providerId:)` feeds `replyBudget` and `shouldCompact`.
    /// A hand-written `defaults write` with a duplicate therefore swapped a known
    /// window for the unrecorded-window fallback and silently disabled the
    /// pre-flight compaction check: the under-compaction failure `window()`'s own
    /// documentation warns about.
    ///
    /// The two views now differ deliberately — the raw view reports what the file
    /// ends with because that is what the editor must show and re-submit, the
    /// parse view reports the last entry anyone could READ — so this pins both
    /// halves of the disagreement.
    func testADuplicateUnreadableEntryDoesNotHideTheReadableOne() {
        let suite = "CloudProfileAddressingTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let key = CloudContextPolicy.defaultsKey(for: "p")

        // Junk last: the editor shows the junk, the send path keeps the window.
        defaults.set("a=200000; a=12k", forKey: key)
        XCTAssertEqual(
            CloudContextPolicy.rawOverrides(for: "p", defaults: defaults)["a"],
            "12k",
            "the editor shows the text the file ends with, so the typo is fixable"
        )
        XCTAssertEqual(
            CloudContextPolicy.overrides(for: "p", defaults: defaults)["a"],
            200_000,
            "the send path keeps the last READABLE window rather than losing the model"
        )
        XCTAssertEqual(
            CloudContextPolicy.window(forModelId: "a", providerId: "p", defaults: defaults),
            200_000,
            "the resolved window follows the parse view, so the ceiling does too"
        )
        XCTAssertEqual(
            CloudContextPolicy.windowSource(forModelId: "a", providerId: "p", defaults: defaults),
            .model(200_000)
        )

        // A sub-plausible last entry is dropped by the reader exactly the way an
        // unreadable one is, and must resolve the same way.
        defaults.set("b=200000; b=512", forKey: key)
        XCTAssertEqual(CloudContextPolicy.overrides(for: "p", defaults: defaults)["b"], 200_000)
        XCTAssertEqual(CloudContextPolicy.rawOverrides(for: "p", defaults: defaults)["b"], "512")

        // Nothing readable at all is still absent, not a guess.
        defaults.set("c=12k; c=oops", forKey: key)
        XCTAssertNil(CloudContextPolicy.overrides(for: "p", defaults: defaults)["c"])
        XCTAssertNil(CloudContextPolicy.window(forModelId: "c", providerId: "p", defaults: defaults))
    }

    /// `setOverride` trims the model id the way the reader trims a stored entry's.
    ///
    /// An untrimmed `" a "` passed the emptiness check and then compared unequal
    /// to the `a` `splitEntry` had trimmed, so saving it APPENDED a duplicate
    /// instead of replacing in place — reaching the duplicate case above through
    /// the API rather than through `defaults write`.
    func testSetOverrideTreatsAnUntrimmedModelIdAsTheSameModel() {
        let suite = "CloudProfileAddressingTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let key = CloudContextPolicy.defaultsKey(for: "p")

        XCTAssertTrue(
            CloudContextPolicy.setOverride(model: "a", window: 128_000, providerId: "p", defaults: defaults)
        )
        XCTAssertTrue(
            CloudContextPolicy.setOverride(
                model: "  a\n",
                window: 200_000,
                providerId: "p",
                defaults: defaults
            )
        )
        XCTAssertEqual(
            defaults.string(forKey: key),
            "a=200000",
            "the padded id replaces the stored entry instead of appending a second one"
        )

        // Whitespace-only is not a model id at all, and still reports false.
        XCTAssertFalse(
            CloudContextPolicy.setOverride(model: "   ", window: 128_000, providerId: "p", defaults: defaults)
        )
        XCTAssertEqual(defaults.string(forKey: key), "a=200000")
    }

    /// "Is this text a window the app can read" is asked in three places — the
    /// Save button, the Cloud tab's unreadable-record note, and the reader
    /// itself — so it lives in ONE place and this pins the three together.
    ///
    /// The Save button's validation and the reader are the pair that matters: a
    /// button that accepts a value the reader drops stores a window that then
    /// reads back as absent while the ceiling never moves.
    func testTheUsableWindowTestIsTheReadersOwnRule() {
        let suite = "CloudProfileAddressingTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let key = CloudContextPolicy.defaultsKey(for: "p")

        for text in ["128000", "1,048,576", "1_048_576", " 8192 "] {
            XCTAssertTrue(CloudContextPolicy.isUsableWindow(text), text)
            defaults.set("m=\(text)", forKey: key)
            XCTAssertEqual(
                CloudContextPolicy.overrides(for: "p", defaults: defaults)["m"],
                CloudContextPolicy.parseWindowValue(text),
                "the button and the reader accept exactly the same text: \(text)"
            )
        }
        for text in [nil, "", "12k", "0", "512", "abc", "1,2,3"] {
            XCTAssertFalse(CloudContextPolicy.isUsableWindow(text), String(describing: text))
            defaults.set("m=\(text ?? "")", forKey: key)
            XCTAssertNil(
                CloudContextPolicy.overrides(for: "p", defaults: defaults)["m"],
                "the button and the reader reject exactly the same text: \(String(describing: text))"
            )
        }
    }

    /// The Help pane quotes the generic profile's budgets, so it has to read
    /// them from that profile. The two figures it printed were literals under a
    /// comment in `ModelProfileManager` arguing against exactly that, and the
    /// Help coverage test counts topic names, not numbers.
    func testTheHelpPaneQuotesTheGenericProfilesOwnBudgets() {
        let coder = GenerationProfileSettings.defaultFor(type: .coder)
        let assistant = GenerationProfileSettings.defaultFor(type: .assistant)

        // The coder budget IS the unrecorded-window ceiling; the Help pane shows
        // whatever this resolves to, so pinning the identity pins the text.
        XCTAssertEqual(coder.maxNewTokens, CloudContextPolicy.unknownWindowReplyCeiling)
        XCTAssertEqual(assistant.maxNewTokens, coder.maxNewTokens / 2)

        // The formatted strings the pane renders, so a change to either budget
        // shows up as a change here rather than as a silently stale sentence.
        XCTAssertEqual(coder.maxNewTokens.formatted(), "16,384")
        XCTAssertEqual(assistant.maxNewTokens.formatted(), "8,192")
    }

    /// `setOverride`'s return answers ONE question: does the store now hold the
    /// requested value. A clear of an absent model satisfies that request, so it
    /// reports true — the same way a save of the value already stored does.
    /// `false` is reserved for a request that could not be honoured at all.
    func testSetOverrideReportsWhetherTheRequestedStateHolds() {
        let suite = "CloudProfileAddressingTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let key = CloudContextPolicy.defaultsKey(for: "p")

        // A clear of a model that was never there: the requested state (absent)
        // is what the store holds, so true — and nothing is written.
        XCTAssertTrue(
            CloudContextPolicy.setOverride(model: "absent", window: nil, providerId: "p", defaults: defaults)
        )
        XCTAssertNil(defaults.string(forKey: key), "a satisfied clear must not create an empty key")

        // Requests that cannot be honoured are the only false answers.
        XCTAssertFalse(
            CloudContextPolicy.setOverride(model: "", window: 128_000, providerId: "p", defaults: defaults)
        )
        XCTAssertFalse(
            CloudContextPolicy.setOverride(model: "m", window: 512, providerId: "p", defaults: defaults),
            "a sub-plausible window would be dropped by the reader and read back as absent"
        )
        XCTAssertNil(defaults.string(forKey: key))

        // A real save, then a real clear, both report true.
        XCTAssertTrue(
            CloudContextPolicy.setOverride(model: "m", window: 128_000, providerId: "p", defaults: defaults)
        )
        XCTAssertTrue(
            CloudContextPolicy.setOverride(model: "m", window: nil, providerId: "p", defaults: defaults)
        )
        XCTAssertNil(defaults.string(forKey: key))
    }

    /// The ceiling's `task(id:)` watches this string, so it has to move when a
    /// window is recorded and stay put when nothing relevant changed. It takes
    /// an injectable `defaults:` for the same reason `setOverride` does: reading
    /// the process domain from the view left this path untestable.
    func testWindowFactsTrackTheProviderEntriesAndTheGlobalFallback() {
        let suite = "CloudProfileAddressingTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }

        let address = "cloud://synthetic/unregistered-model"
        let empty = CloudContextPolicy.windowFacts(forAddress: address, defaults: defaults)
        XCTAssertNotNil(empty, "a cloud address always has facts, even when nothing is recorded")

        // A per-model window changes the facts.
        CloudContextPolicy.setOverride(
            model: "unregistered-model",
            window: 200_000,
            providerId: "synthetic",
            defaults: defaults
        )
        let withWindow = CloudContextPolicy.windowFacts(forAddress: address, defaults: defaults)
        XCTAssertNotEqual(withWindow, empty, "recording a window must invalidate the ceiling")

        // So does the global fallback, which is the other input to the answer.
        defaults.set(64_000, forKey: CloudContextPolicy.globalWindowKey)
        let withFallback = CloudContextPolicy.windowFacts(forAddress: address, defaults: defaults)
        XCTAssertNotEqual(
            withFallback,
            withWindow,
            "the global fallback must invalidate the ceiling too"
        )

        // An unreadable entry also moves the facts, even though it cannot move
        // the number: the string carries the raw text, so correcting a typo
        // re-runs a resolution that lands where it already was. That is a
        // wasted lookup and not a wrong answer — a parse-view identity would
        // instead hide the edit, and the row would keep showing a ceiling the
        // stored bytes no longer describe.
        CloudContextPolicy.setOverride(
            model: "unregistered-model",
            window: nil,
            providerId: "synthetic",
            defaults: defaults
        )
        defaults.set("unregistered-model=12k", forKey: CloudContextPolicy.defaultsKey(for: "synthetic"))
        XCTAssertNotEqual(
            CloudContextPolicy.windowFacts(forAddress: address, defaults: defaults),
            withFallback,
            "an edit the reader cannot parse still has to re-resolve"
        )
        XCTAssertEqual(
            CloudContextPolicy.windowSource(
                forModelId: "unregistered-model",
                providerId: "synthetic",
                defaults: defaults
            ),
            .global(64_000),
            "and the resolution it re-runs still lands on the fallback"
        )

        // A non-cloud address has no such facts, which is what tells the view
        // to price it from its checkpoint instead.
        XCTAssertNil(CloudContextPolicy.windowFacts(forAddress: "/tmp/local-model", defaults: defaults))
        XCTAssertNil(CloudContextPolicy.windowFacts(forAddress: nil, defaults: defaults))
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
        // The explanation must name the FALLBACK, not claim the model recorded a
        // 200,000-token window of its own. The number alone cannot tell the two
        // apart, which is why the backend carries the window's origin; the Cloud
        // tab's own row already says "No window of its own — the global …
        // applies", so a pane saying otherwise contradicts it about one model.
        let globalExplanation = maxOutputCeilingExplanation(
            for: nil,
            fallbackAddress: adHocCloud,
            defaults: defaults
        )
        XCTAssertTrue(
            globalExplanation.contains("global 200,000-token fallback"),
            "expected the fallback to be named, got: \(globalExplanation)"
        )
        XCTAssertFalse(
            globalExplanation.contains("this model's 200,000-token context window"),
            "a fallback value is not this model's own recorded window"
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

        // Local fallback defers to KVCacheProvisioning. Its sentence names the
        // POLICY FLOOR rather than a position limit, because no config was read
        // and the number shown (32,768) is not derived from one.
        XCTAssertNil(
            resolveNonLocalMaxOutputCeiling(model: nil, fallbackAddress: "/tmp/local-model", defaults: defaults)
        )
        let localExplanation = maxOutputCeilingExplanation(
            for: nil,
            fallbackAddress: "/tmp/local-model",
            defaults: defaults
        )
        XCTAssertTrue(localExplanation.contains("policy floor"), localExplanation)
        XCTAssertFalse(
            localExplanation.contains("checkpoint's own"),
            "a 32,768 floor is not the checkpoint's own limit: \(localExplanation)"
        )
    }

    /// A sub-32k checkpoint is capped by the POLICY FLOOR, not by its own
    /// positions, and the sentence beside the slider has to say so. Every 8k
    /// Qwen and older Llama is in this case: the ceiling shown is 32,768 — 24,576
    /// rows ABOVE what the checkpoint declares — so "capped by this checkpoint's
    /// own position limit" told them the opposite of what was binding.
    func testTheLocalCeilingExplanationNamesTheFloorRatherThanASmallCheckpointsLimit() throws {
        let suite = "CloudProfileAddressingTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }

        let floor = KVCacheProvisioning.maxProvisionedRows

        // An 8k checkpoint: the floor binds, and the sentence says so instead of
        // claiming 8,192 positions are the ceiling.
        let small = 8_192
        XCTAssertEqual(KVCacheProvisioning.localMaxOutputTokens(config: nil), floor)
        XCTAssertEqual(
            KVCacheProvisioning.localCeilingBasis(declaredMaxPositionEmbeddings: small),
            .policyFloor(declared: small)
        )
        let smallExplanation = maxOutputCeilingExplanation(
            for: nil,
            fallbackAddress: "/tmp/local-model",
            declaredMaxPositionEmbeddings: small,
            defaults: defaults
        )
        XCTAssertTrue(smallExplanation.contains("8,192 positions"), smallExplanation)
        XCTAssertTrue(smallExplanation.contains("policy floor"), smallExplanation)
        XCTAssertFalse(smallExplanation.contains("checkpoint's own"), smallExplanation)

        // A checkpoint between the floor and the absolute maximum: its own
        // declared limit IS the ceiling, and the sentence says that.
        let wide = 131_072
        XCTAssertEqual(
            KVCacheProvisioning.localCeilingBasis(declaredMaxPositionEmbeddings: wide),
            .checkpointPositions(wide)
        )
        let wideExplanation = maxOutputCeilingExplanation(
            for: nil,
            fallbackAddress: "/tmp/local-model",
            declaredMaxPositionEmbeddings: wide,
            defaults: defaults
        )
        XCTAssertTrue(wideExplanation.contains("131,072"), wideExplanation)
        XCTAssertTrue(wideExplanation.contains("checkpoint's own"), wideExplanation)
        XCTAssertFalse(wideExplanation.contains("policy floor"), wideExplanation)

        // A checkpoint claiming more positions than this app will provision is
        // capped by the policy MAXIMUM, which is a third fact again: 524,288
        // positions do not make a 524,288-token ceiling.
        let huge = 524_288
        XCTAssertEqual(
            KVCacheProvisioning.localCeilingBasis(declaredMaxPositionEmbeddings: huge),
            .policyMaximum(declared: huge)
        )
        let hugeExplanation = maxOutputCeilingExplanation(
            for: nil,
            fallbackAddress: "/tmp/local-model",
            declaredMaxPositionEmbeddings: huge,
            defaults: defaults
        )
        XCTAssertTrue(hugeExplanation.contains("524,288 positions"), hugeExplanation)
        XCTAssertTrue(
            hugeExplanation.contains(KVCacheProvisioning.absoluteMaxProvisionedRows.formatted()),
            hugeExplanation
        )

        // And the three cases agree with the ceiling that is actually shown, so
        // the sentence can never explain a number the slider is not offering.
        for declared in [small, wide, huge] {
            let config = try JSONDecoder().decode(
                ModelConfig.self,
                from: Data("{\"model_type\":\"llama\",\"max_position_embeddings\":\(declared)}".utf8)
            )
            XCTAssertEqual(
                KVCacheProvisioning.localCeilingBasis(config: config),
                KVCacheProvisioning.localCeilingBasis(declaredMaxPositionEmbeddings: declared)
            )
            XCTAssertEqual(
                KVCacheProvisioning.localMaxOutputTokens(config: config),
                KVCacheProvisioning.policyMaxRows(config: config)
            )
        }
    }

    // MARK: - Which window the ceiling was priced against

    /// `windowSource` is what lets an explanation name the right fact: a
    /// per-model record belongs to the model, the single global fallback does
    /// not. The NUMBER must stay the one `window(forModelId:providerId:)`
    /// answers — the two are the same read expressed twice.
    func testTheWindowSourceNamesWhereTheResolvedWindowCameFrom() {
        let suite = "CloudProfileAddressingTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }

        // Nothing recorded anywhere.
        XCTAssertEqual(
            CloudContextPolicy.windowSource(forModelId: "m", providerId: "p", defaults: defaults),
            .none
        )
        XCTAssertNil(CloudContextPolicy.window(forModelId: "m", providerId: "p", defaults: defaults))

        // Only the fallback: it supplies the number but is NOT the model's own.
        defaults.set(200_000, forKey: CloudContextPolicy.globalWindowKey)
        XCTAssertEqual(
            CloudContextPolicy.windowSource(forModelId: "m", providerId: "p", defaults: defaults),
            .global(200_000)
        )
        XCTAssertEqual(
            CloudContextPolicy.window(forModelId: "m", providerId: "p", defaults: defaults),
            200_000,
            "the source and the resolved window must agree"
        )

        // A per-model record wins over the fallback, and says so.
        XCTAssertTrue(
            CloudContextPolicy.setOverride(model: "m", window: 32_768, providerId: "p", defaults: defaults)
        )
        XCTAssertEqual(
            CloudContextPolicy.windowSource(forModelId: "m", providerId: "p", defaults: defaults),
            .model(32_768)
        )
        XCTAssertEqual(CloudContextPolicy.window(forModelId: "m", providerId: "p", defaults: defaults), 32_768)
    }

    /// The Generation tab's ceiling line and the Cloud tab's row have to agree
    /// about the same model: one is not allowed to say "this model's window"
    /// when the other already says "no window of its own".
    func testTheCeilingExplanationNamesAModelsOwnWindowWhenItHasOne() {
        let suite = "CloudProfileAddressingTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }

        let address = "cloud://synthetic/registered-model"
        XCTAssertTrue(
            CloudContextPolicy.setOverride(
                model: "registered-model",
                window: 128_000,
                providerId: "synthetic",
                defaults: defaults
            )
        )
        let explanation = maxOutputCeilingExplanation(
            for: nil,
            fallbackAddress: address,
            defaults: defaults
        )
        XCTAssertTrue(
            explanation.contains("this model's 128,000-token context window"),
            "a recorded window is the model's own fact, got: \(explanation)"
        )
        XCTAssertFalse(explanation.contains("fallback"))
    }

    /// A stored value above the ceiling is KEPT on every backend, but no
    /// backend sends it: the cloud route clamps through `replyBudget`, a local
    /// turn's allocation is lowered by `turnFit`, and the on-device turn is
    /// clamped before the framework sees it. The note has to say which one,
    /// because "the engine still honours it" was false on all three.
    func testTheOverCeilingNoteNamesWhatEachBackendDoesWithTheSavedValue() {
        let suite = "CloudProfileAddressingTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }

        defaults.set(200_000, forKey: CloudContextPolicy.globalWindowKey)
        let cloudCeiling = resolveNonLocalMaxOutputCeiling(
            model: nil,
            fallbackAddress: "cloud://synthetic/unregistered-model",
            defaults: defaults
        )!
        let cloud = maxOutputOverCeilingExplanation(
            savedTokens: 262_144,
            ceiling: cloudCeiling,
            for: nil,
            fallbackAddress: "cloud://synthetic/unregistered-model",
            defaults: defaults
        )
        XCTAssertTrue(cloud.contains("262,144"), "the saved value is named, got: \(cloud)")
        XCTAssertTrue(cloud.contains("\(cloudCeiling.formatted())-token ceiling"), "the ceiling is named")
        XCTAssertTrue(cloud.contains("cloud turn lowers the reply budget"), cloud)
        XCTAssertFalse(cloud.contains("still honours"), "the send path does lower it: \(cloud)")

        let onDevice = maxOutputOverCeilingExplanation(
            savedTokens: 262_144,
            ceiling: AppleFoundationModelService.maxResponseTokenCeiling,
            for: nil,
            fallbackAddress: AppleFoundationModelService.modelPath,
            defaults: defaults
        )
        XCTAssertTrue(onDevice.contains("on-device turn clamps it"), onDevice)

        let local = maxOutputOverCeilingExplanation(
            savedTokens: 262_144,
            ceiling: KVCacheProvisioning.localMaxOutputTokens(config: nil),
            for: nil,
            fallbackAddress: "/tmp/local-model",
            defaults: defaults
        )
        XCTAssertTrue(local.contains("local turn lowers the allocation"), local)
        XCTAssertTrue(local.contains("KV budget"), local)
    }

    /// The Cloud tab's Save button validates with the reader's own parser now.
    /// A second parser is how the validator and the reader drift: the button
    /// would accept a value the parser drops, store it, and the row would then
    /// read it back as absent while the ceiling never moved.
    func testTheSavePathValidatesWithTheReadersParser() {
        let suite = "CloudProfileAddressingTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }

        // Both separator spellings the reader tolerates are accepted…
        XCTAssertEqual(CloudContextPolicy.parseWindowValue("1,048,576"), 1_048_576)
        XCTAssertEqual(CloudContextPolicy.parseWindowValue("1_048_576"), 1_048_576)
        XCTAssertEqual(CloudContextPolicy.parseWindowValue("  65536  "), 65_536)
        // …and what it refuses can never be stored by the save either, because
        // the button now asks this function.
        XCTAssertNil(CloudContextPolicy.parseWindowValue("12k"))
        XCTAssertNil(CloudContextPolicy.parseWindowValue(""))
        XCTAssertNil(CloudContextPolicy.parseWindowValue("1,04,576"))

        // Round trip: what the parser accepts is what the store holds.
        let parsed = CloudContextPolicy.parseWindowValue("1,048,576")!
        XCTAssertTrue(
            CloudContextPolicy.setOverride(model: "m", window: parsed, providerId: "p", defaults: defaults)
        )
        XCTAssertEqual(
            CloudContextPolicy.overrides(for: "p", defaults: defaults)["m"],
            1_048_576,
            "a value the parser accepted must read back as the same window"
        )
    }

    // MARK: - A saved cloud profile is not a local family's fallback

    /// A scratch manager whose seedable family keys are EMPTY entries.
    ///
    /// `seedOfficialProfiles` refills a family key only when it is absent, so
    /// empty entries keep it from adding the official profiles without hiding
    /// the family key from `getProfile`. That leaves any saved cloud key as the
    /// only candidate the step-4 scan could return, which is what makes these
    /// assertions deterministic rather than dependent on dictionary order.
    private func managerWithEmptyFamilySeeds() -> (ModelProfileManager, UserDefaults, String) {
        let suite = "CloudProfileAddressingTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defaults.removePersistentDomain(forName: suite)

        let emptySeeds: [String: [String: GenerationProfileSettings]] = [
            "qwen": [:],
            "ornith": [:],
            "nanbeige": [:]
        ]
        defaults.set(try? JSONEncoder().encode(emptySeeds), forKey: "dynamoe_model_profiles_v1")
        defaults.set(true, forKey: "dynamoe_seeded_official_profiles_v3")
        return (ModelProfileManager(defaults: defaults), defaults, suite)
    }

    /// The one-sided half of the family gate.
    ///
    /// Step 4 already skipped the incoming cloud address, but it scans
    /// `profilesByModel` for any key containing the family name. Once a user
    /// saves a profile for a family-named cloud model, that key is a candidate,
    /// and the unordered dictionary decides whether an unseeded local sibling
    /// receives the cloud model's budget. Family matching is a statement about
    /// local checkpoints, so a cloud key must never be one.
    func testASavedCloudProfileIsNotAFamilyFallbackForLocalModels() {
        let (manager, defaults, suite) = managerWithEmptyFamilySeeds()
        defer { defaults.removePersistentDomain(forName: suite) }

        var cloud = GenerationProfileSettings.defaultFor(type: .assistant)
        cloud.temperature = 0.11
        cloud.maxNewTokens = 40_000
        cloud.presencePenalty = 0.00
        manager.saveProfile(
            for: "cloud-model-synthetic-qwen3-coder",
            type: .assistant,
            settings: cloud
        )

        // An unseeded LOCAL sibling (steps 1-3 miss; step 4 would scan).
        let localSibling = manager.getProfile(
            for: "mlx-community/Qwen3-30B-A3B-4bit",
            type: .assistant
        )
        let localDefault = GenerationProfileSettings.defaultFor(
            type: .assistant,
            modelName: "mlx-community/Qwen3-30B-A3B-4bit"
        )
        XCTAssertEqual(localSibling, localDefault,
                       "a local Qwen sibling must not be answered by a saved cloud Qwen profile")
        XCTAssertEqual(localSibling.maxNewTokens, 4_096)
        XCTAssertEqual(localSibling.presencePenalty, 1.50, accuracy: 0.0001)

        // The filter must not make the cloud profile unreachable by its own
        // address: steps 1-2 are exact-key lookups, not family scans.
        let savedCloud = manager.getProfile(
            for: "cloud-model-synthetic-qwen3-coder",
            type: .assistant
        )
        XCTAssertEqual(savedCloud.maxNewTokens, 40_000)
        XCTAssertEqual(savedCloud.temperature, 0.11, accuracy: 0.0001)
    }

    /// Resetting a seeded family restores ITS official profile, not whichever
    /// other key the family scan happens to find first.
    ///
    /// Nanbeige is seeded by `seedOfficialProfiles` exactly as Ornith and Qwen
    /// are, but `resetProfile` used to omit it from the "store the default under
    /// this key" branch. It therefore removed the entry, and step 4 was free to
    /// return a DIFFERENT Nanbeige profile the user had saved under another key.
    func testResettingASeededFamilyStoresItsOfficialProfile() {
        let (manager, defaults, suite) = managerWithEmptyFamilySeeds()
        defer { defaults.removePersistentDomain(forName: suite) }

        var sibling = GenerationProfileSettings.defaultFor(type: .assistant, modelName: "nanbeige")
        sibling.maxNewTokens = 40_000
        manager.saveProfile(for: "custom/nanbeige-finetune", type: .assistant, settings: sibling)

        var custom = GenerationProfileSettings.defaultFor(type: .assistant, modelName: "nanbeige")
        custom.maxNewTokens = 2_048
        manager.saveProfile(for: "nanbeige/nanbeige4.3-33b", type: .assistant, settings: custom)

        manager.resetProfile(for: "nanbeige/nanbeige4.3-33b", type: .assistant)

        let expected = GenerationProfileSettings.defaultFor(
            type: .assistant,
            modelName: "nanbeige/nanbeige4.3-33b"
        )
        let reset = manager.getProfile(for: "nanbeige/nanbeige4.3-33b", type: .assistant)
        XCTAssertEqual(reset, expected)
        XCTAssertEqual(reset.maxNewTokens, 262_144)
        XCTAssertNotNil(
            manager.profilesByModel[manager.normalizeKey("nanbeige/nanbeige4.3-33b")],
            "a reset seeded family must be stored under its own key, not removed"
        )
    }
}
