//
//  CloudProviderCatalogTests.swift
//  DynaMoE
//
//  Runtime-configured cloud providers: identifier and URL rules, lenient
//  persistence, sentinel parsing with several providers, removal side
//  effects, and the /models envelope shapes a gateway may answer with.
//

import XCTest
import Combine
@testable import DynaMoE

// MARK: - Catalog

final class CloudProviderCatalogTests: XCTestCase {
    private var defaults: UserDefaults!
    private var suiteName: String!

    override func setUp() {
        super.setUp()
        suiteName = "dynamoe.cloud.providers.tests.\(UUID().uuidString)"
        defaults = UserDefaults(suiteName: suiteName)
        defaults.removePersistentDomain(forName: suiteName)
    }

    override func tearDown() {
        defaults.removePersistentDomain(forName: suiteName)
        defaults = nil
        suiteName = nil
        super.tearDown()
    }

    private func zen(
        name: String = "OpenCode Zen",
        identifier: String = "opencode-zen",
        baseURL: String = "https://opencode.ai/zen/v1",
        docsURL: String = ""
    ) throws -> CloudProvider {
        try CloudProviderCatalog.validate(
            displayName: name,
            identifier: identifier,
            baseURL: baseURL,
            docsURL: docsURL,
            existing: CloudProviderCatalog.builtIns
        ).get()
    }

    /// The defaults are deliberately not consulted: `validate` falls back to
    /// `UserDefaults.standard` when `existing` is nil, which makes a URL-rule
    /// test depend on whether this Mac ever saved a provider. Every case here
    /// compares against the built-ins unless it says otherwise.
    private func validationFailure(
        displayName: String = "OpenCode Zen",
        identifier: String = "opencode-zen",
        baseURL: String = "https://opencode.ai/zen/v1",
        docsURL: String = "",
        existing: [CloudProvider]? = nil
    ) -> CloudProviderValidationError? {
        if case .failure(let failure) = CloudProviderCatalog.validate(
            displayName: displayName,
            identifier: identifier,
            baseURL: baseURL,
            docsURL: docsURL,
            existing: existing ?? CloudProviderCatalog.builtIns
        ) {
            return failure
        }
        return nil
    }

    // MARK: Persistence

    func testACustomProviderPersistsAndResolvesItsModels() throws {
        let provider = try zen()
        CloudProviderCatalog.saveCustomProviders([provider], defaults: defaults)

        XCTAssertEqual(CloudProviderCatalog.customProviders(defaults: defaults).map(\.id), ["opencode-zen"])
        XCTAssertEqual(CloudProviderCatalog.all(defaults: defaults).map(\.id), ["synthetic", "opencode-zen"])

        let id = CloudModelAddress.modelId(providerId: provider.id, providerModelId: "qwen3-coder")
        let parsedID = try XCTUnwrap(CloudModelAddress.parseModelId(
            id,
            providers: CloudProviderCatalog.all(defaults: defaults)
        ))
        XCTAssertEqual(parsedID.provider.id, "opencode-zen")
        XCTAssertEqual(parsedID.providerModelId, "qwen3-coder")

        let path = CloudModelAddress.sentinelPath(providerId: provider.id, providerModelId: "qwen3-coder")
        let parsedPath = try XCTUnwrap(CloudModelAddress.parsePath(
            path,
            providers: CloudProviderCatalog.all(defaults: defaults)
        ))
        XCTAssertEqual(parsedPath.provider.id, "opencode-zen")
    }

    func testACredentialedCustomProviderContributesItsModels() throws {
        let provider = try zen()
        CloudProviderCatalog.saveCustomProviders([provider], defaults: defaults)
        CloudModelRegistry.setRegisteredModelIds(["qwen3-coder"], for: provider, defaults: defaults)

        let discovered = CloudModelRegistry.makeDiscoveredModels(
            defaults: defaults,
            providers: CloudProviderCatalog.all(defaults: defaults),
            hasCredential: { $0 == "opencode-zen" }
        )

        XCTAssertEqual(discovered.map(\.id), ["cloud-model-opencode-zen-qwen3-coder"])
        XCTAssertEqual(discovered.first?.displayName, "qwen3-coder (OpenCode Zen)")
        XCTAssertEqual(discovered.first?.snapshotPath, "cloud://opencode-zen/qwen3-coder")
        XCTAssertTrue(discovered.first?.isCloudModel == true)
    }

    func testBuiltInsComeFirstAndDuplicateIdentifiersAreDropped() throws {
        let provider = try zen()
        let syntheticCopy = CloudProvider(
            id: "synthetic",
            displayName: "Not the built-in",
            openAIBaseURL: URL(string: "https://example.com/v1")!
        )

        CloudProviderCatalog.saveCustomProviders(
            [provider, provider, syntheticCopy],
            defaults: defaults
        )

        XCTAssertEqual(
            CloudProviderCatalog.customProviders(defaults: defaults).map(\.id),
            ["opencode-zen"],
            "a duplicate and a built-in collision cost those entries, not the list"
        )
        XCTAssertEqual(
            CloudProviderCatalog.all(defaults: defaults).map(\.id),
            ["synthetic", "opencode-zen"]
        )
        XCTAssertEqual(CloudProviderCatalog.all(defaults: defaults).first?.displayName, "Synthetic")
    }

    func testOneMalformedRecordDoesNotDropTheRest() throws {
        let json = #"""
        [
          {"id":"opencode-zen","displayName":"OpenCode Zen","openAIBaseURL":"https://opencode.ai/zen/v1"},
          {"displayName":"No identifier"},
          {"id":"synthetic","displayName":"Fake built-in","openAIBaseURL":"https://example.com/v1"},
          {"id":"Bad Id!","displayName":"Bad identifier","openAIBaseURL":"https://example.com/v1"},
          {"id":"other","displayName":"Other","openAIBaseURL":"https://other.example/v1"}
        ]
        """#
        defaults.set(Data(json.utf8), forKey: CloudProviderCatalog.customDefaultsKey)

        XCTAssertEqual(
            CloudProviderCatalog.customProviders(defaults: defaults).map(\.id),
            ["opencode-zen", "other"]
        )
    }

    func testUnreadableStorageReadsAsNoProviders() {
        defaults.set(Data("not json".utf8), forKey: CloudProviderCatalog.customDefaultsKey)
        XCTAssertTrue(CloudProviderCatalog.customProviders(defaults: defaults).isEmpty)
        XCTAssertEqual(CloudProviderCatalog.all(defaults: defaults).map(\.id), ["synthetic"])
    }

    // MARK: Identifier rules

    func testIdentifierRules() {
        XCTAssertEqual(validationFailure(identifier: ""), .invalidIdentifier)
        XCTAssertEqual(validationFailure(identifier: "open code"), .invalidIdentifier)
        XCTAssertEqual(validationFailure(identifier: "-zen"), .invalidIdentifier)
        XCTAssertEqual(validationFailure(identifier: "zen-"), .invalidIdentifier)
        XCTAssertEqual(validationFailure(identifier: "zen_key"), .invalidIdentifier)
        XCTAssertEqual(
            validationFailure(identifier: String(repeating: "a", count: 33)),
            .invalidIdentifier
        )
        XCTAssertNil(
            validationFailure(identifier: String(repeating: "a", count: 32)),
            "32 characters is the ceiling, not one past it"
        )
    }

    func testIdentifierIsLowercasedRatherThanRejected() throws {
        let provider = try zen(identifier: "openCode-Zen")
        XCTAssertEqual(provider.id, "opencode-zen")
    }

    func testDuplicateAndReservedIdentifiers() throws {
        let existing = [try zen()]
        XCTAssertEqual(
            validationFailure(existing: CloudProviderCatalog.builtIns + existing),
            .duplicateIdentifier("opencode-zen")
        )
        XCTAssertEqual(validationFailure(identifier: "synthetic"), .reservedIdentifier("synthetic"))
    }

    // MARK: URL rules

    func testBaseURLRules() throws {
        XCTAssertEqual(validationFailure(baseURL: ""), .invalidBaseURL)
        XCTAssertEqual(validationFailure(baseURL: "ftp://opencode.ai/v1"), .invalidBaseURL)
        XCTAssertEqual(validationFailure(baseURL: "not a url"), .invalidBaseURL)
        XCTAssertEqual(
            validationFailure(baseURL: "http://opencode.ai/v1"),
            .insecureBaseURL,
            "plain http must only be allowed on loopback"
        )
        XCTAssertEqual(
            validationFailure(baseURL: "https://opencode.ai/zen/v1/chat/completions"),
            .baseURLIncludesEndpoint
        )
        XCTAssertEqual(
            validationFailure(baseURL: "https://opencode.ai/zen/v1/models"),
            .baseURLIncludesEndpoint
        )
        XCTAssertEqual(
            validationFailure(
                baseURL: "https://my.openai.azure.com/openai/deployments/gpt4?api-version=2024-02-01"
            ),
            .baseURLIncludesQuery,
            "a query is refused rather than silently dropped, so an Azure-style base fails with a reason instead of a 404"
        )

        XCTAssertEqual(
            try zen(baseURL: "https://opencode.ai/zen/v1/").openAIBaseURL.absoluteString,
            "https://opencode.ai/zen/v1",
            "a trailing slash is normalized away"
        )
        XCTAssertEqual(
            try zen(baseURL: "opencode.ai/zen/v1").openAIBaseURL.absoluteString,
            "https://opencode.ai/zen/v1",
            "a missing scheme defaults to https"
        )
        XCTAssertEqual(
            try zen(baseURL: "http://127.0.0.1:8787/v1").openAIBaseURL.absoluteString,
            "http://127.0.0.1:8787/v1"
        )
        XCTAssertEqual(
            try zen(baseURL: "http://localhost:4096/v1").openAIBaseURL.absoluteString,
            "http://localhost:4096/v1"
        )
        XCTAssertEqual(
            try zen(baseURL: "http://[::1]:4096/v1").openAIBaseURL.absoluteString,
            "http://[::1]:4096/v1"
        )
        XCTAssertEqual(
            try zen(baseURL: "HTTPS://OpenCode.ai/zen/v1").openAIBaseURL.absoluteString,
            "https://opencode.ai/zen/v1",
            "the scheme and host are lowercased, so the stored URL is the one displayed and cached"
        )
        XCTAssertEqual(
            validationFailure(baseURL: "https://user:s3cret@opencode.ai/zen/v1"),
            .credentialsInBaseURL,
            "a credential in the URL can never be persisted: it belongs in the Keychain"
        )
        XCTAssertEqual(
            validationFailure(baseURL: "https://user@opencode.ai/zen/v1"),
            .credentialsInBaseURL
        )

        let provider = try zen()
        XCTAssertEqual(provider.chatCompletionsURL.absoluteString, "https://opencode.ai/zen/v1/chat/completions")
        XCTAssertEqual(provider.modelsURL.absoluteString, "https://opencode.ai/zen/v1/models")
    }

    func testDocsLinkIsOptionalAndValidated() throws {
        XCTAssertNil(try zen().docsURL)
        XCTAssertEqual(
            try zen(docsURL: "https://opencode.ai/docs").docsURL?.absoluteString,
            "https://opencode.ai/docs"
        )
        XCTAssertEqual(
            try zen(docsURL: "http://opencode.ai/docs").docsURL?.absoluteString,
            "http://opencode.ai/docs"
        )
        XCTAssertEqual(validationFailure(docsURL: "not a url"), .invalidDocsURL)
        XCTAssertEqual(validationFailure(docsURL: "not-a-url"), .invalidDocsURL)
        XCTAssertEqual(
            validationFailure(docsURL: "file:///etc/passwd"),
            .invalidDocsURL,
            "the docs link is a web link, symmetric with the base URL"
        )
        XCTAssertEqual(validationFailure(docsURL: "javascript:alert(1)"), .invalidDocsURL)
        XCTAssertEqual(
            validationFailure(docsURL: "https://user:s3cret@opencode.ai/docs"),
            .credentialsInDocsURL,
            "the docs link is persisted verbatim in plain text, so a credential there is a leak even though it never reaches a request"
        )
        XCTAssertEqual(
            validationFailure(docsURL: "https://user@opencode.ai/docs"),
            .credentialsInDocsURL
        )
        XCTAssertEqual(
            validationFailure(docsURL: "https://opencode.ai/docs?token=s3cret"),
            .docsURLIncludesQuery,
            "a query is refused because the link is stored verbatim and can carry a credential onto disk"
        )
        XCTAssertEqual(
            try zen(docsURL: "https://opencode.ai/docs#authentication").docsURL?.absoluteString,
            "https://opencode.ai/docs#authentication",
            "a fragment is a section anchor, never sent to a server, so it is kept"
        )
    }

    func testRestoredProvidersAreRevalidated() throws {
        let json = #"""
        [
          {"id":"plain-http","displayName":"Plain HTTP","openAIBaseURL":"http://opencode.ai/zen/v1"},
          {"id":"creds-base","displayName":"Creds","openAIBaseURL":"https://user:s3cret@opencode.ai/zen/v1"},
          {"id":"creds-docs","displayName":"Creds Docs","openAIBaseURL":"https://opencode.ai/zen/v1","docsURL":"https://user:s3cret@opencode.ai/docs"},
          {"id":"query-docs","displayName":"Query Docs","openAIBaseURL":"https://opencode.ai/zen/v1","docsURL":"https://opencode.ai/docs?token=x"},
          {"id":"insecure-anthropic","displayName":"Insecure Anthropic","openAIBaseURL":"https://opencode.ai/zen/v1","anthropicBaseURL":"http://opencode.ai/anthropic"},
          {"id":"loopback","displayName":"Loopback","openAIBaseURL":"http://127.0.0.1:8787/v1"},
          {"id":"trailing","displayName":"Trailing","openAIBaseURL":"https://opencode.ai/zen/v1/"}
        ]
        """#
        defaults.set(Data(json.utf8), forKey: CloudProviderCatalog.customDefaultsKey)

        let restored = CloudProviderCatalog.customProviders(defaults: defaults)
        XCTAssertEqual(
            restored.map(\.id),
            ["loopback", "trailing"],
            "a stored record that no longer satisfies the URL rules is dropped, and costs only itself"
        )
        XCTAssertEqual(
            restored.last?.openAIBaseURL.absoluteString,
            "https://opencode.ai/zen/v1",
            "a restored record is normalized the same way a fresh draft is"
        )
    }

    // MARK: Slug

    func testSlug() {
        XCTAssertEqual(CloudProviderCatalog.slug(forDisplayName: "OpenCode Zen"), "opencode-zen")
        XCTAssertEqual(CloudProviderCatalog.slug(forDisplayName: "  My   Provider!! "), "my-provider")
        XCTAssertEqual(CloudProviderCatalog.slug(forDisplayName: "A/B 2"), "a-b-2")
        XCTAssertEqual(CloudProviderCatalog.slug(forDisplayName: "!!!"), "")
        XCTAssertEqual(
            CloudProviderCatalog.slug(forDisplayName: String(repeating: "a", count: 40)).count,
            CloudProviderCatalog.maximumIdentifierLength
        )
        XCTAssertFalse(
            CloudProviderCatalog.slug(forDisplayName: String(repeating: "ab-", count: 20)).hasSuffix("-"),
            "truncation must not leave a trailing dash"
        )
    }

    // MARK: Address ambiguity

    func testAddressParsingPrefersTheLongestIdentifier() throws {
        let short = CloudProvider(
            id: "my",
            displayName: "My",
            openAIBaseURL: URL(string: "https://a.example/v1")!
        )
        let long = CloudProvider(
            id: "my-provider",
            displayName: "My Provider",
            openAIBaseURL: URL(string: "https://b.example/v1")!
        )
        let providers = [short, long]

        let id = CloudModelAddress.modelId(providerId: "my-provider", providerModelId: "model-x")
        let parsedID = try XCTUnwrap(CloudModelAddress.parseModelId(id, providers: providers))
        XCTAssertEqual(parsedID.provider.id, "my-provider")
        XCTAssertEqual(parsedID.providerModelId, "model-x")

        let path = CloudModelAddress.sentinelPath(providerId: "my-provider", providerModelId: "model-x")
        let parsedPath = try XCTUnwrap(CloudModelAddress.parsePath(path, providers: providers))
        XCTAssertEqual(parsedPath.provider.id, "my-provider")
        XCTAssertEqual(parsedPath.providerModelId, "model-x")
    }

    func testAddressParsingFallsThroughAnEmptyUpstreamID() throws {
        let short = CloudProvider(
            id: "my",
            displayName: "My",
            openAIBaseURL: URL(string: "https://a.example/v1")!
        )
        let long = CloudProvider(
            id: "my-provider",
            displayName: "My Provider",
            openAIBaseURL: URL(string: "https://b.example/v1")!
        )

        let parsed = try XCTUnwrap(CloudModelAddress.parseModelId(
            "cloud-model-my-provider-",
            providers: [short, long]
        ))
        XCTAssertEqual(parsed.provider.id, "my")
        XCTAssertEqual(parsed.providerModelId, "provider-")

        XCTAssertNil(
            CloudModelAddress.parseModelId("cloud-model-my-provider-", providers: [long]),
            "with no shorter provider to fall back to, an empty upstream id stays unresolvable"
        )
    }
}

// MARK: - Store

@MainActor
final class CloudProviderStoreTests: XCTestCase {

    private struct Harness {
        let store: CloudProviderStore
        let defaults: UserDefaults
        let suite: String
        let secretStore: CloudSecretStore

        func cleanUp() {
            // Reap the throwaway service's items as well: a test that adds a
            // provider and never removes it would otherwise leave a live
            // Keychain entry behind on the machine that ran the suite.
            for provider in CloudProviderCatalog.customProviders(defaults: defaults) {
                secretStore.delete(providerId: provider.id)
            }
            defaults.removePersistentDomain(forName: suite)
        }
    }

    private func makeHarness() -> Harness {
        let suite = "dynamoe.cloud.store.tests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defaults.removePersistentDomain(forName: suite)
        let secretStore = CloudSecretStore(service: "DRP.DynaMoE.cloud.tests.\(UUID().uuidString)")
        let store = CloudProviderStore(
            defaults: defaults,
            secretStore: secretStore,
            rescansModels: false,
            updatesSharedRuntime: false
        )
        return Harness(store: store, defaults: defaults, suite: suite, secretStore: secretStore)
    }

    func testAddStoresTheKeyThenRemovalErasesEverything() throws {
        let harness = makeHarness()
        defer { harness.cleanUp() }

        let provider: CloudProvider
        do {
            provider = try harness.store.add(
                displayName: "OpenCode Zen",
                identifier: "opencode-zen",
                baseURL: "https://opencode.ai/zen/v1",
                apiKey: "sk-test-zen"
            )
        } catch {
            try skipIfKeychainUnreachable(error)
            throw error
        }

        XCTAssertEqual(harness.secretStore.get(providerId: provider.id), "sk-test-zen")
        XCTAssertTrue(harness.store.providers.contains { $0.id == "opencode-zen" })

        CloudModelRegistry.setRegisteredModelIds(["qwen3-coder"], for: provider, defaults: harness.defaults)
        CloudModelRegistry.setVisionSupport(
            true,
            forModel: "qwen3-coder",
            provider: provider,
            defaults: harness.defaults
        )
        CloudModelRegistry.setModelIdsWithoutToolSupport(
            ["qwen3-coder"],
            for: provider,
            defaults: harness.defaults
        )

        try harness.store.remove(provider)

        XCTAssertNil(harness.secretStore.get(providerId: provider.id))
        XCTAssertTrue(CloudModelRegistry.registeredModelIds(for: provider, defaults: harness.defaults).isEmpty)
        XCTAssertTrue(CloudModelRegistry.modelIdsWithVisionSupport(for: provider, defaults: harness.defaults).isEmpty)
        XCTAssertTrue(CloudModelRegistry.modelIdsWithoutToolSupport(for: provider, defaults: harness.defaults).isEmpty)
        XCTAssertNil(CloudProviderCatalog.provider(withId: provider.id, defaults: harness.defaults))
        XCTAssertNil(harness.defaults.string(forKey: CloudModelRegistry.modelsDefaultsKey(for: provider.id)))
    }

    func testBuiltInProviderIsImmutable() throws {
        let harness = makeHarness()
        defer { harness.cleanUp() }

        let synthetic = try XCTUnwrap(harness.store.providers.first { $0.id == "synthetic" })
        XCTAssertTrue(harness.store.isBuiltIn(synthetic))

        XCTAssertThrowsError(
            try harness.store.remove(synthetic),
            "removal refuses a built-in the same way update does, rather than no-oping"
        )
        XCTAssertThrowsError(
            try harness.store.update(synthetic, displayName: "Mine", baseURL: "https://example.com/v1")
        )
        XCTAssertNil(
            harness.defaults.data(forKey: CloudProviderCatalog.customDefaultsKey),
            "a refused mutation must not write a custom list, which is the only thing that could resurrect the built-in under another base"
        )
        XCTAssertEqual(harness.store.providers.map(\.id), ["synthetic"])
    }

    func testUpdateKeepsTheIdentifierAndRevalidatesTheBase() throws {
        let harness = makeHarness()
        defer { harness.cleanUp() }

        let provider = try harness.store.add(
            displayName: "OpenCode Zen",
            identifier: "opencode-zen",
            baseURL: "https://opencode.ai/zen/v1"
        )
        try harness.store.update(
            provider,
            displayName: "OpenCode",
            baseURL: "https://opencode.ai/inference/v1/"
        )

        let updated = try XCTUnwrap(
            CloudProviderCatalog.provider(withId: "opencode-zen", defaults: harness.defaults)
        )
        XCTAssertEqual(updated.id, provider.id)
        XCTAssertEqual(updated.displayName, "OpenCode")
        XCTAssertEqual(updated.openAIBaseURL.absoluteString, "https://opencode.ai/inference/v1")

        XCTAssertThrowsError(
            try harness.store.update(updated, displayName: "OpenCode", baseURL: "http://opencode.ai/v1")
        )
        XCTAssertEqual(
            CloudProviderCatalog.provider(withId: "opencode-zen", defaults: harness.defaults)?
                .openAIBaseURL.absoluteString,
            "https://opencode.ai/inference/v1",
            "a refused edit must not be partially applied"
        )
    }

    func testEditingAProviderKeepsItsPromptCachingChoice() throws {
        let harness = makeHarness()
        defer { harness.cleanUp() }

        let provider = try harness.store.add(
            displayName: "OpenCode Zen",
            identifier: "opencode-zen",
            baseURL: "https://opencode.ai/zen/v1"
        )
        XCTAssertNil(provider.cacheHints, "a new provider starts on the safe default")
        try harness.store.setCacheHints(.openAI, for: provider)

        try harness.store.update(
            provider,
            displayName: "OpenCode",
            baseURL: "https://opencode.ai/zen/v1",
            docsURL: "https://opencode.ai/docs"
        )

        let renamed = try XCTUnwrap(
            CloudProviderCatalog.provider(withId: "opencode-zen", defaults: harness.defaults)
        )
        XCTAssertEqual(
            renamed.cacheHints,
            .openAI,
            "the edit form does not own the caching capability, so a rename must not reset it"
        )
        XCTAssertEqual(renamed.resolvedCacheHints, .openAI)

        // A new base URL goes through the same rebuild: the endpoint moved, the
        // capability the user recorded for it did not.
        try harness.store.update(
            renamed,
            displayName: "OpenCode",
            baseURL: "https://gateway.example.com/v1"
        )
        XCTAssertEqual(
            CloudProviderCatalog.provider(withId: "opencode-zen", defaults: harness.defaults)?.cacheHints,
            .openAI
        )
    }

    func testChangingTheBaseURLDropsThePreviousEndpointsModelFacts() throws {
        let harness = makeHarness()
        defer { harness.cleanUp() }

        let provider = try harness.store.add(
            displayName: "OpenCode Zen",
            identifier: "opencode-zen",
            baseURL: "https://opencode.ai/zen/v1"
        )
        CloudModelRegistry.setRegisteredModelIds(["qwen3-coder"], for: provider, defaults: harness.defaults)
        CloudModelRegistry.setVisionSupport(
            true,
            forModel: "qwen3-coder",
            provider: provider,
            defaults: harness.defaults
        )
        CloudModelRegistry.setModelIdsWithoutToolSupport(
            ["qwen3-coder"],
            for: provider,
            defaults: harness.defaults
        )
        // Prices and windows are facts about the MODEL, typed by hand from the
        // provider's own price sheet. Re-pointing the provider says nothing
        // about either, and neither has a Settings control to retype it.
        CloudContextPolicy.setOverrides(
            ["qwen3-coder": 262_144],
            for: provider.id,
            defaults: harness.defaults
        )
        CloudPricingStore.setPricing(
            ["qwen3-coder": CloudPricing(input: 0.3, output: 1.2)],
            for: provider.id,
            defaults: harness.defaults
        )

        try harness.store.update(
            provider,
            displayName: "OpenCode Zen",
            baseURL: "https://gateway.example.com/v1"
        )

        XCTAssertTrue(
            CloudModelRegistry.registeredModelIds(for: provider, defaults: harness.defaults).isEmpty,
            "the new endpoint has not listed its catalog yet, so nothing from the old one may stay pickable"
        )
        XCTAssertTrue(
            CloudModelRegistry.modelIdsWithVisionSupport(for: provider, defaults: harness.defaults).isEmpty
        )
        XCTAssertTrue(
            CloudModelRegistry.modelIdsWithoutToolSupport(for: provider, defaults: harness.defaults).isEmpty
        )
        XCTAssertNil(harness.defaults.string(forKey: CloudModelRegistry.modelsDefaultsKey(for: provider.id)))
        XCTAssertEqual(
            CloudContextPolicy.overrides(for: provider.id, defaults: harness.defaults),
            ["qwen3-coder": 262_144],
            "a window describes the model, not the endpoint"
        )
        XCTAssertEqual(
            CloudPricingStore.pricing(for: provider.id, defaults: harness.defaults),
            ["qwen3-coder": CloudPricing(input: 0.3, output: 1.2)]
        )

        CloudModelRegistry.setRegisteredModelIds(["gateway-chat"], for: provider, defaults: harness.defaults)
        try harness.store.update(
            provider,
            displayName: "Gateway",
            baseURL: "https://gateway.example.com/v1"
        )
        XCTAssertEqual(
            CloudModelRegistry.registeredModelIds(for: provider, defaults: harness.defaults),
            ["gateway-chat"],
            "renaming a provider says nothing about which server it points at"
        )
    }

    func testACredentialedBaseURLIsRefusedAndNeverPersisted() {
        let harness = makeHarness()
        defer { harness.cleanUp() }

        do {
            _ = try harness.store.add(
                displayName: "OpenCode Zen",
                identifier: "opencode-zen",
                baseURL: "https://user:s3cret@opencode.ai/zen/v1"
            )
            XCTFail("a credential in the URL must never reach storage")
        } catch let error as CloudProviderValidationError {
            XCTAssertEqual(error, .credentialsInBaseURL)
        } catch {
            XCTFail("unexpected error: \(error)")
        }

        XCTAssertFalse(harness.store.providers.contains { $0.id == "opencode-zen" })
        XCTAssertNil(harness.defaults.data(forKey: CloudProviderCatalog.customDefaultsKey))
    }

    func testPerModelRemovalLeavesTheRestOfTheCatalog() throws {
        let harness = makeHarness()
        defer { harness.cleanUp() }

        let provider = try harness.store.add(
            displayName: "OpenCode Zen",
            identifier: "opencode-zen",
            baseURL: "https://opencode.ai/zen/v1"
        )
        CloudModelRegistry.setRegisteredModelIds(
            ["qwen3-coder", "gpt-5.1-codex"],
            for: provider,
            defaults: harness.defaults
        )
        CloudModelRegistry.setVisionSupport(
            true,
            forModel: "gpt-5.1-codex",
            provider: provider,
            defaults: harness.defaults
        )
        CloudModelRegistry.setModelIdsWithoutToolSupport(
            ["gpt-5.1-codex"],
            for: provider,
            defaults: harness.defaults
        )
        harness.defaults.set(
            "gpt-5.1-codex",
            forKey: CloudModelRegistry.debugModelDefaultsKey
        )
        // The two registration facts a cache-aware cloud path added: a
        // per-model window and a per-model price. Removing the model must take
        // them with it, or re-adding the same id silently inherits a context
        // ceiling and a rate somebody set for the model that was removed.
        CloudContextPolicy.setOverrides(
            ["gpt-5.1-codex": 400_000, "qwen3-coder": 262_144],
            for: provider.id,
            defaults: harness.defaults
        )
        CloudPricingStore.setPricing(
            [
                "gpt-5.1-codex": CloudPricing(input: 1.25, cachedInput: 0.125, output: 10),
                "qwen3-coder": CloudPricing(input: 0.3, output: 1.2)
            ],
            for: provider.id,
            defaults: harness.defaults
        )

        CloudModelRegistry.removeRegisteredModel(
            "gpt-5.1-codex",
            for: provider,
            defaults: harness.defaults
        )

        XCTAssertEqual(
            CloudModelRegistry.registeredModelIds(for: provider, defaults: harness.defaults),
            ["qwen3-coder"]
        )
        XCTAssertTrue(
            CloudModelRegistry.modelIdsWithVisionSupport(for: provider, defaults: harness.defaults).isEmpty
        )
        XCTAssertTrue(
            CloudModelRegistry.modelIdsWithoutToolSupport(for: provider, defaults: harness.defaults).isEmpty
        )
        XCTAssertEqual(
            CloudContextPolicy.overrides(for: provider.id, defaults: harness.defaults),
            ["qwen3-coder": 262_144]
        )
        XCTAssertEqual(
            Array(CloudPricingStore.pricing(for: provider.id, defaults: harness.defaults).keys),
            ["qwen3-coder"]
        )
        XCTAssertEqual(
            harness.defaults.string(forKey: CloudModelRegistry.debugModelDefaultsKey),
            "gpt-5.1-codex",
            "the debug hook is one global value that another provider can rely on, so a per-provider prune leaves it alone"
        )
    }

    func testClearingAProviderDropsItsWindowsAndPrices() throws {
        let harness = makeHarness()
        defer { harness.cleanUp() }
        let provider = try harness.store.add(
            displayName: "OpenCode Zen",
            identifier: "opencode-zen",
            baseURL: "https://opencode.ai/zen/v1"
        )
        CloudContextPolicy.setOverrides(
            ["qwen3-coder": 262_144],
            for: provider.id,
            defaults: harness.defaults
        )
        CloudPricingStore.setPricing(
            ["qwen3-coder": CloudPricing(input: 0.3, output: 1.2)],
            for: provider.id,
            defaults: harness.defaults
        )
        CloudModelRegistry.clearProviderState(for: provider, defaults: harness.defaults)
        // A re-added identifier starts clean: neither a window nor a rate from
        // the removed registration may describe a model the operator has not
        // configured again.
        XCTAssertTrue(CloudContextPolicy.overrides(for: provider.id, defaults: harness.defaults).isEmpty)
        XCTAssertTrue(CloudPricingStore.pricing(for: provider.id, defaults: harness.defaults).isEmpty)
    }

    func testPruningOneModelKeepsANeighbouringPriceWithATypo() throws {
        // The reader drops entries it cannot decode, so unregistering a model
        // through read-then-write deleted every OTHER model's hand-typed value
        // that merely had a typo — a silent loss of operator data caused by
        // pruning a different row. The removals edit the stored text directly.
        let harness = makeHarness()
        defer { harness.cleanUp() }
        let provider = try harness.store.add(
            displayName: "OpenCode Zen",
            identifier: "opencode-zen",
            baseURL: "https://opencode.ai/zen/v1"
        )
        CloudModelRegistry.setRegisteredModelIds(
            ["pruned", "kept"],
            for: provider,
            defaults: harness.defaults
        )
        harness.defaults.set(
            #"{"kept":{"input":1,"output":2},"typo":{"input":"cheap","output":2},"pruned":{"input":0.3,"output":1.2}}"#,
            forKey: CloudPricingStore.defaultsKey(for: provider.id)
        )
        harness.defaults.set(
            "kept=64000; typo=twelve; pruned=32000",
            forKey: CloudContextPolicy.defaultsKey(for: provider.id)
        )

        CloudModelRegistry.removeRegisteredModel("pruned", for: provider, defaults: harness.defaults)

        let rawPrices = harness.defaults.string(forKey: CloudPricingStore.defaultsKey(for: provider.id)) ?? ""
        XCTAssertTrue(rawPrices.contains("\"typo\""), rawPrices)
        XCTAssertTrue(rawPrices.contains("\"kept\""), rawPrices)
        XCTAssertFalse(rawPrices.contains("\"pruned\""), rawPrices)
        XCTAssertEqual(
            harness.defaults.string(forKey: CloudContextPolicy.defaultsKey(for: provider.id)),
            "kept=64000; typo=twelve",
            "an entry this build cannot parse still belongs to the operator"
        )
        // The model that WAS removed really is gone from both lists.
        XCTAssertNil(CloudPricingStore.pricing(forModel: "pruned", providerId: provider.id, defaults: harness.defaults))
        XCTAssertNil(CloudContextPolicy.overrides(for: provider.id, defaults: harness.defaults)["pruned"])
    }

    func testChangingTheCacheHintReopensALatchedEndpoint() throws {
        // A latch records that a SERVER refused the field. The Settings control
        // is the operator saying they know better, so leaving the latch in
        // place made the option a silent no-op until relaunch.
        let harness = makeHarness()
        defer { harness.cleanUp() }
        let provider = try harness.store.add(
            displayName: "OpenCode Zen",
            identifier: "opencode-zen",
            baseURL: "https://opencode.ai/zen/v1"
        )
        let endpoint = provider.chatCompletionsURL.absoluteString
        CloudCacheHintMemory.shared.declinePromptCacheKey(endpoint: endpoint)
        XCTAssertFalse(OpenAICloudChatService.sendsCacheKey(
            hints: .openAI,
            sessionKey: "abc123",
            endpoint: endpoint
        ))

        try harness.store.setCacheHints(.openAI, for: provider)

        XCTAssertTrue(OpenAICloudChatService.sendsCacheKey(
            hints: .openAI,
            sessionKey: "abc123",
            endpoint: endpoint
        ))
    }

    func testTheShippingStoreMakesTheProviderLiveEverywhere() throws {
        let suite = "dynamoe.cloud.runtime.tests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defaults.removePersistentDomain(forName: suite)
        defer {
            CloudProviderRuntime.shared.refresh()
            defaults.removePersistentDomain(forName: suite)
        }

        CloudProviderRuntime.shared.refresh(defaults: defaults)
        XCTAssertEqual(CloudProviderRuntime.shared.providers.map(\.id), ["synthetic"])

        let store = CloudProviderStore(
            defaults: defaults,
            secretStore: CloudSecretStore(service: "DRP.DynaMoE.cloud.tests.\(UUID().uuidString)"),
            rescansModels: false
        )

        let provider = try store.add(
            displayName: "OpenCode Zen",
            identifier: "opencode-zen",
            baseURL: "https://opencode.ai/zen/v1"
        )

        XCTAssertEqual(
            CloudProviderRuntime.shared.providers.map(\.id),
            ["synthetic", "opencode-zen"],
            "the shipping store updates the snapshot every other surface reads"
        )
        XCTAssertEqual(CloudProviderRegistry.all.map(\.id), ["synthetic", "opencode-zen"])
        XCTAssertEqual(
            CloudProviderRegistry.provider(withId: "opencode-zen")?.openAIBaseURL.absoluteString,
            "https://opencode.ai/zen/v1"
        )
        XCTAssertEqual(
            CloudModelAddress.parseModelId(
                CloudModelAddress.modelId(providerId: "opencode-zen", providerModelId: "qwen3-coder")
            )?.provider.id,
            "opencode-zen",
            "sentinel parsing resolves through the live runtime, so a freshly added provider must resolve there"
        )

        try store.update(provider, displayName: "OpenCode", baseURL: "https://opencode.ai/zen/v2")
        XCTAssertEqual(
            CloudProviderRuntime.shared.providers.first { $0.id == "opencode-zen" }?.displayName,
            "OpenCode"
        )

        try store.remove(provider)
        XCTAssertEqual(CloudProviderRuntime.shared.providers.map(\.id), ["synthetic"])
        XCTAssertEqual(store.providers.map(\.id), ["synthetic"])
    }

    func testReloadingAnUnchangedListDoesNotRepublish() {
        let harness = makeHarness()
        defer { harness.cleanUp() }

        var publications = 0
        let token = harness.store.objectWillChange.sink { _ in publications += 1 }
        defer { token.cancel() }

        harness.store.reloadFromStorage()

        XCTAssertEqual(
            publications,
            0,
            "a pane reappearing must not fire objectWillChange for a list that did not change"
        )
    }
}

// MARK: - /models decoding

final class CloudModelListDecodingTests: XCTestCase {

    private func ids(_ json: String) -> [String] {
        OpenAICloudChatService.decodeModelIds(from: Data(json.utf8))
    }

    func testOpenAIEnvelope() {
        XCTAssertEqual(
            ids(#"{"object":"list","data":[{"id":"qwen3-coder"},{"id":"grok-code"}]}"#),
            ["qwen3-coder", "grok-code"]
        )
    }

    func testStringArrayInsideData() {
        XCTAssertEqual(ids(#"{"data":["kimi-k2","glm-4.7-free"]}"#), ["kimi-k2", "glm-4.7-free"])
    }

    func testModelsKey() {
        XCTAssertEqual(ids(#"{"models":[{"id":"big-pickle"}]}"#), ["big-pickle"])
    }

    func testTopLevelArray() {
        XCTAssertEqual(ids(#"[{"id":"qwen3-coder"},"grok-code"]"#), ["qwen3-coder", "grok-code"])
    }

    func testDuplicatesEmptyIdsAndWhitespaceAreDropped() {
        XCTAssertEqual(
            ids(#"{"data":[{"id":"a"},{"id":" a "},{"id":""},{},{"id":null}]}"#),
            ["a"]
        )
    }

    func testGarbageReadsAsNoModels() {
        XCTAssertTrue(ids("not json").isEmpty)
        XCTAssertTrue(ids("{}").isEmpty)
        XCTAssertTrue(ids(#"{"data":null}"#).isEmpty)
    }
}
