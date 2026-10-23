//
//  ModelConfigThinkingLevelTests.swift
//  DynaMoE
//
//  Pure-logic coverage for graded thinking-level detection. Temporary
//  snapshot fixtures keep this runnable without any cached model weights.
//

import XCTest
@testable import DynaMoE

final class ModelConfigThinkingLevelTests: XCTestCase {

    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("DynaMoEThinkingLevel-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let directory {
            try? FileManager.default.removeItem(at: directory)
        }
    }

    private func write(_ name: String, _ contents: String) throws {
        try contents.write(to: directory.appendingPathComponent(name), atomically: true, encoding: .utf8)
    }

    private func writeConfig(modelType: String, architecture: String? = nil) throws {
        var json: [String: Any] = ["model_type": modelType]
        if let architecture {
            json["architectures"] = [architecture]
        }
        let data = try JSONSerialization.data(withJSONObject: json)
        try data.write(to: directory.appendingPathComponent("config.json"))
    }

    func testSparkTemplateWithReasoningEffortIsDetected() throws {
        try writeConfig(modelType: "spark2_5", architecture: "Spark2_5ForCausalLM")
        try write("chat_template.jinja", "{%- set reasoning_effort = reasoning_effort|default('high') %}")
        XCTAssertTrue(ModelConfig.supportsThinkingLevels(modelPath: directory.path))
    }

    func testNonSparkTemplateWithReasoningEffortIsNotDetected() throws {
        // Detection is deliberately Spark-only: another family advertising the
        // same token would show a level menu the formatter cannot honor.
        try writeConfig(modelType: "qwen3")
        try write("chat_template.jinja", "reasoning_effort")
        XCTAssertFalse(ModelConfig.supportsThinkingLevels(modelPath: directory.path))
    }

    func testSparkTemplateWithoutReasoningEffortIsNotDetected() throws {
        try writeConfig(modelType: "spark2_5")
        try write("chat_template.jinja", "enable_thinking")
        XCTAssertFalse(ModelConfig.supportsThinkingLevels(modelPath: directory.path))
    }

    func testJinjaTemplateWinsOverTheEmbeddedTokenizerTemplate() throws {
        // A stripped jinja template must degrade to plain on/off even though
        // tokenizer_config.json still embeds the original template.
        try writeConfig(modelType: "spark2_5")
        try write("chat_template.jinja", "enable_thinking")
        try write("tokenizer_config.json", "reasoning_effort")
        XCTAssertFalse(ModelConfig.supportsThinkingLevels(modelPath: directory.path))
    }

    func testTokenizerTemplateIsUsedWhenJinjaIsAbsent() throws {
        try writeConfig(modelType: "spark2_5")
        try write("tokenizer_config.json", "reasoning_effort")
        XCTAssertTrue(ModelConfig.supportsThinkingLevels(modelPath: directory.path))
    }

    func testNilAndEmptyPathsAreNotDetected() {
        XCTAssertFalse(ModelConfig.supportsThinkingLevels(modelPath: nil))
        XCTAssertFalse(ModelConfig.supportsThinkingLevels(modelPath: ""))
    }
}
