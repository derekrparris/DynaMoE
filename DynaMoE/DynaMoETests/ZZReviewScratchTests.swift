import XCTest
import SwiftUI
@testable import DynaMoE

@MainActor
final class ZZReviewScratchTests: XCTestCase {
    private func rendered<V: View>(_ view: V, width: CGFloat) -> CGSize {
        let renderer = ImageRenderer(content: view.frame(width: width).environmentObject(AppZoomManager.shared))
        renderer.scale = 1
        return renderer.nsImage?.size ?? .zero
    }

    func testBadgeWrapping() {
        var call = ToolCallRecord(name: "mcp__testserver__very_long_tool_name_indeed")
        call.status = .success
        call.executionDurationSeconds = 12.34
        print("BADGE1 in1152=\(rendered(StepNodeBadge(stepIndex: 1, call: call), width: 1152))")
        print("BADGE1 in120=\(rendered(StepNodeBadge(stepIndex: 1, call: call), width: 120))")
    }

    func testWideTableCellOverflow() {
        let b64 = String(repeating: "QUJD", count: 400)
        let table = MarkdownTableData(headers: ["key", "blob"], alignments: [.leading, .leading], rows: [["a", b64]])
        print("WIDECELL in560=\(rendered(MarkdownTableView(table: table), width: 560))")
        // and inside the markdown view the transcript actually uses
        let md = MarkdownMessageView(content: "", preparsedBlocks: [.table(table)])
        print("WIDECELL md in560=\(rendered(md, width: 560))")
    }

    func testFingerprintCountCost() {
        let big = String(repeating: "abcdefghij", count: 6_000) // 60k chars
        let t0 = CFAbsoluteTimeGetCurrent()
        var sink = 0
        for _ in 0..<2_000 { sink += big.count }
        let per = (CFAbsoluteTimeGetCurrent() - t0) * 1_000_000 / 2_000
        print("FINGERPRINT String.count us=\(String(format: "%.2f", per)) sink=\(sink)")

        // what a COW short-circuit costs, for comparison
        let other = big
        let t1 = CFAbsoluteTimeGetCurrent()
        var same = 0
        for _ in 0..<2_000 { if other == big { same += 1 } }
        print("FINGERPRINT String== us=\(String(format: "%.3f", (CFAbsoluteTimeGetCurrent() - t1) * 1_000_000 / 2_000)) same=\(same)")
    }
}
