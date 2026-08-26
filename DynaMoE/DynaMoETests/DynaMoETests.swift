//
//  DynaMoETests.swift
//  DynaMoETests
//
//  Created by Derek Parris on 8/16/26.
//

import XCTest

final class DynaMoETests: XCTestCase {

    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }

    func testOrnithForward() throws {
        let snapshotDir = "/Users/derekparris/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-35B-A3B-FP8/snapshots/0e048080ccd0ccf4296bfea5638036c196dccc0c"
        guard FileManager.default.fileExists(atPath: snapshotDir) else {
            print("Snapshot not found, skipping.")
            return
        }
        print("Running testOrnithForward...")
    }
}
