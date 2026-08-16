//
//  Item.swift
//  DynaMoE
//
//  Created by Derek Parris on 8/16/26.
//

import Foundation
import SwiftData

@Model
final class Item {
    var timestamp: Date
    
    init(timestamp: Date) {
        self.timestamp = timestamp
    }
}
