//
//  DynaMoE-Bridging-Header.h
//  Exposes the generated UniFFI C API to Swift.
//
//  Reached via SWIFT_OBJC_BRIDGING_HEADER = "DynaMoE-Bridging-Header.h", which
//  resolves relative to SRCROOT (the folder holding DynaMoE.xcodeproj), so this
//  file must live at DynaMoE/DynaMoE-Bridging-Header.h.
//
//  dynamoe_coreFFI.h is not in this folder: it is regenerated into
//  GeneratedFFI/ by the "Build Rust Engine & FFI" phase, which is on
//  HEADER_SEARCH_PATHS. If this file goes missing the build fails with
//  "bridging header ... cannot be found" — see .gitignore, which keeps it
//  tracked on purpose.
//

#ifndef DYNAMOE_BRIDGING_HEADER_H
#define DYNAMOE_BRIDGING_HEADER_H

#import "dynamoe_coreFFI.h"

#endif /* DYNAMOE_BRIDGING_HEADER_H */
