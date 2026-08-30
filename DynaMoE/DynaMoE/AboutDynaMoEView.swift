//
//  AboutDynaMoEView.swift
//  DynaMoE
//
//  Created by Derek Parris on 8/30/26.
//

import SwiftUI
import Metal

struct ThirdPartyLicenseItem: Identifiable {
    let id = UUID()
    let name: String
    let category: String
    let licenseType: String
    let copyrightNotice: String
    let url: String
    let fullLicenseText: String
}

struct AboutDynaMoEView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var showingApacheLicense: Bool = false
    @State private var isCopiedInfo: Bool = false

    private let appVersion = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0.0"
    private let buildNumber = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "2026.1"

    private let thirdPartyLicenses: [ThirdPartyLicenseItem] = [
        ThirdPartyLicenseItem(
            name: "Hugging Face Tokenizers",
            category: "Core Tokenization Engine",
            licenseType: "Apache-2.0",
            copyrightNotice: "Copyright © 2019-2026 Hugging Face, Inc. and contributors.",
            url: "https://github.com/huggingface/tokenizers",
            fullLicenseText: """
            Licensed under the Apache License, Version 2.0 (the "License");
            you may not use this file except in compliance with the License.
            You may obtain a copy of the License at

                http://www.apache.org/licenses/LICENSE-2.0

            Unless required by applicable law or agreed to in writing, software
            distributed under the License is distributed on an "AS IS" BASIS,
            WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
            See the License for the specific language governing permissions and
            limitations under the License.
            """
        ),
        ThirdPartyLicenseItem(
            name: "Hugging Face SafeTensors",
            category: "Zero-Copy Tensor Parsing",
            licenseType: "Apache-2.0",
            copyrightNotice: "Copyright © 2022-2026 Hugging Face, Inc. and contributors.",
            url: "https://github.com/huggingface/safetensors",
            fullLicenseText: """
            Licensed under the Apache License, Version 2.0 (the "License");
            you may not use this file except in compliance with the License.
            You may obtain a copy of the License at

                http://www.apache.org/licenses/LICENSE-2.0

            Unless required by applicable law or agreed to in writing, software
            distributed under the License is distributed on an "AS IS" BASIS,
            WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
            See the License for the specific language governing permissions and
            limitations under the License.
            """
        ),
        ThirdPartyLicenseItem(
            name: "Mozilla UniFFI (uniffi-rs)",
            category: "Multi-Language Rust FFI Bindings",
            licenseType: "MPL-2.0 / Apache-2.0",
            copyrightNotice: "Copyright © Mozilla Corporation and contributors.",
            url: "https://github.com/mozilla/uniffi-rs",
            fullLicenseText: """
            This Source Code Form is subject to the terms of the Mozilla Public
            License, v. 2.0. If a copy of the MPL was not distributed with this
            file, You can obtain one at https://mozilla.org/MPL/2.0/.

            Alternatively, portions of this code are available under the Apache
            License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
            """
        ),
        ThirdPartyLicenseItem(
            name: "Serde & Serde JSON",
            category: "Data Serialization Framework",
            licenseType: "MIT / Apache-2.0",
            copyrightNotice: "Copyright © 2017-2026 David Tolnay and contributors.",
            url: "https://github.com/serde-rs/serde",
            fullLicenseText: """
            Permission is hereby granted, free of charge, to any person obtaining a copy
            of this software and associated documentation files (the "Software"), to deal
            in the Software without restriction, including without limitation the rights
            to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
            copies of the Software, and to permit persons to whom the Software is
            furnished to do so, subject to the following conditions:

            The above copyright notice and this permission notice shall be included in all
            copies or substantial portions of the Software.

            THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
            IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
            FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
            AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
            LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
            OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
            SOFTWARE.
            """
        ),
        ThirdPartyLicenseItem(
            name: "memmap2",
            category: "Cross-Platform Memory Mapping",
            licenseType: "MIT / Apache-2.0",
            copyrightNotice: "Copyright © 2020 Dan Burkert, Colin Rofls, and contributors.",
            url: "https://github.com/yoshuawuyts/memmap2-rs",
            fullLicenseText: """
            Permission is hereby granted, free of charge, to any person obtaining a copy
            of this software and associated documentation files (the "Software"), to deal
            in the Software without restriction, including without limitation the rights
            to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
            copies of the Software, and to permit persons to whom the Software is
            furnished to do so, subject to the following conditions:

            The above copyright notice and this permission notice shall be included in all
            copies or substantial portions of the Software.

            THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
            IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
            FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
            AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
            LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
            OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
            SOFTWARE.
            """
        )
    ]

    var body: some View {
        VStack(spacing: 0) {
            // Header / App Hero
            VStack(spacing: 12) {
                HStack(spacing: 16) {
                    // App Icon
                    ZStack {
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .fill(
                                LinearGradient(
                                    colors: [Color.purple.opacity(0.8), Color.indigo.opacity(0.9), Color.blue.opacity(0.8)],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                )
                            )
                            .frame(width: 68, height: 68)
                            .shadow(color: Color.purple.opacity(0.35), radius: 8, x: 0, y: 4)

                        Image(systemName: "circle.hexagongrid.circle.fill")
                            .font(.system(size: 38))
                            .foregroundColor(.white)
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        HStack(alignment: .firstTextBaseline, spacing: 8) {
                            Text("DynaMoE")
                                .font(.system(size: 24, weight: .bold, design: .rounded))
                            Text("v\(appVersion)")
                                .font(.system(size: 13, weight: .semibold, design: .monospaced))
                                .foregroundColor(.secondary)
                            Text("(Build \(buildNumber))")
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundColor(.secondary.opacity(0.7))
                        }

                        Text("High-Performance MoE Inference Engine for Apple Silicon")
                            .font(.system(size: 12, weight: .medium))
                            .foregroundColor(.secondary)

                        Text("Copyright © 2026 Derek Parris. All rights reserved.")
                            .font(.system(size: 11))
                            .foregroundColor(.secondary.opacity(0.8))
                    }

                    Spacer()
                }
                .padding(.horizontal, 24)
                .padding(.top, 20)
                .padding(.bottom, 12)

                // Quick Action Links
                HStack(spacing: 10) {
                    Button(action: {
                        if let url = URL(string: "https://github.com/derekrparris/DynaMoE") {
                            NSWorkspace.shared.open(url)
                        }
                    }) {
                        Label("GitHub Repository", systemImage: "link")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .buttonStyle(.bordered)

                    Button(action: {
                        showingApacheLicense = true
                    }) {
                        Label("Apache 2.0 License", systemImage: "doc.text")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .buttonStyle(.bordered)

                    Button(action: copySystemSummary) {
                        Label(isCopiedInfo ? "Copied!" : "Copy System Info", systemImage: isCopiedInfo ? "checkmark" : "doc.on.doc")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .buttonStyle(.bordered)

                    Spacer()
                }
                .padding(.horizontal, 24)
                .padding(.bottom, 14)
            }
            .background(Color(NSColor.controlBackgroundColor).opacity(0.6))

            Divider()

            // Scrollable Content
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    // Section 1: Engine Architecture & Platform
                    VStack(alignment: .leading, spacing: 10) {
                        Label("Engine Architecture", systemImage: "cpu.fill")
                            .font(.system(size: 14, weight: .bold))

                        VStack(spacing: 8) {
                            HStack {
                                Text("Metal Compute Device")
                                    .font(.system(size: 12))
                                    .foregroundColor(.secondary)
                                Spacer()
                                Text(MTLCreateSystemDefaultDevice()?.name ?? "Apple Silicon GPU")
                                    .font(.system(size: 12, weight: .semibold, design: .monospaced))
                            }
                            Divider()
                            HStack {
                                Text("Core Runtime")
                                    .font(.system(size: 12))
                                    .foregroundColor(.secondary)
                                Spacer()
                                Text("Rust Engine (libdynamoe_core) + Apple Metal Shaders")
                                    .font(.system(size: 12, weight: .medium))
                            }
                            Divider()
                            HStack {
                                Text("Routing & Storage Pipeline")
                                    .font(.system(size: 12))
                                    .foregroundColor(.secondary)
                                Spacer()
                                Text("Zero-Copy SSD Virtual Paging & Predictive Routing")
                                    .font(.system(size: 12, weight: .medium))
                            }
                        }
                        .padding(14)
                        .background(Color(NSColor.controlBackgroundColor).opacity(0.4))
                        .cornerRadius(10)
                        .overlay(
                            RoundedRectangle(cornerRadius: 10)
                                .stroke(Color.primary.opacity(0.08), lineWidth: 1)
                        )
                    }

                    // Section 2: Third-Party Open Source Software (OSS)
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Label("Third-Party Open-Source Software", systemImage: "shippingbox.fill")
                                .font(.system(size: 14, weight: .bold))
                            Spacer()
                            Text("\(thirdPartyLicenses.count) Bundled Libraries")
                                .font(.system(size: 11))
                                .foregroundColor(.secondary)
                        }

                        Text("DynaMoE incorporates statically linked open-source libraries. Their respective copyright notices, licenses, and conditions are reproduced below in accordance with their terms:")
                            .font(.system(size: 11))
                            .foregroundColor(.secondary)
                            .lineSpacing(2)

                        VStack(spacing: 8) {
                            ForEach(thirdPartyLicenses) { item in
                                DisclosureGroup {
                                    VStack(alignment: .leading, spacing: 8) {
                                        Text(item.copyrightNotice)
                                            .font(.system(size: 11, weight: .semibold))
                                            .foregroundColor(.primary)

                                        ScrollView(.horizontal, showsIndicators: false) {
                                            Text(item.fullLicenseText)
                                                .font(.system(size: 10, design: .monospaced))
                                                .foregroundColor(.secondary)
                                                .padding(8)
                                                .background(Color(NSColor.textBackgroundColor))
                                                .cornerRadius(6)
                                        }

                                        HStack {
                                            Button(action: {
                                                if let url = URL(string: item.url) {
                                                    NSWorkspace.shared.open(url)
                                                }
                                            }) {
                                                Label("Source Repository", systemImage: "arrow.up.right.square")
                                                    .font(.system(size: 10))
                                            }
                                            .buttonStyle(.link)

                                            Spacer()

                                            Button(action: {
                                                NSPasteboard.general.clearContents()
                                                NSPasteboard.general.setString("\(item.name)\n\(item.copyrightNotice)\n\n\(item.fullLicenseText)", forType: .string)
                                            }) {
                                                Label("Copy License", systemImage: "doc.on.doc")
                                                    .font(.system(size: 10))
                                            }
                                            .buttonStyle(.plain)
                                            .foregroundColor(.secondary)
                                        }
                                    }
                                    .padding(.vertical, 6)
                                } label: {
                                    HStack(spacing: 8) {
                                        Text(item.name)
                                            .font(.system(size: 12, weight: .semibold))
                                        Text("(\(item.category))")
                                            .font(.system(size: 11))
                                            .foregroundColor(.secondary)
                                        Spacer()
                                        Text(item.licenseType)
                                            .font(.system(size: 10, weight: .bold, design: .monospaced))
                                            .padding(.horizontal, 6)
                                            .padding(.vertical, 2)
                                            .background(Color.purple.opacity(0.12))
                                            .foregroundColor(.purple)
                                            .cornerRadius(4)
                                    }
                                }
                                .padding(10)
                                .background(Color(NSColor.controlBackgroundColor).opacity(0.3))
                                .cornerRadius(8)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 8)
                                        .stroke(Color.primary.opacity(0.06), lineWidth: 1)
                                )
                            }
                        }
                    }

                    // Section 3: Trademarks & Platform Disclaimers
                    VStack(alignment: .leading, spacing: 8) {
                        Label("Trademarks & Disclaimers", systemImage: "shield.lefthalf.filled")
                            .font(.system(size: 13, weight: .bold))

                        VStack(alignment: .leading, spacing: 8) {
                            Text("• macOS, Metal, Metal Performance Shaders, Apple Silicon, and SwiftUI are trademarks of Apple Inc., registered in the U.S. and other countries.")
                                .font(.system(size: 11))
                                .foregroundColor(.secondary)

                            Text("• Supported model weights, architectures, and tokenizers (including DeepSeek-R1, Qwen 2.5/3.5, Ornith 1.5, Nanbeige 2, and others) are the property of their respective creators and research labs, and are subject to their original model licenses and acceptable use policies.")
                                .font(.system(size: 11))
                                .foregroundColor(.secondary)

                            Text("• DynaMoE is an independent open-source inference engine and is not affiliated with or endorsed by Apple Inc., Hugging Face, or the model creators.")
                                .font(.system(size: 11))
                                .foregroundColor(.secondary)
                        }
                        .padding(12)
                        .background(Color(NSColor.controlBackgroundColor).opacity(0.3))
                        .cornerRadius(8)
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.primary.opacity(0.06), lineWidth: 1)
                        )
                    }
                }
                .padding(24)
            }
        }
        .frame(minWidth: 620, idealWidth: 680, minHeight: 520, idealHeight: 580)
        .background(Color(NSColor.windowBackgroundColor))
        .sheet(isPresented: $showingApacheLicense) {
            ApacheLicenseSheetView()
        }
    }

    private func copySystemSummary() {
        let gpu = MTLCreateSystemDefaultDevice()?.name ?? "Apple Silicon GPU"
        let osVersion = ProcessInfo.processInfo.operatingSystemVersionString
        let text = """
        DynaMoE v\(appVersion) (Build \(buildNumber))
        Copyright © 2026 Derek Parris. All rights reserved.
        Licensed under the Apache License, Version 2.0
        Metal Device: \(gpu)
        OS: macOS \(osVersion)
        """
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        withAnimation {
            isCopiedInfo = true
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            withAnimation {
                isCopiedInfo = false
            }
        }
    }
}

// MARK: - Apache 2.0 License Full Sheet
struct ApacheLicenseSheetView: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Apache License, Version 2.0")
                    .font(.headline)
                    .fontWeight(.bold)
                Spacer()
                Button("Done") {
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
            }
            .padding(16)
            .background(Color(NSColor.controlBackgroundColor).opacity(0.5))

            Divider()

            ScrollView {
                Text(apacheLicenseText)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(.secondary)
                    .lineSpacing(3)
                    .padding(20)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            Divider()

            HStack {
                Button(action: {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(apacheLicenseText, forType: .string)
                }) {
                    Label("Copy License Text", systemImage: "doc.on.doc")
                }
                .buttonStyle(.bordered)

                Spacer()

                Button("Close") {
                    dismiss()
                }
            }
            .padding(12)
            .background(Color(NSColor.controlBackgroundColor).opacity(0.5))
        }
        .frame(width: 580, height: 480)
        .background(Color(NSColor.windowBackgroundColor))
    }

    private var apacheLicenseText: String {
        """
        Apache License
        Version 2.0, January 2004
        http://www.apache.org/licenses/

        TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

        1. Definitions.
        "License" shall mean the terms and conditions for use, reproduction, and distribution as defined by Sections 1 through 9 of this document.
        "Licensor" shall mean the copyright owner or entity authorized by the copyright owner that is granting the License.
        "Legal Entity" shall mean the union of the acting entity and all other entities that control, are controlled by, or are under common control with that entity.
        "You" (or "Your") shall mean an individual or Legal Entity exercising permissions granted by this License.
        "Source" form shall mean the preferred form for making modifications, including but not limited to software source code, documentation source, and configuration files.
        "Object" form shall mean any form resulting from mechanical transformation or translation of a Source form, including but not limited to compiled object code, generated documentation, and conversions to other media types.
        "Work" shall mean the work of authorship, whether in Source or Object form, made available under the License.
        "Derivative Works" shall mean any work, whether in Source or Object form, that is based on (or derived from) the Work.

        2. Grant of Copyright License.
        Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare Derivative Works of, publicly display, publicly perform, sublicense, and distribute the Work and such Derivative Works in Source or Object form.

        3. Grant of Patent License.
        Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Work.

        4. Redistribution.
        You may reproduce and distribute copies of the Work or Derivative Works thereof in any medium, with or without modifications, and in Source or Object form, provided that You meet the following conditions:
        (a) You must give any other recipients of the Work or Derivative Works a copy of this License; and
        (b) You must cause any modified files to carry prominent notices stating that You changed the files; and
        (c) You must retain, in the Source form of any Derivative Works that You distribute, all copyright, patent, trademark, and attribution notices; and
        (d) If the Work includes a "NOTICE" text file, you must include a readable copy of the attribution notices.

        7. Disclaimer of Warranty.
        Unless required by applicable law or agreed to in writing, Licensor provides the Work on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

        8. Limitation of Liability.
        In no event and under no legal theory shall any Contributor be liable to You for damages, including direct, indirect, special, incidental, or consequential damages.

        Copyright 2026 Derek Parris
        """
    }
}
