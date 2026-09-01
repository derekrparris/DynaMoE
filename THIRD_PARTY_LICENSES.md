# Third-Party Licenses and Acknowledgements

DynaMoE incorporates, links with, or is inspired by several open-source libraries, frameworks, and research projects. This document lists these third-party components along with their respective licenses and copyright notices.

---

## Table of Contents

1. [Rust Dependencies](#rust-dependencies)
   - [safetensors](#safetensors)
   - [tokenizers](#tokenizers)
   - [memmap2](#memmap2)
   - [serde & serde_json](#serde--serde_json)
   - [uniffi-rs](#uniffi-rs)
   - [libc](#libc)
2. [Architectural Inspirations & Open-Source Projects](#architectural-inspirations--open-source-projects)
   - [JetSpec](#jetspec)
   - [Flash-MoE](#flash-moe)
   - [Colibri](#colibri)
   - [Osaurus](#osaurus)
3. [Apple Frameworks & Swift Standard Library](#apple-frameworks--swift-standard-library)

---

## Rust Dependencies

### safetensors
* **Repository:** https://github.com/huggingface/safetensors
* **Copyright:** Copyright (c) HuggingFace Inc.
* **License:** Apache License 2.0

```text
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

### tokenizers
* **Repository:** https://github.com/huggingface/tokenizers
* **Copyright:** Copyright (c) HuggingFace Inc.
* **License:** Apache License 2.0

```text
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

### memmap2
* **Repository:** https://github.com/DanBurkert/memmap2-rs
* **Copyright:** 
  - Copyright (c) 2020 Dan Burkert
  - Copyright (c) 2021-2023 Alex Chi
  - Copyright (c) 2023-2024 Yash Dani
* **License:** MIT License / Apache License 2.0

```text
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
```

---

### serde & serde_json
* **Repository:** https://github.com/serde-rs/serde
* **Copyright:** Copyright (c) 2014-2024 Erick Tryzelaar and David Tolnay
* **License:** MIT License / Apache License 2.0

```text
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
```

---

### uniffi-rs
* **Repository:** https://github.com/mozilla/uniffi-rs
* **Copyright:** Copyright (c) Mozilla Foundation
* **License:** Mozilla Public License 2.0 (MPL-2.0)

```text
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
```

---

### libc
* **Repository:** https://github.com/rust-lang/libc
* **Copyright:** Copyright (c) 2014-2024 The Rust Project Developers
* **License:** MIT License / Apache License 2.0

```text
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
```

---

## Architectural Inspirations & Open-Source Projects

### JetSpec
* **Paper / Research:** *JetSpec: Accelerating LLM Decoding via Parallel Tree Drafting and Verification* (Hao AI Lab / UC San Diego — [arXiv:2606.18394](https://arxiv.org/html/2606.18394v2))
* **Description:** Speculative decoding architecture utilizing parallel causal draft heads and tree-causal attention masking for multi-token acceptance per forward step.

---

### Flash-MoE
* **Repository:** https://github.com/danveloper/flash-moe
* **Author:** Dan Woods
* **License:** Apache License 2.0
* **Description:** Pioneered contiguous binary layer repackaging (`packed_experts/layer_XX.bin`) for high-throughput asynchronous POSIX file streaming directly into shared Metal GPU buffers.

```text
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

### Colibri
* **Repository:** https://github.com/JustVugg/colibri
* **Author:** JustVugg
* **License:** MIT License / Apache License 2.0
* **Description:** Pioneering exploration of off-disk model weight execution and memory-mapped inference.

---

### Osaurus
* **Repository:** https://github.com/osaurus-ai/osaurus
* **Author:** Osaurus AI Team
* **License:** MIT License
* **Description:** Local AI harness written in Swift for macOS.

```text
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
```

---

## Apple Frameworks & Swift Standard Library

DynaMoE is built with the Apple Swift programming language and utilizes Apple system frameworks:
* **Metal, MetalKit & Metal Performance Shaders (MPS):** Provided under the Apple SDK and macOS Developer License Agreements.
* **Apple Accelerate (`vDSP`, `vecLib`):** Hardware-vectorized signal processing and linear algebra routines provided under macOS SDK terms.
* **Swift Standard Library:** Licensed under the Apache License 2.0 with Runtime Library Exception.
