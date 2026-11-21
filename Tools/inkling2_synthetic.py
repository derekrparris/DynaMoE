#!/usr/bin/env python3
"""Write a small, structurally complete Inkling 2 Mini checkpoint.

Why: the published checkpoint makes every hypothesis cost a full 48-visit
scalar forward (minutes). A checkpoint at toy widths, loaded by the SAME two
implementations -- the pinned Transformers reference and the Swift host graph
-- turns each hypothesis into seconds, and it is what the support plan's
Phase 0 asks for ("unusual-but-valid small dimensions").

The fixture stores the experts in the CHECKPOINT's own packed layouts with
random-but-valid bytes, so each side runs its OWN dequantizer: the reference's
conversion and DynaMoE's catalog/source. That is the same situation as the
real checkpoint, which is the point -- a fixture that pre-dequantized the
weights would test neither reader.

    /tmp/inkling_venv/bin/python Tools/inkling2_synthetic.py <output-dir>
"""
import json, os, struct, sys
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/inkling2_synth"

# Geometry: 12 layers, a 6-layer core looped three times (24 visits), one full
# layer per six, two experts of top-1, and widths that keep every scale stride
# an exact integer (hidden 64, intermediate 32 for 16-input NVFP4 blocks).
HIDDEN, LAYERS, HEADS, KV, HEAD_DIM = 64, 12, 4, 2, 16
D_REL, REL_EXTENT, WINDOW = 4, 8, 8
EXPERTS, TOP_K, INTER, GATE_RANK = 2, 1, 32, 2
VOCAB, UNPADDED = 128, 120
BLOCK, CORE_START, CORE_END, PASSES = 6, 6, 12, 3
FULL_LAYERS = {5, 11}

rng = np.random.default_rng(20261121)

def to_bf16(values):
    """Round to the BF16 grid and return the STORED 16-bit words.

    The words are what a shard holds, so they are kept as `uint16` end to end:
    converting back to float32 and letting `put` re-cast would reinterpret the
    bit pattern as an integer and store garbage (a whole embedding of zeros was
    the first symptom).
    """
    a = np.asarray(values, dtype=np.float32)
    bits = a.view(np.uint32)
    lsb = (bits >> 16) & 1
    rounded = ((bits + 0x7FFF + lsb) & 0xFFFF0000).astype(np.uint32)
    # The stored word is the TOP half. Reinterpreting the 32-bit value as two
    # uint16 on a little-endian machine yields [low, high], i.e. swapped.
    return (rounded >> 16).astype(np.uint16)

def to_f32(values):
    return np.asarray(values, dtype=np.float32)

tensors = {}
def put(name, array, dtype):
    tensors[name] = (np.ascontiguousarray(array), dtype)

def rnd(*shape, scale=0.2):
    return rng.normal(0, scale, size=shape)

E4M3_CODES = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                      16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,
                      32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,
                      48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,
                      64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,
                      80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,
                      96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,
                      112,113,114,115,116,117,118,119,120,121,122,123,124,125,126], dtype=np.uint8)

# ---- Model level -----------------------------------------------------------
put("model.llm.embed.weight", to_bf16(rnd(VOCAB, HIDDEN)), "BF16")
put("model.llm.embed_norm.weight", to_bf16(1 + rnd(HIDDEN, scale=0.1)), "BF16")
put("model.llm.norm.weight", to_bf16(1 + rnd(HIDDEN, scale=0.1)), "BF16")
put("model.llm.unembed.weight", to_bf16(rnd(VOCAB, HIDDEN)), "BF16")
put("model.llm.attn_res.output_query", to_f32(rnd(HIDDEN)), "F32")
put("model.llm.middle_cycle_cross_pass_queries", to_f32(rnd(2 * (CORE_END-CORE_START), HIDDEN)), "F32")
put("model.llm.middle_cycle_cross_pass_kv_queries", to_f32(rnd(2 * (CORE_END-CORE_START), HIDDEN)), "F32")
for layer in range(1, LAYERS):
    rows = 3 if layer in FULL_LAYERS else 1
    put(f"model.llm.attn_res.attn_queries.{layer-1}", to_f32(rnd(rows, HIDDEN) if rows > 1 else rnd(HIDDEN)), "F32")
for layer in range(LAYERS):
    put(f"model.llm.attn_res.mlp_queries.{layer}", to_f32(rnd(HIDDEN)), "F32")

# ---- Per layer -------------------------------------------------------------
for layer in range(LAYERS):
    p = f"model.llm.layers.{layer}"
    full = layer in FULL_LAYERS
    put(f"{p}.attn.wq_du.weight", to_bf16(rnd(HEADS*HEAD_DIM, HIDDEN)), "BF16")
    put(f"{p}.attn.wk_dv.weight", to_bf16(rnd(KV*HEAD_DIM, HIDDEN)), "BF16")
    put(f"{p}.attn.wv_dv.weight", to_bf16(rnd(KV*HEAD_DIM, HIDDEN)), "BF16")
    put(f"{p}.attn.wr_du.weight", to_bf16(rnd(HEADS*D_REL, HIDDEN)), "BF16")
    put(f"{p}.attn.wo_ud.weight", to_bf16(rnd(HIDDEN, HEADS*HEAD_DIM)), "BF16")
    put(f"{p}.attn.gate_attn.weight", to_bf16(rnd(HEADS*HEAD_DIM, HIDDEN)), "BF16")
    put(f"{p}.attn.attn_sink_proj.weight", to_bf16(rnd(HEADS, HIDDEN)), "BF16")
    put(f"{p}.attn.rel_logits_proj.proj", to_bf16(rnd(D_REL, REL_EXTENT)), "BF16")
    put(f"{p}.attn.q_l2_clip.scale", to_bf16(1 + rnd(HEADS, HEAD_DIM, scale=0.05)), "BF16")
    put(f"{p}.attn.rel_l2_clip.scale", to_bf16(1 + rnd(HEADS, D_REL, scale=0.05)), "BF16")
    put(f"{p}.attn_sconv.weight", to_bf16(rnd(HIDDEN, 1, 4, scale=0.3)), "BF16")
    put(f"{p}.mlp_sconv.weight", to_bf16(rnd(HIDDEN, 1, 4, scale=0.3)), "BF16")
    put(f"{p}.mlp_norm.weight", to_bf16(1 + rnd(HIDDEN, scale=0.1)), "BF16")
    put(f"{p}.mlp.gate.weight", to_f32(rnd(EXPERTS, HIDDEN)), "F32")
    put(f"{p}.mlp.gate.bias", to_f32(rnd(EXPERTS, scale=0.1)), "F32")
    if full:
        for name in ("q", "k", "v"):
            put(f"{p}.{name}_input_norm.weight", to_bf16(1 + rnd(HIDDEN, scale=0.1)), "BF16")
        for name in ("k", "v"):
            put(f"{p}.attn.{name}_sconv.static_w", to_bf16(rnd(4, KV*HEAD_DIM, scale=0.3)), "BF16")
            put(f"{p}.attn.{name}_sconv.weight_proj", to_bf16(rnd(KV*4, HIDDEN)), "BF16")
    else:
        put(f"{p}.attn_norm.weight", to_bf16(1 + rnd(HIDDEN, scale=0.1)), "BF16")
        put(f"{p}.attn.k_sconv.weight", to_bf16(rnd(KV*HEAD_DIM, 1, 4, scale=0.3)), "BF16")
        put(f"{p}.attn.v_sconv.weight", to_bf16(rnd(KV*HEAD_DIM, 1, 4, scale=0.3)), "BF16")
    if CORE_START <= layer < CORE_END:
        for part in ("attn", "k", "v", "mlp"):
            put(f"{p}.controllers.{part}.down_weight", to_bf16(rnd(GATE_RANK, 2*HIDDEN)), "BF16")
            put(f"{p}.controllers.{part}.forget_weight", to_bf16(rnd(HIDDEN, GATE_RANK)), "BF16")
            put(f"{p}.controllers.{part}.gain_weight", to_bf16(rnd(HIDDEN, GATE_RANK)), "BF16")

    # Experts. Layers 0..9 NVFP4 (U8 values, E4M3 scales, amax pair), 10..11
    # MXFP8 (E4M3 values, U8 E8M0 scales) -- both layouts the real split uses.
    mxfp8 = layer >= 10
    gateUpRows = 2 * INTER
    if not mxfp8:
        put(f"{p}.mlp.experts.w13_weight", rng.integers(0, 256, size=(EXPERTS, gateUpRows, HIDDEN//2), dtype=np.uint8), "U8")
        put(f"{p}.mlp.experts.w13_weight_scale", rng.choice(E4M3_CODES, size=(EXPERTS, gateUpRows, HIDDEN//16)), "F8_E4M3")
        put(f"{p}.mlp.experts.w13_weight_amax", to_f32(rng.uniform(0.1, 0.4, size=EXPERTS)), "F32")
        put(f"{p}.mlp.experts.w13_weight_outer_scale_denominator", to_f32([2688.0]), "F32")
        put(f"{p}.mlp.experts.w2_weight", rng.integers(0, 256, size=(EXPERTS, HIDDEN, INTER//2), dtype=np.uint8), "U8")
        put(f"{p}.mlp.experts.w2_weight_scale", rng.choice(E4M3_CODES, size=(EXPERTS, HIDDEN, INTER//16)), "F8_E4M3")
        put(f"{p}.mlp.experts.w2_weight_amax", to_f32(rng.uniform(0.1, 0.4, size=EXPERTS)), "F32")
        put(f"{p}.mlp.experts.w2_weight_outer_scale_denominator", to_f32([2688.0]), "F32")
    else:
        put(f"{p}.mlp.experts.w13_weight", rng.choice(E4M3_CODES, size=(EXPERTS, gateUpRows, HIDDEN)), "F8_E4M3")
        put(f"{p}.mlp.experts.w13_weight_scale", rng.integers(120, 130, size=(EXPERTS, gateUpRows, HIDDEN//32), dtype=np.uint8), "U8")
        put(f"{p}.mlp.experts.w2_weight", rng.choice(E4M3_CODES, size=(EXPERTS, HIDDEN, INTER)), "F8_E4M3")
        put(f"{p}.mlp.experts.w2_weight_scale", rng.integers(120, 130, size=(EXPERTS, HIDDEN, INTER//32), dtype=np.uint8), "U8")

# ---- Vision tower ----------------------------------------------------------
# One or two pre-norm blocks and the projector, at widths the config derives:
# the tower's forward is a later phase, but its TENSORS have to be complete or
# `supportsVision` would advertise a tower the load cannot open.
VISION_WIDTH, VISION_HEADS, VISION_BLOCKS = 32, 2, 1
VISION_PROJ, POSEMB_GRID = 64, 4
merge = 2
put("model.visual.backbone.embeddings.class_embedding", to_bf16(rnd(VISION_WIDTH, scale=0.1)), "BF16")
put("model.visual.backbone.embeddings.cls_position_embedding", to_bf16(rnd(VISION_WIDTH, scale=0.1)), "BF16")
put("model.visual.backbone.embeddings.patch_embedding.weight",
    to_bf16(rnd(VISION_WIDTH, merge * 20 * merge * 20 * 3)), "BF16")
put("model.visual.backbone.embeddings.position_embedding",
    to_bf16(rnd(POSEMB_GRID * POSEMB_GRID, VISION_WIDTH, scale=0.1)), "BF16")
put("model.visual.backbone.ln_pre.weight", to_bf16(1 + rnd(VISION_WIDTH, scale=0.1)), "BF16")
put("model.visual.backbone.ln_pre.bias", to_bf16(rnd(VISION_WIDTH, scale=0.05)), "BF16")
put("model.visual.proj_in.weight", to_bf16(rnd(VISION_PROJ, merge * merge * VISION_WIDTH)), "BF16")
put("model.visual.proj_out.weight", to_bf16(rnd(HIDDEN, VISION_PROJ)), "BF16")
for b in range(VISION_BLOCKS):
    p = f"model.visual.backbone.layers.{b}"
    for stage in ("ln_1", "ln_2"):
        put(f"{p}.{stage}.weight", to_bf16(1 + rnd(VISION_WIDTH, scale=0.1)), "BF16")
        put(f"{p}.{stage}.bias", to_bf16(rnd(VISION_WIDTH, scale=0.05)), "BF16")
    for stage in ("ls_1", "ls_2"):
        put(f"{p}.{stage}.gamma", to_bf16(1 + rnd(1, scale=0.05)), "BF16")
    for name in ("q", "k", "v", "out"):
        put(f"{p}.self_attn.{name}_proj.weight", to_bf16(rnd(VISION_WIDTH, VISION_WIDTH)), "BF16")
        put(f"{p}.self_attn.{name}_proj.bias", to_bf16(rnd(VISION_WIDTH, scale=0.05)), "BF16")
    put(f"{p}.mlp.c_fc.weight", to_bf16(rnd(4 * VISION_WIDTH, VISION_WIDTH)), "BF16")
    put(f"{p}.mlp.c_fc.bias", to_bf16(rnd(4 * VISION_WIDTH, scale=0.05)), "BF16")
    put(f"{p}.mlp.c_proj.weight", to_bf16(rnd(VISION_WIDTH, 4 * VISION_WIDTH)), "BF16")
    put(f"{p}.mlp.c_proj.bias", to_bf16(rnd(VISION_WIDTH, scale=0.05)), "BF16")

# ---- Write safetensors -----------------------------------------------------
DTYPE_SIZE = {"BF16": 2, "F32": 4, "U8": 1, "F8_E4M3": 1}
header = {"__metadata__": {"format": "pt"}}
payload = bytearray()
for name in sorted(tensors):
    array, dtype = tensors[name]
    raw = array.view(np.uint16).tobytes() if dtype == "BF16" else array.tobytes()
    header[name] = {"dtype": dtype, "shape": list(array.shape),
                    "data_offsets": [len(payload), len(payload) + len(raw)]}
    payload.extend(raw)
header_bytes = json.dumps(header).encode()
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "model-00001-of-00001.safetensors"), "wb") as f:
    f.write(struct.pack("<Q", len(header_bytes)))
    f.write(header_bytes)
    f.write(payload)
index = {"metadata": {"total_size": len(payload)},
         "weight_map": {name: "model-00001-of-00001.safetensors" for name in tensors}}
json.dump(index, open(os.path.join(OUT, "model.safetensors.index.json"), "w"))

def write_json(name, value):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(value, f, indent=2)

# Token ids inside the fixture's OWN vocabulary (128 rows): the real
# checkpoint's 200006/200054 would be refused by the strict config, which is
# the validation working, not something to relax for a fixture.
EOS_ID, PAD_ID, IMAGE_ID = VOCAB - 3, VOCAB - 2, VOCAB - 1

write_json("config.json", {
    "dtype": "bfloat16",
    "architectures": ["Inkling2LoopForConditionalGeneration"],
    "model_type": "inkling2_loop",
    "eos_token_id": EOS_ID,
    "pad_token_id": PAD_ID,
    "image_token_id": IMAGE_ID,
    "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
    "text_config": {
        "model_type": "inkling2_loop_text",
        "hidden_size": HIDDEN, "num_hidden_layers": LAYERS,
        "vocab_size": VOCAB, "unpadded_vocab_size": UNPADDED,
        "num_attention_heads": HEADS, "num_key_value_heads": KV, "head_dim": HEAD_DIM,
        "swa_head_dim": HEAD_DIM, "swa_num_attention_heads": HEADS, "swa_num_key_value_heads": KV,
        "sliding_window_size": WINDOW, "d_rel": D_REL, "rel_extent": REL_EXTENT,
        "log_scaling_n_floor": 8, "log_scaling_alpha": 0.1,
        "rms_norm_eps": 1e-06,
        "qk_l2_clip_threshold": 22.627416997969522, "rel_l2_clip_threshold": 8.0,
        "sconv_kernel_size": 4, "final_logit_softcapping": 30.0,
        "n_layers_per_attn_res_block": BLOCK, "attn_res_normalize_values": True,
        "n_routed_experts": EXPERTS, "num_experts_per_tok": TOP_K, "n_shared_experts": 0,
        "intermediate_size": INTER, "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0,
        "route_scale": 8.0, "use_gate_bias": True,
        "max_position_embeddings": 4096,
        "loop_config": {"core_start": CORE_START, "core_end": CORE_END,
                        "num_passes": PASSES, "gate_rank": GATE_RANK, "is_varloop": False},
    },
    "vision_config": {
        "model_type": "inkling2_vision", "text_hidden_size": HIDDEN,
        "temporal_patch_size": 1, "native_patch_size": 20, "adapter_merge_size": 2,
        "projection_hidden_dim": VISION_PROJ, "projector_bias": False, "hidden_size": VISION_WIDTH,
        "num_hidden_layers": VISION_BLOCKS, "num_attention_heads": VISION_HEADS, "mlp_ratio": 4.0,
        "posemb_grid_size": POSEMB_GRID, "layer_norm_eps": 1e-05,
    },
})

def groups():
    nv_w13, nv_w2, mx_w13, mx_w2, dense = [], [], [], [], []
    for layer in range(LAYERS):
        p = f"model.llm.layers.{layer}.mlp.experts"
        if layer >= 10:
            mx_w13.append(f"{p}.w13_weight"); mx_w2.append(f"{p}.w2_weight")
        else:
            nv_w13.append(f"{p}.w13_weight"); nv_w2.append(f"{p}.w2_weight")
    return nv_w13, nv_w2, mx_w13, mx_w2

nv_w13, nv_w2, mx_w13, mx_w2 = groups()
write_json("hf_quant_config.json", {"model": {"parameter_groups": [
    {"compute_dtype": "bfloat16", "output_dtype": "bfloat16", "required_activation_quantization": None,
     "tensors": ["model.llm.embed.weight", "model.llm.unembed.weight"], "weight_block_shape": [], "weight_transform": "cast"},
    {"compute_dtype": "nvfp4", "output_dtype": "uint8",
     "required_activation_quantization": {"amax_type": "per_token", "block_size": 16, "format": "nvfp4_e2m1",
                                          "rounding": "nearest", "scale_dtype": "float8_e4m3fn",
                                          "scale_selection_rule": "fouroversix_mae"},
     "tensors": nv_w13 + nv_w2, "weight_block_shape": [1, 16], "weight_scale_format": "amax_denominator",
     "weight_transform": "nvfp4_nearest"},
    {"compute_dtype": "mxfp8", "output_dtype": "float8_e4m3fn",
     "required_activation_quantization": {"block_size": 32, "format": "mxfp8_e4m3", "rounding": "nearest",
                                          "scale_dtype": "ue8m0"},
     "tensors": mx_w13 + mx_w2, "weight_block_shape": [32, 32], "weight_transform": "mxfp8_nearest"},
]}, "quant_method": "inkling2"})

write_json("generation_config.json", {"do_sample": True, "temperature": 1.0,
                                      "eos_token_id": EOS_ID, "pad_token_id": PAD_ID})
write_json("tokenizer_config.json", {"tokenizer_class": "PreTrainedTokenizerFast",
                                     "eos_token": "<|content_model_end_sampling|>",
                                     "pad_token": "<|endoftext|>"})
write_json("special_tokens_map.json", {"eos_token": "<|content_model_end_sampling|>",
                                       "pad_token": "<|endoftext|>"})
print(f"wrote {len(tensors)} tensors, {len(payload)} bytes to {OUT}")
