#!/usr/bin/env python3
"""Reference goldens for Inkling 2's still-image tower.

The text path has a reference dump per visit, per message channel and for the
final read; the vision tower needs the same thing before a Swift/Metal port can
be checked, because a tower that merely produces finite numbers of the right
shape runs at full speed and answers nonsense.

What is dumped, all from the RELEASED `Inkling2VisionModel` on the checkpoint's
own weights in FP32 (the stored tensors are BF16, so an FP32 load is exactly
what a BF16 read widens to -- there is no rounding to hide behind here):

  gridThw         one image, `[t, rows, cols]` in LANGUAGE-token cells
  pixels          the `pixel_values` tensor's checksum and a few samples: the
                  exact input must be rebuilt by formula, not shipped
  interp          the position-embedding gather indices and weights, so the
                  bilinear resample is pinned separately from the tower
  classToken      `class_embedding + cls_position_embedding`
  embeddings      patch embeddings + interpolated position embeddings, with the
                  class token in front (what `pre_layernorm` receives)
  preLayernorm    after `backbone.ln_pre`
  layers          every encoder layer's full output
  patchStates     the patch rows the merger consumes
  poolerOutput    the merged, projected rows the language model receives

    /tmp/inkling_venv/bin/python Tools/inkling2_vision_goldens.py

Writes `/tmp/inkling2_vision_goldens.json`. The Swift side rebuilds the pixels
from the same formula and asserts them against `pixels.checksum` before it
compares a single activation, so a fixture that drifted cannot pass quietly.
"""
import json

import torch
from transformers import Inkling2LoopForConditionalGeneration

SNAPSHOT = (
    "/Users/derekparris/.cache/huggingface/hub/"
    "models--thinkingmachines--Inkling-2-Mini/snapshots/"
    "c9d028302cd0f035fdf8e7e91f089a4ce94d2195"
)
OUTPUT = "/tmp/inkling2_vision_goldens.json"

# One image, in language-token cells. Three rows by two columns keeps the grid
# asymmetric on purpose: a transposed row/column order passes on a square one.
CELL_ROWS, CELL_COLS = 3, 2
CELL = 40  # pixels per side, the image processor's fixed cell


def pixel(cell: int, row: int, column: int, channel: int) -> float:
    """The deterministic picture both sides build. Must match the Swift helper."""
    seed = ((cell * 131 + row) * 137 + column * 149 + channel * 151) % 251
    return seed / 125.5 - 1.0


def main():
    # The tower is published under `model.visual.backbone.*`, so it has to be
    # taken from the full conditional-generation model's own `vision_tower`
    # submodule. Loading `Inkling2VisionModel` directly from the snapshot root
    # finds none of its names and silently initializes a RANDOM tower -- which
    # is finite, correctly shaped, and worthless as a golden.
    full = Inkling2LoopForConditionalGeneration.from_pretrained(
        SNAPSHOT, dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True
    )
    full.eval()
    model = full.model.vision_tower
    config = model.config

    merge = config.spatial_merge_size
    patch = config.patch_size
    cells = CELL_ROWS * CELL_COLS
    values = torch.zeros(cells, CELL, CELL, config.num_channels)
    for cell in range(cells):
        for row in range(CELL):
            for column in range(CELL):
                for channel in range(config.num_channels):
                    values[cell, row, column, channel] = pixel(cell, row, column, channel)
    grid_thw = torch.tensor([[1, CELL_ROWS, CELL_COLS]])

    captured = {}
    handles = []

    def hook(name):
        def run(_module, _inputs, output):
            tensor = output
            if isinstance(output, (tuple, list)):
                tensor = next(item for item in output if isinstance(item, torch.Tensor))
            captured[name] = tensor.detach().float()
        return run

    handles.append(model.embeddings.register_forward_hook(hook("embeddings")))
    handles.append(model.pre_layernorm.register_forward_hook(hook("pre_layernorm")))
    for index, layer in enumerate(model.layers):
        handles.append(layer.register_forward_hook(hook(f"layer{index}")))
    handles.append(model.merger.register_forward_hook(hook("merger")))

    with torch.no_grad():
        out = model(pixel_values=values, image_grid_thw=grid_thw)
    for handle in handles:
        handle.remove()

    patch_grid = (CELL_ROWS * merge, CELL_COLS * merge)
    patch_tokens = patch_grid[0] * patch_grid[1]
    sequence = patch_tokens + 1  # each image starts with one class token
    assert captured["embeddings"].shape[0] == sequence, captured["embeddings"].shape

    class_token = (
        model.embeddings.class_embedding + model.embeddings.cls_position_embedding
    ).detach().float()

    # The interpolation itself, recomputed exactly as the embeddings call it.
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights

    patch_grid_thw = grid_thw * torch.tensor([1, merge, merge])
    indices, weights = get_vision_interpolation_indices_and_weights(
        patch_grid_thw, num_grid_per_side=config.posemb_grid_size,
        mode="bilinear", align_corners=False, spatial_merge_size=merge, padding="border",
    )

    def rows(tensor):
        return [[float(v) for v in row] for row in tensor.reshape(-1, tensor.shape[-1])]

    pixel_values = values.reshape(-1)
    payload = {
        "checkpoint": SNAPSHOT.rsplit("/", 1)[-1],
        "cellRows": CELL_ROWS,
        "cellColumns": CELL_COLS,
        "cellSize": CELL,
        "patchGrid": list(patch_grid),
        "towerTokens": patch_tokens,
        "sequenceLength": sequence,
        "numChannels": config.num_channels,
        "hiddenSize": config.hidden_size,
        "numLayers": config.num_hidden_layers,
        "pixelCount": int(pixel_values.numel()),
        "pixelChecksum": float(pixel_values.double().sum()),
        "pixelSamples": [float(v) for v in pixel_values[:: max(1, pixel_values.numel() // 32)][:32]],
        "interpolation": {
            "indices": indices.tolist(),
            "weights": weights.tolist(),
        },
        "classToken": [float(v) for v in class_token],
        "embeddings": rows(captured["embeddings"]),
        "preLayernorm": rows(captured["pre_layernorm"]),
        "layers": [rows(captured[f"layer{index}"]) for index in range(len(model.layers))],
        "patchStates": rows(out.last_hidden_state),
        "poolerOutput": rows(out.pooler_output),
    }
    json.dump(payload, open(OUTPUT, "w"))
    print("wrote", OUTPUT)
    print(f"  cells {CELL_ROWS}x{CELL_COLS} -> patch grid {patch_grid[0]}x{patch_grid[1]} "
          f"-> {sequence} tokens (class token + {patch_tokens})")
    print(f"  pooler output {out.pooler_output.shape} (rows per language token: "
          f"{out.pooler_output.shape[0] // cells})")
    print(f"  pixel checksum {payload['pixelChecksum']:.6f}")
    print(f"  layer 0 output[0][:4] {payload['layers'][0][0][:4]}")


if __name__ == "__main__":
    main()
