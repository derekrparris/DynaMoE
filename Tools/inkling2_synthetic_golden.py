#!/usr/bin/env python3
"""Dump reference goldens for the synthetic Inkling 2 checkpoint.

Runs the PINNED Transformers implementation over the fixture that
`inkling2_synthetic.py` writes and records, for the last prompt position:

  * every decoder layer VISIT's output (48 entries for the published schedule,
    24 for the fixture's 12 layers with a 6-layer core),
  * the post-attention convolution of each visit,
  * the final logits.

The per-visit captures come from forward hooks on the layer modules, so they are
the reference's own arithmetic rather than a second written model.

    /tmp/inkling_venv/bin/python Tools/inkling2_synthetic_golden.py [fixture] [out]
"""
import json, sys, torch
from transformers import Inkling2LoopForConditionalGeneration

fixture = sys.argv[1] if len(sys.argv) > 1 else "/tmp/inkling2_synth"
out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/inkling2_synth_goldens.json"
ids = [1, 2, 3, 4, 5, 6, 7, 8]

model = Inkling2LoopForConditionalGeneration.from_pretrained(
    fixture, dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True
)
model.eval()
text = model.model.language_model

visits, post_attention = [], []
for i, layer in enumerate(text.layers):
    def hook(module, args, kwargs, output, idx=i):
        hidden = output[0] if isinstance(output, tuple) else output
        visits.append({"physicalLayer": idx, "output": hidden.detach().float()[-1, -1].tolist()})
    layer.register_forward_hook(hook, with_kwargs=True)
    def conv_hook(module, args, kwargs, output, idx=i):
        value = output[0] if isinstance(output, tuple) else output
        post_attention.append({"physicalLayer": idx, "output": value.detach().float()[-1, -1].tolist()})
    layer.post_attention_convolution.register_forward_hook(conv_hook, with_kwargs=True)

with torch.no_grad():
    out = model(torch.tensor([ids]))

payload = {
    "ids": ids,
    "visits": [entry["output"] for entry in visits],
    "visitLayers": [entry["physicalLayer"] for entry in visits],
    "postAttention": [entry["output"] for entry in post_attention],
    "logits": out.logits[0, -1].float().tolist(),
}
json.dump(payload, open(out_path, "w"))
print("wrote", out_path, "visits", len(visits), "logits", len(payload["logits"]))
