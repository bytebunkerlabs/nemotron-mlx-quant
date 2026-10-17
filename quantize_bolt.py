#!/usr/bin/env python
"""Stage 2 of the first-party Bolt quant: MLX-affine 4-bit quantize the extracted
`nemotron_h_puzzle` text backbone (output of extract_text_backbone.py).

Drives mlx_lm.convert through our `nemotron_h_puzzle` model class, so the load
path stacks the per-layer experts (the extracted dir ships them unstacked)
BEFORE quantizing — producing a checkpoint structurally identical to the
community rapid-mlx quant, and loadable by our port unchanged.

The quant policy is lifted verbatim from rapid-mlx (which modules carry .scales):
  4-bit, group 64 : Mamba in_proj/out_proj, routed experts switch_mlp.fc1/fc2,
                    shared_experts up_proj/down_proj, attention q/k/v/o_proj
  8-bit, group 64 : backbone.embeddings
  FULL PRECISION  : gate (router), fc1_latent_proj, fc2_latent_proj, conv1d,
                    every norm, lm_head  (predicate returns False for these)

Affine quant is deterministic for a fixed quantizer, so an independent run from
the same BF16 source and the same policy reproduces the structure and the
zero-points exactly; the only differences from the community rapid-mlx build are
sub-quant-level rounding deltas between MLX quantizer versions (see verify.py).
The result is an independently computed, numerically equivalent, first-party
quant — not a byte copy.

Usage:  HF_HOME=~/dgx/hf <venv>/bin/python quantize_bolt.py [SRC_DIR] [OUT_DIR]
"""
import os
import sys

from mlx_lm.convert import convert

HOME = os.path.expanduser("~")
SRC = sys.argv[1] if len(sys.argv) > 1 else (
    HOME + "/dgx/hf/bytebunkerlabs/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16-puzzle")
OUT = sys.argv[2] if len(sys.argv) > 2 else (
    HOME + "/dgx/hf/bytebunkerlabs/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit")

# Quantize these leaf modules to 4-bit; everything else (unless it is the
# embedding, below) stays full precision. Matched on the final path component,
# so `fc1_latent_proj`/`fc2_latent_proj` are NOT caught by `fc1`/`fc2`.
Q4 = {"in_proj", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj",
      "fc1", "fc2", "up_proj", "down_proj"}


def predicate(path, module):  # mlx_lm 0.32.0 calls quant_predicate(path, module)
    name = path.split(".")[-1]
    if name == "embeddings":
        return {"group_size": 64, "bits": 8}
    if name in Q4:
        return {"group_size": 64, "bits": 4}
    return False  # gate, *_latent_proj, lm_head, norms, conv1d -> keep FP


if __name__ == "__main__":
    print(f"src : {SRC}\nout : {OUT}")
    convert(
        hf_path=SRC,
        mlx_path=OUT,
        quantize=True,
        q_group_size=64,
        q_bits=4,
        q_mode="affine",
        quant_predicate=predicate,
    )
    print("done ->", OUT)
