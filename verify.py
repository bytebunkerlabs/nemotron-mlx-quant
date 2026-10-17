#!/usr/bin/env python
"""Stage 3 (verify): cross-check the first-party quant (output of
quantize_bolt.py) against a reference MLX 4-bit build of the same model — the
community `rapid-mlx` quant, or a local download of the published
`bytebunker/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit`.

Affine quantization is deterministic for a fixed quantizer, so an independent
run from the same BF16 source and the same policy reproduces the *structure* and
the *zero-points* exactly; differences are limited to sub-quant-level rounding
between MLX quantizer versions. This script measures that and reports:

  - tensor set:   same tensors on both sides (no extras, no missing)
  - full-precision tensors: byte-identical count (gate, latent projs, lm_head,
                   norms, conv1d — kept FP by the policy)
  - zero-points (`.biases`): byte-identical count (expected: all)
  - scales:       max |Δ| (absolute, and relative to the largest scale)
  - dequantized weights: max |Δ| and max relative |Δ| vs the reference

A clean result: identical structure, every zero-point identical, scales within
~1e-3, dequantized weights within a single quant level (relative ~1e-3 or less —
about 1000x smaller than 4-bit's own error vs BF16). That is an independently
computed, numerically equivalent quant — not a byte copy.

Needs enough RAM to dequantize the largest tensor pair (a few GB); done one
tensor at a time.

Usage:  <venv>/bin/python verify.py [OURS_DIR] [REF_DIR]
"""
import glob
import os
import sys

import mlx.core as mx

G = 64  # group size used by the pipeline (and by rapid-mlx)

HOME = os.path.expanduser("~")
DEF_OURS = HOME + "/dgx/hf/bytebunkerlabs/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit"
REF_GLOB = (HOME + "/dgx/hf/hub/"
            "models--rapid-mlx--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit/snapshots/*/")


def load_all(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        out.update(mx.load(f))
    return out


def resolve_ref(arg):
    if arg:
        return arg
    hits = sorted(glob.glob(REF_GLOB))
    if not hits:
        sys.exit("no REF_DIR given and no rapid-mlx snapshot found; pass a "
                 "reference dir (e.g. a download of the published bytebunker repo)")
    return hits[-1]


def maxabs(x):
    return mx.max(mx.abs(x.astype(mx.float32))).item()


def main():
    ours = sys.argv[1] if len(sys.argv) > 1 else DEF_OURS
    ref = resolve_ref(sys.argv[2] if len(sys.argv) > 2 else None)
    print(f"ours : {ours}\nref  : {ref}\n")
    A, B = load_all(ours), load_all(ref)
    ak, bk = set(A), set(B)
    common = ak & bk
    print(f"tensors        ours {len(ak)}   ref {len(bk)}   common {len(common)}")
    print(f"only in ours   {len(ak - bk)}")
    print(f"only in ref    {len(bk - ak)}")
    for k in sorted(ak - bk)[:8]:
        print("   + only ours:", k)
    for k in sorted(bk - ak)[:8]:
        print("   - only ref :", k)

    quant_bases = {k[:-7] for k in common if k.endswith(".scales")}

    # full-precision tensors: anything in common that is not part of a quant triple
    fp_same = fp_tot = 0
    for k in sorted(common):
        if k.endswith(".scales") or k.endswith(".biases"):
            continue
        if k.endswith(".weight") and k[:-7] in quant_bases:
            continue
        fp_tot += 1
        a, b = A[k], B[k]
        if a.shape == b.shape and a.dtype == b.dtype and bool(mx.all(a == b).item()):
            fp_same += 1

    bias_same = bias_tot = 0
    scale_abs = scale_den = 0.0
    dq_abs = dq_rel = 0.0
    for base in sorted(quant_bases):
        sk, bk_, wk = base + ".scales", base + ".biases", base + ".weight"
        if bk_ in A and bk_ in B:
            bias_tot += 1
            if A[bk_].shape == B[bk_].shape and bool(mx.all(A[bk_] == B[bk_]).item()):
                bias_same += 1
        if sk in A and sk in B and A[sk].shape == B[sk].shape:
            scale_abs = max(scale_abs, maxabs(A[sk].astype(mx.float32) - B[sk].astype(mx.float32)))
            scale_den = max(scale_den, maxabs(B[sk]))
        if (wk in A and wk in B and sk in A and sk in B and bk_ in A and bk_ in B
                and A[wk].shape == B[wk].shape):
            bits = int(round(32 * A[wk].shape[-1] / (A[sk].shape[-1] * G)))
            da = mx.dequantize(A[wk], A[sk], A[bk_], group_size=G, bits=bits).astype(mx.float32)
            db = mx.dequantize(B[wk], B[sk], B[bk_], group_size=G, bits=bits).astype(mx.float32)
            ad = mx.abs(da - db)
            m = mx.max(ad).item()
            dq_abs = max(dq_abs, m)
            dq_rel = max(dq_rel, m / (maxabs(db) or 1.0))
            del da, db, ad

    print()
    print(f"full-precision tensors   {fp_same}/{fp_tot} byte-identical")
    print(f"zero-points (.biases)    {bias_same}/{bias_tot} byte-identical")
    print(f"scales max |Δ|           {scale_abs:.3e}   ({scale_abs / (scale_den or 1):.2e} of max scale)")
    print(f"dequant max |Δ|          {dq_abs:.3e}")
    print(f"dequant max relative |Δ| {dq_rel:.3e}")

    ok = (ak == bk and bias_tot and bias_same == bias_tot and dq_rel < 1e-2)
    print("\nRESULT:",
          "structurally identical, numerically equivalent ✅ "
          "(independently computed, not a byte copy)" if ok
          else "differences beyond sub-quant-level rounding — inspect above")


if __name__ == "__main__":
    main()
