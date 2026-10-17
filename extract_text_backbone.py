#!/usr/bin/env python
"""Stage 1 of the first-party Bolt quant: extract the `nemotron_h_puzzle` text
backbone from NVIDIA's Nemotron-3.5-Bolt-74B-A9B *omni* (vision+text) BF16.

The omni checkpoint stores the text model under `language_model.*`, the RADIO
vision tower under `vision_model.*`, and the projector under `mlp1.*`. We keep
`language_model.*` (stripping that prefix -> `backbone.…`, which is what our
mlx_lm port expects), drop the rest, and emit a clean, servable puzzle BF16 dir:
renamed shards + index.json + a config.json built from the omni's `llm_config` +
the tokenizer / chat template.

Streams shard-by-shard, so peak memory is ~one shard. Run AFTER the omni
download completes.

Usage:  <venv>/bin/python extract_text_backbone.py [SRC_SNAPSHOT_DIR] [OUT_DIR]
"""
import glob
import json
import os
import shutil
import sys

import mlx.core as mx

HOME = os.path.expanduser("~")
DEF_SRC_GLOB = (HOME + "/dgx/hf/hub/"
                "models--nvidia--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16/snapshots/*/")
DEF_OUT = HOME + "/dgx/hf/bytebunkerlabs/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16-puzzle"

PREFIX = "language_model."
DROP = ("vision_model.", "mlp1.")
AUX = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
       "generation_config.json", "special_tokens_map.json")


def nbytes(a):
    try:
        return a.nbytes
    except Exception:
        return a.size * (2 if a.dtype in (mx.bfloat16, mx.float16) else 4)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(DEF_SRC_GLOB))[-1]
    out = sys.argv[2] if len(sys.argv) > 2 else DEF_OUT
    os.makedirs(out, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(src, "*.safetensors")))
    assert shards, f"no .safetensors in {src}"
    print(f"src: {src}\nout: {out}\n{len(shards)} input shards")

    # config.json from the omni's llm_config (already a puzzle config), with two
    # fixes so it reads as a clean MLX checkpoint (and matches rapid-mlx):
    #   - drop `auto_map`: HF remote-code autoload, meaningless for an MLX port.
    #   - take the full eos stop-set from generation_config.json ([2, 11]); the
    #     llm_config carried only the backbone EOS (2), so the two would disagree.
    cfg = json.load(open(os.path.join(src, "config.json")))
    llm = cfg["llm_config"]
    llm.pop("auto_map", None)
    try:
        gc = json.load(open(os.path.join(src, "generation_config.json")))
        if gc.get("eos_token_id") is not None:
            llm["eos_token_id"] = gc["eos_token_id"]
    except FileNotFoundError:
        pass
    json.dump(llm, open(os.path.join(out, "config.json"), "w"), indent=2)

    for a in AUX:
        p = os.path.join(src, a)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(out, a))
            print("  aux:", a)

    index = {"metadata": {"total_size": 0}, "weight_map": {}}
    total = 0
    out_n = 0
    kept_total = 0
    for shard in shards:
        w = mx.load(shard)  # lazy / mmap
        keep = {}
        for k, v in w.items():
            if any(k.startswith(d) for d in DROP):
                continue
            keep[k[len(PREFIX):] if k.startswith(PREFIX) else k] = v
        if not keep:
            print("  skip (no text tensors):", os.path.basename(shard))
            del w
            continue
        out_n += 1
        fn = f"model-{out_n:05d}.safetensors"
        mx.save_safetensors(os.path.join(out, fn), keep)
        for k, v in keep.items():
            index["weight_map"][k] = fn
            total += nbytes(v)
        kept_total += len(keep)
        print(f"  [{out_n}] {os.path.basename(shard)} -> {fn}  ({len(keep)} tensors)")
        del w, keep

    index["metadata"]["total_size"] = total
    json.dump(index, open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=2)
    print(f"\ndone: {out_n} shards, {kept_total} tensors, ~{total/1e9:.1f} GB -> {out}")


if __name__ == "__main__":
    main()
