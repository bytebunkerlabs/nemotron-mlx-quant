# nemotron-mlx-quant

Reproduce the 4-bit MLX weights for NVIDIA's **Nemotron-3.5 Bolt-74B-A9B** from the
BF16 release, first-party: extract the text backbone, quantize it affine, and
verify the result — so a 74B mixture-of-experts runs on a single Apple-silicon
Mac at reading speed.

This is the exact pipeline that built
[`bytebunker/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit`](https://huggingface.co/bytebunker/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit).
It starts from NVIDIA's weights and nothing else — no borrowed conversion — and
proves, at the end, that the output is numerically equivalent to the community
`rapid-mlx` build without being a byte copy of it. Affine quantization is
deterministic, so an independent run from the same source and the same policy
lands on the same checkpoint; the walkthrough is in the companion post,
[*Quantize it yourself*](https://blog.bytebunkerlabs.ai/posts/quantize-nemotron-bolt-74b-to-mlx/).

The weights it produces are served by the
[`nemotron_h_puzzle` port for TensorFold / mlx-lm](https://github.com/bytebunkerlabs/dgx-spark-serve/tree/main/contrib/tensorfold-nemotron-h-puzzle)
— the architecture is new enough that stock mlx-lm can't load it yet.

## What it builds

| | |
|---|---|
| **Input** | [`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16) — the multimodal (vision + text) BF16 release, ~150 GB |
| **Output** | MLX **affine 4-bit**, group size 64 — ~43 GB, 9 shards, **4.674 bits/weight**, 1,309 tensors |
| **Model** | hybrid Mamba-2 + attention MoE (`nemotron_h_puzzle`): 74B total, ~9B active, heterogeneous per-layer expert counts, latent MoE, top-k 9 |
| **Runs on** | Apple silicon, 64 GB+ unified memory; ~44 GB resident at inference |
| **Time** | minutes of compute on an M3 Max once the download is in hand (the download dominates) |

## The pipeline

Four short scripts, run in order. Each takes optional `[SRC] [OUT]` arguments and
otherwise uses sensible `~`-relative defaults; override them for your layout.
Run them with the Python that has `mlx` / `mlx-lm` installed (and the port, for
stage 4).

```
extract_text_backbone.py   # 1. omni BF16  ->  text-only puzzle BF16
quantize_bolt.py           # 2. puzzle BF16 ->  affine 4-bit MLX
verify.py                  # 3. cross-check the result is a real reproduction
smoke_test.py              # 4. load it and generate
```

### 1. Extract the text backbone

The BF16 release is an *omni* checkpoint: the text model lives under
`language_model.*`, with a RADIO vision tower (`vision_model.*`) and a projector
(`mlp1.*`) alongside it. Stage 1 keeps only the text tensors (stripping the
`language_model.` prefix to `backbone.*`, which the port expects), drops the
rest, and writes a clean, servable puzzle BF16 directory — renamed shards, an
`index.json`, and a `config.json` built from the omni's `llm_config` (minus the
`auto_map` remote-code hook, with the full `eos` stop-set pulled from
`generation_config.json`). It streams shard-by-shard, so peak memory is about one
shard.

```
python extract_text_backbone.py  [SRC_SNAPSHOT_DIR]  [OUT_DIR]
```

### 2. Quantize, affine 4-bit

Stage 2 drives `mlx_lm.convert` *through the `nemotron_h_puzzle` model class*, so
the load path stacks the per-layer experts (the extracted dir ships them
unstacked) before quantizing. The policy is applied by a predicate matched on the
final path component — see below. Output is the 4-bit MLX directory.

```
HF_HOME=~/dgx/hf  python quantize_bolt.py  [SRC_DIR]  [OUT_DIR]
```

### 3. Verify it's a real reproduction

Stage 3 cross-checks the output against a reference MLX 4-bit build of the same
model — the community `rapid-mlx` quant if you have it, or a local download of
the published `bytebunker` repo (pass its path as the second argument). It reports
the tensor set, how many zero-points are byte-identical, the largest per-group
scale delta, and the largest dequantized-weight difference.

```
python verify.py  [OURS_DIR]  [REF_DIR]
```

### 4. Run it

Stage 4 loads the quant with mlx-lm and generates, to confirm it is coherent.
This needs `nemotron_h_puzzle` support in your mlx-lm — add the one additive
model file from the [port](https://github.com/bytebunkerlabs/dgx-spark-serve/tree/main/contrib/tensorfold-nemotron-h-puzzle)
(the fused engine needs its second file too), or serve it under that engine.

```
python smoke_test.py  [MODEL_DIR_OR_HF_ID]  ["your prompt"]
```

## Quant policy

Lifted verbatim from the community `rapid-mlx` build, so the two are directly
comparable. The predicate in `quantize_bolt.py` matches on the last path
component (so `fc1_latent_proj` is **not** caught by `fc1`):

| Precision | Modules |
|---|---|
| **4-bit, group 64** | Mamba `in_proj` / `out_proj`; attention `q/k/v/o_proj`; routed experts `switch_mlp.fc1` / `fc2`; shared expert `up_proj` / `down_proj` |
| **8-bit, group 64** | token embeddings |
| **full precision** | router / gate, `fc1_latent_proj`, `fc2_latent_proj`, `lm_head`, every norm, Mamba `conv1d` |

The precision-sensitive paths — routing, the latent projections, the output head,
the norms — are the ones kept at full precision, which is where most of 4-bit's
quality cost would otherwise land.

## What "reproduced" means

Affine quantization is deterministic for a fixed quantizer: given the same BF16
source and the same policy, the **structure** and the **zero-points** come out
exactly the same, every time. So a correct independent run should reproduce an
existing build to within quantizer-version rounding — and that is exactly what
`verify.py` measured against `rapid-mlx`:

- **Structurally identical** — the same 1,309 tensors, nothing extra or missing on
  either side, no shape or dtype mismatch.
- **Every zero-point byte-identical** — 273 of 273 `.biases` tensors.
- **Per-group scales agree to ~1e-3.**
- **Dequantized weights agree to within a single quant level** — a relative
  difference of ~7e-4, about 1000× smaller than 4-bit quantization's own error
  against BF16.

That sub-level rounding delta — our MLX version versus theirs — is not noise to
apologize for; it is the fingerprint of weights computed from scratch. An
independent reproduction, not a copy.

## Requirements

- **Apple silicon** (the output is MLX / Metal).
- `pip install mlx mlx-lm` — built and measured on `mlx-lm` 0.32.x.
- The [`nemotron_h_puzzle` port](https://github.com/bytebunkerlabs/dgx-spark-serve/tree/main/contrib/tensorfold-nemotron-h-puzzle)
  installed into your mlx-lm (one additive model file) — stages 2 and 4 load the
  model through it.
- Disk for the ~150 GB BF16 download, the ~148 GB extracted backbone (deletable
  after quantizing), and the ~43 GB output.
- ~64 GB+ unified memory to run the result; a few GB of headroom for `verify.py`
  to dequantize the largest tensor pair.

## Serving the result

The published weights:
**[`bytebunker/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit`](https://huggingface.co/bytebunker/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit)**.
With the port installed:

```python
from mlx_lm import generate, load

model, tokenizer = load("bytebunker/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit")
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain tensor parallelism in two sentences."}],
    add_generation_prompt=True)
print(generate(model, tokenizer, prompt=prompt, verbose=True))
```

Measured on an M3 Max (128 GB): ~44 tok/s decode under plain mlx-lm, ~45 tok/s
single-stream (~51 at four concurrent streams) under the fused TensorFold engine.

## Credits & license

- **Base model & weights:** © NVIDIA Corporation, released under the
  **OpenMDW-1.1** license, which governs the weights this pipeline produces and
  permits quantized derivatives. The 4-bit output ships with a copy of that
  license and credits NVIDIA; see the
  [model card](https://huggingface.co/bytebunker/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit).
  Not affiliated with or endorsed by NVIDIA; "NVIDIA" and "Nemotron" are their
  trademarks.
- **Reference build:** the community
  [`rapid-mlx`](https://huggingface.co/rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit)
  quant, used by `verify.py` as the cross-check.
- **Serving engine:** [TensorFold](https://github.com/ashhart/TensorFold) (upstream,
  by `ashhart`) — ByteBunker contributes the `nemotron_h_puzzle` port, not the engine.
- **This pipeline:** [ByteBunker Labs](https://bytebunkerlabs.ai).

## Related

- The walkthrough: [*Quantize it yourself: a 74B MoE to 4-bit MLX, from NVIDIA's BF16*](https://blog.bytebunkerlabs.ai/posts/quantize-nemotron-bolt-74b-to-mlx/)
- The story of the port: [*A day-one port: Nemotron Bolt 74B on a Mac, at 45 tok/s*](https://blog.bytebunkerlabs.ai/posts/nemotron-bolt-74b-on-a-mac/)
- The serving port: [dgx-spark-serve / contrib](https://github.com/bytebunkerlabs/dgx-spark-serve/tree/main/contrib/tensorfold-nemotron-h-puzzle)
- [ByteBunker Labs](https://bytebunkerlabs.ai) · [blog](https://blog.bytebunkerlabs.ai) · [Hugging Face](https://huggingface.co/bytebunker)
