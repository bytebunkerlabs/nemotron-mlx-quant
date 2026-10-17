#!/usr/bin/env python
"""Stage 4 (run): load the quant with mlx_lm and generate a little text, to
confirm it loads and is coherent.

Needs `nemotron_h_puzzle` support in your mlx_lm — stock mlx_lm 0.32.x does not
load this architecture yet. Add the two additive files from
dgx-spark-serve/contrib/tensorfold-nemotron-h-puzzle (a model module and a
family registration) to your mlx_lm, or use the TensorFold engine.

Usage:  <venv>/bin/python smoke_test.py [MODEL_DIR_OR_HF_ID] ["prompt"]
"""
import sys

from mlx_lm import generate, load

MODEL = sys.argv[1] if len(sys.argv) > 1 else (
    "bytebunker/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit")
PROMPT = sys.argv[2] if len(sys.argv) > 2 else (
    "Explain tensor parallelism in two sentences.")

model, tokenizer = load(MODEL)
prompt = PROMPT
if tokenizer.chat_template is not None:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True)
print(generate(model, tokenizer, prompt=prompt, max_tokens=256, verbose=True))
