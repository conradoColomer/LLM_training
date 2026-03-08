#!/usr/bin/env python3
"""
Export fused MLX model to GGUF with row-major fix (workaround for mlx_lm GGUF export bug).
Run from project root with: .venv/bin/python export_gguf_from_fused.py
"""
import numpy as np
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

# Import after potential path setup
from mlx_lm import load
from mlx_lm.gguf import convert_to_gguf

PROJECT_ROOT = Path(__file__).resolve().parent
FUSED_PATH = PROJECT_ROOT / "fused_model"
OUTPUT_GGUF = FUSED_PATH / "ggml-model-f16.gguf"


def make_row_major(weights_dict):
    """Convert all weight arrays to row-major so mx.save_gguf accepts them."""
    out = {}
    for k, v in weights_dict.items():
        if isinstance(v, mx.array):
            # Force row-major (C-contiguous) via numpy roundtrip
            out[k] = mx.array(np.ascontiguousarray(np.array(v)))
        else:
            out[k] = v
    return out


def main():
    print("Loading fused model from", FUSED_PATH)
    model, tokenizer = load(str(FUSED_PATH), adapter_path=None)
    config = model.config if hasattr(model, "config") else {}
    if not config and (FUSED_PATH / "config.json").exists():
        import json
        with open(FUSED_PATH / "config.json") as f:
            config = json.load(f)

    weights = dict(tree_flatten(model.parameters()))
    print("Making weights row-major...")
    weights = make_row_major(weights)
    print("Converting to GGUF...")
    convert_to_gguf(FUSED_PATH, weights, config, str(OUTPUT_GGUF))
    print("Saved:", OUTPUT_GGUF)


if __name__ == "__main__":
    main()
