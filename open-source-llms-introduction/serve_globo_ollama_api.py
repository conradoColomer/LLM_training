#!/usr/bin/env python3
"""
Ollama-compatible API server for the fused Globomantics model (MLX).
Run from project root: .venv/bin/python serve_globo_ollama_api.py
Then point Open WebUI to http://localhost:11435 (or set GLOBO_API_PORT=11434 to use Ollama's port when Ollama is not running)
"""
from datetime import datetime, timezone
from pathlib import Path
import json

from flask import Flask, request, jsonify, Response

PROJECT_ROOT = Path(__file__).resolve().parent
# Use base model + adapters (fused model on disk has quantized keys that break the loader)
MODEL_PATH = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
ADAPTER_PATH = PROJECT_ROOT / "adapters"
HOST = "127.0.0.1"
# Use 11435 by default so Ollama can stay on 11434; override with GLOBO_API_PORT=11434 if you prefer
PORT = int(__import__("os").environ.get("GLOBO_API_PORT", "11435"))

app = Flask(__name__)

# Lazy load model on first request
_model = None
_tokenizer = None


def get_model():
    global _model, _tokenizer
    if _model is None:
        from mlx_lm import load
        print("Loading model + adapters from", MODEL_PATH, "and", ADAPTER_PATH)
        _model, _tokenizer = load(MODEL_PATH, adapter_path=str(ADAPTER_PATH))
        print("Model loaded.")
    return _model, _tokenizer


@app.route("/api/generate", methods=["POST"])
def generate():
    """Ollama-compatible /api/generate (non-streaming)."""
    try:
        data = request.get_json() or {}
        prompt = data.get("prompt", "")
        stream = data.get("stream", False)
        model_name = data.get("model", "globo-assist")
        max_tokens = data.get("options", {}).get("num_predict", 200)

        model, tokenizer = get_model()
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        from mlx_lm import generate as mlx_gen
        response_text = mlx_gen(
            model, tokenizer, prompt=formatted, max_tokens=max_tokens,
            temp=0.2, verbose=False
        )

        if stream:
            def gen():
                for line in [response_text]:
                    yield json.dumps({
                        "model": model_name,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "response": line,
                        "done": True,
                        "done_reason": "stop",
                    }) + "\n"
            return Response(gen(), mimetype="application/x-ndjson")
        return jsonify({
            "model": model_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "response": response_text,
            "done": True,
            "done_reason": "stop",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/tags", methods=["GET"])
def tags():
    """Ollama-compatible /api/tags so Open WebUI sees the model."""
    return jsonify({
        "models": [
            {
                "name": "globo-assist",
                "modified_at": datetime.now(timezone.utc).isoformat(),
                "size": 0,
                "digest": "",
                "details": {"family": "llama", "parameter_size": "8B"},
            }
        ]
    })


if __name__ == "__main__":
    if not ADAPTER_PATH.exists() or not (ADAPTER_PATH / "adapter_config.json").exists():
        print("Adapter path not found:", ADAPTER_PATH)
        exit(1)
    print(f"Starting Ollama-compatible API at http://{HOST}:{PORT}")
    print("Use model name: globo-assist")
    app.run(host=HOST, port=PORT, threaded=True)
