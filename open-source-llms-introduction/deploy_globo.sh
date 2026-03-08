#!/usr/bin/env bash
# Deploy Globomantics fine-tuned model: fuse adapters → GGUF → Ollama → ready for Open WebUI
# Run from project root: ./deploy_globo.sh

set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=== 1. Fusing LoRA adapters and exporting GGUF ==="
if ! python3 -c "import mlx_lm" 2>/dev/null; then
  echo "Error: mlx_lm not found. Activate your MLX env or run: pip install mlx mlx-lm"
  exit 1
fi

FUSED_DIR="$PROJECT_ROOT/fused_model"
mkdir -p "$FUSED_DIR"

python3 -m mlx_lm fuse \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
  --adapter-path "$PROJECT_ROOT/adapters" \
  --save-path "$FUSED_DIR" \
  --dequantize \
  --export-gguf

if [[ ! -f "$FUSED_DIR/ggml-model-f16.gguf" ]]; then
  echo "Expected $FUSED_DIR/ggml-model-f16.gguf not found. Check mlx_lm.fuse output."
  exit 1
fi
echo "Fused model saved to $FUSED_DIR/ggml-model-f16.gguf"

echo ""
echo "=== 2. Creating Ollama model 'globo-assist' ==="
if ! command -v ollama &>/dev/null; then
  echo "Ollama not found. Install from https://ollama.com/download then run:"
  echo "  ollama create globo-assist -f $PROJECT_ROOT/Modelfile"
  exit 1
fi

ollama create globo-assist -f "$PROJECT_ROOT/Modelfile"
echo ""
echo "=== Done. Run: ollama run globo-assist '¿Qué es Globomantics?' ==="
echo "Then start Open WebUI: pip install open-webui && open-webui serve"
echo "In the UI, set Ollama URL to http://localhost:11434 and select model 'globo-assist'."
