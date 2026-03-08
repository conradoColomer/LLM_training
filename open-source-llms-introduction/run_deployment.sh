#!/usr/bin/env bash
# Run this in your Mac Terminal so MLX can use Metal/GPU.
# From project root: ./run_deployment.sh
# Serves your fine-tuned model (base + adapters) on port 11434 (Ollama-compatible).

set -e
cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"

echo "=== Using project: $PROJECT_ROOT ==="
source "$PROJECT_ROOT/.venv/bin/activate"
pip install -q flask 2>/dev/null || true

echo ""
echo "=== Starting Globomantics API at http://localhost:11435 ==="
echo "    Model: base + adapters (globo-assist)"
echo ""
echo "In another terminal run Open WebUI (needs Python 3.11 or 3.12):"
echo "  See OPEN_WEBUI_SETUP.md for install and connecting to http://localhost:11435"
echo ""
exec .venv/bin/python serve_globo_ollama_api.py
