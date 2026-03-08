# Deploying Your Fine-Tuned Globomantics Model to Open WebUI

This guide gets your MLX-trained LoRA adapters running in **Open WebUI** via **Ollama**. Your base model is **Meta-Llama-3.1-8B-Instruct-4bit** (MLX); adapters live in `adapters/`.

---

## Quick start (copy-paste)

Use the **same Python environment where you ran fine-tuning** (the one that has `mlx` and `mlx-lm`). From the project root:

```bash
# 1. Fuse adapters and export GGUF (requires mlx_lm)
python -m mlx_lm fuse \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
  --adapter-path ./adapters \
  --save-path ./fused_model \
  --dequantize \
  --export-gguf

# 2. Install Ollama (one-time): https://ollama.com/download or: brew install ollama
#    Start Ollama (often automatic), then:

ollama create globo-assist -f Modelfile

# 3. Test in terminal
ollama run globo-assist "¿Qué es Globomantics?"

# 4. Install and run Open WebUI
pip install open-webui
open-webui serve
# → Open http://localhost:8080, set Ollama URL to http://localhost:11434, select model "globo-assist"
```

Or run the script (after fuse): `./deploy_globo.sh` (script does fuse + ollama create; you still need Ollama installed).

---

## 1. Environment summary

| Item | Value |
|------|--------|
| **Base model** | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` |
| **Adapter path** | `adapters/` (contains `adapters.safetensors` + `adapter_config.json`) |
| **Training data** | `02/demos/training_data/train.jsonl` (50 Globomantics Q&A pairs) |

Fine-tuning was done with **MLX** (`mlx_lm lora`), not Unsloth. Open WebUI does not load MLX adapters directly; the path is: **fuse adapters → export GGUF → load in Ollama → use Open WebUI**.

---

## 2. Step 1: Python environment with MLX (if needed)

If you already ran training on this machine, you likely have a venv/conda where `mlx` and `mlx-lm` are installed. Use that same environment for the steps below.

If not, from the project root:

```bash
# Recommended: Python 3.10–3.12 on Apple Silicon
python3 -m venv .venv
source .venv/bin/activate   # or: conda activate <your_env>

pip install mlx mlx-lm
```

Verify:

```bash
python -c "from mlx_lm import load, generate; print('OK')"
```

---

## 3. Step 2: Fuse LoRA adapters and export GGUF

This merges your LoRA weights into the base model and writes a single GGUF file for Ollama.

From the **project root** (`open-source-llms-introduction/`):

```bash
# Activate the same env you used for training (or the one above)
source .venv/bin/activate   # if using venv

# Fuse adapters into base model and export GGUF (fp16)
python -m mlx_lm fuse \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
  --adapter-path ./adapters \
  --save-path ./fused_model \
  --dequantize \
  --export-gguf
```

**What this does:**

- Loads the 4-bit base model and your `adapters/` LoRA weights.
- Fuses them into one model (de-quantized to fp16 for export).
- Writes the fused model to `./fused_model/` and, with `--export-gguf`, creates **`./fused_model/ggml-model-f16.gguf`**.

If `--export-gguf` is not available in your `mlx-lm` version, fuse without it and then use the MLX docs or `mlx-examples` for converting the fused model to GGUF (e.g. llama.cpp’s conversion scripts).

**Check:**

```bash
ls -la ./fused_model/
# Expect: ggml-model-f16.gguf (and possibly other files)
```

---

## 4. Step 3: Install Ollama and create the model

### 4.1 Install Ollama

- **macOS:** <https://ollama.com/download> or `brew install ollama`.
- Start the service (often automatic on install, or run `ollama serve` in the background).

```bash
ollama --version
```

### 4.2 Create the `globo-assist` model from your GGUF

From the **project root** (so the path inside the Modelfile resolves):

```bash
ollama create globo-assist -f Modelfile
```

The included **Modelfile** points to `./fused_model/ggml-model-f16.gguf` and sets a system prompt and Llama 3.1-style template.

**Quick test in the terminal:**

```bash
ollama run globo-assist "¿Qué es Globomantics y quién es su CEO?"
```

You should see answers in line with your training data (e.g. CEO Carina Globos, company description).

---

## 5. Step 4: Install and run Open WebUI

Open WebUI talks to Ollama as the backend, so Ollama must be running with `globo-assist` available.

### 5.1 Install Open WebUI

```bash
# Use a clean env if you prefer (optional)
pip install open-webui
```

### 5.2 Launch the UI

```bash
open-webui serve
```

By default it listens on **http://localhost:8080**. The first time you open it you may be asked to create an admin user.

### 5.3 Connect to Ollama and select the model

1. Open **http://localhost:8080** in your browser.
2. In **Settings** (or the connection/backend configuration), set the **Ollama** endpoint to **http://localhost:11434** (Ollama’s default).
3. In the chat screen, choose the model **globo-assist** from the model selector.

Then ask, for example: *¿Qué es Globomantics?* or *¿Cómo contacto con soporte técnico?* You should get answers consistent with your fine-tuning.

---

## 6. Troubleshooting

| Issue | What to do |
|-------|------------|
| **`ModuleNotFoundError: No module named 'mlx_lm'`** | Use the same Python env where you ran training, or install: `pip install mlx mlx-lm`. |
| **`ollama: command not found`** | Install Ollama from the link above or Homebrew and ensure it’s on your PATH. |
| **`open-webui: command not found`** | Run `pip install open-webui` in the env you use, then `open-webui serve` again. |
| **Fuse fails (e.g. tokenizer error)** | Update: `pip install -U mlx-lm`. If the bug persists, check [mlx-examples](https://github.com/ml-explore/mlx-examples) issues. |
| **Ollama “model not found” in Open WebUI** | Confirm Ollama is running (`curl http://localhost:11434/api/tags`) and that `globo-assist` appears. Point Open WebUI to `http://localhost:11434`. |
| **Wrong or generic answers** | Ensure you selected **globo-assist** in the UI, not another model. Confirm fuse and `ollama create` completed without errors. |

---

## 7. Optional: Streamlit UI (no Open WebUI)

The repo includes a Streamlit app that already targets Ollama and the `globo-assist` model:

```bash
# With Ollama running and globo-assist created
streamlit run 02/demos/interface.py
```

This uses `02/demos/interface.py` and expects `model": "globo-assist"` and `OLLAMA_URL = "http://localhost:11434/api/generate"` (already set).

---

## 8. Summary checklist

- [ ] Python env with `mlx` and `mlx-lm` (same as training if possible).
- [ ] Fuse: `python -m mlx_lm fuse ... --dequantize --export-gguf`.
- [ ] `./fused_model/ggml-model-f16.gguf` exists.
- [ ] Ollama installed and service running.
- [ ] `ollama create globo-assist -f Modelfile` run from project root.
- [ ] `ollama run globo-assist "..."` returns Globomantics-style answers.
- [ ] `pip install open-webui` then `open-webui serve`.
- [ ] In Open WebUI: backend = Ollama at http://localhost:11434, model = **globo-assist**.

Once these are done, your trained model is running in the web UI and ready to use.
