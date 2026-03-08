# Open WebUI setup (Python 3.11 or 3.12 required)

Open WebUI does **not** support Python 3.13. This project’s `.venv` uses 3.13 for MLX, so install and run Open WebUI in a **separate** environment with Python 3.11 or 3.12.

## Option A: Pyenv (if you use pyenv)

```bash
# Create a 3.12 env for Open WebUI
pyenv install -s 3.12.0   # if not already installed
pyenv virtualenv 3.12.0 open-webui
pyenv activate open-webui

pip install open-webui
open-webui serve
```

Then open **http://localhost:8080**. In **Settings → Connections**, add (or set) the Ollama API URL to **http://localhost:11435** and choose model **globo-assist**.

---

## Option B: System or Homebrew Python 3.12

If you have `python3.12` (e.g. from Homebrew):

```bash
python3.12 -m venv ~/open-webui-venv
source ~/open-webui-venv/bin/activate
pip install open-webui
open-webui serve
```

Then in the UI set the Ollama URL to **http://localhost:11435** and use **globo-assist**.

---

## Option C: Docker (no local Python version constraint)

```bash
docker run -d -p 8080:8080 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11435 \
  --add-host=host.docker.internal:host-gateway \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main
```

Then in the Open WebUI settings, set the Ollama API URL to **http://host.docker.internal:11435** so it can reach your Globomantics API on the host.

---

## Port summary

| Service              | Default port | Note                          |
|----------------------|-------------|-------------------------------|
| Globomantics API     | **11435**   | Run with `./run_deployment.sh` |
| Ollama (if you use it) | 11434     | Leave it running if you use it |
| Open WebUI           | 8080        |                               |

In Open WebUI, point the **Ollama** connection to **http://localhost:11435** to use the Globomantics model (**globo-assist**). You can keep Ollama on 11434 for other models and add a second connection for 11435 if you like.
