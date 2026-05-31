# Kimi-K2.5 Quick Start

This guide takes you from a fresh clone to a working setup with the fewest steps.
Two paths are supported:

- **Path A — Hosted API (recommended for most users).** Call Kimi-K2.5 over an
  OpenAI/Anthropic-compatible API. No GPU required. This is what the code
  examples in the [README](README.md#6-model-usage) use.
- **Path B — Self-host the weights.** Run the open weights yourself with vLLM,
  SGLang, or KTransformers. Requires substantial multi-GPU (or CPU+GPU) hardware.

> [!IMPORTANT]
> Kimi-K2.5 is a **1T-parameter** Mixture-of-Experts model (32B activated).
> Even with native INT4 quantization it does **not** fit on a single
> consumer GPU (e.g. an RTX 5090 / 32 GB). Self-hosting targets multi-GPU
> nodes (the deploy guide uses 8× H200 with TP8) or KTransformers
> CPU+GPU heterogeneous inference with very large system RAM. If you only
> have a single workstation GPU, use **Path A**.
>
> Ollama is **not** a supported engine for this model, so there is no
> `OLLAMA_HOST` configuration here — use the OpenAI-compatible client below.

---

## Path A — Hosted API

### 1. Create an isolated environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install "openai>=1.0" requests
```

### 2. Set the required environment variables

| Variable | Required | Purpose | Example |
|----------|----------|---------|---------|
| `MOONSHOT_API_KEY` | yes | API key from <https://platform.moonshot.ai> | `sk-...` |
| `OPENAI_BASE_URL` | no | API endpoint (override only when self-hosting) | `https://api.moonshot.ai/v1` |
| `MODEL_NAME` | no | Model identifier to request | `kimi-k2.5` |

```bash
export MOONSHOT_API_KEY="sk-your-key-here"
# Optional overrides:
# export OPENAI_BASE_URL="https://api.moonshot.ai/v1"
# export MODEL_NAME="kimi-k2.5"
```

### 3. Smoke test

Save as `smoke_test.py` and run `python smoke_test.py`:

```python
import os
import openai

client = openai.OpenAI(
    api_key=os.environ["MOONSHOT_API_KEY"],
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.moonshot.ai/v1"),
)
model_name = os.environ.get("MODEL_NAME", "kimi-k2.5")

resp = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "You are Kimi, an AI assistant created by Moonshot AI."},
        {"role": "user", "content": "Reply with exactly: ok"},
    ],
    max_tokens=16,
)
print("response:", resp.choices[0].message.content)
```

A successful run prints a short model reply. See the
[README usage section](README.md#6-model-usage) for image/video, thinking vs.
instant mode, and tool-calling examples.

---

## Path B — Self-host the weights

### 1. Verify your GPU stack first

Run this sequence before installing an engine — it catches the most common
"it won't start" problems early:

```bash
# 1. Driver + GPUs visible?
nvidia-smi

# 2. CUDA-enabled PyTorch sees every GPU?
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'gpus:', torch.cuda.device_count())"

# 3. Per-GPU memory (need enough aggregate VRAM for a 1T INT4 MoE model)?
python -c "import torch; [print(i, torch.cuda.get_device_name(i), round(torch.cuda.get_device_properties(i).total_memory/1e9,1), 'GB') for i in range(torch.cuda.device_count())]"

# 4. transformers meets the minimum version (>= 4.57.1)?
python -c "import transformers; print('transformers', transformers.__version__)"
```

If step 2 reports `cuda: False`, reinstall PyTorch with the CUDA build matching
your driver before continuing.

### 2. Pick an engine and follow the deploy guide

Detailed, copy-pasteable launch commands for each engine live in the
[Model Deployment Guide](docs/deploy_guidance.md):

- **vLLM** (nightly wheel) — single-node TP8 example.
- **SGLang** (latest main) — single-node TP8 example.
- **KTransformers** — CPU+GPU heterogeneous inference and LoRA fine-tuning.

The minimum `transformers` version is **4.57.1**.

### 3. Point the client at your server

Once your server is up (default OpenAI-compatible endpoint), reuse Path A's
client by overriding the endpoint and model name:

```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export MODEL_NAME="/path/to/Kimi-K2.5"   # whatever you launched the server with
export MOONSHOT_API_KEY="EMPTY"          # most local servers ignore the key
python smoke_test.py
```

For self-hosted vLLM/SGLang, use `extra_body={'chat_template_kwargs': {"thinking": False}}`
to switch to instant mode (the official API uses `extra_body={'thinking': {'type': 'disabled'}}`).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `NameError: client is not defined` | Skipped the client setup block | Construct the client as shown in step 3 / [README](README.md#setting-up-the-client) |
| `openai.AuthenticationError` | `MOONSHOT_API_KEY` unset or wrong | Re-export the key; for local servers set it to `EMPTY` |
| `cuda: False` | CPU-only PyTorch installed | Reinstall the CUDA build of PyTorch |
| Server OOMs on launch | Not enough aggregate VRAM | Add GPUs / use KTransformers CPU+GPU offload (see deploy guide) |
