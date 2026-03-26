# ExLlamaV3 Ampere Deployment Guide

This repo is the working `exllamav3` fork plus local OpenAI-compatible server
scripts used to deploy quantized models on the Ampere machines in this project.

The goal of this README is to be the canonical deployment playbook so future
launches do not require repeated trial-and-error.

## Current best-known deployments

### Fastest stable dual RTX 3060 deployment for Qwen3.5-9B

- Model: `models/Qwen3.5-9B-exl3`
- GPUs: physical `2,3` only
- Backend: layer-split, not TP2
- KV cache: fp16
- Context: `262144`
- GPU split: `9.5,11.0`
- CPU placement: disabled with `EXLLAMA_EMBED_PREFER_CPU=0`
- Measured short decode speed: about `37.8 t/s`
- Thinking is enabled by default in `server_27b_layer.py`, but the server now
  caps the `<think>` phase with `MAX_THINKING_TOKENS` so reasoning cannot consume
  the entire output budget.
- Runner: `run_qwen9b_3060_pair.sh`
- Notes: see `doc/qwen9b_dual_3060_fastest.md`

### Why this is the default recommendation on the 3060 pair

- Native TP2 was not stable on this machine and hit synchronization / CPU-reduce
  timeouts during warmup and prefill.
- NCCL TP2 also failed here because of the CUDA driver/runtime mismatch already
  recorded in project logs.
- Layer-split avoids TP communication problems and is the fastest stable path we
  found that stays fully on GPU.
- The model's default embedding behavior prefers CPU. Setting
  `EXLLAMA_EMBED_PREFER_CPU=0` removes the last CPU placement and keeps the full
  model on the 3060s.

## Golden rules

- Use `conda` and launch from `exl3-dev` unless a model-specific script says
  otherwise.
- Always pin GPUs explicitly with `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
  `CUDA_VISIBLE_DEVICES=...`.
- Prefer fp16 KV cache unless there is a proven need for quantized KV.
- Prefer layer-split on mixed PCIe consumer GPUs when TP2 is unstable.
- Do not assume TP2 is faster just because two GPUs are available. Validate it on
  the exact machine first.
- Keep embeddings on GPU for fully-GPU deployments by setting
  `EXLLAMA_EMBED_PREFER_CPU=0`.
- Use wrapper scripts with fixed model path, GPU selection, split, cache, and log
  path instead of ad-hoc shell commands.
- Treat the model's own `max_position_embeddings` as the hard context limit.

## Environment setup

### Conda

Primary environment:

```bash
source /home/op/miniconda3/etc/profile.d/conda.sh
conda activate exl3-dev
```

Torch libraries on this machine live at:

```bash
/home/op/miniconda3/envs/exl3-dev/lib/python3.11/site-packages/torch/lib
```

Most runner scripts in this repo already export `LD_LIBRARY_PATH` correctly. If
you launch manually, do it yourself:

```bash
export LD_LIBRARY_PATH=/home/op/miniconda3/envs/exl3-dev/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
```

### Models

- Repo-local models: `/home/op/exllamav3_ampere/models`
- External quantized models are also used in some scripts under
  `/home/op/quantized-models`

## Deployment decision tree

### If you are deploying on the two RTX 3060s only

- Start with layer-split.
- Use `server_27b_layer.py` plus a wrapper script.
- Set `EXLLAMA_EMBED_PREFER_CPU=0`.
- Use fp16 KV cache.
- For `Qwen3.5-9B-exl3`, use `run_qwen9b_3060_pair.sh`.

### If you want the fastest deployment method overall

- Use the exact deployment that is already documented and known-good for that
  model and GPU set.
- Reuse an existing wrapper script before inventing a new one.
- For non-ExLlama GPTQ deployments, the `ULTIMATE` vLLM launchers are a good
  reference for wrapper layout, stable logging, and GPU resolution.

### If you are considering TP2

- Use TP2 only when it has already been validated for the exact hardware,
  driver/runtime, and model.
- On this machine, TP2 is not the default recommendation for the 3060 pair.

## Standard deployment workflow

Use this flow for any new model so the result is repeatable.

### 1. Check the target GPUs

```bash
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader
```

If you need a clean 3060-only launch, inspect compute processes on GPUs `2,3`
and stop them before starting the server.

### 2. Choose the backend

- `layer-split`: default for the 3060 pair and for setups where TP comms are the
  main instability risk
- `native TP2`: use only if already proven stable for the target model
- `nccl TP2`: use only if the machine's CUDA driver/runtime combination is known
  to support it cleanly

### 3. Decide the cache target

- Start from the model's `max_position_embeddings`
- Keep fp16 KV unless you have a measured reason to quantize KV
- Align cache tokens to the ExLlama page size (`256` tokens)

### 4. Keep the launch reproducible

Every production-worthy launch should have a wrapper script that fixes:

- conda env
- visible GPUs
- model path
- model id
- GPU split
- cache tokens
- host and port
- log path
- any required environment toggles

### 5. Verify immediately after launch

Run all three:

```bash
curl -fsS http://127.0.0.1:PORT/health
curl -fsS http://127.0.0.1:PORT/v1/models
curl -fsS http://127.0.0.1:PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MODEL_ID",
    "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

Also confirm GPU residency with `nvidia-smi`.

## Recommended dual-3060 Qwen3.5-9B launch

### One-command runner

```bash
/home/op/exllamav3_ampere/run_qwen9b_3060_pair.sh
```

### What it does

- activates `exl3-dev`
- locks execution to `CUDA_VISIBLE_DEVICES=2,3`
- disables CPU embedding fallback with `EXLLAMA_EMBED_PREFER_CPU=0`
- launches `server_27b_layer.py`
- serves `models/Qwen3.5-9B-exl3`
- uses `GPU_SPLIT=9.5,11.0`
- allocates `CACHE_TOKENS=262144`
- writes logs to `logs/qwen9b-3060-pair-current.log`

### Equivalent manual launch

```bash
source /home/op/miniconda3/etc/profile.d/conda.sh
conda activate exl3-dev
export LD_LIBRARY_PATH=/home/op/miniconda3/envs/exl3-dev/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=2,3 \
EXLLAMA_EMBED_PREFER_CPU=0 \
python /home/op/exllamav3_ampere/server_27b_layer.py \
  --host 0.0.0.0 \
  --port 1235 \
  --model /home/op/exllamav3_ampere/models/Qwen3.5-9B-exl3 \
  --gpu-split 9.5,11.0 \
  --cache-tokens 262144
```

### Health checks

```bash
curl -fsS http://127.0.0.1:1235/health
curl -fsS http://127.0.0.1:1235/v1/models
```

### Smoke test

```bash
curl -fsS http://127.0.0.1:1235/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-exl3-3060-dual",
    "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

Note: this model may spend tokens in reasoning first. Use a generous
`max_tokens` value if you need a complete visible answer. For low-latency
direct answers, send `"enable_thinking": false`. For reasoning-heavy prompts,
keep thinking enabled but lower `MAX_THINKING_TOKENS` when the model spends too
long in `<think>`.

## Optimizing Qwen 3.5 scripts

Use these knobs when tuning the Qwen 3.5 launchers and server scripts:

- **Pin the GPUs first.** Keep the 3060 pair on `CUDA_VISIBLE_DEVICES=2,3` and
  use `CUDA_DEVICE_ORDER=PCI_BUS_ID` so the split stays stable across launches.
- **Keep the proven split as the baseline.** Start from `GPU_SPLIT=9.5,11.0` on
  the dual-3060 setup. Only move it if you have a measured VRAM or latency
  reason.
- **Size the cache for the workload.** `CACHE_TOKENS=262144` is the longest
  context configuration that fit reliably in testing. Reduce it if you want more
  headroom or a faster startup path.
- **Control overthinking explicitly.** `MAX_THINKING_TOKENS` is the main lever
  for reasoning-heavy prompts. Smaller values reduce latency and prevent the
  model from spending the entire budget inside `<think>`.
- **Turn thinking off for direct answers.** Set `ENABLE_THINKING=false` in the
  environment or send `"enable_thinking": false` per request when you want a
  short answer with minimal latency.
- **Preserve reasoning only when you need it.** Leave
  `PRESERVE_THINK_OUTPUT=true` if you want to inspect the chain of thought.
  Disable it if you only care about the final answer payload.
- **Validate with the dual-3060 harness.** Run
  `python tests/validate_dual_3060.py` after changing split, cache, or thinking
  settings. It checks health, normal completion, capped thinking, and
  no-thinking behavior.
- **Treat wrapper scripts as the source of truth.** Put the fixed model path,
  GPU pinning, split, cache, and thinking defaults in the wrapper script rather
  than in ad-hoc shell history.

Example tuning profile for fast direct responses:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=2,3 \
EXLLAMA_EMBED_PREFER_CPU=0 \
ENABLE_THINKING=false \
MAX_THINKING_TOKENS=256 \
python /home/op/exllamav3_ampere/server_27b_layer.py \
  --host 0.0.0.0 \
  --port 1235 \
  --model /home/op/exllamav3_ampere/models/Qwen3.5-9B-exl3 \
  --gpu-split 9.5,11.0 \
  --cache-tokens 262144
```

### Qwen3.5-27B on dual RTX 3090s

**TP2 (NCCL) is now the production path.** The previous NCCL failures were
caused by `nvidia-nccl-cu13` (NCCL 2.28.9+cuda13.0) silently overwriting the
correct `nvidia-nccl-cu12` (2.27.5, CUDA 12.8).  After fixing the package and
the `LD_LIBRARY_PATH` order, NCCL initialises cleanly on GPUs 0,1.

**Measured performance**: ~28.5 t/s sustained decode (vs ~25 t/s layer-split,
**+14%**). Peak 29.5 t/s achieved in prior sessions.

Launch with:

```bash
bash /home/op/exllamav3_ampere/launch_27b_gpu01.sh
```

The launcher now:
- Uses `server_27b_tp2.py` (NCCL TP2)
- `CUDA_VISIBLE_DEVICES=0,1`, `GPU_SPLIT=22.0,22.0`
- `CACHE_TOKENS=131072` (128K context)
- `NCCL_ALGO=TREE` for PCIe-connected 3090s
- `LD_LIBRARY_PATH` with `nvidia/cuda_runtime/lib` **first** to ensure CUDA 12.8
  `libcudart.so.12` is found before the conda env's older CUDA 12.1 copy
- `MAX_THINKING_TOKENS=1024` to prevent overthinking

If `nvidia-nccl-cu13` gets reinstalled in the future, NCCL will silently break
again.  Check with:

```bash
conda run -n exl3-dev bash -c \
  'strings $(python -c "import site; print(site.getsitepackages()[0])")/nvidia/nccl/lib/libnccl.so.2 | grep "NCCL version"'
# Must show: NCCL version 2.27.5+cuda12.x  — NOT cuda13
```

## Existing server scripts

These scripts still exist and may be useful for experimentation, benchmarking,
or other GPU layouts:

- `server_openai.py`: single 3060 path for light testing
- `server_3090.py`: single 3090 path
- `server_dual_gpu.py`: older dual-GPU 9B script
- `server_mixed.py`: mixed 3090 + 3060 path
- `server_32k.py`: fixed 32k mixed-GPU 9B variant
- `server_27b_tp2.py`: TP2 server used for TP experiments
- `server_27b_layer.py`: current best general-purpose multi-GPU layer-split
  server in this repo

For new production launches, prefer wrapper scripts over direct use of the older
one-off server entry points.

## Deployment patterns worth reusing

The scripts under `/home/op/ULTIMATE` are a good reference for deployment
discipline even when the backend is different. Reuse these ideas:

- one launcher script per model / port / GPU layout
- fixed log path plus a stable `current.log` symlink
- explicit GPU selection
- health-check script paired with each launcher
- stop script paired with each launcher
- model-specific documented settings kept in repo

## Troubleshooting

### TP2 fails during warmup or prefill

Most likely culprit: `nvidia-nccl-cu13` is installed and has overwritten
`libnccl.so.2` with an NCCL version built for CUDA 13 (requires driver ≥ 575).

```bash
# Diagnose
pip show nvidia-nccl-cu12 nvidia-nccl-cu13
strings /path/to/nvidia/nccl/lib/libnccl.so.2 | grep "NCCL version"
# Fix
pip uninstall nvidia-nccl-cu13 -y
pip install "nvidia-nccl-cu12==2.27.5" --force-reinstall
```

### NCCL backend fails at startup

1. Check which `libnccl.so.2` is being loaded (see above).
2. Ensure `LD_LIBRARY_PATH` prepends `nvidia/cuda_runtime/lib` so CUDA 12.8
   `libcudart.so.12` is resolved before older system or conda copies.
3. As a last resort, fall back to layer-split with `server_27b_layer.py`.

### A module lands on CPU when it should not

- Set `EXLLAMA_EMBED_PREFER_CPU=0`
- Re-launch and confirm residency with `nvidia-smi` and startup logs

### Context launch fails with insufficient VRAM

- Reduce `CACHE_TOKENS`
- Adjust `GPU_SPLIT`
- Keep the model on the intended GPUs only

### The server answers with reasoning but not the final short text

- Raise `max_tokens`
- Reduce `MAX_THINKING_TOKENS` if the model spends too long reasoning
- Set `ENABLE_THINKING=false` for direct-answer workloads
- This is expected for some reasoning-tuned variants

## Related docs

- `doc/qwen9b_dual_3060_fastest.md`
- `doc/quantizer_ampere_findings.md`
- `BENCHMARKED.md`
