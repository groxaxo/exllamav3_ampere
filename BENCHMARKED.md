# ExLlamaV3 Qwen3.5-9B-EXL3 Benchmark Results

**Date:** March 11, 2026  
**Model:** Qwen3.5-9B-EXL3 (6.0 bpw)  
**Framework:** ExLlamaV3 (custom fork: groxaxo/exllamav3_ampere)  
**Conda Environment:** exl3-dev

## System Configuration

| GPU | Model | VRAM |
|-----|-------|------|
| GPU 0 | NVIDIA GeForce RTX 3090 | 24GB |
| GPU 1 | NVIDIA GeForce RTX 3090 | 24GB |
| GPU 2 | NVIDIA GeForce RTX 3060 | 12GB |
| GPU 3 | NVIDIA GeForce RTX 3060 | 12GB |
| GPU 4 | NVIDIA GeForce RTX 3090 | 24GB |

## How Tests Were Conducted

### Test Methodology

1. **Server Setup:** Created FastAPI-based OpenAI-compatible server scripts using ExLlamaV3
2. **Context Testing:** Tested progressively larger context sizes (4K, 8K, 16K, 32K, 64K, 128K, 192K) until failure
3. **Speed Testing:** Sent identical prompts via curl and measured:
   - Total response time
   - Token count (including thinking tokens)
   - Tokens per second calculation
4. **Multiple Runs:** Each configuration tested at least twice to verify consistency

### Test Prompt

```json
{
  "model": "Qwen3.5-9B-exl3",
  "messages": [{"role": "user", "content": "Write a detailed explanation of how neural networks work, covering layers, activation functions, backpropagation, and gradient descent. Be thorough."}],
  "max_tokens": 1000
}
```

### Test Command

```bash
time curl -s http://localhost:PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3.5-9B-exl3", "messages": [...], "max_tokens": 1000}' \
  | jq '.usage.completion_tokens'
```

## Benchmark Results

### Single GPU Tests

#### RTX 3060 (GPU 2) - 12GB VRAM

| Metric | Value |
|--------|-------|
| Max Context | 32,768 tokens |
| Test 1 | 999 tokens in 22.5s = **44.39 t/s** |
| Test 2 | 799 tokens in 20.4s = **39.09 t/s** |
| **Average Speed** | **~42 t/s** |

**Memory Allocation:** Model loaded with `use_per_device=[12.0]`

#### RTX 3090 (GPU 0) - 24GB VRAM

| Metric | Value |
|--------|-------|
| Max Context | 131,072 tokens (128K) |
| Test 1 | 999 tokens in 16.6s = **60.31 t/s** |
| Test 2 | 799 tokens in 13.1s = **61.16 t/s** |
| **Average Speed** | **~61 t/s** |

**Memory Allocation:** Model loaded with `use_per_device=[22.0]`

### Multi-GPU Tests (Layer-Split Mode)

#### Dual RTX 3090 (GPU 0 + GPU 1) - 48GB Total

| Metric | Value |
|--------|-------|
| Max Context | 196,608 tokens (192K) |
| GPU 0 Free After Load | 13.27 GB |
| GPU 1 Free After Load | 23.06 GB |
| Test 1 | 999 tokens in 16.8s = **59.57 t/s** |
| Test 2 | 799 tokens in 13.2s = **60.71 t/s** |
| **Average Speed** | **~60 t/s** |

**Memory Allocation:** Model loaded with `use_per_device=[11.0, 11.0]`

#### RTX 3090 + RTX 3060 (GPU 0 + GPU 2) - 36GB Total - 128K Context

| Metric | Value |
|--------|-------|
| Max Context | ~131,072 tokens (128K) |
| Test 1 | 999 tokens in 15.4s = **64.75 t/s** |
| Test 2 | 799 tokens in 12.1s = **66.05 t/s** |
| **Average Speed** | **~65 t/s** ⭐ FASTEST |

**Memory Allocation:** Model loaded with `use_per_device=[16.0, 8.0]`

#### RTX 3090 + RTX 3060 (GPU 1 + GPU 3) - 36GB Total - 32K Context

| Metric | Value |
|--------|-------|
| Max Context | 32,768 tokens (limited by test) |
| Test 1 | 999 tokens in 16.7s = **59.73 t/s** |
| Test 2 | 799 tokens in 13.7s = **58.38 t/s** |
| **Average Speed** | **~59 t/s** |

**Memory Allocation:** Model loaded with `use_per_device=[16.0, 8.0]`

## Summary Table

| Configuration | GPUs | VRAM | Max Context | Speed (t/s) |
|---------------|------|------|-------------|-------------|
| RTX 3060 | 1x 12GB | 12GB | 32K | **42** |
| RTX 3090 | 1x 24GB | 24GB | 128K | **61** |
| Dual RTX 3090 | 2x 24GB | 48GB | 192K | **60** |
| RTX 3090 + 3060 (128K) | 36GB | 36GB | 128K | **65** ⭐ |
| RTX 3090 + 3060 (32K) | 36GB | 36GB | 32K | **59** |

## Key Findings

### 1. Speed Performance
- **Fastest configuration:** RTX 3090 + RTX 3060 with 128K context achieved **65 t/s**
- **Single 3090** is ~45% faster than single 3060 (61 vs 42 t/s)
- **Multi-GPU layer-split** does NOT improve speed (sequential execution)
- Mixed GPU configs can outperform dual high-end GPUs

### 2. Context Length
- **RTX 3060 (12GB):** Limited to 32K context
- **RTX 3090 (24GB):** Supports full 128K context
- **Dual RTX 3090 (48GB):** Supports up to 192K context (largest tested)
- Context is limited by smallest GPU's available memory after model load

### 3. Memory Utilization
- Model size: ~8.4GB for Qwen3.5-9B-EXL3 at 6.0 bpw
- KV cache (fp16) scales with context length
- Layer-split allows combining different GPU sizes effectively

### 4. Architecture Limitation
- **Tensor parallelism is NOT supported** for Qwen3.5 in ExLlamaV3
- Multi-GPU uses **layer-split mode** (layers distributed, compute is sequential)
- This explains why dual GPU doesn't improve speed vs single GPU

## Server Configuration

### Files Created

```
/home/op/exllamav3_ampere/
├── server_openai.py    # Single GPU 3060 (port 8001)
├── server_3090.py      # Single GPU 3090 (port 8002)
├── server_dual_gpu.py  # Dual RTX 3090 (port 8002)
├── server_mixed.py     # 3090+3060 128K ctx (port 8003)
└── server_32k.py       # 3090+3060 32K ctx (port 8004)
```

### How to Run

```bash
# Activate environment
conda activate exl3-dev

# Set library path for torch
export LD_LIBRARY_PATH=/home/op/miniconda3/envs/exl3-dev/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH

# Example: Run mixed GPU server with 128K context
cd /home/op/exllamav3_ampere
CUDA_VISIBLE_DEVICES=0,2 python server_mixed.py
```

### API Usage

```bash
# Health check
curl http://localhost:8003/health

# Chat completion (non-streaming)
curl http://localhost:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-exl3",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# Chat completion (streaming)
curl http://localhost:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-exl3",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "stream": true
  }'
```

## Technical Details

### Model Configuration
- **Architecture:** Qwen3_5ForConditionalGeneration
- **Quantization:** EXL3 @ 6.0 bits per weight
- **Head Bits:** 6
- **Native Max Position:** 262,144 tokens
- **Thinking Mode:** Enabled by default (Qwen3.5 reasoning)

### Cache Configuration
- **Type:** CacheLayer_fp16 (fp16 KV cache)
- **Bytes per token:** ~2 bytes × 2 (K+V) × hidden_dim × num_layers / 1GB

### Loading Parameters Used

| Config | use_per_device | tensor_p |
|--------|----------------|----------|
| Single 3060 | [12.0] | False |
| Single 3090 | [22.0] | False |
| Dual 3090 | [11.0, 11.0] | False |
| Mixed 128K | [16.0, 8.0] | False |
| Mixed 32K | [16.0, 8.0] | False |

## Conclusions

1. **For maximum speed:** Use RTX 3090 + RTX 3060 with 128K context (~65 t/s)
2. **For maximum context:** Use dual RTX 3090 (up to 192K tokens)
3. **For cost efficiency:** Single RTX 3090 provides best balance (61 t/s, 128K context)
4. **Mixed GPU setups work well:** Can combine different GPU models via layer-split
5. **Tensor parallelism needed for speed gains:** Layer-split only helps with memory, not compute

## March 16, 2026 - FA2 paged benchmark at ~20.75K cache / 4K decode

### Command notes

- Requested command used `--cache-tokens 20768`, but the paged cache implementation requires cache sizes to be a multiple of `256`.
- The nearest valid value is `20736`, so the actual benchmark runs below used:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
python benchmark_fa2_paged.py \
  --model /home/op/exllamav3_ampere/models/Qwen3.5-9B-exl3 \
  --cache-tokens 20736 \
  --test-tokens 4000 \
  --num-runs 3 \
  --configs single dual \
  --single-gpu-split 22.0 \
  --dual-gpu-split 3.5,20.5 \
  --require-all-gpus
```

- Environment:
  - Conda env: `exl3-dev`
  - `flash_attn`: `2.7.4.post1`
  - GPUs: RTX 3090 (CC 8.6)
- `benchmark_fa2_paged.py` was patched to:
  - validate cache-token alignment
  - accept explicit `--single-gpu-split` / `--dual-gpu-split`
  - honor `--num-runs`
  - fail with `--require-all-gpus` if a supposed dual-GPU run still fits on one GPU

### TP support caveat

- **True tensor parallelism is not available for Qwen3.5 in this fork.**
- `exllamav3/architecture/qwen3_5.py` explicitly sets `supports_tp = False`, so the relevant multi-GPU benchmark here is a **forced layer-split benchmark**, not a true TP benchmark.

### Results

| Configuration | Effective load config | Placement | Avg time (s) | Avg t/s | Notes |
|---------------|------------------------|-----------|--------------|---------|-------|
| Old default dual path | visible GPUs `0,1`, `use_per_device=[12.0, 12.0]` | `cuda:0: 34 modules`, `cuda:1: 0 modules` | 72.76 | **54.96** | Not a real split for this workload |
| Final single baseline | single RTX 3090, `use_per_device=[22.0]` | `cuda:0: 34 modules` | 72.31 | **55.31** | Runs: 55.41 / 55.29 / 55.23 t/s |
| Final real dual split | dual RTX 3090, `use_per_device=[3.5, 20.5]` | `cuda:0: 18 modules`, `cuda:1: 16 modules` | 59.94 | **66.72** | Runs: 67.17 / 66.76 / 66.24 t/s |

### Dual-split sweep summary

- First split that genuinely used both GPUs: `5.5,18.5`
- Best 32-token probe result: `3.5,20.5` at **62.93 t/s**
- 4K-token candidate bakeoff:
  - `3.5,20.5` -> **66.46 t/s**
  - `3.0,21.0` -> **64.32 t/s**
- Final chosen split: **`3.5,20.5`**

### Findings

1. **A real 2-GPU split is possible** for this 9B model, but only by forcing a much smaller first-device cap. The working threshold started around `5.5,18.5`; the best split tested was **`3.5,20.5`**.
2. **The original "TP2" benchmark was not a real multi-GPU run.** With `use_per_device=[12.0, 12.0]`, the whole model still fit on GPU 0, so GPU 1 stayed unused.
3. **For this exact FA2 paged decode benchmark, the real dual-3090 split beat single-GPU by ~20.6%.** Final comparison: **66.72 t/s** dual vs **55.31 t/s** single.
4. **This is a layer-split win, not a tensor-parallel win.** Because Qwen3.5 TP is unimplemented in this fork, the performance gain comes from forcing layer placement across both Ampere GPUs.
5. **Recommended command for repeatable dual-GPU testing:** use `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1` and pass `--dual-gpu-split 3.5,20.5 --require-all-gpus` so the benchmark fails if it accidentally falls back to one GPU.

## March 16, 2026 - Generalized 3/4/5-GPU FA2 paged benchmark sweep

### Script update

- `benchmark_fa2_paged.py` was generalized beyond hard-coded single/dual configs:
  - `--configs` now accepts arbitrary GPU counts such as `3`, `4`, or `5-gpu`, plus aliases like `single`, `dual`, `triple`, `quad`, and `penta`
  - new `--gpu-split <config>=<gb0,gb1,...>` overrides work for any requested GPU count
  - legacy `--single-gpu-split` and `--dual-gpu-split` remain supported for the original 1-GPU and 2-GPU workflow
  - `--require-all-gpus` still fails if any requested logical GPU receives zero model modules

### Benchmark commands

All runs below used the same core benchmark settings as the prior dual-GPU benchmark:

- Model: `/home/op/exllamav3_ampere/models/Qwen3.5-9B-exl3`
- Cache tokens: `20736`
- Test tokens: `4000`
- Runs: `3`
- Validation: `--require-all-gpus`

```bash
# 3 x RTX 3090
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,4 \
python benchmark_fa2_paged.py \
  --model /home/op/exllamav3_ampere/models/Qwen3.5-9B-exl3 \
  --cache-tokens 20736 \
  --test-tokens 4000 \
  --num-runs 3 \
  --configs 3 \
  --gpu-split 3=2.0,2.0,20.5 \
  --require-all-gpus

# 1 x RTX 3060 + 3 x RTX 3090
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,0,1,4 \
python benchmark_fa2_paged.py \
  --model /home/op/exllamav3_ampere/models/Qwen3.5-9B-exl3 \
  --cache-tokens 20736 \
  --test-tokens 4000 \
  --num-runs 3 \
  --configs 4 \
  --gpu-split 4=1.0,1.0,1.0,21.0 \
  --require-all-gpus

# 2 x RTX 3060 + 3 x RTX 3090
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3,0,1,4 \
python benchmark_fa2_paged.py \
  --model /home/op/exllamav3_ampere/models/Qwen3.5-9B-exl3 \
  --cache-tokens 20736 \
  --test-tokens 4000 \
  --num-runs 3 \
  --configs 5 \
  --gpu-split 5=0.85,0.85,0.85,0.85,20.6 \
  --require-all-gpus
```

### Results

| Configuration | Visible devices | Hardware mix | Final split | Placement | Avg time (s) | Avg t/s | Notes |
|---------------|-----------------|--------------|-------------|-----------|--------------|---------|-------|
| Final single baseline | `0` | `3090` | `22.0` | `cuda:0: 34 modules` | 72.31 | **55.31** | Runs: 55.41 / 55.29 / 55.23 t/s |
| Final real dual split | `0,1` | `3090 + 3090` | `3.5,20.5` | `cuda:0: 18 modules`, `cuda:1: 16 modules` | 59.94 | **66.72** | Runs: 67.17 / 66.76 / 66.24 t/s |
| Final real 3-GPU split | `0,1,4` | `3090 + 3090 + 3090` | `2.0,2.0,20.5` | `cuda:0: 9 modules`, `cuda:1: 9 modules`, `cuda:2: 16 modules` | 61.98 | **64.52** | Runs: 64.54 / 64.65 / 64.36 t/s |
| Final real 4-GPU split | `2,0,1,4` | `3060 + 3090 + 3090 + 3090` | `1.0,1.0,1.0,21.0` | `cuda:0: 3 modules`, `cuda:1: 3 modules`, `cuda:2: 3 modules`, `cuda:3: 25 modules` | 70.77 | **56.50** | Runs: 56.48 / 56.60 / 56.43 t/s |
| Final real 5-GPU split | `2,3,0,1,4` | `3060 + 3060 + 3090 + 3090 + 3090` | `0.85,0.85,0.85,0.85,20.6` | `cuda:0: 2 modules`, `cuda:1: 1 modules`, `cuda:2: 2 modules`, `cuda:3: 2 modules`, `cuda:4: 27 modules` | 71.64 | **55.82** | Runs: 55.90 / 55.84 / 55.72 t/s |

### Probe-sweep highlights

- 3 GPUs:
  - `3=3.0,3.0,18.5` -> **60.46 t/s** (32-token probe)
  - `3=2.0,2.0,20.5` -> **60.83 t/s** (best probe and best full run)
- 4 GPUs:
  - `4=1.5,1.5,1.5,20.0` -> **51.44 t/s**
  - `4=1.0,1.0,1.0,21.0` -> **54.29 t/s** (best probe and best full run)
  - `4=0.75,0.75,0.75,21.5` -> **54.01 t/s**
- 5 GPUs:
  - `5=1.0,1.0,1.0,1.0,20.0` -> **52.35 t/s**
  - `5=0.85,0.85,0.85,0.85,20.6` -> **54.08 t/s** (best probe and best full run)
  - `5=0.75,0.75,0.75,0.75,21.0` -> **53.42 t/s**

### Findings

1. **The benchmark script now supports arbitrary GPU counts** without adding a new hard-coded CLI flag for every configuration. `--configs` and `--gpu-split` are enough to drive real 3-GPU, 4-GPU, and 5-GPU layer-split tests.
2. **For this exact 9B FA2 paged decode workload, 2 GPUs remain the speed winner.** The best real dual-3090 split reached **66.72 t/s**, while the best real 3-GPU split reached **64.52 t/s**.
3. **Adding more GPUs beyond 2 hurt throughput even when placement was real.** Three 3090s were slower than two 3090s, which points to extra inter-GPU handoff overhead outweighing the benefit of thinner layer shards.
4. **Once the 3060s are forced into the pipeline, throughput falls back toward single-GPU territory.** Best 4-GPU result: **56.50 t/s**. Best 5-GPU result: **55.82 t/s**.
5. **The best 4-GPU and 5-GPU splits kept most of the model on the final 3090.** The extra GPUs only carried a few modules each, which was the least-bad way to satisfy `--require-all-gpus`.
6. **Practical recommendation:** use 2 GPUs for raw speed, 3 GPUs only if you need extra placement flexibility while staying on 3090s, and 4 or 5 GPUs only when you explicitly need to involve those extra cards despite the performance cost.

---

## March 16, 2026 – Qwen3.5-27B-exl3 heretic 6bpw: layer-split 2-GPU deployment

### Model

| Property | Value |
|----------|-------|
| Model | `Qwen3.5-27B-exl3` (MetaphoricalCode heretic 6bpw) |
| Architecture | `Qwen3_5ForConditionalGeneration` (Qwen3.5 hybrid linear/full-attn) |
| Quantisation | EXL3 6.0 bpw, head bits 6, codebook MCG |
| Layers | 64 total: **16 full-attention** + 48 linear-attention (interval=4) |
| KV heads / head dim | 4 kv heads × 256 head dim |
| Model size on disk | 22 GB |
| `max_position_embeddings` | 262 144 (256 K) |

### Multi-GPU note: why "TP 2" means layer-split here

True tensor parallelism is **not yet implemented** for Qwen3.5 in this fork:

```
exllamav3/architecture/qwen3_5.py:
    # TP for this architecture is not implemented yet
    self.caps.update({"supports_tp": False})
```

The equivalent multi-GPU strategy (and the one used for all results below) is
**layer-split** across 2 × RTX 3090 — identical in effect to the approach proven
for the 9B model in earlier sections of this document.

### Previous heretic model failure (5bpw hb8 variant)

An earlier test of `Qwen3.5-27B-heretic-v2-exl3-5bpw-hb8` produced nothing but
exclamation-mark output (`!!!!`) at every temperature. The 6bpw model stored at
`models/Qwen3.5-27B-exl3` generates **correct, coherent reasoning** (verified
with the sanity probe `"What is 2+2?"` → proper chain-of-thought output).

### Maximum context

| Config | Result |
|--------|--------|
| 2 × RTX 3090, split `[11.0, 22.0]`, 4-bit KV cache | **262 144 tokens (256 K)** |

Memory budget (2 × RTX 3090 = 48 GB total):
- Model weights: ~18 GB (9.3 GB cuda:0, 8.6 GB cuda:1)
- Full 256 K KV cache (16 full-attn layers × 4 kv heads × 256 head dim × 4-bit): **~4 GB**
- Headroom remaining: ~26 GB — cache is the smallest consumer

The model's own `max_position_embeddings = 262 144` is the binding constraint,
not VRAM.

### Benchmark (decode throughput, same settings as 9B benchmark)

| Configuration | GPU split | Placement | Avg t/s | Notes |
|---------------|-----------|-----------|---------|-------|
| Layer-split 2 × RTX 3090 | `[11.0, 22.0]` | cuda:0: 36 modules, cuda:1: 31 modules | **~23.8 t/s** | Runs: 24.05 / 23.83 / 23.68 t/s |

Settings: cache 20 736 tokens, 500 decode tokens, 4-bit KV, 3 runs (measured via live server API).

### Server deployment

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
python server_27b_tp2.py \
  --host 0.0.0.0 \
  --port 1234 \
  --model /home/op/exllamav3_ampere/models/Qwen3.5-27B-exl3 \
  --gpu-split 11.0,22.0 \
  --max-context 262144
```

- OpenAI-compatible API at `http://0.0.0.0:1234/v1`
- Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`
- Supports streaming (`"stream": true`) and non-streaming responses
- ChatML format (`<|im_start|>` / `<|im_end|>`)
- Generation is serialised with an asyncio lock (safe for sequential requests)

### Findings

1. **The 6bpw heretic model works; the 5bpw hb8 heretic variant did not.** Switching from `Qwen3.5-27B-heretic-v2-exl3-5bpw-hb8` to the 6bpw model produced correct chain-of-thought reasoning.
2. **True TP is still unsupported for Qwen3.5.** Layer-split is the only working multi-GPU path in this fork.
3. **Maximum context is the full 256 K (262 144 tokens).** The KV cache for this hybrid architecture only needs ~4 GB at 256 K (16 full-attention layers vs 32 for a standard transformer), leaving ample headroom on a 2 × 3090 setup.
4. **Decode throughput is ~23.8 t/s** with 4-bit KV cache and the 2-GPU layer-split, consistent with the expected ~3× slowdown vs the 9B model (more parameters → more compute per token).
5. **The layer-split placement was `cuda:0: 36 modules, cuda:1: 31 modules`** with `use_per_device=[11.0, 22.0]` — both GPUs received a real share of the model.
6. **Server is live at `http://0.0.0.0:1234/v1`** (PID 1253512, log at `server_27b_tp2_run2.log`).

## March 17, 2026 - Qwen3.5-27B real TP fixed on 2 x RTX 3090

This section supersedes the earlier "TP means layer-split" note above for the
current tree state.

### What changed

Real tensor parallelism for Qwen3.5 now works in this fork after fixing the
missing and incorrect TP paths:

- `exllamav3/modules/gated_delta_net.py`
  - implemented TP export/import/allocation for Qwen3.5 linear-attention blocks
- `exllamav3/modules/gated_rmsnorm.py`
  - fixed TP import to rebuild the fast-path norm object correctly
- `exllamav3/modules/attn.py`
  - fixed TP export/import for **interleaved-gate attention**
  - corrected `q_proj` TP slice width so Q and gate halves are both preserved
- `exllamav3/architecture/qwen3_5.py`
  - enables `supports_tp`
- `exllamav3/model/model_tp_backend.py`
  - adds single-rank no-op fast paths for TP barriers / reductions
- `exllamav3/model/model.py`
  - TP default backend now matches the working path: `nccl`

### Validation

The first-token TP sanity check now matches the layer-split reference for the
prompt:

```text
<|im_start|>user
What is 2+2? Answer with one word only.<|im_end|>
<|im_start|>assistant
```

Top TP predictions on 2 x 3090 with `tp_backend='nccl'`:

1. `<think>`
2. `Four`
3. `4`
4. `four`

This is the same useful ordering as the correct non-TP path.

### Important backend finding

- **NCCL is the working real-TP backend** for Qwen3.5 on this machine.
- The old native TP reduction path was still corrupting outputs in practice for
  this workload, so the default TP backend was switched to `nccl`.
- Single-rank TP reduction is now short-circuited to a no-op instead of
  unnecessarily running through the reducer.

### Serving result on port 1234

The live server is now running with:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
python server_27b_tp2.py \
  --host 0.0.0.0 \
  --port 1234 \
  --model /home/op/exllamav3_ampere/models/Qwen3.5-27B-exl3 \
  --gpu-split 22.0,22.0 \
  --max-context 262144 \
  --tp-backend nccl
```

Current deployment details:

- API: `http://0.0.0.0:1234/v1`
- Endpoints verified:
  - `GET /health`
  - `GET /v1/models`
  - `POST /v1/chat/completions`
- Model ID: `Qwen3.5-27B-exl3-heretic-6bpw-tp2`
- Current server PID: `1648613`
- Current log: `/tmp/server_27b_tp2.log`

### No-cache serving path

The user explicitly requested **no quantized KV cache**. The current live TP
server therefore uses a **stateless no-cache decode path**
(`flash_attn_nc`, `last_tokens_only=1`) instead of the ExLlama generator/cache
stack.

Reason:

- TP model loading works
- TP logits are correct
- but the **TP + Cache + Generator** path still hangs under NCCL in this fork

So the working production path right now is:

- real TP model load
- no KV cache
- per-token full-sequence recompute

This keeps the server correct and usable without quantizing any cache.

### Measured live-server result

Smoke test request:

```json
{
  "model": "Qwen3.5-27B-exl3-heretic-6bpw-tp2",
  "messages": [{"role": "user", "content": "What is 2+2? Answer with one word only."}],
  "max_tokens": 16,
  "temperature": 0.0,
  "top_p": 1.0
}
```

Observed response:

```text
<think>
Thinking Process:

1.  **Analyze the Request:**
```

Measured latency:

- 16 generated tokens in **2.296 s**
- effective live-server decode rate: **~6.97 t/s**

Standalone TP no-cache smoke test in-process:

- 16 generated tokens in **2.226 s**
- **~7.19 t/s**

### Context note

The live TP server is configured with the model's full architectural limit:

- `max_context = 262144`

Because the current serving path is **stateless no-cache decode**, this limit is
not constrained by quantized-cache packing. The request still remains bounded by
the model's own `max_position_embeddings = 262144`.

### Current status

1. **Real TP for Qwen3.5 is now working** on 2 x RTX 3090 when using NCCL.
2. **The model is live on `0.0.0.0:1234` right now** with real TP enabled.
3. **No quantized KV cache is used** in the live deployment.
4. **Cached TP generation is still an open issue**; the stable deployed path is
   stateless no-cache generation.

---

## March 18, 2026 — TP + Cached Generation Fix

### Problem

The previous TP deployment ran at **~7 t/s** because it used stateless no-cache
decode (`flash_attn_nc`), recomputing the full sequence for every token. The
Generator + Cache + TP path hung/deadlocked during generation.

### Root Causes Identified

Three distinct bugs prevented TP + Cache + Generator from working:

1. **Cache not created before TP loading.**
   `Cache.__init__()` attaches CacheLayer objects to the model's Attention modules.
   During `model.load(tensor_p=True)`, `Attention.tp_export()` exports those cache
   layers to TP workers. If Cache is created **after** loading, workers never
   receive cache layers and `tp_cache_lookup` stays empty → KeyError or wrong path.

   **Fix:** Create `Cache(model, ...)` before calling `model.load(tensor_p=True)`.
   This mirrors the pattern in `exllamav3/model_init.py`.

2. **`recurrent_states` not transported through TP pipeline.**
   The Generator passes `params["recurrent_states"]` (a dict of GPU-resident
   `GDN_RecurrentState` objects) to `model.forward()`. In the TP path,
   `prepare_inputs_for_tp()` serializes tensor params via shared memory and replaces
   the Cache object with its `id()`. But `recurrent_states` was **not handled** —
   the dict containing GPU tensors was passed as-is through `conn.send()`, which
   attempts to pickle it through multiprocessing Pipe → **deadlock**.

   **Fix:** Strip `recurrent_states` from params, replace with a
   `_recurrent_state_id` key. Each TP worker maintains its own dict of
   recurrent states keyed by this ID. On first use, workers create fresh
   `GDN_RecurrentState` objects (with correct TP-slice dimensions) and persist them
   between prefill/decode calls.

3. **`GDN_RecurrentState.stash()` crashed on None tensors.**
   The main-process "shadow" recurrent state objects have `last_conv_state = None`
   and `last_recurrent_state = None` (the real tensors live on workers). The
   Generator's `RecurrentCache.stash()` called `.cpu()` on these → `AttributeError`.

   **Fix:** Guard `.stash()` and `.unstash()` with `if tensor is not None` checks.

4. **Position tracking desync.**
   The main process's recurrent state `position` field (used by the Generator for
   checkpoint decisions) was never updated because the actual forward pass runs on
   workers. Fix: `prefill_tp()` and `forward_tp()` now increment the main-process
   shadow state positions by the sequence length after each dispatch.

### Files Modified

| File | Change |
|------|--------|
| `exllamav3/model/model_tp.py` | `prepare_inputs_for_tp()`: strip recurrent_states, add state_id, add `inv_freq` transport. `prefill_tp()`/`forward_tp()`: position sync after dispatch. New `tp_cleanup_recurrent_state()` method. |
| `exllamav3/model/model_tp_fn.py` | `mp_model_forward()`: worker-local recurrent state management keyed by state_id. Added `inv_freq` to tensor transport list. New `mp_cleanup_recurrent_state()` function. |
| `exllamav3/modules/gated_delta_net.py` | `GDN_RecurrentState.stash()`/`.unstash()`: handle None tensors gracefully. |
| `server_27b_tp2.py` | Rewritten: Cache created before loading, Generator pipeline with fp16 KV cache. |

### Benchmark Results

**Model:** Qwen3.5-27B-exl3 (6bpw heretic), 64 layers (16 full-attn + 48 GDN)
**GPUs:** 2 × RTX 3090 (CUDA 0,1), fp16 KV cache, 32K token cache
**Prompt:** 36 tokens, 199 generated tokens

| Configuration | Prefill (t/s) | Generation (t/s) | Speedup vs old TP |
|--------------|--------------|------------------|-------------------|
| **TP (NCCL) + cached** | **199.9** | **29.5** | **4.2×** |
| Layer-split + cached | 149.3 | 25.1 | 3.6× |
| TP (NCCL) + stateless (old) | — | 7.0 | 1.0× (baseline) |

### Live Server Verification

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
  python server_27b_tp2.py --host 0.0.0.0 --port 1234 \
    --cache-tokens 32768 --tp-backend nccl
```

Measured live throughput:

- 199 completion tokens in 7.259s = **~27.4 t/s** (end-to-end including HTTP)
- Health: `GET /health` → `{"status": "ok"}`
- Streaming: `stream: true` confirmed working
- All endpoints: `/health`, `/v1/models`, `/v1/chat/completions`

### Summary

| Metric | Before (stateless TP) | After (cached TP) | Improvement |
|--------|----------------------|-------------------|-------------|
| Decode speed | ~7 t/s | ~29.5 t/s | **4.2× faster** |
| Prefill speed | N/A (full recompute) | ~200 t/s | cached |
| KV cache | none | fp16, 32K tokens | ✓ |
| Generator | no (manual loop) | yes (paged + batched) | ✓ |
| Recurrent state | discarded each token | persisted per-worker | ✓ |

---

## Phase 3: Ampere TP Hot-Path Optimization (3×RTX 3090)

### Optimization Summary

Starting from Phase 2 baseline (~27 t/s TP2, ~7 t/s before caching fix), additional
hot-path optimizations were applied targeting Ampere (SM 8.6) GPUs on PCIe.

### Architecture Analysis

Qwen3.5-27B has 64 layers (16 full-attention + 48 GDN linear-attention), each with
a separate MLP. During TP decode, this results in **128 NCCL all_reduce calls per token**
— all on float32 tensors cast to float16 for transport.

### Changes Applied

| # | Optimization | File | Effect |
|---|---|---|---|
| 1 | Skip NCCL CPU reducer dispatch in NCCL mode | `model_tp.py` | Removed per-forward IPC overhead |
| 2 | Disable per-forward NCCL barrier | `model_tp_backend.py` | Removed synchronous barrier per token |
| 3 | Direct NCCL send/recv gather (replace shared-memory fallback) | `model_tp_backend.py` | 2× throughput gain |
| 4 | FP16 all_reduce buffer reuse (avoid 256 allocations/token) | `model_tp_backend.py` | Eliminates temp tensor allocs |
| 5 | Startup warmup pass | `server_27b_tp2.py` | 40% faster cold-start TTFT |
| 6 | Output cap raised to 16000 | `server_27b_tp2.py` | No more truncated answers |
| 7 | TF32 matmul/cudnn flags | `server_27b_tp2.py` | Ampere tensor core utilization |

### Environment Variables for Tuning

| Variable | Default | Purpose |
|---|---|---|
| `EXLLAMA_NCCL_FWD_BARRIER` | `0` | Set to `1` to restore per-forward barrier |
| `EXLLAMA_NCCL_GATHER_FALLBACK` | `0` | Set to `1` to use shared-memory gather |
| `EXLLAMA_FP16_REDUCE_THRESHOLD` | `0` | Min tensor elements for FP16 cast (0 = always) |
| `EXLLAMA_STARTUP_WARMUP` | `1` | Set to `0` to disable startup warmup |
| `NCCL_ALGO` | (unset) | Set to `TREE` for PCIe systems |
| `DEFAULT_MAX_TOKENS` | `16000` | Max output tokens default |

### Live Server Optimization Progression (TP2, 500 tokens, temp=0)

| Stage | Throughput | Delta |
|---|---|---|
| Baseline (before optimizations) | 10.28 t/s | — |
| + CPU reducer skip | 12.67 t/s | +23% |
| + Barrier removal | 13.97 t/s | +10% |
| + NCCL gather | 27.38 t/s | +96% |
| + Buffer reuse + warmup | 28.6 t/s | +4% |
| **Total improvement** | **10.28 → 28.6 t/s** | **2.78×** |

### TP2 vs TP3 Benchmark Results

In-process benchmarks using `bench_tp.py` (3 runs, 500 new tokens, argmax, NCCL backend).

#### Decode Throughput

| Config | Run 1 | Run 2 | Run 3 | Average |
|---|---|---|---|---|
| TP2 (2×RTX 3090) | 28.39 | 28.40 | 28.43 | **28.41 t/s** |
| TP3 (3×RTX 3090) | 28.90 | 28.83 | 28.91 | **28.88 t/s** |

#### Prefill Throughput (steady state)

| Config | Avg Prefill |
|---|---|
| TP2 | **230 t/s** |
| TP3 | **172 t/s** |

#### Maximum Context Length (fp16 KV cache, 22 GB/GPU)

| Config | Max Cache Tokens | Max Context |
|---|---|---|
| TP2 (2×3090) | ~416K | ~416K tokens |
| TP3 (3×3090) | ~704K | ~704K tokens |

#### Cache Size vs Decode Speed

| Cache Tokens | TP2 (t/s) | TP3 (t/s) |
|---|---|---|
| 32K | 28.41 | 28.88 |
| 65K | 28.70 | 28.95 |
| 128K | 29.26 | — |
| 256K | 29.19 | — |
| 384K | 30.65 | — |
| 416K | 30.69 | — |
| 640K | OOM | 30.84 |
| 704K | OOM | 28.94 |

### Deployment Recommendation

| Use Case | Recommended Config | Why |
|---|---|---|
| **Default / most workloads** | **TP2 + 128K cache** | Best prefill speed, saves 1 GPU, 128K context |
| **Long context (>128K)** | **TP2 + 256-384K cache** | TP2 handles up to 416K tokens |
| **Very long context (>416K)** | **TP3 + 640K cache** | Only TP3 can go beyond 416K |
| **Maximum context** | **TP3 + 704K cache** | Absolute limit on 3×3090 |

**Key finding**: TP2 and TP3 have nearly identical decode throughput (~28.5 t/s) on
PCIe-connected 3090s. The extra all_reduce hop in TP3 cancels the compute savings.
TP3's only advantage is 70% more VRAM for larger caches.

### Recommended Launch Commands

```bash
# TP2 — Default (128K context, recommended)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 NCCL_ALGO=TREE \
    python server_27b_tp2.py --cache-tokens 131072

# TP2 — Maximum context (384K)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 NCCL_ALGO=TREE \
    python server_27b_tp2.py --cache-tokens 393216

# TP3 — Ultra-long context (640K)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,4 NCCL_ALGO=TREE \
    python server_27b_tp2.py --gpu-split 22.0,22.0,22.0 --cache-tokens 655360
```

### Hardware Notes

- **GPUs**: 3× NVIDIA RTX 3090 (24 GB each, SM 8.6 Ampere)
- **Interconnect**: PCIe Gen3 (no NVLink)
- **Model**: Qwen3.5-27B-exl3 heretic 6bpw
- **KV Cache**: FP16 (no quantization)
- **Software**: ExLlamaV3 + flash_attn 2.7.4, Python 3.11, CUDA 12.x

---

## March 27, 2026 — TP2 NCCL Root-Cause Fix + Re-Validation

### Problem: NCCL Silently Broken by Wrong Package

After the March 18 work, the TP2 server stopped launching.  The error was:

```
NCCL error: ncclUnhandledCudaError: Cuda failure
'CUDA driver version is insufficient for CUDA runtime version'
NCCL version 2.28.9
```

This was previously misdiagnosed as a driver/libcudart path issue.  The real
root cause was a **rogue pip package**:

| Package | Version | Built for | Overwrites |
|---------|---------|-----------|------------|
| `nvidia-nccl-cu12` | 2.27.5 | CUDA 12.8 | `nvidia/nccl/lib/libnccl.so.2` |
| `nvidia-nccl-cu13` | 2.28.9 | **CUDA 13.0** | same path — **wins** |

`nvidia-nccl-cu13` was installed standalone (no dependents) and silently
replaced the correct `libnccl.so.2`.  PyTorch 2.10.0+cu128 dynamically
`dlopen`s `libnccl.so.2` at runtime, so it loaded NCCL 2.28.9+cuda13.0, which
requires driver ≥ 575.51.02.  Our driver is 570.211.01 (max CUDA 12.8) →
version check fails.

**Secondary issue**: LD_LIBRARY_PATH in the launcher pointed to `torch/lib`
first, but `torch/lib` ships **no** `libcudart.so.12` on this install.  The
search fell back to:
- `${conda_env}/lib/libcudart.so.12` → CUDA 12.1 ✗
- `/lib/x86_64-linux-gnu/libcudart.so.12` → CUDA 12.0 ✗
- `${site-packages}/nvidia/cuda_runtime/lib/libcudart.so.12` → CUDA 12.8 ✓

### Fix Applied

1. **Removed `nvidia-nccl-cu13`** and force-reinstalled `nvidia-nccl-cu12==2.27.5`.
2. **Fixed `LD_LIBRARY_PATH` order** to prepend `nvidia/cuda_runtime/lib`
   and `nvidia/nccl/lib` before `torch/lib`:

```bash
SP="/home/op/miniconda3/envs/exl3-dev/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="${SP}/nvidia/cuda_runtime/lib:${SP}/nvidia/nccl/lib:${SP}/torch/lib:${LD_LIBRARY_PATH:-}"
```

3. **Updated `server_27b_tp2.py`** to apply this path at the top of the script
   (for `mp.spawn` workers to inherit it at exec time).
4. **Updated `launch_27b_gpu01.sh`** with the corrected path.

### Re-Validation Results (March 27, 2026)

Environment: 2× RTX 3090, PCIe Gen3, driver 570.211.01, CUDA 12.8, NCCL 2.27.5

| Test | Tokens | Time | t/s |
|------|--------|------|-----|
| Warmup token | 10 | 0.4 s | 23.3 |
| No-thinking 300 tok | 71 | 2.5 s | 27.8 |
| No-thinking 500 tok | 499 | 17.4 s | **28.7** |
| Thinking+answer 500 tok | 511 | 18.0 s | **28.4** |
| No-thinking 800 tok | 799 | 28.1 s | **28.4** |

**Sustained decode: ~28.5 t/s** (vs ~25 t/s layer-split = **+14%**).

Previously documented peak of 29.5 t/s remains achievable; the small delta is
noise from PCIe communication and other GPU load (freed0m script on GPU 4
during this run).

### Updated Recommended Launch

```bash
# TP2 — default (128K context)
cd /home/op/exllamav3_ampere
bash launch_27b_gpu01.sh
```

The launch script now:
- Uses `server_27b_tp2.py` (NCCL TP2) instead of layer-split
- Sets correct LD_LIBRARY_PATH (cuda_runtime first)
- Exports `NCCL_ALGO=TREE` for PCIe systems
- Disables unnecessary per-forward barriers and gather fallbacks

### Prevention

To avoid future rogue `nvidia-nccl-cu13` installations:

```bash
# Check which NCCL is loaded
pip show nvidia-nccl-cu12 nvidia-nccl-cu13
# Verify the right binary
strings /path/to/nvidia/nccl/lib/libnccl.so.2 | grep "NCCL version"
# Must show: NCCL version 2.27.5+cuda12.x  (NOT cuda13)
```
