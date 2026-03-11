# ExLlamaV3 Ampere Inference Notes

This repo is a custom `exllamav3` fork with a small set of local OpenAI-compatible inference scripts for Ampere GPUs.

The current setup keeps the scripts created on `2026-03-11` as the base set and folds the `Qwen3.5-27B` mixed-GPU work into that set instead of keeping a separate one-off script from `2026-03-12`.

## Included inference scripts

### `server_openai.py`
- Single `RTX 3060`
- Model path: `models/Qwen3.5-9B-exl3`
- Port: `8001`
- Intended for lower-VRAM single-GPU testing

### `server_3090.py`
- Single `RTX 3090`
- Model path: `models/Qwen3.5-9B-exl3`
- Port: `8002`
- Intended for higher-context single-GPU testing

### `server_dual_gpu.py`
- Dual `RTX 3090`
- Model path: `models/Qwen3.5-9B-exl3`
- Port: `8002`
- Uses layer-split loading across two 3090s

### `server_mixed.py`
- Mixed `RTX 3090 + RTX 3060`
- Model path: `models/Qwen3.5-27B-exl3`
- Port: `8003`
- Uses quantized KV cache (`4-bit K/V`)
- Configured for about `150k` context via `CACHE_TOKENS = 150016`
- This is the script to use for the current `Qwen3.5-27B` mixed-GPU inference setup

### `server_32k.py`
- Mixed `RTX 3090 + RTX 3060`
- Model path: `models/Qwen3.5-9B-exl3`
- Port: `8004`
- Fixed `32k` context variant for the 9B setup

## What changed

- Removed `server_qwen27b_150k_mixed.py`
- Moved its useful configuration into `server_mixed.py`
- Replaced the downloaded model-card-style top-level README with repo-specific documentation

## Environment

- **AFM** (ArceeForCausalLM)
- **Apertus** (ApertursForCausalLM)
- **Command-R** etc. (CohereForCausalLM)
- **Command-A**, **Command-R7B**, **Command-R+** etc. (Cohere2ForCausalLM)
- **DeciLM**, **Nemotron** (DeciLMForCausalLM)
- **dots.llm1** (Dots1ForCausalLM)
- **ERNIE 4.5** (Ernie4_5_ForCausalLM, Ernie4_5_MoeForCausalLM)
- **EXAONE 4.0** (Exaone4ForCausalLM)
- **Gemma 2** (Gemma2ForCausalLM)
- **Gemma 3** (Gemma3ForCausalLM, Gemma3ForConditionalGeneration) *- multimodal*
- **GLM 4**, **GLM 4.5**, **GLM 4.5-Air**, **GLM 4.6** (Glm4ForCausalLM, Glm4MoeForCausalLM)
- **GLM 4.1V**, **GLM 4.5V** (Glm4vForConditionalGeneration, Glm4vMoeForConditionalGeneration) *- multimodal*
- **HyperCLOVAX** (HyperCLOVAXForCausalLM, HCXVisionV2ForCausalLM) *- multimodal*
- **IQuest-Coder** (IQuestCoderForCausalLM)
- **Llama**, **Llama 2**, **Llama 3**, **Llama 3.1-Nemotron** etc. (LlamaForCausalLM)
- **MiMo-RL** (MiMoForCausalLM)
- **MiniMax-M2** (MiniMaxM2ForCausalLM)
- **Mistral**, **Ministral 3**, **Devstral 2** etc. (MistralForCausalLM, Mistral3ForConditionalGeneration) *- multimodal*
- **Mixtral** (MixtralForCausalLM)
- **NanoChat** (NanoChatForCausalLM)
- **Olmo 3.1** (Olmo3ForCausalLM)
- **Olmo-Hybrid** (OlmoHybridForCausalLM)
- **Phi3**, **Phi4** (Phi3ForCausalLM)
- **Qwen 2**, **Qwen 2.5**, **Qwen 2.5 VL** (Qwen2ForCausalLM, Qwen2_5_VLForConditionalGeneration) *- multimodal*
- **Qwen 3** (Qwen3ForCausalLM, Qwen3MoeForCausalLM)
- **Qwen 3-Next** (Qwen3NextForCausalLM)
- **Qwen 3-VL** (Qwen3VLForConditionalGeneration)  *- multimodal*
- **Qwen 3-VL MoE** (Qwen3VLMoeForConditionalGeneration) *- multimodal*
- **Qwen 3.5** (Qwen3_5ForConditionalGeneration) *- multimodal*
- **Qwen 3.5 MoE** (Qwen3_5MoeForConditionalGeneration) *- multimodal*
- **Seed-OSS** (SeedOssForCausalLM)
- **SmolLM** (SmolLM3ForCausalLM)
- **SolarOpen** (SolarOpenForCausalLM)
- **Step 3.5 Flash** (Step3p5ForCausalLM)

- Conda env: `exl3-dev`
- Torch libraries at `/home/op/miniconda3/envs/exl3-dev/lib/python3.11/site-packages/torch/lib`
- Models under `/home/op/exllamav3_ampere/models/`

Activate the environment before starting a server:

```bash
conda activate exl3-dev
export LD_LIBRARY_PATH=/home/op/miniconda3/envs/exl3-dev/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH
```

## Launch commands

### Single 3060

```bash
CUDA_VISIBLE_DEVICES=2 python server_openai.py
```

### Single 3090

```bash
CUDA_VISIBLE_DEVICES=0 python server_3090.py
```

### Dual 3090

```bash
CUDA_VISIBLE_DEVICES=0,1 python server_dual_gpu.py
```

### Mixed 3090 + 3060 for Qwen3.5-27B

```bash
CUDA_VISIBLE_DEVICES=0,2 python server_mixed.py
```

### Mixed 3090 + 3060 at 32K for Qwen3.5-9B

```bash
CUDA_VISIBLE_DEVICES=1,3 python server_32k.py
```

## API checks

Health:

```bash
curl http://localhost:8003/health
```

List models:

```bash
curl http://localhost:8003/v1/models
```

Chat completion:

```bash
curl http://localhost:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-27B-exl3",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128
  }'
```

## Notes

- `server_mixed.py` is now the main mixed-GPU script for the 27B model.
- The older 9B scripts are kept because they are still useful for comparison and lighter-weight testing.
- Files such as logs, downloaded model blobs, cache directories, and local benchmark output are intentionally not documented as committed project artifacts.
