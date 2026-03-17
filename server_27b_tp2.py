#!/usr/bin/env python3
"""
OpenAI-compatible TP server for Qwen3.5-27B-exl3 (6bpw heretic variant).

Uses real tensor-parallel (NCCL) with fp16 KV cache and the Generator pipeline
for cached, high-throughput generation.

Supports TP2, TP3, or any GPU count — set CUDA_VISIBLE_DEVICES accordingly.

Usage:
  # TP2 (2×RTX 3090)
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \\
      python server_27b_tp2.py --gpu-split 22.0,22.0

  # TP3 (3×RTX 3090)
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,4 \\
      python server_27b_tp2.py --gpu-split 22.0,22.0,22.0

Environment variables (override CLI defaults):
  MODEL_DIR, GPU_SPLIT, CACHE_TOKENS, HOST, PORT, TP_BACKEND,
  DEFAULT_MAX_TOKENS (16000), MIN_MAX_TOKENS (12), MAX_MAX_TOKENS (16000)
"""

import sys
import os
import argparse
import asyncio
import time
import json
import uuid

# ------------------------------------------------------------------
# Torch library path (required on this machine before importing torch)
# ------------------------------------------------------------------
torch_lib = "/home/op/miniconda3/envs/exl3-dev/lib/python3.11/site-packages/torch/lib"
if os.path.exists(torch_lib):
    os.environ.setdefault(
        "LD_LIBRARY_PATH",
        torch_lib + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
    )

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

from exllamav3 import Config, Model, Tokenizer, Cache, Generator, Job
from exllamav3.cache import CacheLayer_fp16
from exllamav3.generator.sampler.presets import ComboSampler, ArgmaxSampler
from exllamav3.constants import PAGE_SIZE

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
DEFAULT_MODEL_DIR = os.getenv(
    "MODEL_DIR", "/home/op/exllamav3_ampere/models/Qwen3.5-27B-exl3"
)
DEFAULT_GPU_SPLIT = [
    float(x) for x in os.getenv("GPU_SPLIT", "22.0,22.0").split(",")
]
DEFAULT_CACHE_TOKENS = int(os.getenv("CACHE_TOKENS", "32768"))
DEFAULT_HOST = os.getenv("HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("PORT", "1234"))
DEFAULT_TP_BACKEND = os.getenv("TP_BACKEND", "nccl")
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "16000"))
MIN_MAX_TOKENS = int(os.getenv("MIN_MAX_TOKENS", "12"))
MAX_MAX_TOKENS = int(os.getenv("MAX_MAX_TOKENS", "16000"))

# ------------------------------------------------------------------
# Globals (populated during startup)
# ------------------------------------------------------------------
_model: Optional[Model] = None
_tokenizer: Optional[Tokenizer] = None
_generator: Optional[Generator] = None
_model_name: str = "Qwen3.5-27B-exl3-heretic-6bpw-tp2"
_cache_tokens: int = DEFAULT_CACHE_TOKENS
_gen_lock: asyncio.Lock

app = FastAPI(title="Qwen3.5-27B heretic exl3 - TP multi-GPU server")


# ------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "Qwen3.5-27B-exl3-heretic-6bpw-tp2"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    min_p: Optional[float] = 0.08
    top_k: Optional[int] = 0
    stream: Optional[bool] = False


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def align_to_page(n: int) -> int:
    return (n // PAGE_SIZE) * PAGE_SIZE


def _normalize_max_tokens(requested: Optional[int]) -> int:
    if requested is None or requested <= 0:
        return DEFAULT_MAX_TOKENS
    return max(MIN_MAX_TOKENS, min(requested, MAX_MAX_TOKENS))


def format_chatml(messages: List[ChatMessage], add_assistant: bool = True) -> str:
    prompt = ""
    for msg in messages:
        role = msg.role.strip().lower()
        if role == "system":
            prompt += f"<|im_start|>system\n{msg.content}<|im_end|>\n"
        elif role == "user":
            prompt += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
        elif role == "assistant":
            prompt += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"
    if add_assistant:
        prompt += "<|im_start|>assistant\n"
    return prompt


def make_sampler(request: ChatCompletionRequest):
    from exllamav3.generator.sampler.custom import SS_Temperature, SS_TopK, SS_TopP, SS_MinP, SS_Sample, SS_Argmax
    temp = request.temperature if request.temperature is not None else 0.7
    if temp <= 0:
        return ArgmaxSampler()
    steps = []
    min_p = request.min_p or 0.0
    if min_p > 0:
        steps.append(SS_MinP(min_p))
    top_k = request.top_k or 0
    if top_k > 0:
        steps.append(SS_TopK(top_k))
    top_p = request.top_p if request.top_p is not None else 1.0
    if 0 < top_p < 1.0:
        steps.append(SS_TopP(top_p))
    steps.append(SS_Temperature(temp))
    steps.append(SS_Sample())
    from exllamav3.generator.sampler.custom import CustomSampler
    return CustomSampler(steps)


def generate_with_generator(input_ids: torch.Tensor, request: ChatCompletionRequest, max_new: int, stop_conditions: set):
    """Generate tokens using the cached Generator pipeline."""
    sampler = make_sampler(request)
    job = Job(
        input_ids=input_ids,
        max_new_tokens=max_new,
        stop_conditions=stop_conditions,
        sampler=sampler,
    )
    _generator.enqueue(job)

    total_new_tokens = 0
    while _generator.num_remaining_jobs():
        for r in _generator.iterate():
            stage = r.get("stage", "")
            if stage == "streaming":
                text = r.get("text", "")
                eos = r.get("eos", False)
                if r.get("new_tokens") is not None:
                    total_new_tokens = r.get("new_tokens")
                if text or eos:
                    yield text, eos, total_new_tokens
                if eos:
                    return


# ------------------------------------------------------------------
# Startup
# ------------------------------------------------------------------
@app.on_event("startup")
async def load_model():
    global _model, _tokenizer, _generator, _gen_lock, _cache_tokens

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--gpu-split", default=",".join(str(x) for x in DEFAULT_GPU_SPLIT))
    parser.add_argument("--cache-tokens", type=int, default=DEFAULT_CACHE_TOKENS)
    parser.add_argument("--tp-backend", default=DEFAULT_TP_BACKEND)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args, _ = parser.parse_known_args()

    gpu_split = [float(x) for x in args.gpu_split.split(",")]
    _cache_tokens = align_to_page(args.cache_tokens)

    print(f"\n{'='*60}")
    num_gpus = len(gpu_split)
    print(f"  Qwen3.5-27B-exl3 heretic - TP {num_gpus}-GPU server (cached)")
    print(f"{'='*60}")
    print(f"  Model dir    : {args.model}")
    print(f"  GPU split    : {gpu_split}")
    print(f"  TP backend   : {args.tp_backend}")
    print(f"  Cache tokens : {_cache_tokens:,}")
    print(f"  KV cache     : fp16 (no quantization)")
    print(f"  Decode path  : Generator + paged flash_attn (cached)")
    print(f"  Max tokens   : {DEFAULT_MAX_TOKENS} (cap {MAX_MAX_TOKENS})")
    print(f"  Endpoint     : http://{args.host}:{args.port}/v1")
    print()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    print(f"  CUDA_VISIBLE_DEVICES : {visible}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}  ({p.total_memory / 1024**3:.1f} GB)")
    print()

    # Ampere (SM 8.x) optimizations: TF32 for matmul, cudnn
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("Loading model config...")
    cfg = Config.from_directory(args.model)
    _model = Model.from_config(cfg)

    print(f"Creating fp16 KV cache ({_cache_tokens:,} tokens)...")
    cache = Cache(_model, max_num_tokens=_cache_tokens, layer_type=CacheLayer_fp16)

    print("Loading weights (tensor parallel)...")
    _model.load(
        tensor_p=True,
        use_per_device=gpu_split,
        tp_output_device=0,
        tp_backend=args.tp_backend,
        max_chunk_size=2048,
        max_output_size=1,
        progressbar=True,
    )

    for i in range(torch.cuda.device_count()):
        mem_alloc = torch.cuda.memory_allocated(i) / 1024**3
        mem_res = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  GPU {i}: alloc={mem_alloc:.2f}GB  reserved={mem_res:.2f}GB")

    print("\nLoading tokenizer...")
    _tokenizer = Tokenizer.from_config(cfg)

    print("Creating Generator...")
    _generator = Generator(
        model=_model,
        cache=cache,
        tokenizer=_tokenizer,
    )

    _gen_lock = asyncio.Lock()
    print(f"\nServer ready.  Cache = {_cache_tokens:,} tokens.\n")


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "model": _model_name, "cache_tokens": _cache_tokens}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": _model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
                "context_length": _cache_tokens,
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if _model is None or _tokenizer is None or _generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    prompt = format_chatml(request.messages)
    input_ids = _tokenizer.encode(prompt, add_bos=True, encode_special_tokens=True)
    if isinstance(input_ids, tuple):
        input_ids = input_ids[0]

    prompt_len = input_ids.shape[-1]
    if prompt_len >= _cache_tokens:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt length {prompt_len} exceeds cache size {_cache_tokens}",
        )

    request.max_tokens = _normalize_max_tokens(request.max_tokens)
    max_new = min(request.max_tokens, _cache_tokens - prompt_len - 1)

    eos_token_id = _tokenizer.eos_token_id
    im_end_id = _tokenizer.single_id("<|im_end|>")
    stop_conditions = {t for t in [eos_token_id, im_end_id] if t is not None}

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if request.stream:
        async def generate_stream():
            async with _gen_lock:
                for chunk, eos, _ntok in generate_with_generator(input_ids, request, max_new, stop_conditions):
                    if chunk:
                        data = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": _model_name,
                            "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                        await asyncio.sleep(0)
                data = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": _model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    async with _gen_lock:
        pieces = []
        completion_tokens = 0
        for piece, eos, ntok in generate_with_generator(input_ids, request, max_new, stop_conditions):
            pieces.append(piece)
            completion_tokens = ntok
        response_text = "".join(pieces)

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": _model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_len,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_len + completion_tokens,
        },
    }


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3.5-27B-exl3 heretic TP N-GPU server")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--gpu-split", default=",".join(str(x) for x in DEFAULT_GPU_SPLIT))
    parser.add_argument("--cache-tokens", type=int, default=DEFAULT_CACHE_TOKENS)
    parser.add_argument("--tp-backend", default=DEFAULT_TP_BACKEND)
    args = parser.parse_args()

    uvicorn.run(
        "server_27b_tp2:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )
