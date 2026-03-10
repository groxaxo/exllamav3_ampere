# Qwen3.5 FLA / Ampere notes

This writeup covers a focused experiment on the optional Flash Linear Attention (FLA) and `causal-conv1d` paths for Qwen3.5 on Ampere GPUs, along with the small ExLlamaV3 fork change made to exercise the FLA fused recurrent kernel safely.

## What changed in this fork

The Qwen3.5 `GatedDeltaNet` path in `exllamav3/modules/gated_delta_net.py` now:

- treats FLA as an optional dependency more defensively, so an incompatible FLA install does not break the Qwen path outright,
- adds an `EXLLAMA_GDN_RECURRENT_BACKEND` environment variable with values:
  - `auto` (default): use FLA fused recurrent only for the decode path (`seqlen == 1`) when available,
  - `ext`: force the existing ExLlama extension recurrent kernel,
  - `fla`: force the FLA fused recurrent kernel for the non-chunked recurrent path.

The default stayed conservative: `auto` targets decode-heavy interactive usage and falls back cleanly to the existing extension path.

## Test setup

- Current fork: `/home/op/exllamav3_ampere`
- Upstream reference: `/home/op/external_repos/exllamav3_upstream`
- Conda env: `exl3-dev`
- 1x GPU target: RTX 3060 (`CUDA_VISIBLE_DEVICES=2`)
- 2x GPU target: RTX 3060 pair (`CUDA_VISIBLE_DEVICES=2,3`)
- Qwen model: `models/Qwen3.5-9B-exl3`
- 2x fallback model: `/home/op/models/Llama-3.1-8B-Instruct-exl3-4.0bpw`

Benchmark commands used:

```sh
python eval/perf.py -m models/Qwen3.5-9B-exl3 -cs 2048 -max_length 2048 -chunk_size 1024
python eval/perf.py -m /home/op/models/Llama-3.1-8B-Instruct-exl3-4.0bpw -tp -tpb native -gs 10,10 -cs 4096 -max_length 4096 -chunk_size 2048
```

Qwen3.5 tensor parallelism is still unsupported here, so the 2x regression check uses the existing Llama EXL3 model as a control.

## Baseline before optional kernels

1x RTX 3060, Qwen3.5-9B EXL3:

| Repo | Prefill avg. | Decode avg. |
| --- | ---: | ---: |
| Current fork | `2361.52` tokens/s | `64.20` tokens/s |
| Upstream | `2358.85` tokens/s | `64.11` tokens/s |

The pre-change fork and upstream were effectively identical on this workload.

## Optional kernel setup notes

- `causal-conv1d` built and imported successfully in `exl3-dev`.
- `flash-linear-attention` installed, but its default eager import path tripped on Triton `3.1.0` with:

```text
ValueError: 'STAGE' is not in list
```

This came from FLA's eager chunk-kernel imports, not from the fused recurrent kernel itself. For the local benchmark environment, the cloned FLA checkout was patched to use lighter package inits so the fused recurrent module could be imported directly. A cleaner long-term route is to use Triton `>= 3.2.0` or upstream an equivalent lazy-import fix to FLA.

With that local patch in place on this machine:

- `chunk_gated_delta_rule` remained unavailable,
- `fused_recurrent_gated_delta_rule_fwd` was available and usable.

## Correctness spot-check

A representative single-token Qwen decode step was run twice on the current fork:

- `EXLLAMA_GDN_RECURRENT_BACKEND=ext`
- `EXLLAMA_GDN_RECURRENT_BACKEND=fla`

Observed result:

- greedy token matched (`argmax = 222` in both runs),
- `max_abs_diff = 0.078125`,
- `mean_abs_diff = 0.0100`.

That is close enough for an inference-side optimization experiment, and the greedy output matched on the checked step.

## Qwen3.5 results after optional kernels

1x RTX 3060, Qwen3.5-9B EXL3:

| Mode | Prefill avg. | Decode avg. | vs current `ext` |
| --- | ---: | ---: | --- |
| Current fork, `ext` | `2330.81` | `64.52` | control |
| Current fork, `auto` | `2311.92` | `64.99` | `-0.81%` prefill, `+0.73%` decode |
| Current fork, `fla` | `2357.25` | `64.74` | `+1.13%` prefill, `+0.34%` decode |
| Upstream, post-install | `2352.47` | `64.53` | `+0.93%` prefill, `+0.02%` decode |

### Interpretation

- `auto` gave the best decode average on the tested RTX 3060 setup.
- forcing `fla` gave the best prefill average among the edited-fork modes, but its decode gain was smaller than `auto`.
- upstream with the same optional installs behaved very close to the current fork's `ext` control, which is what we would expect from the code paths.

For interactive decode-heavy use, `auto` is the better default.

For prefill-heavy experiments, `EXLLAMA_GDN_RECURRENT_BACKEND=fla` is worth trying if the local FLA/Triton stack is healthy.

## 2x RTX 3060 regression check

2x RTX 3060, native TP, Llama-3.1-8B EXL3 fallback:

| Setup | Prefill avg. | Decode avg. |
| --- | ---: | ---: |
| Current baseline | `596.79` tokens/s | `59.67` tokens/s |
| Current post-install / edited fork | `596.46` tokens/s | `59.92` tokens/s |

Delta vs baseline:

- prefill: `-0.06%`
- decode: `+0.42%`

That is effectively noise-level movement and indicates no obvious regression in the unrelated 2x TP fallback path.

## Recommendations

1. Keep `EXLLAMA_GDN_RECURRENT_BACKEND=auto` as the default behavior.
2. Document `fla` as an opt-in for prefill-oriented Qwen workloads.
3. Prefer Triton `>= 3.2.0` before investing further in FLA chunk-kernel benchmarking on Ampere.
4. If deeper FLA integration is desired later, avoid depending on FLA's heavy top-level package init for optional kernels.
5. `causal-conv1d` is viable in this env, but the dominant measured win here came from recurrent-backend selection rather than the 2x fallback path.
