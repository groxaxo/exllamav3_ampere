# ExLlamaV3 on Ampere GPUs

This note collects the repository's current evidence around Ampere performance, outlines quality-preserving ways to investigate slow inference, and turns the likely optimization work into a practical todo list.

## Current status in this repo

Two existing docs already call out that Ampere is not fully tuned yet:

- `README.md`: the Marlin-inspired GEMM kernel is "roughly memory-bound" under optimal conditions on RTX 4090, but still needs work on Ampere GPUs and at lower bitrates.
- `doc/convert.md`: the quantization kernel is currently tuned for Ada GPUs, and Ampere is expected to improve with more optimization.

That matches what the kernel sources say today:

- `exllamav3/exllamav3_ext/quant/exl3_gemm_inner.cuh` has TODOs for:
  - rearranging tiles into an `ldmatrix`-friendly layout while loading,
  - resolving shared-memory bank conflicts,
  - reducing bank conflicts in the reduction path.
- `exllamav3/exllamav3_ext/ptx.cuh` explicitly warns against an emulated `m8n8k4` tensor-core path on Ampere and later.

## What "without sacrificing quality" should mean here

For ExLlamaV3, quality-preserving optimization should prioritize:

- better kernel scheduling,
- better memory layout,
- better overlap of global-memory and shared-memory work,
- better kernel-shape selection for Ampere,
- better runtime settings such as chunk sizes and batching,
- better profiling so regressions are measured before and after tuning.

It should **not** rely on:

- lowering bitrate below the model's acceptable quality target,
- lowering `head_bits` just to chase throughput,
- changing sampling settings in ways that alter output quality,
- switching to approximation paths that change logits or attention semantics.

## Probable reasons Ampere inference is slower

### 1. Kernel tuning is biased toward Ada-era sweet spots

The repo's own docs say the fast path is tuned around RTX 4090-like behavior. Ampere cards such as A100 (`sm_80`) and RTX 30-series (`sm_86`) have different balance points for:

- register pressure,
- shared-memory throughput,
- `ldmatrix` layout sensitivity,
- occupancy vs. tile size,
- memory bandwidth utilization at lower bitrates.

If the current kernel shapes were primarily validated on Ada, Ampere can end up leaving tensor cores or memory bandwidth underutilized.

### 2. Shared-memory bank conflicts are still visible in the core GEMM path

The most relevant code is already annotated:

```cpp
// exllamav3/exllamav3_ext/quant/exl3_gemm_inner.cuh
for (int i = 0; i < load_a_iters; ++i)
{
    // TODO: Rearrange into ldmatrix friendly layout while loading?
    if (pred_a_gl[i]) cp_async(sh + EXL3_GEMM_BASE_THREADS * i + t, gl + load_a_gl[i]);
}

// TODO: Resolve bank conflicts
int r = (lane_id % 8) + 8 * ((lane_id / 8) % 2);
int c = lane_id / 16;

// TODO: Shuffle to avoid bank conflicts here? Doesn't seem to be a bottleneck
```

On Ampere, bank conflicts and suboptimal `ldmatrix` packing can erase the expected win from quantized matmul, especially at small batch sizes or low bitrates where overhead becomes more visible.

### 3. Low-bitrate inference is likely becoming launch- and scheduling-limited

The README explicitly says the kernel still needs work "to remain memory-bound at lower bitrates." That is a strong hint that below the best-case operating point, the bottleneck shifts away from raw bandwidth and toward:

- kernel launch overhead,
- dequantization overhead,
- poor tile utilization,
- imperfect shape selection.

### 4. Kernel selection likely needs Ampere-specific retuning

`science/qgemm_benchmark.py` exists specifically to compare kernel shapes, but the repo does not currently document any Ampere-specific kernel map or retuning pass. That makes it likely that Ampere is using generic or Ada-favored choices that are correct functionally, but not optimal for throughput.

## Practical things to try now that should not reduce quality

These are the safest near-term actions for diagnosing or improving slow Ampere inference with the current codebase.

### Measure prefill and decode separately

Use the existing perf script so you can see whether the slowdown is in prompt ingestion, token generation, or both:

```sh
cd /path/to/exllamav3
python eval/perf.py -m /path/to/model-exl3 --chunk_size 4096 --max_length 8192
```

If prefill is much slower than decode, the issue is probably in chunked attention/prefill behavior or large GEMMs. If decode collapses first, small-shape GEMM/GEMV efficiency is the better target.

### Sweep chunk size instead of assuming one value is best

`eval/perf.py` exposes `--chunk_size`, and Ampere may want a different tradeoff from Ada:

```sh
cd /path/to/exllamav3
python eval/perf.py -m /path/to/model-exl3 --chunk_size 2048 --max_length 8192
python eval/perf.py -m /path/to/model-exl3 --chunk_size 4096 --max_length 8192
python eval/perf.py -m /path/to/model-exl3 --chunk_size 8192 --max_length 8192
```

This changes runtime scheduling, not model quality.

### Rebuild for the Ampere target you actually use

When building from source, it is reasonable to target Ampere explicitly so the extension is compiled for the hardware you care about. For example, `8.0;8.6` covers common A100 and RTX 30-series targets; add other SM versions if your Ampere deployment needs them:

```sh
cd /path/to/exllamav3
export TORCH_CUDA_ARCH_LIST="8.0;8.6"
pip install .
```

This does not change model outputs. It only changes how the CUDA extension is compiled.

### Benchmark kernel-shape behavior directly

The repository already contains a focused QGEMM benchmarking script:

```sh
cd /path/to/exllamav3
python science/qgemm_benchmark.py
```

Before using it on a single Ampere GPU, set the `devices` list in the script to the actual CUDA device you want to test. The important output is whether the preferred kernel shape is consistently close to the fastest kernel for the shapes that dominate inference.

### Use multi-GPU quantization settings for conversion work, not quality tradeoffs

If part of the complaint is "Ampere is slow" during conversion rather than runtime inference, use the existing multi-GPU switches first:

```sh
cd /path/to/exllamav3
python convert.py -i /path/to/model \
                  -o /path/to/model-exl3 \
                  -w /path/to/workdir \
                  -b 4.0 \
                  -d 0,1 \
                  -pm
```

This changes where compute runs, not the quality target.

## Probable code-level optimizations for Ampere

These are the most plausible engineering directions based on the current source tree.

### A. Repack A/B tiles into an Ampere-friendly `ldmatrix` layout during `cp_async`

**Why it helps:** the current GEMM inner loop already hints that shared-memory layout is not ideal. Doing the rearrangement at load time can trade a little indexing work for better tensor-core feed efficiency later.

**Relevant file:** `exllamav3/exllamav3_ext/quant/exl3_gemm_inner.cuh`

**Potential direction:**

```cpp
// Pseudocode only: keep math identical, change layout and load order.
if (pred_a_gl[i])
{
    int4 v = gl[load_a_gl[i]];
    int dst = ampere_ldmatrix_swizzle(i, t, TILESIZE_K);
    cp_async(sh + dst, &v);
}
```

This is a performance-only change if the mathematical layout seen by the MMA path remains unchanged.

### B. Add Ampere-specific bank-conflict mitigation

**Why it helps:** the kernel already flags bank conflicts in both fragment loading and reduction. Ampere is especially sensitive when shared-memory access patterns are just slightly off the ideal swizzle.

**Potential solutions:**

- add per-row padding in shared-memory staging buffers,
- change lane-to-row/lane-to-column mapping,
- split the reduction scratch so neighboring lanes do not hammer the same banks,
- benchmark whether a small shared-memory footprint increase pays back in occupancy-adjusted throughput.

### C. Retune kernel-shape selection using Ampere benchmark data

**Why it helps:** even a very fast kernel underperforms if the wrong shape is chosen for common `(m, k, n)` combinations during decode and prefill.

**Relevant files:**

- `science/qgemm_benchmark.py`
- `tests/test_qgemm.py`
- `exllamav3/exllamav3_ext/quant/...` kernel map code

**Potential solution:** regenerate or retune the kernel preference table from Ampere benchmark runs instead of assuming Ada-favored winners generalize.

### D. Retune for low-bitrate operating points instead of only the 4 bpw sweet spot

The README suggests the kernel is happiest at 4 bpw. If Ampere slows down below that point, the fix may be to:

- use different tile sizes for low-bitrate paths,
- reduce dequant staging overhead,
- increase work per launch,
- specialize decode-heavy kernels for the small-`m` shapes common in generation.

None of those require changing quantization quality; they only change how the same weights are executed.

### E. Reduce register pressure where it hurts occupancy on Ampere

The inner GEMM kernel keeps multiple fragment arrays in registers. That is great when occupancy remains healthy, but on Ampere a seemingly small register increase can cut active warps enough to lose latency hiding.

**Likely experiments:**

- fewer pipeline stages,
- smaller tile shapes for selected paths,
- separate tuning for decode (`m` small) and prefill (`m` larger),
- selective unrolling changes on Ampere only.

## Suggested todo list

### Immediate profiling and validation

- [ ] Run `python eval/perf.py` on at least one `sm_80` target and one `sm_86` target.
- [ ] Record separate prefill and generation throughput for chunk sizes `2048`, `4096`, and `8192`.
- [ ] Run `python science/qgemm_benchmark.py` on Ampere and capture the fastest kernel per hot shape.
- [ ] Cross-check kernel winners against the currently preferred kernel choices.
- [ ] Keep quality fixed while testing: same model, same quantization, same sampler settings.

### Kernel work with the best chance of helping inference

- [ ] Prototype `ldmatrix`-friendly tile rearrangement while loading A tiles.
- [ ] Prototype shared-memory padding or lane remapping to reduce bank conflicts.
- [ ] Test whether the reduction scratch path needs a different shuffle/padding pattern on Ampere.
- [ ] Measure occupancy, shared-memory efficiency, and tensor-core utilization before and after each change.
- [ ] Add Ampere-specific kernel preferences if benchmarks show stable winners.

### Low-bitrate optimization without quality loss

- [ ] Profile decode-heavy runs for models below the 4 bpw sweet spot.
- [ ] Add shape-specialized paths for the small-`m` decode case if that is where Ampere falls behind.
- [ ] Check whether dequant or scale application dominates below 4 bpw.
- [ ] Keep output logits bit-identical within normal floating-point tolerance while tuning.

### Conversion-side optimization

- [ ] Benchmark Ampere conversion throughput with `-d` and `-pm`.
- [ ] Compare whether parallel-mode layer assignment maps well to Ampere SM counts.
- [ ] Revisit the workload split heuristic in `exllamav3/conversion/convert_model.py`.
- [ ] Document recommended device ratios for mixed Ampere setups once measured.

## A conservative optimization order

If slow Ampere inference is the main problem, the safest order is:

1. measure with `eval/perf.py`,
2. benchmark kernel shapes with `science/qgemm_benchmark.py`,
3. retune kernel selection,
4. fix shared-memory layout and bank conflicts,
5. only then experiment with deeper kernel rewrites.

That order gives the best chance of finding speedups that keep output quality unchanged.
