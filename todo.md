# Ampere Kernel Optimizations Tasks

- `[/]` Rebuild native extension (`python3 setup.py build_ext --inplace`)
- `[ ]` Run smoke test (`tests/smoke_qwen3_5_arch.py` if present, or basic test)
- `[ ]` Run GEMM kernel validation (`tests/test_qgemm.py`)
- `[ ]` Run quantization function validation (`tests/test_quant_fn.py`)

## Done (2026-04-15)

- Fixed misplaced `#pragma unroll` in exl3_gemv_kernel.cuh (moved to directly precede the loop). Reason: compiler requires the pragma immediately before the loop; prevented warning and ensured intended unrolling.
- Removed unused `bit_end` in exl3_dq.cuh. Reason: eliminated dead variable and associated warning; clarified bit-range calculation.
- Corrected split-k behavior in exl3_gemv_kernel.cuh and exl3_gemv.cu:
  - Used `k_slice = blockIdx.z` to compute `k_start`/`k_end` and offset `tile_k` loops in both Direct and Staged paths.
  - When `split_k > 1`, store partial results using `atomicAdd` and zeroed `C` on the host before kernel launch. Reason: ensure correct accumulation across k-slices and avoid redundant full-K overwrites.
- Removed unused `frag_col` in `reduce_fr_c` to silence a compiler warning.
- Rebuilt extension; the shared object `exllamav3_ext.cpython-311-x86_64-linux-gnu.so` was produced and the build no longer emits the original warnings.

## Notes / Next steps

- Import of the compiled extension failed in this shell due to `libc10.so` not being found (runtime linker / conda env issue). Activate the `exl3-dev` conda environment or ensure libtorch/libc10 are in `LD_LIBRARY_PATH` before importing `exllamav3_ext`.
- Run the smoke tests and the GEMM/quant unit tests listed above to validate runtime correctness.
- Consider adding tests to cover `split_k` execution paths to prevent regressions.
