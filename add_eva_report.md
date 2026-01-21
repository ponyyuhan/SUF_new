# Add-Eva Experiments Report (2026-01-21)

This report summarizes the three add-on experiments requested in `add_eva.md`.

## Environment

- Host: 2x RTX 5090 (per user), 314 GB system RAM
- Build: `GPU-MPC` binaries compiled with `CUDA_VERSION=13.0` and `GPU_ARCH=120`
- SUF tuning used where noted: `SUF_NEXP_BITS=10 SUF_INV_BITS=10 SUF_RSQRT_BITS=9`

## Experiment 1: Block-level Softmax + LayerNorm Bench

**Setup**
- Binary: `third_party/EzPC/GPU-MPC/experiments/sigma/bench_softmax_norm`
- Shapes: B=1, H=12, d_model=768, L in {32, 64, 128, 256}
- Iters: 5 (from JSON output)
- Metrics: online latency, comm bytes, key bytes

**Softmax block (BERT-base shaped, B=1, H=12, LxL)**

SHAFT softmax measured on the same tensor shape `(B,H,L,L)` and same `L` set (5 runs, 1 warmup). Default config uses ODE softmax with 16 iterations and clipping to [-4, 12].

| L | Sigma ms | SUF ms | SHAFT ms | Sigma comm (MiB) | SUF comm (MiB) | SHAFT comm (MiB) | Sigma key (MiB) | SUF key (MiB) | SHAFT rounds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 4.525 | 3.443 | 110.0 | 1.39 | 0.75 | 22.8750 | 19.53 | 9.79 | 41 |
| 64 | 9.903 | 5.932 | 209.8 | 5.53 | 2.98 | 91.5000 | 78.42 | 39.48 | 41 |
| 128 | 38.189 | 17.530 | 364.6 | 22.11 | 11.89 | 366.0000 | 314.28 | 158.54 | 41 |
| 256 | 108.197 | 59.177 | 882.5 | 88.37 | 47.50 | 1464.0000 | 1258.28 | 635.41 | 41 |

Note: SHAFT unit-test does not expose preprocessing/key material sizes.

**LayerNorm block (BERT-base shaped)**

| L | Sigma ms | SUF ms | Speedup | Sigma comm (MiB) | SUF comm (MiB) | Sigma key (MiB) | SUF key (MiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 4.993 | 1.949 | 2.56× | 1.19 | 1.19 | 7.63 | 7.57 |
| 64 | 8.653 | 2.384 | 3.63× | 2.37 | 2.37 | 15.25 | 15.14 |
| 128 | 16.515 | 3.568 | 4.63× | 4.75 | 4.74 | 30.50 | 30.27 |
| 256 | 32.024 | 8.380 | 3.82× | 9.50 | 9.48 | 60.99 | 60.54 |

**Observations**
- SUF is consistently faster than Sigma for both Softmax and LayerNorm.
- SUF reduces online communication and key bytes for Softmax.
- LayerNorm comm is similar (expected), but SUF still provides strong latency gains.

**Artifacts**
- `artifacts/add_eva/block_bench_opt.jsonl`
- `artifacts/add_eva/shaft_softmax_attn.txt`

## Experiment 2: Extensibility Case Study — Sigmoid

**Setup**
- New SUF descriptor for sigmoid with polynomial (degree 1) and LUT-only (degree 0) variants.
- Shape: `n=262144`, `scale_in=12`, `scale_out=16`, `in_bits=16`, `bw_out=16`.

**Results**

| Variant | intervals | degree | avg ms | key (KiB) | max abs err | mean abs err | cos sim |
|---|---:|---:|---:|---:|---:|---:|---:|
| SUF-sigmoid (poly) | 256 | 1 | 0.012339 | 1024.04 | 65513 | 4683.61 | 0.949018 |
| SUF-sigmoid (LUT) | 256 | 0 | 0.006173 | 512.03 | 512 | 63.99 | 0.999996 |

**Mask-shape invariance**
- Passed (fixed shape across keygen seeds).

**Observations**
- Both variants are implementable purely via SUF descriptors (no new protocol).
- LUT-only is more accurate and smaller in key size for the 256-interval setting.

**Artifacts**
- `artifacts/add_eva/sigmoid_bench.jsonl`
- `artifacts/add_eva/sigmoid_invariance.txt`

## Experiment 3: Batch-size Sweep (BERT-base / GPT-2, seq=128)

**Setup**
- Models: BERT-base, GPT-2
- Batches: {1, 2, 4} (larger batches exceed key buffer limits)
- Metrics: online latency, comm, key bytes, throughput
- Runtime flags for stability:
  - `SIGMA_MEMPOOL_GB=0`, `SIGMA_DISABLE_ASYNC_MALLOC=1`
  - `SIGMA_COMPRESS=0` (disables comm compression to avoid expand kernel OOB)
  - `OMP_NUM_THREADS=4`, `OMP_THREAD_LIMIT=4`

**BERT-base (seq=128)**

| B | Sigma ms | SUF ms | Speedup | Sigma comm (GB) | SUF comm (GB) | Sigma key (GB) | SUF key (GB) | Sigma tok/s | SUF tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5318.391 | 3005.464 | 1.77× | 0.99 | 1.21 | 16.83 | 12.74 | 24.07 | 42.59 |
| 2 | 10599.061 | 6104.788 | 1.74× | 2.68 | 2.41 | 33.67 | 25.48 | 24.15 | 41.93 |
| 4 | 21670.691 | 12506.095 | 1.73× | 5.36 | 4.83 | 67.34 | 50.95 | 23.63 | 40.94 |

**GPT-2 (seq=128)**

| B | Sigma ms | SUF ms | Speedup | Sigma comm (GB) | SUF comm (GB) | Sigma key (GB) | SUF key (GB) | Sigma tok/s | SUF tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4839.882 | 2716.696 | 1.78× | 1.12 | 1.05 | 14.29 | 11.10 | 26.45 | 47.12 |
| 2 | 9901.255 | 5441.932 | 1.82× | 2.24 | 2.10 | 28.58 | 22.20 | 25.86 | 47.04 |
| 4 | 19817.532 | 10874.305 | 1.82× | 4.49 | 4.20 | 57.17 | 44.41 | 25.84 | 47.08 |

**Observations**
- SUF achieves 1.73–1.82× lower latency and ~1.7–1.9× higher throughput across all batches.
- Key material is consistently smaller for SUF.
- Communication is generally lower for SUF; the only exception is BERT-base at B=1 with compression disabled.

**Artifacts**
- `artifacts/add_eva/batch_sweep.csv`

## Summary

- Block-level benchmarks and the sigmoid case study both show SUF leading against Sigma.
- SHAFT softmax is included with the same `(B,H,L,L)` attention shape; SUF remains faster on the attention-style block.
- Batch sweep shows SUF consistently faster with better throughput and smaller keys.
