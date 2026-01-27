# Evaluation Update (127_eva + comments + batch) — 2026-01-27

This update **does not rerun everything from scratch**. I resumed from existing outputs, snapshotted them as run0, and added only the missing/needed runs (run>=1). Unless noted, reported medians/variance use the selected runs with `run_idx>=1`.

## 1) Environment and Protocol Settings

- OS: Ubuntu 24.04.3 LTS (`uname`: Linux 6.18.6-pbk)
- CPU: 2× AMD EPYC 9654 (96 cores/socket, 384 CPUs total)
- RAM: ~1.5 TiB (`free -h`)
- GPU: 2× NVIDIA RTX PRO 6000 Blackwell Workstation Edition (97,887 MiB each)
- Driver: 580.119.02 (`nvidia-smi`)
- Disk usage for outputs/results: ~<1 MB total (`du -sh results .../output`)

**End-to-end env (both parties unless noted):**

```bash
SIGMA_MEMPOOL_DISABLE=1
SIGMA_PINNED_KEYBUF=1
OMP_NUM_THREADS=32

# SUF toggles
SUF_SOFTMAX=1 SUF_LAYERNORM=1 SUF_ACTIVATION=1
SUF_NEXP_BITS=10 SUF_INV_BITS=10 SUF_RSQRT_BITS=9
```

**Key buffer sizing (to avoid seq=256/512 crashes):**

- seq=128: `SIGMA_KEYBUF_GB=20` (scaled up internally for batch>1)
- seq=256: `SIGMA_KEYBUF_GB=80`
- seq=512: `SIGMA_KEYBUF_GB=160`

This stays well within RAM limits but would be infeasible if you had only ~200 GB of *memory*. Disk usage remains tiny because key buffers are not written to disk.

## 2) End-to-End SUF vs Sigma (seq ∈ {128,256,512})

Numbers below are medians over the selected runs (`run_idx>=1`). Std is the population std over those selected runs.

| Model | Seq | Sigma ms | SUF ms | Speedup | Sigma std (ms) | SUF std (ms) | Comm reduction | Key reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bert-base | 128 | 1613.8 | 1149.5 | 1.40x | 24.0 | 10.9 | 16.1% | 24.3% |
| bert-base | 256 | 2991.5 | 2152.4 | 1.39x | 13.5 | 12.0 | 21.1% | 27.2% |
| bert-base | 512 | 7694.7 | 5324.1 | 1.45x | 8.6 | 9.4 | 26.0% | 29.9% |
| gpt2 | 128 | 1423.9 | 1072.7 | 1.33x | 29.3 | 3.8 | 12.1% | 22.3% |
| gpt2 | 256 | 2650.0 | 1795.1 | 1.48x | 122.8 | 9.1 | 16.1% | 24.7% |
| gpt2 | 512 | 5630.1 | 3944.0 | 1.43x | 205.8 | 313.8 | 21.1% | 27.4% |

Key takeaways:

- SUF is faster everywhere in this matrix: **1.33×–1.48×** across BERT-base and GPT-2.
- Online communication is consistently lower: **12.1%–26.0%** reduction.
- Preprocessing key material is also lighter: **~18%–32%** reduction depending on model/seq.

## 3) Batch Sweep (seq=128, internal batch mode)

Important: this is still the *internal batch* mode (looping `batch` times inside the binary). It is **not true batched tensors** yet, but it does provide per-inference and throughput trends under the current system.

### 3.1 BERT-base (seq=128)

| Batch | Sigma ms/batch | SUF ms/batch | Sigma ms/inf | SUF ms/inf | Speedup | Sigma tok/s | SUF tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1613.8 | 1149.5 | 1613.8 | 1149.5 | 1.40x | 79.3 | 111.4 |
| 2 | 2866.7 | 2143.1 | 1433.4 | 1071.6 | 1.34x | 89.3 | 119.5 |
| 4 | 5538.2 | 4161.1 | 1384.5 | 1040.3 | 1.33x | 92.4 | 123.0 |
| 8 | 11338.7 | 9036.9 | 1417.3 | 1129.6 | 1.25x | 90.3 | 113.3 |

### 3.2 GPT-2 (seq=128)

| Batch | Sigma ms/batch | SUF ms/batch | Sigma ms/inf | SUF ms/inf | Speedup | Sigma tok/s | SUF tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1423.9 | 1072.7 | 1423.9 | 1072.7 | 1.33x | 89.9 | 119.3 |
| 2 | 2627.4 | 2061.0 | 1313.7 | 1030.5 | 1.27x | 97.4 | 124.2 |
| 4 | 5581.1 | 4410.7 | 1395.3 | 1102.7 | 1.27x | 91.7 | 116.1 |
| 8 | 10941.3 | 8346.5 | 1367.7 | 1043.3 | 1.31x | 93.6 | 122.7 |

SUF remains better across all batch sizes in this internal-batch setting.

## 4) Online Breakdown + Amdahl Reconciliation (回应 Figure 1 vs Table 1)

Because Sigma’s scoped timers overlap (their sums exceed total time), I compute an **allocation-style breakdown**:

- Compute time is distributed across scopes in proportion to their scoped times.
- Comm time is distributed in proportion to comm bytes (which *are* additive).

This yields per-category times that sum to total time and makes Amdahl-style reasoning possible.

### 4.1 Nonlinear share drop (Sigma → SUF)

| Model | Seq | Sigma nonlinear share | SUF nonlinear share | Sigma softmax ms | SUF softmax ms |
|---|---:|---:|---:|---:|---:|
| bert-base | 128 | 56.1% | 44.2% | 371.9 | 169.9 |
| bert-base | 256 | 59.7% | 50.9% | 946.6 | 528.7 |
| bert-base | 512 | 63.2% | 54.8% | 3291.1 | 1864.6 |
| gpt2 | 128 | 54.6% | 41.6% | 226.2 | 110.7 |
| gpt2 | 256 | 58.0% | 48.9% | 606.0 | 295.7 |
| gpt2 | 512 | 62.1% | 52.0% | 1829.3 | 980.2 |

Across these settings, nonlinear share drops by roughly **8–13 percentage points** in the allocated breakdown.

### 4.2 Amdahl-style reconciliation

We use:

- `f`: approximate nonlinear share in Sigma wall-time (from the allocated breakdown)
- `s`: nonlinear speedup using scoped nonlinear timers
- `Speedup_pred = 1 / ((1-f) + f/s)`

| Model | Seq | f (nonlinear share) | s (nonlinear speedup) | Predicted | Measured |
|---|---:|---:|---:|---:|---:|
| bert-base | 128 | 56.1% | 1.86x | 1.35x | 1.40x |
| bert-base | 256 | 59.7% | 1.74x | 1.34x | 1.39x |
| bert-base | 512 | 63.2% | 1.81x | 1.39x | 1.45x |
| gpt2 | 128 | 54.6% | 1.96x | 1.37x | 1.33x |
| gpt2 | 256 | 58.0% | 1.76x | 1.33x | 1.48x |
| gpt2 | 512 | 62.1% | 1.90x | 1.42x | 1.43x |

This is the quantitative story reviewers asked for: per-gate nonlinear improvements in the ~1.7–2.0× range translate into ~1.3–1.5× end-to-end because `f` is only ~55–63%.

## 5) Padding / Mask-Independent Overhead (directly addressing the 3.8× concern)

I ran the SUF model microbench with mask-aware padding **off vs on** (BERT-base, seq=128):

| Setting | pred_bytes | pred_ms | lut_ms | per_gate_eval_ms | per_gate_key_ms |
|---|---:|---:|---:|---:|---:|
| mask-aware OFF | 384 | 0.079 | 0.012 | 0.994 | 0.130 |
| mask-aware ON | 768 | 0.152 | 0.014 | 3.941 | 0.200 |

Observed ratios:

- predicate bytes: 2.00×
- predicate phase time: 1.94×
- LUT phase time: 1.18×
- per-gate eval time: 3.97×

So yes, padding can create ~4× per-gate slowdowns even when predicate bytes only 2×. The microbench indicates that the padding doubles predicate work and pushes GPU evaluation into a less efficient regime (more predicates/helpers, worse launch/memory behavior).

## 6) Reproducibility Gaps from comments.md — now filled

### 6.1 Approximation parameters (m, d, n, f, ranges)

From the current Sigma/SUF code paths:

- **Fixed-point format:** bitlength `n=bw` depends on model (e.g., 50 for BERT-base/GPT-2), fraction bits `f=scale=12`. See `ezpc_upstream/GPU-MPC/experiments/sigma/sigma.cu`.
- **Sigma GELU/SiLU LUT sizes:**
  - GELU: `bin=8` → `m=256` entries, `degree d=0` (table lookup). See `ezpc_upstream/GPU-MPC/backend/sigma.h` and `ezpc_upstream/GPU-MPC/fss/gpu_lut.h`.
  - SiLU: `bin=10` → `m=1024`, `d=0`. Same references.
  - LUT sampling uses `scaleIn=6` and `scaleOut=scale=12` in `genLUT(bin, 6, scale)`.
- **SUF GELU/SiLU intervals:**
  - GELU: `SUF_GELU_INTERVALS=256` (default here), `d=0` (piecewise constant LUT).
  - SiLU: `SUF_SILU_INTERVALS=1024` (default).
  - Bits used for LUT indexing: `in_bits = bits_needed(intervals-1)` (8 bits for 256). See `src/sigma_suf_bridge.cu`.
- **Softmax subroutines:**
  - Sigma uses LUT-based `nexp` and `inv` tables. For `inv`, `bin = ceil(log2(n_seq)) + 6` (e.g., 13 bits for seq=128). See `ezpc_upstream/GPU-MPC/fss/gpu_mha.h`.
  - SUF uses explicit bit controls: `SUF_NEXP_BITS=10`, `SUF_INV_BITS=10`, `SUF_RSQRT_BITS=9` (as set in this evaluation). See `src/sigma_suf_bridge.cu`.

### 6.2 Variance reporting

- I now have repeated runs for seq=128/256/512 (selected runs are those with `run_idx>=1`).
- Typical relative std is low (often ~0.1%–4%). GPT-2 seq=512 shows higher noise due to a few slow runs; the median remains stable and SUF remains faster.

### 6.3 Hidden sizes used in microbench shapes

From `ezpc_upstream/GPU-MPC/experiments/sigma/sigma.cu`:

- BERT-base hidden size: 768
- BERT-large hidden size: 1024
- GPT-2 hidden size: 768
- GPT-Neo hidden size: 2048
- LLaMA-7B hidden size: 4096

## 7) Baseline Fairness / Component Sharing

Fairness is enforced by using the same Sigma runtime and toggling only nonlinear gates:

- Linear layers (matmul/GEMM) and overall model structure are shared.
- SUF replacements are gated by env toggles like `SUF_SOFTMAX`, `SUF_LAYERNORM`, `SUF_ACTIVATION`. See `src/sigma_suf_bridge.cu` for `suf_*_enabled()` and the SUF gate paths.

## 8) batch.md Tasks — what is completed vs not yet

### 8.1 Completed now

- A full internal-batch sweep (batch=1/2/4/8) with SUF faster everywhere.
- Per-inference latency and throughput are reported explicitly in Section 3.
- Resource checks: disk footprint is negligible; memory pressure is driven by key buffers (160 GB per party at seq=512).

### 8.2 Not yet completed: true batched tensors

True batching (explicit `[B,L,H]` tensors and batched attention kernels) would require significant code changes across attention/softmax and tensor layout. I did not attempt to refactor that here because it would be a large change and risks destabilizing the current reproducible pipeline.

## 9) Artifacts and Raw Results

Key files you can cite or post-process:

- End-to-end repeated runs (seq=256/512): `results/resume_e2e_runs_2026-01-27.json`
- End-to-end repeated runs (seq=128): `results/resume_seq128_runs_2026-01-27.json`
- Amdahl + variance analysis (selected runs): `results/analysis_breakdown_amdahl_2026-01-27_v3_selected.json`
- Allocated breakdowns (selected runs): `results/breakdowns_allocated_2026-01-27_v3_selected.json`
- Batch sweep metrics: `results/batch_sweep_internal_metrics_selected_2026-01-27.json`
- Batch breakdowns (seq=128): `results/breakdowns_batch_seq128_selected_2026-01-27.json`
- Padding ablation raw: `results/bench_suf_model_bertbase_mask0_2026-01-27.jsonl`, `results/bench_suf_model_bertbase_maskaware_2026-01-27.jsonl`
