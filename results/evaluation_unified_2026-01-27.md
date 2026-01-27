# Unified Evaluation (all useful results) — 2026-01-27

Measurement policy (统一口径):
- End-to-end and gate microbench are derived from end-to-end logs.
- Selection: use `run_idx >= 1` and take the median across selected runs.
- LAN/WAN projection uses `T = comp_time + 2 * comm_bytes / bandwidth + rounds * latency`.
- Internal batch sweep uses `SIGMA_BATCH` (looping), not true `[B,L,H]` batching.

Primary sources:
- `results/resume_seq128_runs_2026-01-27.json`
- `results/resume_seq128_more_models_2026-01-27.json`
- `results/resume_e2e_runs_2026-01-27.json`
- `results/resume_seq32_64_runs_2026-01-27.json`
- `results/ablation_mask0_bertbase_2026-01-27.jsonl`
- `results/ablation_maskaware_bertbase_2026-01-27.jsonl`
- `results/ablation_cache_on_bertbase_2026-01-27.jsonl`
- `results/ablation_cache_off_bertbase_2026-01-27.jsonl`
- `results/ablation_cpu_vs_gpu_bertbase_2026-01-27.jsonl`
- `results/batch_sweep_internal_metrics_selected_2026-01-27.json`

## 0) Experiment configuration (aligned with evaluation_latest.md)

### 0.1 Environment
- OS: Ubuntu 24.04.3 LTS
- CPU: 2x AMD EPYC 9654 (96 cores/socket, 384 CPUs total)
- RAM: ~1.5 TiB
- GPU: 2x NVIDIA RTX PRO 6000 Blackwell Workstation Edition (97,887 MiB each, sm_120)
- CUDA: 13.0 (`nvcc` build cuda_13.0.r13.0), driver 580.119.02
- Sigma baseline: `ezpc_upstream/GPU-MPC/experiments/sigma` with `build/gpu_mpc_upstream/sigma`
- SUF: `third_party/EzPC_vendor/GPU-MPC/experiments/sigma` with `build/gpu_mpc_vendor/sigma`
- GPU binding: party-0 uses GPU0, party-1 uses GPU1 (via `CUDA_VISIBLE_DEVICES=0/1`)

### 0.2 Network model (for LAN/WAN projection only)
Projection formula:
```text
T = comp_time + 2 * comm_bytes / bandwidth + rounds * latency
```
- LAN: 1 GB/s, 0.5 ms
- WAN: 400 MB/s, 4 ms
- `comp_time` is estimated as `(total_us - comm_us)` from `evaluator.txt`

Rounds used for projection (protocol-determined constants reused across reruns):

| Model / Seq | Sigma rounds | SUF rounds |
|---|---:|---:|
| bert-tiny-128 | 188 | 186 |
| bert-base-32 | 1080 | 1068 |
| bert-base-64 | 1104 | 1092 |
| bert-base-128 | 1128 | 1116 |
| bert-large-128 | 2256 | 2232 |
| gpt2-128 | 1128 | 1116 |
| gpt2-256 | 1152 | 1140 |
| gpt-neo-128 | 2256 | 2232 |

### 0.3 Run settings (env flags and selection policy)
Common env flags (both Sigma and SUF unless noted):
- `SIGMA_MEMPOOL_DISABLE=1`
- `SIGMA_PINNED_KEYBUF=1`
- `OMP_NUM_THREADS=32`

SUF-specific flags (both parties):
```text
SUF_SOFTMAX=1 SUF_LAYERNORM=1 SUF_ACTIVATION=1
SUF_NEXP_BITS=10 SUF_INV_BITS=10 SUF_RSQRT_BITS=9
```

Selection and aggregation:
- Each setting is run multiple times (typically 3 runs).
- Reported values use runs with `run_idx >= 1` and take the median.
- `run_idx=0` is treated as warmup/outlier by default.

### 0.4 Key buffer sizing (`SIGMA_KEYBUF_GB`)
Key buffer overrides used to avoid OOM/segfault and keep the pipeline stable:

| Setting | `SIGMA_KEYBUF_GB` |
|---|---:|
| bert-tiny, seq=128 | 4 |
| bert-base, seq in {32, 64} | 10 |
| bert-base/gpt2, seq=128 | 20 |
| bert-large, seq=128 | 60 |
| gpt-neo, seq=128 | 80 |
| bert-base/gpt2, seq=256 | 80 |
| bert-base/gpt2, seq=512 | 160 |

Notes:
- When `SIGMA_BATCH > 1` and no override is provided, Sigma scales key buffers internally.
- These overrides are explicitly set for reproducibility and stability.

### 0.5 Repro commands (same-mouth runs)
End-to-end runs (example: BERT-base, seq=128).

Sigma (party 0 then party 1):
```bash
cd /workspace/SUF_new/ezpc_upstream/GPU-MPC/experiments/sigma
SIGMA_MEMPOOL_DISABLE=1 SIGMA_PINNED_KEYBUF=1 SIGMA_KEYBUF_GB=20 \
OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=0 \
/workspace/SUF_new/build/gpu_mpc_upstream/sigma bert-base 128 0 127.0.0.1 32
SIGMA_MEMPOOL_DISABLE=1 SIGMA_PINNED_KEYBUF=1 SIGMA_KEYBUF_GB=20 \
OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=1 \
/workspace/SUF_new/build/gpu_mpc_upstream/sigma bert-base 128 1 127.0.0.1 32
```

SUF (party 0 then party 1):
```bash
cd /workspace/SUF_new/third_party/EzPC_vendor/GPU-MPC/experiments/sigma
SIGMA_MEMPOOL_DISABLE=1 SIGMA_PINNED_KEYBUF=1 SIGMA_KEYBUF_GB=20 \
OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=0 \
SUF_SOFTMAX=1 SUF_LAYERNORM=1 SUF_ACTIVATION=1 \
SUF_NEXP_BITS=10 SUF_INV_BITS=10 SUF_RSQRT_BITS=9 \
/workspace/SUF_new/build/gpu_mpc_vendor/sigma bert-base 128 0 127.0.0.1 32
SIGMA_MEMPOOL_DISABLE=1 SIGMA_PINNED_KEYBUF=1 SIGMA_KEYBUF_GB=20 \
OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=1 \
SUF_SOFTMAX=1 SUF_LAYERNORM=1 SUF_ACTIVATION=1 \
SUF_NEXP_BITS=10 SUF_INV_BITS=10 SUF_RSQRT_BITS=9 \
/workspace/SUF_new/build/gpu_mpc_vendor/sigma bert-base 128 1 127.0.0.1 32
```

For other settings, change `(model, seq)` and use the keybuf table above.

Ablations (microbench; BERT-base GELU gate):
```bash
cd /workspace/SUF_new
./build/bench_suf_model --model bert-base --seq 128 --iters 50 --json
./build/bench_suf_model --model bert-base --seq 128 --iters 50 --mask-aware --mask 123456789 --json
./build/bench_suf_model --model bert-base --seq 128 --iters 50 --mask-aware --mask 123456789 --no-template-cache --json
./build/bench_suf_model --model bert-base --seq 128 --iters 1 --mask-aware --mask 123456789 --cpu-eval --json
```

Internal batch sweep (not true batching):
```bash
cd /workspace/SUF_new/ezpc_upstream/GPU-MPC/experiments/sigma
SIGMA_BATCH=4 SIGMA_MEMPOOL_DISABLE=1 SIGMA_PINNED_KEYBUF=1 SIGMA_KEYBUF_GB=20 \
OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=0 \
/workspace/SUF_new/build/gpu_mpc_upstream/sigma bert-base 128 0 127.0.0.1 32
```

## 1) Seq=128 end-to-end (absolute values + LAN/WAN)

| Model | Sigma online (ms) | SUF online (ms) | Speedup | Sigma comm (GiB) | SUF comm (GiB) | Comm ↓ | Sigma keygen (s) | SUF keygen (s) | Keygen ↓ | Sigma key (GiB) | SUF key (GiB) | Key ↓ | Sigma LAN (s) | SUF LAN (s) | Sigma WAN (s) | SUF WAN (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bert-tiny-128 | 63.9 | 42.6 | 1.50x | 0.020 | 0.017 | 16.2% | 0.07 | 0.06 | 12.4% | 0.326 | 0.250 | 23.3% | 0.19 | 0.17 | 0.92 | 0.87 |
| bert-base-128 | 1613.8 | 1149.5 | 1.40x | 0.989 | 0.830 | 16.1% | 1.32 | 1.08 | 18.2% | 16.835 | 12.739 | 24.3% | 3.99 | 3.36 | 11.12 | 9.94 |
| bert-large-128 | 4034.5 | 2997.9 | 1.35x | 2.638 | 2.213 | 16.1% | 3.21 | 2.48 | 22.6% | 45.448 | 34.529 | 24.0% | 10.18 | 8.45 | 26.58 | 23.39 |
| gpt2-128 | 1423.9 | 1072.7 | 1.33x | 0.824 | 0.724 | 12.1% | 1.20 | 0.99 | 17.7% | 14.292 | 11.101 | 22.3% | 3.55 | 3.07 | 10.16 | 9.31 |
| gpt-neo-128 | 6326.2 | 5115.8 | 1.24x | 4.029 | 3.648 | 9.5% | 5.42 | 4.35 | 19.8% | 76.187 | 61.215 | 19.7% | 15.30 | 13.54 | 36.18 | 33.10 |

### 1.1 Seq=128 absolute breakdown (time + comm)

| Model | Variant | comm time (ms) | comp time (ms) | softmax time (ms) | GELU time (ms) | layernorm time (ms) | truncate time (ms) | total comm (GiB) | softmax comm (GiB) | GELU comm (GiB) | layernorm comm (GiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bert-tiny-128 | sigma | 6.3 | 57.6 | 16.4 | 4.2 | 14.2 | 5.1 | 0.020 | 0.005 | 0.003 | 0.002 |
| bert-tiny-128 | suf | 5.6 | 37.0 | 9.8 | 1.6 | 7.6 | 4.5 | 0.017 | 0.003 | 0.003 | 0.002 |
| bert-base-128 | sigma | 316.2 | 1297.5 | 356.1 | 211.9 | 145.8 | 193.6 | 0.989 | 0.259 | 0.159 | 0.111 |
| bert-base-128 | suf | 129.0 | 1020.4 | 160.4 | 42.5 | 80.3 | 203.6 | 0.830 | 0.139 | 0.120 | 0.111 |
| bert-large-128 | sigma | 645.5 | 3388.9 | 785.8 | 516.6 | 309.4 | 601.0 | 2.638 | 0.691 | 0.425 | 0.297 |
| bert-large-128 | suf | 417.2 | 2580.7 | 399.3 | 118.8 | 205.2 | 504.4 | 2.213 | 0.372 | 0.319 | 0.296 |
| gpt2-128 | sigma | 204.9 | 1219.0 | 228.5 | 197.0 | 151.8 | 221.8 | 0.824 | 0.131 | 0.159 | 0.111 |
| gpt2-128 | suf | 119.7 | 953.0 | 101.8 | 40.4 | 83.5 | 181.4 | 0.724 | 0.071 | 0.120 | 0.111 |
| gpt-neo-128 | sigma | 801.7 | 5524.5 | 442.8 | 921.5 | 518.6 | 1081.2 | 4.029 | 0.357 | 0.867 | 0.604 |
| gpt-neo-128 | suf | 523.6 | 4592.2 | 252.7 | 232.8 | 323.2 | 1010.2 | 3.648 | 0.193 | 0.650 | 0.604 |

## 2) Sequence sweeps (absolute values)

### 2.1 BERT-base (32/64/128, same-mouth rerun)

| Seq | Sigma online (ms) | SUF online (ms) | Speedup | Sigma comm (GiB) | SUF comm (GiB) | Comm ↓ | Sigma keygen (s) | SUF keygen (s) | Keygen ↓ | Sigma key (GiB) | SUF key (GiB) | Key ↓ | Sigma LAN (s) | SUF LAN (s) | Sigma WAN (s) | SUF WAN (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 642.3 | 545.5 | 1.18x | 0.185 | 0.167 | 9.4% | 0.60 | 0.56 | 7.3% | 3.706 | 3.024 | 18.4% | 1.52 | 1.39 | 5.89 | 5.66 |
| 64 | 947.4 | 696.0 | 1.36x | 0.411 | 0.361 | 12.1% | 0.98 | 0.70 | 28.4% | 7.431 | 5.839 | 21.4% | 2.27 | 1.93 | 7.46 | 6.92 |
| 128 | 1613.8 | 1149.5 | 1.40x | 0.989 | 0.830 | 16.1% | 1.32 | 1.08 | 18.2% | 16.835 | 12.739 | 24.3% | 3.99 | 3.36 | 11.12 | 9.94 |

### 2.2 BERT-base (128/256/512)

| Seq | Sigma online (ms) | SUF online (ms) | Speedup | Sigma comm (GiB) | SUF comm (GiB) | Comm ↓ | Sigma keygen (s) | SUF keygen (s) | Keygen ↓ | Sigma key (GiB) | SUF key (GiB) | Key ↓ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1613.8 | 1149.5 | 1.40x | 0.989 | 0.830 | 16.1% | 1.32 | 1.08 | 18.2% | 16.835 | 12.739 | 24.3% |
| 256 | 2991.5 | 2152.4 | 1.39x | 2.647 | 2.088 | 21.1% | 2.99 | 2.03 | 32.2% | 43.460 | 31.619 | 27.2% |
| 512 | 7694.7 | 5324.1 | 1.45x | 7.966 | 5.891 | 26.0% | 7.08 | 5.01 | 29.2% | 127.981 | 89.704 | 29.9% |

### 2.3 GPT-2 (128/256/512)

| Seq | Sigma online (ms) | SUF online (ms) | Speedup | Sigma comm (GiB) | SUF comm (GiB) | Comm ↓ | Sigma keygen (s) | SUF keygen (s) | Keygen ↓ | Sigma key (GiB) | SUF key (GiB) | Key ↓ | Sigma LAN (s) | SUF LAN (s) | Sigma WAN (s) | SUF WAN (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1423.9 | 1072.7 | 1.33x | 0.824 | 0.724 | 12.1% | 1.20 | 0.99 | 17.7% | 14.292 | 11.101 | 22.3% | 3.55 | 3.07 | 10.16 | 9.31 |
| 256 | 2650.0 | 1795.1 | 1.48x | 1.983 | 1.663 | 16.1% | 2.11 | 1.65 | 21.9% | 33.191 | 24.984 | 24.7% | 7.04 | 5.69 | 17.46 | 15.04 |
| 512 | 5630.1 | 3944.0 | 1.43x | 5.302 | 4.183 | 21.1% | 5.16 | 3.73 | 27.8% | 86.687 | 62.975 | 27.4% | - | - | - | - |

## 3) Absolute communication and key sizes (GiB)

| Model | Seq | Sigma comm (GiB) | SUF comm (GiB) | Comm ↓ | Sigma key (GiB) | SUF key (GiB) | Key ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| bert-base | 32 | 0.185 | 0.167 | 9.4% | 3.706 | 3.024 | 18.4% |
| bert-base | 64 | 0.411 | 0.361 | 12.1% | 7.431 | 5.839 | 21.4% |
| bert-base | 128 | 0.989 | 0.830 | 16.1% | 16.835 | 12.739 | 24.3% |
| bert-base | 256 | 2.647 | 2.088 | 21.1% | 43.460 | 31.619 | 27.2% |
| bert-base | 512 | 7.966 | 5.891 | 26.0% | 127.981 | 89.704 | 29.9% |
| bert-large | 128 | 2.638 | 2.213 | 16.1% | 45.448 | 34.529 | 24.0% |
| bert-tiny | 128 | 0.020 | 0.017 | 16.2% | 0.326 | 0.250 | 23.3% |
| gpt-neo | 128 | 4.029 | 3.648 | 9.5% | 76.187 | 61.215 | 19.7% |
| gpt2 | 128 | 0.824 | 0.724 | 12.1% | 14.292 | 11.101 | 22.3% |
| gpt2 | 256 | 1.983 | 1.663 | 16.1% | 33.191 | 24.984 | 24.7% |
| gpt2 | 512 | 5.302 | 4.183 | 21.1% | 86.687 | 62.975 | 27.4% |

## 4) Gate microbench (same-mouth; derived from end-to-end logs)

Per-gate GELU is computed as `gelu_us / n_layer` and GELU comm per-gate as `gelu_comm_bytes / n_layer`.

### 4.1 Seq=128 across models

| Model | Sigma GELU /gate (ms) | SUF GELU /gate (ms) | Gate speedup | Sigma GELU comm /gate (MiB) | SUF GELU comm /gate (MiB) | Comm ↓ |
|---|---:|---:|---:|---:|---:|---:|
| bert-tiny-128 | 2.11 | 0.80 | 2.63x | 1.66 | 1.30 | 21.7% |
| bert-base-128 | 17.66 | 3.54 | 4.99x | 13.59 | 10.22 | 24.8% |
| bert-large-128 | 21.53 | 4.95 | 4.35x | 18.12 | 13.62 | 24.8% |
| gpt2-128 | 16.42 | 3.37 | 4.87x | 13.59 | 10.22 | 24.8% |
| gpt-neo-128 | 38.40 | 9.70 | 3.96x | 37.00 | 27.75 | 25.0% |

### 4.2 BERT-base across seq lengths (32/64/128)

| Seq | Sigma GELU /gate (ms) | SUF GELU /gate (ms) | Gate speedup | Sigma GELU comm /gate (MiB) | SUF GELU comm /gate (MiB) | Comm ↓ |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 5.33 | 1.10 | 4.84x | 3.40 | 2.55 | 24.8% |
| 64 | 8.72 | 1.84 | 4.74x | 6.80 | 5.11 | 24.8% |
| 128 | 17.66 | 3.54 | 4.99x | 13.59 | 10.22 | 24.8% |

## 5) Ablations (microbench; local GPU/CPU timings)

### 5.1 Mask-independent shapes via padding (BERT-base GELU gate)

| Variant | mask_aware | per_gate_eval_ms | total_eval_ms | per_gate_key_bytes | pred_bytes | lut_bytes | init_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| mask-dependent | false | 0.995 | 11.946 | 2464 | 384 | 2080 | 1.255 |
| fixed-shape | true | 3.950 | 47.396 | 2848 | 768 | 2080 | 0.762 |

Padding overhead: eval 3.97x; key bytes 1.16x.

### 5.2 Program reuse / template cache

| Variant | template_cache | #program_inits | init metric (ms) | total_eval_ms | per_gate_eval_ms |
|---|---:|---:|---:|---:|---:|
| cached | true | 1 | init_ms=0.761 | 47.393 | 3.949 |
| no cache | false | 12 | total_init_ms=20.435 | 47.392 | 3.949 |

No-cache extra init: 20.435 ms over 12 gates.

### 5.3 GPU secure vs CPU reference

| Mode | per_gate_eval_ms | total_eval_ms | speedup vs CPU |
|---|---:|---:|---:|
| GPU secure | 3.956 | 47.473 | 92.0x |
| CPU ref | 363.798 | 4365.570 | 1.0x |

## 6) Internal batch sweep (seq=128; not true batching)

These results come from `SIGMA_BATCH` (looping batch times), not a single batched tensor.

### 6.1 bert-base

| Batch | Sigma ms/inf | SUF ms/inf | Speedup | Sigma tok/s | SUF tok/s | Sigma comm/inf (GiB) | SUF comm/inf (GiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1613.8 | 1149.5 | 1.40x | 79.3 | 111.4 | 0.989 | 0.830 |
| 2 | 1433.4 | 1071.6 | 1.34x | 89.3 | 119.5 | 0.989 | 0.830 |
| 4 | 1384.5 | 1040.3 | 1.33x | 92.4 | 123.0 | 0.989 | 0.830 |
| 8 | 1417.3 | 1129.6 | 1.25x | 90.3 | 113.3 | 0.989 | 0.830 |

### 6.2 gpt2

| Batch | Sigma ms/inf | SUF ms/inf | Speedup | Sigma tok/s | SUF tok/s | Sigma comm/inf (GiB) | SUF comm/inf (GiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1423.9 | 1072.7 | 1.33x | 89.9 | 119.3 | 0.824 | 0.724 |
| 2 | 1313.7 | 1030.5 | 1.27x | 97.4 | 124.2 | 0.824 | 0.724 |
| 4 | 1395.3 | 1102.7 | 1.27x | 91.7 | 116.1 | 0.824 | 0.724 |
| 8 | 1367.7 | 1043.3 | 1.31x | 93.6 | 122.7 | 0.824 | 0.724 |
