# Evaluation Results with Rounds + LAN/WAN Projection (2026-01-22)

## 1. Environment
- **OS**: Ubuntu 24.04.3 LTS
- **CPU**: AMD EPYC 9374F 32-Core Processor (1 socket, 64 threads)
- **RAM**: 314 GiB (swap 8.0 GiB)
- **GPU**: 2× NVIDIA GeForce RTX 5090 (sm_120)
- **CUDA**: 13.0 (nvcc 13.0.88)
- **Sigma baseline**: `ezpc_upstream/GPU-MPC` (async malloc enabled, mempool disabled)
- **SUF**: `third_party/EzPC/GPU-MPC` with SUF bridge enabled
- **SUF allocator**: async malloc enabled with `SIGMA_MEMPOOL_GB=0`

## 2. Network model
Projection formula:
```
T = comp_time + 2 * comm_bytes / bandwidth + rounds * latency
```
- **LAN**: 1 GB/s, 0.5 ms
- **WAN**: 400 MB/s, 4 ms
- **comp_time** is estimated as `avg_time_per_batch - comm_time/batch` (all in seconds).
- **bandwidth units** use bytes/s (1 GB = 1e9, 400 MB = 400e6).

## 3. SUF vs Sigma (end-to-end, seq=128)
| Model | Sigma online (ms) | SUF online (ms) | Speedup | Sigma comm (GB) | SUF comm (GB) | Sigma rounds | SUF rounds | Sigma LAN (s) | SUF LAN (s) | Sigma WAN (s) | SUF WAN (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bert-tiny-128 | 45.04 | 34.72 | 1.30x | 0.020 | 0.017 | 188 | 186 | 0.17 | 0.15 | 0.90 | 0.86 |
| bert-base-128 | 1072.57 | 922.80 | 1.16x | 0.989 | 0.830 | 1128 | 1116 | 3.48 | 2.98 | 10.62 | 9.56 |
| bert-large-128 | 2867.65 | 2281.95 | 1.26x | 2.638 | 2.213 | 2256 | 2232 | 8.96 | 7.45 | 25.36 | 22.40 |
| gpt2-128 | 933.82 | 711.68 | 1.31x | 0.824 | 0.724 | 1128 | 1116 | 3.04 | 2.63 | 9.65 | 8.87 |
| gpt-neo-128 | 4856.00 | 3935.95 | 1.23x | 4.029 | 3.648 | 2256 | 2232 | 13.26 | 11.77 | 34.14 | 31.33 |

### 3.1 Keygen and key size
| Model | Sigma keygen (s) | SUF keygen (s) | Sigma key (GB) | SUF key (GB) |
|---|---|---|---|---|
| bert-tiny-128 | 0.37 | 0.05 | 0.326 | 0.250 |
| bert-base-128 | 3.26 | 0.55 | 16.835 | 12.739 |
| bert-large-128 | 8.54 | 1.44 | 45.448 | 34.529 |
| gpt2-128 | 3.67 | 0.72 | 14.292 | 11.101 |
| gpt-neo-128 | 8.37 | 2.54 | 76.187 | 61.215 |

## 4. Additional sequence points (GPT-2 / GPT-Neo)
| Model | Sigma time (ms) | SUF time (ms) | Speedup | Sigma comm (GB) | SUF comm (GB) | Sigma rounds | SUF rounds | Sigma LAN (s) | SUF LAN (s) | Sigma WAN (s) | SUF WAN (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt2-64 | 542.19 | 441.65 | 1.23x | 0.370 | 0.335 | 1104 | 1092 | 1.77 | 1.59 | 6.83 | 6.49 |
| gpt2-128 | 933.82 | 711.68 | 1.31x | 0.824 | 0.724 | 1128 | 1116 | 3.04 | 2.63 | 9.65 | 8.87 |
| gpt2-256 | 2211.57 | 1453.51 | 1.52x | 1.983 | 1.663 | 1152 | 1140 | 6.37 | 5.12 | 16.79 | 14.47 |
| gpt-neo-64 | 2837.95 | 2371.88 | 1.20x | 1.900 | 1.750 | 2208 | 2184 | 7.40 | 6.73 | 21.24 | 20.02 |
| gpt-neo-128 | 4856.00 | 3935.95 | 1.23x | 4.029 | 3.648 | 2256 | 2232 | 13.26 | 11.77 | 34.14 | 31.33 |

## 5. Scaling (BERT-base seq sweep)
| Seq | Sigma time (ms) | SUF time (ms) | Speedup | Sigma comm (GB) | SUF comm (GB) | Sigma rounds | SUF rounds | Sigma LAN (s) | SUF LAN (s) | Sigma WAN (s) | SUF WAN (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 379.37 | 323.30 | 1.17x | 0.185 | 0.167 | 1080 | 1068 | 1.25 | 1.15 | 5.63 | 5.43 |
| 64 | 602.43 | 440.93 | 1.37x | 0.411 | 0.361 | 1104 | 1092 | 1.89 | 1.66 | 7.08 | 6.65 |
| 128 | 1072.57 | 922.80 | 1.16x | 0.989 | 0.830 | 1128 | 1116 | 3.48 | 2.98 | 10.62 | 9.56 |

**Seq=256**: both Sigma and SUF failed with `cudaMemcpy` invalid argument / runtime error (same as prior report).

## 6. Kernel microbench (activation)
| Model / Gate | Sigma per-gate eval (ms) | SUF per-gate eval (ms) | Speedup | SUF per-gate key (bytes) |
|---|---|---|---|---|
| bert-base GELU | 20.070 | 0.285 | 70.5x | 2464 |
| bert-large GELU | 25.457 | 0.363 | 70.2x | 2464 |
| gpt2 GELU | 22.375 | 0.285 | 78.4x | 2464 |
| llama7b SILU | 56.976 | 1.114 | 51.1x | 8672 |

**Notes**: Sigma activation tests were run with `SIGMA_DISABLE_ASYNC_MALLOC=1` and `SIGMA_COMPRESS=0` to avoid CUDA illegal access seen with compression on sm_120.

## 7. SHAFT baselines (local runs)
All SHAFT runs use `CUDA_VISIBLE_DEVICES=0` (launcher does not bind GPUs by rank).
Projection formula:
```
T = comp_time + 2 * comm_bytes / bandwidth + rounds * latency
```

### 7.1 Unit-test microbench (SHAFT)
**Softmax** (`examples/unit-test/run_test_softmax.py`):
| L | Time (s) | Bytes (MB) | Rounds |
|---:|---:|---:|---:|
| 32 | 0.0908 | 0.0596 | 41 |
| 64 | 0.0805 | 0.1191 | 41 |
| 128 | 0.0814 | 0.2383 | 41 |
| 256 | 0.0786 | 0.4766 | 41 |

**GELU** (`examples/unit-test/run_test_gelu.py`):
- Max error: **0.0045**, Avg error: **0.000739**

| Shape | Time (s) | Bytes (MB) | Rounds |
|---|---:|---:|---:|
| (128, 3072) | 0.2352 | 354 | 19 |
| (128, 4096) | 0.2454 | 472 | 19 |

### 7.2 End-to-end transformer inference (SHAFT, seq=128)
| Model | Comp (s) | Comm (GB) | Rounds | LAN (s) | WAN (s) |
|---|---:|---:|---:|---:|---:|
| bert-base-128 | 3.09 | 10.46 | 1496 | 24.76 | 61.37 |
| bert-large-128 | 7.60 | 28.46 | 2936 | 65.99 | 161.64 |

**GPT-2**: local run failed in `run_generation_private.py` with device mismatch (`cuda:0` vs `cpu`), see `/tmp/shaft_gpt2_64_comp.log`.
