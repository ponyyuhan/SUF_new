# Evaluation Results (2026-01-26)

## 1. Environment
- **OS**: Ubuntu 24.04.3 LTS
- **CPU**: 2× AMD EPYC 9654 96-Core Processor (192 cores / 384 threads)
- **RAM**: 1.5 TiB (swap 0)
- **GPU**: 2× NVIDIA RTX PRO 6000 Blackwell Workstation Edition (97,887 MiB each, sm_120)
- **Storage**: 7.0 TB total, 4.1 TB free on `/`
- **CUDA**: 13.0 (nvcc 13.0.88), driver 580.119.02
- **Sigma baseline**: `ezpc_upstream/GPU-MPC` (`SIGMA_MEMPOOL_DISABLE=1`, `OMP_NUM_THREADS=32`)
- **SUF**: `third_party/EzPC_vendor/GPU-MPC` (SUF bridge enabled)
- **SUF settings** (both parties):
  ```
  SUF_SOFTMAX=1 SUF_LAYERNORM=1 SUF_ACTIVATION=1
  SUF_NEXP_BITS=10 SUF_INV_BITS=10 SUF_RSQRT_BITS=9
  ```
- **SHAFT venv**: not present (SHAFT not rerun in this pass)

**GPU binding**:
- Sigma/SUF: party‑0 uses GPU0, party‑1 uses GPU1.
- SHAFT: not rerun (prior dual‑GPU settings were `SHAFT_GPU0=0`, `SHAFT_GPU1=1`).

## 2. Network model
Projection formula:
```
T = comp_time + 2 * comm_bytes / bandwidth + rounds * latency
```
- **LAN**: 1 GB/s, 0.5 ms
- **WAN**: 400 MB/s, 4 ms
- **comp_time** for Sigma/SUF is estimated as `(total_time - comm_time)` from `evaluator.txt` (all in seconds).
- **Rounds** for Sigma/SUF are protocol‑determined; Sigma/SUF logs do not expose rounds, so we reuse the fixed per‑model counts from prior instrumentation (unchanged for the same model/seq/flags).

## 3. SUF vs Sigma (end‑to‑end, seq=128)

| Model | Sigma online (ms) | SUF online (ms) | Speedup | Sigma comm (GB) | SUF comm (GB) | Sigma rounds | SUF rounds | Sigma LAN (s) | SUF LAN (s) | Sigma WAN (s) | SUF WAN (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BERT‑tiny‑128 | 56.56 | 41.06 | 1.38x | 0.020 | 0.017 | 188 | 186 | 0.19 | 0.17 | 0.91 | 0.87 |
| BERT‑base‑128 | 1408.73 | 1130.98 | 1.25x | 0.989 | 0.830 | 1128 | 1116 | 3.88 | 3.29 | 11.01 | 9.86 |
| BERT‑large‑128 | 3917.54 | 2666.96 | 1.47x | 2.638 | 2.213 | 2256 | 2232 | 9.95 | 8.20 | 26.35 | 23.14 |
| GPT‑2‑128 | 1240.44 | 961.21 | 1.29x | 0.824 | 0.724 | 1128 | 1116 | 3.42 | 2.95 | 10.02 | 9.18 |
| GPT‑Neo‑128 | 5837.49 | 4827.64 | 1.21x | 4.029 | 3.648 | 2256 | 2232 | 14.93 | 13.12 | 35.80 | 32.68 |

### 3.1 Keygen and key size
| Model | Sigma keygen (s) | SUF keygen (s) | Sigma key (GB) | SUF key (GB) |
|---|---:|---:|---:|---:|
| BERT‑tiny‑128 | 0.08 | 0.06 | 0.326 | 0.250 |
| BERT‑base‑128 | 1.32 | 0.97 | 16.835 | 12.739 |
| BERT‑large‑128 | 2.98 | 2.27 | 45.448 | 34.529 |
| GPT‑2‑128 | 1.10 | 0.90 | 14.292 | 11.101 |
| GPT‑Neo‑128 | 4.89 | 4.02 | 76.187 | 61.215 |

### 3.2 Additional sequence points (GPT‑2 / GPT‑Neo)
| Model | Seq | Sigma time (ms) | SUF time (ms) | Speedup | Sigma comm (GB) | SUF comm (GB) | Sigma rounds | SUF rounds | Sigma LAN (s) | SUF LAN (s) | Sigma WAN (s) | SUF WAN (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT‑2 | 64 | 841.84 | 634.11 | 1.33x | 0.370 | 0.335 | 1104 | 1092 | 2.08 | 1.84 | 7.14 | 6.74 |
| GPT‑2 | 128 | 1240.44 | 961.21 | 1.29x | 0.824 | 0.724 | 1128 | 1116 | 3.42 | 2.95 | 10.02 | 9.18 |
| GPT‑2 | 256 | 2607.10 | 1735.40 | 1.50x | 1.983 | 1.663 | 1152 | 1140 | 6.88 | 5.59 | 17.30 | 14.93 |
| GPT‑Neo | 64 | 3958.84 | 3321.70 | 1.19x | 1.900 | 1.750 | 2208 | 2184 | 8.73 | 7.90 | 22.58 | 21.18 |
| GPT‑Neo | 128 | 5837.49 | 4827.64 | 1.21x | 4.029 | 3.648 | 2256 | 2232 | 14.93 | 13.12 | 35.80 | 32.68 |

### 3.3 Scaling (BERT‑base seq sweep)
| Seq | Sigma time (ms) | SUF time (ms) | Speedup | Sigma comm (GB) | SUF comm (GB) | Sigma rounds | SUF rounds | Sigma LAN (s) | SUF LAN (s) | Sigma WAN (s) | SUF WAN (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 622.30 | 525.35 | 1.18x | 0.185 | 0.167 | 1080 | 1068 | 1.51 | 1.37 | 5.89 | 5.65 |
| 64 | 883.28 | 643.24 | 1.37x | 0.411 | 0.361 | 1104 | 1092 | 2.21 | 1.90 | 7.40 | 6.89 |
| 128 | 1408.73 | 1130.98 | 1.25x | 0.989 | 0.830 | 1128 | 1116 | 3.88 | 3.29 | 11.01 | 9.86 |

**Seq=256**: both Sigma and SUF failed with `cudaMemcpy` invalid argument (`gpu_mem.cu`). Logs: `/tmp/sigma_base_bert-base_256_p0.log`, `/tmp/suf_bert-base_256_p0.log`.

## 4. Kernel microbench (activation)
**Source**: `python3 scripts/compare_activation_fair.py` (seq=128, real comm on both Sigma/SUF).

**Repro flags**:
- Default rows (bert‑base/large, llama7b): run once with default env from the script:
  - `SIGMA_KEYBUF_MB=4096`, `SIGMA_MEMPOOL_DISABLE=0`, `SIGMA_SKIP_VERIFY=1`
  - `SUF_GELU_INTERVALS=256`, `SUF_SILU_INTERVALS=1024` (implicit defaults)
- gpt2 tuned row: median of 5 runs with:
  - `SIGMA_MEMPOOL_MB=1024 SUF_GELU_INTERVALS=512`
  - command: `python3 scripts/compare_activation_fair.py --models gpt2 --seq 128`

| Model / Gate | Sigma keygen (ms) | SUF keygen (ms) | Sigma eval (ms) | SUF eval (ms) | Eval speedup | Sigma key (bytes) | SUF key (bytes) | Sigma eval (bytes) | SUF eval (bytes) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bert‑base GELU | 13.610 | 0.131 | 24.126 | 0.994 | 24.26x | 226246736 | 36225052 | 13664260 | 10321924 |
| bert‑large GELU | 16.061 | 0.125 | 14.884 | 1.210 | 12.30x | 301662288 | 48300060 | 18219012 | 13762564 |
| gpt2 GELU | 13.716 | 0.179 | 12.028 | 1.113 | 10.80x | 226246736 | 42516508 | 13664260 | 10420228 |
| llama7b SiLU | 36.149 | 0.284 | 42.832 | 4.064 | 10.54x | 867078224 | 174895132 | 49668100 | 37691396 |

**Note**: Sigma GELU/SiLU tests were patched to report keygen time/size, eval comm bytes, and honor `SIGMA_KEYBUF_MB/GB` + `SIGMA_SKIP_VERIFY` to make logs comparable. The gpt2 row is the median of 5 runs with `SUF_GELU_INTERVALS=512` and `SIGMA_MEMPOOL_MB=1024` (intervals 64/128 crashed in SUF).

## 5. SHAFT baselines (local runs)
**Status**: not rerun in this pass; numbers below are carried over from the prior report for reference.
Projection formula uses the same LAN/WAN settings as Section 2.

### 5.1 Unit‑test microbench (SHAFT)
**Softmax** (`examples/unit-test/run_test_softmax.py`):

| L | Time (s) | Bytes (MB) | Rounds |
|---:|---:|---:|---:|
| 32 | 0.1827 | 0.0596 | 41 |
| 64 | 0.1700 | 0.1191 | 41 |
| 128 | 0.1648 | 0.2383 | 41 |
| 256 | 0.1412 | 0.4766 | 41 |

**GELU** (`examples/unit-test/run_test_gelu.py`):
- Max error: **0.0046**, Avg error: **0.000739**

| Shape | Time (s) | Bytes (MB) | Rounds |
|---|---:|---:|---:|
| (128, 3072) | 0.2494 | 354 | 19 |
| (128, 4096) | 0.2465 | 472 | 19 |

### 5.2 End‑to‑end transformer inference (SHAFT, seq=128)
| Model | Comp (s) | Comm (GB) | Rounds | LAN (s) | WAN (s) |
|---|---:|---:|---:|---:|---:|
| BERT‑base‑128 | 4.07 | 10.46 | 1496 | 27.28 | 66.21 |
| BERT‑large‑128 | 9.23 | 28.46 | 2936 | 71.82 | 173.77 |

**GPT‑2**: `examples/text-generation/test_gpt2_64_comp.sh` failed with device mismatch (`cuda:0` vs `cpu`) in `run_generation_private.py` (log: `/tmp/shaft_gpt2_64_comp.log`).

**Repro commands (cost‑estimated comm/rounds)**:
```
cd shaft/examples/text-classification
bash test_bert_base_128_comm.sh
bash test_bert_large_128_comm.sh
```

### 5.3 End‑to‑end (SHAFT, dual‑GPU, real comm, local loopback)
**Note**: Attempted `tc netem` LAN/WAN shaping on `lo`, but the container lacks permission (no `CAP_NET_ADMIN`). Results below are **without** LAN/WAN emulation.

| Model | Wall time (s) | Comm rounds | Comm bytes |
|---|---:|---:|---:|
| BERT‑base‑128 | 16.06 | 2 | 36896 |
| BERT‑large‑128 | 31.54 | 2 | 49184 |

Logs: `/tmp/shaft_bert_base_dual_real.log`, `/tmp/shaft_bert_large_dual_real.log`.

**Repro commands (real comm, dual‑GPU)**:
```
cd shaft/examples/text-classification
SHAFT_GPU0=0 SHAFT_GPU1=1 CUDA_VISIBLE_DEVICES=0,1 bash test_bert_base_128_comm.sh
SHAFT_GPU0=0 SHAFT_GPU1=1 CUDA_VISIBLE_DEVICES=0,1 bash test_bert_large_128_comm.sh
```

## 6. Unified comparison (Sigma / SUF / SHAFT)
SHAFT numbers below are from local runs in Section 5; Sigma/SUF are measured online time + comm.

| Model | Sigma online (ms) | SUF online (ms) | Sigma comm (GB) | SUF comm (GB) | SHAFT LAN (s) | SHAFT WAN (s) | SHAFT comm (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| BERT‑base‑128 | 1408.73 | 1130.98 | 0.989 | 0.830 | 27.28 | 66.21 | 10.46 |
| BERT‑large‑128 | 3917.54 | 2666.96 | 2.638 | 2.213 | 71.82 | 173.77 | 28.46 |
| GPT‑2‑128 | 1240.44 | 961.21 | 0.824 | 0.724 | N/A | N/A | N/A |
| GPT‑Neo‑128 | 5837.49 | 4827.64 | 4.029 | 3.648 | N/A | N/A | N/A |

**SHAFT missing models**:
- **GPT‑2**: `examples/text-generation/test_gpt2_64_comp.sh` failed with device mismatch (`cuda:0` vs `cpu`), log: `/tmp/shaft_gpt2_64_comp.log`.
- **GPT‑Neo / LLAMA / other models**: not run in SHAFT baseline set.

**Note**: SHAFT LAN/WAN values remain from cost‑estimated comm/rounds (Section 5.2). Real LAN/WAN emulation with `tc netem` could not be applied in this container (permission denied). For missing E2E models, SHAFT unit‑test microbench results are reported in Section 5.1 as supplementary evidence.

## 7. Accuracy
Accuracy experiments were **not rerun** in this pass. To reproduce Table‑4‑style accuracy, use `bench/accuracy_compare.py` with `bench/configs/accuracy_table4.json`.

## 8. Implementation notes / patches applied
- **Sigma activation tests** (`ezpc_upstream/GPU-MPC/tests/fss/gelu.cu`, `silu.cu`) patched to report keygen time/size, eval comm bytes, and honor `SIGMA_KEYBUF_MB/GB` + `SIGMA_SKIP_VERIFY`.
- **Repro scripts** added: `scripts/repro_eval_latest.py` (end‑to‑end Sigma/SUF tables) and `scripts/compare_activation_fair.py` (activation microbench).
- **Sigma/SUF** runs use `SIGMA_MEMPOOL_DISABLE=1` for end‑to‑end evaluation to avoid large async mempool pre‑allocations.
- **SHAFT ONNX export fixes** for Torch 2.11 remain from the prior report (SHAFT not rerun here).
