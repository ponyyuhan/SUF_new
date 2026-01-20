# Evaluation Results (2026-01-20)

## 1. Environment
- **OS**: Ubuntu 24.04.3 LTS
- **CPU**: AMD EPYC 9374F 32-Core Processor (1 socket, 64 threads)
- **RAM**: 314 GiB (swap 8.0 GiB)
- **GPU**: 2× NVIDIA GeForce RTX 5090 (32,607 MiB each, sm_120)
- **Storage**: 2.0 TB total, 1.3 TB free on `/`
- **CUDA**: 13.0 (nvcc 13.0.88)
- **Sigma/SUF build**: `CUDA_VERSION=13.0`, `GPU_ARCH=120`, `SIGMA_MEMPOOL_DISABLE=1`
- **SHAFT venv**: `/workspace/SUF_new/shaft/.venv`
  - **PyTorch**: 2.11.0.dev20260119+cu128 (CUDA 12.8)
  - **CrypTen**: 1.0.0
  - **Transformers**: 4.45.0
- **Note**: Llama‑13B is not evaluated (per request).

## 2. SUF vs Sigma (end‑to‑end, 2 GPUs)
**SUF settings** (both parties):
```
SUF_SOFTMAX=1 SUF_LAYERNORM=1 SUF_ACTIVATION=1
SUF_NEXP_BITS=10 SUF_INV_BITS=10 SUF_RSQRT_BITS=9
CUDA_VISIBLE_DEVICES=0/1, CPU threads=32
```
**Sigma baseline**: built from `ezpc_upstream/GPU-MPC`.

| Model | Sigma online (ms) | SUF online (ms) | Speedup | Sigma comm (GB) | SUF comm (GB) | Sigma keygen (s) | SUF keygen (s) | Sigma key (GB) | SUF key (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BERT‑tiny‑128 | 154.00 | 34.33 | 4.49x | 0.020 | 0.017 | 0.29 | 0.04 | 0.326 | 0.250 |
| BERT‑base‑128 | 1631.10 | 928.72 | 1.76x | 0.989 | 0.830 | 2.97 | 0.54 | 16.835 | 12.739 |
| BERT‑large‑128 | 4960.80 | 2405.52 | 2.06x | 2.638 | 2.213 | 7.63 | 1.61 | 45.448 | 34.529 |
| GPT‑2‑128 | 1679.97 | 804.87 | 2.09x | 0.824 | 0.724 | 3.36 | 0.50 | 14.292 | 11.101 |
| GPT‑Neo‑128 | 6053.52 | 4271.14 | 1.42x | 4.029 | 3.648 | 7.71 | 2.88 | 76.187 | 61.215 |

**Observation**: SUF is faster and communicates less on all tested models while also shrinking key size and keygen time.

**Not run (resource/operational constraints)**:
- **GPT‑Neo‑large**: previous attempt overloaded the server; per request, we did not rerun that command.
- **Llama‑7B / Llama‑13B**: Sigma key buffers are hard‑coded to **300 GB** and **450 GB** per party, requiring ~600 GB / 900 GB total; with 314 GiB total RAM (~227 GiB available during runs), these exceed memory and are not feasible in this environment.

### 2.1 Additional sequence points (GPT‑2 / GPT‑Neo)
| Model | Seq | Sigma time (ms) | SUF time (ms) | Speedup | Sigma comm (GB) | SUF comm (GB) |
|---|---:|---:|---:|---:|---:|---:|
| GPT‑2 | 64 | 621.83 | 474.10 | 1.31x | 0.370 | 0.335 |
| GPT‑2 | 128 | 1679.97 | 804.87 | 2.09x | 0.824 | 0.724 |
| GPT‑2 | 256 | 2494.41 | 1573.46 | 1.59x | 1.983 | 1.663 |
| GPT‑Neo | 64 | 3320.82 | 2726.24 | 1.22x | 1.900 | 1.750 |
| GPT‑Neo | 128 | 6053.52 | 4271.14 | 1.42x | 4.029 | 3.648 |

### 2.2 SHAFT comparison (per 2025‑2287‑paper.pdf)
SHAFT computes reported runtime as:
```
T = comp_time + 2 * comm_bytes / bandwidth + rounds * latency
```
Network settings in the paper: **LAN (1 GB/s, 0.5 ms)**, **WAN (400 MB/s, 4 ms)**.

| Model | SHAFT comp (s) | Comm (GB) | Rounds | Projected LAN (s) | Projected WAN (s) |
|---|---:|---:|---:|---:|---:|
| BERT‑base‑128 | 2.82 | 10.46 | 1496 | 24.49 | 61.10 |
| BERT‑large‑128 | 7.28 | 28.46 | 2936 | 65.67 | 161.32 |

**Notes**:
- SHAFT logs provide comp time, comm bytes, and rounds; projected times above follow the paper’s formula/network settings.
- Sigma/SUF logs do not expose round counts, so we report measured online time and comm only; direct projected‑time comparison is not possible.

### 2.3 Unified comparison (Sigma / SUF / SHAFT)
SHAFT numbers below are taken from Table VII in the SHAFT paper (2025‑2287‑paper.pdf). Times are **projected** using their LAN/WAN settings (LAN 1 GB/s, 0.5 ms; WAN 400 MB/s, 4 ms); comm is in GB.

| Model | Sigma online (ms) | SUF online (ms) | Sigma comm (GB) | SUF comm (GB) | SHAFT LAN (s) | SHAFT WAN (s) | SHAFT comm (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| BERT‑base‑128 | 1631.10 | 928.72 | 0.989 | 0.830 | 28.60 | 66.46 | 10.46 |
| BERT‑large‑128 | 4960.80 | 2405.52 | 2.638 | 2.213 | 77.13 | 176.21 | 28.46 |
| GPT‑2‑128 | 1679.97 | 804.87 | 0.824 | 0.724 | 32.59 | 73.47 | 11.27 |
| GPT‑2‑64 | 621.83 | 474.10 | 0.370 | 0.335 | 19.32 | 42.37 | 5.76 |
| ViT‑base | — | — | — | — | 45.66 | 108.24 | 18.41 |

**Note**: SHAFT rows are from the paper’s Table VII; our local SHAFT GPT‑2 run failed, so we do not report local GPT‑2 measurements.

## 3. Scaling (BERT‑base, seq length sweep)
**Setup**: same as Section 2.

| Seq | Sigma time (ms) | SUF time (ms) | Speedup | Sigma comm (GB) | SUF comm (GB) |
|---:|---:|---:|---:|---:|---:|
| 32 | 1137.27 | 337.71 | 3.37x | 0.185 | 0.167 |
| 64 | 1126.43 | 496.39 | 2.27x | 0.411 | 0.361 |
| 128 | 1631.10 | 928.72 | 1.76x | 0.989 | 0.830 |

**Seq=256**: both Sigma and SUF fail with `cudaMemcpy` invalid argument (`gpu_mem.cu`), logs in `/tmp/sigma_up_bert_base_256_*.log` and `/tmp/suf_bert_base_256_*.log`.

## 4. Kernel microbench (SUF vs Sigma, activation)
**Source**: `python3 scripts/compare_activation.py`.

| Model / Gate | Sigma per‑gate eval (ms) | SUF per‑gate eval (ms) | Speedup | SUF per‑gate key (bytes) |
|---|---:|---:|---:|---:|
| BERT‑base GELU | 14.933 | 0.553 | 27.0x | 2464 |
| BERT‑large GELU | 17.869 | 0.721 | 24.8x | 2464 |
| GPT‑2 GELU | 16.021 | 0.555 | 28.9x | 2464 |
| Llama‑7B SiLU | 50.638 | 2.242 | 22.6x | 8672 |

**Note**: Sigma test binaries do not report key/comm bytes; those fields are 0 in the raw logs.

## 5. SHAFT baselines
All SHAFT runs use `CUDA_VISIBLE_DEVICES=0` (both parties share GPU0 because the launcher does not bind GPUs by rank).
The SHAFT paper (2025‑2287) reports runtime using:
```
T = comp_time + 2 * comm_bytes / bandwidth + rounds * latency
```
with LAN **(1 GB/s, 0.5 ms)** and WAN **(400 MB/s, 4 ms)**.

### 5.1 Unit‑test microbench
**Softmax** (`examples/unit-test/run_test_softmax.py`):
| L | Time (s) | Bytes (MB) | Rounds |
|---:|---:|---:|---:|
| 32 | 0.0837 | 0.0596 | 41 |
| 64 | 0.0743 | 0.1191 | 41 |
| 128 | 0.0727 | 0.2383 | 41 |
| 256 | 0.0710 | 0.4766 | 41 |

**GELU** (`examples/unit-test/run_test_gelu.py`):
- Max error: **0.0045**, Avg error: **0.000739**

| Shape | Time (s) | Bytes (MB) | Rounds |
|---|---:|---:|---:|
| (128, 3072) | 0.2642 | 354 | 19 |
| (128, 4096) | 0.2774 | 472 | 19 |

### 5.2 End‑to‑end transformer inference (SHAFT)
**BERT‑base‑128 (QNLI)**:
- **Compute time**: 2.82s (`/tmp/shaft_bert_base_128_comp.log`)
- **Comm**: 10.46 GB, rounds 1496 (`/tmp/shaft_bert_base_128_comm.log`)

**BERT‑large‑128 (QNLI)**:
- **Compute time**: 7.28s (`/tmp/shaft_bert_large_128_comp.log`)
- **Comm**: 28.46 GB, rounds 2936 (`/tmp/shaft_bert_large_128_comm.log`)

**GPT‑2‑128**:
- **Failed** in `run_generation_private.py` with CrypTen operator issue (`Less.forward() missing 1 required positional argument: 'y'`).
- Log: `/tmp/shaft_gpt2_128_comp.log`.

## 6. Accuracy
- `accuracy_sweep/bert_tiny_accuracy.csv`: MAE / RMSE / MaxAbs = 0 (matches Sigma baseline in this sweep).

## 7. Implementation notes / patches applied
- Added `SIGMA_MEMPOOL_DISABLE` guard in `ezpc_upstream/GPU-MPC/utils/gpu_mem.cu` and `third_party/EzPC_vendor/GPU-MPC/utils/gpu_mem.cu` to avoid large mempool pre‑alloc.
- Updated SUF/Sigma softmax + layernorm key parsing to skip SUF‑specific nExp/inv/rsqrt bytes when `SUF_HAVE_CUDA` is enabled.
- Extended `scripts/compare_activation.py` with explicit GPU selection and more robust Sigma log parsing.
- **Note**: `third_party/EzPC/GPU-MPC` contains local edits and is not identical to the vendored `EzPC_vendor` copy; all evaluation runs here use `EzPC_vendor` (SUF) and `ezpc_upstream` (Sigma baseline).
- Removed failed Llama‑13B key artifacts in `third_party/EzPC/GPU-MPC/experiments/sigma/keys/llama13b-suf-nexp9-inv9-rsqrt8` (≈335 GB) after confirming they were crash leftovers.

## 8. Remaining gaps vs Evaluation.md
- **Block‑level bench** (`bench_softmax_norm`) and **extensibility case study** are not present in this repo; not executed.
- **Batch‑size sweep** is not supported by the current Sigma harness CLI.
