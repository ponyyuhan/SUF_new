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
| BERT‑base‑128 | 1631.10 | 928.72 | 1.76x | 0.989 | 0.830 | 2.97 | 0.54 | 16.835 | 12.739 |
| BERT‑large‑128 | 4960.80 | 2405.52 | 2.06x | 2.638 | 2.213 | 7.63 | 1.61 | 45.448 | 34.529 |
| GPT‑2‑128 | 1679.97 | 804.87 | 2.09x | 0.824 | 0.724 | 3.36 | 0.50 | 14.292 | 11.101 |

**Observation**: SUF is faster and communicates less on all three models while also shrinking key size and keygen time.

### 2.1 SUF vs SHAFT (overlapping models)
SHAFT reports **compute time only** (communication is reported separately), so the comparison below is conservative for SHAFT.

| Model | SUF online (s) | SHAFT compute (s) | Speedup | SUF comm (GB) | SHAFT comm (GB) | Comm reduction |
|---|---:|---:|---:|---:|---:|---:|
| BERT‑base‑128 | 0.929 | 2.82 | 3.04x | 0.830 | 10.46 | 12.6x |
| BERT‑large‑128 | 2.406 | 7.28 | 3.03x | 2.213 | 28.46 | 12.9x |

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

## 8. Remaining gaps vs Evaluation.md
- **Block‑level bench** (`bench_softmax_norm`) and **extensibility case study** are not present in this repo; not executed.
- **Batch‑size sweep** is not supported by the current Sigma harness CLI.
