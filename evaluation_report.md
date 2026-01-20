# Evaluation Results (2026-01-20)

## 1. Environment
- **OS**: Ubuntu 24.04.3 LTS
- **CPU**: AMD EPYC 9374F 32-Core Processor ×2 (128 logical CPUs)
- **GPU**: 2× NVIDIA RTX PRO 6000 Blackwell Workstation Edition (sm_120)
- **CUDA**: 13.0 (nvcc 13.0.88)
- **Sigma/SUF build**: `CUDA_VERSION=13.0`, `GPU_ARCH=120`
- **SHAFT venv**: `/workspace/SUF_new/shaft/.venv`
  - **PyTorch**: 2.11.0.dev20260119+cu128 (CUDA 12.8)
  - **CrypTen**: 1.0.0
  - **Transformers**: 4.45.0

## 2. SUF vs Sigma (end-to-end, 2 GPUs)
**SUF settings** (both parties):
```
SUF_SOFTMAX=1 SUF_LAYERNORM=1 SUF_ACTIVATION=1
SUF_NEXP_BITS=10 SUF_INV_BITS=10 SUF_RSQRT_BITS=9
CUDA_VISIBLE_DEVICES=0/1, CPU threads=32
```
**Sigma baseline**: built from upstream `ezpc_upstream/GPU-MPC` (no SUF modifications). 

| Model | Sigma online (ms) | SUF online (ms) | Speedup | Sigma comm (GB) | SUF comm (GB) | Sigma keygen (s) | SUF keygen (s) | Sigma key (GB) | SUF key (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BERT-base-128 | 1017.17 | 722.54 | 1.41x | 1.062 | 0.891 | 7.03 | 0.48 | 18.076 | 13.678 |
| BERT-large-128 | 2785.84 | 1805.22 | 1.54x | 2.833 | 2.376 | 13.48 | 1.21 | 48.800 | 37.076 |
| GPT2-128 | 801.38 | 699.68 | 1.15x | 0.885 | 0.778 | 7.15 | 0.44 | 15.346 | 11.920 |

**Observation**: SUF is faster and communicates less on all three models while also shrinking key size and keygen time.

## 3. Scaling (BERT-base, seq length sweep)
**Setup**: Same as Section 2. 

| Seq | Sigma time (ms) | SUF time (ms) | Speedup | Sigma comm (GB) | SUF comm (GB) |
|---:|---:|---:|---:|---:|---:|
| 32 | 391.24 | 298.62 | 1.31x | 0.198 | 0.180 |
| 64 | 579.78 | 433.68 | 1.34x | 0.442 | 0.388 |
| 128 | 1017.17 | 722.54 | 1.41x | 1.062 | 0.891 |

**Seq=256**: both Sigma and SUF runs failed with `cudaMemcpy` invalid argument (see `/tmp/*_bert_base_256_*.log`).

## 4. SHAFT baselines
### 4.1 Unit-test microbench
**Softmax** (`examples/unit-test/run_test_softmax.py`):
| L | Time (s) | Bytes (MB) | Rounds |
|---:|---:|---:|---:|
| 32 | 0.0940 | 0.0596 | 41 |
| 64 | 0.1092 | 0.1191 | 41 |
| 128 | 0.1101 | 0.2383 | 41 |
| 256 | 0.1099 | 0.4766 | 41 |

**GELU** (`examples/unit-test/run_test_gelu.py`):
- Max error: **0.3010**, Avg error: **0.000775**

| Shape | Time (s) | Bytes (MB) | Rounds |
|---|---:|---:|---:|
| (128, 3072) | 0.2622 | 354 | 19 |
| (128, 4096) | 0.3286 | 472 | 19 |

### 4.2 End-to-end transformer inference (SHAFT)
**BERT-base-128 (QNLI)**:
- **Compute time**: 2.79s (`/tmp/shaft_bert_base_128_comp.log`)
- **Comm**: 10.46 GB, rounds 1496 (`/tmp/shaft_bert_base_128_comm.log`)

**BERT-large-128 (QNLI)**:
- **Compute time**: 6.80s (`/tmp/shaft_bert_large_128_comp.log`)
- **Comm**: 28.46 GB, rounds 2936 (`/tmp/shaft_bert_large_128_comm.log`)

**GPT-2-128**:
- **Failed** in `run_generation_private.py` with CrypTen operator issues (e.g., `Less.forward() missing 1 required positional argument: 'y'`). 
- Logs: `/tmp/shaft_gpt2_128_comp.log` (failure during private model forward).

## 5. Accuracy
- SUF vs Sigma accuracy sweep (bert-tiny): `accuracy_sweep/bert_tiny_accuracy.csv` shows **MAE/RMSE/MaxAbs = 0** for the tested LUT bitwidths (nExp/inv/rsqrt sweep). 

## 6. Implementation notes / patches applied
- **Upstream Sigma build fixes** (GCC 13 + Ubuntu 24.04):
  - `SEAL/native/src/seal/util/locks.h`: added `#include <mutex>`
  - `ext/sytorch/ext/sci/src/cleartext_library_float.cpp`: added `#include <cstdint>`
- **SHAFT on Blackwell (sm_120)**:
  - Required **nightly PyTorch (cu128)** for sm_120 support.
  - CrypTen patches for torch nightly compatibility:
    - ONNX converter import fallback (handle missing registry modules)
    - ONNX export: `dynamo=False`, dynamic input names for multi-input models
    - `Add` module: device alignment for plaintext tensors
  - Added dependency: `onnxscript` for torch.onnx exporter.

## 7. Outstanding gaps vs Evaluation.md
- **Block-level bench** (`bench_softmax_norm`) and **extensibility case study** are not present in this repo; not executed.
- **Seq=256** scaling failed for both Sigma and SUF due to CUDA memcpy invalid argument.
- **SHAFT GPT-2** e2e failed due to CrypTen op incompatibility in current environment.

