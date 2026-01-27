# Internal batch (>1) crash reproduction, fix, and SUF vs Sigma results (2026-01-27)

## Scope and goal

- Goal: make `SIGMA_BATCH > 1` run stably, then compare SUF vs Sigma under the same internal-batch setup for BERT-base and GPT-2.
- This document focuses on internal batch (one process loops `batch` times inside the binary), not true batched tensors.

## Key findings (bottom line first)

- bert-base: SUF is faster by **1.23x-1.33x**, reduces online comm by **16.1%**, and reduces key size by **24.3%**.
- gpt2: SUF is faster by **1.32x-1.44x**, reduces online comm by **12.1%**, and reduces key size by **22.3%**.

## Root cause of batch>1 crash and the fix

### Symptom

- With `SIGMA_BATCH=2` (or higher), runs aborted/segfaulted during MHA key reads (`readGPUDPFKey -> readGPUTrFloorKey -> SIGMA::mha`).

### Root cause

- `SIGMAKeygen::output` consumes output-mask bytes and advances `keyBuf`, but `SIGMA::output` did not advance `keyBuf`.
- In internal-batch mode, that misaligns the key stream after the first forward, so the second forward reads garbage headers and crashes.

### Fix applied

- Advance the key pointer in online output: `keyBuf += memSz;` where `memSz = N * sizeof(T)`.
- Applied in both upstream Sigma and SUF/vendor Sigma:
  - `ezpc_upstream/GPU-MPC/backend/sigma.h:294`
  - `third_party/EzPC_vendor/GPU-MPC/backend/sigma.h:289`

### Optional debug instrumentation (left in, gated by env var)

- Key-offset debug helpers (enabled by `SIGMA_DEBUG_KEYS=1`):
  - `ezpc_upstream/GPU-MPC/backend/sigma.h:74`
  - `ezpc_upstream/GPU-MPC/experiments/sigma/sigma.cu:252`
- DPF header guardrails for faster failure and better logs:
  - `ezpc_upstream/GPU-MPC/fss/gpu_dpf.h:104`

## Experiment configuration

### Hardware and runtime layout

- 2 GPUs, one party per GPU:
  - P0: `CUDA_VISIBLE_DEVICES=0`
  - P1: `CUDA_VISIBLE_DEVICES=1`
- Threads: `OMP_NUM_THREADS=32`
- Address: `127.0.0.1` (same host)

### Common environment variables

- `SIGMA_BATCH in {1,2,4,8}`
- `SIGMA_MEMPOOL_DISABLE=1`
- `SIGMA_PINNED_KEYBUF=1` (important; pinned off can severely degrade performance)

### SUF-specific environment variables

- `SUF_SOFTMAX=1`
- `SUF_LAYERNORM=1`
- `SUF_ACTIVATION=1`
- `SUF_NEXP_BITS=10`
- `SUF_INV_BITS=10`
- `SUF_RSQRT_BITS=9`

### Models and sequence length

- Models: `bert-base`, `gpt2`
- Sequence length: `seq=128`
- Batch sizes: `1, 2, 4, 8`

## Build commands used for the patched binaries

These commands rebuild only the `sigma` objects/binaries, without touching unrelated targets.

### Upstream Sigma binary

```bash
nvcc -std=c++17 -O3 -lineinfo -Xcompiler -fopenmp \
  -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120 \
  -I/workspace/SUF_new/ezpc_upstream/GPU-MPC \
  -I/workspace/SUF_new/ezpc_upstream/GPU-MPC/ext/sytorch/include \
  -I/workspace/SUF_new/ezpc_upstream/GPU-MPC/ext/sytorch/ext/llama/include \
  -I/workspace/SUF_new/ezpc_upstream/GPU-MPC/ext/sytorch/ext/cryptoTools \
  -I/workspace/SUF_new/ezpc_upstream/GPU-MPC/ext/sytorch/ext/cryptoTools/cryptoTools \
  -I/workspace/SUF_new/ezpc_upstream/GPU-MPC/ext/cutlass/include \
  -I/workspace/SUF_new/ezpc_upstream/GPU-MPC/ext/cutlass/tools/util/include \
  -c /workspace/SUF_new/ezpc_upstream/GPU-MPC/experiments/sigma/sigma.cu \
  -o /workspace/SUF_new/build/gpu_mpc_upstream/obj/sigma.o

objs=$(ls /workspace/SUF_new/build/gpu_mpc_upstream/obj/*.o | rg -v "/test_")
nvcc -std=c++17 -O3 -Xcompiler -fopenmp \
  -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120 \
  -o /workspace/SUF_new/build/gpu_mpc_upstream/sigma \
  $objs -lcurand -lcudart -lcuda -lstdc++fs -lpthread -ldl
```

### SUF/vendor Sigma binary

```bash
nvcc -std=c++17 -O3 -lineinfo -Xcompiler -fopenmp \
  -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120 \
  -I/workspace/SUF_new/include \
  -I/workspace/SUF_new/third_party/EzPC_vendor/GPU-MPC \
  -I/workspace/SUF_new/third_party/EzPC_vendor/GPU-MPC/ext/sytorch/include \
  -I/workspace/SUF_new/ezpc_upstream/GPU-MPC/ext/sytorch/include \
  -I/workspace/SUF_new/third_party/EzPC_vendor/GPU-MPC/ext/sytorch/ext/llama/include \
  -I/workspace/SUF_new/third_party/EzPC_vendor/GPU-MPC/ext/sytorch/ext/cryptoTools \
  -I/workspace/SUF_new/third_party/EzPC_vendor/GPU-MPC/ext/sytorch/ext/cryptoTools/cryptoTools \
  -I/workspace/SUF_new/third_party/EzPC_vendor/GPU-MPC/ext/cutlass/include \
  -I/workspace/SUF_new/third_party/EzPC_vendor/GPU-MPC/ext/cutlass/tools/util/include \
  -c /workspace/SUF_new/third_party/EzPC_vendor/GPU-MPC/experiments/sigma/sigma.cu \
  -o /workspace/SUF_new/build/gpu_mpc_vendor/obj/sigma.o

objs=$(ls /workspace/SUF_new/build/gpu_mpc_vendor/obj/*.o | rg -v "/test_")
nvcc -std=c++17 -O3 -Xcompiler -fopenmp \
  -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120 \
  -o /workspace/SUF_new/build/gpu_mpc_vendor/sigma \
  $objs -lcurand -lcudart -lcuda -lstdc++fs -lpthread -ldl
```

## Run commands (internal batch, pinned key buffer)

The following pattern was used for both Sigma and SUF. The only differences are the binary path and the SUF-specific env vars.

### Sigma (upstream)

```bash
cd /workspace/SUF_new/ezpc_upstream/GPU-MPC/experiments/sigma

# P0
SIGMA_BATCH=<B> SIGMA_MEMPOOL_DISABLE=1 SIGMA_PINNED_KEYBUF=1
OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=0
/workspace/SUF_new/build/gpu_mpc_upstream/sigma <model> 128 0 127.0.0.1 32

# P1
SIGMA_BATCH=<B> SIGMA_MEMPOOL_DISABLE=1 SIGMA_PINNED_KEYBUF=1
OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=1
/workspace/SUF_new/build/gpu_mpc_upstream/sigma <model> 128 1 127.0.0.1 32
```

### SUF (vendor)

```bash
cd /workspace/SUF_new/third_party/EzPC_vendor/GPU-MPC/experiments/sigma

# P0
SIGMA_BATCH=<B> SIGMA_MEMPOOL_DISABLE=1 SIGMA_PINNED_KEYBUF=1
SUF_SOFTMAX=1 SUF_LAYERNORM=1 SUF_ACTIVATION=1
SUF_NEXP_BITS=10 SUF_INV_BITS=10 SUF_RSQRT_BITS=9
OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=0
/workspace/SUF_new/build/gpu_mpc_vendor/sigma <model> 128 0 127.0.0.1 32

# P1
SIGMA_BATCH=<B> SIGMA_MEMPOOL_DISABLE=1 SIGMA_PINNED_KEYBUF=1
SUF_SOFTMAX=1 SUF_LAYERNORM=1 SUF_ACTIVATION=1
SUF_NEXP_BITS=10 SUF_INV_BITS=10 SUF_RSQRT_BITS=9
OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=1
/workspace/SUF_new/build/gpu_mpc_vendor/sigma <model> 128 1 127.0.0.1 32
```

## Results: SUF vs Sigma under internal batch

All numbers below are per inference (total divided by `batch`). Outputs were parsed from each run's `dealer.txt` and `evaluator.txt`.

### bert-base

| Batch | Sigma online (ms/inf) | SUF online (ms/inf) | Speedup | Sigma comm (GB/inf) | SUF comm (GB/inf) | Comm reduction | Sigma keygen (s/inf) | SUF keygen (s/inf) | Keygen speedup | Sigma key (GB/inf) | SUF key (GB/inf) | Key reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1392.7 | 1064.4 | 1.31x | 0.989 | 0.830 | 16.1% | 1.190 | 0.966 | 1.23x | 16.83 | 12.74 | 24.3% |
| 2 | 1421.1 | 1151.4 | 1.23x | 0.989 | 0.830 | 16.1% | 1.192 | 1.008 | 1.18x | 16.83 | 12.74 | 24.3% |
| 4 | 1378.3 | 1039.0 | 1.33x | 0.989 | 0.830 | 16.1% | 1.191 | 0.918 | 1.30x | 16.83 | 12.74 | 24.3% |
| 8 | 1370.2 | 1041.9 | 1.32x | 0.989 | 0.830 | 16.1% | 1.164 | 0.913 | 1.28x | 16.83 | 12.74 | 24.3% |

- Sigma output dir example: `/workspace/SUF_new/ezpc_upstream/GPU-MPC/experiments/sigma/output/P0/models/bert-base-128`
- SUF output dir example: `/workspace/SUF_new/third_party/EzPC_vendor/GPU-MPC/experiments/sigma/output/P0/models/bert-base-128`

### gpt2

| Batch | Sigma online (ms/inf) | SUF online (ms/inf) | Speedup | Sigma comm (GB/inf) | SUF comm (GB/inf) | Comm reduction | Sigma keygen (s/inf) | SUF keygen (s/inf) | Keygen speedup | Sigma key (GB/inf) | SUF key (GB/inf) | Key reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1266.2 | 931.1 | 1.36x | 0.824 | 0.724 | 12.1% | 1.072 | 0.875 | 1.23x | 14.29 | 11.10 | 22.3% |
| 2 | 1270.1 | 961.6 | 1.32x | 0.824 | 0.724 | 12.1% | 1.072 | 0.848 | 1.26x | 14.29 | 11.10 | 22.3% |
| 4 | 1378.6 | 956.6 | 1.44x | 0.824 | 0.724 | 12.1% | 1.045 | 0.848 | 1.23x | 14.29 | 11.10 | 22.3% |
| 8 | 1288.4 | 973.2 | 1.32x | 0.824 | 0.724 | 12.1% | 1.086 | 0.849 | 1.28x | 14.29 | 11.10 | 22.3% |

- Sigma output dir example: `/workspace/SUF_new/ezpc_upstream/GPU-MPC/experiments/sigma/output/P0/models/gpt2-128`
- SUF output dir example: `/workspace/SUF_new/third_party/EzPC_vendor/GPU-MPC/experiments/sigma/output/P0/models/gpt2-128`

## Interpretation notes

- SUF remains strictly better than Sigma across all tested batches for both models under this internal-batch setup.
- Internal batch in this codebase is still a loop over `batch` forwards, not a true batched tensor dimension. This fix makes that loop correct and stable.
- The communication and key-size reductions are stable across batch sizes because they are largely per-layer structural differences (SUF vs Sigma), not batch-dependent effects.

## Artifacts

- Primary machine-readable results:
  - `/workspace/SUF_new/batch_scaling_internal_pinnedon_manual_2026-01-27.json`
- This markdown report:
  - `/workspace/SUF_new/batch_scaling_internal_pinnedon_2026-01-27.md`
