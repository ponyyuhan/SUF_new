# Evaluation Results (2026-01-23)

## 1. Environment
- **OS**: Ubuntu 24.04.3 LTS
- **CPU**: 2× AMD EPYC 7642 48-Core Processor (96 cores / 192 threads)
- **RAM**: 1.0 TiB (swap 8.0 GiB)
- **GPU**: 2× NVIDIA RTX PRO 6000 Blackwell Workstation Edition (97,887 MiB each, sm_120)
- **Storage**: 200 GB total, 189 GB free on `/`
- **CUDA**: 13.0 (nvcc 13.0.88), driver 580.105.08
- **Sigma baseline**: `ezpc_upstream/GPU-MPC` (`SIGMA_MEMPOOL_DISABLE=1`, `OMP_NUM_THREADS=32`)
- **SUF**: `third_party/EzPC_vendor/GPU-MPC` (SUF bridge enabled)
- **SUF settings** (both parties):
  ```
  SUF_SOFTMAX=1 SUF_LAYERNORM=1 SUF_ACTIVATION=1
  SUF_NEXP_BITS=10 SUF_INV_BITS=10 SUF_RSQRT_BITS=9
  ```
- **SHAFT venv**: `/workspace/new/SUF_new/shaft/.venv`
  - **PyTorch**: 2.11.0.dev20260122+cu128
  - **Transformers**: 4.45.0
  - **ONNX**: 1.20.1, **onnxscript**: 0.5.7

**GPU binding**:
- Sigma/SUF: party‑0 uses GPU0, party‑1 uses GPU1.
- SHAFT (dual‑GPU runs): `SHAFT_GPU0=0`, `SHAFT_GPU1=1` (launcher binds GPUs by rank). Earlier single‑GPU baselines used `CUDA_VISIBLE_DEVICES=0`.

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
| BERT‑tiny‑128 | 69.84 | 55.20 | 1.27x | 0.020 | 0.017 | 188 | 186 | 0.19 | 0.17 | 0.91 | 0.88 |
| BERT‑base‑128 | 1682.04 | 1313.86 | 1.28x | 0.989 | 0.830 | 1128 | 1116 | 4.11 | 3.41 | 11.25 | 9.99 |
| BERT‑large‑128 | 4311.51 | 3125.36 | 1.38x | 2.638 | 2.213 | 2256 | 2232 | 10.42 | 8.34 | 26.81 | 23.28 |
| GPT‑2‑128 | 1513.57 | 1073.32 | 1.41x | 0.824 | 0.724 | 1128 | 1116 | 3.55 | 2.95 | 10.15 | 9.19 |
| GPT‑Neo‑128 | 7078.19 | 5414.84 | 1.31x | 4.029 | 3.648 | 2256 | 2232 | 15.42 | 13.40 | 36.30 | 32.96 |

### 3.1 Keygen and key size
| Model | Sigma keygen (s) | SUF keygen (s) | Sigma key (GB) | SUF key (GB) |
|---|---:|---:|---:|---:|
| BERT‑tiny‑128 | 0.08 | 0.06 | 0.326 | 0.250 |
| BERT‑base‑128 | 1.28 | 0.99 | 16.835 | 12.739 |
| BERT‑large‑128 | 3.21 | 2.18 | 45.448 | 34.529 |
| GPT‑2‑128 | 1.14 | 0.88 | 14.292 | 11.101 |
| GPT‑Neo‑128 | 4.95 | 4.04 | 76.187 | 61.215 |

### 3.2 Additional sequence points (GPT‑2 / GPT‑Neo)
| Model | Seq | Sigma time (ms) | SUF time (ms) | Speedup | Sigma comm (GB) | SUF comm (GB) | Sigma rounds | SUF rounds | Sigma LAN (s) | SUF LAN (s) | Sigma WAN (s) | SUF WAN (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT‑2 | 64 | 823.29 | 573.03 | 1.44x | 0.370 | 0.335 | 1104 | 1092 | 2.04 | 1.73 | 7.10 | 6.64 |
| GPT‑2 | 128 | 1513.57 | 1073.32 | 1.41x | 0.824 | 0.724 | 1128 | 1116 | 3.55 | 2.95 | 10.15 | 9.19 |
| GPT‑2 | 256 | 3171.60 | 2165.38 | 1.46x | 1.983 | 1.663 | 1152 | 1140 | 7.51 | 5.84 | 17.92 | 15.19 |
| GPT‑Neo | 64 | 4232.44 | 3457.93 | 1.22x | 1.900 | 1.750 | 2208 | 2184 | 8.89 | 7.74 | 22.74 | 21.02 |
| GPT‑Neo | 128 | 7078.19 | 5414.84 | 1.31x | 4.029 | 3.648 | 2256 | 2232 | 15.42 | 13.40 | 36.30 | 32.96 |

### 3.3 Scaling (BERT‑base seq sweep)
| Seq | Sigma time (ms) | SUF time (ms) | Speedup | Sigma comm (GB) | SUF comm (GB) | Sigma rounds | SUF rounds | Sigma LAN (s) | SUF LAN (s) | Sigma WAN (s) | SUF WAN (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 619.98 | 464.27 | 1.34x | 0.185 | 0.167 | 1080 | 1068 | 1.42 | 1.27 | 5.80 | 5.55 |
| 64 | 903.87 | 684.70 | 1.32x | 0.411 | 0.361 | 1104 | 1092 | 2.18 | 1.88 | 7.37 | 6.86 |
| 128 | 1682.04 | 1313.86 | 1.28x | 0.989 | 0.830 | 1128 | 1116 | 4.11 | 3.41 | 11.25 | 9.99 |

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
| bert‑base GELU | 19.095 | 2.767 | 18.153 | 15.066 | 1.20x | 226246736 | 36225052 | 13664260 | 10321924 |
| bert‑large GELU | 23.689 | 3.660 | 23.948 | 15.365 | 1.56x | 301662288 | 48300060 | 18219012 | 13762564 |
| gpt2 GELU | 19.040 | 3.495 | 18.644 | 18.035 | 1.03x | 226246736 | 42516508 | 13664260 | 10420228 |
| llama7b SiLU | 43.543 | 8.293 | 53.147 | 27.240 | 1.95x | 867078224 | 174895132 | 49668100 | 37691396 |

**Note**: Sigma GELU/SiLU tests were patched to report keygen time/size, eval comm bytes, and honor `SIGMA_KEYBUF_MB/GB` + `SIGMA_SKIP_VERIFY` to make logs comparable. The gpt2 row is the median of 5 runs with `SUF_GELU_INTERVALS=512` and `SIGMA_MEMPOOL_MB=1024` (intervals 64/128 crashed in SUF).

## 5. SHAFT baselines (local runs)
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
| BERT‑base‑128 | 1682.04 | 1313.86 | 0.989 | 0.830 | 27.28 | 66.21 | 10.46 |
| BERT‑large‑128 | 4311.51 | 3125.36 | 2.638 | 2.213 | 71.82 | 173.77 | 28.46 |
| GPT‑2‑128 | 1513.57 | 1073.32 | 0.824 | 0.724 | N/A | N/A | N/A |
| GPT‑Neo‑128 | 7078.19 | 5414.84 | 4.029 | 3.648 | N/A | N/A | N/A |

**SHAFT missing models**:
- **GPT‑2**: `examples/text-generation/test_gpt2_64_comp.sh` failed with device mismatch (`cuda:0` vs `cpu`), log: `/tmp/shaft_gpt2_64_comp.log`.
- **GPT‑Neo / LLAMA / other models**: not run in SHAFT baseline set.

**Note**: SHAFT LAN/WAN values remain from cost‑estimated comm/rounds (Section 5.2). Real LAN/WAN emulation with `tc netem` could not be applied in this container (permission denied). For missing E2E models, SHAFT unit‑test microbench results are reported in Section 5.1 as supplementary evidence.
| GPT‑2‑128 | 1513.57 | 1073.32 | 0.824 | 0.724 | — | — | — |

## 7. Accuracy
Accuracy experiments were **not rerun** in this pass. To reproduce Table‑4‑style accuracy, use `bench/accuracy_compare.py` with `bench/configs/accuracy_table4.json`.

## 8. Implementation notes / patches applied
- **SHAFT ONNX export fixes** for Torch 2.11: capture missing exception in `crypten/nn/__init__.py`, relax symbolic‑registry import in `crypten/nn/onnx_converter.py`, generate per‑input `dynamic_axes`, and force `dynamo=False` to enable custom `shaft::Embedding`/`shaft::GELU` export.
- **Sigma/SUF** runs use `SIGMA_MEMPOOL_DISABLE=1` to avoid large async mempool pre‑allocations.
