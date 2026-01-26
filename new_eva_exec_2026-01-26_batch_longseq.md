# new_eva.md Executed Experiments (2026-01-26, batch + long-seq update)

## Summary
- Fixed batch scaling by **not disabling pinned key buffers** (`SIGMA_PINNED_KEYBUF` left at default). This removed the SUF transfer‑time inflation and makes **SUF faster than Sigma for all batches**.
- **D' Batch scaling**: end‑to‑end Sigma vs SUF at seq=128 for **BERT‑base** and **GPT‑2**, batch ∈ {1,2,4,8}. Method = serial batching (run batch times with SIGMA_BATCH=1, aggregate totals). Plots + tables generated.
- **D'' Longer sequence**: decoder end‑to‑end GPT‑2 seq=512 already completed (see below). Encoder end‑to‑end BERT‑base seq=256 still fails (segfault). Block‑level long‑seq results provided via activation microbench + Sigma softmax.

## Environment
- **OS**: Ubuntu 24.04.3 LTS
- **CPU**: 2× AMD EPYC 9654 (192 cores / 384 threads)
- **RAM**: 1.5 TiB (swap 0)
- **GPU**: 2× NVIDIA RTX PRO 6000 Blackwell (97,887 MiB each, sm_120)
- **CUDA**: 13.0 (nvcc 13.0.88), driver 580.119.02

## Binaries
- **Sigma (baseline)**: `/workspace/SUF_new/build/gpu_mpc_upstream/sigma`
- **SUF (bridge)**: `/workspace/SUF_new/build/gpu_mpc_vendor/sigma`
- **Activation microbench**: `/workspace/SUF_new/build/bench_suf_model`, `/workspace/SUF_new/build/gpu_mpc_upstream/gelu`
- **Softmax tests**: `/workspace/SUF_new/build/gpu_mpc_upstream/softmax`, `/workspace/SUF_new/build/gpu_mpc_vendor/softmax`

## Global config (batch scaling)
- `OMP_NUM_THREADS=32`
- `SIGMA_MEMPOOL_DISABLE=1`
- **Pinned key buffer**: default (no `SIGMA_PINNED_KEYBUF` override)

## SUF config (batch scaling)
```
SUF_SOFTMAX=1 SUF_LAYERNORM=1 SUF_ACTIVATION=1
SUF_NEXP_BITS=10 SUF_INV_BITS=10 SUF_RSQRT_BITS=9
```

---

# D' Batch Scaling (End‑to‑End, serial batching)

**Why serial batching**: internal SIGMA_BATCH>1 crashed in keygen/eval (std::bad_array_new_length / segfault). To keep correct per‑inference semantics, batch size is implemented as **repeat‑runs** and then aggregated.

**Command**:
```
./scripts/run_batch_scaling.py \
  --models bert-base,gpt2 --seq 128 --batches 1,2,4,8 --threads 32 \
  --addr 127.0.0.1 --gpu0 0 --gpu1 1 \
  --out /workspace/SUF_new/batch_scaling_bert_pinned.json

./scripts/run_batch_scaling.py \
  --models gpt2 --seq 128 --batches 1,2,4,8 --threads 32 \
  --addr 127.0.0.1 --gpu0 0 --gpu1 1 \
  --out /workspace/SUF_new/batch_scaling_gpt2_pinned.json
```

**Tables**: `/workspace/SUF_new/batch_scaling_tables.md`

**Plots** (SVG fallback due to missing matplotlib):
- Throughput: `/workspace/SUF_new/batch_plots/batch_throughput.svg`
- Latency: `/workspace/SUF_new/batch_plots/batch_latency.svg`

**Results (per‑inference)**

### bert-base (seq=128)
| Batch | Sigma ms/inf | SUF ms/inf | Speedup | Sigma tokens/s | SUF tokens/s | Sigma comm (GB/inf) | SUF comm (GB/inf) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1372.2 | 1035.7 | 1.32x | 93.3 | 123.6 | 0.989 | 0.830 |
| 2 | 1314.9 | 1026.1 | 1.28x | 97.3 | 124.7 | 0.989 | 0.830 |
| 4 | 1356.3 | 1047.3 | 1.30x | 94.4 | 122.2 | 0.989 | 0.830 |
| 8 | 1373.7 | 1070.2 | 1.28x | 93.2 | 119.6 | 0.989 | 0.830 |

### gpt2 (seq=128)
| Batch | Sigma ms/inf | SUF ms/inf | Speedup | Sigma tokens/s | SUF tokens/s | Sigma comm (GB/inf) | SUF comm (GB/inf) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1313.9 | 910.8 | 1.44x | 97.4 | 140.5 | 0.824 | 0.724 |
| 2 | 1174.3 | 917.3 | 1.28x | 109.0 | 139.5 | 0.824 | 0.724 |
| 4 | 1200.1 | 928.3 | 1.29x | 106.7 | 137.9 | 0.824 | 0.724 |
| 8 | 1232.7 | 923.9 | 1.33x | 103.8 | 138.5 | 0.824 | 0.724 |

**Key size per inference (batch=1)**:
- **BERT‑base**: Sigma 16.835 GB, SUF 12.739 GB
- **GPT‑2**: Sigma 14.292 GB, SUF 11.101 GB

**Note (fix applied)**: earlier runs had `SIGMA_PINNED_KEYBUF=0`, which inflated SUF transfer time. Leaving pinned keybuf at default restores SUF’s advantage across all batches.

---

# D'' Longer Sequence Scaling

## D''‑A. Decoder end‑to‑end (GPT‑2 seq=512)
(Repeated here from `new_eva_exec_2026-01-26.md` for completeness.)

**Key config**: `SIGMA_PINNED_KEYBUF=0` due to extremely large key buffer at seq=512.

| Variant | Keygen (s) | Key size (GB) | Online time (ms) | Comm time (ms) | Comm (GB) | Throughput (tokens/s) |
|---|---:|---:|---:|---:|---:|---:|
| Sigma | 43.37 | 86.687 | 79209.84 | 19468.88 | 5.302 | 6.46 |
| SUF | 31.72 | 62.975 | 36519.43 | 23884.61 | 4.183 | 14.02 |

**Speedups (SUF vs Sigma)**: online 2.17×, keygen 1.37×, throughput 2.17×.

## D''‑B. Encoder end‑to‑end (BERT‑base seq=256) — blocked
Attempted with:
```
SIGMA_MEMPOOL_DISABLE=1 OMP_NUM_THREADS=32 \
  /workspace/SUF_new/build/gpu_mpc_upstream/sigma bert-base 256 0 127.0.0.1 32
SIGMA_MEMPOOL_DISABLE=1 OMP_NUM_THREADS=32 \
  /workspace/SUF_new/build/gpu_mpc_upstream/sigma bert-base 256 1 127.0.0.1 32
```
Both parties **segfaulted before producing logs**. Logs were empty: `/tmp/sigma_bert256_p0.log`, `/tmp/sigma_bert256_p1.log`.

## D''‑C. Block‑level activation sweep (GELU) @ seq={256,512,1024}
Commands (per seq):
```
python3 scripts/compare_activation_fair.py --seq <SEQ> --models bert-base,gpt2 \
  --iters 20 --runs 1 --sigma-keybuf-mb <KB> --sigma-mempool-mb 1024
```

### seq=256 (`/tmp/activation_seq256.log`)
| Model / Gate | Sigma keygen (ms) | SUF keygen (ms) | Sigma eval (ms) | SUF eval (ms) | Eval speedup | Sigma key (bytes) | SUF key (bytes) | Sigma eval (bytes) | SUF eval (bytes) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bert-base GELU | 21.499 | 0.129 | 22.002 | 1.859 | 11.84x | 452493392 | 72450076 | 27328516 | 20643844 |
| gpt2 GELU | 21.605 | 0.132 | 21.867 | 1.859 | 11.76x | 452493392 | 72450076 | 27328516 | 20643844 |

### seq=512 (`/tmp/activation_seq512.log`)
| Model / Gate | Sigma keygen (ms) | SUF keygen (ms) | Sigma eval (ms) | SUF eval (ms) | Eval speedup | Sigma key (bytes) | SUF key (bytes) | Sigma eval (bytes) | SUF eval (bytes) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bert-base GELU | 37.287 | 0.128 | 45.141 | 3.588 | 12.58x | 904986704 | 144900124 | 54657028 | 41287684 |
| gpt2 GELU | 37.638 | 0.128 | 43.798 | 3.588 | 12.21x | 904986704 | 144900124 | 54657028 | 41287684 |

### seq=1024 (`/tmp/activation_seq1024.log`)
| Model / Gate | Sigma keygen (ms) | SUF keygen (ms) | Sigma eval (ms) | SUF eval (ms) | Eval speedup | Sigma key (bytes) | SUF key (bytes) | Sigma eval (bytes) | SUF eval (bytes) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bert-base GELU | 69.102 | 0.129 | 81.765 | 7.157 | 11.42x | 1809973328 | 289800220 | 109314052 | 82575364 |
| gpt2 GELU | 69.191 | 0.125 | 82.487 | 7.156 | 11.53x | 1809973328 | 289800220 | 109314052 | 82575364 |

## D''‑D. Block‑level softmax (Sigma, L=256/512)
Upstream test binary: `/workspace/SUF_new/build/gpu_mpc_upstream/softmax`

| L | Time (ms) | Comm (MB) | Log |
|---:|---:|---:|---|
| 256 | 42.561 | 46.666 | `/tmp/softmax_upstream_256_p0.log` |
| 512 | 207.166 | 185.967 | `/tmp/softmax_upstream_512_p0.log` |

**SUF softmax (vendor)**: `/workspace/SUF_new/build/gpu_mpc_vendor/softmax` failed with illegal memory access
in `gpu_avgpool.cu:90` (log: `/tmp/softmax_vendor_256_p0.log`).

---

## Artifacts
- Batch JSON: `batch_scaling_bert_pinned.json`, `batch_scaling_gpt2_pinned.json`
- Batch tables: `batch_scaling_tables.md`
- Batch plots: `batch_plots/batch_throughput.svg`, `batch_plots/batch_latency.svg`
- Activation logs: `/tmp/activation_seq256.log`, `/tmp/activation_seq512.log`, `/tmp/activation_seq1024.log`
- Softmax logs: `/tmp/softmax_upstream_256_p0.log`, `/tmp/softmax_upstream_512_p0.log`, `/tmp/softmax_vendor_256_p0.log`

