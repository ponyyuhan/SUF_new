# New run results (2026-01-26)

This file records only the **newly run** results from this session.
All runs use `scripts/run_batch_scaling.py` with:
- OMP_NUM_THREADS=32
- SIGMA_MEMPOOL_DISABLE=1 (script default)
- SIGMA_PINNED_KEYBUF=1 (pinned-keybuf on)
- SUF_SOFTMAX=1 SUF_LAYERNORM=1 SUF_ACTIVATION=1
- SUF_NEXP_BITS=10 SUF_INV_BITS=10 SUF_RSQRT_BITS=9

Additional per-run overrides:
- BERT seq=256: SIGMA_KEYBUF_GB=80
- BERT seq=512: SIGMA_KEYBUF_GB=220
- GPT-2 seq=256: SIGMA_KEYBUF_GB=64
- GPT-2 seq=512: SIGMA_KEYBUF_GB=140

## Updated summary sentence (BERT-base + GPT-2, seq=128/256/512, batch=1)
Based on the new runs in this file, SUF delivers **1.27x-1.65x** end-to-end speedup and reduces online communication by **12.14%-26.05%** on BERT-base and GPT-2. Preprocessing is also lighter, with **17.91%-33.64%** lower key-generation time and **22.33%-29.91%** smaller key size.

## BERT-base sequence sweep (batch=1)

| Seq | Sigma online ms | SUF online ms | Sigma tokens/s | SUF tokens/s | Sigma comm GB | SUF comm GB | Sigma keygen s | SUF keygen s | Sigma key GB | SUF key GB | Sigma rounds | SUF rounds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1319.295 | 1041.672 | 97.022 | 122.879 | 0.989 | 0.830 | 1.427 | 0.947 | 16.835 | 12.739 | 1128 | 1116 |
| 256 | 2888.296 | 1978.612 | 88.634 | 129.384 | 2.647 | 2.088 | 2.493 | 1.846 | 43.460 | 31.619 | None | None |
| 512 | 6845.579 | 4847.335 | 74.793 | 105.625 | 7.966 | 5.891 | 6.497 | 4.631 | 127.981 | 89.704 | None | None |

Result files:
- `batch_scaling_bert_seq128_rerun.json`
- `batch_scaling_bert_seq256_rerun.json`
- `batch_scaling_bert_seq512_rerun.json`

### SUF vs Sigma delta (BERT-base, batch=1)
| Seq | SUF speedup (x) | Latency reduction (%) | Comm reduction (GB) | Comm reduction (%) |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1.267 | 21.04 | 0.159 | 16.11 |
| 256 | 1.460 | 31.50 | 0.558 | 21.10 |
| 512 | 1.412 | 29.19 | 2.075 | 26.05 |

## GPT-2 batch scaling (seq=128)

| Batch | Sigma online ms/inf | SUF online ms/inf | Sigma tokens/s | SUF tokens/s | Sigma comm GB/inf | SUF comm GB/inf | Sigma rounds | SUF rounds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1198.563 | 931.429 | 106.795 | 137.423 | 0.824 | 0.724 | 1128 | 1116 |
| 2 | 1175.450 | 948.197 | 108.894 | 134.993 | 0.824 | 0.724 | 2256 | 2232 |
| 4 | 1348.578 | 944.308 | 94.915 | 135.549 | 0.824 | 0.724 | 4512 | 4464 |
| 8 | 1217.743 | 987.269 | 105.112 | 129.651 | 0.824 | 0.724 | 9024 | 8928 |

Result file: `batch_scaling_gpt2_seq128_rerun.json`

### SUF vs Sigma delta (GPT-2 batch scaling, seq=128)
| Batch | SUF speedup (x) | Latency reduction (%) | Comm reduction (GB) | Comm reduction (%) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.287 | 22.29 | 0.100 | 12.14 |
| 2 | 1.240 | 19.33 | 0.100 | 12.14 |
| 4 | 1.428 | 29.98 | 0.100 | 12.14 |
| 8 | 1.233 | 18.93 | 0.100 | 12.14 |

## GPT-2 sequence sweep (batch=1)

| Seq | Sigma online ms | SUF online ms | Sigma tokens/s | SUF tokens/s | Sigma comm GB | SUF comm GB | Sigma keygen s | SUF keygen s | Sigma key GB | SUF key GB | Sigma rounds | SUF rounds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1198.563 | 931.429 | 106.795 | 137.423 | 0.824 | 0.724 | 1.051 | 0.863 | 14.292 | 11.101 | 1128 | 1116 |
| 256 | 2254.409 | 1639.911 | 113.555 | 156.106 | 1.983 | 1.663 | 1.995 | 1.545 | 33.191 | 24.984 | None | None |
| 512 | 5791.808 | 3517.568 | 88.401 | 145.555 | 5.302 | 4.183 | 4.532 | 3.325 | 86.687 | 62.975 | None | None |

Result files:
- `batch_scaling_gpt2_seq128_rerun.json`
- `batch_scaling_gpt2_seq256_rerun.json`
- `batch_scaling_gpt2_seq512_rerun.json`

### SUF vs Sigma delta (GPT-2 sequence sweep, batch=1)
| Seq | SUF speedup (x) | Latency reduction (%) | Comm reduction (GB) | Comm reduction (%) |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1.287 | 22.29 | 0.100 | 12.14 |
| 256 | 1.375 | 27.26 | 0.320 | 16.13 |
| 512 | 1.647 | 39.27 | 1.119 | 21.10 |

## Error vs evaluation_latest.md (2026-01-23)
Only overlapping points are compared (BERT-base-128, GPT-2-128, GPT-2-256).

| Model-Seq | Sigma online ms | SUF online ms | Sigma comm GB | SUF comm GB |
| --- | --- | --- | --- | --- |
| bert-base-128 | 1682.040 -> 1319.295 (-362.745, -21.57%) | 1313.860 -> 1041.672 (-272.188, -20.72%) | 0.989 -> 0.989 (+0.000428, +0.043%) | 0.830 -> 0.830 (-0.000017, -0.002%) |
| gpt2-128 | 1513.570 -> 1198.563 (-315.007, -20.81%) | 1073.320 -> 931.429 (-141.891, -13.22%) | 0.824 -> 0.824 (+0.000357, +0.043%) | 0.724 -> 0.724 (+0.000319, +0.044%) |
| gpt2-256 | 3171.600 -> 2254.409 (-917.191, -28.92%) | 2165.380 -> 1639.911 (-525.469, -24.27%) | 1.983 -> 1.983 (-0.000118, -0.006%) | 1.663 -> 1.663 (+0.000057, +0.003%) |

## Incomplete / failed
- BERT seq=384 failed on Sigma with `cudaMallocAsync` OOM.
  - log: `/tmp/sigma_batch_bert-base_384_b1_r0_p0.log`
  - attempted with SIGMA_KEYBUF_GB=140; rerun with SIGMA_KEYBUF_GB=180 was interrupted.
