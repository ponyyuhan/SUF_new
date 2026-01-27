# Absolute Communication / Key Sizes (selected-run median) — 2026-01-27

口径说明（与 2026-01-27 主表一致）：
- 仅使用 `run_idx >= 1` 的结果（丢弃 run0 作为 warmup/异常值）。
- 对选中 runs 取 `median` 作为最终值。
- 数据来源：`results/resume_seq128_runs_2026-01-27.json` 与 `results/resume_e2e_runs_2026-01-27.json`。

## 1) bert-base — end-to-end + absolute comm/key

| Seq | Sigma total (ms) | SUF total (ms) | Speedup | Sigma comm (GiB) | SUF comm (GiB) | Comm ↓ | Sigma key (GiB) | SUF key (GiB) | Key ↓ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1613.8 | 1149.5 | 1.40x | 0.989 | 0.830 | 16.1% | 16.835 | 12.739 | 24.3% |
| 256 | 2991.5 | 2152.4 | 1.39x | 2.647 | 2.088 | 21.1% | 43.460 | 31.619 | 27.2% |
| 512 | 7694.7 | 5324.1 | 1.45x | 7.966 | 5.891 | 26.0% | 127.981 | 89.704 | 29.9% |

### More absolute values (time + communication breakdown)

| Seq | Variant | keygen (ms) | online comm time (ms) | softmax time (ms) | GELU time (ms) | layernorm time (ms) | truncate time (ms) | total comm (GiB) | softmax comm (GiB) | GELU comm (GiB) | layernorm comm (GiB) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | sigma | 1324.4 | 316.2 | 356.1 | 211.9 | 145.8 | 193.6 | 0.989 | 0.259 | 0.159 | 0.111 |
| 128 | suf | 1083.0 | 129.0 | 160.4 | 42.5 | 80.3 | 203.6 | 0.830 | 0.139 | 0.120 | 0.111 |
| 256 | sigma | 2994.6 | 455.0 | 976.7 | 359.2 | 211.0 | 377.2 | 2.647 | 1.036 | 0.319 | 0.223 |
| 256 | suf | 2031.1 | 289.8 | 523.6 | 73.6 | 129.6 | 379.5 | 2.088 | 0.557 | 0.240 | 0.222 |
| 512 | sigma | 7081.6 | 1084.4 | 3441.5 | 707.8 | 382.8 | 708.5 | 7.966 | 4.141 | 0.637 | 0.445 |
| 512 | suf | 5012.4 | 859.9 | 1816.2 | 156.4 | 235.5 | 694.1 | 5.891 | 2.225 | 0.479 | 0.444 |

### Absolute deltas (SUF vs Sigma)

| Seq | total speedup | keygen ↓ | online comm time ↓ | softmax time ↓ | GELU time ↓ | layernorm time ↓ | total comm ↓ | softmax comm ↓ | GELU comm ↓ | layernorm comm ↓ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1.40x | 18.2% | 59.2% | 55.0% | 80.0% | 44.9% | 16.1% | 46.2% | 24.8% | 0.2% |
| 256 | 1.39x | 32.2% | 36.3% | 46.4% | 79.5% | 38.6% | 21.1% | 46.2% | 24.8% | 0.2% |
| 512 | 1.45x | 29.2% | 20.7% | 47.2% | 77.9% | 38.5% | 26.0% | 46.3% | 24.8% | 0.2% |

## 2) gpt2 — end-to-end + absolute comm/key

| Seq | Sigma total (ms) | SUF total (ms) | Speedup | Sigma comm (GiB) | SUF comm (GiB) | Comm ↓ | Sigma key (GiB) | SUF key (GiB) | Key ↓ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1423.9 | 1072.7 | 1.33x | 0.824 | 0.724 | 12.1% | 14.292 | 11.101 | 22.3% |
| 256 | 2650.0 | 1795.1 | 1.48x | 1.983 | 1.663 | 16.1% | 33.191 | 24.984 | 24.7% |
| 512 | 5630.1 | 3944.0 | 1.43x | 5.302 | 4.183 | 21.1% | 86.687 | 62.975 | 27.4% |

### More absolute values (time + communication breakdown)

| Seq | Variant | keygen (ms) | online comm time (ms) | softmax time (ms) | GELU time (ms) | layernorm time (ms) | truncate time (ms) | total comm (GiB) | softmax comm (GiB) | GELU comm (GiB) | layernorm comm (GiB) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | sigma | 1204.6 | 204.9 | 228.5 | 197.0 | 151.8 | 221.8 | 0.824 | 0.131 | 0.159 | 0.111 |
| 128 | suf | 991.3 | 119.7 | 101.8 | 40.4 | 83.5 | 181.4 | 0.724 | 0.071 | 0.120 | 0.111 |
| 256 | sigma | 2112.5 | 441.4 | 596.8 | 351.0 | 243.6 | 393.6 | 1.983 | 0.522 | 0.319 | 0.223 |
| 256 | suf | 1648.9 | 243.8 | 297.9 | 72.7 | 149.9 | 379.1 | 1.663 | 0.281 | 0.240 | 0.222 |
| 512 | sigma | 5163.5 | 836.6 | 1886.3 | 724.1 | 387.8 | 771.0 | 5.302 | 2.078 | 0.637 | 0.445 |
| 512 | suf | 3725.6 | 579.8 | 927.4 | 159.7 | 221.6 | 678.5 | 4.183 | 1.119 | 0.479 | 0.444 |

### Absolute deltas (SUF vs Sigma)

| Seq | total speedup | keygen ↓ | online comm time ↓ | softmax time ↓ | GELU time ↓ | layernorm time ↓ | total comm ↓ | softmax comm ↓ | GELU comm ↓ | layernorm comm ↓ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1.33x | 17.7% | 41.6% | 55.5% | 79.5% | 45.0% | 12.1% | 45.9% | 24.8% | 0.2% |
| 256 | 1.48x | 21.9% | 44.8% | 50.1% | 79.3% | 38.5% | 16.1% | 46.1% | 24.8% | 0.2% |
| 512 | 1.43x | 27.8% | 30.7% | 50.8% | 77.9% | 42.9% | 21.1% | 46.2% | 24.8% | 0.2% |

## 3) Combined absolute comm/key table (paper-friendly)

| Model | Seq | Sigma comm (GiB) | SUF comm (GiB) | Comm ↓ | Sigma key (GiB) | SUF key (GiB) | Key ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| bert-base | 128 | 0.989 | 0.830 | 16.1% | 16.835 | 12.739 | 24.3% |
| bert-base | 256 | 2.647 | 2.088 | 21.1% | 43.460 | 31.619 | 27.2% |
| bert-base | 512 | 7.966 | 5.891 | 26.0% | 127.981 | 89.704 | 29.9% |
| gpt2 | 128 | 0.824 | 0.724 | 12.1% | 14.292 | 11.101 | 22.3% |
| gpt2 | 256 | 1.983 | 1.663 | 16.1% | 33.191 | 24.984 | 24.7% |
| gpt2 | 512 | 5.302 | 4.183 | 21.1% | 86.687 | 62.975 | 27.4% |
