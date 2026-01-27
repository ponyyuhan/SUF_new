# Absolute Results + LAN/WAN + Ablation Consistency (2026-01-27)

本报告用同一口径汇总：
- 端到端：仅使用 `run_idx >= 1` 的 runs，并对这些 runs 取 median。
- LAN/WAN 投影：使用公式 `T = comp_time + 2 * comm_bytes / bandwidth + rounds * latency`。
- gate microbench / ablation：不做 LAN/WAN 投影，只报告本地 GPU/CPU 计时。

数据来源：
- `results/resume_seq128_runs_2026-01-27.json`
- `results/resume_seq128_more_models_2026-01-27.json`
- `results/resume_e2e_runs_2026-01-27.json`
- `results/ablation_*_bertbase_2026-01-27.jsonl`

## 1) Seq=128 end-to-end absolute values + LAN/WAN projections

| Model | Sigma online (ms) | SUF online (ms) | Speedup | Sigma comm (GiB) | SUF comm (GiB) | Comm ↓ | Sigma keygen (s) | SUF keygen (s) | Keygen ↓ | Sigma key (GiB) | SUF key (GiB) | Key ↓ | Sigma LAN (s) | SUF LAN (s) | Sigma WAN (s) | SUF WAN (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bert-tiny-128 | 63.9 | 42.6 | 1.50x | 0.020 | 0.017 | 16.2% | 0.07 | 0.06 | 12.4% | 0.326 | 0.250 | 23.3% | 0.19 | 0.17 | 0.92 | 0.87 |
| bert-base-128 | 1613.8 | 1149.5 | 1.40x | 0.989 | 0.830 | 16.1% | 1.32 | 1.08 | 18.2% | 16.835 | 12.739 | 24.3% | 3.99 | 3.36 | 11.12 | 9.94 |
| bert-large-128 | 4034.5 | 2997.9 | 1.35x | 2.638 | 2.213 | 16.1% | 3.21 | 2.48 | 22.6% | 45.448 | 34.529 | 24.0% | 10.18 | 8.45 | 26.58 | 23.39 |
| gpt2-128 | 1423.9 | 1072.7 | 1.33x | 0.824 | 0.724 | 12.1% | 1.20 | 0.99 | 17.7% | 14.292 | 11.101 | 22.3% | 3.55 | 3.07 | 10.16 | 9.31 |
| gpt-neo-128 | 6326.2 | 5115.8 | 1.24x | 4.029 | 3.648 | 9.5% | 5.42 | 4.35 | 19.8% | 76.187 | 61.215 | 19.7% | 15.30 | 13.54 | 36.18 | 33.10 |

### 1.1 More absolute breakdown (seq=128)

| Model | Variant | comm time (ms) | comp time (ms) | softmax time (ms) | GELU time (ms) | layernorm time (ms) | truncate time (ms) | total comm (GiB) | softmax comm (GiB) | GELU comm (GiB) | layernorm comm (GiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bert-tiny-128 | sigma | 6.3 | 57.6 | 16.4 | 4.2 | 14.2 | 5.1 | 0.020 | 0.005 | 0.003 | 0.002 |
| bert-tiny-128 | suf | 5.6 | 37.0 | 9.8 | 1.6 | 7.6 | 4.5 | 0.017 | 0.003 | 0.003 | 0.002 |
| bert-base-128 | sigma | 316.2 | 1297.5 | 356.1 | 211.9 | 145.8 | 193.6 | 0.989 | 0.259 | 0.159 | 0.111 |
| bert-base-128 | suf | 129.0 | 1020.4 | 160.4 | 42.5 | 80.3 | 203.6 | 0.830 | 0.139 | 0.120 | 0.111 |
| bert-large-128 | sigma | 645.5 | 3388.9 | 785.8 | 516.6 | 309.4 | 601.0 | 2.638 | 0.691 | 0.425 | 0.297 |
| bert-large-128 | suf | 417.2 | 2580.7 | 399.3 | 118.8 | 205.2 | 504.4 | 2.213 | 0.372 | 0.319 | 0.296 |
| gpt2-128 | sigma | 204.9 | 1219.0 | 228.5 | 197.0 | 151.8 | 221.8 | 0.824 | 0.131 | 0.159 | 0.111 |
| gpt2-128 | suf | 119.7 | 953.0 | 101.8 | 40.4 | 83.5 | 181.4 | 0.724 | 0.071 | 0.120 | 0.111 |
| gpt-neo-128 | sigma | 801.7 | 5524.5 | 442.8 | 921.5 | 518.6 | 1081.2 | 4.029 | 0.357 | 0.867 | 0.604 |
| gpt-neo-128 | suf | 523.6 | 4592.2 | 252.7 | 232.8 | 323.2 | 1010.2 | 3.648 | 0.193 | 0.650 | 0.604 |

## 2) Sequence sweep absolute values (selected-run median)

### 2.1 bert-base (32/64/128, same-mouth rerun)

| Seq | Sigma online (ms) | SUF online (ms) | Speedup | Sigma comm (GiB) | SUF comm (GiB) | Comm ↓ | Sigma key (GiB) | SUF key (GiB) | Key ↓ | Sigma keygen (s) | SUF keygen (s) | Keygen ↓ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 642.3 | 545.5 | 1.18x | 0.185 | 0.167 | 9.4% | 3.706 | 3.024 | 18.4% | 0.60 | 0.56 | 7.3% |
| 64 | 947.4 | 696.0 | 1.36x | 0.411 | 0.361 | 12.1% | 7.431 | 5.839 | 21.4% | 0.98 | 0.70 | 28.4% |
| 128 | 1613.8 | 1149.5 | 1.40x | 0.989 | 0.830 | 16.1% | 16.835 | 12.739 | 24.3% | 1.32 | 1.08 | 18.2% |

### 2.2 bert-base (128/256/512, longer-seq sweep)

| Seq | Sigma online (ms) | SUF online (ms) | Speedup | Sigma comm (GiB) | SUF comm (GiB) | Comm ↓ | Sigma key (GiB) | SUF key (GiB) | Key ↓ | Sigma keygen (s) | SUF keygen (s) | Keygen ↓ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1613.8 | 1149.5 | 1.40x | 0.989 | 0.830 | 16.1% | 16.835 | 12.739 | 24.3% | 1.32 | 1.08 | 18.2% |
| 256 | 2991.5 | 2152.4 | 1.39x | 2.647 | 2.088 | 21.1% | 43.460 | 31.619 | 27.2% | 2.99 | 2.03 | 32.2% |
| 512 | 7694.7 | 5324.1 | 1.45x | 7.966 | 5.891 | 26.0% | 127.981 | 89.704 | 29.9% | 7.08 | 5.01 | 29.2% |

### 2.3 gpt2

| Seq | Sigma online (ms) | SUF online (ms) | Speedup | Sigma comm (GiB) | SUF comm (GiB) | Comm ↓ | Sigma key (GiB) | SUF key (GiB) | Key ↓ | Sigma keygen (s) | SUF keygen (s) | Keygen ↓ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1423.9 | 1072.7 | 1.33x | 0.824 | 0.724 | 12.1% | 14.292 | 11.101 | 22.3% | 1.20 | 0.99 | 17.7% |
| 256 | 2650.0 | 1795.1 | 1.48x | 1.983 | 1.663 | 16.1% | 33.191 | 24.984 | 24.7% | 2.11 | 1.65 | 21.9% |
| 512 | 5630.1 | 3944.0 | 1.43x | 5.302 | 4.183 | 21.1% | 86.687 | 62.975 | 27.4% | 5.16 | 3.73 | 27.8% |

## 3) Gate-level activation signals derived from end-to-end logs

这里用端到端日志中的 GELU scoped 计时与 GELU 通信量，按层数平均为 “每个 activation gate” 的绝对值，作为 gate-level 的同口径参考（不是独立 microbench）。

### 3.1 Seq=128 across models

| Model | Sigma GELU /gate (ms) | SUF GELU /gate (ms) | Gate speedup | Sigma GELU comm /gate (MiB) | SUF GELU comm /gate (MiB) | Comm ↓ |
|---|---:|---:|---:|---:|---:|---:|
| bert-tiny-128 | 2.11 | 0.80 | 2.63x | 1.66 | 1.30 | 21.7% |
| bert-base-128 | 17.66 | 3.54 | 4.99x | 13.59 | 10.22 | 24.8% |
| bert-large-128 | 21.53 | 4.95 | 4.35x | 18.12 | 13.62 | 24.8% |
| gpt2-128 | 16.42 | 3.37 | 4.87x | 13.59 | 10.22 | 24.8% |
| gpt-neo-128 | 38.40 | 9.70 | 3.96x | 37.00 | 27.75 | 25.0% |

### 3.2 BERT-base across seq lengths (32/64/128)

| Seq | Sigma GELU /gate (ms) | SUF GELU /gate (ms) | Gate speedup | Sigma GELU comm /gate (MiB) | SUF GELU comm /gate (MiB) | Comm ↓ |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 5.33 | 1.10 | 4.84x | 3.40 | 2.55 | 24.8% |
| 64 | 8.72 | 1.84 | 4.74x | 6.80 | 5.11 | 24.8% |
| 128 | 17.66 | 3.54 | 4.99x | 13.59 | 10.22 | 24.8% |

## 4) Ablation consistency check (mask-aware padding + program reuse)

Ablation microbench 不应用 LAN/WAN 投影公式；只看本地 GPU/CPU 计时与 key/shape 体积。

### 4.1 Mask-independent shapes via padding (BERT-base GELU gate)

| Variant | mask_aware | per_gate_eval_ms | total_eval_ms | per_gate_key_bytes | pred_bytes | lut_bytes | init_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| mask-dependent | false | 0.995 | 11.946 | 2464 | 384 | 2080 | 1.255 |
| fixed-shape | true | 3.950 | 47.396 | 2848 | 768 | 2080 | 0.762 |

Padding overhead (per_gate_eval_ms): 3.97x; key bytes: 1.16x.

### 4.2 Program reuse / template cache

| Variant | template_cache | #program_inits | init metric (ms) | total_eval_ms | per_gate_eval_ms |
|---|---:|---:|---:|---:|---:|
| cached | true | 1 | init_ms=0.761 | 47.393 | 3.949 |
| no cache | false | 12 | total_init_ms=20.435 | 47.392 | 3.949 |

No-cache extra init: 20.435 ms (BERT-base has 12 gates).

### 4.3 GPU vs CPU reference (same gate workload)

| Mode | per_gate_eval_ms | total_eval_ms | speedup vs CPU |
|---|---:|---:|---:|
| GPU secure | 3.956 | 47.473 | 92.0x |
| CPU ref | 363.798 | 4365.570 | 1.0x |

与 `xiaorong_latest.md` 的数值相比，绝对值会变，但关键比值（padding 约 4x、cache 显著减少 init、GPU 远快于 CPU）保持同一结论口径。

## 5) 公式口径说明（避免混用）

- LAN/WAN 投影公式只用于端到端 online latency（有 rounds / comm_bytes / comm_time 的场景）。
- gate microbench / ablation 是本地 kernel/程序开销，不应套用 rounds/latency 公式。
- 本报告对端到端统一使用 `run_idx>=1` + median；microbench 直接用单次 JSON 输出（因为它本身已是多次 iters 的均值）。
