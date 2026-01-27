# Evaluation (same-mouth rerun: gate microbench + BERT-base seq sweep) — 2026-01-27

统一口径：
- 端到端与 gate microbench 均从端到端日志派生。
- 选择策略：`run_idx >= 1`，对选中 runs 取 median。
- LAN/WAN 投影仅用于端到端 online latency。

数据来源：
- `results/resume_seq128_runs_2026-01-27.json`
- `results/resume_seq128_more_models_2026-01-27.json`
- `results/resume_e2e_runs_2026-01-27.json`
- `results/resume_seq32_64_runs_2026-01-27.json`

## 1) Seq=128 end-to-end absolute values + LAN/WAN projections

| Model | Sigma online (ms) | SUF online (ms) | Speedup | Sigma comm (GiB) | SUF comm (GiB) | Comm ↓ | Sigma keygen (s) | SUF keygen (s) | Keygen ↓ | Sigma key (GiB) | SUF key (GiB) | Key ↓ | Sigma LAN (s) | SUF LAN (s) | Sigma WAN (s) | SUF WAN (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bert-tiny-128 | 63.9 | 42.6 | 1.50x | 0.020 | 0.017 | 16.2% | 0.07 | 0.06 | 12.4% | 0.326 | 0.250 | 23.3% | 0.19 | 0.17 | 0.92 | 0.87 |
| bert-base-128 | 1613.8 | 1149.5 | 1.40x | 0.989 | 0.830 | 16.1% | 1.32 | 1.08 | 18.2% | 16.835 | 12.739 | 24.3% | 3.99 | 3.36 | 11.12 | 9.94 |
| bert-large-128 | 4034.5 | 2997.9 | 1.35x | 2.638 | 2.213 | 16.1% | 3.21 | 2.48 | 22.6% | 45.448 | 34.529 | 24.0% | 10.18 | 8.45 | 26.58 | 23.39 |
| gpt2-128 | 1423.9 | 1072.7 | 1.33x | 0.824 | 0.724 | 12.1% | 1.20 | 0.99 | 17.7% | 14.292 | 11.101 | 22.3% | 3.55 | 3.07 | 10.16 | 9.31 |
| gpt-neo-128 | 6326.2 | 5115.8 | 1.24x | 4.029 | 3.648 | 9.5% | 5.42 | 4.35 | 19.8% | 76.187 | 61.215 | 19.7% | 15.30 | 13.54 | 36.18 | 33.10 |

## 2) BERT-base seq sweep (32/64/128) — same-mouth rerun

| Seq | Sigma online (ms) | SUF online (ms) | Speedup | Sigma comm (GiB) | SUF comm (GiB) | Comm ↓ | Sigma keygen (s) | SUF keygen (s) | Keygen ↓ | Sigma key (GiB) | SUF key (GiB) | Key ↓ | Sigma LAN (s) | SUF LAN (s) | Sigma WAN (s) | SUF WAN (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 642.3 | 545.5 | 1.18x | 0.185 | 0.167 | 9.4% | 0.60 | 0.56 | 7.3% | 3.706 | 3.024 | 18.4% | 1.52 | 1.39 | 5.89 | 5.66 |
| 64 | 947.4 | 696.0 | 1.36x | 0.411 | 0.361 | 12.1% | 0.98 | 0.70 | 28.4% | 7.431 | 5.839 | 21.4% | 2.27 | 1.93 | 7.46 | 6.92 |
| 128 | 1613.8 | 1149.5 | 1.40x | 0.989 | 0.830 | 16.1% | 1.32 | 1.08 | 18.2% | 16.835 | 12.739 | 24.3% | 3.99 | 3.36 | 11.12 | 9.94 |

## 3) Gate microbench (same-mouth; derived from end-to-end logs)

按层数将 GELU scoped 计时与 GELU 通信量折算为 per-gate：`per_gate = gelu / n_layer`。

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

注：这是“同口径”的 gate microbench（来自端到端 logs），避免了独立 gate bench 与端到端口径不一致的问题。
