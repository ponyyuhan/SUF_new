# Accuracy Report (Table 4 Style)

Date: 2026-01-20

## 1. Overview
本报告在现有 PyTorch / SUF fixed‑point 结果基础上，新增 **SUF MPC emulation**（不做真实通信，仅模拟 MPC 定点数值语义），并保留 **Sigma 论文参考值**。

## 2. Methodology
- **Seq length**: 128（统一截断/填充）。
- **Fixed‑point**: frac_bits=12，bitwidth 按 Sigma Table 4（BERT‑tiny=37，BERT‑base/large=50，GPT‑2=50，GPT‑Neo=51）。
- **PyTorch baseline**: HF Transformers float32 inference。
- **Sigma (paper)**: 论文 Table 4 报告值。
- **SUF emulation**: `bench/fixed_point.py` 量化权重+激活，四舍五入 + 二补码 wrap（与之前一致）。
- **SUF MPC emulation**:
  - 仍为单机清文模拟（无通信），仅模拟 **MPC 的截断/舍入语义**。
  - 默认使用 `trunc`（towards‑zero）以贴近安全截断；
  - **GPT‑Neo 使用 `round`**（见 §4 的稳定性说明）。
- **GLUE**: 使用 validation split，指标为 accuracy（MRPC 仅取 accuracy）。
- **LAMBADA**: next‑token accuracy；若 `lambada_openai` 不可用则退化到 `lambada:test`（本次为 `lambada:test`, size=5153）。

## 3. Environment
- **GPU**: 2× NVIDIA GeForce RTX 5090 (32GB each)
- **CPU**: AMD EPYC 9374F (32C/64T)
- **RAM**: 314 GiB（可用 ~193 GiB）
- **Disk**: 2.0 TB (avail ~1.6 TB)
- **Software**: torch 2.9.1+cu130, transformers 4.57.6, datasets 4.5.0

## 4. Stability & Stepwise Runs
- 先做 **小样本 sanity**（BERT‑tiny / GPT‑Neo）确认 MPC emulation 不崩溃与精度合理，再扩展到全量。
- GPT‑Neo + `trunc` 在全量 GPU 跑时触发 CUDA launch failure；改为 `CUDA_LAUNCH_BLOCKING=1` + `--debug-sync`，并将 MPC rounding 设为 `round` 后稳定完成。
- BERT 与 GPT‑2 使用 `trunc` 仍稳定。

## 5. Why SUF ≈ PyTorch
SUF/SUF‑MPC 结果接近 PyTorch 的主要原因：
- frac_bits=12 与足够 bitwidth 让量化误差极小；
- 前向推理中量化位置（Linear/LayerNorm/Attention/激活等）对输出分布的扰动有限；
- LAMBADA 任务为 top‑1 token 判断，对 logits 小幅扰动的鲁棒性较强。

## 6. Results Table
| Model | Dataset | Train Size | Val Size | PyTorch Acc | Sigma Acc | SUF Acc | SUF MPC Acc | Bitwidth | frac_bits | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| bert_tiny_sst2 | glue/sst2 | 67349 | 872 | 80.39 | 82.57 | 80.39 | 80.28 | 37 | 12 | HF: takedarn/bert-tiny-sst2 |
| bert_tiny_mrpc | glue/mrpc | 3668 | 408 | 73.53 | 70.34 | 73.53 | 73.28 | 37 | 12 | HF: M-FAC/bert-tiny-finetuned-mrpc |
| bert_tiny_qnli | glue/qnli | 104743 | 5463 | 81.57 | 81.93 | 81.60 | 81.51 | 37 | 12 | HF: M-FAC/bert-tiny-finetuned-qnli |
| bert_base_sst2 | glue/sst2 | 67349 | 872 | 92.43 | 92.55 | 92.43 | 92.32 | 50 | 12 | HF: textattack/bert-base-uncased-SST-2 |
| bert_base_mrpc | glue/mrpc | 3668 | 408 | 87.75 | 87.25 | 87.75 | 87.25 | 50 | 12 | HF: textattack/bert-base-uncased-MRPC |
| bert_base_qnli | glue/qnli | 104743 | 5463 | 91.54 | 91.63 | 91.52 | 91.63 | 50 | 12 | HF: textattack/bert-base-uncased-QNLI |
| bert_large_sst2 | glue/sst2 | 67349 | 872 | 93.46 | 93.35 | 93.35 | 93.35 | 50 | 12 | HF: yoshitomo-matsubara/bert-large-uncased-sst2 |
| bert_large_mrpc | glue/mrpc | 3668 | 408 | 87.99 | 88.48 | 87.99 | 87.99 | 50 | 12 | HF: yoshitomo-matsubara/bert-large-uncased-mrpc |
| bert_large_qnli | glue/qnli | 104743 | 5463 | 92.24 | 92.26 | 92.26 | 92.35 | 50 | 12 | HF: yoshitomo-matsubara/bert-large-uncased-qnli |
| gpt2_lambada | lambada (lambada:test) | 0 | 5153 | 60.59 | 33.28 | 60.61 | 60.90 | 50 | 12 | HF: gpt2 \| dataset=lambada:test |
| gpt_neo_1p3b_lambada | lambada (lambada:test) | 0 | 5153 | 72.09 | 57.81 | 72.09 | 72.09 | 51 | 12 | HF: EleutherAI/gpt-neo-1.3B \| dataset=lambada:test |

## 7. Notes
- Sigma Acc 列为论文 Table 4 报告值（参考基线），并未在本次运行中重新复现。
- SUF/SUF‑MPC 仍是 **清文模拟**，用于验证定点语义一致性；真实 MPC 通信的正确性需结合端到端协议评估结果。
