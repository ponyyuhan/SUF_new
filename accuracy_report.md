| Model | Dataset | Train Size | Val Size | PyTorch Acc | Sigma Acc | SUF Acc | Bitwidth | frac_bits | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| bert_tiny_sst2 | glue/sst2 | 67349 | 872 | 80.39 | 82.57 | 80.39 | 37 | 12 | HF: takedarn/bert-tiny-sst2 |
| bert_tiny_mrpc | glue/mrpc | 3668 | 408 | 73.53 | 70.34 | 73.53 | 37 | 12 | HF: M-FAC/bert-tiny-finetuned-mrpc |
| bert_tiny_qnli | glue/qnli | 104743 | 5463 | 81.57 | 81.93 | 81.60 | 37 | 12 | HF: M-FAC/bert-tiny-finetuned-qnli |
| bert_base_sst2 | glue/sst2 | 67349 | 872 | 92.43 | 92.55 | 92.43 | 50 | 12 | HF: textattack/bert-base-uncased-SST-2 |
| bert_base_mrpc | glue/mrpc | 3668 | 408 | 87.75 | 87.25 | 87.75 | 50 | 12 | HF: textattack/bert-base-uncased-MRPC |
| bert_base_qnli | glue/qnli | 104743 | 5463 | 91.54 | 91.63 | 91.52 | 50 | 12 | HF: textattack/bert-base-uncased-QNLI |
| bert_large_sst2 | glue/sst2 | 67349 | 872 | 93.46 | 93.35 | 93.35 | 50 | 12 | HF: yoshitomo-matsubara/bert-large-uncased-sst2 |
| bert_large_mrpc | glue/mrpc | 3668 | 408 | 87.99 | 88.48 | 87.99 | 50 | 12 | HF: yoshitomo-matsubara/bert-large-uncased-mrpc |
| bert_large_qnli | glue/qnli | 104743 | 5463 | 92.24 | 92.26 | 92.26 | 50 | 12 | HF: yoshitomo-matsubara/bert-large-uncased-qnli |
| gpt2_lambada | lambada (lambada:test) | 0 | 5153 | 60.59 | 33.28 | 60.61 | 50 | 12 | HF: gpt2 \| dataset=lambada:test |
| gpt_neo_1p3b_lambada | lambada (lambada:test) | 0 | 5153 | 72.09 | 57.81 | 72.09 | 51 | 12 | HF: EleutherAI/gpt-neo-1.3B \| dataset=lambada:test |
