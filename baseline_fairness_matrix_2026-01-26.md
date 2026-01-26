# SUF vs Baselines — Fair/Best‑Effort Comparison Matrix (2026‑01‑26)

## Goal
You asked for **same task / same sequence length** and a list of **BOLT / Bumblebee comparisons that are fair or as fair as possible** versus SUF. This document provides:
1) **What is strictly comparable**, 2) **Best‑effort comparable** items, and 3) **exact steps** to align task/sequence length.

---

## 1) Fairness Criteria (what “comparable” means)
To call two runs “fairly comparable”, we should match:
- **Task**: same model + same objective (e.g., BERT MRPC classification, GPT‑2 next‑token generation)
- **Sequence length**: identical max token length (e.g., 128 or 256)
- **Batch size**: identical batch
- **Protocol / threat model**: ideally same; if not, we must label comparison as *best‑effort only*
- **Hardware**: ideally same; if not, label *best‑effort only*
- **Metrics**: end‑to‑end runtime, online runtime, communication, rounds, keygen (if applicable)

### Reality of this repo
- **SUF**: GPU‑centric FSS (two‑server pre‑processing). No CPU build in this repo.
- **BOLT**: CPU‑only 2PC + HE (EzPC/SCI + SEAL). No GPU path.
- **Bumblebee (OpenBumbleBee)**: SPU 2PC runs on CPU; GPU plaintext works, but SPU + JAX 0.9.0 fails due to XLA HLO importer overflow. So SPU end‑to‑end GPU is not currently viable.

**Implication:** Strict fairness between SUF and BOLT/Bumblebee is **not achievable on hardware**, but **best‑effort matching** on **task/seq/batch** is still possible and meaningful **if clearly labeled**.

---

## 2) What is strictly comparable today
### ✅ SUF vs Sigma (same codebase / same protocol)
This is the **most fair** baseline available.
- Same threat model, same hardware (GPU), same code path
- We can match **task, seq length, batch, and metrics** exactly

**Where to find results**
- BERT batch/seq: `batch_scaling_bert*.json` and logs
- GPT‑2 batch/seq: `batch_scaling_gpt2.json`

**Action**: Use these for the **primary comparison**; they already satisfy the fairness criteria.

---

## 3) Best‑effort comparisons (BOLT / Bumblebee vs SUF)
Below are comparisons that can be made **with matched task/seq/batch**, but are still **best‑effort** because the protocols and hardware differ.

### A) BOLT vs SUF (BERT MRPC‑style classification)
**Why best‑effort only**: BOLT is CPU 2PC+HE; SUF is GPU FSS.

**How to align task/seq length**
BOLT evaluation uses **MRPC‑style BERT** in SCI:
- Code path: `baselines/EzPC/SCI/tests/bert_iron/iron_bert.cpp`
- Inputs loaded from: `.../weights_txt_right/inputs_<id>_data.txt` and `inputs_<id>_mask.txt`
- **Sequence length is embedded in input file shape** (token length = input length)

**Concrete alignment steps**
1) **Determine BOLT seq length** from input file shape:
   ```bash
   python3 - <<'PY'
   import numpy as np
   data = np.loadtxt('/path/to/mrpc/weights_txt_right/inputs_0_data.txt', delimiter=',')
   mask = np.loadtxt('/path/to/mrpc/weights_txt_right/inputs_0_mask.txt', delimiter=',')
   print('input shape:', data.shape)
   print('mask shape:', mask.shape)
   PY
   ```
   The second dimension corresponds to **sequence length**.
2) **Set SUF to the exact same seq length**, e.g. `seq=128` or `seq=256`.
3) **Batch size**: BOLT uses `num_sample` in `iron_bert.cpp`. Set to `1` for batch‑1 to match SUF.
4) **Task alignment**: both are **BERT classification**; record as “MRPC‑style” or “BERT‑cls” for SUF.

**Status today**
- BOLT MRPC end‑to‑end is **available** (CPU) but **task/seq length must be checked**.
- SUF BERT seq=128 batch=1 exists; can be re‑run to match exact seq length once confirmed.

**What to report**
- Runtime (end‑to‑end & online)
- Comm (bytes / inference)
- Rounds
- Hardware and protocol disclaimer

### B) Bumblebee vs SUF (BERT / GPT‑2)
**Why best‑effort only**: Bumblebee SPU is CPU 2PC; SUF is GPU FSS.

**Two possible modes**
1) **Bumblebee SPU (CPU)** — closer protocol match, slower hardware
2) **Bumblebee plaintext (GPU JAX)** — closer hardware, no cryptographic protocol

#### Option B1: Bumblebee SPU (CPU) — protocol‑aligned, hardware‑mismatched
- Requires **JAX 0.4.x** (JAX 0.9.0 currently breaks SPU HLO importer)
- Run end‑to‑end BERT/GPT‑2 with **same seq length and batch** as SUF

#### Option B2: Bumblebee plaintext (GPU) — hardware‑aligned, protocol‑mismatched
- Works with JAX 0.9.0 + CUDA (current environment)
- Use **SKIP_SPU=1** to bypass SPU
- BERT/GPT‑2 runtime is plaintext only (not secure)

**Concrete alignment steps (BERT/GPT‑2)**
- **BERT** (`baselines/OpenBumbleBee/examples/python/ml/flax_bert/flax_bert.py`):
  - Ensure tokenizer uses a fixed length:
    ```python
    tokenizer(..., max_length=SEQ, padding='max_length', truncation=True)
    ```
  - Set `SEQ` to match SUF (e.g., 128 or 256)
- **GPT‑2** (`baselines/OpenBumbleBee/examples/python/ml/flax_gpt2/flax_gpt2.py`):
  - Build input with length `SEQ` (pad/truncate to match)
  - Run with `SKIP_SPU=1` for plaintext GPU, or SPU for CPU

**Status today**
- GPU plaintext **BERT** (≈3.08s) and **GPT‑2** (≈32.14s) are recorded.
- SPU end‑to‑end on JAX 0.9.0 fails (`proto.id() > INT_MAX`).

---

## 4) Recommended “best‑effort” comparison sets (matching task/seq/batch)
Below are **specific** candidate comparisons you can ask me to run. Each has a **fairness label**.

### Set 1 — Most Fair (Recommended)
| Pair | Task | Seq | Batch | Fairness | Notes |
|---|---|---:|---:|---|---|
| **SUF vs Sigma** | BERT‑cls | 128 | 1 | **Strict** | Same GPU‑MPC stack |
| **SUF vs Sigma** | BERT‑cls | 256 | 1 | **Strict** | Same GPU‑MPC stack |
| **SUF vs Sigma** | GPT‑2 | 128 | 1 | **Strict** | Same GPU‑MPC stack |

### Set 2 — Best‑Effort Protocol Match (CPU vs GPU)
| Pair | Task | Seq | Batch | Fairness | Notes |
|---|---|---:|---:|---|---|
| **BOLT vs SUF** | BERT MRPC | (match input) | 1 | **Best‑effort** | CPU 2PC+HE vs GPU FSS |
| **Bumblebee SPU vs SUF** | BERT | 128 | 1 | **Best‑effort** | Requires JAX 0.4.x CPU SPU |
| **Bumblebee SPU vs SUF** | GPT‑2 | 128 | 1 | **Best‑effort** | Requires JAX 0.4.x CPU SPU |

### Set 3 — Best‑Effort Hardware Match (GPU plaintext vs GPU MPC)
| Pair | Task | Seq | Batch | Fairness | Notes |
|---|---|---:|---:|---|---|
| **Bumblebee plaintext GPU vs SUF** | BERT | 128 | 1 | **Best‑effort** | Protocol mismatch |
| **Bumblebee plaintext GPU vs SUF** | GPT‑2 | 128 | 1 | **Best‑effort** | Protocol mismatch |

---

## 5) Concrete next steps to satisfy “same task / seq length”
### Step A — Identify exact seq length for BOLT MRPC
Run shape check on BOLT input files:
```bash
python3 - <<'PY'
import numpy as np
x = np.loadtxt('/path/to/mrpc/weights_txt_right/inputs_0_data.txt', delimiter=',')
print(x.shape)
PY
```
If it prints `(1, 128)` → seq=128. Then **set SUF seq=128** and re‑run SUF BERT.

**Observed (this machine):** BOLT MRPC input shape is **(128, 768)** and mask shape **(128,)** → **seq=128 confirmed**.

### Step B — Force Bumblebee BERT/GPT‑2 to a fixed seq length
Update tokenization in Bumblebee to enforce `SEQ`:
- BERT: use `max_length=SEQ, padding='max_length', truncation=True`
- GPT‑2: pad/truncate to `SEQ`

### Step C — Record comparison with explicit fairness label
Every table/result should include:
- **Hardware** (CPU vs GPU)
- **Protocol** (FSS / 2PC+HE / plaintext)
- **Seq / batch**
- **Runtime / comm / rounds**

---

## 6) Known blockers
- **Bumblebee SPU (GPU)** currently fails due to XLA HLO ID overflow with JAX 0.9.0.
- **BOLT** code is in `baselines/EzPC` (SCI), not in the BOLT repo itself.

---

## 8) Latest aligned measurements (seq=128, batch=1)
### SUF (GPU, FSS) — BERT base
- **Runtime (online)**: **~1.042 s** (from `batch_scaling_bert_seq128_rerun.json`)
- **Comm**: **~0.830 GB**

### Sigma (GPU, FSS) — BERT base
- **Runtime (online)**: **~1.319 s** (same run file)

### Bumblebee plaintext (GPU JAX) — BERT base
- **Runtime**: **~3.286 s** (`/tmp/bumble_bert_e2e_gpu_seq128.log`, `SKIP_SPU=1`)
- **Note**: plaintext only (not secure)

### Bumblebee plaintext (GPU JAX) — GPT‑2
- **Runtime**: **~40.606 s** (`/tmp/bumble_gpt2_e2e_gpu_seq128.log`, `SKIP_SPU=1`)
- **Note**: plaintext only (not secure)

### Bumblebee SPU (CPU, 2PC) — BERT base
- **Runtime**: **~289.84 s** (`/tmp/bumble_bert_spu_cpu_seq128.log`, JAX 0.4.26)
- **Note**: SPU run on CPU, protocol‑aligned but hardware‑mismatched

---

## 7) Ask / Decision Needed
Please choose one or more of the following, and I will execute:
1) **Check BOLT MRPC input seq length** and re‑run SUF BERT with the same seq
2) **Pin Bumblebee to a fixed seq length** for BERT/GPT‑2 plaintext GPU and record results
3) **Attempt Bumblebee SPU on CPU** by downgrading JAX (protocol‑aligned best‑effort)
