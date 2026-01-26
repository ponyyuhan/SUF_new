# new_eva results (2026-01-26)

This file follows the order of `new_eva.md` and records **experiment results**, **configs**, and **analysis/notes** for what was actually run in this workspace.  
All times are wall-clock unless noted. SUF/Sigma runs are GPU-MPC (two-GPU local 2-party). BOLT/Bumblebee are best-effort baselines (different threat model or plaintext).

---

## 1) Batch scaling (D'): end-to-end batch scaling (BERT-base + GPT-2, seq=128)

### Setup / config
- Script: `scripts/run_batch_scaling.py` (mode=`serial`, batch in {1,2,4,8}).
- Models: `bert-base`, `gpt2`.
- Sequence length: **128** (fixed).
- Batch sizes: **1, 2, 4, 8**.
- Hardware: 2 GPUs (CUDA devices 0/1) + CPU for host orchestration.
- Threads: `OMP_NUM_THREADS=32` (default in script).
- Common env:
  - `SIGMA_MEMPOOL_DISABLE=1`
  - `CUDA_VISIBLE_DEVICES=0/1` per party
- SUF flags (as in script):
  - `SUF_SOFTMAX=1`, `SUF_LAYERNORM=1`, `SUF_ACTIVATION=1`
  - `SUF_NEXP_BITS=10`, `SUF_INV_BITS=10`, `SUF_RSQRT_BITS=9`
- BERT batch scaling file: `batch_scaling_bert_pinned.json` (run with pinned keybuf; see filename).
- GPT-2 batch scaling file: `batch_scaling_gpt2.json`.
- Validation rerun (BERT, batch=1): `batch_scaling_bert_seq128_rerun.json`.

### Results -- BERT-base (seq=128)
**Table columns** are per-inference latency (ms), throughput (tokens/s), online comm per inference (GB), and projected time per inference (LAN/WAN).  
(For the projected LAN/WAN time, we divide the script's total by batch.)

| batch | sigma_lat_ms | suf_lat_ms | sigma_tps | suf_tps | sigma_comm_gb | suf_comm_gb | sigma_lan_s | suf_lan_s | sigma_wan_s | suf_wan_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1372.246 | 1035.723 | 93.278 | 123.585 | 0.989 | 0.830 | 3.873 | 3.249 | 11.009 | 9.828 |
| 2 | 1314.936 | 1026.139 | 97.343 | 124.739 | 0.989 | 0.830 | 3.832 | 3.234 | 10.967 | 9.813 |
| 4 | 1356.296 | 1047.272 | 94.375 | 122.222 | 0.989 | 0.830 | 3.861 | 3.248 | 10.996 | 9.827 |
| 8 | 1373.749 | 1070.195 | 93.176 | 119.604 | 0.989 | 0.830 | 3.853 | 3.266 | 10.988 | 9.845 |

**Batch=1 validation rerun** (`batch_scaling_bert_seq128_rerun.json`):
- Sigma: **1319.295 ms**, 0.989 GB
- SUF: **1041.672 ms**, 0.830 GB

**SUF vs Sigma (BERT)**: **SUF is better at every batch** (lower latency, higher throughput, lower comm).

### Results -- GPT-2 (seq=128)
| batch | sigma_lat_ms | suf_lat_ms | sigma_tps | suf_tps | sigma_comm_gb | suf_comm_gb | sigma_lan_s | suf_lan_s | sigma_wan_s | suf_wan_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2012.132 | 5770.907 | 63.614 | 22.180 | 0.824 | 0.724 | 4.179 | 7.751 | 10.782 | 13.990 |
| 2 | 2154.994 | 2457.706 | 59.397 | 52.081 | 0.824 | 0.724 | 4.305 | 3.745 | 10.909 | 9.984 |
| 4 | 5631.801 | 2927.680 | 22.728 | 43.721 | 0.824 | 0.724 | 7.416 | 4.857 | 14.019 | 11.096 |
| 8 | 3009.558 | 2825.802 | 42.531 | 45.297 | 0.824 | 0.724 | 5.182 | 4.629 | 11.786 | 10.868 |

**SUF vs Sigma (GPT-2)**: **SUF is worse at batch=1/2, but better at batch=4/8**. This needs follow-up (see analysis below).

### Analysis (batch scaling)
- **BERT** behaves as expected: SUF is consistently faster and lower-comm across all batches.
- **GPT-2** shows **SUF slower at small batches** (b=1/2). Possible causes:
  1) kernel launch/overhead dominates for GPT-2 at small batch;
  2) block schedule or SUF kernel fusion may be less effective in GPT-2 at b=1/2;
  3) a configuration mismatch (e.g., op fusion flags) between Sigma and SUF runs.
- **Next step**: rerun GPT-2 with explicit profiling + confirm that Sigma/SUF use identical kernel paths and same pinned-keybuf policy; if still slower, optimize GPT-2 kernels or adjust SUF parameters.

---

## 2) Longer sequence scaling (D'')

### BERT-base seq=256 (end-to-end)
- **Status**: **Not complete**. End-to-end BERT seq=256 still **segfaults** (reported earlier as `cudaMemcpy invalid argument` in `gpu_mem.cu`).
- **Current data**: no valid seq=256 BERT measurements in this workspace.
- **Planned fix path** (not yet executed in this run):
  - instrument buffer sizes and memcpy byte counts to confirm overflow/size mismatch;
  - check `MAX_*` constants and scratch sizes for seq=256;
  - if necessary, split attention/softmax/norm into blocks for seq=256 to avoid oversized buffers.

### GPT-2 seq=256 / 512
- **Status**: not run in this session. No seq=256/512 GPT-2 results recorded here.

**SUF vs Sigma (longer seq)**: **No conclusion yet** due to missing BERT seq=256 and GPT-2 seq=256/512 results.

---

## (Baseline systems context)
Below are open-source baselines discussed in `new_eva.md`. We record reproducibility status and the experiments that were actually run.

## 1) Open-source status check (BOLT / Bumblebee / IRON)

### BOLT (S&P'24, EzPC/SCI + SEAL)
- **Status**: **Runnable** from EzPC/SCI build in this repo.
- Code path: `baselines/EzPC/SCI/tests/bert_bolt/bert.cpp` (BOLT_BERT binary).
- Weights: external (Google Drive); not committed. Local path used: `baselines/BOLT_weights/bolt/quantize/mrpc`.

### Bumblebee (NDSS'25, OpenBumbleBee)
- **Status**: **GPU plaintext runs OK**; SPU GPU is **not viable** with current JAX/XLA.
- Repo path: `baselines/OpenBumbleBee`.
- We modified the example scripts to **fix seq=128** and add **JAX compatibility shims**.

### IRON
- **Status**: **Not runnable as secure baseline** in this repo (public repo only includes plaintext fixed-point scripts; secure inference code is not released).
- Action: **report as "not directly comparable / not reproducible"** with citation and explanation.

---

## 2) Baseline comparison experiments and results (best-effort only)

### A. Presentation strategy (fairness grouping)
- **Primary comparison**: SUF vs Sigma (same protocol and hardware) -- most fair.
- **Secondary (appendix) comparisons**: BOLT and Bumblebee -- **different threat model**, so best-effort only.

### B. BOLT vs SUF (BERT MRPC, seq=128, batch=1)

**Alignment checks**
- BOLT input shape: `inputs_0_data.txt` is **(128, 768)** and mask `(128,)` -> **seq=128 confirmed**.
- SUF uses **BERT-base seq=128 batch=1**.

**BOLT command** (two processes on localhost):
```
/workspace/SUF_new/baselines/EzPC/SCI/build/bin/BOLT_BERT \
  r=1 p=8000 ip=127.0.0.1 \
  path=/workspace/SUF_new/baselines/BOLT_weights/bolt/quantize/mrpc \
  num_sample=1 id=0 prune=0 output=/tmp/bolt_bert_out_p1.txt

/workspace/SUF_new/baselines/EzPC/SCI/build/bin/BOLT_BERT \
  r=2 p=8000 ip=127.0.0.1 \
  path=/workspace/SUF_new/baselines/BOLT_weights/bolt/quantize/mrpc \
  num_sample=1 id=0 prune=0 output=/tmp/bolt_bert_out_p2.txt
```
- `OMP_NUM_THREADS=4`.
- Instrumentation: `bert.cpp` updated to print **total** rounds and **total** comm (`Communication rounds (total)` / `Communication overhead (total)`).

**BOLT result (CPU, 2PC+HE)**
- End-to-end: **P1 295.439 s**, **P2 354.371 s**
- Comm (total): **P1 28,882,609,673 bytes**, **P2 29,146,859,639 bytes**
  - Sum: **58,029,469,312 bytes (~54.04 GiB)**
- Rounds (total): **124,180**
- Logs: `/tmp/bolt_bert_p1_seq128_total2.log`, `/tmp/bolt_bert_p2_seq128_total2.log`

**SUF reference (GPU FSS, seq=128, batch=1)**
- Online time: **1.041672 s** (`batch_scaling_bert_seq128_rerun.json`)
- Comm: **0.829983 GB**
- Rounds: **1116**

**SUF vs BOLT (BERT)**: **SUF is better** (orders of magnitude faster, far less comm).  
**Important**: This is **best-effort only**, since BOLT is CPU 2PC+HE and SUF is GPU FSS.

### C. Bumblebee vs SUF (GPU plaintext, seq=128)

**Bumblebee settings**
- Repo: `baselines/OpenBumbleBee`.
- GPU plaintext only: `SKIP_SPU=1` (SPU GPU path is experimental / failing).
- Fixed seq=128 in `flax_bert.py` and `flax_gpt2.py`.
- JAX: 0.9.0 with CUDA devices available.

**Bumblebee plaintext GPU results**
- **BERT**: **3.286 s** (`/tmp/bumble_bert_e2e_gpu_seq128.log`)
- **GPT-2**: **40.606 s** (`/tmp/bumble_gpt2_e2e_gpu_seq128.log`)

**SUF references (GPU FSS, seq=128, batch=1)**
- BERT SUF: **1.041672 s**, 0.830 GB
- GPT-2 SUF: **5.770907 s**, 0.724 GB

**SUF vs Bumblebee plaintext (GPU)**: **SUF is better** for both BERT and GPT-2 (lower latency).

**Note**: plaintext GPU is a hardware-aligned but **protocol-mismatched** comparison.

---

## 3) IRON -- why we do not report it as an executable baseline
- Public repo is **plaintext fixed-point scripts only**, no secure inference implementation.
- Threat model differs (2PC client-server vs SUF two-server preprocessing).
- We will cite the repo and explain that it is **not reproducible as a secure baseline**.

---

## Summary of "SUF vs baseline" outcomes
- **SUF vs Sigma (BERT)**: **SUF better at all batches** (lower latency, higher throughput, lower comm).
- **SUF vs Sigma (GPT-2)**: **SUF worse at batch=1/2, better at batch=4/8** -> needs follow-up.
- **SUF vs BOLT (BERT MRPC)**: **SUF better** (much faster, far less comm).
- **SUF vs Bumblebee plaintext (BERT/GPT-2)**: **SUF better**.

