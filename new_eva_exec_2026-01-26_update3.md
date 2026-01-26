# Update: 2026-01-26 (Batch=1 recheck + BumbleBee microbench)

## A. BERT batch=1 recheck (seq=128)
**Conclusion**: SUF faster than Sigma (**1.34×** speedup).

**Command**:
```
python3 /workspace/SUF_new/scripts/run_batch_scaling.py --models bert-base --seq 128 --batches 1 \
  --threads 32 --mode serial --out /workspace/SUF_new/batch_scaling_bert_fix_2026-01-26b.json
```

**Results (per‑inference)**
- Sigma: 3096.332 ms, 41.34 tokens/s, comm 0.989 GB, key 16.835 GB
- SUF:   2313.589 ms, 55.33 tokens/s, comm 0.830 GB, key 12.739 GB

**Notes**
- If you ever see SUF slower for batch=1, check for global overrides (e.g., `SIGMA_PINNED_KEYBUF=0`) or heavy CPU contention.

## B. OpenBumbleBee microbench (SPU/Cheetah, CPU fallback)
**Environment**: conda env `bb` (CPU, no CUDA‑enabled jaxlib). All runs via `bazel run -c opt`.

### GELU (input size = 1024)
**Command**:
```
/root/miniconda3/bin/conda run -n bb --no-capture-output bash -lc \
  'cd /workspace/SUF_new/baselines/OpenBumbleBee && bazel run -c opt //examples/python/microbench:gelu'
```
**Result** (from `/tmp/bumble_gelu.log`):
- Total time: **0.725589 s**
- Comm: send **1.7002 MiB**, recv **1.9208 MiB**

### Softmax (input shape = 128 × 32)
**Command**:
```
/root/miniconda3/bin/conda run -n bb --no-capture-output bash -lc \
  'cd /workspace/SUF_new/baselines/OpenBumbleBee && bazel run -c opt //examples/python/microbench:softmax'
```
**Result** (from `/tmp/bumble_softmax.log`):
- Total time: **1.436310 s**
- Comm: send **6.4037 MiB**, recv **7.4708 MiB**

### Batch MatMul (batch=16, 64×128 × 128×256)
**Command**:
```
/root/miniconda3/bin/conda run -n bb --no-capture-output bash -lc \
  'cd /workspace/SUF_new/baselines/OpenBumbleBee && bazel run -c opt //examples/python/microbench:matmul'
```
**Result** (from `/tmp/bumble_matmul.log`):
- Total time: **2.633057 s**
- Comm: send **20.0104 MiB**, recv **20.0101 MiB**

## C. BOLT BERT end‑to‑end (MRPC weights)
**Status**: completed.

**Command** (two parties):
```
/workspace/SUF_new/baselines/EzPC/SCI/build/bin/BOLT_BERT r=1 p=40400 ip=127.0.0.1 \
  path=/workspace/SUF_new/baselines/BOLT_weights/bolt/quantize/mrpc num_class=2 id=0 num_sample=1 \
  output=/workspace/SUF_new/baselines/BOLT_weights/bolt/output.txt
/workspace/SUF_new/baselines/EzPC/SCI/build/bin/BOLT_BERT r=2 p=40400 ip=127.0.0.1 \
  path=/workspace/SUF_new/baselines/BOLT_weights/bolt/quantize/mrpc num_class=2 id=0 num_sample=1 \
  output=/workspace/SUF_new/baselines/BOLT_weights/bolt/output.txt
```

**Results** (from logs):
- Party‑1 end‑to‑end: **403.426 s**
- Party‑2 end‑to‑end: **435.943 s**
- Total comm (sum of per‑op bytes in party‑1 log): **26.88 GiB**
- Total rounds (sum of per‑op rounds in party‑1 log): **123,927**

Logs: `/tmp/bolt_bert_p1.log`, `/tmp/bolt_bert_p2.log`
Output: `/workspace/SUF_new/baselines/BOLT_weights/bolt/output.txt`

## D. BERT‑base seq=256 end‑to‑end (rerun)
**Fix**: runs complete when overriding key buffer (e.g., `SIGMA_KEYBUF_GB=80`, `SIGMA_PINNED_KEYBUF=0`).

**Important**: run with working directory set to the experiment folder so `output/P0|P1` exists.

**Commands** (workdir = `/workspace/SUF_new/ezpc_upstream/GPU-MPC/experiments/sigma`):
Sigma:
```
(env SIGMA_MEMPOOL_DISABLE=1 SIGMA_KEYBUF_GB=80 SIGMA_PINNED_KEYBUF=0 OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=0 \
  /workspace/SUF_new/build/gpu_mpc_upstream/sigma bert-base 256 0 127.0.0.1 32 > /tmp/sigma_bert256_k80_p0.log 2>&1) &
(env SIGMA_MEMPOOL_DISABLE=1 SIGMA_KEYBUF_GB=80 SIGMA_PINNED_KEYBUF=0 OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=1 \
  /workspace/SUF_new/build/gpu_mpc_upstream/sigma bert-base 256 1 127.0.0.1 32 > /tmp/sigma_bert256_k80_p1.log 2>&1) &
wait
```
SUF (workdir = `/workspace/SUF_new/third_party/EzPC_vendor/GPU-MPC/experiments/sigma`):
```
(env SIGMA_MEMPOOL_DISABLE=1 SIGMA_KEYBUF_GB=80 SIGMA_PINNED_KEYBUF=0 OMP_NUM_THREADS=32 \
  SUF_SOFTMAX=1 SUF_LAYERNORM=1 SUF_ACTIVATION=1 SUF_NEXP_BITS=10 SUF_INV_BITS=10 SUF_RSQRT_BITS=9 \
  CUDA_VISIBLE_DEVICES=0 /workspace/SUF_new/build/gpu_mpc_vendor/sigma bert-base 256 0 127.0.0.1 32 > /tmp/suf_bert256_k80_p0.log 2>&1) &
(env SIGMA_MEMPOOL_DISABLE=1 SIGMA_KEYBUF_GB=80 SIGMA_PINNED_KEYBUF=0 OMP_NUM_THREADS=32 \
  SUF_SOFTMAX=1 SUF_LAYERNORM=1 SUF_ACTIVATION=1 SUF_NEXP_BITS=10 SUF_INV_BITS=10 SUF_RSQRT_BITS=9 \
  CUDA_VISIBLE_DEVICES=1 /workspace/SUF_new/build/gpu_mpc_vendor/sigma bert-base 256 1 127.0.0.1 32 > /tmp/suf_bert256_k80_p1.log 2>&1) &
wait
```

**Results (per‑inference, from evaluator/dealer)**
- Sigma: **41,852.627 ms**, comm **2.647 GB**, key **43.460 GB**
  - Logs: `/workspace/SUF_new/ezpc_upstream/GPU-MPC/experiments/sigma/output/P0/models/bert-base-256/`
- SUF: **7,465.184 ms**, comm **2.088 GB**, key **31.619 GB**
  - Logs: `/workspace/SUF_new/third_party/EzPC_vendor/GPU-MPC/experiments/sigma/output/P0/models/bert-base-256/`
