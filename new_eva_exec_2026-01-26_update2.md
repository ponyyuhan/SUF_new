# Update: 2026-01-26 (Batch/Seq256/Baselines)

## A. BERT batch=1 recheck (seq=128)
**Conclusion**: SUF faster than Sigma.

**Command** (auto pinned keybuf, no global override):
```
python3 scripts/run_batch_scaling.py --models bert-base --seq 128 --batches 1 --threads 32 \
  --mode serial --out /workspace/SUF_new/batch_scaling_bert_fix_2026-01-26.json
```

**Results (per‑inference)**
- Sigma: 1343.389 ms, 95.28 tokens/s, comm 0.989 GB, key 16.835 GB
- SUF:   1022.730 ms, 125.16 tokens/s, comm 0.830 GB, key 12.739 GB

**Notes**
- The script now clears inherited `SIGMA_PINNED_KEYBUF` when `--pinned-keybuf auto` to avoid accidental global overrides.
- If SUF ever appears slower here, check the environment for `SIGMA_PINNED_KEYBUF=0` (it inflates transfer time on SUF).

## B. BERT‑base seq=256 end‑to‑end (segfault investigation)
**Conclusion**: With larger key buffer, both Sigma and SUF run; SUF faster.

**Key config**:
- `SIGMA_KEYBUF_GB=80` (override)
- `SIGMA_PINNED_KEYBUF=0`
- `SIGMA_MEMPOOL_DISABLE=1`

**Commands**
Sigma:
```
cd /workspace/SUF_new/ezpc_upstream/GPU-MPC/experiments/sigma
(env SIGMA_MEMPOOL_DISABLE=1 SIGMA_KEYBUF_GB=80 SIGMA_PINNED_KEYBUF=0 OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=0 \
  /workspace/SUF_new/build/gpu_mpc_upstream/sigma bert-base 256 0 127.0.0.1 32 > /tmp/sigma_bert256_k80_p0.log 2>&1) &
(env SIGMA_MEMPOOL_DISABLE=1 SIGMA_KEYBUF_GB=80 SIGMA_PINNED_KEYBUF=0 OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=1 \
  /workspace/SUF_new/build/gpu_mpc_upstream/sigma bert-base 256 1 127.0.0.1 32 > /tmp/sigma_bert256_k80_p1.log 2>&1) &
wait
```
SUF:
```
cd /workspace/SUF_new/third_party/EzPC_vendor/GPU-MPC/experiments/sigma
(env SIGMA_MEMPOOL_DISABLE=1 SIGMA_KEYBUF_GB=80 SIGMA_PINNED_KEYBUF=0 OMP_NUM_THREADS=32 \
  SUF_SOFTMAX=1 SUF_LAYERNORM=1 SUF_ACTIVATION=1 SUF_NEXP_BITS=10 SUF_INV_BITS=10 SUF_RSQRT_BITS=9 \
  CUDA_VISIBLE_DEVICES=0 /workspace/SUF_new/build/gpu_mpc_vendor/sigma bert-base 256 0 127.0.0.1 32 > /tmp/suf_bert256_k80_p0.log 2>&1) &
(env SIGMA_MEMPOOL_DISABLE=1 SIGMA_KEYBUF_GB=80 SIGMA_PINNED_KEYBUF=0 OMP_NUM_THREADS=32 \
  SUF_SOFTMAX=1 SUF_LAYERNORM=1 SUF_ACTIVATION=1 SUF_NEXP_BITS=10 SUF_INV_BITS=10 SUF_RSQRT_BITS=9 \
  CUDA_VISIBLE_DEVICES=1 /workspace/SUF_new/build/gpu_mpc_vendor/sigma bert-base 256 1 127.0.0.1 32 > /tmp/suf_bert256_k80_p1.log 2>&1) &
wait
```

**Results (per‑inference, from evaluator.txt)**
- Sigma: 19,712.664 ms, comm 2.647 GB, key 43.460 GB
  - Logs: `/workspace/SUF_new/ezpc_upstream/GPU-MPC/experiments/sigma/output/P0/models/bert-base-256/`
- SUF:   6,940.940 ms, comm 2.088 GB, key 31.619 GB
  - Logs: `/workspace/SUF_new/third_party/EzPC_vendor/GPU-MPC/experiments/sigma/output/P0/models/bert-base-256/`

**Observation**: This suggests the previous segfault/`cudaMemcpy invalid argument` likely stemmed from insufficient key buffer or earlier output dir creation issues. With `SIGMA_KEYBUF_GB=80` and explicit output dirs, the run completes.

## C. Baseline: BOLT (EzPC bert branch)
**Repo**:
- BOLT wrapper: `/workspace/SUF_new/baselines/BOLT`
- Code: `/workspace/SUF_new/baselines/EzPC` (branch `bert`)
**Weights** (downloaded from provided Drive link):
- `/workspace/SUF_new/baselines/BOLT_weights/bolt/quantize/mrpc/weights_txt`

**Build status**:
- `cmake` + `cmake --build` succeeded for BOLT/IRON binaries.
- Some unrelated targets (SCI-Beacon, SCI-SecfloatML) failed; BOLT binaries were still produced.
- **Patch applied**: `tests/bert_bolt/nonlinear.h` set `MAX_THREADS=12` (was 64) to avoid bind errors on 64×(base/50/100) ports.

**BOLT microbench (N=128, nt=4)**
Commands (each uses two parties):
```
cd /workspace/SUF_new/baselines/EzPC/SCI/build/bin
./BOLT-gelu r=1 p=32000 ip=127.0.0.1 N=128 nt=4
./BOLT-gelu r=2 p=32000 ip=127.0.0.1 N=128 nt=4

./BOLT-softmax r=1 p=32010 ip=127.0.0.1 N=128 nt=4
./BOLT-softmax r=2 p=32010 ip=127.0.0.1 N=128 nt=4

./BOLT-layer_norm r=1 p=32020 ip=127.0.0.1 N=128 nt=4
./BOLT-layer_norm r=2 p=32020 ip=127.0.0.1 N=128 nt=4
```

Results (party 1 log; party 2 includes ULP checks):
- **GELU**: 68.854 ms, 728,048 bytes (ops/s 1859.01)
  - Logs: `/tmp/bolt_gelu_p1.log`, `/tmp/bolt_gelu_p2.log`
- **Softmax**: 314.757 ms, 61,118,720 bytes (ops/s 406.66)
  - Logs: `/tmp/bolt_softmax_p1.log`, `/tmp/bolt_softmax_p2.log`
- **LayerNorm**: 1410.6 ms, 424,770,384 bytes (ops/s 90.74)
  - Logs: `/tmp/bolt_layernorm_p1.log`, `/tmp/bolt_layernorm_p2.log`

**BOLT BERT end‑to‑end (MRPC weights)**:
- Command (port base 40100):  
  `/workspace/SUF_new/baselines/EzPC/SCI/build/bin/BOLT_BERT r={1,2} p=40100 ip=127.0.0.1 path=/workspace/SUF_new/baselines/BOLT_weights/bolt/quantize/mrpc num_class=2 id=0 num_sample=1 output=/workspace/SUF_new/baselines/BOLT_weights/bolt/output.txt`
- **Status (before fix)**: **segfault** during attention layer 0 (both parties).  
  Logs: `/tmp/bolt_bert_p1.log`, `/tmp/bolt_bert_p2.log`  
  Exit codes: rc0=139, rc1=139 (from session `45659`).
- **Fix attempt**: set `NL_NTHREADS=12` in `tests/bert_bolt/bert.h` to match `MAX_THREADS=12`.
- **Status (after fix)**: rerun on port base `40300` started; progressing through attention layers (latest log: `Layer - 2: Linear #1 done HE`), **still running** at the time of update.

**Microbench comparison (BOLT vs SUF, GELU‑like size)**:
- SUF (GPU) activation bench, bert‑base seq=128:  
  `./build/bench_suf_model --model bert-base --seq 128 --intervals 256 --degree 0 --helpers 2 --iters 5 --json`  
  → **per_gate_eval_ms = 0.995 ms** (total_eval_ms 11.944 ms for 12 layers)
- BOLT GELU microbench (CPU 2PC): **68.854 ms** for one 128×3072 GELU
- **Conclusion (microbench only)**: SUF is **much faster** (≈69× per‑gate) on this hardware.  
  Note: protocol/hardware differ (GPU FSS vs CPU HE/GC), so this is not a strict apples‑to‑apples end‑to‑end comparison.

**SUF sanity check (stability)**:
- iters=20 → per_gate_eval_ms **0.9945 ms**
- iters=100 → per_gate_eval_ms **0.9941 ms**

## D. Baseline: OpenBumbleBee
- Repo cloned at `/workspace/SUF_new/baselines/OpenBumbleBee`.
- Installed deps: gcc-11/g++-11, lld-15, conda, bazelisk, rust/cargo.
- Conda env created at `/venv/bb` (python 3.10) and `requirements-dev.txt` installed inside it.
- **Build in progress**: `bazel build -c opt examples/python/microbench:gelu` (using conda env).
