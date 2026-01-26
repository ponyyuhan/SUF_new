# Baseline Status (Bumblebee & BOLT) — 2026-01-26

## Bumblebee (OpenBumbleBee) end-to-end
**Environment**: CPU-only (CUDA-enabled jaxlib not installed). SPU runtime on localhost (2PC).

### BERT (Flax BERT)
- **Status**: **Completed** (CPU + SPU).
- **Input**: dummy sentence fallback (HuggingFace `glue/cola` test split returned 404; script now falls back to a local sentence).
- **CPU runtime**: **6.453161 s**
- **SPU runtime**: **51.627365 s**
- **Log**: `/tmp/bumble_bert_e2e.log`

### GPT-2 (Flax GPT-2)
- **Status**: **Completed** (CPU + SPU).
- **CPU runtime**: **48.913096 s**
- **SPU runtime**: **287.307342 s**
- **Log**: `/tmp/bumble_gpt2_e2e.log`

**Notes / Fixes applied**:
- Added compatibility shims in:
  - `baselines/OpenBumbleBee/examples/python/ml/flax_bert/flax_bert.py`
  - `baselines/OpenBumbleBee/examples/python/ml/flax_gpt2/flax_gpt2.py`
  to handle newer JAX API changes (missing `define_bool_state`, `DeviceArray`, `linear_util`, `ShapedArray`, `KeyArray`, `default_prng_impl`).
- BERT dataset fallback to a dummy sentence if `glue/cola` fails to load.

## BOLT vs SUF comparison
**Can we compare?** **Yes, but only as a rough reference.**
- **BOLT**: CPU 2PC + HE (EzPC/SCI + SEAL), single-host 2-party run.
- **SUF**: GPU FSS with two-server preprocessing threat model.

**Current BOLT end-to-end result (BERT MRPC)**:
- End-to-end: **403.426 s (P1), 435.943 s (P2)**
- Comm (sum of per-op bytes): **26.88 GiB**
- Rounds: **123,927**

**SUF reference (BERT base, seq=128, batch=1)**:
- End-to-end: **2313.589 ms**
- Comm: **0.830 GB**

**Conclusion**: SUF is much faster and lower-comm on this hardware, but **not apples-to-apples** due to different threat models and CPU vs GPU execution.

## If SUF runs on CPU, can it be compared to BOLT?
**Currently no** — SUF codebase is GPU-centric (CUDA kernels + GPU MPC stack). A CPU version is not available in this repo. 
**If** a CPU port is implemented, then a **more fair hardware comparison** against BOLT becomes possible, but **protocol/threat-model differences** would still remain (BOLT 2PC client–server vs SUF two-server preprocessing).
