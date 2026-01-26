# new_eva.md Executed Experiments (2026-01-26)

## Selection rationale (resources + feasibility)
- The current codebase does **not expose a batch parameter** in the end‑to‑end Sigma/SUF binaries (`experiments/sigma/sigma.cu` only accepts model + seq), so **Batch scaling (D’)** is not directly executable without code changes.
- Encoder **BERT‑base seq=256** is known to fail with `cudaMemcpy invalid argument` (see earlier logs). Fixing that requires code changes (buffer sizing / alignment) and was **out of scope** for this run.
- Given available resources (2× RTX PRO 6000 Blackwell, 1.5 TiB RAM, 7 TB disk), the **decoder long‑sequence end‑to‑end** experiment is executable and directly addresses **D’’ (GPT‑2 seq=512)**.

**Chosen executable experiment**:
- **D’’ (Longer sequence scaling, decoder)**: GPT‑2 seq=512 end‑to‑end, **Sigma vs SUF**, batch=1.

## Environment
- **OS**: Ubuntu 24.04.3 LTS
- **CPU**: 2× AMD EPYC 9654 96‑Core Processor (192 cores / 384 threads)
- **RAM**: 1.5 TiB (swap 0)
- **GPU**: 2× NVIDIA RTX PRO 6000 Blackwell Workstation Edition (97,887 MiB each, sm_120)
- **Storage**: 7.0 TB total, 4.1 TB free on `/`
- **CUDA**: 13.0 (nvcc 13.0.88), driver 580.119.02

## Binaries
- **Sigma baseline**: `/workspace/SUF_new/build/gpu_mpc_upstream/sigma`
- **SUF**: `/workspace/SUF_new/build/gpu_mpc_vendor/sigma` (SUF bridge enabled)

## GPU binding
- Party‑0 → GPU0
- Party‑1 → GPU1

## Global config (both parties)
- `OMP_NUM_THREADS=32`
- `SIGMA_MEMPOOL_DISABLE=1`
- `SIGMA_PINNED_KEYBUF=0`  (avoids ~120 GB pinned host allocation for seq=512)

## SUF config (both parties)
```
SUF_SOFTMAX=1 SUF_LAYERNORM=1 SUF_ACTIVATION=1
SUF_NEXP_BITS=10 SUF_INV_BITS=10 SUF_RSQRT_BITS=9
```

## Commands (GPT‑2 seq=512)
Sigma (baseline):
```
cd /workspace/SUF_new/ezpc_upstream/GPU-MPC/experiments/sigma
mkdir -p output/P0/models output/P1/models
rm -rf output/P0/models/gpt2-512 output/P1/models/gpt2-512
SIGMA_MEMPOOL_DISABLE=1 SIGMA_PINNED_KEYBUF=0 OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=0 \
  /workspace/SUF_new/build/gpu_mpc_upstream/sigma gpt2 512 0 127.0.0.1 32 > /tmp/sigma_base_gpt2_512_p0.log 2>&1 &
SIGMA_MEMPOOL_DISABLE=1 SIGMA_PINNED_KEYBUF=0 OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=1 \
  /workspace/SUF_new/build/gpu_mpc_upstream/sigma gpt2 512 1 127.0.0.1 32 > /tmp/sigma_base_gpt2_512_p1.log 2>&1 &
wait
```

SUF (bridge enabled):
```
cd /workspace/SUF_new/third_party/EzPC_vendor/GPU-MPC/experiments/sigma
mkdir -p output/P0/models output/P1/models
rm -rf output/P0/models/gpt2-512 output/P1/models/gpt2-512
export SUF_SOFTMAX=1 SUF_LAYERNORM=1 SUF_ACTIVATION=1
export SUF_NEXP_BITS=10 SUF_INV_BITS=10 SUF_RSQRT_BITS=9
SIGMA_MEMPOOL_DISABLE=1 SIGMA_PINNED_KEYBUF=0 OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=0 \
  /workspace/SUF_new/build/gpu_mpc_vendor/sigma gpt2 512 0 127.0.0.1 32 > /tmp/suf_gpt2_512_p0.log 2>&1 &
SIGMA_MEMPOOL_DISABLE=1 SIGMA_PINNED_KEYBUF=0 OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=1 \
  /workspace/SUF_new/build/gpu_mpc_vendor/sigma gpt2 512 1 127.0.0.1 32 > /tmp/suf_gpt2_512_p1.log 2>&1 &
wait
```

## Result files
- Sigma: `ezpc_upstream/GPU-MPC/experiments/sigma/output/P0/models/gpt2-512/{dealer.txt,evaluator.txt}`
- SUF: `third_party/EzPC_vendor/GPU-MPC/experiments/sigma/output/P0/models/gpt2-512/{dealer.txt,evaluator.txt}`
- Logs: `/tmp/sigma_base_gpt2_512_p0.log`, `/tmp/sigma_base_gpt2_512_p1.log`, `/tmp/suf_gpt2_512_p0.log`, `/tmp/suf_gpt2_512_p1.log`

## Parsed results (GPT‑2, seq=512)
**Definitions**:
- Online time = `Total time` from `evaluator.txt` (party‑0)
- Comm time = `Comm time` from `evaluator.txt`
- Comm bytes = `Total Comm` from `evaluator.txt`
- Keygen time + key size from `dealer.txt`
- Throughput ≈ `seq / online_time_s` (tokens/s)

| Variant | Keygen (s) | Key size (GB) | Online time (ms) | Comm time (ms) | Comm (GB) | Throughput (tokens/s) |
|---|---:|---:|---:|---:|---:|---:|
| Sigma | 43.37 | 86.687 | 79209.84 | 19468.88 | 5.302 | 6.46 |
| SUF | 31.72 | 62.975 | 36519.43 | 23884.61 | 4.183 | 14.02 |

**Speedups (SUF vs Sigma)**:
- Online latency: **2.17×**
- Keygen time: **1.37×**
- Throughput: **2.17×**

## Notes / caveats
- GPT‑2 seq=512 requires a **large key buffer**; to avoid excessive pinned memory, `SIGMA_PINNED_KEYBUF=0` was used for both Sigma and SUF.
- Encoder long‑sequence (BERT‑base seq≥256) remains blocked by the existing `cudaMemcpy invalid argument` failure and needs buffer/size diagnostics to proceed.
- Batch scaling is not runnable without adding a batch dimension to the Sigma/SUF end‑to‑end binaries.
