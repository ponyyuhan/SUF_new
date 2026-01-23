# SUF Ablation Report (2026-01-23)

This report follows the ablation plan in `xiaorong.md`, focusing on **attribution of SUF speedups** via four controllable factors:

- **P (Padding / fixed‑shape):** mask‑aware compilation (fixed public shape) vs mask‑dependent shape.
- **T (Template reuse / dispatch):** reuse a single compiled program vs re‑instantiating per layer.
- **B (Batching / packing):** larger packed workloads to test GPU throughput scaling.
- **K (GPU kernel engineering):** GPU secure evaluation vs CPU reference evaluation.

All experiments were run on the current server (2× RTX PRO 6000 Blackwell, CUDA 13.0).  
Unless stated otherwise (P/T/B/K sections below), model = **BERT‑base**, seq = **128**, GELU gate, intervals = **256**, degree = **0**, helpers = **2**.

Artifacts:
- `artifacts/ablation/p0_mask_unaware.json`
- `artifacts/ablation/p1_mask_aware.json`
- `artifacts/ablation/t1_template_cache.json`
- `artifacts/ablation/t0_no_cache.json`
- `artifacts/ablation/k_cpu_vs_gpu.json`
- `artifacts/ablation/batch_n1.txt`
- `artifacts/ablation/batch_n2.txt`
- `artifacts/ablation/batch_n4.txt`

---

## P: Padding / fixed‑shape vs mask‑dependent shape

We compare **mask‑aware compilation** (fixed public shapes via padded partitions) with **mask‑unaware** (shape depends on real cutpoints).  
This directly measures the overhead of enforcing shape‑independence and padding.

| Config | mask_aware | per_gate_eval_ms | total_eval_ms | total_key_bytes | pred_bytes | lut_bytes | init_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 (mask‑shape) | false | 0.264 | 3.174 | 29568 | 384 | 2080 | 0.440 |
| P1 (fixed‑shape) | true | 1.014 | 12.169 | 34176 | 768 | 2080 | 0.930 |

**Observation**
- Enforcing fixed shape increases online GELU gate cost by **~3.83×** and key size by **~1.16×**.
- The cost comes from additional predicates and padded partitions; the LUT payload size is unchanged.

---

## T: Template reuse vs per‑layer re‑instantiation

Here we isolate the overhead of **template/program re‑instantiation**.  
`T1` compiles once and reuses; `T0` re‑instantiates for each of 12 layers.

| Config | template_cache | init_ms (one program) | total_init_ms (all layers) | per_gate_eval_ms | total_eval_ms |
|---|---:|---:|---:|---:|---:|
| T1 (cached) | true | 0.903 | 0.000 | 1.013 | 12.161 |
| T0 (no cache) | false | 0.000 | 10.389 | 1.015 | 12.184 |

**Observation**
- Re‑instantiation adds **~10.39 ms** of total setup overhead for 12 layers, while online evaluation time is essentially unchanged.
- This isolates the *template/dispatch* cost from the actual gate evaluation cost.

---

## B: Packing / batching effect (secure GPU program)

To emulate runtime packing, we scale the secure gate workload by **B×** using `bench_suf_gpu` with mask‑aware compilation.

| Batch (B) | n elems | avg_ms | throughput (elem/s) |
|---:|---:|---:|---:|
| 1 | 393216 | 379.856 | 1.035e6 |
| 2 | 786432 | 722.880 | 1.088e6 |
| 4 | 1572864 | 1417.990 | 1.109e6 |

**Observation**
- Throughput improves slightly with larger packed workloads (**~7.1%** from B=1 → B=4), reflecting better GPU utilization.
- This quantifies the **packing/batching benefit** at the kernel level.

---

## K: GPU kernel engineering vs CPU reference

We compare GPU secure evaluation with CPU reference evaluation for the same gate workload.
CPU reference uses `iters=1` to keep runtime bounded; GPU timing is reported with the same setting.

| Mode | per_gate_eval_ms | total_eval_ms | speedup vs CPU |
|---|---:|---:|---:|
| GPU (secure) | 1.025 | 12.295 | 335.3× |
| CPU (ref) | 343.551 | 4122.610 | 1.0× |

**Observation**
- GPU secure evaluation provides **~335× speedup** vs CPU reference for the same gate workload.
- This captures the impact of GPU kernel engineering in SUF.

---

## Summary of Attribution

- **P (fixed‑shape padding)**: increases per‑gate compute and key size, but enables mask‑independent shapes and cacheability.
- **T (template reuse)**: reduces setup/dispatch overhead by ~10.39 ms for BERT‑base (12 layers), with negligible impact on online eval.
- **B (packing)**: improves throughput modestly with larger packed workloads.
- **K (GPU kernels)**: largest single‑factor speedup, delivering ~335× vs CPU reference on the same workload.

These ablations isolate and quantify the distinct contributors to SUF’s performance profile as requested in `xiaorong.md`.
