# SUF Ablation Report (2026-01-21)

This report follows the ablation plan in `xiaorong.md`, focusing on **attribution of SUF speedups** via four controllable factors:

- **P (Padding / fixed-shape):** mask-aware compilation (fixed public shape) vs mask-dependent shape.
- **T (Template reuse / dispatch):** reuse a single compiled program vs re-instantiating per layer.
- **B (Batching / packing):** larger packed workloads to test GPU throughput scaling.
- **K (GPU kernel engineering):** GPU secure evaluation vs CPU reference evaluation.

All experiments were run on the current server (2× RTX 5090, CUDA 13.0).  
Unless stated otherwise (P/T/B/K sections below), model = **BERT-base**, seq = **128**, GELU gate, intervals = **256**, degree = **0**, helpers = **2**.

Artifacts:
- `artifacts/ablation/suf_ablation_p0.json`
- `artifacts/ablation/suf_ablation_p1.json`
- `artifacts/ablation/suf_ablation_t0.json`
- `artifacts/ablation/suf_ablation_k.json`
- `artifacts/ablation/batch_n1.txt`
- `artifacts/ablation/batch_n2.txt`
- `artifacts/ablation/batch_n4.txt`
- `artifacts/ablation/suf_vs_non_suf.csv`
- `artifacts/ablation/suf_vs_non_suf.json`

---

## P: Padding / fixed-shape vs mask-dependent shape

We compare **mask-aware compilation** (fixed public shapes via padded partitions) with **mask-unaware** (shape depends on real cutpoints).  
This directly measures the overhead of enforcing shape-independence and padding.

| Config | mask_aware | per_gate_eval_ms | total_eval_ms | total_key_bytes | pred_bytes | lut_bytes | init_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 (mask-shape) | false | 0.288 | 3.453 | 29568 | 384 | 2080 | 0.174 |
| P1 (fixed-shape) | true | 1.098 | 13.179 | 34176 | 768 | 2080 | 0.187 |

**Observation**
- Enforcing fixed shape increases online GELU gate cost by **~3.82×** and key size by **~1.16×**.
- The cost comes from additional predicates and padded partitions; the LUT payload size is unchanged.

---

## T: Template reuse vs per-layer re-instantiation

Here we isolate the overhead of **template/program re-instantiation**.  
`T1` compiles once and reuses; `T0` re-instantiates for each of 12 layers.

| Config | template_cache | init_ms (one program) | total_init_ms (all layers) | per_gate_eval_ms | total_eval_ms |
|---|---:|---:|---:|---:|---:|
| T1 (cached) | true | 0.187 | 0.187 | 1.098 | 13.179 |
| T0 (no cache) | false | 0.000 | 3.364 | 1.101 | 13.212 |

**Observation**
- Re-instantiation adds **~3.36 ms** of total setup overhead for 12 layers, while online evaluation time is essentially unchanged.
- This isolates the *template/dispatch* cost from the actual gate evaluation cost.

---

## B: Packing / batching effect (secure GPU program)

To emulate runtime packing, we scale the secure gate workload by **B×** using `bench_suf_gpu` with mask-aware compilation.

| Batch (B) | n elems | avg_ms | throughput (elem/s) |
|---:|---:|---:|---:|
| 1 | 393216 | 412.546 | 953145.00 |
| 2 | 786432 | 786.771 | 999569.00 |
| 4 | 1572864 | 1533.470 | 1025690.00 |

**Observation**
- Throughput improves slightly with larger packed workloads (**~7.6%** from B=1 → B=4), reflecting better GPU utilization.
- This quantifies the **packing/batching benefit** at the kernel level.

---

## K: GPU kernel engineering vs CPU reference

We compare GPU secure evaluation with CPU reference evaluation for the same gate workload.
CPU reference uses `iters=1` to keep runtime bounded; GPU timing is reported with the same setting.

| Mode | per_gate_eval_ms | total_eval_ms | speedup vs CPU |
|---|---:|---:|---:|
| GPU (secure) | 1.107 | 13.281 | 33.6× |
| CPU (ref) | 37.234 | 446.814 | 1.0× |

**Observation**
- GPU secure evaluation provides **~33.6× speedup** vs CPU reference for the same gate workload.
- This captures the impact of GPU kernel engineering in SUF.

---

---

## Summary of Attribution

- **P (fixed-shape padding)**: increases per-gate compute and key size, but enables mask-independent shapes and cacheability.
- **T (template reuse)**: reduces setup/dispatch overhead by ~3.36 ms for BERT-base (12 layers), with negligible impact on online eval.
- **B (packing)**: improves throughput modestly with larger packed workloads.
- **K (GPU kernels)**: largest single-factor speedup, delivering ~33× vs CPU reference on the same workload.

These ablations isolate and quantify the distinct contributors to SUF’s performance profile as requested in `xiaorong.md`.
