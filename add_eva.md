According to your current evaluation notes, the three missing pieces are: (1) a **block-level benchmark** (`bench_softmax_norm`), (2) an **extensibility case study**, and (3) a **batch-size sweep** (not supported by the current Sigma harness CLI).

Below are **complete, ICML-style experimental plans** for these three additions.

---

## Experiment 1: Block-level Benchmark for Softmax and LayerNorm

### Goal and claim

Demonstrate that SUF’s advantages are **not only end-to-end**, but are **most pronounced exactly where SUF applies**: fixed-point nonlinear blocks and helper kernels. Specifically:

* SUF reduces **online latency** and **online communication** for **Softmax** and **LayerNorm** blocks.
* SUF reduces **preprocessing material** (keys + correlated randomness) for these blocks.
* SUF provides better **GPU batching efficiency** and **cleaner breakdown**: “≤ 2 tFSS calls + uniform post-processing”.

This directly strengthens the paper’s core story (typed IR + compilation → fewer bespoke stages) by isolating the nonlinear-heavy regions.

### Workloads (what to benchmark)

Benchmark two blocks used in standard transformer inference:

1. **Attention Softmax block**

    * Input: attention scores `S` shaped `(B, H, L, L)`.
    * Softmax is applied along the last dimension (length `L`) for each `(B,H,L)` row.
    * Use typical configs:

        * BERT-base-like: `H=12`, `L∈{32,64,128,256}` (if memory allows), `B=1` for this experiment.
        * Also run GPT-2-like (`H=12`, same `L`) to show consistency.

2. **LayerNorm block**

    * Input: hidden activations `X` shaped `(B, L, d_model)`.
    * LayerNorm reduces over `d_model` and applies affine parameters (public).
    * Use typical configs:

        * BERT-base: `d_model=768`, `L∈{32,64,128,256}`, `B=1`.
        * BERT-large: `d_model=1024`, `L∈{32,64,128}` (optional).

### Baselines

* **Sigma**: block implementations from the Sigma GPU codebase (your `ezpc_upstream/GPU-MPC`).
* **SUF**: your SUF-enabled build (your `EzPC_vendor` copy).
* **SHAFT (optional, for Softmax only)**: run SHAFT’s `unit-test` softmax script for `L∈{32,64,128,256}` and report its time/bytes/rounds as context (it is already structured for this block). This is optional because SHAFT doesn’t provide a clean LayerNorm unit test in the same style in many setups.

### Metrics (report all)

For each `(block, model-shape, L)`:

* **Online latency (ms)**: wall-clock end-to-end for the block execution in the 2-party setting.
* **Online communication (bytes/MB/GB)**: total bytes exchanged.
* **# rounds** (recommended): if your comm layer can count message flushes; otherwise state “not available”.
* **Preprocessing size**:

    * Total generated key material size on disk (per party) for that block.
    * Optional: breakdown into `PackCmp` and `IntervalLUT` key bytes for SUF.
* **GPU kernel breakdown** (high value for ICML):

    * time spent in predicate extraction (PackCmp),
    * time spent in coefficient lookup (IntervalLUT),
    * time spent in post-processing (Horner + AND/B2A).

### Experimental protocol (how to run fairly)

* Fix all SUF approximation settings to match end-to-end evaluation settings (e.g., same nExp/inv/rsqrt parameters).
* Use identical:

    * bitwidth `n`, fractional bits `f`,
    * truncation rules,
    * clipping/range-reduction (if applicable).
* Run **warm-up** (e.g., 5 iterations) and then **measure** over 30 iterations.
* Report **median** and **p10–p90** (or mean±std if you prefer).
* Pin each party to a separate GPU (same as your main eval).

### Implementation plan (what Codex should build)

**Deliverable 1: a new benchmark binary (or script)**
Create a new benchmark target, e.g.

* `GPU-MPC/benchmarks/bench_softmax_norm.cu` (C++/CUDA), or
* `GPU-MPC/experiments/sigma/bench_softmax_norm.sh` calling existing internal kernels.

The benchmark should support:

* `--block softmax|layernorm`
* `--L`, `--H`, `--d_model`
* `--batch B`
* `--iters N`
* `--warmup W`
* `--mode sigma|suf` (or build two binaries and run both)

**Deliverable 2: measurement hooks**
Codex should:

* add a timing utility (CUDA events + CPU wall clock),
* add comm byte counters (reuse whatever your harness uses for end-to-end).

**Deliverable 3: structured output**
Make the bench print JSON lines like:

```json
{
  "block":"softmax",
  "backend":"suf",
  "B":1,"H":12,"L":128,"d_model":768,
  "online_ms":72.3,
  "comm_bytes":2490368,
  "key_bytes_party0":..., "key_bytes_party1":...,
  "breakdown_ms":{"packcmp":..., "lut":..., "post":...}
}
```

**Deliverable 4: plotting**
Add a Python script `scripts/plot_block_bench.py` that outputs:

* latency vs L (two curves: Sigma, SUF),
* comm vs L,
* preprocessing size vs L (if block preprocessing depends on L).

### Expected analysis write-up (what this experiment should show)

* Softmax/LayerNorm are “nonlinear-dense”; SUF should show a larger speedup than end-to-end in many cases.
* Breakdown should show SUF concentrates work into:

    * one batched predicate extraction,
    * one payload lookup,
    * uniform share-based post-processing.

---

## Experiment 2: Extensibility Case Study — Add a New Nonlinearity via SUF

### Goal and claim

Make extensibility a **measurable** result:

* Add a **new scalar nonlinearity** (not already supported as a specialized protocol) by writing only a **SUF descriptor**, with no bespoke masked-predicate protocol design.
* Show that the new gate:

    1. is correct (matches fixed-point reference),
    2. fits the “≤2 template evaluations” pattern,
    3. has reasonable performance and preprocessing cost,
    4. preserves **mask-independent public shape**.

This is the “compiler/IR value” experiment ICML reviewers look for.

### Choose the case-study function

Pick a function that is common in modern transformers but often missing in older secure inference stacks:

**Recommended: Sigmoid (for GLU/SwiGLU gating).**

* Used in gating variants and easy to motivate.
* Has saturating behavior (needs piecewise handling), so it’s a good test for SUF.

Alternative (if you want a second one): `tanh`.

### What to implement

1. **SUF descriptor for Sigmoid**

    * Partition canonical input range into intervals (e.g., saturate outside a bounded range).
    * For each interval, approximate sigmoid with a low-degree polynomial over `R`.
    * Include helper bits needed for fixed-point semantics if your implementation requires them (e.g., sign/MSB predicates, low-bit predicates if you do special rounding).

2. **A “naive LUT baseline” (for this experiment only)**
   To make the experiment stronger, include an internal baseline:

    * implement sigmoid as a **pure lookup table** (piecewise constant, degree 0) using the same `IntervalLUT` template, with many intervals (e.g., 256 or 512 buckets).
      This gives a clear tradeoff curve:
    * polynomial SUF: fewer intervals, smaller keys, fewer bytes
    * LUT: bigger keys/bytes but potentially higher accuracy

This is not Sigma/SHAFT, but it provides a meaningful *within-framework* baseline for the extensibility story when external baselines don’t support the primitive.

### Baselines

* **SUF-Sigmoid (poly)**: your proposed SUF descriptor.
* **SUF-Sigmoid (LUT-only)**: a descriptor variant that uses only interval lookup (degree 0).
* **Sigma**: if Sigma does not implement sigmoid, explicitly report “not supported as a built-in primitive”; optionally implement a *composition baseline* if feasible (e.g., approximate sigmoid using existing Sigma primitives), but this is not required if it becomes a large engineering project.
* **SHAFT**: optional if SHAFT provides a secure sigmoid primitive; if not, report “not available”.

### Metrics

Report three categories:

**A. Developer effort / extensibility**

* Files added/modified.
* Lines of code added (descriptor + plumbing).
* Whether any backend/FSS code changed (should be “no” for SUF).

**B. Correctness / numerical quality**

* Compare against:

    * floating-point PyTorch sigmoid (ground truth),
    * and/or a fixed-point reference implementation (your emulation path).
* Report:

    * max absolute error,
    * mean absolute error,
    * maybe cosine similarity on a representative tensor.

**C. Performance / cost**
For a representative tensor shape:

* For gating: `(B, L, 4*d_model)` typical MLP intermediate.

    * Example: BERT-base-like: `(1, 128, 3072)` or `(1,128,4096)`.
* Measure:

    * online latency,
    * online comm,
    * preprocessing size (keys),
    * emitted shape parameters `(T, {k_t}, M, p)`.

### Additional “mask-shape invariance” sub-test (highly recommended)

This is particularly aligned with your paper’s security story:

* Generate keys for the same sigmoid gate **K times** (e.g., K=50) with different random masks.
* Check that:

    * key sizes are identical,
    * emitted template shapes `(T, {k_t}, M, p)` are identical,
    * any padding logic is working as intended.

This turns a “the compiler enforces mask-independent shape” claim into an empirical validation.

### Implementation plan (Codex task list)

**Deliverable 1: a sigmoid descriptor format**
If your SUF descriptors are coded as C++ structs or JSON, implement:

* `descriptors/sigmoid_{poly,lut}.(json|h|cc)`

**Deliverable 2: coefficient generation tool**
If you don’t already have one:

* add `tools/fit_univariate_poly.py` that:

    * fits piecewise polynomials on a bounded range,
    * exports integer coefficients in `R`,
    * prints evaluation error.

**Deliverable 3: unit test**
Add `tests/test_suf_sigmoid.(py|cc)`:

* generate random inputs,
* run (a) reference (float or fixed-point), (b) SUF emulation, (c) MPC (optional),
* assert error bounds.

**Deliverable 4: microbench**
Add `scripts/bench_sigmoid.py`:

* runs the compiled sigmoid gate repeatedly,
* reports latency/comm/key bytes.

**Deliverable 5: invariance check**
Add `scripts/check_shape_invariance.py`:

* runs keygen K times,
* asserts constant sizes and shapes.

### Expected analysis write-up

* “Adding sigmoid required only writing a SUF descriptor (no new cryptographic protocol).”
* Show accuracy numbers and performance.
* Show polynomial-vs-LUT tradeoff.
* Show mask-independent shapes empirically.

---

## Experiment 3: Batch-size Sweep — Latency/Throughput vs Batch

### Goal and claim

ICML reviewers often expect throughput scaling:

* Show how SUF scales from interactive **low-latency** inference (`B=1`) to higher-throughput batching (`B>1`).
* Demonstrate that SUF’s improvements persist (or improve) with batching due to:

    * fewer template invocations,
    * better GPU amortization,
    * reduced per-element overhead.

This also addresses the limitation you noted: batch-size sweep is missing because Sigma CLI doesn’t support it yet.

### Workloads

Pick 1–2 representative end-to-end models to keep runtime manageable:

* **BERT-base, seq=128**
* **GPT-2, seq=128** (optional but recommended)

Batch sizes:

* `B ∈ {1,2,4,8,16}` (stop earlier if memory limits)

### Baselines

* **Sigma** end-to-end inference (same model/seq).
* **SUF** end-to-end inference (same model/seq).
* **SHAFT** (optional): only if it supports batching and runtime is feasible; otherwise omit and note.

### Metrics

For each `(model, B)`:

* Online latency (ms)
* Online communication (GB)
* **Throughput**:

    * sequences/sec, and/or tokens/sec
    * tokens/sec = `(B * seq_len) / (online_time_sec)`
* Preprocessing size:

    * total key bytes per party
    * optionally per-inference amortized key bytes (divide by B for reporting clarity)

Also report GPU memory usage if you can query it (optional but nice).

### Key design detail: what exactly does “batch” mean?

To be meaningful, batch-size must run **concurrently on GPU**:

* the batch dimension must be part of tensor shapes and kernel launches
* not “run B times sequentially”

### Implementation plan (Codex task list)

**Step 1: extend Sigma harness to accept `--batch`**

* Add CLI flag `--batch B` in the Sigma end-to-end driver.
* Plumb `B` into:

    * input tensor allocation,
    * embedding, attention, MLP, softmax, layernorm shapes,
    * any loops that assume `B=1`.

**Step 2: extend SUF harness similarly**

* Ensure SUF env flags still work:

    * `SUF_SOFTMAX=1 SUF_LAYERNORM=1 SUF_ACTIVATION=1`
* Ensure preprocessing/keygen scales with batch: total number of gate instances increases ∝ B.

**Step 3: logging**
Make both Sigma and SUF print:

* `B`, `seq`, `online_ms`, `comm_bytes`, `key_bytes`

**Step 4: run script**
Add:

* `scripts/run_batch_sweep.sh`
* outputs a CSV:

    * `model,B,backend,online_ms,comm_gb,key_gb,throughput_tok_s`

**Step 5: plot**
Add:

* `scripts/plot_batch_sweep.py` producing:

    1. latency vs batch (expect increasing, but sublinear ideally),
    2. throughput vs batch (expect increasing, saturating),
    3. comm per token vs batch.

### Experimental protocol

* Fix seq length and approximation parameters.
* Run each configuration ≥ 5 times; report median.
* Warm-up each run with one dummy inference to stabilize GPU clocks.

### Expected analysis write-up

* SUF should keep a consistent advantage vs Sigma across batch sizes.
* Throughput curves: SUF should reach higher throughput earlier, especially for nonlinear-heavy regimes.
* If speedup decreases at large batch due to linear layers dominating, explicitly explain it (this is acceptable if framed correctly).

---

If you want, I can also turn these three plans into “Evaluation section subsections” (ready-to-paste LaTeX prose + table/figure suggestions), but the above is the complete design + implementation checklist for Codex to execute.
