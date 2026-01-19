Below is a concrete, **deployable** restructuring plan that stays faithful to your paper’s core thesis (“**SUF → two non‑interactive tFSS templates (PackCmp + IntervalLUT) + uniform postproc**”), and targets the two structural blockers you’ve identified:

* **(1) Softmax round barrier / pfss.open_flushes explosion**
* **(2) key_bytes / keygen_time structural blow‑up (GapARS/StepDcf/Pred dominating)**

I’m going to organize this as **Round‑by‑Round deliverables**. Each round ends with **exact code touch points** so it is not hand‑wavy. I’ll also explicitly point out where we need a new crypto interface (multi‑point/interval DPF + prefix aggregation + vector payload correction), and I’ll only propose constructs that already exist in literature/implementations (and therefore are “landable”). For references, multi‑point DPF work and codebases exist (e.g., DMPF), incremental/prefix DPF exists (IDPF), and GPU DPF evaluation patterns exist and are public. ([Ben-Gurion University Research Portal][1])

---

## What we must change structurally (not “LUT_6~9 constant fixes”)

Right now, the big‑model default path is structurally **not** the paper:

* You are **not running “two templates per gate”** for the dominant nonlinear ops.

    * Large models default into `GeluSigmaGpuTask / NExpSigmaGpuTask / InvSigmaGpuTask / RsqrtSigmaGpuTask` and `RowMaxDiffSigmaGpuTask`.
    * Those tasks are *multi‑round state machines* (`open → wait → pfss → open → wait → mul → ...`), so flushes are fundamentally high.

* The “IntervalLUT template” that the paper narrative depends on is effectively **not real** in the default big‑model path:

    * Bench defaults to `SUF_COEFF_MODE=ppcs` (compiles away IntervalLUT).
    * The existing IntervalLUT implementation is **SIL1** (DCF/StepDCF style), and its key format **multiplies out_words into per‑level material**, and then you multiply by `intervals` again — exactly the blow‑up you summarized.

* Runtime batching/pipelining isn’t being exploited where it matters:

    * `pfss.num_jobs ≈ pfss.num_flushes` means you’re paying job granularity overhead instead of amortizing.

So to beat Sigma, we need two “big levers”:

1. **Make IntervalLUT “true” (SIL2)**: DPF/CDPF‑based, single/near‑single traversal with vector payload correction, i.e., kill the `intervals × in_bits × out_words` material multiplication that SIL1 has.

2. **Make Softmax’s repeated DReLU reduction actually batch at scale** (across blocks/heads/layers) so `open_flushes` becomes roughly `(#rounds × constant)` rather than `(#blocks × #rounds × constant)`.

The rest (adaptive limits, double buffering staging) supports these levers, but can’t replace them.

---

## Round 1: Make the runtime/compiler “SIL2‑ready” and make Softmax batchable

This round is **purely structural plumbing**. It will not magically shrink key_bytes yet, but it removes the “not landable” blockers and sets the project up so SIL2 and batched softmax can be dropped in without refactoring again.

### 1.1 Lock down “shape invariance” explicitly for IntervalLUT keys

You already pad rotated partitions to a fixed `m+1` intervals in:

* `src/compiler/suf_to_pfss.cpp` (the interval‑padding logic is already there; good)

That means SIL2 can **assume fixed `intervals=M` from the SUF description**, and never leak mask‑dependent split count.

**Action items**

* Add a comment + invariant check where interval LUT programs are built:

    * File: `src/compiler/suf_to_pfss.cpp`
    * Function: the IntervalLUT build in the `CoeffMode::kIntervalLut` branch
    * After padding, assert `cd.intervals.size() == m+1` always.
* Expose `M` to the backend in a stable way:

    * It’s already in `CoeffProgramDesc::intervals` and in the header you write.

This is necessary because SIL2 will need a fixed `M` for “vector payload correction slots” even if some are dummy.

### 1.2 Introduce an explicit IntervalLUT “version selector” in one place

You already have reserved key formats:

* `SecureIntervalLutHeaderV2` is literally annotated as **reserved for true IntervalLUT (DPF/CDPF‑based)** in `include/proto/secure_pfss_key_formats.hpp`.

**Action items**

* Add a single env knob that selects SIL1 vs SIL2:

    * `SUF_INTERVAL_LUT_VERSION=1|2`
* Implement the dispatch *only* inside `SecureGpuPfssBackend`:

    * File: `cuda/pfss_backend_gpu.cu`
    * Functions:

        * `SecureGpuPfssBackend::gen_interval_lut(...)`
        * `SecureGpuPfssBackend::eval_interval_lut_many_device(...)`
* Key parsing: read `hdr.magic` and route:

    * `"SIL1"` → current path
    * `"SIL2"` → new path (stub in Round 1; real in Round 2)

Why this matters: all higher levels (`CompositeEvalBatchBackend::eval_interval_lut_many_*`) stay unchanged.

### 1.3 Make Softmax max‑reduction batchable across blocks (the real flush lever)

Your own diagnosis is correct: the reduction has sequential dependence **across rounds**, but there is huge batchability **within a round** across:

* blocks (row blocks),
* heads,
* possibly layers (if you pipeline).

Right now, each `RowMaxDiffSigmaGpuTask` is its own little state machine; PhaseExecutor can’t easily coalesce them into one PFSS job per round.

**Structural change**: introduce a “round‑synchronous batcher” for RowMaxDiff, so per reduction round you do:

* one big open of all `diff_hat` for all blocks in this softmax phase
* one big PFSS eval DReLU for all those diffs
* one big open for `c = d ⊕ r` for all those diffs
* one big batch of Beaver muls

This does **not** reduce the theoretical round count, but it **collapses `pfss.num_jobs` and open flushes** by amortizing across blocks.

**Concrete code insertion points**

* Softmax orchestration:

    * `include/nn/softmax_block_task.hpp`
    * Today: it instantiates `RowMaxDiffSigmaGpuTask` per block.
    * Replace with:

        * `RowMaxDiffSigmaGpuRoundBatcher` (new) owned by `SoftmaxBlockTask` (or owned per “Softmax phase” object)
        * Each block registers its per-round device buffers into the batcher.

* New batcher file:

    * Create `include/nn/row_maxdiff_batch.hpp` (and `.cpp` if you prefer)
    * The batcher API should look like:

      ```cpp
      struct RowMaxDiffRoundBatcher {
        void register_block(BlockId id,
                            DeviceSpan<uint64_t> diffs_share,
                            DeviceSpan<uint64_t> r_diff_share,
                            DreluKeySpan keys,
                            // output handles: d_xor_share, etc
                            ...);
  
        Need step(PhaseRuntime& R);  // runs one reduction round across ALL registered blocks
        bool done() const;
      };
      ```

* Beaver mul batching:

    * Currently `MulTask` is per-instance; it opens internally and forces flush boundaries.
    * Introduce a `MulBatcher`:

        * file: `include/runtime/phase_tasks.hpp` or `include/runtime/mul_batch.hpp`
        * new method: `enqueue_many_mul(...)` that creates **one** open of all `(e,f)` across all multiplications in the round.

This is absolutely “landable”: it’s scheduling and buffering, not new crypto.

### 1.4 Turn “adaptive lazy limits” from a global gamble into phase-specific guardrails

You already have the framework (`AdaptiveLazyConfig` + `tuned_limits_for_phase()`), but it’s off by default and can regress due to budget+ demand flush double counting.

Instead of “just turn it on”, do this:

* Keep `SUF_PHASE_EXEC_ADAPTIVE_LIMITS=0` as global default, but:
* Add a **phase tag override** for Softmax and maybe MatMul:

    * For Softmax phases only, enable adaptive limits with **hard guardrails**:

        * Minimum payload threshold before budget flush triggers (avoid tiny budget flushes).
        * A monotonic “don’t flush twice back-to-back within K steps” rule.

**Where**

* `include/runtime/phase_executor.hpp`:

    * Extend `AdaptiveLazyConfig` to include:

        * `min_open_wire_bytes_before_budget_flush`
        * `min_pfss_hatx_words_before_budget_flush`
        * `cooldown_steps_after_budget_flush`
* `src/nn/transformer_layer.cpp`:

    * When building phases, set those knobs differently for softmax vs others.

This supports the batched RowMaxDiff batcher by letting it accumulate more before flush, while preventing “budget flush storms”.

---

## Round 2: Implement SIL2 — the true IntervalLUT template (DPF/CDPF-based, vector payload correction)

This is the core “paper alignment” and the key_bytes lever.

### 2.1 What SIL2 must compute (in your existing interface)

You already build a padded partition:

* `intervals = M` (fixed)
* Each interval has:

    * `[lo_i, hi_i)` in the **hatx domain**
    * `payload_i` = vector of `out_words` u64 (flattened coeffs)

Evaluation on input `u = hatx` must output secret shares of `payload_j` such that `u ∈ interval_j`.

SIL1 currently does:

* choose `v_last`
* compute deltas and sum `delta_i * 1[u < cut_i]` using M DCF evaluations with vector payload → **M × DCF_key_bytes**.

SIL2 goal:

* one near‑single traversal, no `M` DCFs.

### 2.2 The crypto interface we need (and it exists)

We need a **sparse / multi-point programmable primitive** plus **prefix aggregation**:

* A multi-point programmable function (DMPF / sparse DPF) exists in research and open implementations. ([Ben-Gurion University Research Portal][1])
* Prefix/Incremental DPF exists as a standard construct and is implemented publicly (IDPF). ([GitHub][2])

GPU evaluation patterns for DPF are also public and show that DPF-like tree traversals map well to GPUs. ([GitHub][3])

So we are not inventing new assumptions; we are **implementing an existing class of primitives** inside your already “reserved” SIL2 slot.

### 2.3 SIL2 design that is implementable in *your* codebase

Here is the design that fits your current compiler output and avoids mask-dependent shapes:

#### Step 1: Convert partition into a “step function via sparse deltas”

Let payload vectors be `P[0..M-1]`.

Define deltas:

* `D[0] = P[0]`
* `D[i] = P[i] - P[i-1]` for `i>=1`  (ring subtraction, per word)

Let boundary points be `B[i] = lo_i` for each interval start.
Because the partition is sorted by `lo` and padded, you have exactly `M` starts; `lo_0` should be 0 (or 0 in ring with your sentinel conventions).

Then:

* `P[j] = Σ_{i=0..j} D[i]` (prefix sum of deltas)

So IntervalLUT reduces to:

* find `j = rank(u)` such that `u` is in interval j
* output prefix sum up to j

#### Step 2: Build a **sparse point function** over domain `[0, 2^{in_bits})`

Define sparse function `G(x)`:

* `G(B[i]) = D[i]` for i in [0..M-1]
* `G(x) = 0` otherwise

Then desired output is:

* `Out(u) = PrefixSum(G, u)` but **not over all x**: we want sum of deltas whose boundary is <= u (monotone).

So we need a primitive that allows:

* programming **M points** with vector payloads
* evaluating **prefix aggregate** at `u` in one/near-one traversal

This is exactly where “multi-point/interval DPF + prefix aggregation + vector payload correction” lands.

### 2.4 SIL2 key format in your repo

Use your reserved header:

* File: `include/proto/secure_pfss_key_formats.hpp`
* Struct: `SecureIntervalLutHeaderV2` (magic `"SIL2"`)

**Proposed blob layout (per party key)**

```
[SecureIntervalLutHeaderV2  (packed)]
[base_share: out_words*u64]     // share of P[0] or equivalently D[0]
[SIL2_core_key_bytes]           // the sparse/prefix-DPF key
[optional: padding to key_bytes]
```

Where `intervals=M` is fixed and public (from the SUF desc), and `SIL2_core_key_bytes` is fixed given `(in_bits, M, out_words)`.

### 2.5 Implementation plan (code-level)

#### (A) CPU reference implementation first (correctness + key size accounting)

You need a CPU reference to validate correctness before GPU.

* New files:

    * `include/proto/sil2_ref.hpp`
    * `src/proto/sil2_ref.cpp`

Expose:

```cpp
std::pair<std::vector<uint8_t>, std::vector<uint8_t>>
gen_sil2_keypair(const IntervalPayload* intervals, size_t M,
                 int in_bits, int out_words, PrgSeed seed);

void eval_sil2_cpu(uint64_t u_hat, const uint8_t* key_bytes,
                   uint64_t* out_words);
```

This CPU ref can be slow; it’s purely for correctness tests.

#### (B) GPU implementation

* Add GPU kernel:

    * File: `cuda/pfss_kernels.cu`
    * Kernel: `interval_lut_sil2_many_kernel(...)`

It must:

* operate SoA where possible

* treat `in_bits` <= 64

* output `out_words` u64 per element

* Add backend hooks:

    * File: `cuda/pfss_backend_gpu.cu`
    * Functions:

        * `SecureGpuPfssBackend::gen_interval_lut(...)` → when SIL2, call new SIL2 keygen
        * `SecureGpuPfssBackend::eval_interval_lut_many_device(...)` → when SIL2, call new kernel

* Keep SIL1 code untouched.

#### (C) Tests

Add a deterministic test that compares SIL1 and SIL2 outputs on random partitions and random u:

* `src/tests/test_interval_lut_sil2.cpp`
* Cases:

    * random payloads, random u
    * include padded dummy split scenario (your compiler can create it)
    * include full-ring interval

This is critical because SIL2 will be security-sensitive and easy to “almost get right” while being wrong.

---

## Round 3: Make the paper path dominant end-to-end (disable sigma_gpu defaults)

Once SIL2 exists and is faster/cheaper, you can flip the system back into the paper’s execution shape:

### 3.1 Switch large-model defaults back to SUF templates

* File: `src/demo/bench_suf_transformer.cpp`
* Today it sets:

    * `SUF_GELU_MODE=sigma_gpu`, `SUF_NEXP_MODE=sigma_gpu`, etc.
    * `SUF_COEFF_MODE=ppcs`

New default strategy:

* If `SUF_INTERVAL_LUT_VERSION==2` and GPU backend supports SIL2:

    * set `SUF_COEFF_MODE=interval_lut`
    * set all *_MODE to the SUF composite path (i.e., do **not** force sigma_gpu)

### 3.2 Recompile helper gates (Trunc/ARS) away from StepDcf where possible

Your own stats show `GapARS/StepDcf/Pred` dominating key_bytes.

After SIL2 lands, the correct move is:

* reduce reliance on `CoeffMode::kStepDcf` for helper gates that can be represented as piecewise polynomials in hatx
* let the compiler push them into IntervalLUT payload selection + uniform postproc

Where:

* `src/compiler/suf_to_pfss.cpp`
* In `build_coeff_step(...)`, add a policy:

    * if degree/piece structure exists and `interval_lut_v2` is available, prefer interval_lut

This is how you convert that GB-scale StepDcf material into **one SIL2 + one PackCmp**, matching the paper’s story.

---

## Why this is the “right” structural bet vs Sigma

Sigma beats you today because:

* it is already a “tight set of primitives + good batching” system (DPF/DCFs, max reduction, etc.)
* while your default big-model path is a **hybrid**: SUF IR exists, but the heavy ops are still sigma-like state machines + SIL1/StepDcf material blow-ups.

SIL2 + round-synchronous batching moves you back to:

* **one template call** for coefficient selection (IntervalLUT), instead of StepDcf forests
* **packaged predicate extraction** (PackCmp)
* and runtime that can finally amortize PFSS jobs, in the way your paper narrative requires.

Multi-point functions and incremental/prefix DPF aren’t speculative; they exist as constructs and public implementations. ([Ben-Gurion University Research Portal][1])
GPU DPF evaluation patterns are also real and already demonstrated. ([GitHub][3])

---

## Immediate “Round 1” checklist you can implement right now in this repo

I’m listing these as exact edits so it’s actionable:

1. **Add IntervalLUT version knob + dispatch**

* `cuda/pfss_backend_gpu.cu`

    * dispatch on `hdr.magic` and/or `SUF_INTERVAL_LUT_VERSION`
* `include/proto/secure_pfss_key_formats.hpp`

    * ensure `"SIL2"` header is documented as active, not just reserved

2. **Add RowMaxDiff round batcher skeleton**

* Create `include/nn/row_maxdiff_batch.hpp`
* Modify `include/nn/softmax_block_task.hpp` to instantiate batcher and register blocks
* Add minimal `MulBatcher` for round-level Beaver opens (even if initially it just wraps existing MulTask calls, the shape is important)

3. **Phase-specific adaptive limits guardrails**

* `include/runtime/phase_executor.hpp`

    * add per-phase knobs: min payload threshold + cooldown
* `src/nn/transformer_layer.cpp`

    * enable adaptive limits only for Softmax phase initially (not globally)

4. **Add SIL2 CPU reference + tests (even before GPU)**

* `src/tests/test_interval_lut_sil2.cpp` with SIL1-vs-SIL2 equivalence on random instances

That’s the fastest way to “make it landable”: once this scaffolding is in, Round 2 is “just” implementing the SIL2 core and GPU kernel without re-threading the whole system again.

---

If you want the next step in the next message, I’ll go one level deeper and propose a concrete SIL2 core-key algorithm that fits your current DPF/ChaCha primitives and your “key shape must not depend on sampled masks” constraint, including an explicit byte layout for `SIL2_core_key_bytes` and how to implement its GPU kernel in the same style as your existing `drelu1_kernel_keyed` / SoA packcmp kernels.

[1]: https://cris.bgu.ac.il/en/publications/improved-constructions-for-distributed-multi-point-functions-2/ "
        Improved Constructions for Distributed Multi-Point Functions
      \-  Ben-Gurion University Research Portal"
[2]: https://github.com/google/distributed_point_functions "GitHub - google/distributed_point_functions: An Implementation of Incremental Distributed Point Functions in C++"
[3]: https://github.com/facebookresearch/GPU-DPF "GitHub - facebookresearch/GPU-DPF: GPU-based Distributed Point Functions (DPF) and 2-server private information retrieval (PIR)."
