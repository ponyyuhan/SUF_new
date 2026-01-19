the next steps that are both:

* **consistent with your paper’s core idea** (typed IR + compiler emitting standard templates + shape‑safe),
* **and definitely implementable** (no new primitives like CDPF / seed‑homomorphic PRG),

are:

1. **Make packed pred actually fast** by fixing the key layout / staging for SPC1 PackCmp (your stated root cause).
2. **Collapse flushes** by changing buffering + scheduling policy (not cryptography).
3. **Overlap open communication with PFSS GPU work** (again scheduling).

Below is a concrete plan you can execute in your repo with minimal “conceptual” changes.

---

# Part A — Make `SUF_TRUNC_PACKED_PRED=1` a win (fix SPC1 PackCmp key layout)

You already diagnosed the truth: **AoS-per-element packing** causes the GPU to read keys with huge stride → uncoalesced → bandwidth collapse → packed mode loses.

This is fixable without touching security or SUF: it’s purely **key blob layout** + **staging** + **kernel indexing**.

## A1) Decide which dimension your warp maps to (and layout to match)

Your SPC1 “fused path” is now `N * 32` threads. In practice it’s almost certainly:

* 1 warp processes **one (element, chunk-of-32-queries)**, where lane = query-in-chunk.

If that is true (and your description strongly suggests it), then the right layout is:

> **Level-major (SoA-by-query) inside each element**, i.e. for each level `ℓ`, store the correction words for queries `[q0..q31]` contiguously.

This way, at level `ℓ`, lane 0..31 loads `cw[ℓ][q]` from **contiguous** addresses → coalesced.

### What you likely have today (bad)

```
[element][query][level]   // query-major: each query owns its whole CW array
```

At a fixed level, lanes read addresses spaced by `sizeof(one_query_key)` → huge stride.

### What you want (good)

```
[element][level][query]   // level-major: at each level, CWs for all queries contiguous
```

If (in some places) your kernel maps lane = element and iterates queries sequentially, then you’d instead want **query-major across elements** (true SoA across elements). But given you already went to `N*32`, the common case is lane=query. So implement `element→level→query`.

## A2) Implement a “transposed key layout v2” for SPC1 PackCmp

### File touch points

* `include/proto/secure_pfss_key_formats.hpp`
* wherever SPC1 PackCmp keys are produced/serialized for secure GPU (likely in keygen material builders)
* `cuda/pfss_kernels.cu` around your SPC1 eval kernel (`:1735`)
* `cuda/pfss_backend_gpu.cu` around staging/launch (`:2610`, `:4095`)

### Key format design (concrete)

Define a device-facing key blob layout like:

```cpp
// All offsets are in bytes from base pointer.
// Everything aligned to 16 bytes (important for vectorized loads).
struct Spc1PackCmpKeyV2Header {
  uint16_t num_bits;      // <= 255
  uint16_t num_queries;   // total queries in this group (descriptor-fixed)
  uint16_t queries_per_chunk; // usually 32
  uint16_t num_chunks;    // ceil(num_queries / 32)
  uint16_t out_words;     // ceil(num_queries / 64)  (or per chunk, if you emit chunk outputs)
  uint16_t reserved;

  uint32_t off_root;      // roots for all queries/chunks (if per-query roots exist)
  uint32_t off_cw;        // correction words block (transposed)
  uint32_t off_aux;       // optional (view shifts / metadata), if needed
};
```

Then for the correction words block, store:

* For each `chunk` (32 queries):
* For each level `ℓ in [0..num_bits-1]`:

    * a packed structure per lane/query:

        * seed correction material (whatever your SPC1 uses)
        * t-bit correction (if present)
        * output correction for multiword mask (if SPC1 needs it per level; if it only needs at leaf, store separately)

In memory:

```
cw_base +
  chunk_id * (num_bits * 32 * CW_STRIDE)
+ level     * (32 * CW_STRIDE)
+ lane      * CW_STRIDE
```

Where `CW_STRIDE` is aligned (16 or 32 bytes). You want `CW_STRIDE` small and aligned.

### Why this is safe w.r.t your paper

* It does not change the template semantics (still PackCmp).
* No new leakage: public shape is still (T, {k_t}).
* No use of plaintext thresholds.
* Only the internal storage layout changes.

## A3) Change key generation / packing to output V2 directly (avoid runtime transpose)

Do **not** “copy AoS then transpose on GPU” if you can avoid it. That would add online overhead.

Because preprocessing is already offline, the best path is:

* Dealer generates keys as usual per query,
* When serializing to the per-element material blob, write CWs into V2 transposed positions.

Where to implement:

* The code that currently builds the “secure GPU material” for PackCmp groups (likely near your other material generation for sigma_gpu gates, but for generic PackCmp). If you don’t have a single spot, add a helper in the same file that defines the key format (`secure_pfss_key_formats.hpp`) so both CPU and GPU share layout.

## A4) Update the kernel to use vectorized loads in the transposed layout

In `cuda/pfss_kernels.cu` (SPC1 kernel):

* At each level, each lane loads its CW struct:

    * use `uint4` / `ulonglong2` loads if CW_STRIDE >= 16
    * keep CW in registers
* Avoid per-level pointer chasing (compute a single base pointer per chunk + level, then lane offset)

This one detail often makes the difference between “packed pred 2× slower” and “packed pred 2× faster”.

## A5) Re-enable trunc packed pred only after layout fix

Once V2 layout is in, re-test:

* `SUF_TRUNC_PACKED_PRED=1` should **reduce**:

    * `pfss_flush_eval_eval_ns`
    * possibly `pfss.num_flushes` (because more work per PFSS job group)
* You should also see fewer PFSS key bytes for trunc predicates if SPC1 truly packs multiple thresholds into fewer evals.

**Concrete success criteria** (for this specific step):

* A microbench that runs only the trunc predicate extraction for a large tensor should show:

    * packed pred throughput ≥ non-packed throughput
    * GPU global load efficiency improved (Nsight: gld transactions/req down, L2 hit up)

---

# Part B — Kill the real wall-time killer: too many flushes

Even if PFSS kernels are optimal, your counters show you’re paying fixed costs 100–300 times.

You don’t need new crypto to fix this; you need a different scheduling policy.

## B1) Increase buffer capacities so “BUFFER_FULL” flushes disappear

Flush counts that high often come from staging buffers being too small.

### PFSS side

In `cuda/pfss_backend_gpu.cu`:

* identify the host pinned staging buffer size and device key arena size.
* increase them aggressively (start with **32–128MB** per PFSS family).

Make them configurable:

* `SUF_PFSS_STAGING_MB=...`
* `SUF_PFSS_DEVICE_ARENA_MB=...`

If you currently allocate per flush, switch to persistent buffers.

### Open side

In your OpenCollector implementation:

* increase open pack buffers similarly (e.g., **16–64MB**), configurable:

    * `SUF_OPEN_STAGING_MB=...`

This alone can reduce open flushes by 2–4× if you were flushing due to size.

## B2) Add “flush reason” counters (you can’t optimize blind)

You already have `open_flushes` and `pfss.num_flushes`. Add per-reason breakdown:

* `AWAIT_DEPENDENCY`
* `BUFFER_FULL`
* `EXPLICIT_TASK_FLUSH`
* `SHAPE_SWITCH` (if you flush when switching kernel variants)

This is implementable as a few counters incremented at the call sites that trigger flush.

Goal:

* After B1, **BUFFER_FULL should be near 0**.
* Remaining flushes should be only **AWAIT_DEPENDENCY**, which you reduce via scheduling (B3/B4).

## B3) Stop flushing from inside leaf tasks: convert to enqueue/await discipline

Right now your flush counts are telling you: tasks are still frequently “forcing progress”.

The implementable change (without rewriting everything into a full DAG scheduler) is:

* Introduce a “scope” that defers flushes until the end of a block/layer.

### Concrete mechanism

Add a lightweight RAII guard to the runtime executor, e.g.:

```cpp
struct ScopedDeferFlush {
  PhaseExecutor& ex;
  ScopedDeferFlush(PhaseExecutor& ex) : ex(ex) { ex.defer_flush_begin(); }
  ~ScopedDeferFlush() { ex.defer_flush_end(); }
};
```

When `defer_flush` is active:

* `open.request(...)` and `pfss.request(...)` only enqueue
* any “flush_if_needed” becomes a no-op unless buffer would overflow

Then place `ScopedDeferFlush` around:

* the whole MLP block
* the whole attention/softmax block
* or at least around “all SUF scalar gates in the layer”

Where to wire:

* `src/nn/mlp_block.cpp`
* softmax/attention block file(s)

At the end of the scope call:

* `ex.flush_all()` once (or a small fixed sequence: open→pfss→beaver)

This typically drops flushes massively with minimal code churn.

## B4) Coalesce opens: one open for all hatx in a block (and don’t await early)

Open is expensive mainly because it forces comm sync.

Implement these two rules:

1. **Do not await an open handle until you truly need the value.**
2. **When you do need it, await a big batch, not a single buffer.**

Concretely:

* In `include/runtime/phase_tasks.hpp`, audit any place that does:

    * enqueue open → immediate flush/await → then enqueue pfss
* Replace with:

    * enqueue open(s) for all hatx first
    * flush opens once
    * then enqueue all pfss jobs that depend on those opens
    * flush pfss once

This aligns with your paper: online reveals masked openings anyway; batching doesn’t change leakage.

---

# Part C — Overlap open communication with PFSS GPU work (hide `open_comm_ns`)

`open_comm_ns ≈ 0.116s` is large. If you can overlap most of it under PFSS eval + post-processing, you can cut wall time significantly.

This is implementable if your OpenCollector supports “in-flight flush” and doesn’t block the main thread.

## C1) Make OpenCollector support multiple in-flight flush contexts

You previously hit a bug where OpenCollector retained only the most recent flush buffer. Fixing this properly unlocks overlap.

Implement OpenCollector as a queue of contexts:

```cpp
struct OpenFlushCtx {
  HostPinnedBuffer send, recv;
  DeviceBuffer pack_dev, scatter_dev;
  cudaEvent_t pack_done, scatter_done;
  std::promise<void> comm_done;
  // mapping from handle -> offset in recv buffer
};
```

Then:

* `open.flush_async()` launches GPU pack (async), starts comm on a separate thread after pack_done, and returns immediately.
* Later `open.await_all()` waits for comm_done then launches scatter to GPU.

This lets you do:

* start open comm
* run PFSS kernels while comm is in flight
* scatter results later

## C2) Scheduling rule: start opens as early as possible

Within a block:

* enqueue opens for the *next* set of hatx as soon as the corresponding shares exist
* flush async
* continue scheduling other work (PFSS for previous opens, Horner, etc.)

Even if you don’t fully pipeline across layers, pipelining within a block can hide tens of ms.

---

# Part D — Small but real: compiler-level atom sharing that is descriptor-safe

You’re already careful about “no mask-dependent dedup”. You can still do **descriptor-safe CSE**:

### Key idea

In masked rewrite:
[
1[x<\beta] = 1[\hat{x}<\theta(\beta)] \oplus 1[\hat{x}<r] \oplus w(\beta).
]

The term **`1[\hat{x}<r]`** is shared across all `β` for that element.

You should ensure the compiler emits **exactly one** query for `1[\hat{x}<r]` per gate instance (or even per wire), and reuses it in all rewrites. This does *not* depend on sampled masks; it depends only on the rewrite rule structure.

Where to implement:

* `src/compiler/suf_to_pfss.cpp`

Mechanically:

* represent “threshold kinds” as symbolic refs:

    * `ThresholdKind::RIN` (the `r` term)
    * `ThresholdKind::SHIFTED_BOUNDARY(i)` for each boundary
* build the query list from these symbolic refs in a fixed order
* reuse indices in boolean expression lowering

This reduces:

* PackCmp query count
* PFSS eval work
* key bytes

And it is fully consistent with your paper’s “fixed ordering, mask-independent shapes” requirement.

---

# Part E — What to expect if you implement A + B correctly

Your current numbers:

* `suf online ≈ 0.6257s`
* `pfss_flush_eval_eval_ns ≈ 0.276s`
* `open_comm_ns ≈ 0.116s`
* `pfss.num_flushes=104`, `open_flushes=264`

After:

### (A) SPC1 key transposition → packed pred becomes fast

You should be able to turn `SUF_TRUNC_PACKED_PRED=1` back on without regression and reduce PFSS eval time.

### (B) batching policy + buffer sizing

You should see:

* `pfss.num_flushes` drop from 104 → **~30–50**
* `open_flushes` drop from 264 → **~60–120**

Even before overlap, this reduces fixed costs (launch, staging, pack/scatter) substantially.

### (C) overlap

If you overlap most opens under PFSS + other GPU work, `open_comm_ns` becomes much less visible in wall time.

That combination is exactly the kind of “structural” improvement needed to close a 3× gap without new primitives.

---

# A minimal “do this next” checklist

If you want the shortest sequence that is *guaranteed implementable* and highest ROI:

1. **Implement SPC1 PackCmp KeyV2 (transposed layout)** and update the kernel to consume it.
2. **Re-enable trunc packed pred** and verify it speeds up the trunc-heavy microbench.
3. **Increase PFSS/Open staging buffer sizes** and add flush-reason counters.
4. **Add ScopedDeferFlush** and wrap one entire transformer block so PFSS/Open flush only at block barriers.
5. **OpenCollector multi-in-flight** so you can flush async and overlap comm with GPU work.

This stays squarely inside your paper’s story:

* still PackCmp/PPCS/standard opens,
* still mask-correct,
* still shape-safe,
* just faster backend + better batching.

If you want, paste (or summarize) the current **SPC1 fused kernel’s thread mapping** (lane meaning, loop order over queries/levels), and I can give you a *precise* byte-level KeyV2 layout (offset math + alignment + exact indexing) that matches your kernel with zero ambiguity.
