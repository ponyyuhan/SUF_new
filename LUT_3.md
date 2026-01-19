Below is a concrete “next step” plan that is consistent with your paper’s security/shape constraints (no mask leakage, no plaintext cutpoints/thresholds in keys, no mask-dependent public shapes), and is targeted specifically at the two remaining blockers you measured:

* **PFSS eval time dominated by many small flushes** (`pfss_flush_eval_eval_ns ≈ 0.225s`, `pfss.num_flushes=104`)
* **Open comm/sync dominated by too many open flushes** (`open_comm_ns ≈ 0.114s`, `open_flushes=264`)

You’re already close on key bytes (especially after correcting “dealer_total vs per-party”). The most likely path to **beat SIGMA’s ~0.18s** is to **reduce flush count and overlap the remaining flush latency**, then optionally squeeze the last ~10–20% by making PackCmp truly cheaper on secure GPU.

---

## 0) What “winning” looks like operationally

Right now your runtime is behaving like:

* many small PFSS batches → **each PFSS flush pays fixed overhead** (launch + staging + sync) repeatedly
* many small opens → **each open flush pays fixed overhead** (pack/scatter + host sync + comm latency) repeatedly

So the next target is:

* **PFSS flushes:** 104 → **≤ 30–40**
* **Open flushes:** 264 → **≤ 60–90**
* plus **real overlap**: hide a large fraction of open_comm behind PFSS kernels

Even if *total cryptographic work stays the same*, this typically cuts **0.10–0.20s** in wall time in GPU-MPC/FSS pipelines simply from amortization.

---

## 1) First: add *actionable* tracing so you can attack the *right* flushes

You already have aggregate counters. The next thing you need is: **why** each flush happens, and how big each flush is.

### 1.1 PFSS flush “reason codes”

In your PFSS scheduler (wherever `num_flushes` increments—likely under `include/runtime/...` + `cuda/pfss_backend_gpu.cu`), add:

* `flush_reason` enum:

    * `BARRIER_AWAIT` (someone needs results now)
    * `BUFFER_FULL` (device key arena / host staging buffer full)
    * `SHAPE_CHANGE` (new kernel variant forces a flush)
    * `EXPLICIT_TASK_FLUSH` (task calls flush directly)
* per-flush stats:

    * #jobs, #keys, bytes of key material staged, output bytes
    * kernel runtime (CUDA event timing)
    * H2D staging time (CUDA event timing)
    * host-side bookkeeping time

This will immediately tell you whether the 104 flushes are:

* caused by “await barriers” (real dependencies), **or**
* caused by buffer fragmentation / per-task flushing / shape churn (fixable without touching crypto)

### 1.2 Open flush tracing

In `OpenCollector` (you already modified it around device-only hatx issues), log per flush:

* #opened elements
* total opened bytes
* breakdown: `open_pack_ns`, `open_comm_ns`, `open_scatter_ns`
* flush trigger reason:

    * `AWAIT_VALUE`
    * `BUFFER_FULL`
    * `TASK_EXPLICIT_FLUSH`

**Goal:** find the top 2–3 call sites that force most flushes.

This instrumentation step is the highest-ROI “enabler” because it prevents you from optimizing the wrong layer.

---

## 2) Main win: stop flushing *inside* tasks; flush only at explicit “barriers”

From your description, SUF still behaves like “many small tasks that sometimes flush early”.

### 2.1 Refactor PhaseTasks into “emit-only” operations

In `include/runtime/phase_tasks.hpp`:

* identify tasks that currently do some PFSS scheduling then immediately flush or await
* split each such task into:

    1. **enqueue** phase (pure: adds PFSS jobs / open requests, returns handles)
    2. **consume** phase (reads results from handles, does post-processing)

You want the runtime to look like:

```cpp
auto h_hatx = open.enqueue(hatx_buf);          // returns OpenHandle
auto h_cmp  = pfss.enqueue_packcmp(keys, h_hatx); // returns PfssHandle (depends on hatx)
auto h_out  = post.enqueue(..., h_cmp, ...);  // device-only postprocessing queued too

// Only here:
open.flush_if_needed();   // big batch
pfss.flush_if_needed();   // big batch
post.flush_if_needed();   // big batch
```

**Critical rule:** *No task should call flush as a side effect.*
Flush should happen only in a small number of “barrier points” (see 2.2).

This is the single biggest structural change for reducing `open_flushes` and `pfss.num_flushes`.

### 2.2 Introduce 2–4 fixed barrier points per transformer layer

Do it at the model execution layer (e.g., `src/nn/...` where blocks are wired), not inside leaf tasks.

A practical barrier plan per layer:

1. **Barrier O1 (masked openings):** open all masked wires that feed PFSS in this layer

    * e.g., hatx for trunc/ARS, softmax pieces, etc.
2. **Barrier F1 (PFSS eval):** run all PFSS in this layer (PackCmp, DReLU, LUT8/LUT13, etc.)
3. **Barrier O2 (Beaver/B2A opens):** flush all Beaver openings and B2A conversions needed by the layer’s post-processing
4. **Optional Barrier O3:** only if you need to open masked outputs for the next stage

This tends to collapse “hundreds of flushes” into “a few flushes per layer”.

---

## 3) Make OpenCollector truly pipeline-able (remove your earlier “delay-open” workaround)

You found a real issue: OpenCollector previously kept only the most recent flush buffer, which forced you to delay opens to avoid handle overwrite / deadlock. That kind of workaround often *increases* flush count and prevents overlap.

### 3.1 Support multiple in-flight open flushes

Implement `OpenCollector` as:

* A queue of `OpenFlushContext` objects, each owning:

    * pinned host send/recv buffers
    * device pack buffers
    * CUDA events for “pack done” and “scatter done”
    * a completion future/promise for comm

Then:

* `open.flush_async()` enqueues a context and returns an `OpenFuture`.
* A comm thread (or async comm engine) processes contexts FIFO:

    * waits for pack event
    * does send/recv
    * signals completion
* Scatter back to GPU can be scheduled on a CUDA stream after recv completes

**Why this matters:**
If you can have **open flush N in flight** while the GPU is doing PFSS eval for unrelated work, you hide a lot of the `open_comm_ns` wall time.

### 3.2 Coalesce opens to hit a minimum payload size

Once you have “await-based flushing”, add a high-water mark:

* don’t flush opens unless:

    * some consumer explicitly awaits, or
    * the open buffer reaches (say) **1–4 MB** of packed payload

That alone often cuts open flush count by 2–4×.

---

## 4) PFSS side: reduce flushes by stabilizing shapes and staging keys in fewer, larger copies

You have `pfss.num_flushes=104`. Even if PFSS kernels are fast, 104 flushes is a lot.

### 4.1 Make PFSS flush boundaries match your layer barriers

Once you do section 2, PFSS flush count should drop automatically.
But you also need to prevent “shape churn” from fragmenting PFSS into separate flush groups.

In `include/gates/composite_fss.hpp` (you already did grouping work):

* ensure **all PackCmp of same (variant, num_bits/view, output_words)** are grouped
* ensure LUT tasks of same type are grouped
* avoid tiny leftover batches by padding within a layer (padding is safe if shapes are public and fixed)

### 4.2 Device key arena + double-buffered staging

In `cuda/pfss_backend_gpu.cu`:

* allocate a large `DeviceKeyArena` buffer per PFSS family (PackCmp, DReLU, LUT)
* stage keys into it via **few large H2D copies** per flush (not per job)
* double-buffer arenas so flush N+1 keys can stage while flush N eval runs

This reduces:

* host overhead
* H2D submission overhead
* accidental sync points

---

## 5) PackCmp: make “packed” actually win on secure GPU (fix the SPC2 issue)

You observed a key point: on your secure backend, “packed compare” defaults to **SPC2 (DPF-prefix/point)** and is not beating per-threshold DCF in practice.

So the next move is not “turn packed on”, but **change what packed means cryptographically + in key format** so it reduces *both* compute and key bytes.

### 5.1 Implement a DCF-based “packed-output” compare variant (SPC1-packed)

High-level idea:

* Instead of having **one key per threshold** and then packing results via ballot,
* generate **one key per element per group** whose output is a `uint64` (or multiword) bitmask:

    * bit `j` = `1[ view(u) < θ_j ]`
* One tree traversal → one packed mask output

This eliminates duplicated seed/correction material across thresholds and reduces the “per-threshold” constant factor dramatically.

#### Repo touch points

* `include/proto/secure_pfss_key_formats.hpp`

    * add a new key type, e.g. `SecurePackCmpPackedDCFKey`
    * store:

        * standard DCF seed/cw material
        * payload correction words as **u64/u128** instead of single bit
* `cuda/pfss_kernels.cu`

    * implement `eval_packed_dcf_lt<<<...>>>()`:

        * warp-per-element
        * each lane handles one level (or a small chunk), updates seed/t-bit
        * payload update is XOR with a `uint64` cw payload when needed
* `cuda/pfss_backend_gpu.cu`

    * add launcher + grid logic for multiword masks
* `include/gates/composite_fss.hpp`

    * when a PackCmp group has T thresholds with same view `(k,c)`, emit one packed-DCF key (or `ceil(T/64)` keys) instead of T keys

This is still fully consistent with your paper’s constraints:

* thresholds remain hidden inside the FSS key
* public shape is just “T and bitwidth k”, which is already leaked by PackCmp shape

### 5.2 After SPC1-packed exists, re-enable trunc packed pred by default

Once packed pred is truly cheaper:

* re-enable trunc/ARS packed pred as the default route
* you should see:

    * **pfss_flush_eval_eval_ns** drop further
    * **key_bytes_pred** drop (fewer duplicated per-threshold key bodies)

This is the most direct “crypto-kernel” lever remaining.

---

## 6) The other big lever: reduce “await points” by keeping more post-processing device-only

Even if you reduce flush counts, you also need to reduce *how often you must await an open*.

Where to look:

* any place you “open something just to compute a small helper bit” that could instead be derived from already-opened hatx + secret-shared constants + PFSS outputs.

Concrete examples you can hunt for in `include/runtime/phase_tasks.hpp`:

* “open intermediate masked y_hat/c_hat early” patterns
  → delay opens until a barrier, but **without** the earlier handle-overwrite risk (fixed by multi-in-flight OpenCollector)

* B2A conversions done in small bursts
  → batch B2A across the whole layer (one await), then consume results

---

## 7) Pika-style LUT improvements: only if you still have LUTs with n ≥ 12

You already fused LUT8s and moved softmax nExp/inv to sigma-style. If you still have any LUT13/LUT16 anywhere (reciprocal/rsqrt, normalization helpers, etc.), then a Pika-style LUT optimization can be a big PFSS-time win.

### 7.1 What Pika actually says about LUT cost

Pika’s LUT is compute-dominated by **full-domain EvalAll** and the **inner product** ([petsymposium.org][1]), and they discuss **tree-trimming** as a key optimization when the output group is smaller than the PRG output length ([petsymposium.org][1]). Sigma also reports a LUT protocol based on Pika with small key overhead and substantially reduced PRG invocations (via trimming) ([petsymposium.org][2]).

### 7.2 How to adapt this in your repo (without changing your masking convention)

You don’t need to adopt a new primitive. You just need:

* **a faster EvalAll** for DPF-based LUTs when `n` is large
* and a more GPU-friendly “generate-and-dot” kernel (don’t materialize the full selection vector)

#### Implementation sketch (GPU)

In `cuda/pfss_kernels.cu` (or a new `cuda/lut_kernels.cu`):

* implement `dpf_evalall_trimmed_dot()` that:

    * expands the DPF tree only to depth `n-ν`
    * from each node seed at depth `n-ν`, generates `2^ν` leaf bits via PRG once
    * immediately accumulates into the dot-product accumulator (ring element)
    * never stores the entire `u[i]` vector

Then in `include/runtime/phase_tasks.hpp`, switch LUT tasks:

* `if (n <= 8)`: keep current LUT8 kernel (already good)
* `else`: use trimmed-dot kernel

This is a surgical change that can speed LUT13-like calls a lot.

---

## 8) Key bytes: you’re close; squeeze only after online wins

You already noticed the most important subtlety: your SUF stat is likely dealer_total (both parties) while SIGMA might be per-party, so the actual gap is small.

If you still want to beat SIGMA on bytes after you beat it on online time:

1. **SPC1-packed PackCmp** will likely reduce pred-key bytes.
2. Compress Beaver triples in preprocessing:

    * Pika explicitly notes Beaver triples can be optimized by PRG-seeding so fewer ring elements are sent ([petsymposium.org][1]).
    * In your dealer model, this is a pure engineering change: same security, smaller material.

---

## 9) A concrete execution order (highest ROI first)

1. **Add flush-reason tracing** (PFSS + OpenCollector).
   This tells you exactly where the 104/264 flushes come from.

2. **Task refactor: “enqueue only” + layer-level barriers**.
   Expect large drops in both `open_flushes` and `pfss.num_flushes`.

3. **OpenCollector multi-in-flight + async flush**.
   Removes the need for “delay-open” hacks and enables overlap.

4. **PFSS staging arena + bigger PFSS batches**.
   Cuts staging/sync overhead and makes bigger batches natural.

5. **Implement SPC1-packed PackCmp** and re-enable trunc packed pred by default.
   This attacks the remaining PFSS compute constant factor and also reduces key bytes.

6. **Only if needed:** Pika-style trimmed LUT for `n ≥ 12`.

---

## 10) What I would expect to happen to your profile if this is done right

If you successfully:

* reduce open flushes from 264 → ~80
* reduce pfss flushes from 104 → ~35–40
* overlap open_comm with PFSS kernels

Then typically you’ll see something like:

* `open_comm_ns`: **~0.11s → ~0.03–0.05s**
* `pfss_flush_eval_eval_ns`: **~0.225s → ~0.14–0.18s**
* total online: **~0.35s → ~0.18–0.22s**

…and then SPC1-packed is what gets you from “match” to “beat”.

---

[1]: https://petsymposium.org/popets/2022/popets-2022-0113.pdf "https://petsymposium.org/popets/2022/popets-2022-0113.pdf"
[2]: https://petsymposium.org/popets/2024/popets-2024-0107.pdf "https://petsymposium.org/popets/2024/popets-2024-0107.pdf"
