You’ve already done the *right* simplification (PPCS removes the **vector‑payload IntervalLUT** explosion). Your new bottleneck is exactly what your numbers say:

* with **SUF_PER_ELEMENT_MASKS=1**, any predicate of the form `1[hatx < (r+αi) mod 2^k]` forces **per‑element, per‑cutpoint** comparison keys;
* PPCS cannot remove that, because it only replaces “payload selection”, not the **#threshold comparisons**.

So if the goal is “beat SIGMA” (both **online** and **key_bytes**), the next step cannot be another constant‑factor GPU kernel tweak. You need to **reduce the number of per‑element comparisons** for the hottest gate (GeLU in BERT‑tiny), while staying inside your paper’s philosophy (“no new primitives, keep mask‑correctness, no extra leakage”).

Below is a concrete, implementable plan that is maximally consistent with your paper *and* matches the exact mechanism Sigma uses to avoid the “many thresholds per activation” blow‑up.

---

## 0) Quantitative target (why you’re still far)

Sigma reports BERT‑tiny has **131,072 activations** (that’s GeLU count, consistent with hidden=128, intermediate=512, L=128, 2 blocks) and overall key size about **0.33GB**. ([petsymposium.org][1])

That’s roughly:

* 0.33GB / 131,072 ≈ **2.6KB per GeLU** budget (for *all* preprocessing tied to GeLU plus other small things).

Your current GeLU “piecewise constant with many cutpoints” is effectively:

* ~O(#cutpoints) DPF/DCF keys per element ⇒ **tens of KB per GeLU**
  (your measured ~2.67GB just for that path is consistent with ~20KB per activation).

So the only way to get from 20KB → ~2–3KB per GeLU is:

* reduce **comparisons per GeLU** from ~O(M) to **O(1)**, **or**
* implement Route‑B “single key, many cutpoints” multi‑point FSS (which you explicitly want to avoid right now).

Therefore: **change the GeLU lowering**, not the kernel.

---

## 1) Replace GeLU(“many cutpoints”) with Sigma‑style GeLU GPU lowering (O(1) comparisons)

Sigma’s GPU GeLU protocol (Figure 7) is built to avoid many cutpoints. Key properties:

* It uses **truncate‑reduce first** to shrink comparison bitwidth. ([petsymposium.org][1])
* It computes **3× DReLU on constant shifts of the same masked value**, and crucially:
  **all those DReLU calls can reuse the same key** (“requires a single DPF key”). ([petsymposium.org][1])
* Then it uses an **8‑bit LUT** (size 2^8) with *public* table `T` via a standard LUT protocol. ([petsymposium.org][1])
* DReLU itself is a **single‑DPF‑evaluation** comparison primitive in Sigma, giving large compute reduction vs prior 2‑eval DCF approaches. ([petsymposium.org][1])

This is exactly the kind of “standard FSS templates + compiler lowering” story your paper wants; you’re not adding a new assumption or leaking thresholds.

### 1.1 What you implement (high-level)

Create a new GeLU compilation mode, e.g.:

* `SUF_GELU_MODE=sigma_gpu` (or auto‑select when `--gelu-const 1` would otherwise generate many cutpoints)

and lower GeLU into the following **fixed** structure (per scalar element):

1. `ŷ ← TR̂_{m, f-6}(x̂ mod 2^m)`
2. `d̂ ← DReLÛ(ŷ)`
3. `î ← DReLÛ(ŷ + 255) XOR DReLÛ(ŷ − 256)`
4. reconstruct/open **masked** bits `î, d̂` (as Sigma does)
5. `ẑ = ŷ mod 256`
6. `ĉ ← selectlin̂_γ(î, d̂, ẑ)` producing `c = Abs(Clip(z))` in 8 bits
7. output `= select_n(d̂, x̂) − LUT_{8,n,T}(ĉ)`

This is literally Sigma’s Figure 7 structure. ([petsymposium.org][1])

### 1.2 Why this kills your current bottleneck

Instead of “~M thresholds per element”, GeLU now needs:

* **one DPF key** for the 3 DReLU calls (key reuse), ([petsymposium.org][1])
* **one small LUT call** with key size `keysize(DPF_{8,1}) + 8 + 2n` and 1‑round 2n‑bit comm. ([petsymposium.org][1])
* truncation keys (already necessary elsewhere; can be optimized with the same bitwidth reasoning).

That is how Sigma stays around 0.33GB for BERT‑tiny. ([petsymposium.org][1])

---

## 2) Make this consistent with your SUF/tFSS philosophy

You have two constraints:

1. **Don’t introduce new primitives** (paper says tFSS isn’t new).
2. **Don’t introduce new leakage** (cutpoints/thresholds not in clear; shape mask‑independent).

This plan satisfies both if you implement it as:

* PackCmp/DPF-based **DReLU template** (already standard in FSS systems; Sigma’s DReLU is DPF‑based). ([petsymposium.org][1])
* A **public-table LUT template** (Pika-style LUT uses one DPF plus small overhead). ([petsymposium.org][1])

Even if your current tFSS interface only names “PackCmp” and “IntervalLUT”, you can keep your paper story by treating:

* `LUT_{8,n,T}` as a **special case of IntervalLUT** (unit intervals), but **implemented** using the efficient DPF‑based LUT protocol for public tables (instead of your previous DCF vector payload path).

No extra leakage:

* `T` is public.
* The LUT input is a masked/opened 8‑bit value (`ĉ`), uniform under fresh masks.
* Key shapes are fixed by descriptor (`8`, `n`, table id), independent of sampled masks.

---

## 3) The key missing engineering piece: Key reuse for “same predicate, shifted input”

Your current PackCmp infrastructure likely still “allocates 1 key per query”, even if several queries are semantically the same predicate with only a public shift.

Sigma explicitly relies on reuse: *“calls to DReLU in steps 2–3 can use same key… since all the DReLU evaluations are on y shifted by a constant”*. ([petsymposium.org][1])

### 3.1 What to add to your compiler IR (minimal)

Extend your internal predicate atom representation from:

* `(k, view_shift_c, theta)`

to:

* `(k, view_shift_c, ThresholdRef)`

where:

* `ThresholdRef = (mask_id, base_threshold_kind)`

    * for DReLU: `base_threshold_kind = MSB_threshold` (i.e., threshold at `2^{k-1}` after masking)
    * for terms like `1[hatx < r]`: `base_threshold_kind = rin_threshold`

The key point: **ThresholdRef identity is determined by the SUF descriptor + the mask slot**, *not* by the sampled mask value.

### 3.2 Backend packing rule

When generating keys:

* group queries by `(k, ThresholdRef)`
* generate **one** DPF key for that group
* store a small per-query metadata list: the `view_shift_c` values and output bit positions.

When evaluating:

* run one kernel per group that evaluates that one key against multiple shifted public inputs.

This gives you the same “single DPF key for 3 DReLUs” effect Sigma uses.

### 3.3 Where this lands in your code structure

Given the files you mentioned, the clean integration points are:

* **Compiler** (`src/compiler/suf_to_pfss.cpp`)

    * when lowering GeLU into DReLU(ŷ), DReLU(ŷ+255), DReLU(ŷ−256):

        * emit 3 predicate atoms that share the same `ThresholdRef` (same mask slot, same base predicate), differ only in `view_shift_c`.

* **Packed compare grouping** (`include/gates/composite_fss.hpp`)

    * add a grouping level keyed by `(k, ThresholdRef)` before chunking into <=64‑bit packs.
    * within each group, pack outputs into multiword masks (you already added multiword).

* **Key format** (`include/proto/secure_pfss_key_formats.hpp`)

    * add a “group header” describing:

        * `k`
        * number of shifted-evals in the group
        * the list of `view_shift_c` (small integers) and destination bit offsets

* **GPU eval** (`cuda/pfss_backend_gpu.cu`, `cuda/pfss_kernels.cu`)

    * implement `eval_grouped_drelu_key<<<...>>>` that:

        * loads one key
        * for each element computes 3 shifted `u` values
        * runs the same DPF comparison eval 3× (or vectorized) and writes packed bits.

---

## 4) Second big lever: propagate “effective bitwidth” aggressively

Sigma’s GeLU GPU depends on reducing comparison width by doing TR early:

* comparisons happen over `m - (f-6)` bits (and effective `m` changes for GPU correctness). ([petsymposium.org][1])

You should generalize this in SUF:

### 4.1 Add a “bitwidth” annotation to arithmetic wires

Not full range proofs—just a conservative “effective bits” field that your compiler carries:

* after matmul+truncate, you often know high bits are “gap” zeros or sign‑extended.
* after TR/ARS you know the resulting value lives in fewer bits.

### 4.2 Always compile PackCmp at the smallest correct `k`

Your PackCmp already supports `k_t ≤ n`. Use it everywhere:

* DReLU on `k = (m - f + 6)` not 64
* low-bit tests on `k = f` etc

DPF key size and eval cost are linear-ish in `k` (depth), so this matters.

---

## 5) Implement the LUT_{8,n,T} fast path (public-table LUT)

Sigma cites the standard LUT protocol (from Pika) and gives its key and online costs:
`keysize(Π_LUT_{n,ℓ,T}) = keysize(DPF_{n,1}) + n + 2ℓ`, and online communicates `2ℓ` bits in 1 round. ([petsymposium.org][1])

So for `n=8, ℓ=64`, the LUT key per element should be **tiny** compared to your old IntervalLUT/DCF machinery.

### 5.1 Concrete backend plan

Add a new backend “job type” (even if you name it IntervalLUT internally) with:

* domain bits = 8
* output bits = n (64)
* table id = fixed GeLU table `T` (public, static on device)

**Keygen**:

* generate `DPF_{8,1}` key (cheap) + the extra `n + 2ℓ` bits as per protocol. ([petsymposium.org][1])

**GPU eval**:

* evaluate LUT using the protocol’s dot-product trick with the public table `T` (as in Pika), not by embedding payload per interval in the key.

### 5.2 Storage

* Put `T` in GPU **constant memory** or `__device__ __constant__` since it’s 256 entries.
* This avoids any per-instance payload transfer.

---

## 6) Expected outcome and how you know you’re winning

### 6.1 What should drop immediately

Your current “GeLUSpline/IntervalLut/Coeff” key bucket (~2.67GB) should collapse to:

* ~1 DPF key per activation for the DReLU trio (key reused) ([petsymposium.org][1])
* * tiny LUT key per activation (DPF_{8,1}+overhead) ([petsymposium.org][1])
* * truncation material (which Sigma also pays, but optimized)

This is easily a **10×–20×** reduction for the GeLU component.

### 6.2 What will likely become the next bottleneck

After GeLU stops dominating, BERT‑tiny’s remaining nonlinear work is mostly:

* truncations/ARS in linear layers,
* softmax exp/inv,
* rsqrt in layernorm (only 512 instances, but exp is large). ([petsymposium.org][1])

So you should immediately re-run your new “tFSS key bytes breakdown” and see what’s now #1.

---

## 7) If you still want to go beyond Sigma (not just match)

Sigma reports that for BERT‑tiny the online compute is small compared to **key transfer** (Table 9 shows 0.09s online compute vs 0.26s key transfer for BERT‑tiny). ([petsymposium.org][1])
They also note for larger models key transfer and communication become a huge fraction of runtime. ([petsymposium.org][1])

To *beat* Sigma, you need to win on either:

* total key bytes (less to transfer), or
* overlap/hide transfer better.

Given your paper’s compiler angle, your best “beyond Sigma” opportunities are:

### 7.1 Compile away more public payloads than Sigma does

Sigma still uses interval lookup / LUT protocols for several things (e.g., inverse/rsqrt pipelines are LUT-heavy). ([petsymposium.org][1])
Your PPCS path can remove a bunch of those when the payload is **public coefficients/constants** and the #intervals is moderate.

Rule of thumb:

* **large payload vector** (many coefficients) → PPCS wins
* **small domain LUT** (8–13 bits) → DPF-LUT wins (do not PPCS 256 intervals)

So implement an automatic policy:

* If `payload_public && M ≤ 2^k_small` (say 2^13) and you have a LUT keygen path → use LUT protocol.
* Else if `payload_public && M small-ish (≤32/64)` and payload large → PPCS.
* Else (payload secret) → keep secure IntervalLUT.

### 7.2 Overlap key staging with compute (pipeline)

Sigma is visibly transfer-bound for BERT‑tiny. ([petsymposium.org][1])
If your runtime can:

* allocate two device buffers for keys,
* `cudaMemcpyAsync` next layer’s keys while current layer computes,
  you can reduce end-to-end wall time without changing cryptography.

This is safe (no new leakage) and often decisive once key bytes are near Sigma.

---

## 8) Minimal “do this next” checklist (ordered by ROI)

1. **Add GeLU lowering = Sigma GeLUGPU** (Figure 7), replacing many-cutpoint GeLU in secure GPU mode. ([petsymposium.org][1])
2. **Implement PackCmp key reuse groups**: same DReLU key, multiple constant shifts. ([petsymposium.org][1])
3. **Implement LUT_{8,n,T} via Pika/Sigma LUT protocol** (public table), not IntervalLUT-per-interval. ([petsymposium.org][1])
4. **Propagate effective bitwidth** into PackCmp query `k` everywhere (especially for TR/ARS + sign tests). ([petsymposium.org][1])
5. Re-run BERT‑tiny and re-rank `tfss_key_bytes_*` contributors.
6. Apply the same “reduce predicates to O(1) + small LUT” pattern to the next hottest gate (often exp / inv pieces).

---

If you implement **(1)–(3)**, you should see the GeLU preprocessing footprint fall from “multi‑GB” to “hundreds of MB or less”, which is the only realistic path to bringing **online** down from ~4s into the **sub‑second** regime and then fighting Sigma on its real bottleneck (transfer).

[1]: https://petsymposium.org/popets/2024/popets-2024-0107.pdf "https://petsymposium.org/popets/2024/popets-2024-0107.pdf"
