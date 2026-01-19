Below is a concrete, *implementable* optimization plan that keeps the **paper.tex security posture** (“one‑key‑private tFSS; leakage = public masked openings + public shape only”), and gives you a realistic path to **beating Sigma** rather than just shaving small constants.

I’m going to be blunt about what matters: **you will not outrun Sigma by micro‑optimizing the current SigmaFSS‑DCF GPU kernel alone.** Sigma’s paper explicitly avoids “classic DCF everywhere” and instead uses *DPF‑based* comparison + spline/interval machinery with *memoization/packing* to avoid (n\times k) PRG work. If your secure backend is still structurally “(k) thresholds ⇒ (k) DCF evaluations”, you are fighting a losing battle on GPU.

Everything below is written to be consistent with the current repo structure and your latest secure key format direction.

---

## 0) First: is it “allowed” that secure GPU path is DCF‑only?

Security-wise: **yes**—“only SigmaFSS one‑key‑private DCF” is *not weaker* than your paper’s assumption. It’s simply a *different instantiation* of the abstract tFSS interface.

Paper alignment-wise: it’s **not OK if you keep performance claims** that rely on the two-template structure being *cheap* in practice, because:

* Your paper abstracts tFSS as **PackCmp + IntervalLUT** templates that are assumed to be *standard* and *efficient*.
* If “IntervalLUT” degenerates into “many DCFs + lots of PRG work,” then your system is no longer the engineered thing you claim it is, and you won’t match the claimed speedups.

So: **DCF-only is fine for security, but it caps performance hard.** If the goal is “SUF(gpu, secure) > Sigma(gpu, secure)”, you need to *match Sigma’s asymptotic structure* for comparisons/LUTs (DPF-based + packed/memoized), not just wrap DCF.

---

## 1) Why you’re slow (the root cause in one sentence)

Your current secure PackCmp/IntervalLUT path is still effectively doing:

[
\text{cost} \approx #\text{(comparisons)} \times n \times \text{(heavy PRG)}
]

where your PRG is “variable‑key AES with key schedule per level”, and your LUT/packed work is decomposed into **many independent DCF subkeys**, so there is **no cross-threshold sharing** of the expensive per-level tree expansion.

Sigma’s implementation avoids exactly this. (The Sigma paper’s text even emphasizes “DPF-based comparison is >2× lower than DCF” and that spline/interval lookup avoids the naive (n\times k) behavior via memoization.)

---

## 2) Judge your proposed ideas (which ones are correct / worth it)

You listed five candidate directions. Here’s the honest verdict:

### (1) “Optimize DCF PRG: fixed-key AES-CTR / ChaCha”

* **Fixed-key AES-CTR with public key is *not* a drop-in replacement** for the PRG used inside GGM/DPF/DCF trees. If the construction expects a *PRG with stretch*, using a public permutation directly is distinguishable (inversion test exists).
* **ChaCha is viable** *if used as a PRF/PRG keyed by the seed* (i.e., still “variable-key”), because ChaCha’s “key schedule” is trivial (load key words).
* Conclusion: ✅ **Switching to an ARX PRG (ChaCha/Ascon-like) keyed by the seed is plausible and can help**, but it **won’t be enough** if the algorithmic structure remains “(k) thresholds ⇒ (k) full tree evals.”

### (2) “Compress keys: short seed + correction words; PRG-expand v”

* If you mean **removing the huge per-level/per-output material** (the `v`-style tables) and deriving it from seeds:
  ✅ Correct direction, but **requires changing the underlying FSS construction**, not just refactoring serialization.
* In other words: you can’t “just compress SigmaFSS DCF keys” without re-deriving correctness/security for a modified scheme.
* Conclusion: ✅ Big potential win, but it’s a **new backend**, not a patch.

### (3) “Reduce kernel/flush granularity”

* ✅ Absolutely correct and *immediately implementable*.
* But if PFSS compute dominates by 10–20×, reducing launches alone won’t close the gap.

### (4) “Key memory layout SoA”

* ✅ Correct, good GPU hygiene.
* Usually a **1.2×–2×** win if you’re bandwidth-bound. If you’re PRG compute-bound, it may be smaller.

### (5) “Profile-driven”

* ✅ Necessary; your numbers already show PFSS dominates.

**Bottom line:** (3)+(4) are necessary “plumbing”. (1)+(2) matter only if paired with the real fix: **implement packed/memoized DPF-based templates**.

---

## 3) The path that can actually beat Sigma (while keeping your paper’s security)

### Goal

Make the secure GPU backend’s “tFSS cost per gate” look like:

* **One packed predicate extraction** with *shared tree expansion* across all predicates needed for a gate, and
* **One interval lookup** with *shared tree expansion* across all interval boundaries and returning a vector payload,

so that per element you’re closer to:

[
O(n) \text{ PRG work per template} \quad \text{instead of}\quad O(n \cdot k)
]

This is the *only* plausible way to reach Sigma-class performance.

---

## 4) Concrete step-by-step plan (implementation-first)

### Step A — Immediate wins (do these even if you later replace the crypto)

These steps are quick and reduce overhead without touching security.

#### A1) Default to fused evaluation on GPU for PackCmp

In your secure GPU backend, you already have a fused kernel path (`packed_cmp_sigma_dcf_kernel_keyed`). Right now, your latest description suggests you’ve improved D2H header reads and device parsing, but **you must ensure you’re not launching “one kernel per bit”** in the secure path.

**Action:**

* In `SecureGpuPfssBackend::eval_packed_lt_many_device`, make `fuse=true` by default when:

    * `keys_flat` is device pointer (cached material),
    * `num_bits <= 64` (current kernel limitation),
    * `N` is large enough to amortize.
* If you need more than 64 predicate bits, extend the kernel to support larger `out_words` (it currently hard-codes 2 words in places—fix that).

**Why this matters:** it reduces launches and improves cache locality. It won’t solve the PRG cost, but it’s a baseline.

#### A2) Merge PFSS jobs more aggressively (reduce `num_flushes`)

Your profiling shows `pfss.num_flushes` is huge. That is pure overhead + kills overlap.

**Action:**

* In the superbatch scheduler (look at `src/runtime/pfss_superbatch.cpp`), add a policy:

    * **Do not flush** after each gate.
    * Flush only at dependency barriers:

        * after opening a batch of (\hat{x}),
        * before consuming predicate shares / coeff shares if needed.
* Increase maximum queued jobs; group by `(template_kind, in_bits, out_words, key_bytes)` to maximize reuse.

**Acceptance test:** `pfss.num_flushes` should drop by **>10×** (e.g., 152 → <15) on bert-tiny L128 B1.

#### A3) Make key blobs SoA-friendly *inside device cache*

Even if `keys_flat` is AoS on disk/host, once you cache it on GPU you can pretranspose hotspots.

**Action:**

* Add a one-time “cache transform” step:

    * split headers, seeds, correction material into separate arrays
    * align to 16/32 bytes
* Update kernels to read from SoA.

This is a classic win when keys are large.

---

### Step B — The real fix: implement **DPF-based packed comparison** (Sigma-style) in your backend

This is where you start matching Sigma, and it’s also the most likely to make SUF win because SUF reduces *how many* times you need these operations.

#### B1) Stop using SigmaFSS-DCF for comparisons; implement DPF-based `lt`

Sigma’s paper uses a DPF-based comparison (`Eval_n^<`) that is *strictly cheaper than DCF*.

**What to implement:**

* A new backend primitive: `PackedLT_DPF` that outputs XOR shares of many `1[u < theta_t]` queries.
* Internally, it uses a DPF construction (GGM tree) and Sigma’s less-than evaluation method.

**Keep the current DCF path** as a fallback:

* `secure_gpu_backend.mode = {DCF, DPF}`
* default `DPF` once stable.

#### B2) Choose a GPU-friendly PRG keyed by seed

You *cannot* keep “AES key schedule per node per level” and expect to win on GPU.

**Implement PRG as:**

* `ChaCha12(seed, counter)` (or ChaCha8 if you accept weaker margin for prototype)
* seed is 256-bit key; derive from your 128-bit seed by (seed || seed) or HKDF-like expansion (done in keygen).

This keeps the “seed-as-key” property needed for DPF security proofs, but avoids AES key expansion cost.

**Where to implement:**

* Replace the PRG used in `prg_two_blocks()` / `convert_group()`-like logic with a ChaCha block function.
* You’ll need the same PRG in keygen (CPU or GPU). If keygen stays CPU, implement ChaCha there too.

#### B3) Make PackCmp truly “one template eval”

Right now, even with fused kernel, you are still doing **T independent DCFs**. That’s not what you want.

Instead, make PackCmp produce all T bits with **one tree traversal per input**.

**Implementation approach that is feasible:**

* Build a single multi-output predicate extractor key per gate instance:

    * output is a packed bitvector of size T
    * evaluation traverses tree once and emits packed output shares

This is essentially “FSS for vector-valued functions” where output type is ({0,1}^T). The key size will scale with T (inevitable), but **evaluation should not do T separate tree walks**.

**Practical compromise (still beats current):**

* Partition predicates into groups of up to 64 bits (or 128) and do 1 traversal per group.
* Still reduces “T traversals” → “ceil(T/64) traversals”.

---

### Step C — Implement **IntervalLUT as a real interval template**, not “sum of DCFs”

This is the other half of your paper’s “two-template theorem” engineering reality.

Right now your secure IntervalLUT kernel loops `intervals` times and calls a full DCF each time. That’s the exact (n\times k) failure mode Sigma avoids.

#### C1) Implement a “0-degree spline / piecewise constant FSS” template (Grotto/Sigma style)

You want a key that encodes:

* secret cutpoints (mask-dependent), and
* vector payload per interval (coeff vectors)

with **one traversal** (or near-one) per input.

**Feasible implementation strategy:**

* Represent the piecewise constant function by a tree where each node stores:

    * correction seeds (like DPF),
    * and a *group delta accumulator* for the payload.
* Evaluation does a single walk down the tree (depth n), updating the payload share.

The key trick is to attach the payload contributions to correction words in a way that makes the *sum of the two parties’ traversals* reconstruct the correct payload for the interval.

This is exactly the kind of “template” your paper assumes exists.

#### C2) Enforce **mask-independent shape** by padding

Whatever construction you choose, **do not let key size depend on:**

* where the wrap happens,
* how many intervals “effectively split”,
* any dedup/sparsity that depends on cutpoints.

Always allocate:

* exactly `M = m+1` intervals (or fixed worst-case),
* exactly fixed payload dimension `p`,
* fixed key bytes per instance.

This preserves your paper’s leakage function.

---

## 5) A “Codex-ready” implementation roadmap (keep DCF, add fast secure templates)

Here’s an instruction sequence you can literally hand to a coding agent.

### Phase 1: plumbing + batching (safe quick wins)

1. **Enable fused PackCmp kernel by default** for secure GPU backend:

    * In `SecureGpuPfssBackend::eval_packed_lt_many_device`, set `fuse=true` when `num_bits<=64`.
    * Extend kernel to support `num_bits>64` by allowing `out_words>2` (fix hard-coded loops).
2. **Reduce flushes**:

    * In `pfss_superbatch.cpp`, delay flush until dependency barriers; increase max in-flight jobs; group jobs by template kind/shape.
3. **SoA cache transform**:

    * When caching `keys_flat` on device, transform into SoA arrays for hot fields (headers, seeds, v/cw blobs).

### Phase 2: fast secure PRG (still keep DCF fallback)

4. Introduce a new PRG implementation (`ChaCha12`) in CUDA:

    * `chacha_block(seed, counter)` returns 64 bytes.
    * Use it to implement:

        * `PRG_expand(seed) -> (seedL, tL, seedR, tR)` (extract from bytes)
        * `PRG_group(seed, group_size)` (emit group masks)
5. Gate with build flag:

    * `SUF_SECURE_PRG=CHACHA` uses ChaCha in eval for new backend
    * Keep `SUF_SECURE_PRG=AES` for legacy DCF path.

### Phase 3: DPF-based comparison primitive (Sigma-style)

6. Add a new key type + eval path:

    * `PackedLTKeyDPF` encoding T comparisons in one structure (or chunks of 64).
7. Implement `Eval_lt_DPF_many(inputs_hatx, key_blob) -> XOR-shared packed bits`
8. Add runtime switch:

    * `SecureGpuPfssBackend::eval_packed_lt_many_device` chooses DPF-packed path if available, else falls back to DCF.

### Phase 4: real IntervalLUT template

9. Implement `IntervalLUTKeyDPF` (piecewise constant FSS) with vector payload:

    * one traversal per input
    * returns additive shares of payload vector
10. Switch SUF coefficient selection to use this backend path.
11. Keep old “sum of DCFs” IntervalLUT as fallback behind a flag.

### Phase 5: prove you beat Sigma

12. Microbench:

* throughput of `packed_lt` and `interval_lut` kernels alone

13. End-to-end bench:

* bert-tiny L128 B1 per-element masks =1

14. Acceptance criteria:

* PFSS time no longer dominates matmul by >1000×
* online time ≤ Sigma, key_bytes ≤ Sigma (or explain tradeoff)

---

## 6) Key warning: don’t accidentally break your paper’s leakage model

Two common “optimizations” that silently break the paper:

* **Sorting/deduplicating cutpoints based on secret mask-derived values** to reduce key size.
  This makes key shape mask-dependent → leaks mask bits.
* **Variable work / early exits** based on secret thresholds in GPU kernels.
  This leaks through timing/memory access.

If you do prefix-sharing/memoization, do it in a way whose **public schedule depends only on the descriptor** (or pad to a fixed schedule).

---

## 7) If you want the shortest path to “better than Sigma”

The fastest engineering route is:

1. **Wrap Sigma’s own optimized GPU DPF/spline primitives (if accessible in your environment)** as the tFSS backend, rather than reimplementing cryptography.
2. Then SUF’s compiler/batching advantages can show up immediately.

If you can’t reuse Sigma code, then your best “independent” route is:

* implement **DPF-based comparison with ARX PRG**, and
* implement **interval LUT as a real piecewise-constant FSS template** (single traversal), and
* keep DCF fallback for correctness + debugging.

That is the minimum set that plausibly closes a 10×–20× gap.

