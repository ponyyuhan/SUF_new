Sigma’s security/semantics are *wire-level masking*: for each input wire (w_i) with value (x_i), the owner computes and reveals a **masked value** (\hat{x}_i = x_i + r_i) using that wire’s own mask (r_i), and the FSS key for a gate is generated for the corresponding input/output masks ((r_i, r_j)).

If you instead do what your current bridge effectively does (open a full vector all masked with the **same** (r)) then:
[
\hat{x}_a - \hat{x}_b = (x_a + r) - (x_b + r) = x_a - x_b,
]
so you leak **all pairwise differences inside the tensor** (and many downstream statistics: ordering, equality patterns, variance-ish signals, etc.). That is *strictly more leakage* than Sigma’s “masked openings” allow, and it is not simulatable from Sigma’s allowed transcript under the same definition. In other words: **you’re paying less because you’re revealing more.**

Below is a **complete, landable** design to get back to Sigma-level semantics **without killing performance**, and **without changing your numeric approximation** (so accuracy doesn’t drop). It does require changing the crypto path: *you cannot keep the “shifted full table in the key” trick* once masks become per-wire.

---

## 1) What you must match to claim “Sigma-level security”

Minimum bar (from Sigma’s protocol structure):

1. **Every value you `reconstruct/open` must be masked with its own wire mask** (r_i), not a tensor-global mask.

2. For each gate (g), preprocessing must generate keys for the **input mask(s)** and **output mask(s)** (Sigma denotes this as Gen(*F(r*\text{in}, r_\text{out})), then Eval on (\hat{x}=x+r_\text{in})).

3. The implementation must avoid any code path where **one party learns the plaintext output** of a non-linear gate (unless that output wire is explicitly owned/opened by that party by the protocol). If your “general poly” path still does local plaintext evaluation, that path is **not Sigma-secure** if it is ever used on secret wires.

So: **per-wire masks + per-wire keys + no plaintext non-linear outputs**.

---

## 2) Why your current “one `r_in` per tensor” is fast — and why it can’t survive the fix

Your current speedup is structurally coming from this equivalence:

* With a **single** (r), you can build **one** “shifted table share” indexed by (\hat{x}) and reuse it for all elements.
* With **per-wire** (r_i), you would need a **different shifted table per element**, which is (O(n\cdot 2^{\text{in_bits}})) space/key material — totally infeasible.

So once you restore Sigma semantics, the only viable class of solutions is:

> **Public/shared table stored once + small per-wire FSS/DPF keys + batched GPU eval**

Which is also exactly how Sigma thinks about LUTs on GPU (evaluate the LUT protocol using DPF kernels; optimize key transfer and bitwidth).

---

## 3) The concrete “no-accuracy-loss, Sigma-secure, still-fast”方案

### Core idea

Implement **SIL2-style LUT evaluation**:

* **Table (T)** is **public/shared** and stored **once** on GPU (registry/cache).
* Each element/wire has its own masks (r_i) (input) and (r'_i) (output).
* Preprocessing generates a **small DPF/leaf-block key** per element tied to (r_i).
* Online:

    1. open (\hat{x}_i = x_i + r_i) (Sigma-style),
    2. run one **batched GPU kernel** to produce shares of (\hat{y}_i = f(x_i) + r'_i),
    3. open (\hat{y}) (or keep as masked wire, depending on the surrounding circuit).

This meets Sigma’s definition/structure.

---

## 4) The performance key: keep LUT domains small (Sigma’s own trick)

If you try to do LUT over a full 16-bit domain, you’ll die (Sigma explicitly notes LUT size (2^n) is expensive).

Sigma’s GPU performance comes from *forcing LUT inputs onto small domains* via clipping/truncation/effective-bitwidth reasoning:

* GeLU uses **LUT8** (input reduced to 8 bits after clip/abs/truncation mod 256) and they explain reducing domain while keeping accuracy.
* They use **effective bitwidth** (m=n-f) after the preceding truncation, so comparisons/LUTs run on fewer bits.
* nExp is reduced from a 16-bit LUT to **two 8-bit LUTs** (Seedot trick).
* They even state typical LUT sizes: (2^8) for GeLU and exponential, (2^{10}) for SiLU, (2^{13}) for rsqrt, etc.

### How you preserve *your* accuracy (not just Sigma’s)

You said you can’t lose accuracy. The safe move is:

* **Do not change your approximation** (your cutpoints/interval outputs).
* **Change only the representation of the LUT index** so you never evaluate over a huge domain.

For your piecewise-constant “intervals = 256 / 1024” style activations, the function already only has 256/1024 distinct outputs. So you can define an **exact equivalent LUT** on:

* `k = log2(intervals)` bits (8 or 10),
* index = “which bucket/interval am I in?”

That doesn’t change the approximation at all.

---

## 5) The one tricky part: deriving a small-bit LUT index without breaking masking

If your underlying ring value has `bw` bits but your LUT has `k` bits, you need a correct masked-opening for the `k`-bit index.

There are two deployment-grade ways (choose one):

### Option A (closest to Sigma): restructure the math so the LUT input is *already* small-bit

This is what Sigma does (clip + truncation + mod 256, effective bitwidth).

If you can move the “reduce bitwidth” step **before** your LUT gate in the compiled circuit (i.e., represent the wire as `uint8/uint10` masked wire), then the LUT sees a small domain and everything is straightforward.

### Option B (if you insist on starting from opened (\hat{x}) at bw bits): exact bucketization with a single extra compare on low bits

For uniform buckets of size (2^{s}) where (s=bw-k):

* Write (x = x_\text{hi}\cdot 2^s + x_\text{lo}),
* (r = r_\text{hi}\cdot 2^s + r_\text{lo}),
* (\hat{x} = x + r) opened.

Then:
[
\hat{x}*\text{hi} = x*\text{hi} + r_\text{hi} + \text{carry},
\quad
\text{carry} = 1{x_\text{lo}+r_\text{lo}\ge 2^s}.
]

But **carry can be computed as**:
[
\text{carry} = 1{\hat{x}*\text{lo} < r*\text{lo}}
]
because (\hat{x}*\text{lo}=(x*\text{lo}+r_\text{lo})\bmod 2^s).

So you can:

1. open (\hat{x}) (already),
2. run **one DCF-based “public vs secret” compare** on `s` bits to get secret-shared `carry`,
3. form a *masked* `k`-bit value (\widehat{x_\text{hi}} = \hat{x}*\text{hi} - \text{carry}), which equals (x*\text{hi}+r_\text{hi}),
4. now do LUT on `k` bits with mask (r_\text{hi}).

This adds **one tiny compare on `s` bits**, but keeps your approximation exactly unchanged.

---

## 6) What to implement in *your* code (SUF_new-main.zip) — concrete changes

### 6.1 Replace the “tensor-global r_in” contract

In `src/sigma_suf_bridge.cu`:

* **Delete the idea of `gate.r_in_share` being a scalar.**
* Either:

    * use Sigma’s existing per-element input masks (best), or
    * generate a fresh per-element `r_in[i]` (dealer/preprocess) and store shares on GPU.

Concretely change:

```cpp
struct SigmaLutGateStateU64 {
  u64 r_in_share;               // <-- remove
  u64* d_output_mask;           // keep
  IntervalLutKeyV2 lut_key;     // replace with per-wire keys or table_id + per-wire dpf keys
  ...
};
```

To something like:

```cpp
struct SigmaLutGateStateU64 {
  // per-wire input mask shares (device pointer, length n)
  const u64* d_r_in_share;        // alias sigma's d_input_mask if available
  u64*       d_r_out_share;       // already exists as d_output_mask

  // SIL2 LUT:
  u32 table_id;
  u8  in_bits;                   // k (8/10/13)
  u8  out_words;
  DpfKeyDeviceView dpf_keys;      // per-wire DPF keys on device (SoA)
};
```

And in eval:

* stop calling `kernel_remask(..., gate.r_in_share, ...)`.
* open masked input directly from the wire’s own mask share.

This alone fixes the *biggest* semantic gap: you no longer reuse one mask across an entire tensor.

### 6.2 Remove “DirectTable” mode for any secure path with per-wire masks

With per-wire masks, DirectTable implies per-wire shifted table → not viable.
Keep it only behind a clearly-named benchmark flag, e.g.:

* `SUF_INSECURE_BENCH_SHARED_MASK=1` → allows old mode
* default off

### 6.3 Implement “Vector PikaLUT / DPF-LUT” (SIL2 LUT) with a table registry

You need:

**(A) TableRegistry**

* Key: `(function_id, in_bits, out_words, scale/clip params…)`
* Value: `device_ptr` to SoA table `T[w][idx]`

**(B) Per-wire DPF keys**

* Preprocessing: for each wire i, generate DPF key for input mask `r_in[i]`.
* Key bytes per wire are ~O(in_bits·λ), independent of intervals.

**(C) One batched GPU kernel**
`lut_dpf_dot_vec<<<grid>>> (hat_x, dpf_keys, table_ptr, r_out_share, out_share)`

Sigma explicitly expects LUTs to be realized via DPF evaluation over the domain, and they emphasize keeping bitwidths low and packing to reduce key transfer—your implementation should follow that style.

### 6.4 Keep accuracy identical

* Build the LUT table from **your existing approximation**.
* Only change how the index is represented/evaluated (Section 5 Option A/B).

If you also want to adopt Sigma’s own GeLU/nExp reductions, Sigma explicitly discusses that some early truncation changes the least significant bit but does not affect accuracy in their experiments; but if your constraint is “identical to my current approximation,” then keep your table exactly.

---

## 7) What is the *minimum* you must “补齐” to claim Sigma-level security

If your *actual deployed path* is “Sigma-SUF bridge + LUT-like nonlinearities”, then the minimum set is:

1. **Per-wire input masking for every opened masked tensor** (no collapsing to a single `r_in`).
2. **A per-wire-keyed LUT protocol** (DPF-LUT / PikaLUT) with shared/public tables; DirectTable-with-shift must be removed in secure mode.
3. Ensure no path returns **plaintext non-linear outputs** to a single party (disable/guard the local `eval_poly_only`-style path if it can ever run on secret wires).

Everything else (shape leakage, public parameters) can remain as-is because Sigma also assumes circuit structure/bitwidths are public.

---

## 8) Reality check: can you keep *exactly* the same performance advantage?

If your current win comes mainly from:

* “one mask for the whole tensor” **and**
* “one shifted secret-shared table reused for all elements”,

then **you cannot keep that win and also meet Sigma’s definition**. That win is *paid for* by the extra leakage.

What you *can* do (and what Sigma does) is move the performance battle to the places that matter:

* keep LUT domains tiny (8/10/13 bits),
* pack/stream keys efficiently,
* keep everything batched in one GPU launch per gate,
* use low-bitwidth datatypes (uint8/uint16) wherever possible.

That gives you a very plausible path to **match Sigma security** and still be competitive (and possibly still faster if your SUF compilation reduces the number of calls/gates or improves batching).

---

If you want, I can take *your exact current bridge functions* (`keygen_activation_u64`, `eval_activation_u64`, `keygen_nexp_u64`, `keygen_inv_u64`, `keygen_rsqrt_u64`) and write a **drop-in “secure mode” redesign map**: for each function, what `in_bits` becomes (8/10/13), which table to cache, which keys to generate, and which extra step (if any) is needed to produce the correct small-bit LUT input **without any change to your approximation table**.
