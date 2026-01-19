Below is a **concrete, landable “true SIL2” IntervalLUT** design that **actually improves efficiency** (key bytes + online time) in your codebase, **supports vector payload** (out_words>1), and fits the “DPF/CDPF single/near‑single traversal” intent of `SecureIntervalLutHeaderV2` (“SIL2”).
I will be very explicit about: (1) what SIL2 must *really* be to win, (2) the exact key format + kernels you can implement immediately, (3) how to configure/compiler‑select it, and (4) why **DMPF is not your B1**.

---

# 0) Why your current IntervalLUT is still SIL1 and why it cannot win

Your current “IntervalLUT” path on secure GPU is **SIL1**:

* It **generates (intervals‑1) DCF keys**, and each DCF key has per‑level VCWs ⇒ key bytes scale like
  **O(intervals × in_bits × out_words)**
* Online GPU eval runs a kernel that essentially iterates DCF‑like logic repeatedly ⇒ expensive.

This is exactly why you still see key bytes exploding and why “paper claim” is not realized.

So, **true SIL2 must remove the `intervals` multiplier from key size**, and ideally also remove the repeated traversal in evaluation.

---

# 1) What “true SIL2 IntervalLUT” should be in practice

The only *real* way to get SIL2‑level efficiency (key size independent of `intervals`) is:

✅ **Do not generate one DCF per cutpoint.**
✅ Encode only **the input mask** (or an equivalent secret point) in a DPF/CDPF key.
✅ Keep the LUT payload (the table values / per‑interval payload vectors) **public and shared**, stored once, not embedded per element.
✅ Use **one DPF traversal** to derive the selection and compute the output.

This is exactly the philosophy behind Sigma/Grotto “interval lookup” results: key size depends on DPF, not on k.

---

# 2) The B1‑landable SIL2 we can implement **now** (and it’s already 80% in your repo)

The good news: your repo already contains the necessary building blocks:

* `dpf_leafblock_*` (Pika leaf‑block DPF)
* `pikalut_dpf_dot_kernel_keyed` GPU kernel (scalar)
* `eval_pika_lut_many_device` plumbing
* `SecurePikaLutHeader` / `SecureLut8Header` patterns
* `SecureIntervalLutHeaderV2` reserved for SIL2

So the **fastest, highest‑impact SIL2 IntervalLUT** is:

## **SIL2 IntervalLUT = Vector PikaLUT**

That is:
**a single leaf‑block DPF key per element + one kernel launch that outputs out_words u64s**.

### Scope (B1):

* Works when IntervalLUT input domain is small enough to be **densified**: typically `in_bits <= 13` (8192 entries) or `<= 8` (256 entries).
* This matches the *real hot path* uses: LUT8 / LUT13‑style approximations, inverse/exp splines, etc.
* **Vector payload** supported: each table entry stores `out_words` u64s.

### Why densify?

IntervalLUT originally stores `(cutpoints, per‑interval payload)`.
For small domain we build a dense table:

`DenseTable[x] = payload(interval_of_x)`

Then IntervalLUT evaluation reduces to a standard masked LUT lookup.

This is the **exact same trick Sigma uses**: interval LUT becomes “database lookup”, and the DPF key encodes only the secret mask.

---

# 3) Exact key format (SecureIntervalLutHeaderV2 “SIL2”)

You already have:

```cpp
struct SecureIntervalLutHeaderV2 { // 32 bytes
  char magic[4]; // "SIL2"
  u8 in_bits;
  u8 out_words;
  u16 intervals;
  u32 dcf_key_bytes;
  u8 dcf_in_bits;
  u8 dcf_out_bytes;
  u8 dcf_shift_bits;
  u8 dcf_out_sem;
  u8 interval_lut_mode;
  u8 reserved;
  u16 reserved2;
  u32 reserved3;
};
```

We repurpose fields as:

| Field             | SIL2 meaning                                  |
| ----------------- | --------------------------------------------- |
| magic             | `"SIL2"`                                      |
| in_bits           | logN of LUT domain (8 or 13)                  |
| out_words         | vector payload length                         |
| intervals         | original interval count (optional, for debug) |
| dcf_key_bytes     | **dpf_key_bytes** (leafblock key length)      |
| interval_lut_mode | `0=PikaLeafblock`, `1=LUT8Point`              |
| reserved          | `sign_w` (the Pika sign bit)                  |
| reserved3         | **table_id** in a global LUT registry         |

**Body layout:**

```
[ SecureIntervalLutHeaderV2 ]
[ out_mask_share[out_words] : u64 each ]
[ dpf_leafblock_key_bytes ]
```

This mirrors your existing `SecurePikaLutHeader`, except:

* we store **out_words masks**, not 1
* we store **table_id**, not table values.

✅ Key bytes become:
`32 + out_words*8 + dpf_key_bytes`
and **do not depend on intervals**.

---

# 4) How evaluation works on GPU (single traversal, vector output)

You already have scalar Pika kernel:

* `pikalut_dpf_dot_kernel_keyed(...)` in `cuda/pfss_kernels.cu`

We implement:

## `pikalut_dpf_dot_kernel_keyed_vec(...)`

Same grid layout (128 threads per element), but each thread keeps `acc[out_words]` registers.

Pseudo logic:

1. For each `j` index chunk:

    * compute `idx = j ^ rot` (same rotation scheme)
    * evaluate DPF bit `b = dpf_leafblock_eval_bit_dev(key, idx)`
    * if b==1: for w in [0..out_words): `acc[w] += table[w][idx]`
2. Warp reduce acc[w]
3. Apply party sign normalization exactly like scalar:

    * `if party==1: acc[w] = -acc[w]`
    * `if sign_w<0: acc[w] = -acc[w]`
4. Add output mask shares:

    * `out[w] = acc[w] + out_mask_share[w]`

**Important table storage layout** for coalescing:

* store as SoA: `table_word[w][idx]`
* contiguous block per w: length N
* allocate once on GPU.

✅ Online eval is:

* 1 kernel launch per interval‑LUT gate
* one DPF traversal per element
* cost scales **linearly in out_words**, not in intervals.

---

# 5) LUT/Table registry (to avoid embedding payload in keys)

We need a runtime registry on both dealer and evaluator side:

* Compute a `hash(table_values)` (e.g., xxhash64)
* `table_id = registry.register(hash, device_ptr, in_bits, out_words)`
* Store `table_id` into header.reserved3.

In eval:

* lookup `device_ptr = registry.ptr(table_id)`
* pass pointer to vector Pika kernel.

This makes payload **public and shared once**.

---

# 6) Key generation (dealer side): gen_interval_lut SIL2 path

Inside `SecureGpuPfssBackend::gen_interval_lut(...)` add branch:

### Eligibility checks (B1)

* `desc.in_bits <= SIL2_MAX_IN_BITS` (default 13)
* payload is **public / same across all elements** (see compiler constraints below)

Then:

1. Build dense table from `(cutpoints, payload_flat)`:

    * for each interval i: fill x in `[cut[i], cut[i+1])` with payload[i]
    * table size `1<<in_bits`
2. Register dense table ⇒ `table_id`
3. Generate leaf‑block DPF key pair at secret alpha = `r_in` (Pika style)
4. Write SIL2 header + out_mask_share + dpf key bytes.

---

# 7) Compiler / configuration: how to actually use SIL2 in your pipeline

This is the critical part people usually miss:

### SIL2 only works when payload is **public across elements**

If your payload depends on `r_in` (like `flatten_coeffs_public_hatx` shifted coefficients), then there is **no way** to share the table across elements, and SIL2 won’t reduce memory.

So the *real* configuration rule is:

✅ Use SIL2 for **degree‑0 piecewise constant** interval LUT (lookup tables), and any case where payload does **not** depend on per‑element mask.
❌ Do not use SIL2 for “shifted coefficient payload” unless you redesign to avoid per‑element shift in payload.

### Practical selection rule (B1)

In `suf_to_pfss.cpp` compilation, for functions where:

* each piece degree == 0 (constant payload)
* in_bits <= 13
* intervals can be arbitrary but domain small enough to densify

then compile as “IntervalLUT SIL2 mode” and **do not shift cutpoints by r_in**.

Instead:

* keep the function in x‑domain (public)
* SIL2 key handles mask via DPF (exactly Sigma style).

### Env knobs (recommended)

Add flags:

* `SUF_SECURE_GPU_INTERVAL_LUT_VER=2` (enable SIL2 path)
* `SUF_SIL2_MAX_IN_BITS=13`
* `SUF_SIL2_FORCE_DEG0_ONLY=1` (safe default)

And in backend:

* if not eligible ⇒ fallback to SIL1.

---

# 8) Why DMPF is NOT your B1 (and what it’s for)

You asked if DMPF is suitable.

**Answer: DMPF is NOT the right primitive for B1**, because:

* DMPF key size and keygen time scale with number of programmed points `t`.
* IntervalLUT needs either:

    * a prefix/range aggregation capability, or
    * selection over k intervals
* DMPF by itself returns value at specific points, not range sum / prefix.
* You would still need a prefix‑sum layer ⇒ additional O(t log N) structure.

So for the efficiency win you want (remove `intervals` multiplier), the correct approach is **single DPF key at alpha=mask + public table**, i.e., LUT/Pika style.

DMPF becomes relevant only in **B2** if you insist on supporting large‑domain arbitrary cutpoints without densifying and without public payload tables. But that is *not* the fastest route to improve your system now.

---

# 9) Expected impact (what you will actually see)

For a typical LUT13 (8192 entries), `out_words=4`:

* SIL1: key bytes per element ≈ `intervals * in_bits * out_words * 8` (explodes)
* SIL2: key bytes per element ≈ `32 + 4*8 + dpf_key_bytes`

    * `dpf_key_bytes` ~ O(in_bits) seeds/cws (a few hundred bytes)
    * total maybe ~ 400–700 bytes per element

Online:

* SIL1: repeated DCF steps per interval
* SIL2: one Pika kernel pass (already optimized in your repo)

This is the kind of **order‑of‑magnitude** reduction you need to make the paper claims real.

---

# 10) Implementation checklist (exact files you will touch)

### A) Key format + parsing

* `include/proto/secure_pfss_key_formats.hpp`

    * finalize semantics of `SecureIntervalLutHeaderV2`

### B) Backend generation + eval

* `cuda/pfss_backend_gpu.cu`

    * in `gen_interval_lut(...)`: add SIL2 branch, emit `"SIL2"` keys
    * in `eval_interval_lut_many_device(...)`: detect `"SIL2"`, dispatch to new kernel

### C) New vector Pika kernel

* `cuda/pfss_kernels.cu`

    * implement `pikalut_dpf_dot_kernel_keyed_vec(...)`

### D) Table registry

* new `TableRegistry` helper (C++ in backend or runtime)

    * map table_hash/table_id → device_ptr
    * allocate once, reuse

### E) Compiler gating

* `src/compiler/suf_to_pfss.cpp`

    * if function is deg0 and small domain: compile as SIL2‑eligible interval LUT
    * avoid per‑element shifting of cutpoints/payload for these

---

# 11) What is still incomplete (so you can plan next steps)

To be transparent: the SIL2 design above is the **high‑impact, immediately landable B1** that will make your system faster and smaller for LUT‑type interval LUTs.

It does **not** magically solve the “shifted polynomial coefficients per element” case.
To make coefficient selection also SIL2‑like, you would need a bigger redesign (B2): avoid per‑element payload dependence on r, or embed polynomial evaluation differently (sigma/grotto spline protocols), or accept Beaver multiplications.

But B1 is the right thing to implement **first** because:

* it uses existing DPF/Pika code you already have
* it removes the worst key blow‑ups for LUT tasks
* it is fully GPU‑friendly and single traversal
* it matches your reserved “SIL2” comment perfectly

---

If you want, I can also provide:

1. a concrete C++ struct definition for the registry + exact serialization code,
2. the precise kernel signature and register pressure mitigation strategies, and
3. compiler criteria for “public payload detection” so you don’t accidentally miscompile secret payload into SIL2.
