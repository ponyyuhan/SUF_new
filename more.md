Below is the **full, concrete “delta-at-cutpoints + prefix aggregation” definition** you asked for **for SIL2/SPC2**, written in a way that Codex can implement.

I’m going to be very explicit about:

1. **What function is being FSS‑shared** (the delta-at-cutpoints representation).
2. **Where correction words live** (byte layout + placement).
3. **Padding rules** (to prevent multiplicity / wrap leakage).
4. **GPU eval + writeback strategy** (warp‑safe, coalesced, no atomics).

⚠️ **Important caveat / honesty upfront:**
A *true* “single key / single traversal / supports **multiple nonzero cutpoints in one DPF tree**” requires a **multi‑point programmable DPF / CDPF**, which is *not* what SigmaFSS SFD1 implements and not what libdpf provides out of the box (libdpf gives single‑point + prefix parity).

So what I’m giving you here is the **correct SIL2/SPC2 definition that matches your key‑format doc intent** and is implementable in your repo **today**:

* **Delta‑at‑cutpoints** function definition (exact),
* built using a **fixed‑count array of per‑cutpoint DCF/DPF-prefix keys**,
* evaluated in **one fused GPU kernel** (one traversal loop over bits; inner loop over cutpoints fixed and padded),
* so the runtime cost becomes **O(n_bits · K_fixed)** with **K_fixed small and constant**, and **no leak**.

This matches your “near single traversal” requirement in the sense used in your optimize/security notes: **one kernel, one pass over bits per input element**, no launch multiplicity, fixed schedule.

If later you want the **true multi‑point CDPF** (single α‑tree encoding all cutpoints), that’s a different cryptographic construction (requires programmable nodes or seed‑homomorphic PRG). I can outline it after, but this answer is the implementable version.

---

# 0) Core functional form (delta-at-cutpoints + prefix aggregation)

Both **SPC2** and **SIL2** are instances of the same piecewise-constant map:

Let cutpoints (sorted in the opened domain) be:

[
c_0 < c_1 < \dots < c_{M-1}
]

Let payload vectors be:

* SPC2: payload is a packed bitmask word(s)
* SIL2: payload is a vector of `out_words` u64

Define interval payloads (V_0,\dots,V_{M-1}) where interval (j) is:

[
I_j = [c_j, c_{j+1}) \quad (\text{cyclic, } c_M := 2^n)
]

Define cyclic deltas:

[
\Delta_0 = V_0 - V_{M-1}
]
[
\Delta_j = V_j - V_{j-1} \quad \text{for } j=1..M-1
]

Pick base:

[
B = V_{M-1}
]

Then for any opened input (u),

[
F(u) = B + \sum_{j=0}^{M-1} \Delta_j \cdot 1[u \ge c_j]
]

That’s the **delta-at-cutpoints + prefix aggregation** form.
All correctness reduces to implementing the **step functions**:

[
S_j(u) = 1[u \ge c_j]
]

---

# 1) What SIL2/SPC2 keys must contain

## Public leakage allowed:

* `in_bits`
* fixed `M` (interval count / threshold count)
* `out_words`

## Must be hidden (one-key private):

* actual cutpoint values (c_j)
* delta values (\Delta_j)
* multiplicities / collisions due to wrap or duplicates

So **key shape must be fixed**:

* always exactly `M` cutpoints
* always exactly `M` delta payload blocks
* always exactly `M` DCF-prefix keys (padded with dummy zero deltas if needed)

---

# 2) Exact correction-word placement (DCF-prefix / “DPF-based <”)

This is the part you said is “the hard one”: where CWs go in the key and how eval uses them.

### We use the standard *comparison DCF* layout, but we enforce:

* payload is vector (`out_words`)
* **no per-level group expansion** in the inner loop beyond generating the one PRG block needed for τ/state
* deltas are injected only via a final payload correction term (one block per cutpoint) + optional per-level scalar masks

Because you already have SFD1 (SigmaFSS) and its CW layout in your kernel, Codex can implement SIL2/SPC2 by **reusing the same CW layout**, but changing:

* `v_ptr` (the per-level group corrections) → removed/seeded/padded to zero
* only final correction `g` holds the full delta payload

### Concretely, per cutpoint j:

**DCF key blob:**

```
struct DcfKey {
  u64 s0;                 // root seed (or 2x u64 if 128-bit seed)
  u8  t0;                 // root control bit
  // For level ℓ=0..(in_bits-1):
  u64 cw_s[ℓ];            // seed correction word
  u8  cw_t[ℓ][2];         // 2 bits: t-corrections for (0-branch,1-branch)
  // Optional scalar CW for prefix aggregation; for SIL2 we treat these as 0.
  // Final payload correction (delta share):
  u64 g[out_words];       // delta payload correction
}
```

### Placement inside SIL2 key:

SIL2 key layout = header + base share + array of M DCF keys:

```
[ SecureIntervalLutV2Header ]
[ base_share: out_words * u64 ]
for j in 0..M-1:
  [ DcfKey_j bytes ]
```

SPC2 is identical except `out_words = ceil(num_thresh / 64)` and payload group op is XOR not add.

---

# 3) Padding rules (no leakage of multiplicity / wrap)

This is critical. You asked specifically “avoid leaking cutpoint multiplicities.”

### Rule A: fix M at compile time / descriptor time

* `M` must be public and fixed per template instance type.
* Even if some cutpoints coincide after masking (wrap), you still keep exactly M slots.

### Rule B: never deduplicate cutpoints

If two cutpoints map to the same opened value:

* keep both entries
* set one delta to 0 (or split delta arbitrarily)
* generate *full DCF key* for each, indistinguishable from real.

Reason: removing/merging would make key length or structure depend on multiplicity.

### Rule C: pad unused slots with dummy keys

If logical cutpoints < M (happens in PackCmp when predicate set smaller than max width):

* fill remaining slots with random cutpoint value and **delta = 0**
* generate full DCF keys anyway

### Rule D: sorting must not leak

Dealer sorts cutpoints to compute cyclic deltas, but **key order must not reflect sort** unless keys are pseudorandom. Since DCF keys are one-key private, ordering is safe, but easiest is:

* fix stable order = index order of original intervals/predicates, not sorted order
* store `perm[j]` internally when computing deltas

---

# 4) Evaluation definition (prefix aggregation)

Given opened `u` and party key K_b:

```
acc = base_share_b   // vector out_words
for j in 0..M-1:
    step_share = EvalDCFPrefix(K_b.dcf[j], u)   // returns additive share of delta_j if u>=c_j else 0
    acc += step_share
return acc
```

Same for SPC2 but operations are XOR and acc is packed bitmask.

**Important:** You must not branch on secret bits.
So EvalDCFPrefix must be **branchless** and produce correct share for all u.

---

# 5) GPU kernel strategy (writeback + coalescing)

Your repo already has:

* `packed_cmp_sigma_dcf_kernel_keyed`
* `interval_lut_sigma_dcf_kernel_keyed`

These kernels currently loop `for b in intervals` and call `eval_dcf_sigma_sfd1...`

### For SIL2/SPC2 v2:

You do **one kernel**, one grid over input elements. Each thread handles one element.

**Kernel skeleton:**

```cuda
idx = blockIdx.x * blockDim.x + threadIdx.x;
if idx >= n: return;

u = xs[idx];
acc[w] = base_share[w];  // register array

for j in 0..M-1:
    // Evaluate step function for cutpoint j
    tmp[w] = EvalDCF_v2(dcf_keys[j], u);   // register array
    acc[w] += tmp[w];

store acc[w] to out[idx*out_words + w];
```

### Writeback strategies

#### Strategy 1 (best): “thread-local accumulation”

* Keep `acc[w]` in registers
* Only store once at end
* No atomics, no shared memory
* Coalesced store if `out_words` small

#### Strategy 2: vectorized store for out_words<=4

Use `ulonglong2/4` or `uint4` vector types for coalesced global writes.

#### Strategy 3: warp‑striped for larger out_words

If `out_words` big (>=8), do:

* one warp cooperatively handles one element
* lane i handles word i, i+warpSize, …
* reduce / store without bank conflicts

But in SUF, `out_words` is usually small (coeff vector sizes <= 8).

---

# 6) Practical advice for Codex integration

### Minimal invasive path

1. Add new key headers:

    * `SecurePackedLtV2Header`
    * `SecureIntervalLutV2Header`

2. Add new keygen:

    * identical delta computation to SIL1/SPC1
    * but DCF keygen uses the v2 DCF format (no per-level v array)

3. Add new kernels:

    * `packed_cmp_dcf_v2_kernel_keyed`
    * `interval_lut_dcf_v2_kernel_keyed`

4. Wire into `eval_*_many_device` via new `desc.key_kind`.

### Fused kernel launch

* do not launch one kernel per cutpoint
* one kernel per template call
* fixed loop count M (public), so schedule is constant-time and leakage-safe

---

# What is still missing (and why)

You asked for *the true single-prefix-DPF multi-cutpoint scheme*.
That requires a primitive that can encode **multiple cutpoints into one tree** (multi-point programmable DPF or CDPF).

Your repo currently does not include that construction; neither SigmaFSS SFD1 nor libdpf support it directly in additive payload mode.

What I gave you **is the exact SIL2/SPC2 definition that is implementable** and preserves:

✅ delta-at-cutpoints correctness
✅ fixed-shape / padded leakage discipline
✅ explicit CW placement + layout
✅ GPU evaluation with constant schedule and optimal writeback

If you want, I can next provide:

* the **true multi-point CDPF** construction (requires either:

    * seed-homomorphic PRG, or
    * explicit programmable internal nodes + padded branching),
* and show how to encode **all Δ_j in one key** with one traversal.

But that is a deeper cryptographic redesign.

---

If you tell me which direction you want:

**(A)** implementable “fused fixed-M DCF v2” path (what I specified),
or
**(B)** true multi-point CDPF single-key scheme (cryptographically heavier),

…I can continue with the exact pseudocode + byte layouts for either.
