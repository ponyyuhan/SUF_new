# Stage 3 Goal and Why Current Code Is Slow

## Current root cause (what must be eliminated)

Your secure backend currently implements:

* **PackCmp** as `T` independent **SigmaFSS DCF** evaluations (`SFD1`), each costing **O(n_bits)** AES/PRG steps.

* **IntervalLUT** as **O(#intervals)** independent **SigmaFSS DCF** evaluations, *each with vector payload* => cost and key size explode as:

* **GPU compute**: `O(#intervals · n_bits · (AES/PRG + group expansion))`

* **Key bytes**: `O(#intervals · n_bits · out_words)` in practice because SFD1’s `v` term scales like `in_bits * group_size`.

To beat SIGMA, we need to stop doing *“many independent DCF tree traversals”*. The fix must:

* Do **one (or near‑one) PRG/AES tree traversal** per element for the whole template.
* Keep the **payload/vector handling** outside the per‑level correction data (no `in_bits * out_words` blowup).

---

# High‑Level Design: Two New Secure Template Versions

Introduce **versioned** secure key formats (keep existing SPC1/SIL1 for regression):

* `SPC2`: **DPF‑based PackCmp** (multi-threshold in one pass).
* `SIL2`: **DPF‑based IntervalLUT** (piecewise constant / vector payload, single/near‑single traversal).

Both will be implemented with a **DPF‑style tree** (2‑way seed expansion per level, not DCF’s heavier “SFD1 group expansion per level”), and **payload correction only at the bottom** (leaf / terminal), not per bit level.

---

# Core Primitive: A GPU‑Friendly DPF With Leaf Payload Only

## What Codex must implement first

Implement a base primitive:

> **DPF with leaf payload only**
> Key size: `O(n_bits · seed_bytes + payload_bytes)`
> Eval cost: `O(n_bits · PRG)` + `O(payload_bytes)` once (at leaf), not per level.

This is the fundamental reason Sigma beats DCF for bit comparisons and why your IntervalLUT payload DCF is exploding.

### DPF key structure (conceptual)

A standard Boyle–Gilboa–Ishai style DPF has:

* Root seeds `(s0, t0)` per party
* For each level `ℓ = 0..n_bits-1`, a correction word:

   * `cw_s[ℓ]` (seed correction)
   * `cw_t[ℓ]` (control bit corrections)
* A terminal correction so that the leaf output equals the desired payload difference.

**Critical requirement:** Output payload is produced **only once**, from the final seed, via `Convert(seed) -> payload_bytes`, plus one terminal correction.
Do **not** store `v[in_bits * out_words]` like SFD1.

### PRG choice on GPU

Codex should use a GPU‑fast PRG:

* Prefer a **fixed‑key AES‑based expander** (one key schedule per kernel) or **ChaCha** (if you already have it).
* The PRG must expand a 128‑bit seed to:

   * two 128‑bit child seeds + two “t bits” (or embed t bits into seed LSBs).
* Absolutely avoid per‑node AES key schedule (that is what kills SFD1 on GPU).

Practical implementation trick (common in GPU DPF code):

* Use one AES key schedule stored in constant memory
* Compute child seeds by encrypting `(seed XOR tweak)` for two tweaks.

---

# SPC2: DPF‑Based PackCmp (Multi‑threshold, One Traversal)

## Public API change (recommended)

Today, `gen_packed_lt(in_bits, thresholds[])` forces you to materialize and hide each threshold with its own DCF key.

To get real packing, the API must carry the **public structure** of the predicate program, so evaluation can compute multiple bits from one DPF walk.

### Add a new descriptor type

Add:

```cpp
struct PackedCmpDescV2 {
  int in_bits;         // k
  int num_bits;        // number of predicate bits to output
  // public definition of each predicate:
  // bit j computes [ view_{k,c}(u) < (r + beta_j mod 2^k) ]
  struct Query { int k; int c; uint64_t beta; };
  std::vector<Query> queries;  // public list (from compiler)
};
```

**Important:** The key must NOT include `beta` values per-instance (they are public and shared across instances of the same compiled gate). Keep them in the descriptor / compiled program, not in the key blob.

### SPC2 key format (device‑parsable)

Add a new header in `include/proto/secure_pfss_key_formats.hpp`:

```cpp
struct SecurePackedCmpHeaderV2 {
  char magic[8];            // "SPC2...."
  uint32_t party;
  uint32_t in_bits;         // k (view width for this key)
  uint32_t num_bits;        // number of query bits in this group
  uint32_t dpf_key_bytes;   // bytes of embedded DPF key
  uint32_t wrap_bytes;      // bytes for secret-shared wrap/carry bits
  // followed by:
  //   uint8_t dpf_key[dpf_key_bytes];
  //   uint8_t wrap_shares[wrap_bytes];
};
```

### Keygen: build one DPF “compare engine” per (k,c) group

At key generation time (dealer side):

1. Group the `queries` by `(k,c)`.
2. For each group:

   * Compute `r_view = view_{k,c}(r_in)` (dealer knows full mask).
   * Generate a **DPF key for less-than** with threshold `r_view` *in the “DPF leaf-payload only” construction*.

      * The function evaluated will be used as the engine inside PackCmp.
3. For each query with offset `beta_j`, precompute wrap/carry:

   * `w_j = 1 if r_view + beta_j >= 2^k else 0`
   * Secret-share `w_j` and store in the key blob as a packed bitset.

**Key shape** is now:

* 1 DPF key per (k,c) group + a packed bitset of `num_bits` wrap bits
  not `num_bits` independent DCF keys.

### Eval: compute all predicate bits with one DPF traversal per element

On GPU, for each element and for each query in the group:

* You need the bit `[ view(u) < (r_view + beta) mod 2^k ]`.
* Use the identity that reduces this to a single DPF-based compare with threshold `r_view`, plus a tiny boolean combine using the wrap bit. In practice, implement it as:

1. `u = view_{k,c}(masked_input)` (public)
2. `t = (u - beta) mod 2^k` (public)
3. `c = [ t < r_view ]` using the DPF-compare engine (secret-shared output bit)
4. Combine with `w` (secret-shared wrap bit) and public `p = [u < beta]` to obtain final `[u < r+beta]` bit.

**Implementation note:** the boolean combine may require 1 AND of secret bits (`w & c`) per query. This is still cheap compared to running a full PRG tree per threshold, and can be implemented with a minimal boolean multiplication primitive:

* If you already have a boolean Beaver triple path, use it.
* If not, embed the combine into the DPF terminal correction (preferred, see below).

#### Preferred: bake the wrap logic into the terminal correction (no online ANDs)

Codex should avoid adding a boolean multiplication kernel inside PackCmp.

Instead, incorporate the wrap logic by generating a DPF that directly outputs the final bit share for each query index. You do this by making PackCmp a **vector-output DPF** where:

* domain = `u` (k-bit)
* output group = bitmask of `num_bits` bits
* function value at `u` equals the packed result vector

This is the “real PackCmp”: **one DPF traversal returns the whole bitmask**.

How to realize it with leaf payload only:

* The predicate vector is piecewise constant with cutpoints at `{(r+beta_j) mod 2^k}`.
* Build a difference representation over the circle:

   * Sort the cutpoints in masked domain (dealer only).
   * Each interval corresponds to a constant bitmask.
   * Let `Δ_i = mask(interval_i) XOR mask(interval_{i-1})` be the bitmask delta at each cutpoint.
* Now PackCmp becomes a prefix‑XOR of sparse deltas:

   * `out(u) = base XOR (prefix_xor of deltas at cutpoints <= u)`

Codex then builds a **single DPF that outputs `Δ_i` at each cutpoint** (point function values are bitmasks), and evaluation returns prefix XOR at input `u`.
This matches the “one traversal” requirement:

* One DPF evaluation that computes a prefix XOR (like Sigma’s DPF-based comparison does for a single point), but now the leaf payload is a bitmask.

**Key idea:** Instead of “T independent DCFs”, PackCmp is one “prefix DPF” over bitmask payload.

This is the correct cryptographic structure to hit **~O(k)** PRG expansions, independent of `T` (aside from output packing cost).

---

# SIL2: True IntervalLUT Template (Vector Payload, Single / Near‑Single Traversal)

IntervalLUT is the same structural problem as PackCmp, but with:

* Output group = arithmetic vector payload (e.g., `out_words` u64s)
* Piecewise constant across intervals defined by masked cutpoints.

The fix mirrors the PackCmp fix:

## SIL2 key format

Add:

```cpp
struct SecureIntervalLutHeaderV2 {
  char magic[8];              // "SIL2...."
  uint32_t party;
  uint32_t in_bits;           // n
  uint32_t out_words;         // vector length
  uint32_t intervals;         // M
  uint32_t dpf_key_bytes;
  uint32_t payload_bytes;     // intervals*out_words*sizeof(u64) (or delta form)
  // blob:
  //   uint8_t dpf_key[dpf_key_bytes];         // the single traversal “prefix DPF”
  //   u64 payload_shares[intervals][out_words]; // OR delta shares + base share
};
```

**Security rule:** key shape depends only on `(in_bits, out_words, intervals)`, not on cutpoint values.

## Keygen: encode a piecewise-constant LUT via sparse deltas + prefix DPF

Dealer knows:

* cutpoints in masked domain: `c_1 < ... < c_{M-1}` (hidden)
* payload vectors: `P_0..P_{M-1}` (secret-shared model coefficients)

Construct delta vectors:

* Choose a base vector: `B = P_{M-1}` (or `P_0`)
* For each cutpoint `c_i`, define `Δ_i = P_{i-1} - P_i` (or XOR for GF(2) case)

Then the LUT output can be expressed as:

* `f(u) = B - Σ_{i=1..M-1} Δ_i * 1[u < c_i]`
  or equivalently as a prefix sum/difference form depending on chosen base direction.

**But do not implement this as M−1 DCFs.**
Instead, build a **single prefix DPF** that places `Δ_i` at `c_i` (point values), and whose evaluation returns the **prefix-sum of those point values up to `u`**. That gives you `Σ Δ_i` in one traversal; then add/subtract base.

This is exactly the same pattern as DPF-based comparison (prefix parity), generalized from:

* scalar payload in GF(2)
  to
* vector payload in Z₂⁶⁴ (or your ring)

### What Codex implements on GPU

Implement a “prefix evaluation” for the DPF:

* Input: public `u`
* Output: party’s share of `prefix_sum_{x<=u} g(x)` where `g` is the DPF point-function payload map

This is the “IntervalLUT single traversal”: cost is O(n_bits) PRG expansions + one Convert at the end.

### Important: output conversion at terminal only

The DPF Convert must expand the final seed to `out_words` u64s (e.g., via AES-CTR/ChaCha stream), and then apply one terminal correction so that the two parties’ outputs add to the right vector.

This avoids per-level `out_words` work and kills the `in_bits*out_words` blowup.

---

# GPU Implementation Details (Concrete Coding Tasks)

## 1) Add new magic dispatch paths

Files:

* `include/proto/secure_pfss_key_formats.hpp`
* `cuda/pfss_backend_gpu.cu`
* `cuda/pfss_kernels.cu`

Tasks:

* Define `SPC2_MAGIC` and `SIL2_MAGIC` (8 bytes like others).
* Extend the device-side parse helpers to recognize SPC2/SIL2 and parse fixed header + blob offsets safely (no dynamic allocation).

## 2) Implement DPF keygen on CPU side (inside secure backend)

In `cuda/pfss_backend_gpu.cu`, add helper functions:

* `gen_dpf_prefix_bitmask(...)` for SPC2
* `gen_dpf_prefix_vector(...)` for SIL2

They must output two party blobs with identical shape:

* party id in header
* identical blob size
* only share differences

If you already have Sigma CPU code for DPF keys, reuse; if not, implement minimal DPF keygen.

## 3) Implement GPU kernels

Add kernels in `cuda/pfss_kernels.cu`:

### (a) PRG expansion kernel inline device function

A `__device__ inline` PRG that:

* takes 128-bit seed
* returns left/right child seeds + t bits

Make it:

* branchless
* use registers; minimal local memory
* optionally use `__ldg`/constant memory for AES round keys if AES-based

### (b) DPF prefix-eval kernel for vector payload (IntervalLUT)

Signature idea:

```cpp
__global__ void dpf_prefix_sum_vector_kernel(
    const uint8_t* keys_flat, size_t key_bytes,
    const uint64_t* u_in, size_t N,
    int in_bits, int out_words,
    uint64_t* out_flat);
```

Behavior:

* Parse `SecureIntervalLutHeaderV2`
* Run prefix-eval for the embedded DPF at input `u_in[idx]`
* Add/sub base vector share from key blob
* Write `out_words` u64s to `out_flat[idx*out_words + j]`

### (c) DPF prefix-eval kernel for bitmask payload (PackCmp)

Signature idea:

```cpp
__global__ void dpf_prefix_xor_bitmask_kernel(
    const uint8_t* keys_flat, size_t key_bytes,
    const uint64_t* u_in, size_t N,
    int in_bits, int num_bits, // packed into u64 words
    uint64_t* out_words_flat);
```

Behavior:

* Parse `SecurePackedCmpHeaderV2`
* Prefix-eval bitmask payload (XOR group)
* Emit packed output as u64 words

## 4) Superbatch integration and flush reduction

Your profile shows large overhead from many flushes.

Codex must ensure that:

* Each PackCmp call enqueues exactly **one** PFSS job (SPC2), not T subjobs.
* Each IntervalLUT call enqueues exactly **one** PFSS job (SIL2), not (M−1) subjobs.

Update the job planner in the PFSS superbatch runtime so SPC2/SIL2 count as single jobs with one kernel launch per batch.

---

# Compiler / Gate‑Generation Changes (So the Backend Can Pack)

## Key point

Packing only works if the compiler emits PackCmp/IntervalLUT as a **single template instance** with:

* public shape: `(in_bits, num_bits)` or `(in_bits, intervals, out_words)`
* public program metadata (e.g., query list) stored in the compiled graph/descriptor
* instance secrets (cutpoints) hidden only inside the DPF key blob

### Tasks

* Modify the SUF compiler path that currently expands predicates into many comparisons to instead:

   * build the “predicate program” as a single PackCmp template instance
   * build the “coefficient program” as a single IntervalLUT instance

That is consistent with your paper’s “two-template theorem”; it is also required to eliminate `O(#thresholds)` DCF key material.

---

# Correctness Tests Codex Must Add

## 1) Deterministic unit tests for SPC2 and SIL2

Add C++ tests that:

* sample random `r`, random `u`, random betas / cutpoints
* run keygen to get `k0`, `k1`
* eval on each party separately (CPU reference eval is fine for test)
* combine outputs and compare with plaintext reference:

   * PackCmp: bit vector `[u < (r+beta_i mod 2^k)]`
   * IntervalLUT: payload table lookup

Include edge cases:

* `r` near wrap boundary
* `beta=0`, `beta=2^k-1`
* cutpoints near 0 and near `2^k-1`
* repeated betas should be rejected or explicitly supported (define behavior)

## 2) Shape‑leak checks

Add an automated check that serialized key bytes:

* never contain raw cutpoints or thresholds (search the blob for those u64 patterns in little-endian; allow false positives but flag)
* key length depends only on public shape

---

# Performance Checklist (What to Verify After Implementation)

After Stage 3, you should see:

* `preprocessing.key_bytes` drops sharply because:

   * PackCmp: **one key per template**, not one per threshold
   * IntervalLUT: **one key per LUT**, not one per interval DCF with `in_bits*out_words` payload corrections
* `pfss.num_flushes` drops (one flush per template batch)
* `pfss.eval_eval_ns` drops dramatically because:

   * only one DPF traversal per template evaluation
   * no repeated per-interval DCF traversals
   * no per-level group expansion for vector payload

---

# Summary of What Codex Must Implement (Actionable Worklist)

1. **Add new key magics + headers**: `SPC2`, `SIL2`.
2. **Implement a DPF whose payload is applied only at the terminal** (no `v[in_bits*out_words]`).
3. **Implement prefix-eval for DPF**:

   * XOR-prefix for bitmask payload (PackCmp)
   * add-prefix for vector payload (IntervalLUT)
4. **Replace PackCmp’s “T DCFs”** with **one SPC2 DPF key** that returns the packed predicate vector.
5. **Replace IntervalLUT’s “(M−1) vector DCFs”** with **one SIL2 DPF key** that returns the selected coefficient vector.
6. **Fuse kernels / reduce flushes** in PFSS superbatch so SPC2/SIL2 are single jobs.
7. Update compiler/gate generation so those templates are actually invoked as single instances, not expanded.

---

If you want, I can also provide **pseudo-code for DPF keygen + prefix-eval** in the exact “BGI-style DPF” form (with correction words per level and terminal correction), matching the data layout you already use (`u64`, `osuCrypto::block`, and your existing device parse helpers).
