Below is the next “non-hand-wavy” chunk: **how to actually land SIL2/SPC2 in *this* repo**, and what structural changes you should make so the system follows the paper’s “two-template per gate instance” idea instead of falling back to StepDcf / SigmaGPU-style multi-round tasks.

I’m going to be very explicit about *where to change code*, *what key formats to define*, and *what the first landable version should look like*, even if it isn’t the final “optimal” version yet.

---

## 1) First structural change you should do immediately: stop gating IntervalLUT on `ell==0`

Right now you *already* have the compiler emitting IntervalLUT-capable coefficient descriptions (`build_coeff_interval_lut` exists), but in the composite gate generator you only select IntervalLUT when `F.l_out == 0`:

**File:** `include/gates/composite_fss.hpp`
**Function:** `composite_gen_backend_with_masks(...)`
**Current logic (problem):**

```cpp
if (F.l_out == 0 && lut_backend != nullptr) {
  coeff_mode = CoeffMode::kIntervalLut;
}
```

This means any gate that has boolean outputs (e.g., wrap/carry bits, sign bits, etc.) stays on `StepDcf` and drags in **StepDcf/Pred** key blow-ups.

### Change it to:

* Always prefer IntervalLUT when backend supports it, regardless of `ell`.
* Keep StepDcf only as a fallback if IntervalLUT is unavailable.

Concretely:

```cpp
auto* lut_backend = dynamic_cast<PfssIntervalLutExt*>(backend.get());
if (lut_backend != nullptr) {
  coeff_mode = CoeffMode::kIntervalLut;
} else {
  coeff_mode = CoeffMode::kStepDcf;
}
```

You will still generate the boolean predicates (`cut_pred_keys`) for `ell>0` paths exactly as you already do—this preserves the paper’s **two-template structure**:

* Template 1: PackedCmp (predicate bits for boolean outputs and for selecting intervals)
* Template 2: IntervalLUT (coeff selection)

This single change doesn’t magically shrink keys yet, but it stops the compiler from “doing the wrong thing structurally.”

---

## 2) Why SIL1 can’t win: the current DCF payload path is inherently “payload × in_bits”

In your secure GPU backend, vector-payload DCF uses SFD2 and stores per-level `v0` of shape:

```
in_bits × group_size × 8 bytes
```

That’s exactly the giant term that kills you.

**File:** `cuda/pfss_backend_gpu.cu`
**Function:** `SecureGpuPfssBackend::gen_dcf_sfd2_`
You can see it allocates/stores:

* per-level `cw` (16B) AND
* per-level `vcw` (group_size * 8B)

So for IntervalLUT you’re doing this *for every cutpoint*.

This matches the known issue that comparison-style FSS (DCF) typically introduces per-layer output-group correction material (often described as per-layer “VCW”). ([NTT Research, Inc.][1])
So: **any design that keeps “vector payload inside DCF” will keep paying “in_bits × out_words”.**

That’s why you reserved `SecureIntervalLutHeaderV2` as “DPF/CDPF-based”—you need a different primitive.

---

## 3) A landable SIL2 that actually removes `out_words × in_bits` from keys

You do **not** need to solve the most general FSS problem to get a big win.

For IntervalLUT, the function class is very special:

* domain: `u ∈ {0..2^n-1}` (public opened masked input)
* output: one of `m` vectors (coeff rows), piecewise-constant on `m` consecutive intervals
* `m` is typically small (often ≤ 32, maybe ≤ 64)

### SIL2 v0 approach: “Interval automaton” FSS (branching-program flavored, but specialized)

Instead of implementing IntervalLUT as a sum of **m-1 DCFs**, represent it as a *small deterministic automaton* that refines the interval as you scan bits of `u`.

#### State definition (crucial)

At prefix length `ℓ` (MSB-first), the set of all numbers matching prefix `p` is a contiguous range:

```
R(p) = [p<< (n-ℓ), (p+1)<< (n-ℓ) - 1]
```

Intersect this numeric range with your cutpoints list; the resulting possible interval indices form a contiguous index range:

```
state = (lo, hi)  where  0 ≤ lo ≤ hi < m
```

Start state is always `(0, m-1)`.

Transition on next input bit `b ∈ {0,1}` is:

* compute child prefix `p' = (p<<1) | b`
* compute `R(p')`
* compute which intervals overlap `R(p')` → yields new `(lo', hi')`

Dealer can compute these transitions because dealer knows:

* the shifted cutpoints (mask-aware)
* the interval partition (already in your keygen)

**Important:** The number of distinct `(lo,hi)` states encountered is ≤ m(m+1)/2. For m=32, that’s 528 states. Totally manageable.

At the end (after n bits), `R(p)` is a single number, so `lo==hi` and you know the interval index.

#### What the FSS has to do

Privately evaluate this deterministic automaton on public bits of `u` and output the payload vector of the final state.

This is exactly the type of function secret sharing that Boyle–Gilboa–Ishai show can be done via branching-program style constructions.
But here you can implement it with a **DPF-like seed-walk** specialized to deterministic width-W layered graphs.

#### Key size intuition

If you store:

* per-level correction material that scales with **#states × λ** (λ=16 bytes)
* final payload material that scales with **m × out_words × 8**

Then your key size becomes:

```
O(n * #states * 16) + O(m * out_words * 8)
```

And critically: **no “n × out_words” term.**

That’s the exact structural property you need to beat Sigma on key bytes.

---

## 4) How to integrate SIL2 into *your* codebase

You already reserved the header:

**File:** `include/proto/secure_pfss_key_formats.hpp`

```cpp
struct SecureIntervalLutHeaderV2 { ... };
```

### 4.1 Proposed binary layout after `SecureIntervalLutHeaderV2`

After the header, store:

1. **State table**

* `uint16_t num_states_total`
* `uint16_t level_offsets[n+1]`  // prefix-sums into a flat state array
* flat array of states:

    * for each state id:

        * `uint16_t lo, hi`
        * `uint16_t next0, next1`   // next state ids for bit 0/1

2. **FSS seed material**
   You want DPF-like evaluation: each party holds one seed for current state; expanding a seed yields candidate child seeds; correction words enforce consistency between parties so sums yield correct payload.

Store per-level:

* `cw_seed[level][state]`: 16B per state per level (or smaller if you compress)
* small control bits as needed

3. **Output payload shares**
   At end, once state is determined, output row should be:

* party0 gets random mask share
* party1 gets payload - mask share

So store `out_table[state]` as the party’s additive share vector (length out_words, u64).

This is just:

```
num_terminal_states(≈m) * out_words * 8 bytes
```

or store for all states but only terminal states used (better store only terminal `lo==hi` states).

### 4.2 Backend API touch points

**Keygen path:**
`SecureGpuPfssBackend::gen_interval_lut(...)` in `cuda/pfss_backend_gpu.cu`

Add a version switch:

* if env `SUF_SECURE_INTERVAL_LUT_VERSION=1` → current SIL1
* if env `...=2` → new SIL2

You already pass in:

* `in_bits`
* `out_words`
* `cutpoints`
* `output_rows`

So SIL2 keygen should:

1. build the `(lo,hi)` transition graph per element
2. generate FSS seed/correction material per element
3. secret-share payload rows into terminal state outputs

**Eval path:**
IntervalLUT evaluation currently calls:

* `lut_backend->eval_interval_lut(...)` indirectly via composite eval

You’ll add a new entry point (or branch inside existing) that recognizes `ILT2` magic and launches a new CUDA kernel.

### 4.3 New CUDA kernel you need

**File:** `cuda/pfss_kernels.cu`
Add something like:

* `interval_lut2_eval_kernel(...)`

Kernel logic per element:

1. read `u` (opened masked input)
2. initialize current state id = root (0)
3. for bit position `i` from MSB to LSB:

    * read input bit `b`
    * look up next state id: `state = next[b]`
    * update per-party seed using correction words (DPF-like)
4. at end, fetch `out_table[state_terminal]` share and write to output buffer

This is **one kernel launch** and exactly **one open flush for `u`** (same as today), but avoids the “m times DCF with vector payload” blowup.

---

## 5) What about SPC2 (PackedCmp) and Softmax?

You said the real runtime barrier is Softmax `RowMaxDiffSigmaGpuTask` with log2(cols) rounds and lots of opens. That’s a separate structural problem (reduction is adaptive).

But you *can* still do a big structural win in two steps:

### Step A: Use SPC2/SIL2 to eliminate StepDcf/Pred everywhere

This cuts key bytes and often reduces the number of “open flush phases” because fewer sub-tasks remain that force intermediate opens.

### Step B: Replace “interactive max-reduction” with a fixed-depth selection circuit compiled to FSS

Max reduction is hard to do with zero adaptivity if you insist on exact max. Two practical directions:

* **approx/max-clipping strategy**: trade a small accuracy hit to remove adaptive rounds
* **FSS branching-program for max**: build a fixed branching program that computes max over `cols` with fixed depth (but width grows). Might still be viable for typical Transformer softmax sizes if you exploit SIMD in CUDA.

I’m not going to pretend Step B is small; but Step A is the prerequisite, and SIL2 is the core missing piece.

---

## 6) What you should measure to know SIL2 is “working”

Add two counters in your benchmarks:

1. **Key bytes per gate type**:

    * interval lut keys should stop scaling with `out_words * in_bits`
2. **GPU time breakdown per phase**:

    * `open` phases count should not increase
    * IntervalLUT eval should be one kernel instead of looping over cutpoints

---

## 7) Minimal “next commit” checklist (so you can actually land this)

1. **Compiler / selector change**

* `include/gates/composite_fss.hpp`: remove the `F.l_out==0` condition for IntervalLUT choice.

2. **SIL2 plumbing**

* `include/proto/secure_pfss_key_formats.hpp`: keep `SecureIntervalLutHeaderV2` and document your chosen layout.
* `cuda/pfss_backend_gpu.cu`: add `gen_interval_lut_v2_` and a version switch.
* `cuda/pfss_kernels.cu`: add `interval_lut2_eval_kernel`.

3. **Config**

* env var `SUF_SECURE_INTERVAL_LUT_VERSION=2` (default to 1 for now).

4. **Correctness microtests**

* For random cutpoints and random payload rows:

    * generate keys for both parties
    * sample random `u`
    * run eval on both parties
    * verify reconstructed output equals expected interval row

---

If you want, in the next round I can:

* sketch the exact per-level correction word logic for the DFA-FSS (so you can code it directly), and
* show how to reuse the same seed-expansion primitives you already have for DPF-prefix code in `pfss_kernels.cu` to avoid inventing new PRGs.

But the key point is: **SIL2 should not be “DCF-with-vector-payload repeated m times.”** It should be “small-width state machine over input bits,” so the per-bit overhead depends on **λ and #states**, not **out_words**.

[1]: https://ntt-research.com/wp-content/uploads/2023/01/Programmable-Distributed-Point-Functions.pdf "https://ntt-research.com/wp-content/uploads/2023/01/Programmable-Distributed-Point-Functions.pdf"
