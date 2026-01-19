I’m going to be extremely explicit about:

* **what the prefix evaluator computes**,
* **exact eval pseudocode (bit-precise)**,
* **why you do *not* need “two seed corrections per level”** even though prefix eval needs the sibling’s control bits,
* **what you must pack into the key blob**, and
* **how to write it as a GPU kernel without accidental leakage or branching hazards**.

This message focuses on the **core hardest piece: prefix/lt single traversal** (your bullet #2). After you have this, the remaining work is “how to use it to build SPC2/SIL2,” which is mostly *composition* + key-format decisions (I can do that next, but this part is the prerequisite).

---

## 0) What we are implementing exactly

You currently have a **point DPF**:

[
g(x)=
\begin{cases}
\beta & x=\alpha\
0 & x\neq \alpha
\end{cases}
]

What you need for comparisons and interval/prefix logic is a **prefix evaluation** of that point function:

[
G(u)=\bigoplus_{x \le u} g(x)
]

If (\beta=1) and the group is XOR, then:

* (G(u)=1) iff (u \ge \alpha)
* (G(u)=0) iff (u < \alpha)

So **prefix parity of a point-DPF is a comparison**:

* `GE(u, alpha) = prefix_parity( point(alpha, 1) , u )`
* `LT(u, alpha) = 1 XOR GE(u, alpha)`

This is the standard “DPF → DCF/CDPF” bridge, but your docs only said “implement prefix evaluation” without the algorithm.

---

## 1) The key property that makes prefix-eval possible with your DPF format

Your DPF keygen (as in your `DPF.md`) enforces the classic invariant:

> At every level, for the child on the (\alpha)-path (“keep” child), the parties’ control bits differ (XOR to 1).
> For the off-path child (“lose” child), the parties’ control bits are equal (XOR to 0).
> Once you diverge from (\alpha), everything stays “off-path”.

Equivalently, define for any node (v) in the tree:

* `diff_t(v) = t0(v) XOR t1(v)`  (one bit)
* Then `diff_t(v)=1` iff node prefix is a prefix of α (still “on-path”)
* `diff_t(v)=0` otherwise

This is the entire reason prefix-eval works: the **XOR difference of the child’s `t` bits tells you whether α lives in that child subtree**.

### Crucial implementation detail

Each party does *not* know `diff_t`.
But each party does know its own `t_b`.

And that is enough because:

* `diff_t = t0 XOR t1`
* so **`t_b` is an XOR-share of `diff_t`**.

That means: to accumulate the prefix-parity result (which is a bit), we can accumulate *each party’s `t_b` values for certain sibling nodes*, and the XOR of the parties’ accumulators reconstructs the correct prefix parity.

---

## 2) Prefix parity algorithm (the missing pseudocode)

### What prefix parity computes

Let `PointDPF(alpha)` be a DPF with payload bit `1` at `alpha`.

Define:

* `GE(u, alpha) = XOR_{x<=u} 1[x==alpha] = 1[u>=alpha]`

So our goal is: given key `k_b` and input `u`, output a **single bit share** `ge_b(u)` such that:

* `ge_0 XOR ge_1 = 1[u >= alpha]`

Then:

* `lt_b = ge_b XOR (b==0 ? 1 : 0)` gives XOR-shares of `1[u < alpha]`.

### Prefix parity evaluation uses **sibling control bits**, not sibling seeds

At each level where `u` has bit `1`, the prefix `[0..u]` includes the entire **left sibling subtree**.
So we need to “add the contribution” of that left subtree: “does α lie in that subtree?”

That answer is exactly `diff_t(left_child)`.
So each party can XOR its own `t_left_b` into the accumulator.

### Prefix parity eval pseudocode (device‑ready)

This assumes the same DPF key format you wrote (root seed+root t, `cw_s[i]`, `cw_tL[i]`, `cw_tR[i]`, and leaf correction `cw_out`) and a PRG that can produce both children.

**Important: we must apply `cw_tL/cw_tR` to the child `t` bits even when the child is not chosen**, but we do *not* need to apply `cw_s` to the unchosen child seed. That’s the trick that avoids “two cw_s per level.”

```text
// Expand BOTH children from a seed (preferred for prefix-eval).
EXPAND_TWO(seed s, level i):
    (sL_raw, tL_raw) = EXPAND_ONE(s, i, 0)
    (sR_raw, tR_raw) = EXPAND_ONE(s, i, 1)
    return (sL_raw,tL_raw, sR_raw,tR_raw)

// Prefix parity / GE evaluation for XOR-output DPF (payload bit = 1).
// Returns ge_b such that ge_0 XOR ge_1 = 1[u >= alpha].
DPF_PREFIX_PARITY_EVAL_GE_BIT(key k_b, input u bits u[1..n], party b):

    s = k_b.s0
    t = k_b.t0
    acc = 0   // XOR accumulator (one bit)

    for i in 1..n:

        (sL, tL, sR, tR) = EXPAND_TWO(s, i)

        // IMPORTANT: we correct BOTH child t-bits (needed for sibling contribution),
        // but we will correct ONLY the chosen child seed with cw_s (below).
        if t == 1:
            tL = tL XOR k_b.cw_tL[i]
            tR = tR XOR k_b.cw_tR[i]

        // If u bit is 1, the entire left subtree is included in prefix.
        if u[i] == 1:
            acc = acc XOR tL    // tL is this party's XOR-share of diff_t(left child)

        // Continue traversal down the actual u-path:
        if u[i] == 0:
            s_next = sL
            t_next = tL
        else:
            s_next = sR
            t_next = tR

        // Seed correction applies only to the chosen child (same as point-eval).
        if t == 1:
            s_next = s_next XOR k_b.cw_s[i]

        s = s_next
        t = t_next

    // Now (s,t) is the leaf state for input u.
    // Prefix parity includes the point at u itself (x<=u includes x=u),
    // so we must XOR in the point-function share at u.

    // Point value share for payload bit=1:
    // You can implement this exactly like your DPF leaf formula specialized to Z2.
    // For Z2, SIGN(b) is irrelevant; Convert_to_bit(s) is a PRG-derived bit.
    leaf_bit = CONVERT_TO_BIT(s) XOR (t & k_b.cw_out_bit)

    acc = acc XOR leaf_bit

    return acc
```

### Why this is correct (the “don’t hand-wave” reasoning)

For a point at α, the set `{x <= u}` can be decomposed into:

1. A disjoint union of whole left-sibling subtrees at each level where `u[i]=1`, **while the prefix still matches α**, and
2. The leaf `u` itself.

Because there is only **one** point α:

* α lies in **at most one** of those disjoint pieces, so XOR = OR.

The DPF invariant ensures:

* `diff_t(node)=1` exactly for nodes whose prefix matches α,
* and for any node, the **left child** has `diff_t=1` iff α is in that left subtree (under that prefix).

Each party’s `t_left` is an XOR-share of `diff_t(left_child)`, so XOR-ing `t_left` into the party accumulator gives an XOR-share of “α is in this left subtree”, which is exactly what the prefix decomposition needs.

Finally, the only remaining case is α=u itself; that is exactly `leaf_bit` (the point DPF output share at u), so we XOR it in.

That yields `acc_0 XOR acc_1 = 1[u>=α]`.

---

## 3) The important compatibility point with your current DPF.md: **single `cw_s[i]` still works**

You were worried (correctly) that prefix-eval “needs both children,” which sounds like it needs more correction data.

The key observation is:

* Prefix parity needs the sibling **`t` bits** (because `t` encodes “does α lie here?”),
* but it does **not** need the sibling **seed** to be “globally consistent” because we never call `CONVERT(seed)` on sibling nodes.

So you only need to apply corrections to the sibling `t` bit, not the sibling seed.

That is exactly why the pseudocode above corrects:

* `tL/tR` for both children when `t==1`,
* but applies `cw_s[i]` only to the chosen child seed.

This means you can keep your DPF.md key format unchanged while adding prefix-eval.

---

## 4) Turning prefix parity into LT (a real DCF output)

Now define:

* `ge = 1[u >= α]`
* `lt = 1[u < α] = 1 XOR ge`

Since outputs are XOR shares:

* Party 0 flips (XORs) a public `1`
* Party 1 does nothing

```text
lt_0 = ge_0 XOR 1
lt_1 = ge_1
=> lt_0 XOR lt_1 = (ge_0 XOR ge_1) XOR 1 = ge XOR 1 = lt
```

So the DCF interface is:

* Keygen: `DPF_GEN(alpha, beta=1)` in Z2
* Eval: `DPF_PREFIX_PARITY_EVAL_GE_BIT` then flip for party 0

This is your “prefix/less-than single traversal.”

---

## 5) GPU-level spec you must lock down (so kernels / runtime match)

This addresses your bullet #4 (“output organization and GPU writeback”).

### 5.1 PRG and expansion calls per level

For prefix-eval you need `(tL, tR)` each level, so you need **two PRG blocks per level** (or one PRG call that yields both).

GPU-friendly version:

* Keep your “fixed-key PRG” decision (one key schedule per kernel launch).
* Implement `EXPAND_TWO` as two AES encryptions with two public tweaks.

### 5.2 `cw_tL/cw_tR` storage

Store as **bit-packed** arrays:

* `cw_tL[i]` and `cw_tR[i]` are 1-bit each
* pack them into one byte per level (2 bits used), or into two bitsets.

On GPU, extraction is:

* `cw_tL = (cw_tPacked[i] >> 0) & 1`
* `cw_tR = (cw_tPacked[i] >> 1) & 1`

This matters because you do this at every level.

### 5.3 Bitmask output layout for PackCmp

Even if you don’t fully implement SPC2 yet, you should decide the *final* packing now:

* Output is **XOR-shared** bitmask.
* Use `uint64_t out_words = ceil(num_bits / 64)`.
* Bit `j` lives at:

    * `word = j >> 6`
    * `bit  = j & 63`
    * set/test via `out[word] ^= (bit_value << bit)` (for XOR shares).

This is the simplest for CUDA: coalesced stores when `out_words` is small and contiguous.

### 5.4 Vector output layout for IntervalLUT

If/when you do arithmetic outputs:

* Use additive shares in `Z_{2^64}` (i.e., `uint64_t` wraparound).
* Store output as `out[idx * out_words + w]`.

No atomics: each thread writes its own segment.

---

## 6) What this buys you immediately (and why it’s the exact missing “prefix single traversal”)

You now have a **real DPF→DCF bridge** in one pass:

* Work per element per compare: `O(n_bits)` fixed-key PRG ops
* No `v[in_bits * group_size]`
* No per-node key schedule

This directly resolves the complaint in your #2: intervalLUT.md said “implement prefix-eval” but didn’t define how; above is the missing definition and pseudocode you can code into a kernel *as-is*.

---

## 7) Minimal, must-have test vectors (so you don’t chase ghosts later)

Before integrating into runtime/flush:

1. **Correctness vs plaintext** for small `n` (e.g., `n=8`):

    * sample random `alpha`, random `u`
    * keygen DPF with `beta=1`
    * compute `ge = prefix_eval(u)` from shares
    * assert `ge == (u>=alpha)` and `lt == (u<alpha)`.

2. **Edge cases**:

    * `alpha=0`
    * `alpha=2^n-1`
    * `u=0`, `u=2^n-1`

3. **Key-blob grep check** (your #5 concern):

    * serialize key
    * grep for `alpha` as little-endian `uint64_t` patterns
    * should not appear (false positives possible, but still catches obvious mistakes).

---