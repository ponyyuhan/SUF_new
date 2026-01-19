Below is the highest‑ROI next step to close the remaining gap (Softmax / nExp predicate blow‑up) **without introducing any new security assumptions** and while staying aligned with your paper’s theme (mask‑aware compilation + standard FSS templates):

## 1) Stop fighting `PackedPred` for Softmax: replace Softmax’s nExp + inverse with Sigma-style GPU kernels

Right now your bottleneck is exactly where Sigma wins: **Softmax’s “exp of negative values + inverse”**. Sigma avoids “many translated cutpoints per element” and instead uses a small constant set of primitives: **DPF‑based DReLU + TR + small LUTs**. Their nExp uses **two LUT8 instead of a LUT16** and explicitly notes that this shrinks online PRG work by ~100× in that subroutine. ([Ashish Panwar][1])

You already proved this pattern works in your repo with `SUF_GELU_MODE=sigma_gpu`. Do the same for softmax.

### 1.1 Implement `SUF_NEXP_MODE=sigma_gpu` (Sigma Figure 8)

Sigma’s nExp protocol is:

* 1× `DReLU_m`
* 1× `select_16`
* 1× `TR_{16,8}`
* 2× `LUT8`
* 1× `Mul_n`
* 1× `GapARS_{n,f}` ([Ashish Panwar][1])

This removes the entire “many cutpoints → many comparisons” structure that your `PackedPred` is currently paying for.

#### Concrete repo plan

**(A) Add a new gate header**
Create:

* `include/gates/nexp_sigma_gpu.hpp`

Mirror the structure of:

* `include/gates/gelu_sigma_gpu.hpp`

It should expose something like:

```cpp
struct NExpSigmaGpuMaterial {
  DReLUKey drelu_key;   // DPF-based
  LutKey lut0_key;      // LUT8
  LutKey lut1_key;      // LUT8
  // masks / r_in, r_out shares, and any TR/GapARS material you already store
  DeviceTableHandle T0; // 256-entry, public
  DeviceTableHandle T1; // 256-entry, public
};
```

**(B) Key/material generation**
Add a `GenNExpSigmaGpuMaterial(...)` next to your existing GeLU generator in `gelu_sigma_gpu.hpp` (or a parallel file). The only “secret” things are masks and DPF keys; `T0/T1` are public tables.

Sigma’s tables are defined as (for fixed-point with `f` fractional bits): ([Ashish Panwar][1])

* `T1[i] = floor( nExp(i / 2^4)  * 2^f )`
* `T0[i] = floor( nExp(i / 2^12) * 2^f )`

You can generate these on the host once (at init / build time) and upload to the GPU **once per process** (not per layer / not per inference). Put them in constant memory if small enough.

**(C) Runtime task**
In:

* `include/runtime/phase_tasks.hpp`

Add `NExpSigmaGpuTask` right next to your existing `Sigma-style GeLU sigma_gpu` task (you mentioned it around `#L3486`). Make it run fully on GPU like your GeLU.

Pseudo-flow inside the task (following Sigma Figure 8): ([Ashish Panwar][1])

1. `d_hat = DReLU_m( (x_hat - 2^16) mod 2^m ) XOR 1`
2. `c_hat = select_16(d_hat, (x_hat - (2^16 - 1)) mod 2^16) + (2^16 - 1)`
3. `(c1_hat, c0_hat) = TR_{16,8}(c_hat)`  // upper/lower bytes
4. `t1_hat = LUT8(T1, c1_hat)`
5. `t0_hat = LUT8(T0, c0_hat)`
6. `t_hat = Mul_n(t0_hat, t1_hat)`
7. `y_hat = GapARS_{n,f}(t_hat)`

**(D) Wire it into softmax path**
Find where Softmax calls your current nExp SUF gate and add:

* `SUF_NEXP_MODE=sigma_gpu` (or `SUF_SOFTMAX_MODE=sigma_gpu` that internally switches both nExp+inv)

Likely locations (based on your GeLU wiring style):

* `src/nn/attention_block.cpp` / `src/nn/softmax_block.cpp` (whichever hosts softmax)
* config plumbed similarly to `src/nn/mlp_block.cpp#L597` for GeLU

### 1.2 Implement `SUF_INV_MODE=sigma_gpu` (Sigma inverse protocol)

Sigma’s inverse for Softmax reduces the bitwidth and uses a **LUT with q bits** after truncation: ([Ashish Panwar][1])

* `p = f + ceil(log2(k+1))`
* `q = 6 + ceil(log2(k+1))`
* `z_hat = TR_{p, f-6}( z_hat mod 2^p )`
* output via `LUT_{q,n}(T)` where `T[i] = floor(2^(f+6) / i)` ([Ashish Panwar][1])

This is specifically designed to make inverse cheap in Softmax.

#### Concrete repo plan

**(A) Add**

* `include/gates/inv_sigma_gpu.hpp`

**(B) Add LUTq support (q=13 for k=128, f=12)**
For BERT‑tiny with `k=128`, `ceil(log2(k+1))=7`, so:

* `q = 6 + 7 = 13` → **LUT13**
* `p = f + 7` (if `f=12`, `p=19`)

So you need a `LUT13` implementation that’s fast on GPU.

Sigma references Pika’s LUT protocol and gives its key/online cost:
`keysize(LUT_{n,ℓ,T}) = keysize(DPF_{n,1}) + n + 2ℓ`, and online uses `2^(n-ν-1)` PRG calls with `ν = log2(λ+1)` plus `2ℓ` bits comm in 1 round. ([Ashish Panwar][1])

That’s *exactly* what you want for LUT13.

---

## 2) “If Pika is fast”: make your LUT engine truly Pika-style (especially for LUT13)

You already built LUT8 as “DPF EvalAll + dot” and it works. For LUT13, **don’t implement naive full 8192‑point EvalAll per element** unless you have to.

Sigma’s summary of Pika LUT is the key: the protocol is designed so online work scales like `2^(n-ν-1)` PRG calls (with `ν ≈ 7` for λ=128), *not* `2^n`. ([Ashish Panwar][1])
For `n=13`, that’s roughly `2^(13-7-1)=32` PRG expansions (plus a small number of evaluations), which is dramatically cheaper than 8192.

### 2.1 How to adapt inside your repo

#### 2.1.1 Unify LUT into one generic backend

Right now you likely have LUT8 wired “just for GeLU sigma_gpu.” Generalize it into:

* `include/proto/pika_lut.hpp` (new)
* `include/proto/secure_pfss_key_formats.hpp` (extend key format)
* `cuda/pfss_kernels.cu` (add LUT kernel(s))
* `cuda/pfss_backend_gpu.cu` (launch logic & batching)

Expose:

```cpp
template<int Nbits, int OutBits /* n in your ring */>
struct PikaLutKey { ... };

template<int Nbits, int OutBits>
void GenPikaLut(...);

template<int Nbits, int OutBits>
__global__ void EvalPikaLutKernel(...);
```

Then have:

* GeLU sigma_gpu use `PikaLut<8>`
* nExp sigma_gpu use two `PikaLut<8>`
* inverse sigma_gpu use `PikaLut<13>`

#### 2.1.2 Device table placement

* LUT8 tables: 256 entries → best in constant memory
* LUT13 table: 8192 entries → still small enough for GPU global memory; consider:

    * a single contiguous `uint64_t` array in device memory
    * optionally `__ldg`/read-only cache path

**Important:** table is public, so you can store it once globally and reuse across all layers/inferences.

#### 2.1.3 Batch shape / memory layout

To beat Sigma, LUT should be bandwidth-friendly:

* **Keys:** store as Structure-of-Arrays (SoA) so each warp loads contiguous words.
* **Inputs:** pack `x_hat` indices into `uint16`/`uint32` if `Nbits <= 16` to cut bandwidth.
* **Outputs:** write masked outputs directly into the next consumer buffer (avoid intermediate host staging).

---

## 3) Use Sigma’s “effective bitwidth” everywhere in Softmax (this is low-effort, big payoff)

Sigma’s global optimization is crucial for Softmax:

* max comparisons can be done with effective bitwidth `m = n - f + 1`
* the input to nExp has effective bitwidth `m = n - f + 2` ([Ashish Panwar][1])

So with `n=64, f=12`:

* max compare bitwidth: `m=53`
* nExp bitwidth: `m=54`

That reduces:

* DPF key sizes (linear in bitwidth)
* DPF eval work (linear-ish in depth)

If your current DReLU/compare is still hardcoded to 64-bit, update the softmax path to pass `m` down so all DReLU/compare/TR operate on `m` when safe.

---

## 4) After nExp+inv go sigma_gpu, the next bottleneck will move: be ready with these two follow-ups

### 4.1 Max-reduction comparisons: switch to DReLU-based compare (Sigma’s approach)

Sigma compares `x >= y` by rewriting to `x - y >= 0` and using a call to `DReLU_n` (with a domain constraint they enforce). ([Ashish Panwar][1])
Key size for `DReLU_n` is `keysize(DPF_{n-1,1}) + 1`, and online needs one DPF evaluation. ([Ashish Panwar][1])

If your max currently uses a heavier PackCmp/DCF path, replace it with:

* subtract (ring)
* DReLU
* select

And apply effective bitwidth (`m=n-f+1`) for those compares. ([Ashish Panwar][1])

### 4.2 Fuse kernels to cut memory traffic (often the real GPU killer)

Once you adopt sigma_gpu nExp and inverse, you’ll have a very “kernel-y” pipeline. The next big win is **fusion**, not more crypto:

* Fuse `(subtract max) + (x_hat - 2^16) mod 2^m` into the same kernel that produces the DReLU input.
* Fuse `TR16,8` + LUT index formation.
* In nExp, fuse the two LUT8 evaluations into one kernel launch if your LUT engine can take two tables (or one table and two indices).
* Fuse “mul + GapARS” if your GapARS is implemented as a small set of operations.

This is how you typically close the last 2–4× gap vs a highly tuned baseline like Sigma (once the protocol structure matches).

---

## 5) Practical “what to do next” checklist (in the order I would implement)

1. **Add `nexp_sigma_gpu`** (Figure 8) and plug it into Softmax. ([Ashish Panwar][1])
2. **Add `inv_sigma_gpu`** (inverse protocol) and plug it into Softmax. ([Ashish Panwar][1])
3. **Generalize LUT to `PikaLut<Nbits>` and implement LUT13 using the Pika-style cost model** (avoid naive EvalAll for 8192 points). ([Ashish Panwar][1])
4. **Enable effective bitwidth for softmax compares and nExp input** (`m=n-f+1`, `m=n-f+2`). ([Ashish Panwar][1])
5. Re-profile: if max-reduction dominates, **rewrite max compare to DReLU-based compare**, and keep it at effective bitwidth. ([Ashish Panwar][1])
6. Only after that: kernel fusion + device-resident caching of LUT tables and key material.

---

## Why this is the “most on-theme” option for your paper

* You are not inventing new primitives; you are adopting **standard FSS building blocks** (DPF-based DReLU, TR, LUT) and simply choosing a better lowering for Softmax than “many cutpoints comparisons.” Sigma explicitly frames LUT via Pika and DReLU as standard FSS components. ([Ashish Panwar][1])
* It doesn’t leak thresholds/cutpoints in plaintext (unlike sigmafast-style approaches you explicitly can’t use).
* It turns your remaining “PackedPred explosion” into a small constant set of masked, FSS-evaluable operations—the exact story you want: **compiler/IR chooses a structured decomposition**.

If you execute the steps above, your Softmax/NExp key material should stop being the dominant term, and your runtime should move toward “Sigma-like,” after which GPU engineering (fusion + caching) becomes the lever to actually *beat* Sigma.

[1]: https://apanwariisc.github.io/publications/pets-2024-sigma/sigma.pdf "https://apanwariisc.github.io/publications/pets-2024-sigma/sigma.pdf"
