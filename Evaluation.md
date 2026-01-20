Below is a concrete, ICML-grade Evaluation plan **and** a “Codex workplan” that starts from setting up **Sigma** and **SHAFT**, runs a suite of **kernel + block + end-to-end** benchmarks, and ends with **aggregation + plots + LaTeX tables + analysis text**.

I’m grounding the baseline run/build details in your repo docs (SUF/Sigma harness) and in the SHAFT artifact appendix.
Sigma/SUF harness + build pointers:
SHAFT scripts + timing formula + example outputs:

---

## 1) What ICML reviewers will expect you to demonstrate

Your paper’s contribution is not “a faster exp” but a **compiler + typed IR** that turns many brittle fixed-point scalar kernels into a uniform “two-template + share post-processing” pipeline. Your Evaluation should therefore hit **four axes**:

1. **Performance (online):** latency + communication for nonlinear/helper-heavy workloads.
2. **Preprocessing footprint:** key/triple material and keygen time (your compiler makes this dramatically smaller / better structured).
3. **Generality / extensibility:** new kernels can be added without bespoke FSS decompositions or key formats.
4. **Accuracy:** fixed-point + approximations don’t harm task accuracy beyond baseline.

To compare fairly:

* **Sigma** is your closest baseline (same “masked-wire + FSS on (\hat{x})” paradigm, GPU optimized).
* **SHAFT** is a strong “non-FSS secure transformer inference” baseline with a full artifact for **softmax + GELU + transformer inference cost/accuracy**.

---

## 2) Experimental setup section (what to write + what to measure)

### Hardware/OS/Software

Report:

* GPU model (e.g., A100 80GB / RTX 4090), CPU model, RAM
* CUDA version, compiler, OS
* For Python baselines: Python version, PyTorch version
* Threading settings (important for single-process 2-party harness; your README explicitly warns OpenMP oversubscription can inflate open/comm time)

### Network model

Do **both**:

* **Raw protocol metrics**: (i) computation time, (ii) communicated bytes, (iii) number of rounds/flushes.
* **Projected wall-clock under network settings**:
  [
  T_{\text{proj}} = T_{\text{comp}} + \frac{2\cdot \text{bytes}}{\text{bandwidth}} + \text{rounds}\cdot \text{RTT}
  ]
  SHAFT already uses this style and provides scripts that output comp time / bytes / rounds, plus an example LAN calculation.

**Important:** be explicit about units (bytes vs bits). Keep the same convention across all three systems.

### Repetitions

ICML reviewers like stability:

* Warmup 1 run.
* Repeat 5–10 times.
* Report median + IQR or mean ± std (choose one and stick to it).

---

## 3) Baselines: what you need to build/run

### 3.1 Sigma (EzPC GPU-MPC)

In your repo, there is already a build helper and a harness:

* Build Sigma via `bash scripts/build_sigma.sh`.
* End-to-end harness: `python3 bench/run_sigma_vs_suf.py --config ...`.

Also note: your benchmark report already parses Sigma outputs from `external/sigma_ezpc/.../sigma/output/`.

### 3.2 SHAFT

SHAFT provides:

* **Unit-test microbenches** for softmax and GELU (Python scripts).
* **End-to-end transformer inference cost scripts**: `test_<model>_<in_size>_{comp,comm}.sh` and an explicit formula for projected time.
* **Accuracy scripts** (e.g., SST-2/QNLI/CoLA).

---

## 4) Benchmark suite design (what experiments to run)

### Experiment A: Scalar-kernel microbenchmarks (core claim: compiler-generated “two-template” kernels are fast)

**Goal:** show SUF reduces online latency/bytes for the *exact kernels that dominate transformers*: truncation/ARS, GeLU/SiLU, nExp, reciprocal/rsqrt.

**Workloads (minimum set):**

1. **Truncation / ARS helper** (and any variants you support)
2. **GeLU** (spline/poly approximation)
3. **SiLU** (or similar smooth activation)
4. **nExp** (exp approximation used inside softmax)
5. **Reciprocal** (softmax normalization)
6. **Rsqrt** (LayerNorm)

**Metrics:**

* time (ms) per 1M elements (or per token), bytes communicated, number of opens/rounds, plus predicate count (T) and LUT payload dimension (p) as sanity that you’re staying “two-template.”

**Baselines:**

* Sigma for all kernels it supports.
* SHAFT for (at least) GELU and Softmax-related microbench outputs; their artifact explicitly runs `python run_test_gelu.py` and provides output format (time/bytes/rounds).

**Figures/Tables:**

* **Figure 1:** “Per-kernel online time” bars (Sigma vs SUF; plus SHAFT where available).
* **Figure 2:** “Per-kernel online communication” bars.
* **Table 1:** kernel shapes: ((m,d,r,\ell,T,M,p)) for each SUF gate (this directly supports your “structured IR → predictable backend shapes”).

---

### Experiment B: Block-level benchmarks (softmax / layernorm)

**Goal:** reviewers will ask “kernel speedups are nice, do they translate into real blocks?”

Run:

* Softmax block (max-reduction + nExp + sum + reciprocal + multiply)
* LayerNorm (mean/var reductions + rsqrt + affine)

Your repo already has a block bench target: `build_ninja/bench_softmax_norm`.

Compare to:

* Sigma’s block performance (if it exposes similar benchmarks, otherwise measure inside end-to-end and attribute by phase).
* SHAFT softmax microbench (`python run_test_softmax.py`) per artifact appendix.

**Figures/Tables:**

* **Figure 3:** softmax and layernorm block latency/bytes vs batch size and sequence length.

---

### Experiment C: End-to-end transformer inference (the “headline table”)

**Goal:** show real end-to-end wins; must include at least one encoder and one decoder.

**Models (choose overlap across baselines):**

* BERT-base (seq=128, batch=1) — supported by Sigma harness and SHAFT scripts.
* BERT-large (seq=128, batch=1) — supported by Sigma harness and SHAFT paper claims.
* GPT-2 (seq=128, batch=1) — supported by both (SHAFT has text-generation folder; SUF harness already includes gpt2).
  Optional (nice-to-have):
* ViT-base (SHAFT supports; include if SUF has it, otherwise omit or clearly state “not implemented”).

**Metrics:**

* Online time (s), online bytes (GB), online rounds/flushes.
* Preprocessing key material size and keygen time (Sigma vs SUF). Your repo already reports these and shows large gaps.
* For SHAFT: comp time, comm bytes, rounds from `test_*_{comp,comm}.sh` scripts.

**Primary table (ICML-style):**

* Table 2: End-to-end online (Sigma vs SUF) + SHAFT cost metrics for overlapping models.
* Table 3: Preprocessing (Sigma vs SUF): keygen time + key size.

You already have a template table structure in `benchmark_report.md` showing “Sigma online vs SUF online” and “Sigma keygen/key vs SUF keygen/key”. Use the same format but extend it to include SHAFT columns where applicable.

---

### Experiment D: Scaling experiments (seq length + batch size)

**Goal:** demonstrate your wins are not a single-point artifact.

**Sweep:**

* Sequence length: 32, 64, 128, 256 (and 512 if feasible)
* Batch size: 1, 2, 4 (or 1/4/8 if memory allows)

Report:

* throughput (tokens/s) and latency (s)
* bytes and rounds
* show “speedup vs Sigma” vs seq length.

---

### Experiment E: Extensibility case study (this is *your* differentiator vs Sigma/SHAFT)

**Goal:** quantify engineering overhead and validate correctness for a “new” primitive.

Pick a function that is realistic in transformers but not necessarily in baseline:

* e.g., **Softplus**, **Tanh**, **Sigmoid**, or a new **custom quantized clamp** used in some model.

What to show:

* a SUF descriptor file / code (~tens of lines),
* compiler emits exactly the same backend templates (PackCmp + IntervalLUT) with mask-independent shapes,
* performance is comparable to existing kernels.

Metrics:

* “Lines of code changed” (or number of descriptor lines) + “no backend changes”.
* compile time and emitted shapes.

Include:

* Table 4: “Adding a new nonlinearity: SUF vs hand-coded baseline” (qualitative + quantitative).

---

### Experiment F: Accuracy

Two layers:

1. **Function-level numeric error** (max/avg error) vs float32 reference for each scalar kernel.
2. **Task-level accuracy** for BERT tasks (SST-2, QNLI, CoLA are standard and are exactly what SHAFT uses).

Your repo already has an “accuracy bench” scaffold comparing PyTorch/Sigma/SUF fixed-point emulation, with frac_bits and bitwidth matching Sigma for comparability. Leverage that; extend to include SHAFT if needed.

---

### Experiment G: Ablations (reviewer-proofing)

Ablations that directly validate your design claims:

1. **Open packing ON/OFF** and device pack threshold effects (SUF has explicit toggles; GPU defaults described in README).
2. **PFSS backend choice** (sigmafast vs grotto/libdpf) if you want to show backend modularity.
3. **Composite-FSS block size** (`SUF_COMPOSITE_BLOCK`) vs performance/bytes (to show batching behavior).
4. **Shape padding overhead**: “always allocate (M=m+1)” vs “exact (M)” (measure overhead to justify leakage discipline).

---

## 5) Codex implementation plan (from setup → running → parsing → plots → LaTeX)

### Phase 0: Create a unified experiment harness layout

**Codex task 0.1:** add this directory structure (if not already):

```
bench/
  runners/
    run_suf.py
    run_sigma.py        # if not already covered by run_sigma_vs_suf.py
    run_shaft.py
  parse/
    parse_suf_json.py
    parse_sigma_out.py
    parse_shaft_logs.py
  analysis/
    aggregate.py
    plots.py
    latex_tables.py
  results/
    raw/
    aggregated/
```

Acceptance: you can run one command and it produces `bench/results/aggregated/results.csv` and `results.json`.

---

### Phase 1: Sigma + SUF runner (likely mostly already exists)

Your repo already provides:

* `bench/run_sigma_vs_suf.py` and configs.
* `build_ninja/bench_suf_transformer` with GPU defaults and profiling toggles.

**Codex task 1.1:** extend the existing harness to optionally:

* sweep `seq_len` and `batch`,
* run `n_reps` times,
* write raw logs into `bench/results/raw/<framework>/...json`,
* and write a normalized record (common schema) per run.

Common schema suggestion:

```json
{
  "framework": "suf|sigma|shaft",
  "bench_kind": "kernel|block|e2e",
  "bench_name": "gelu|softmax|bert_base|gpt2|...",
  "device": "gpu|cpu",
  "seq_len": 128,
  "batch": 1,
  "time_comp_s": 0.0,
  "time_total_s": 0.0,
  "comm_bytes": 0,
  "rounds": 0,
  "preproc_bytes": 0,
  "preproc_time_s": 0.0,
  "notes": { "n_bits": 50, "frac_bits": 12, "T": 123, "M": 17, "p": 64 }
}
```

For SUF:

* parse from SUF JSON logs. Your README points out where to inspect settings like `preprocessing.open_pack_device_min_words` in JSON logs and how to enable per-phase timing breakdowns with `SUF_BENCH_PROFILE=1`.

For Sigma:

* parse from Sigma output directory already used by your report.

---

### Phase 2: SHAFT runner + parser (new)

SHAFT gives you two types of experiments:

#### 2.A Unit-test microbench (softmax, gelu)

Artifact commands:

* `python run_test_gelu.py` (outputs max/avg error + time/bytes/rounds lines)
* `python run_test_softmax.py` (similar style; referenced in artifact)

**Codex task 2.1:** implement `bench/runners/run_shaft.py` with subcommands:

* `--mode unit --which gelu|softmax`
* runs the script inside the SHAFT repo
* captures stdout to `bench/results/raw/shaft/<timestamp>_<which>.log`
* parses per-input-size lines into normalized JSON entries.

Parsing examples from artifact:

* GELU output example includes:

    * `max error: ... avg error: ...`
    * `(128,3072) time: 0.2203s, bytes: 354 MB, rounds: 19`

So parser should:

* extract `time`, `bytes`, `rounds`, input shape(s), and error stats.

#### 2.B End-to-end transformer inference cost + comm

Artifact scripts:

* `bash test_bert_base_128_comp.sh`
* `bash test_bert_base_128_comm.sh`
  They output:
* `comp time: ...`
* `comm byte: ... GB, round: ...`

**Codex task 2.2:** extend `run_shaft.py`:

* `--mode e2e --model bert_base|bert_large|gpt2|vit_base --seq 128`
* runs both `_comp.sh` and `_comm.sh`,
* parses comp time, comm bytes, rounds,
* writes a normalized record.

**Codex task 2.3:** implement a “projected runtime calculator” in `bench/analysis/aggregate.py`:

* takes (bytes, rounds, comp_time) and network presets, outputs projected times.
* include presets:

    * LAN: 1 Gbps, RTT 0.5 ms (as SHAFT describes)
    * WAN: pick a realistic one (e.g., 100 Mbps, 20 ms)
    * Datacenter: 10–25 Gbps, 0.1 ms

---

### Phase 3: Aggregation + plots + LaTeX tables

**Codex task 3.1:** `bench/analysis/aggregate.py`

* Input: all normalized JSON entries in `bench/results/raw/**`.
* Output:

    * `bench/results/aggregated/results.csv`
    * `bench/results/aggregated/results.json`
    * `bench/results/aggregated/summary_by_model.csv` (median + std)

**Codex task 3.2:** `bench/analysis/plots.py`
Generate:

* `fig_kernel_time.pdf`
* `fig_kernel_comm.pdf`
* `fig_e2e_time.pdf`
* `fig_preproc_size.pdf`
* `fig_scaling_seq.pdf`

Use matplotlib; add error bars.

**Codex task 3.3:** `bench/analysis/latex_tables.py`
Emit paste-ready LaTeX tables:

* `tab_e2e.tex` (Sigma vs SUF vs SHAFT for overlapping models)
* `tab_preproc.tex` (Sigma vs SUF keygen + key size)
* `tab_kernels.tex` (kernel microbench)

You already have a working “Sigma vs SUF” summary format in `benchmark_report.md`. Mirror that layout in LaTeX.

---

## 6) How to write the Results + Analysis narrative (what to emphasize)

### 6.1 Kernel-level

Explain **why** SUF wins:

* constant number of backend calls (≤2),
* batching-friendly,
* uniform post-processing (Horner + Boolean combining),
* fewer bespoke mask-correction stages.

Show a breakdown when possible using SUF’s profiling fields (`online_profile.*`) enabled by `SUF_BENCH_PROFILE=1`.

### 6.2 End-to-end

Highlight:

* wins are larger on models with more nonlinear/helper pressure (often BERT-large, GPT-like).
* communication and key transfer often dominate once compute is fast (Sigma also observes this in its own discussion; you can reference in related work, not necessarily eval).

### 6.3 Preprocessing footprint

This is your “compiler leverage” story:

* Sigma key sizes can be huge; SUF key sizes are dramatically smaller under your template interface and padding discipline (your report already shows large gaps).
* Explain that SUF prevents mask-dependent shapes and encourages reuse/batching.

### 6.4 Extensibility case study

Make it concrete:

* “We added Softplus in X lines (descriptor only) and got competitive performance with no backend changes.”
* Provide emitted shapes and confirm they are mask-independent by running shape checks on many random masks.

### 6.5 Accuracy

Report:

* function error (max/avg), and
* task accuracy vs plaintext / baseline.

SHAFT’s artifact uses GLUE tasks and provides guidance for SST-2 / QNLI / CoLA; your SUF repo already has an accuracy scaffold aligned to “Table 4 style” with fixed-point rounding choices.

---

## 7) Minimal “ICML Evaluation Package” you should aim to produce

If you want to keep scope tight but strong, do:

* **Table 2 (headline):** End-to-end (BERT-base, BERT-large, GPT2) with online time/bytes + projected LAN/WAN time; include SHAFT where overlapping.
* **Table 3:** Preprocessing (Sigma vs SUF) key size + keygen time.
* **Figure 1:** Kernel microbench time.
* **Figure 2:** Kernel microbench bytes.
* **Figure 3:** Scaling vs sequence length (BERT-base and GPT2).
* **Table 4:** Extensibility case study (new function + shapes + performance + LOC).
* **Appendix:** Full sweep configs + environment details + all raw metrics.

---

If you want, paste your current repo layout and the CLI you already use for `bench_suf_transformer` / `bench_softmax_norm`, and I’ll turn the above into **exact Codex prompts** (one prompt per file) with function signatures and parsing regexes tailored to your actual log formats.

