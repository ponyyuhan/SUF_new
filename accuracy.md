TASK: Implement a complete, reproducible “Table 4 style” accuracy benchmark that compares:
(1) PyTorch float32 accuracy,
(2) Sigma Table 4 reported accuracy (as a reference baseline),
(3) SUF fixed-point emulation accuracy (frac_bits = 12, bitwidth matched to Sigma),
on the same tasks/datasets used in Sigma:

- GLUE: SST-2, MRPC, QNLI for BERT-tiny / BERT-base / BERT-large.
- LAMBADA: next-token (next-word) prediction accuracy for GPT-2 and GPT-Neo-1.3B.
- (Optional / skipped by default): Llama2-7B and Llama2-13B on LAMBADA (Sigma includes them but repo may not have weights).

Sigma Table 4 numbers (hardcode these in config as the “sigma_reference”):
BERT-tiny:
SST2: train 67k, val 872, pytorch 82.45, sigma 82.57, bitwidth 37
MRPC: train 3.7k, val 408, pytorch 71.07, sigma 70.34, bitwidth 37
QNLI: train 105K, val 5463, pytorch 81.42, sigma 81.93, bitwidth 37
BERT-base (bitwidth 50):
SST2: pytorch 92.55, sigma 92.55
MRPC: pytorch 84.31, sigma 87.25
QNLI: pytorch 91.60, sigma 91.63
BERT-large (bitwidth 50):
SST2: pytorch 93.58, sigma 93.35
MRPC: pytorch 87.99, sigma 88.48
QNLI: pytorch 92.23, sigma 92.26
GPT2 on Lambada (val 5153): pytorch 32.46, sigma 33.28, bitwidth 50
GPT-Neo-1.3B on Lambada (val 5153): pytorch 57.46, sigma 57.81, bitwidth 51
Llama2-7B Lambada: pytorch 70.17, sigma 69.92, bitwidth 48 (optional)
Llama2-13B Lambada: pytorch 73.14, sigma 72.99, bitwidth 48 (optional)

Protocol constraints to match Sigma:
- Evaluate with sequence length = 128 (truncate/pad accordingly).
- Use fixed-point precision f = 12 (call this frac_bits = 12).
- Use the same bitwidth per model as above (n_bits).
- IMPORTANT: This is an accuracy (numerical fidelity) evaluation, not secure two-party runtime.
  We do cleartext evaluation for:
    * PyTorch float32 baseline (HF Transformers),
    * SUF emulation (fixed-point rounding with frac_bits=12; bitwidth matching Sigma),
    * Sigma baseline is *reported* numbers from Table 4 for reference.

Where to implement:
- Primary entrypoint: bench/accuracy_compare.py (exists as a scaffold per README).
- Config: bench/configs/accuracy_table4.json (create/update as needed).
- Outputs:
    - --out-json: machine-readable results
    - --out-md: a Markdown table in the style of Sigma Table 4 with an extra SUF column.

STEP 0 — Inspect current repo state
1) Open and read existing files:
    - bench/accuracy_compare.py
    - bench/configs/accuracy_table4.json (if exists)
    - any existing helper code for “SUF fixed-point emulation” (search for keywords: fixed-point, frac_bits, quantize, emulation, suf_emulation, sim_harness, clear backend).
2) Prefer reusing existing SUF cleartext / emulation utilities if present.
   If the repo already has a clear backend that runs the SUF approximations, wire that in.
   Otherwise implement a pragmatic Python-level fixed-point emulation wrapper (see STEP 3B).

STEP 1 — Add Python deps (minimal and pinned where reasonable)
If the repo has a Python requirements file, add or document these dependencies:
- torch
- transformers
- datasets
- evaluate (or implement accuracy metrics directly)
- tqdm
  Also ensure code runs on CPU by default; GPU optional via --device.

Do not break existing benchmark scripts.

STEP 2 — Define config schema (bench/configs/accuracy_table4.json)
Design a JSON schema that can represent each “row” of the Table 4 evaluation.
Recommended structure:

{
"global": {
"seed": 0,
"seq_len": 128,
"frac_bits": 12,
"default_device": "cpu",
"cache_dir": null,
"num_workers": 4
},
"rows": [
{
"id": "bert_tiny_sst2",
"family": "bert_seqcls",
"task": "glue/sst2",
"hf_checkpoint": "<HF hub id or local path>",
"tokenizer": "<optional override>",
"n_bits": 37,
"frac_bits": 12,
"sigma_ref": {
"train_size": 67000,
"val_size": 872,
"pytorch_acc": 82.45,
"sigma_acc": 82.57
},
"skip": false,
"notes": ""
},
...
]
}

Include rows for:
- BERT-tiny: SST2/MRPC/QNLI (n_bits=37)
- BERT-base: SST2/MRPC/QNLI (n_bits=50)
- BERT-large: SST2/MRPC/QNLI (n_bits=50)
- GPT-2: LAMBADA (n_bits=50)
- GPT-Neo-1.3B: LAMBADA (n_bits=51)
- Llama2 rows present but default skip=true (if weights gated)

IMPORTANT checkpoint requirement:
- GLUE tasks require task-specific fine-tuned checkpoints.
- The config must allow local checkpoints if the repo includes them.
- If the repo already has “known good” fine-tuned checkpoints referenced in README, keep them and pin revision/sha if possible.

STEP 3 — Implement evaluation engine in bench/accuracy_compare.py
Add/ensure the CLI supports:
--config <path>
--out-json <path>
--out-md <path>
--device cpu|cuda
--max-examples <int> (optional; for quick smoke runs)
--batch-size <int> (default 8 for BERT tasks; 1 or small for LAMBADA)
--seed <int> (override)
--cache-dir <path> (optional)
--strict-sigma-match (optional): if enabled, assert pytorch acc is within tolerance vs sigma_ref.pytorch_acc.

The script must:
A) Load config.
B) For each row not skipped:
1) Load dataset and compute train/val sizes (or use sigma_ref if provided).
2) Run PyTorch float32 evaluation (baseline).
3) Run SUF fixed-point emulation evaluation using SAME model weights/checkpoint and SAME tokenization/truncation rules.
4) Record results including deltas vs PyTorch and vs Sigma.
   C) Write JSON and Markdown outputs.

STEP 3A — PyTorch float32 evaluation (HF)
Implement two evaluation paths:

(1) BERT GLUE classification (family == "bert_seqcls")
- Use datasets.load_dataset("glue", subset) where subset in { "sst2", "mrpc", "qnli" }.
- Use the official validation split (GLUE “validation” split).
- Tokenization:
    - Use AutoTokenizer.from_pretrained(checkpoint)
    - max_length = seq_len (=128)
    - truncation=True
    - padding="max_length"
- Model:
    - AutoModelForSequenceClassification.from_pretrained(checkpoint)
    - model.eval(); torch.inference_mode()
- Metric:
    - SST2/QNLI: accuracy
    - MRPC: report both accuracy and F1 if easy, but “table accuracy” should be accuracy to match Sigma style.
- Output as percentage with two decimals (but also store raw fraction).

(2) Causal LM LAMBADA (family == "gpt_lm")
- Dataset:
  Prefer “lambada_openai” if available; otherwise load “lambada” and implement the standard prompt/label split.
  Validation set size should be 5153 (Sigma Table 4). If using a fallback dataset variant, print a warning and record the actual size used.
- Metric: next-token accuracy:
  For each example:
    - tokenize full text to input_ids
    - label_id = last token id
    - prompt_ids = all tokens except last
    - truncate prompt_ids to last seq_len tokens (128) if longer
    - run model(prompt_ids) and take logits at last position
    - pred = argmax(logits_last)
    - correct if pred == label_id
      Use batch evaluation where possible (pad prompts to same length, use attention_mask; gather logits at each sample’s last non-pad position).
- For GPT-family tokenizer, ensure proper handling of special tokens and padding:
    - GPT2 tokenizer often has no pad token; set tokenizer.pad_token = tokenizer.eos_token for batching.

STEP 3B — SUF fixed-point emulation
This is the key part.

Preferred option (REUSE SUF CLEAR/EMULATION IF IT EXISTS):
- Search the repo for an existing cleartext backend or “emulation” path that runs SUF approximations without MPC.
- If found, integrate it to compute model outputs/logits for each batch.
- Ensure the emulation uses frac_bits=12 and n_bits from config, and matches the SUF numerical semantics used in the benchmark harness.

Fallback option (IMPLEMENT PYTHON FIXED-POINT EMULATION WRAPPER):
If no in-repo clear emulation exists, implement a best-effort fixed-point emulation in Python:
- Create a module (e.g., bench/fixed_point.py) with:
    - class FixedPointConfig(n_bits:int, frac_bits:int, signed:bool=True)
    - quantize(x: torch.Tensor, cfg) -> torch.Tensor
        * q = round(x * 2^f)
        * wrap to signed n_bits two’s complement (mod 2^n) to mimic ring overflow
        * return q / 2^f as float tensor
    - maybe quantize_weights for Linear/Embedding weights once.
- Wrap HF models using forward hooks or module replacement so that activations are quantized frequently enough to approximate fixed-point inference:
  Minimal acceptable quantization points:
    - after Embedding output
    - after every Linear projection (Q/K/V/out proj; MLP projections)
    - after LayerNorm
    - after activation (GELU for BERT; GELU/SiLU if present)
    - after attention softmax output and after attention matmul result
    - after residual additions
      Practical implementation options:
    - Register forward hooks on modules (Linear, LayerNorm, Embedding, GELU/SiLU, Softmax)
    - Or use torch.fx to rewrite the graph (more complex; only do if repo already uses fx).
- Ensure SUF emulation uses float operations but inserts quantization to emulate fixed-point rounding and bitwidth constraints.
- IMPORTANT: Keep the “baseline float32” model untouched; instantiate a separate emulated model or deep copy weights.

The objective is that SUF emulated accuracy should match (or be extremely close to) PyTorch float32, similar to Sigma’s claim.

STEP 4 — Output format
1) JSON output: list per row:
   {
   "id": ...,
   "family": ...,
   "dataset": ...,
   "seq_len": 128,
   "n_bits": ...,
   "frac_bits": 12,
   "train_size": ...,
   "val_size": ...,
   "pytorch": { "acc": ..., "acc_pct": ..., "extra": {...} },
   "suf": { "acc": ..., "acc_pct": ..., "extra": {...} },
   "sigma_ref": { ... },
   "delta": {
   "suf_minus_pytorch_pp": ...,
   "suf_minus_sigma_pp": ...,
   "pytorch_minus_sigma_pp": ...
   },
   "notes": "...",
   "runtime_s": ...
   }
2) Markdown output: a clean table similar to Sigma Table 4, plus SUF column:
   Columns:
   Model | Dataset | Train Size | Val Size | PyTorch Acc | Sigma Acc | SUF Acc | Bitwidth | frac_bits | Notes
- Use percentages with two decimals.
- For skipped rows: include row with “SKIPPED (reason)” or omit (config-driven).

STEP 5 — Reproducibility + usability
- Seed everything (python random, numpy if used, torch).
- Use torch.inference_mode and model.eval.
- Add --max-examples for quick runs.
- Add clear logging and progress bars (tqdm).
- Make sure script exits non-zero on configuration errors (missing checkpoint, dataset load failure), but allows skipping a row with skip=true.

STEP 6 — Documentation
- Update README.md “Accuracy Bench (Table 4 style)” section if needed:
    - Confirm command line usage
    - Explain that Sigma numbers are taken from Sigma Table 4 for reference and SUF is cleartext fixed-point emulation with frac_bits=12 and Sigma-matched bitwidth.

STEP 7 — Minimal validation
Add a lightweight smoke test (optional) that runs:
- One BERT row and one GPT row with --max-examples 20 on CPU
  and confirms the pipeline runs end-to-end and produces output files.

DELIVERABLES
- Updated/implemented bench/accuracy_compare.py
- Updated/created bench/configs/accuracy_table4.json
- Any helper modules you add (bench/fixed_point.py etc)
- Updated README snippet (if necessary)
- Example output files under bench/results/accuracy/ (do not commit large model weights; just commit small example outputs if repo policy allows)

Focus on correctness, clarity, and making the evaluation easy to reproduce for an ICML-style paper artifact.
