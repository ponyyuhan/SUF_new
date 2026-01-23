# Activation microbench rerun (2026-01-23)

## Parameters
**compare_activation_fair.py**
- command: `SIGMA_DISABLE_ASYNC_MALLOC=1 SIGMA_COMPRESS=0 python3 scripts/compare_activation_fair.py --models bert-base,bert-large,gpt2,llama7b --seq 128 --sigma-mempool-disable --csv artifacts/activation_2026-01-23/compare_activation_fair.csv --json artifacts/activation_2026-01-23/compare_activation_fair.json`
- env: `SIGMA_DISABLE_ASYNC_MALLOC=1`, `SIGMA_COMPRESS=0`, `SIGMA_MEMPOOL_DISABLE=1`, `SIGMA_KEYBUF_MB=4096`, `SIGMA_SKIP_VERIFY=1`
- SUF defaults (from `src/sigma_suf_bridge.cu`): `SUF_GELU_INTERVALS=256`, `SUF_SILU_INTERVALS=1024`
- LAN/WAN projection (from evaluation_rounds.md): `LAN=1e9 B/s, 0.5 ms`, `WAN=400e6 B/s, 4 ms`
- rounds source: `Eval rounds` printed by test binaries (Sigma=ezpc_upstream, SUF=EzPC_vendor)
- GPU binding: defaults (`SIGMA_GPU0=0`, `SIGMA_GPU1=1`, `SUF_GPU0=0`, `SUF_GPU1=1`) unless overridden

**compare_activation.py**
- command: `SIGMA_DISABLE_ASYNC_MALLOC=1 SIGMA_COMPRESS=0 python3 scripts/compare_activation.py --models bert-base,bert-large,gpt2,llama7b --seq 128 --csv artifacts/activation_2026-01-23/compare_activation.csv --json artifacts/activation_2026-01-23/compare_activation.json`
- env: `SIGMA_DISABLE_ASYNC_MALLOC=1`, `SIGMA_COMPRESS=0`, `SIGMA_KEYBUF_MB=4096`, `SIGMA_MEMPOOL_MB=4096`, `SIGMA_SKIP_VERIFY=1`
- SUF bench args: `--iters 20 --helpers 2 --degree 0 --intervals-gelu 256 --intervals-silu 1024`

## Environment
- timestamp (UTC): 2026-01-23 21:50:39
- OS: Ubuntu 24.04.3 LTS (Noble Numbat)
- kernel: Linux 6.18.6-pbk #1 SMP PREEMPT_DYNAMIC (2026-01-19)
- CPU: AMD EPYC 9654 96-Core Processor (2 sockets, 384 CPUs)
- RAM: 1.5 TiB
- GPU: 2x NVIDIA RTX PRO 6000 Blackwell Workstation Edition (97,887 MiB each)
- NVIDIA driver/CUDA: 580.119.02 / CUDA 13.0
- nvcc: 13.0.88
- Python: 3.12.12
- git commit: 1f8df53768130a1e715df4b0188fa31841a0772c (dirty)

## Results: compare_activation_fair.py (per‑gate)
> Note: `_x` columns are Sigma/SUF ratios (higher means SUF is better; `inf` means SUF value is 0).

| model | gate | seq | gate_elems | sigma_key_bytes | sigma_key_ms | sigma_eval_ms | sigma_eval_bytes | sigma_eval_rounds | sigma_lan_s | sigma_wan_s | suf_key_bytes | suf_key_ms | suf_eval_ms | suf_eval_bytes | suf_eval_rounds | suf_lan_s | suf_wan_s | key_bytes_x | key_ms_x | eval_ms_x | eval_bytes_x | lan_x | wan_x |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bert-base | gelu | 128 | 393216 | 226246736 | 16.039 | 15.472 | 13664260 | 7 | 0.044 | 0.110 | 36225052 | 10.305 | 9.145 | 10321924 | 4 | 0.030 | 0.075 | 6.25 | 1.56 | 1.69 | 1.32 | 1.45 | 1.45 |
| bert-large | gelu | 128 | 524288 | 301662288 | 17.971 | 17.072 | 18219012 | 7 | 0.055 | 0.134 | 48300060 | 17.152 | 10.277 | 13762564 | 4 | 0.038 | 0.093 | 6.25 | 1.05 | 1.66 | 1.32 | 1.44 | 1.44 |
| gpt2 | gelu | 128 | 393216 | 226246736 | 15.789 | 14.729 | 13664260 | 7 | 0.044 | 0.109 | 36225052 | 10.444 | 8.961 | 10321924 | 4 | 0.030 | 0.075 | 6.25 | 1.51 | 1.64 | 1.32 | 1.44 | 1.45 |
| llama7b | silu | 128 | 1409024 | 867078224 | 42.320 | 58.520 | 49668100 | 7 | 0.148 | 0.322 | 174895132 | 16.022 | 23.799 | 37691396 | 4 | 0.095 | 0.222 | 4.96 | 2.64 | 2.46 | 1.32 | 1.56 | 1.45 |

## Results: compare_activation.py (per‑gate)
> Note: `_x` columns are Sigma/SUF ratios (higher means SUF is better; `inf` means SUF value is 0).

| model | gate | seq | gate_elems | gate_count | suf_intervals | suf_degree | suf_helpers | suf_pred_bytes | suf_lut_bytes | suf_key_bytes | suf_key_ms | suf_eval_ms | sigma_key_bytes | sigma_key_ms | sigma_eval_bytes | sigma_eval_ms | key_bytes_x | key_ms_x | eval_ms_x | eval_bytes_x |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bert-base | gelu | 128 | 393216 | 12 | 256 | 0 | 2 | 384 | 2080 | 2464 | 0.020 | 0.990 | 226246736 | 49.712 | 20054020 | 25.078 | 91820.92 | 2486.84 | 25.33 | inf |
| bert-large | gelu | 128 | 524288 | 24 | 256 | 0 | 2 | 384 | 2080 | 2464 | 0.020 | 1.207 | 301662288 | 36.573 | 26738692 | 28.346 | 122427.88 | 1818.65 | 23.48 | inf |
| gpt2 | gelu | 128 | 393216 | 12 | 256 | 0 | 2 | 384 | 2080 | 2464 | 0.023 | 0.990 | 226246736 | 34.409 | 20054020 | 25.325 | 91820.92 | 1470.47 | 25.59 | inf |
| llama7b | silu | 128 | 1409024 | 32 | 1024 | 0 | 2 | 448 | 8224 | 8672 | 0.060 | 4.054 | 867078224 | 57.733 | 74678276 | 68.230 | 99985.96 | 961.43 | 16.83 | inf |

## Raw JSON outputs
**compare_activation_fair.json**
```json
[
  {
    "model": "bert-base",
    "gate": "gelu",
    "seq": 128,
    "gate_elems": 393216,
    "gate_count": 12,
    "sigma_per_gate_key_bytes": 226246736,
    "sigma_per_gate_key_ms": 16.039,
    "sigma_per_gate_eval_ms": 15.472,
    "sigma_per_gate_eval_bytes": 13664260,
    "sigma_per_gate_eval_rounds": 7,
    "sigma_per_gate_comm_ms": 2.17,
    "sigma_per_gate_comp_ms": 13.302,
    "sigma_lan_s": 0.04413052,
    "sigma_wan_s": 0.10962329999999999,
    "suf_per_gate_key_bytes": 36225052,
    "suf_per_gate_key_ms": 10.305,
    "suf_per_gate_eval_ms": 9.145,
    "suf_per_gate_eval_bytes": 10321924,
    "suf_per_gate_eval_rounds": 4,
    "suf_per_gate_comm_ms": 1.372,
    "suf_per_gate_comp_ms": 7.773000000000001,
    "suf_lan_s": 0.030416848000000003,
    "suf_wan_s": 0.07538262000000001
  },
  {
    "model": "bert-large",
    "gate": "gelu",
    "seq": 128,
    "gate_elems": 524288,
    "gate_count": 24,
    "sigma_per_gate_key_bytes": 301662288,
    "sigma_per_gate_key_ms": 17.971,
    "sigma_per_gate_eval_ms": 17.072,
    "sigma_per_gate_eval_bytes": 18219012,
    "sigma_per_gate_eval_rounds": 7,
    "sigma_per_gate_comm_ms": 2.352,
    "sigma_per_gate_comp_ms": 14.72,
    "sigma_lan_s": 0.054658024,
    "sigma_wan_s": 0.13381506,
    "suf_per_gate_key_bytes": 48300060,
    "suf_per_gate_key_ms": 17.152,
    "suf_per_gate_eval_ms": 10.277,
    "suf_per_gate_eval_bytes": 13762564,
    "suf_per_gate_eval_rounds": 4,
    "suf_per_gate_comm_ms": 1.934,
    "suf_per_gate_comp_ms": 8.343,
    "suf_lan_s": 0.037868128,
    "suf_wan_s": 0.09315582
  },
  {
    "model": "gpt2",
    "gate": "gelu",
    "seq": 128,
    "gate_elems": 393216,
    "gate_count": 12,
    "sigma_per_gate_key_bytes": 226246736,
    "sigma_per_gate_key_ms": 15.789,
    "sigma_per_gate_eval_ms": 14.729,
    "sigma_per_gate_eval_bytes": 13664260,
    "sigma_per_gate_eval_rounds": 7,
    "sigma_per_gate_comm_ms": 1.934,
    "sigma_per_gate_comp_ms": 12.795000000000002,
    "sigma_lan_s": 0.04362352,
    "sigma_wan_s": 0.1091163,
    "suf_per_gate_key_bytes": 36225052,
    "suf_per_gate_key_ms": 10.444,
    "suf_per_gate_eval_ms": 8.961,
    "suf_per_gate_eval_bytes": 10321924,
    "suf_per_gate_eval_rounds": 4,
    "suf_per_gate_comm_ms": 1.411,
    "suf_per_gate_comp_ms": 7.550000000000001,
    "suf_lan_s": 0.030193848000000002,
    "suf_wan_s": 0.07515962000000001
  },
  {
    "model": "llama7b",
    "gate": "silu",
    "seq": 128,
    "gate_elems": 1409024,
    "gate_count": 32,
    "sigma_per_gate_key_bytes": 867078224,
    "sigma_per_gate_key_ms": 42.32,
    "sigma_per_gate_eval_ms": 58.52,
    "sigma_per_gate_eval_bytes": 49668100,
    "sigma_per_gate_eval_rounds": 7,
    "sigma_per_gate_comm_ms": 13.18,
    "sigma_per_gate_comp_ms": 45.339999999999996,
    "sigma_lan_s": 0.1481762,
    "sigma_wan_s": 0.32168050000000004,
    "suf_per_gate_key_bytes": 174895132,
    "suf_per_gate_key_ms": 16.022,
    "suf_per_gate_eval_ms": 23.799,
    "suf_per_gate_eval_bytes": 37691396,
    "suf_per_gate_eval_rounds": 4,
    "suf_per_gate_comm_ms": 6.013,
    "suf_per_gate_comp_ms": 17.786,
    "suf_lan_s": 0.095168792,
    "suf_wan_s": 0.22224297999999998
  }
]
```
**compare_activation.json**
```json
[
  {
    "model": "bert-base",
    "gate": "gelu",
    "seq": 128,
    "gate_elems": 393216,
    "gate_count": 12,
    "suf_intervals": 256,
    "suf_degree": 0,
    "suf_helpers": 2,
    "suf_pred_bytes": 384,
    "suf_lut_bytes": 2080,
    "suf_per_gate_key_bytes": 2464,
    "suf_total_key_bytes": 29568,
    "suf_pred_ms": 0.00656,
    "suf_lut_ms": 0.00303,
    "suf_per_gate_key_ms": 0.01999,
    "suf_total_key_ms": 0.23988,
    "suf_per_gate_eval_ms": 0.989995,
    "suf_total_eval_ms": 11.8799,
    "suf_per_gate_eval_bytes": 0,
    "suf_total_eval_bytes": 0,
    "sigma_per_gate_key_bytes": 226246736,
    "sigma_total_key_bytes": 2714960832,
    "sigma_per_gate_key_ms": 49.712,
    "sigma_total_key_ms": 596.5440000000001,
    "sigma_per_gate_eval_ms": 25.078,
    "sigma_total_eval_ms": 300.936,
    "sigma_per_gate_eval_bytes": 20054020,
    "sigma_total_eval_bytes": 240648240
  },
  {
    "model": "bert-large",
    "gate": "gelu",
    "seq": 128,
    "gate_elems": 524288,
    "gate_count": 24,
    "suf_intervals": 256,
    "suf_degree": 0,
    "suf_helpers": 2,
    "suf_pred_bytes": 384,
    "suf_lut_bytes": 2080,
    "suf_per_gate_key_bytes": 2464,
    "suf_total_key_bytes": 59136,
    "suf_pred_ms": 0.0068,
    "suf_lut_ms": 0.00307,
    "suf_per_gate_key_ms": 0.02011,
    "suf_total_key_ms": 0.48264,
    "suf_per_gate_eval_ms": 1.20739,
    "suf_total_eval_ms": 28.9774,
    "suf_per_gate_eval_bytes": 0,
    "suf_total_eval_bytes": 0,
    "sigma_per_gate_key_bytes": 301662288,
    "sigma_total_key_bytes": 7239894912,
    "sigma_per_gate_key_ms": 36.573,
    "sigma_total_key_ms": 877.752,
    "sigma_per_gate_eval_ms": 28.346,
    "sigma_total_eval_ms": 680.304,
    "sigma_per_gate_eval_bytes": 26738692,
    "sigma_total_eval_bytes": 641728608
  },
  {
    "model": "gpt2",
    "gate": "gelu",
    "seq": 128,
    "gate_elems": 393216,
    "gate_count": 12,
    "suf_intervals": 256,
    "suf_degree": 0,
    "suf_helpers": 2,
    "suf_pred_bytes": 384,
    "suf_lut_bytes": 2080,
    "suf_per_gate_key_bytes": 2464,
    "suf_total_key_bytes": 29568,
    "suf_pred_ms": 0.00657,
    "suf_lut_ms": 0.00302,
    "suf_per_gate_key_ms": 0.0234,
    "suf_total_key_ms": 0.2808,
    "suf_per_gate_eval_ms": 0.989574,
    "suf_total_eval_ms": 11.8749,
    "suf_per_gate_eval_bytes": 0,
    "suf_total_eval_bytes": 0,
    "sigma_per_gate_key_bytes": 226246736,
    "sigma_total_key_bytes": 2714960832,
    "sigma_per_gate_key_ms": 34.409,
    "sigma_total_key_ms": 412.908,
    "sigma_per_gate_eval_ms": 25.325,
    "sigma_total_eval_ms": 303.9,
    "sigma_per_gate_eval_bytes": 20054020,
    "sigma_total_eval_bytes": 240648240
  },
  {
    "model": "llama7b",
    "gate": "silu",
    "seq": 128,
    "gate_elems": 1409024,
    "gate_count": 32,
    "suf_intervals": 1024,
    "suf_degree": 0,
    "suf_helpers": 2,
    "suf_pred_bytes": 448,
    "suf_lut_bytes": 8224,
    "suf_per_gate_key_bytes": 8672,
    "suf_total_key_bytes": 277504,
    "suf_pred_ms": 0.007719,
    "suf_lut_ms": 0.01639,
    "suf_per_gate_key_ms": 0.060049,
    "suf_total_key_ms": 1.92157,
    "suf_per_gate_eval_ms": 4.05351,
    "suf_total_eval_ms": 129.712,
    "suf_per_gate_eval_bytes": 0,
    "suf_total_eval_bytes": 0,
    "sigma_per_gate_key_bytes": 867078224,
    "sigma_total_key_bytes": 27746503168,
    "sigma_per_gate_key_ms": 57.733,
    "sigma_total_key_ms": 1847.456,
    "sigma_per_gate_eval_ms": 68.23,
    "sigma_total_eval_ms": 2183.36,
    "sigma_per_gate_eval_bytes": 74678276,
    "sigma_total_eval_bytes": 2389704832
  }
]
```
