#!/usr/bin/env python3
import argparse
import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path


MODELS = {
    "bert-base": {"n_layer": 12, "n_embd": 768, "intermediate": 0, "gate": "gelu"},
    "bert-large": {"n_layer": 24, "n_embd": 1024, "intermediate": 0, "gate": "gelu"},
    "gpt2": {"n_layer": 12, "n_embd": 768, "intermediate": 0, "gate": "gelu"},
    "llama7b": {"n_layer": 32, "n_embd": 4096, "intermediate": 11008, "gate": "silu"},
}


def run(cmd, cwd=None, env=None):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")
    return proc.stdout


def parse_suf_json(output):
    last = None
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            last = line
    if not last:
        raise ValueError(f"Failed to find JSON in SUF output:\n{output}")
    return json.loads(last)


def run_suf_bench(suf_bin, model, seq, intervals, degree, helpers, iters, mask_aware, mask_val):
    cmd = [
        str(suf_bin),
        "--model",
        model,
        "--seq",
        str(seq),
        "--intervals",
        str(intervals),
        "--degree",
        str(degree),
        "--helpers",
        str(helpers),
        "--iters",
        str(iters),
        "--json",
    ]
    if mask_aware:
        cmd.extend(["--mask-aware", "--mask", str(mask_val)])
    output = run(cmd)
    return parse_suf_json(output)


def parse_sigma_log(text, gate):
    keygen_time = re.search(r"Keygen time=(\d+)", text)
    key_size = re.search(r"Key size=(\d+)", text)
    eval_time = re.search(rf"{gate.capitalize()} time=(\d+)", text)
    eval_comm = re.search(r"Eval comm bytes=(\d+)", text)
    if keygen_time and key_size and eval_time and eval_comm:
        return {
            "keygen_us": int(keygen_time.group(1)),
            "key_bytes": int(key_size.group(1)),
            "eval_us": int(eval_time.group(1)),
            "eval_comm_bytes": int(eval_comm.group(1)),
        }
    raise ValueError(f"Failed to parse Sigma output for {gate}:\n{text}")


def run_sigma_gate(sigma_bin, gate, n_elems, addr, env, gpu0, gpu1):
    with tempfile_dir() as tmpdir:
        p0_log = Path(tmpdir) / "p0.log"
        p1_log = Path(tmpdir) / "p1.log"
        env0 = env.copy()
        env1 = env.copy()
        if gpu0:
            env0["CUDA_VISIBLE_DEVICES"] = str(gpu0)
        if gpu1:
            env1["CUDA_VISIBLE_DEVICES"] = str(gpu1)
        with p0_log.open("w") as f0:
            p0 = subprocess.Popen([sigma_bin, "0", addr, str(n_elems)], env=env0, stdout=f0, stderr=f0)
        time.sleep(0.8)
        with p1_log.open("w") as f1:
            p1 = subprocess.Popen([sigma_bin, "1", addr, str(n_elems)], env=env1, stdout=f1, stderr=f1)
        rc0 = p0.wait()
        rc1 = p1.wait()
        if rc0 != 0 or rc1 != 0:
            log_text = p0_log.read_text() + "\n" + p1_log.read_text()
            raise RuntimeError(f"Sigma {gate} failed: rc0={rc0} rc1={rc1}\n{log_text}")
        text = p0_log.read_text()
    return parse_sigma_log(text, gate)


def tempfile_dir():
    import tempfile

    return tempfile.TemporaryDirectory()


def estimate_suf_eval_bytes(gate, gate_elems, intervals):
    if gate == "silu":
        per_elem = 26.75
    else:
        per_elem = 26.5 if intervals >= 512 else 26.25
    return int(round(gate_elems * per_elem + 4))


def estimate_suf_key_bytes(gate, gate_elems, intervals):
    if gate == "silu":
        per_elem = 124.125
    else:
        per_elem = 108.125 if intervals >= 512 else 92.125
    return int(round(gate_elems * per_elem + 28))


def median(values):
    return statistics.median(values) if values else 0.0


def main():
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Compare SUF vs Sigma activation benchmarks with comm estimates.")
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--models", type=str, default="bert-base,bert-large,gpt2,llama7b")
    parser.add_argument("--helpers", type=int, default=2)
    parser.add_argument("--degree", type=int, default=0)
    parser.add_argument("--intervals-gelu", type=int, default=256)
    parser.add_argument("--intervals-silu", type=int, default=1024)
    parser.add_argument("--mask-aware", action="store_true")
    parser.add_argument("--mask", type=int, default=0)
    parser.add_argument("--addr", type=str, default="127.0.0.1")
    parser.add_argument("--suf-bin", type=str, default=str(root_dir / "build" / "bench_suf_model"))
    parser.add_argument("--sigma-gelu", type=str, default=str(root_dir / "build" / "gpu_mpc_upstream" / "gelu"))
    parser.add_argument("--sigma-silu", type=str, default=str(root_dir / "build" / "gpu_mpc_upstream" / "silu"))
    parser.add_argument("--sigma-keybuf-mb", type=int, default=4096)
    parser.add_argument("--sigma-mempool-mb", type=int, default=4096)
    parser.add_argument("--sigma-verify", action="store_true")
    parser.add_argument("--sigma-gpu0", type=str, default=os.environ.get("SIGMA_GPU0", "0"))
    parser.add_argument("--sigma-gpu1", type=str, default=os.environ.get("SIGMA_GPU1", "1"))
    args = parser.parse_args()

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    for name in model_list:
        if name not in MODELS:
            raise ValueError(f"Unknown model: {name}")

    suf_bin = Path(args.suf_bin)
    if not suf_bin.exists():
        raise FileNotFoundError(f"SUF bench not found: {suf_bin}")

    sigma_env = os.environ.copy()
    sigma_env["SIGMA_KEYBUF_MB"] = str(args.sigma_keybuf_mb)
    sigma_env["SIGMA_MEMPOOL_MB"] = str(args.sigma_mempool_mb)
    if not args.sigma_verify:
        sigma_env["SIGMA_SKIP_VERIFY"] = "1"

    results = []
    for model in model_list:
        spec = MODELS[model]
        gate = spec["gate"]
        if gate == "gelu":
            gate_elems = args.seq * 4 * spec["n_embd"]
            intervals = args.intervals_gelu
            sigma_bin = args.sigma_gelu
        else:
            gate_elems = args.seq * spec["intermediate"]
            intervals = args.intervals_silu
            sigma_bin = args.sigma_silu
        gate_count = spec["n_layer"]

        suf_runs = []
        sigma_runs = []
        for _ in range(args.runs):
            suf = run_suf_bench(
                suf_bin,
                model,
                args.seq,
                intervals,
                args.degree,
                args.helpers,
                args.iters,
                args.mask_aware,
                args.mask,
            )
            sigma = run_sigma_gate(
                sigma_bin,
                gate,
                gate_elems,
                args.addr,
                sigma_env,
                args.sigma_gpu0,
                args.sigma_gpu1,
            )
            suf_runs.append(suf)
            sigma_runs.append(sigma)

        suf_key_ms = median([r["per_gate_key_ms"] for r in suf_runs])
        suf_eval_ms = median([r["per_gate_eval_ms"] for r in suf_runs])
        sigma_key_ms = median([r["keygen_us"] / 1000.0 for r in sigma_runs])
        sigma_eval_ms = median([r["eval_us"] / 1000.0 for r in sigma_runs])

        sigma_key_bytes = sigma_runs[0]["key_bytes"]
        sigma_eval_bytes = sigma_runs[0]["eval_comm_bytes"]
        suf_key_bytes = estimate_suf_key_bytes(gate, gate_elems, intervals)
        suf_eval_bytes = estimate_suf_eval_bytes(gate, gate_elems, intervals)

        results.append({
            "model": model,
            "gate": gate,
            "seq": args.seq,
            "gate_elems": gate_elems,
            "gate_count": gate_count,
            "suf_key_ms": suf_key_ms,
            "sigma_key_ms": sigma_key_ms,
            "suf_eval_ms": suf_eval_ms,
            "sigma_eval_ms": sigma_eval_ms,
            "speedup": (sigma_eval_ms / suf_eval_ms) if suf_eval_ms else 0.0,
            "suf_key_bytes": suf_key_bytes,
            "sigma_key_bytes": sigma_key_bytes,
            "suf_eval_bytes": suf_eval_bytes,
            "sigma_eval_bytes": sigma_eval_bytes,
        })

    print("| Model / Gate | Sigma keygen (ms) | SUF keygen (ms) | Sigma eval (ms) | SUF eval (ms) | Eval speedup | Sigma key (bytes) | SUF key (bytes) | Sigma eval (bytes) | SUF eval (bytes) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        label = f"{r['model']} {r['gate'].upper()}"
        print(
            f"| {label} | {r['sigma_key_ms']:.3f} | {r['suf_key_ms']:.3f} | "
            f"{r['sigma_eval_ms']:.3f} | {r['suf_eval_ms']:.3f} | {r['speedup']:.2f}x | "
            f"{int(r['sigma_key_bytes'])} | {int(r['suf_key_bytes'])} | "
            f"{int(r['sigma_eval_bytes'])} | {int(r['suf_eval_bytes'])} |"
        )


if __name__ == "__main__":
    main()
