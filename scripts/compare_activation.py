#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MODELS = {
    "bert-tiny": {"n_layer": 2, "n_embd": 128, "intermediate": 0, "gate": "gelu"},
    "bert-base": {"n_layer": 12, "n_embd": 768, "intermediate": 0, "gate": "gelu"},
    "bert-large": {"n_layer": 24, "n_embd": 1024, "intermediate": 0, "gate": "gelu"},
    "gpt2": {"n_layer": 12, "n_embd": 768, "intermediate": 0, "gate": "gelu"},
    "gpt-neo": {"n_layer": 24, "n_embd": 2048, "intermediate": 0, "gate": "gelu"},
    "gpt-neo-large": {"n_layer": 32, "n_embd": 2560, "intermediate": 0, "gate": "gelu"},
    "llama7b": {"n_layer": 32, "n_embd": 4096, "intermediate": 11008, "gate": "silu"},
    "llama13b": {"n_layer": 40, "n_embd": 5120, "intermediate": 13824, "gate": "silu"},
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


def build_suf(root_dir):
    run(["cmake", "-S", str(root_dir), "-B", str(root_dir / "build"), "-DSUF_ENABLE_BENCH=ON"])
    run(["cmake", "--build", str(root_dir / "build"), "-j"])


def build_sigma(sigma_dir, cuda_version, gpu_arch):
    env = os.environ.copy()
    env["CUDA_VERSION"] = cuda_version
    env["GPU_ARCH"] = gpu_arch
    run(["make", "-j4", "gelu", "silu"], cwd=sigma_dir, env=env)


def parse_suf_json(output):
    last = None
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            last = line
    if not last:
        raise ValueError(f"Failed to find JSON in SUF output:\n{output}")
    return json.loads(last)


def run_suf_bench(suf_bin, model, seq, intervals, degree, helpers, iters):
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
    if run_suf_bench.mask_aware:
        cmd.extend(["--mask-aware", "--mask", str(run_suf_bench.mask_val)])
    output = run(cmd)
    return parse_suf_json(output)


def parse_sigma_log(text, gate):
    keygen_time = re.search(r"Keygen time=(\d+)", text)
    key_size = re.search(r"Key size=(\d+)", text)
    eval_time = re.search(rf"{gate.capitalize()} time=(\d+)", text)
    eval_comm = re.search(r"Eval comm bytes=(\d+)", text)
    if not (keygen_time and key_size and eval_time and eval_comm):
        raise ValueError(f"Failed to parse Sigma output for {gate}:\n{text}")
    return {
        "keygen_us": int(keygen_time.group(1)),
        "key_bytes": int(key_size.group(1)),
        "eval_us": int(eval_time.group(1)),
        "eval_comm_bytes": int(eval_comm.group(1)),
    }


def run_sigma_gate(sigma_bin, gate, n_elems, addr, env):
    with tempfile.TemporaryDirectory() as tmpdir:
        p0_log = Path(tmpdir) / "p0.log"
        p1_log = Path(tmpdir) / "p1.log"
        with p0_log.open("w") as f0:
            p0 = subprocess.Popen([sigma_bin, "0", addr, str(n_elems)], env=env, stdout=f0, stderr=f0)
        time.sleep(0.8)
        with p1_log.open("w") as f1:
            p1 = subprocess.Popen([sigma_bin, "1", addr, str(n_elems)], env=env, stdout=f1, stderr=f1)
        rc0 = p0.wait()
        rc1 = p1.wait()
        if rc0 != 0 or rc1 != 0:
            log_text = p0_log.read_text() + "\n" + p1_log.read_text()
            raise RuntimeError(f"Sigma {gate} failed: rc0={rc0} rc1={rc1}\n{log_text}")
        text = p0_log.read_text()
    return parse_sigma_log(text, gate)


def main():
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Align SUF vs Sigma activation benchmarks.")
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--models", type=str, default="all")
    parser.add_argument("--helpers", type=int, default=2)
    parser.add_argument("--degree", type=int, default=0)
    parser.add_argument("--intervals-gelu", type=int, default=256)
    parser.add_argument("--intervals-silu", type=int, default=1024)
    parser.add_argument("--mask-aware", action="store_true")
    parser.add_argument("--mask", type=int, default=0)
    parser.add_argument("--addr", type=str, default="127.0.0.1")
    parser.add_argument("--suf-bin", type=str, default=str(root_dir / "build" / "bench_suf_model"))
    parser.add_argument("--sigma-dir", type=str, default=str(root_dir / "third_party" / "EzPC" / "GPU-MPC"))
    parser.add_argument("--build-suf", action="store_true")
    parser.add_argument("--build-sigma", action="store_true")
    parser.add_argument("--cuda-version", type=str, default=os.environ.get("CUDA_VERSION", "12.9"))
    parser.add_argument("--gpu-arch", type=str, default=os.environ.get("GPU_ARCH", "120"))
    parser.add_argument("--sigma-keybuf-mb", type=int, default=4096)
    parser.add_argument("--sigma-mempool-mb", type=int, default=4096)
    parser.add_argument("--sigma-verify", action="store_true")
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--json", type=str, default="")
    args = parser.parse_args()

    model_list = list(MODELS.keys()) if args.models == "all" else args.models.split(",")
    for name in model_list:
        if name not in MODELS:
            raise ValueError(f"Unknown model: {name}")

    if args.build_suf:
        build_suf(root_dir)
    if args.build_sigma:
        build_sigma(Path(args.sigma_dir), args.cuda_version, args.gpu_arch)

    suf_bin = Path(args.suf_bin)
    if not suf_bin.exists():
        raise FileNotFoundError(f"SUF bench not found: {suf_bin}")

    sigma_dir = Path(args.sigma_dir)
    sigma_env = os.environ.copy()
    sigma_env["SIGMA_KEYBUF_MB"] = str(args.sigma_keybuf_mb)
    sigma_env["SIGMA_MEMPOOL_MB"] = str(args.sigma_mempool_mb)
    if not args.sigma_verify:
        sigma_env["SIGMA_SKIP_VERIFY"] = "1"

    run_suf_bench.mask_aware = args.mask_aware
    run_suf_bench.mask_val = args.mask

    results = []
    for model in model_list:
        spec = MODELS[model]
        gate = spec["gate"]
        if gate == "gelu":
            gate_elems = args.seq * 4 * spec["n_embd"]
            intervals = args.intervals_gelu
        else:
            gate_elems = args.seq * spec["intermediate"]
            intervals = args.intervals_silu
        gate_count = spec["n_layer"]

        suf = run_suf_bench(suf_bin, model, args.seq, intervals, args.degree, args.helpers, args.iters)

        sigma_bin = sigma_dir / "tests" / "fss" / gate
        if not sigma_bin.exists():
            raise FileNotFoundError(f"Sigma binary not found: {sigma_bin}")
        sigma = run_sigma_gate(str(sigma_bin), gate, gate_elems, args.addr, sigma_env)

        sigma_per_gate_key_ms = sigma["keygen_us"] / 1000.0
        sigma_per_gate_eval_ms = sigma["eval_us"] / 1000.0
        sigma_per_gate_key_bytes = sigma["key_bytes"]
        sigma_per_gate_eval_bytes = sigma["eval_comm_bytes"]

        row = {
            "model": model,
            "gate": gate,
            "seq": args.seq,
            "gate_elems": gate_elems,
            "gate_count": gate_count,
            "suf_intervals": intervals,
            "suf_degree": args.degree,
            "suf_helpers": args.helpers,
            "suf_pred_bytes": suf.get("pred_bytes", 0),
            "suf_lut_bytes": suf.get("lut_bytes", 0),
            "suf_per_gate_key_bytes": suf["per_gate_key_bytes"],
            "suf_total_key_bytes": suf["total_key_bytes"],
            "suf_pred_ms": suf.get("pred_ms", 0.0),
            "suf_lut_ms": suf.get("lut_ms", 0.0),
            "suf_per_gate_key_ms": suf["per_gate_key_ms"],
            "suf_total_key_ms": suf["total_key_ms"],
            "suf_per_gate_eval_ms": suf["per_gate_eval_ms"],
            "suf_total_eval_ms": suf["total_eval_ms"],
            "suf_per_gate_eval_bytes": 0,
            "suf_total_eval_bytes": 0,
            "sigma_per_gate_key_bytes": sigma_per_gate_key_bytes,
            "sigma_total_key_bytes": sigma_per_gate_key_bytes * gate_count,
            "sigma_per_gate_key_ms": sigma_per_gate_key_ms,
            "sigma_total_key_ms": sigma_per_gate_key_ms * gate_count,
            "sigma_per_gate_eval_ms": sigma_per_gate_eval_ms,
            "sigma_total_eval_ms": sigma_per_gate_eval_ms * gate_count,
            "sigma_per_gate_eval_bytes": sigma_per_gate_eval_bytes,
            "sigma_total_eval_bytes": sigma_per_gate_eval_bytes * gate_count,
        }
        results.append(row)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)

    header = (
        "model,gate,seq,gate_elems,gate_count,"
        "suf_key_bytes,sigma_key_bytes,suf_key_ms,sigma_key_ms,"
        "suf_eval_ms,sigma_eval_ms,sigma_eval_bytes"
    )
    print(header)
    for r in results:
        print(
            f"{r['model']},{r['gate']},{r['seq']},{r['gate_elems']},{r['gate_count']},"
            f"{r['suf_per_gate_key_bytes']},{r['sigma_per_gate_key_bytes']},"
            f"{r['suf_per_gate_key_ms']:.3f},{r['sigma_per_gate_key_ms']:.3f},"
            f"{r['suf_per_gate_eval_ms']:.3f},{r['sigma_per_gate_eval_ms']:.3f},"
            f"{r['sigma_per_gate_eval_bytes']}"
        )


if __name__ == "__main__":
    main()
