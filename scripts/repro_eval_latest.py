#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


MODEL_DISPLAY = {
    "bert-tiny": "BERT-tiny",
    "bert-base": "BERT-base",
    "bert-large": "BERT-large",
    "gpt2": "GPT-2",
    "gpt-neo": "GPT-Neo",
}

ROUND_COUNTS = {
    ("bert-tiny", 128): (188, 186),
    ("bert-base", 32): (1080, 1068),
    ("bert-base", 64): (1104, 1092),
    ("bert-base", 128): (1128, 1116),
    ("bert-large", 128): (2256, 2232),
    ("gpt2", 64): (1104, 1092),
    ("gpt2", 128): (1128, 1116),
    ("gpt2", 256): (1152, 1140),
    ("gpt-neo", 64): (2208, 2184),
    ("gpt-neo", 128): (2256, 2232),
}


def run_cmd(cmd, cwd, env, log_path):
    with open(log_path, "w") as f:
        return subprocess.Popen(cmd, cwd=cwd, env=env, stdout=f, stderr=f)


def ensure_output_dirs(run_dir):
    (run_dir / "output" / "P0" / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "output" / "P1" / "models").mkdir(parents=True, exist_ok=True)


def run_sigma_pair(run_dir, bin_path, model, seq, threads, addr, env_base, gpu0, gpu1, tag):
    ensure_output_dirs(run_dir)
    out_dir0 = run_dir / "output" / "P0" / "models" / f"{model}-{seq}"
    out_dir1 = run_dir / "output" / "P1" / "models" / f"{model}-{seq}"
    if out_dir0.exists():
        for p in out_dir0.glob("*"):
            p.unlink()
        out_dir0.rmdir()
    if out_dir1.exists():
        for p in out_dir1.glob("*"):
            p.unlink()
        out_dir1.rmdir()

    env0 = env_base.copy()
    env1 = env_base.copy()
    if gpu0 is not None:
        env0["CUDA_VISIBLE_DEVICES"] = str(gpu0)
    if gpu1 is not None:
        env1["CUDA_VISIBLE_DEVICES"] = str(gpu1)

    cmd0 = [str(bin_path), model, str(seq), "0", addr, str(threads)]
    cmd1 = [str(bin_path), model, str(seq), "1", addr, str(threads)]

    log0 = Path(f"/tmp/{tag}_{model}_{seq}_p0.log")
    log1 = Path(f"/tmp/{tag}_{model}_{seq}_p1.log")

    p0 = run_cmd(cmd0, run_dir, env0, log0)
    time.sleep(1.0)
    p1 = run_cmd(cmd1, run_dir, env1, log1)
    rc0 = p0.wait()
    rc1 = p1.wait()
    if rc0 != 0 or rc1 != 0:
        raise RuntimeError(
            f"{tag} failed for {model}-{seq} (rc0={rc0} rc1={rc1}). Logs: {log0}, {log1}"
        )
    return out_dir0


def parse_dealer(path):
    text = path.read_text()
    total_us = int(re.search(r"Total time=(\d+) us", text).group(1))
    key_bytes = int(re.search(r"Key size=(\d+) B", text).group(1))
    return {"keygen_us": total_us, "key_bytes": key_bytes}


def parse_evaluator(path):
    text = path.read_text()
    total_us = int(re.search(r"Total time=(\d+) us", text).group(1))
    comm_us = int(re.search(r"Comm time=(\d+) us", text).group(1))
    total_comm_bytes = int(re.search(r"Total Comm=(\d+) B", text).group(1))
    return {
        "total_us": total_us,
        "comm_us": comm_us,
        "total_comm_bytes": total_comm_bytes,
    }


def bytes_to_gb(bytes_val):
    return bytes_val / (1024.0 ** 3)


def project_times(total_us, comm_us, comm_bytes, rounds, bandwidth, latency_s):
    comp_time = (total_us - comm_us) / 1e6
    return comp_time + 2.0 * comm_bytes / bandwidth + rounds * latency_s


def collect_result(out_dir):
    dealer = parse_dealer(out_dir / "dealer.txt")
    evaluator = parse_evaluator(out_dir / "evaluator.txt")
    return {
        **dealer,
        **evaluator,
        "online_ms": evaluator["total_us"] / 1000.0,
        "comm_gb": bytes_to_gb(evaluator["total_comm_bytes"]),
        "keygen_s": dealer["keygen_us"] / 1e6,
        "key_gb": bytes_to_gb(dealer["key_bytes"]),
    }


def fmt_float(x, places=2):
    return f"{x:.{places}f}"


def fmt_speedup(a, b):
    if b == 0:
        return "inf"
    return f"{a / b:.2f}x"


def make_tables(results):
    lan_bw = 1e9
    wan_bw = 400e6
    lan_lat = 0.0005
    wan_lat = 0.004

    def lan_wan(model, seq, variant):
        rounds = ROUND_COUNTS[(model, seq)][0 if variant == "sigma" else 1]
        r = results[(model, seq, variant)]
        lan = project_times(r["total_us"], r["comm_us"], r["total_comm_bytes"], rounds, lan_bw, lan_lat)
        wan = project_times(r["total_us"], r["comm_us"], r["total_comm_bytes"], rounds, wan_bw, wan_lat)
        return rounds, lan, wan

    # Section 3 main table (seq=128)
    base_models = ["bert-tiny", "bert-base", "bert-large", "gpt2", "gpt-neo"]
    base_rows = []
    for model in base_models:
        seq = 128
        sigma = results[(model, seq, "sigma")]
        suf = results[(model, seq, "suf")]
        s_rounds, s_lan, s_wan = lan_wan(model, seq, "sigma")
        u_rounds, u_lan, u_wan = lan_wan(model, seq, "suf")
        base_rows.append({
            "model": f"{MODEL_DISPLAY[model]}-{seq}",
            "sigma_ms": sigma["online_ms"],
            "suf_ms": suf["online_ms"],
            "speedup": fmt_speedup(sigma["online_ms"], suf["online_ms"]),
            "sigma_comm": sigma["comm_gb"],
            "suf_comm": suf["comm_gb"],
            "sigma_rounds": s_rounds,
            "suf_rounds": u_rounds,
            "sigma_lan": s_lan,
            "suf_lan": u_lan,
            "sigma_wan": s_wan,
            "suf_wan": u_wan,
        })

    # Keygen/key size table
    key_rows = []
    for model in base_models:
        seq = 128
        sigma = results[(model, seq, "sigma")]
        suf = results[(model, seq, "suf")]
        key_rows.append({
            "model": f"{MODEL_DISPLAY[model]}-{seq}",
            "sigma_keygen": sigma["keygen_s"],
            "suf_keygen": suf["keygen_s"],
            "sigma_key": sigma["key_gb"],
            "suf_key": suf["key_gb"],
        })

    # Additional seq (GPT-2 / GPT-Neo)
    extra_rows = []
    for model, seqs in [("gpt2", [64, 128, 256]), ("gpt-neo", [64, 128])]:
        for seq in seqs:
            sigma = results[(model, seq, "sigma")]
            suf = results[(model, seq, "suf")]
            s_rounds, s_lan, s_wan = lan_wan(model, seq, "sigma")
            u_rounds, u_lan, u_wan = lan_wan(model, seq, "suf")
            extra_rows.append({
                "model": MODEL_DISPLAY[model],
                "seq": seq,
                "sigma_ms": sigma["online_ms"],
                "suf_ms": suf["online_ms"],
                "speedup": fmt_speedup(sigma["online_ms"], suf["online_ms"]),
                "sigma_comm": sigma["comm_gb"],
                "suf_comm": suf["comm_gb"],
                "sigma_rounds": s_rounds,
                "suf_rounds": u_rounds,
                "sigma_lan": s_lan,
                "suf_lan": u_lan,
                "sigma_wan": s_wan,
                "suf_wan": u_wan,
            })

    # Scaling (BERT-base)
    scale_rows = []
    for seq in [32, 64, 128]:
        sigma = results[("bert-base", seq, "sigma")]
        suf = results[("bert-base", seq, "suf")]
        s_rounds, s_lan, s_wan = lan_wan("bert-base", seq, "sigma")
        u_rounds, u_lan, u_wan = lan_wan("bert-base", seq, "suf")
        scale_rows.append({
            "seq": seq,
            "sigma_ms": sigma["online_ms"],
            "suf_ms": suf["online_ms"],
            "speedup": fmt_speedup(sigma["online_ms"], suf["online_ms"]),
            "sigma_comm": sigma["comm_gb"],
            "suf_comm": suf["comm_gb"],
            "sigma_rounds": s_rounds,
            "suf_rounds": u_rounds,
            "sigma_lan": s_lan,
            "suf_lan": u_lan,
            "sigma_wan": s_wan,
            "suf_wan": u_wan,
        })

    return base_rows, key_rows, extra_rows, scale_rows


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Reproduce Sigma/SUF evaluation tables.")
    parser.add_argument("--sigma-bin", type=str, default=str(root / "build" / "gpu_mpc_upstream" / "sigma"))
    parser.add_argument("--suf-bin", type=str, default=str(root / "build" / "gpu_mpc_vendor" / "sigma"))
    parser.add_argument("--sigma-run-dir", type=str, default=str(root / "ezpc_upstream" / "GPU-MPC" / "experiments" / "sigma"))
    parser.add_argument("--suf-run-dir", type=str, default=str(root / "third_party" / "EzPC_vendor" / "GPU-MPC" / "experiments" / "sigma"))
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--addr", type=str, default="127.0.0.1")
    parser.add_argument("--gpu0", type=str, default="0")
    parser.add_argument("--gpu1", type=str, default="1")
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    sigma_bin = Path(args.sigma_bin)
    suf_bin = Path(args.suf_bin)
    sigma_dir = Path(args.sigma_run_dir)
    suf_dir = Path(args.suf_run_dir)

    tasks = [
        ("bert-tiny", 128),
        ("bert-base", 32),
        ("bert-base", 64),
        ("bert-base", 128),
        ("bert-large", 128),
        ("gpt2", 64),
        ("gpt2", 128),
        ("gpt2", 256),
        ("gpt-neo", 64),
        ("gpt-neo", 128),
    ]

    results = {}
    sigma_env = os.environ.copy()
    sigma_env["SIGMA_MEMPOOL_DISABLE"] = "1"
    sigma_env["OMP_NUM_THREADS"] = str(args.threads)

    suf_env = os.environ.copy()
    suf_env["SIGMA_MEMPOOL_DISABLE"] = "1"
    suf_env["OMP_NUM_THREADS"] = str(args.threads)
    suf_env["SUF_SOFTMAX"] = "1"
    suf_env["SUF_LAYERNORM"] = "1"
    suf_env["SUF_ACTIVATION"] = "1"
    suf_env["SUF_NEXP_BITS"] = "10"
    suf_env["SUF_INV_BITS"] = "10"
    suf_env["SUF_RSQRT_BITS"] = "9"

    for model, seq in tasks:
        for variant, run_dir, bin_path, env, tag in [
            ("sigma", sigma_dir, sigma_bin, sigma_env, "sigma_base"),
            ("suf", suf_dir, suf_bin, suf_env, "suf"),
        ]:
            if not args.no_run:
                out_dir = run_sigma_pair(
                    run_dir,
                    bin_path,
                    model,
                    seq,
                    args.threads,
                    args.addr,
                    env,
                    args.gpu0,
                    args.gpu1,
                    tag,
                )
            else:
                out_dir = run_dir / "output" / "P0" / "models" / f"{model}-{seq}"
            if not out_dir.exists():
                raise FileNotFoundError(f"Missing output: {out_dir}")
            results[(model, seq, variant)] = collect_result(out_dir)

    base_rows, key_rows, extra_rows, scale_rows = make_tables(results)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(
                {
                    "results": {f"{k[0]}:{k[1]}:{k[2]}": v for k, v in results.items()},
                    "base": base_rows,
                    "key": key_rows,
                    "extra": extra_rows,
                    "scale": scale_rows,
                },
                f,
                indent=2,
            )

    def print_table(headers, rows, keys, fmts):
        print("| " + " | ".join(headers) + " |")
        print("|" + "|".join(["---"] * len(headers)) + "|")
        for row in rows:
            cells = []
            for k, fmt in zip(keys, fmts):
                val = row[k]
                if fmt:
                    cells.append(fmt(val))
                else:
                    cells.append(str(val))
            print("| " + " | ".join(cells) + " |")

    print("\n## SUF vs Sigma (end-to-end, seq=128)\n")
    print_table(
        ["Model", "Sigma online (ms)", "SUF online (ms)", "Speedup", "Sigma comm (GB)", "SUF comm (GB)",
         "Sigma rounds", "SUF rounds", "Sigma LAN (s)", "SUF LAN (s)", "Sigma WAN (s)", "SUF WAN (s)"],
        base_rows,
        ["model", "sigma_ms", "suf_ms", "speedup", "sigma_comm", "suf_comm",
         "sigma_rounds", "suf_rounds", "sigma_lan", "suf_lan", "sigma_wan", "suf_wan"],
        [None, lambda v: fmt_float(v, 2), lambda v: fmt_float(v, 2), None,
         lambda v: fmt_float(v, 3), lambda v: fmt_float(v, 3),
         None, None, lambda v: fmt_float(v, 2), lambda v: fmt_float(v, 2),
         lambda v: fmt_float(v, 2), lambda v: fmt_float(v, 2)],
    )

    print("\n### Keygen and key size\n")
    print_table(
        ["Model", "Sigma keygen (s)", "SUF keygen (s)", "Sigma key (GB)", "SUF key (GB)"],
        key_rows,
        ["model", "sigma_keygen", "suf_keygen", "sigma_key", "suf_key"],
        [None, lambda v: fmt_float(v, 2), lambda v: fmt_float(v, 2),
         lambda v: fmt_float(v, 3), lambda v: fmt_float(v, 3)],
    )

    print("\n### Additional sequence points (GPT-2 / GPT-Neo)\n")
    print_table(
        ["Model", "Seq", "Sigma time (ms)", "SUF time (ms)", "Speedup", "Sigma comm (GB)", "SUF comm (GB)",
         "Sigma rounds", "SUF rounds", "Sigma LAN (s)", "SUF LAN (s)", "Sigma WAN (s)", "SUF WAN (s)"],
        extra_rows,
        ["model", "seq", "sigma_ms", "suf_ms", "speedup", "sigma_comm", "suf_comm",
         "sigma_rounds", "suf_rounds", "sigma_lan", "suf_lan", "sigma_wan", "suf_wan"],
        [None, None, lambda v: fmt_float(v, 2), lambda v: fmt_float(v, 2), None,
         lambda v: fmt_float(v, 3), lambda v: fmt_float(v, 3), None, None,
         lambda v: fmt_float(v, 2), lambda v: fmt_float(v, 2),
         lambda v: fmt_float(v, 2), lambda v: fmt_float(v, 2)],
    )

    print("\n### Scaling (BERT-base seq sweep)\n")
    print_table(
        ["Seq", "Sigma time (ms)", "SUF time (ms)", "Speedup", "Sigma comm (GB)", "SUF comm (GB)",
         "Sigma rounds", "SUF rounds", "Sigma LAN (s)", "SUF LAN (s)", "Sigma WAN (s)", "SUF WAN (s)"],
        scale_rows,
        ["seq", "sigma_ms", "suf_ms", "speedup", "sigma_comm", "suf_comm",
         "sigma_rounds", "suf_rounds", "sigma_lan", "suf_lan", "sigma_wan", "suf_wan"],
        [None, lambda v: fmt_float(v, 2), lambda v: fmt_float(v, 2), None,
         lambda v: fmt_float(v, 3), lambda v: fmt_float(v, 3), None, None,
         lambda v: fmt_float(v, 2), lambda v: fmt_float(v, 2),
         lambda v: fmt_float(v, 2), lambda v: fmt_float(v, 2)],
    )


if __name__ == "__main__":
    main()
