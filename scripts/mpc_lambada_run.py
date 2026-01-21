#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def run(cmd, env=None, cwd=None, capture=False):
    if capture:
        return subprocess.run(cmd, env=env, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    return subprocess.run(cmd, env=env, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run real MPC LAMBADA accuracy (dealer + server + client).")
    parser.add_argument("--model", required=True, choices=["gpt2", "gpt_neo_1p3b"])
    parser.add_argument("--binary", default="third_party/EzPC/GPU-MPC/experiments/accuracy/mpc_lambada")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--keys-dir", default=None)
    parser.add_argument("--offline-only", action="store_true", help="Run dealer only to generate keys")
    parser.add_argument("--online-only", action="store_true", help="Run server+client only, using existing keys")
    parser.add_argument("--reuse-keys", action="store_true", help="Skip dealer if server.dat/client.dat already exist")
    parser.add_argument("--dealer-gpu", default="0")
    parser.add_argument("--server-gpu", default="0")
    parser.add_argument("--client-gpu", default="1")
    parser.add_argument("--omp-threads", type=int, default=4)
    args = parser.parse_args()

    root = Path("artifacts") / "mpc_lambada" / args.model
    weights = (Path(args.weights) if args.weights else root / "weights.dat").resolve()
    data_dir = (Path(args.data_dir) if args.data_dir else root / "inputs").resolve()
    labels = (Path(args.labels) if args.labels else root / "labels.txt").resolve()
    keys_dir = Path(args.keys_dir) if args.keys_dir else (root / "keys")
    keys_dir.mkdir(parents=True, exist_ok=True)

    binary = Path(args.binary).resolve()
    if not binary.exists():
        print(f"[error] binary not found: {binary}", file=sys.stderr)
        return 1
    if not weights.exists() or not data_dir.exists() or not labels.exists():
        print("[error] missing weights/data/labels. Run mpc_lambada_export.py first.", file=sys.stderr)
        return 1

    cmd = [
        str(binary),
        "{party}",
        args.model,
        str(weights),
        str(data_dir),
        str(labels),
        str(args.start),
        str(args.count),
        args.ip,
    ]

    env_base = os.environ.copy()
    env_base["OMP_NUM_THREADS"] = str(args.omp_threads)
    env_base["MKL_NUM_THREADS"] = str(args.omp_threads)
    env_base["OPENBLAS_NUM_THREADS"] = str(args.omp_threads)

    server_dat = keys_dir / "server.dat"
    client_dat = keys_dir / "client.dat"

    if not args.online_only:
        if args.reuse_keys and server_dat.exists() and client_dat.exists():
            print("[info] reusing existing keys")
        else:
            dealer_env = env_base.copy()
            dealer_env["CUDA_VISIBLE_DEVICES"] = args.dealer_gpu
            dealer_cmd = [c.format(party="1") for c in cmd]
            print("[info] running dealer...")
            run(dealer_cmd, env=dealer_env, cwd=keys_dir)
        if args.offline_only:
            return 0

    # Server + client: online run
    server_env = env_base.copy()
    server_env["CUDA_VISIBLE_DEVICES"] = args.server_gpu
    client_env = env_base.copy()
    client_env["CUDA_VISIBLE_DEVICES"] = args.client_gpu

    server_cmd = [c.format(party="2") for c in cmd]
    client_cmd = [c.format(party="3") for c in cmd]

    print("[info] running server + client...")
    server_proc = subprocess.Popen(server_cmd, env=server_env, cwd=keys_dir)
    time.sleep(2)
    client_proc = subprocess.Popen(client_cmd, env=client_env, cwd=keys_dir, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    client_out, _ = client_proc.communicate()
    server_rc = server_proc.wait()

    if client_proc.returncode != 0 or server_rc != 0:
        print("[error] MPC run failed", file=sys.stderr)
        if client_out:
            print(client_out)
        return 1

    if client_out:
        print(client_out.strip())
        for line in client_out.splitlines():
            if line.startswith("MPC_ACC"):
                parts = line.split()
                if len(parts) >= 4:
                    print(f"[result] correct={parts[1]} total={parts[2]} acc_pct={parts[3]}")
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
