#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIGMA_DIR="$ROOT_DIR/third_party/EzPC/GPU-MPC"

BUILD_SIGMA=false
RUN_SIGMA=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-sigma) BUILD_SIGMA=true; shift ;;
    --run-sigma) RUN_SIGMA=true; shift ;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

echo "[SUF] configure/build"
cmake -S "$ROOT_DIR" -B "$ROOT_DIR/build" -DSUF_ENABLE_BENCH=ON -DSUF_ENABLE_TESTS=ON
cmake --build "$ROOT_DIR/build" -j

echo "[SUF] bench"
SECURE_FLAG=""
if [[ "${SUF_SECURE:-0}" -eq 1 ]]; then
  SECURE_FLAG="--secure"
fi
"$ROOT_DIR/build/bench_suf_gpu" --n 1048576 --iters 50 --intervals 16 --degree 3 --helpers 4 ${SECURE_FLAG}

if $BUILD_SIGMA; then
  echo "[Sigma] build"
  if [[ ! -d "$SIGMA_DIR" ]]; then
    echo "Sigma repo not found at $SIGMA_DIR"; exit 1
  fi
  pushd "$SIGMA_DIR" >/dev/null
  : "${CUDA_VERSION:=12.9}"
  : "${GPU_ARCH:=120}"
  export CUDA_VERSION GPU_ARCH
  echo "Using CUDA_VERSION=$CUDA_VERSION GPU_ARCH=$GPU_ARCH"
  if [[ ! -f .suf_sigma_setup_done ]]; then
    echo "Running setup.sh (may take a while)..."
    bash setup.sh || {
      echo "setup.sh failed. You may need to set a compatible CUTLASS branch."; exit 1;
    }
    touch .suf_sigma_setup_done
  fi
  make sigma
  popd >/dev/null
fi

if $RUN_SIGMA; then
  echo "[Sigma] run (localhost, two parties)"
  echo "This starts two local processes. Adjust CPU threads as needed."
  pushd "$SIGMA_DIR/experiments/sigma" >/dev/null
  if [[ ! -x ./sigma ]]; then
    echo "sigma binary not found; build it with --build-sigma"; exit 1
  fi
  ./sigma bert-tiny 128 0 127.0.0.1 8 > /tmp/sigma_p0.log 2>&1 &
  P0_PID=$!
  sleep 1
  ./sigma bert-tiny 128 1 127.0.0.1 8 > /tmp/sigma_p1.log 2>&1 &
  P1_PID=$!
  wait $P0_PID || true
  wait $P1_PID || true
  echo "Sigma logs: /tmp/sigma_p0.log /tmp/sigma_p1.log"
  popd >/dev/null
fi
