#!/bin/bash
# Runs Part A (the rendering check) on a rented machine that starts from
# nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04. Mirrors safeDreamerImage/Dockerfile.
# Usage on the machine:  bash vast_part_a.sh setup | download | run | summary
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
WORK=/workspace/partA
REPO=$WORK/evoWMs
CKPT=$WORK/checkpoints
OUT=$WORK/results
COMMIT=95e1dd4
SOURCE=https://huggingface.co/Weidong-Huang/SafeDreamer/resolve/80ef19ee7a7a8a105489100c1dc6e1681df2c97b

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 LC_ALL=C.UTF-8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/usr/local/cuda-11.8/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda-11.8 --xla_gpu_autotune_level=0"
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # several evaluations share the one GPU
export TF_FORCE_GPU_ALLOW_GROWTH=true

case "$1" in
setup)
    mkdir -p "$WORK"
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends python3 python3-pip python3-dev build-essential git \
        ca-certificates curl libegl1 libgl1 libopengl0 libglvnd0 libosmesa6 libglib2.0-0 >/dev/null
    # These machines already supply this file, read-only; only write it where it is missing.
    mkdir -p /usr/share/glvnd/egl_vendor.d 2>/dev/null || true
    [ -e /usr/share/glvnd/egl_vendor.d/10_nvidia.json ] || printf '%s\n' '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json || true
    [ -d "$REPO" ] || git clone -q https://github.com/Sachin2911/evoWMs.git "$REPO"
    git -C "$REPO" checkout -q "$COMMIT"
    python3 -m pip install -q --no-cache-dir pip==24.0 setuptools==69.5.1 wheel==0.43.0
    python3 -m pip install -q --no-cache-dir -r "$REPO/safeDreamerImage/requirements.txt"
    [ -d /opt/safety-gymnasium ] || git clone -q https://github.com/PKU-Alignment/safety-gymnasium.git /opt/safety-gymnasium
    git -C /opt/safety-gymnasium checkout -q 064c0c3f1dac523ffdf1e4daddba8262779344d4
    python3 -m pip install -q --no-cache-dir --no-deps -e /opt/safety-gymnasium
    [ -d /opt/SafeDreamer ] || git clone -q https://github.com/PKU-Alignment/SafeDreamer.git /opt/SafeDreamer
    git -C /opt/SafeDreamer checkout -q 529221f02b54d5093e304d877eb2aaf6da63a9ff
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    nproc; grep -m1 'model name' /proc/cpuinfo
    python3 -c "import jax, mujoco, safety_gymnasium, tensorflow_probability; print('imports ok: jax', jax.__version__, 'mujoco', mujoco.__version__, jax.devices())"
    # The repo's own check, once with each renderer: tells us whether GPU rendering works here.
    for gl in osmesa egl; do
        echo "== check_env with MUJOCO_GL=$gl"
        MUJOCO_GL=$gl python3 "$REPO/safeDreamerImage/check_env.py" --out "$WORK/check_$gl" 2>&1 | grep -E 'PASS|FAIL|per second|Error' || true
    done
    ;;
download)
    mkdir -p "$CKPT"
    # Machine K (0 to 3) takes every fourth checkpoint of the twelve, in file-name order.
    grep -E '_(dreamerv3|bsrp_lag)_' "$REPO/safeDreamerBaselines/checkpoints.sha256" \
        | awk '{n = split($2, a, "/"); print a[n], $1, $2}' | sort | awk -v k="$K" '(NR - 1) % 4 == k {print $2, $3}' \
        | while read -r checksum path; do
        target="$CKPT/$(basename "$path")"
        if [ -f "$target" ] && echo "$checksum  $target" | sha256sum --check --status; then echo "have $(basename "$path")"; continue; fi
        curl --fail --location --silent --show-error --retry 3 --output "$target.part" "$SOURCE/$path"
        echo "$checksum  $target.part" | sha256sum --check --status || { echo "checksum mismatch: $path" >&2; exit 1; }
        mv "$target.part" "$target"; echo "got  $(basename "$path")"
    done
    ls "$CKPT" | wc -l; du -sh "$CKPT"
    ;;
run)
    # run RENDERER REP PARALLEL: every checkpoint, 50 episodes, results in $OUT/RENDERER-REP/
    gl=$2; rep=$3; par=${4:-6}
    mkdir -p "$OUT/$gl-$rep" "$WORK/logs"
    cd "$REPO/safeDreamerBaselines"
    ls "$CKPT"/*.ckpt | LP_NUM_THREADS=1 MUJOCO_GL=$gl xargs -P "$par" -I{} sh -c \
        'name=$(basename {} .ckpt); python3 evaluate.py --checkpoint {} --out '"$OUT/$gl-$rep"'/$name.jsonl > '"$WORK/logs/$gl-$rep"'-$name.log 2>&1 || echo "FAILED $name"'
    wc -l "$OUT/$gl-$rep"/*.jsonl | tail -1
    ;;
all)
    bash "$0" download
    # GPU rendering first (quick), then CPU rendering; two repeats of each, side by side.
    for gl in egl osmesa; do
        bash "$0" run $gl 1 3 & bash "$0" run $gl 2 3 & wait
        echo "finished $gl at $(date +%H:%M)"
    done
    echo ALLDONE
    ;;
summary)
    wc -l "$OUT"/*/*.jsonl 2>/dev/null | tail -1
    grep -L 'episodes: reward' "$WORK"/logs/*.log 2>/dev/null | head || true
    ;;
esac
