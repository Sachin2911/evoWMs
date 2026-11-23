#!/bin/bash
# Runs the adapter recipe on a rented machine that has no Apptainer, for when the cluster
# cannot be used.
#
# The machine starts from the image the cluster's own image is built on,
# nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04, and `setup` installs what
# safeDreamerImage/Dockerfile installs. Everything is kept under /workspace:
#
#   /workspace/evoWMs                    this repo
#   /workspace/data                      checkpoints, episodes and predictors, laid out as /datasets/$USER is
#   /workspace/runs/NAME/task-NNNN       one folder per run, as ms makes them, with its task.log
#   /workspace/queue                     runs that wait for a GPU
#
#   bash vast.sh setup COMMIT            install everything, and check that the simulator renders on every GPU
#   bash vast.sh checkpoints TASK ...    download the reward-only and the BSRP-Lag checkpoints of each task
#   bash vast.sh workers N               start N workers per GPU; each takes one run at a time from the queue
#   bash vast.sh sweep NAME "GRID" COMMAND ...
#                                        queue one run per combination of the grid, with its settings added to
#                                        the end of the command, as ms submit --grid does. COMMAND runs in
#                                        adapterRecipe/, so it is usually: bash on_task.sh SCRIPT.py ...
#   bash vast.sh status                  how many runs wait, run, are done and have failed
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

WORK=/workspace
REPO=$WORK/evoWMs
SOURCE=https://huggingface.co/Weidong-Huang/SafeDreamer/resolve/80ef19ee7a7a8a105489100c1dc6e1681df2c97b

# What the cluster's image sets (safeDreamerImage/Dockerfile and safeDreamer.def).
export MUJOCO_GL=egl PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 LC_ALL=C.UTF-8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/lib64:/usr/local/cuda-11.8/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda-11.8 --xla_gpu_autotune_level=0"
# Several runs share a GPU, so none may take all of its memory at the start.
export XLA_PYTHON_CLIENT_PREALLOCATE=false TF_FORCE_GPU_ALLOW_GROWTH=true
# Many runs share the processor, so each keeps its arithmetic to two threads.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
# on_task.sh and prepare_task.sh: where the data is, and that there is no image to run in.
export ADAPTER_DATA=$WORK/data ADAPTER_IMAGE=none

gpus() { nvidia-smi --query-gpu=index --format=csv,noheader | wc -l; }

case "${1:-}" in
setup)
    commit=$2
    mkdir -p "$WORK"/{data,runs,queue,claimed,workers}
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends python3 python3-pip python3-dev build-essential git rsync \
        ca-certificates curl libegl1 libgl1 libopengl0 libglvnd0 libosmesa6 libglib2.0-0 >/dev/null
    # Tells the graphics library to render on the NVIDIA card. Rented machines usually supply this file already.
    mkdir -p /usr/share/glvnd/egl_vendor.d 2>/dev/null || true
    [ -e /usr/share/glvnd/egl_vendor.d/10_nvidia.json ] || printf '%s\n' \
        '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json || true
    [ -d "$REPO" ] || git clone -q https://github.com/Sachin2911/evoWMs.git "$REPO"
    git -C "$REPO" fetch -q origin
    git -C "$REPO" checkout -q "$commit"
    python3 -m pip install -q --no-cache-dir pip==24.0 setuptools==69.5.1 wheel==0.43.0
    python3 -m pip install -q --no-cache-dir -r "$REPO/safeDreamerImage/requirements.txt"
    [ -d /opt/safety-gymnasium ] || git clone -q https://github.com/PKU-Alignment/safety-gymnasium.git /opt/safety-gymnasium
    git -C /opt/safety-gymnasium checkout -q 064c0c3f1dac523ffdf1e4daddba8262779344d4
    python3 -m pip install -q --no-cache-dir --no-deps -e /opt/safety-gymnasium
    [ -d /opt/SafeDreamer ] || git clone -q https://github.com/PKU-Alignment/SafeDreamer.git /opt/SafeDreamer
    git -C /opt/SafeDreamer checkout -q 529221f02b54d5093e304d877eb2aaf6da63a9ff
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
    echo "$(nproc) processors, $(free -g | awk '/Mem:/ {print $2}') GB of memory"
    python3 -c "import jax, mujoco, safety_gymnasium, scipy; print('imports ok: jax', jax.__version__, 'mujoco', mujoco.__version__, jax.devices())"
    # The repo's own check of the simulator, rendering on each GPU in turn.
    for gpu in $(seq 0 $(($(gpus) - 1))); do
        echo "== check_env on GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu python3 "$REPO/safeDreamerImage/check_env.py" \
            --out "$WORK/check_$gpu" 2>&1 | grep -E 'PASS|FAIL|per second|Error' || true
    done
    ;;
checkpoints)
    shift
    for task in "$@"; do
        grep -E "_(dreamerv3|bsrp_lag)_([0-9.]+_)?safetygym_Safety${task}-v0_" "$REPO/safeDreamerBaselines/checkpoints.sha256" \
            | while read -r checksum path; do
            target="$ADAPTER_DATA/safedreamer_checkpoints/$path"
            if [ -f "$target" ] && echo "$checksum  $target" | sha256sum --check --status; then echo "have  $path"; continue; fi
            mkdir -p "$(dirname "$target")"
            curl --fail --location --silent --show-error --retry 3 --output "$target.part" "$SOURCE/$path"
            echo "$checksum  $target.part" | sha256sum --check --status || { echo "checksum mismatch: $path" >&2; exit 1; }
            mv "$target.part" "$target"
            echo "got   $path"
        done
    done
    ;;
workers)
    for gpu in $(seq 0 $(($(gpus) - 1))); do
        for slot in $(seq 1 "$2"); do
            nohup setsid bash "$0" worker "$gpu" > "$WORK/workers/gpu$gpu-slot$slot.log" 2>&1 &
        done
    done
    echo "$(($(gpus) * $2)) workers started"
    ;;
worker)
    # Takes the oldest run in the queue, again and again. Moving its file is what claims it:
    # only one worker's move of a file succeeds.
    gpu=$2
    while true; do
        task=$(ls "$WORK/queue" | sort | head -1)
        if [ -z "$task" ] || ! mv "$WORK/queue/$task" "$WORK/claimed/$task" 2>/dev/null; then
            sleep $((10 + RANDOM % 10))
            continue
        fi
        run=$(sed -n 's/^export MS_RUN_DIR=//p' "$WORK/claimed/$task")
        mkdir -p "$run"
        echo "start $(date -u +%FT%TZ) gpu=$gpu host=$(hostname)" >> "$run/task.log"
        code=0
        CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu bash "$WORK/claimed/$task" >> "$run/task.log" 2>&1 || code=$?
        echo "exit $code $(date -u +%FT%TZ)" >> "$run/task.log"
        echo "$code" > "$run/exit_code"
    done
    ;;
sweep)
    name=$2; grid=$3; shift 3
    mkdir -p "$WORK/runs/$name"
    git -C "$REPO" rev-parse HEAD > "$WORK/runs/$name/commit.txt"
    printf '%q ' "$@" > "$WORK/runs/$name/command.txt"
    # Every combination of the grid's comma lists, one per line.
    python3 - "$grid" > "$WORK/runs/$name/tasks.txt" <<'EOF'
import itertools
import sys

parts = [part.split('=', 1) for part in sys.argv[1].split()]
for values in itertools.product(*[listed.split(',') for _, listed in parts]):
    print(' '.join(f'{name}={value}' for (name, _), value in zip(parts, values)))
EOF
    number=0
    while read -r settings; do
        # Written beside the queue and then moved in, so a worker never takes a half-written file.
        file=$(date +%s%N)-$name-$(printf '%04d' "$number")
        {
            echo "export MS_RUN_DIR=$WORK/runs/$name/task-$(printf '%04d' "$number")"
            echo "cd $REPO/adapterRecipe"
            printf 'exec '; printf '%q ' "$@"; echo "$settings"
        } > "$WORK/workers/$file"
        mv "$WORK/workers/$file" "$WORK/queue/$file"
        number=$((number + 1))
    done < "$WORK/runs/$name/tasks.txt"
    echo "queued $number runs of $name"
    ;;
status)
    echo "waiting $(ls "$WORK/queue" | wc -l), claimed $(ls "$WORK/claimed" | wc -l)"
    for sweep in "$WORK"/runs/*/; do
        all=$(wc -l < "$sweep/tasks.txt")
        done_=$(cat "$sweep"/task-*/exit_code 2>/dev/null | grep -c '^0$' || true)
        failed=$(cat "$sweep"/task-*/exit_code 2>/dev/null | grep -vc '^0$' || true)
        echo "$(basename "$sweep"): $all runs, $done_ done, $failed failed"
    done
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
    uptime
    ;;
*)
    sed -n '2,24p' "$0"
    exit 1
    ;;
esac
