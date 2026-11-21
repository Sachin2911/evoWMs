#!/bin/bash
# Part B (the cost-predictor test) on a rented machine that part_a.sh has already set up.
# Usage on the machine:  COMMIT=<git commit> bash part_b.sh collect | train
#
# collect: plays each reward-only checkpoint on the machine for 200 episodes (seeds 1000 to
#          1199, kept apart from the evaluation seeds 0 to 49), saving every step and the
#          world model's internal state at every decision.
# train:   trains and tests the cost predictors on those recordings.
set -euo pipefail
WORK=/workspace/partB
REPO=$WORK/evoWMs
CKPT=/workspace/partA/checkpoints

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 LC_ALL=C.UTF-8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/usr/local/cuda-11.8/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda-11.8 --xla_gpu_autotune_level=0"
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # several processes share the one GPU
export MUJOCO_GL=egl                         # cameras rendered on the GPU, as in the baselines

mkdir -p "$WORK/logs"
[ -d "$REPO" ] || git clone -q https://github.com/Sachin2911/evoWMs.git "$REPO"
git -C "$REPO" fetch -q origin
git -C "$REPO" checkout -q "$COMMIT"

case "$1" in
collect)
    cd "$REPO/safeDreamerBaselines"
    # Four processes of 50 episodes each per checkpoint, each writing to its own folder.
    for checkpoint in "$CKPT"/*_dreamerv3_*.ckpt; do
        for chunk in 0 1 2 3; do
            python3 evaluate.py --checkpoint "$checkpoint" --episodes 50 --first-seed $((1000 + 50 * chunk)) \
                --record "$WORK/recordings/chunk$chunk" --record-latents \
                > "$WORK/logs/collect-$(basename "$checkpoint" .ckpt)-$chunk.log" 2>&1 &
        done
    done
    wait
    cat "$WORK"/recordings/chunk*/*/episodes.jsonl | wc -l
    du -sh "$WORK/recordings"
    echo COLLECTED
    ;;
train)
    for folder in "$WORK"/recordings/chunk0/*/; do
        name=$(basename "$folder")
        python3 "$REPO/costPredictor/train_cost_predictor.py" \
            --recordings "$WORK"/recordings/chunk*/"$name" --out "$WORK/results/$name" \
            > "$WORK/logs/train-$name.log" 2>&1
        tail -n 3 "$WORK/logs/train-$name.log"
    done
    echo TRAINED
    ;;
esac
