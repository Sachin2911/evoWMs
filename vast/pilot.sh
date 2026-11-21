#!/bin/bash
# The adapter pilot on a rented machine that starts from
# nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04.
#
# Usage on the machine:  COMMIT=<git commit> bash pilot.sh COMMAND ...
#
#   setup                       the SafeDreamer environment (part_a.sh), the cma package and
#                               the PointGoal1 reward-only checkpoint
#   unpack                      unpacks recordings/*.tar, copied here from the laptop
#   efficiency NAME ...         for each recorded checkpoint NAME: the cost predictors trained
#                               on 10, 25, 50, 100 and 150 episodes
#   run LABEL READS SCORING SEED
#                               one evolution run on PointGoal1 and the evaluation of the
#                               adapters it keeps, in runs/LABEL
set -euo pipefail
WORK=/workspace/pilot
REPO=${REPO:-$WORK/evoWMs}
CKPT=$WORK/checkpoints
SOURCE=https://huggingface.co/Weidong-Huang/SafeDreamer/resolve/80ef19ee7a7a8a105489100c1dc6e1681df2c97b
POINTGOAL1=20240327-144037_dreamerv3_safetygym_SafetyPointGoal1-v0_0
INITIAL=${INITIAL:-50}        # recorded episodes the cost network learns from
BUDGET=${BUDGET:-1000000}     # simulator steps per evolution run

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 LC_ALL=C.UTF-8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/usr/local/cuda-11.8/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda-11.8 --xla_gpu_autotune_level=0"
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # several processes share the one GPU
export TF_FORCE_GPU_ALLOW_GROWTH=true
export MUJOCO_GL=egl                         # cameras rendered on the GPU, as in the baselines

mkdir -p "$WORK/logs" "$CKPT"
if [ "$1" != setup ] && [ "$1" != unpack ] && [ -z "${REPO_FIXED:-}" ]; then
    # Several commands may start at once on a machine; they take turns here, and code that is
    # already at the right commit is left alone while others are running from it.
    exec 9> "$WORK/checkout.lock"
    flock 9
    [ -d "$REPO" ] || git clone -q https://github.com/Sachin2911/evoWMs.git "$REPO"
    if [ "$(git -C "$REPO" rev-parse HEAD)" != "$(git -C "$REPO" rev-parse --verify -q "$COMMIT^{commit}")" ]; then
        git -C "$REPO" fetch -q origin
        git -C "$REPO" checkout -q "$COMMIT"
    fi
    flock -u 9
fi

case "$1" in
setup)
    bash "$(dirname "$0")/part_a.sh" setup
    python3 -m pip install -q --no-cache-dir cma==3.3.0
    python3 -c "import cma; print('cma', cma.__version__)"
    line=$(grep "/$POINTGOAL1.ckpt" /workspace/partA/evoWMs/safeDreamerBaselines/checkpoints.sha256)
    checksum=${line%% *}; path=${line##* }
    target="$CKPT/$POINTGOAL1.ckpt"
    if ! { [ -f "$target" ] && echo "$checksum  $target" | sha256sum --check --status; }; then
        curl --fail --location --silent --show-error --retry 3 --output "$target.part" "$SOURCE/$path"
        echo "$checksum  $target.part" | sha256sum --check --status || { echo "checksum mismatch" >&2; exit 1; }
        mv "$target.part" "$target"
    fi
    echo SETUP-DONE
    ;;
unpack)
    for archive in "$WORK"/recordings/*.tar; do tar -xf "$archive" -C "$WORK/recordings"; done
    ls "$WORK"/recordings/chunk0
    ;;
efficiency)
    shift
    for name in "$@"; do
        for episodes in 10 25 50 100 150; do
            out="$WORK/efficiency/$name/episodes_$episodes"
            [ -f "$out/results.json" ] && continue
            python3 "$REPO/costPredictor/train_cost_predictor.py" --labelled-episodes "$episodes" \
                --predictors latent deter privileged shuffled \
                --recordings "$WORK"/recordings/chunk*/"$name" --out "$out" \
                > "$WORK/logs/efficiency-$name-$episodes.log" 2>&1
            rm -f "$out"/latent_*.npz
        done
        echo "done $name"
    done
    echo EFFICIENCY-DONE
    ;;
run)
    label=$2; reads=$3; scoring=$4; seed=$5
    out="$WORK/runs/$label"
    mkdir -p "$out"
    recordings=("$WORK"/recordings/chunk*/"$POINTGOAL1")
    {
        cd "$REPO/adapterPilot"
        [ -f "$out/cost_net.npz" ] || python3 cost_features.py --recordings "${recordings[@]}" --episodes "$INITIAL" --out "$out/cost_net.npz"
        [ -f "$out/finished.json" ] || python3 evolve.py --checkpoint "$CKPT/$POINTGOAL1.ckpt" --recordings "${recordings[@]}" \
            --cost-net "$out/cost_net.npz" --initial-episodes "$INITIAL" --reads "$reads" --scoring "$scoring" \
            --seed "$seed" --budget "$BUDGET" --out "$out" ${EVOLVE_OPTIONS:-}
        cd "$REPO/safeDreamerBaselines"
        # The adapters that were kept, on the 50 evaluation seeds the baselines were measured on.
        for kept in adapter_half_budget adapter_full_budget; do
            [ -f "$out/$kept.npz" ] && python3 evaluate.py --checkpoint "$CKPT/$POINTGOAL1.ckpt" \
                --adapter "$out/$kept.npz" --episodes "${EVALUATION_EPISODES:-50}" --out "$out/evaluation_$kept.jsonl"
        done
        echo RUN-DONE
    } > "$WORK/logs/run-$label.log" 2>&1
    tail -n 3 "$WORK/logs/run-$label.log"
    ;;
esac
