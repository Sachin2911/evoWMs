#!/bin/bash
# Runs one of this folder's scripts inside the image, for the task named with --task=NAME.
#
#   bash on_task.sh search.py --seed 0 --predictor PREDICTOR --task=PointGoal1
#
# --task=NAME is replaced by --checkpoint and the reward-only checkpoint of that task.
# --baseline-task=NAME is replaced by --checkpoint and SafeDreamer's own BSRP-Lag checkpoint
# of that task, for playing the baseline on the same seeds:
#
#   bash on_task.sh ../safeDreamerBaselines/evaluate.py --episodes 300 --first-seed 5000 --baseline-task=PointGoal1
# In every other argument the word TASK is replaced by the task's name, and the word
# PREDICTOR by the task's cost predictor, which is kept in one place for every task:
#
#   /datasets/$USER/adapterRecipe/predictors/NAME/cost_predictor.npz
#
# A task whose name begins with mw- is a Meta-World task of TD-MPC2: it runs in that
# agent's image, and its checkpoint is the released single-task agent, mw-NAME-1.pt.
#
# So one command serves every task, and a sweep over tasks is a grid over --task:
#
#   ../msCli/ms submit --grid "--task=PointGoal1,PointButton1 --seed=0,1,2" -t 6h -n search -- \
#       bash on_task.sh search.py --predictor PREDICTOR
#
# Off the cluster (vast.sh), ADAPTER_DATA names the folder that stands in for /datasets/$USER,
# and ADAPTER_IMAGE=none says that python3 itself has everything and there is no image.
set -euo pipefail

DATA=${ADAPTER_DATA:-/datasets/$USER}
IMAGE=${ADAPTER_IMAGE:-$HOME/images/safeDreamer.sif}
CHECKPOINTS=$DATA/safedreamer_checkpoints/unsafe_dreamerv3
PREDICTORS=$DATA/adapterRecipe/predictors

script=$1
shift

task=
for argument in "$@"; do
    case "$argument" in
        --task=*) task=${argument#--task=}; variant=dreamerv3 ;;
        --baseline-task=*) task=${argument#--baseline-task=}; variant=bsrp_lag ;;
    esac
done
if [ -z "$task" ]; then
    echo "on_task.sh: no --task=NAME among the arguments" >&2
    exit 1
fi
if [ "${task#mw-}" != "$task" ]; then
    checkpoint=$DATA/tdmpc2_checkpoints/metaworld/$task-1.pt
    IMAGE=${ADAPTER_IMAGE:-$HOME/images/tdmpc2.sif}
elif [ "$variant" = dreamerv3 ]; then
    checkpoint=$(ls "$CHECKPOINTS"/*_dreamerv3_safetygym_Safety"$task"-v0_0.ckpt)
else
    # Some tasks have several, trained with different seeds; the first in name order is taken.
    checkpoint=$(ls "$DATA"/safedreamer_checkpoints/safedreamer_bsrplag/*_bsrp_lag_*safetygym_Safety"$task"-v0_*.ckpt | head -1)
fi

arguments=()
for argument in "$@"; do
    case "$argument" in
        --task=*|--baseline-task=*) arguments+=(--checkpoint "$checkpoint") ;;
        PREDICTOR) arguments+=("$PREDICTORS/$task/cost_predictor.npz") ;;
        *) arguments+=("${argument//TASK/$task}") ;;
    esac
done

if [ "$IMAGE" = none ]; then
    exec env XLA_PYTHON_CLIENT_PREALLOCATE=false python3 "$script" "${arguments[@]}"
fi
# --nv gives the container the node's GPU; --bind makes /datasets visible inside it. Without
# the preallocation setting, the first of several runs on a node takes all of the GPU's memory.
exec apptainer exec --nv --bind "$DATA" --env XLA_PYTHON_CLIENT_PREALLOCATE=false "$IMAGE" \
    python3 "$script" "${arguments[@]}"
