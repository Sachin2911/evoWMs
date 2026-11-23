#!/bin/bash
# Runs one of this folder's scripts inside the image, for the task named with --task=NAME.
#
#   bash on_task.sh search.py --seed 0 --predictor PREDICTOR --task=PointGoal1
#
# --task=NAME is replaced by --checkpoint and the reward-only checkpoint of that task.
# In every other argument the word TASK is replaced by the task's name, and the word
# PREDICTOR by the task's cost predictor, which is kept in one place for every task:
#
#   /datasets/$USER/adapterRecipe/predictors/NAME/cost_predictor.npz
#
# So one command serves every task, and a sweep over tasks is a grid over --task:
#
#   ../msCli/ms submit --grid "--task=PointGoal1,PointButton1 --seed=0,1,2" -t 6h -n search -- \
#       bash on_task.sh search.py --predictor PREDICTOR
set -euo pipefail

IMAGE=$HOME/images/safeDreamer.sif
CHECKPOINTS=/datasets/$USER/safedreamer_checkpoints/unsafe_dreamerv3
PREDICTORS=/datasets/$USER/adapterRecipe/predictors

script=$1
shift

task=
for argument in "$@"; do
    case "$argument" in --task=*) task=${argument#--task=} ;; esac
done
if [ -z "$task" ]; then
    echo "on_task.sh: no --task=NAME among the arguments" >&2
    exit 1
fi
checkpoint=$(ls "$CHECKPOINTS"/*_dreamerv3_safetygym_Safety"$task"-v0_0.ckpt)

arguments=()
for argument in "$@"; do
    case "$argument" in
        --task=*) arguments+=(--checkpoint "$checkpoint") ;;
        PREDICTOR) arguments+=("$PREDICTORS/$task/cost_predictor.npz") ;;
        *) arguments+=("${argument//TASK/$task}") ;;
    esac
done

# --nv gives the container the node's GPU; --bind makes /datasets visible inside it. Without
# the preallocation setting, the first of several runs on a node takes all of the GPU's memory.
exec apptainer exec --nv --bind "/datasets/$USER" --env XLA_PYTHON_CLIENT_PREALLOCATE=false "$IMAGE" \
    python3 "$script" "${arguments[@]}"
