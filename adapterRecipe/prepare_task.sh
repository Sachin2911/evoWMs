#!/bin/bash
# Makes what a search of one task begins with: episodes of the unadapted agent, and the
# first cost predictor fitted to them.
#
#   bash prepare_task.sh --task=CarGoal1
#
# 80 episodes are played on seeds 1000 upward: 40 that only score the predictor, 20 that
# choose its strength and 20 it is fitted to. The 40 a search counts as its own are the
# last two. They are kept in one place for every task, where on_task.sh looks for them:
#
#   /datasets/$USER/adapterRecipe/episodes/NAME/
#   /datasets/$USER/adapterRecipe/predictors/NAME/cost_predictor.npz
#
# Episodes that are already there are not played again. For many tasks at once:
#
#   ../msCli/ms submit --grid "--task=CarGoal1,PointPush1" -t 3h -n prepare -- bash prepare_task.sh
set -euo pipefail

DATA=${ADAPTER_DATA:-/datasets/$USER}
IMAGE=${ADAPTER_IMAGE:-$HOME/images/safeDreamer.sif}
EPISODES=$DATA/adapterRecipe/episodes
PREDICTORS=$DATA/adapterRecipe/predictors

task=
for argument in "$@"; do
    case "$argument" in --task=*) task=${argument#--task=} ;; esac
done
if [ -z "$task" ]; then
    echo "prepare_task.sh: no --task=NAME among the arguments" >&2
    exit 1
fi

bash on_task.sh episodes.py --episodes 80 --out "$EPISODES/$task" --task="$task"
if [ "$IMAGE" = none ]; then
    exec python3 cost_predictor.py --episodes "$EPISODES/$task" --train 20 --out "$PREDICTORS/$task"
fi
# The predictor is fitted on the CPU, so the container needs no GPU.
apptainer exec --bind "$DATA" "$IMAGE" \
    python3 cost_predictor.py --episodes "$EPISODES/$task" --train 20 --out "$PREDICTORS/$task"
