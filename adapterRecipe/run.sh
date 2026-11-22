#!/bin/bash
# One evolution run on the cluster: task N is line N+1 of a list of runs.
#
# Submit from this folder, one task per line of the list, several tasks to a node:
#   ../msCli/ms submit --array 47 --per-node 4 -t 4h -n round1 -- bash run.sh configs/round1.txt {i}
#
# Each line of the list is: a label, a task (PointGoal1, CarGoal1, PointButton1, PointPush1
# or PointGoal2), and then the options for evolve.py. Results go to the task's run folder.
# A third argument names another script of this folder to run in place of evolve.py.
set -euo pipefail

IMAGE=$HOME/images/safeDreamer.sif
CHECKPOINTS=/datasets/$USER/safedreamer_checkpoints/unsafe_dreamerv3
# The cma package is not in the image; it is installed once into a folder of its own.
TOOLS=$HOME/pylibs/evo-tools

list=$1
task=$2
script=${3:-evolve.py}
line=$(sed -n "$((task + 1))p" "$list")
if [ -z "$line" ]; then
    echo "$list has no line $((task + 1))" >&2
    exit 1
fi
read -r label world options <<< "$line"
case "$world" in
    PointGoal1)   checkpoint=20240327-144037_dreamerv3_safetygym_SafetyPointGoal1-v0_0.ckpt ;;
    CarGoal1)     checkpoint=20240327-000000_dreamerv3_safetygym_SafetyCarGoal1-v0_0.ckpt ;;
    PointButton1) checkpoint=20240327-144037_dreamerv3_safetygym_SafetyPointButton1-v0_0.ckpt ;;
    PointPush1)   checkpoint=20240327-144037_dreamerv3_safetygym_SafetyPointPush1-v0_0.ckpt ;;
    PointGoal2)   checkpoint=20240327-144638_dreamerv3_safetygym_SafetyPointGoal2-v0_0.ckpt ;;
    *) echo "unknown task $world" >&2; exit 1 ;;
esac

out=${MS_RUN_DIR:-$PWD/local-run}
mkdir -p "$out" "$(dirname "$TOOLS")"
echo "$label $world $options" > "$out/run.txt"

# Tasks on one node start together; they take turns here so cma is installed only once.
(
    flock 9
    if [ ! -d "$TOOLS/cma" ]; then
        apptainer exec "$IMAGE" python3 -m pip install --quiet --target "$TOOLS" --no-deps cma==3.3.0
    fi
) 9> "$TOOLS.lock"

# The run uses its own copy of the code, so syncing a newer commit cannot change a run that has started.
code=$out/code
if [ ! -d "$code" ]; then
    mkdir -p "$code"
    cp -r ../adapterRecipe ../adapterPilot ../safeDreamerBaselines "$code/"
    # Which code this is: the git commit, or the VERSION file of a copy that was sent without git.
    if [ -f ../VERSION ]; then cp ../VERSION "$code/VERSION"; else git rev-parse HEAD > "$code/VERSION"; fi
fi

# --nv gives the container the node's GPU; --bind makes /datasets visible inside it.
# Several runs share the node's one GPU, so none of them may take all of its memory.
# shellcheck disable=SC2086
exec apptainer exec --nv --bind "/datasets/$USER" \
    --env "PYTHONPATH=$TOOLS" --env XLA_PYTHON_CLIENT_PREALLOCATE=false --env TF_FORCE_GPU_ALLOW_GROWTH=true \
    "$IMAGE" python3 "$code/adapterRecipe/$script" \
    --checkpoint "$CHECKPOINTS/$checkpoint" --out "$out" $options
