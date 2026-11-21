#!/bin/bash
# Evaluates one released checkpoint: task N is the checkpoint on line N+1 of checkpoints.sha256.
#
# Submit from this folder, as one task per checkpoint. The first 8 lines of the list are
# the PointGoal1 checkpoints, so for those alone:
#   ../msCli/ms submit --array 8 -t 6h -n baseline -- bash run_baseline.sh {i}
# and for all 31:
#   ../msCli/ms submit --array 31 -t 6h -n baseline -- bash run_baseline.sh {i}
#
# Anything after the task number is passed on to evaluate.py, e.g. --episodes 10.
# Each task writes its results to episodes.jsonl in its own run folder.
set -euo pipefail

IMAGE=$HOME/images/safeDreamer.sif
CHECKPOINTS=/datasets/$USER/safedreamer_checkpoints

task=$1
shift

# Each line of the list is a checksum and then the checkpoint's path; we want the path.
path=$(sed -n "$((task + 1))p" checkpoints.sha256 | awk '{print $2}')
if [ -z "$path" ]; then
    echo "checkpoints.sha256 has no line $((task + 1))" >&2
    exit 1
fi

# --nv gives the container the node's GPU; --bind makes /datasets visible inside it.
exec apptainer exec --nv --bind "/datasets/$USER" "$IMAGE" \
    python3 evaluate.py --checkpoint "$CHECKPOINTS/$path" "$@"
