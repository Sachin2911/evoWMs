#!/bin/bash
# Starts the live demo (live.py) on this machine, inside the TD-MPC2 image, and prints the address to open.
#
#   bash run.sh                  # then open http://127.0.0.1:8000
#   bash run.sh --task mw-window-open --port 8001
#
# It needs, on this machine:
#   the image             ~/images/tdmpc2.sif, built from tdmpc2Probe/tdmpc2.def
#   the released agents   ~/evoData/tdmpc2_checkpoints/metaworld/mw-TASK-1.pt (tdmpc2Probe/checkpoints.sha256)
# ADAPTER_MW_IMAGE and ADAPTER_DATA name other places for them, as they do for on_task.sh.
set -euo pipefail

IMAGE=${ADAPTER_MW_IMAGE:-$HOME/images/tdmpc2.sif}
DATA=${ADAPTER_DATA:-$HOME/evoData}

cd "$(dirname "$0")"
exec apptainer exec --nv "$IMAGE" python3 live.py --checkpoints "$DATA/tdmpc2_checkpoints/metaworld" "$@"
