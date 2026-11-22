#!/bin/bash
# Runs collect.py on the cluster, inside the image, with the Hugging Face package it needs.
#
#   ../msCli/ms submit -t 1h -n collect -- bash collect.sh --sweeps round1=/home-mscluster/$USER/runs/RUN \
#       --out /datasets/$USER/adapter_recipe_search --upload ACCOUNT/NAME
set -euo pipefail

IMAGE=$HOME/images/safeDreamer.sif
# huggingface_hub is not in the image; build_dataset.sh installed it here.
TOOLS=$HOME/pylibs/dataset-tools

exec apptainer exec --bind "/datasets/$USER" --env "PYTHONPATH=$TOOLS" "$IMAGE" python3 collect.py "$@"
