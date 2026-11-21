#!/bin/bash
# Builds the Hugging Face dataset from the recordings, and uploads it if asked to.
#
# First record every checkpoint, from this folder:
#   ../msCli/ms submit --array 31 -t 6h -n record -- bash run_baseline.sh {i} \
#       --record /datasets/smohan/safedreamer_evals/recordings --video-seeds 0 1 2 3 4
# Then build, and upload to a private dataset. --earlier names the run folders of an earlier,
# separate evaluation of the same episodes, to mark which episodes came out the same twice:
#   ../msCli/ms submit -t 2h -n build-dataset -- bash build_dataset.sh \
#       --earlier /home-mscluster/smohan/runs/RUN1 /home-mscluster/smohan/runs/RUN2 --upload ACCOUNT/NAME
# Without --upload it only builds, into /datasets/$USER/safedreamer_evals/dataset.
set -euo pipefail

IMAGE=$HOME/images/safeDreamer.sif
EVALS=/datasets/$USER/safedreamer_evals
# Two packages the image does not have, installed once into a folder of their own.
# They only write the step tables and do the upload, so they are not part of the image.
TOOLS=$HOME/pylibs/dataset-tools

if [ ! -d "$TOOLS/pyarrow" ] || [ ! -d "$TOOLS/huggingface_hub" ]; then
    apptainer exec "$IMAGE" python3 -m pip install --quiet --target "$TOOLS" \
        pyarrow==17.0.0 huggingface_hub==0.25.2
fi

# PYTHONPATH makes Python look in the tools folder as well as in the image.
apptainer exec --bind "/datasets/$USER" --env "PYTHONPATH=$TOOLS" "$IMAGE" \
    python3 build_dataset.py --recordings "$EVALS/recordings" --out "$EVALS/dataset" "$@"
