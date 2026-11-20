#!/bin/bash
# Builds the SafeDreamer image on a cluster node and puts it in ~/images.
#
# Submit from this folder, on bigbatch (see the /tmp note below):
#   ../msCli/ms submit -p bigbatch -t 2h -n build-image -- bash build_image.sh
set -euo pipefail

IMAGE_DIR="$HOME/images"
WORK_DIR="/tmp/$USER-safedreamer-build"

# Apptainer downloads and unpacks the base image in its temporary folders, about 25 GB in all.
# They go in /tmp, which is on the node itself; home is a shared network folder and too slow for this.
# On the cluster /tmp is held in RAM, and only bigbatch nodes have enough of it.
free_gb=$(df --output=avail -BG /tmp | tail -1 | tr -dc '0-9')
if [ "$free_gb" -lt 40 ]; then
    echo "Only ${free_gb} GB free in /tmp on $(hostname); the build needs a bigbatch node." >&2
    exit 1
fi

export APPTAINER_TMPDIR="$WORK_DIR/tmp"
export APPTAINER_CACHEDIR="$WORK_DIR/cache"
mkdir -p "$IMAGE_DIR" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
# /tmp is the node's memory, so clear it whether the build succeeds or fails.
trap 'rm -rf "$WORK_DIR"' EXIT

apptainer build --fakeroot "$WORK_DIR/safeDreamer.sif" safeDreamer.def

# One large copy to shared storage, done only after the build has succeeded.
mv "$WORK_DIR/safeDreamer.sif" "$IMAGE_DIR/safeDreamer.sif"
ls -lh "$IMAGE_DIR/safeDreamer.sif"
