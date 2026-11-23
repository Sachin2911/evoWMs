#!/bin/bash
# Builds the TD-MPC2 image on a cluster node and puts it in ~/images.
#
# Submit from this folder, on bigbatch (see the /tmp note below):
#   ../msCli/ms submit -p bigbatch -t 3h -n build-tdmpc2-image -- bash build_image.sh
set -euo pipefail

IMAGE_DIR="$HOME/images"
WORK_DIR="/tmp/$USER-tdmpc2-build"
CACHE_DIR="/datasets/$USER/tdmpc2_image_cache"
RECIPE="$PWD/tdmpc2.def"

# TD-MPC2's code, at the commit the published image was built for: the package list inside
# that image is the same file, byte for byte, as docker/environment.yaml at this commit.
TDMPC2_COMMIT=02b18a48b1c4e4eae0425eada444fafcb43bca0a
# MuJoCo 2.1.0, the release mujoco-py in the image was compiled against.
MUJOCO_URL=https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
MUJOCO_SHA256=a436ca2f4144c38b837205635bbd60ffe1162d5b44c87df22232795978d7d012

# Apptainer unpacks the published image (9.8 GB compressed, in 24 layers) in its temporary
# folder and writes the new image beside it: by our estimate about 35 GB in all. That goes
# in /tmp, which is on the node itself; shared storage is too slow for so many small files.
# On the cluster /tmp is held in RAM, and only bigbatch nodes have enough of it.
free_gb=$(df --output=avail -BG /tmp | tail -1 | tr -dc '0-9')
if [ "$free_gb" -lt 45 ]; then
    echo "Only ${free_gb} GB free in /tmp on $(hostname); the build needs a bigbatch node." >&2
    exit 1
fi

# The downloaded layers are a few large files, so they can sit on shared storage. That
# leaves /tmp to the unpacking, and a build that fails later does not download them again.
export APPTAINER_TMPDIR="$WORK_DIR/tmp"
export APPTAINER_CACHEDIR="$CACHE_DIR"
rm -rf "$WORK_DIR"
mkdir -p "$IMAGE_DIR" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$WORK_DIR/stage"
# /tmp is the node's memory, so clear it whether the build succeeds or fails.
trap 'rm -rf "$WORK_DIR"' EXIT

# The two folders the recipe copies in. It looks for them in the folder the build is
# started from, so the build is started here.
cd "$WORK_DIR/stage"

git clone --quiet https://github.com/nicklashansen/tdmpc2.git tdmpc2
git -C tdmpc2 checkout --quiet "$TDMPC2_COMMIT"
if [ "$(git -C tdmpc2 rev-parse HEAD)" != "$TDMPC2_COMMIT" ]; then
    echo "TD-MPC2 is not at commit $TDMPC2_COMMIT" >&2
    exit 1
fi
rm -rf tdmpc2/.git

curl --fail --location --silent --show-error --retry 3 --output mujoco210.tar.gz "$MUJOCO_URL"
if ! echo "$MUJOCO_SHA256  mujoco210.tar.gz" | sha256sum --check --status; then
    echo "checksum mismatch: mujoco210.tar.gz" >&2
    exit 1
fi
tar -xzf mujoco210.tar.gz
rm mujoco210.tar.gz

# No --fakeroot: the recipe only copies files, so it needs no admin rights inside the image.
apptainer build "$WORK_DIR/tdmpc2.sif" "$RECIPE"

# The two folders must have landed where the recipe says, not one level deeper.
for path in /opt/mujoco210/bin/libmujoco210.so /opt/tdmpc2/tdmpc2/tdmpc2.py; do
    if ! apptainer exec "$WORK_DIR/tdmpc2.sif" test -f "$path"; then
        echo "The image was built, but $path is not in it." >&2
        exit 1
    fi
done

# One large copy to shared storage, done only after the build has succeeded.
mv "$WORK_DIR/tdmpc2.sif" "$IMAGE_DIR/tdmpc2.sif"
rm -rf "$CACHE_DIR"
ls -lh "$IMAGE_DIR/tdmpc2.sif"
