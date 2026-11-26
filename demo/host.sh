#!/bin/bash
# Serves the live demo (live.py) from a rented machine, for showing it on a phone or any other screen.
#
# The machine starts from nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 with port 8000 open, as
# adapterRecipe/vast.sh's machines do, and runs TD-MPC2 the way `vast.sh tdmpc2` sets it up: its authors'
# image, pulled with udocker. Everything is kept under /workspace. On the machine:
#
#   bash host.sh setup          install udocker, TD-MPC2's image and code, this repo and the released agents
#   bash host.sh start KEY      serve the demo on port 8000 to anyone who opens it with ?key=KEY, and start
#                               it again if it stops. The log is /workspace/demo.log
#   bash host.sh stop
#
# The pictures are drawn at 400 pixels, packed harder than on the laptop, and sent at most 6 times a second: a
# phone's screen is small, every decision sends two pictures, and a phone may be paying for the data. SIZE,
# QUALITY and RATE in the environment of `start` set others.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

WORK=/workspace
REPO=$WORK/evoWMs
DATA=$WORK/data
export UDOCKER_DIR=$WORK/td/udocker TDMPC2_DIR=$WORK/td/tdmpc2

case "${1:-}" in
setup)
    mkdir -p "$WORK/td" "$DATA"
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends python3 python3-pip git ca-certificates curl >/dev/null
    [ -d "$REPO" ] || git clone -q --branch "${2:-adapterMemory}" https://github.com/Sachin2911/evoWMs.git "$REPO"
    python3 -m pip install -q udocker
    udocker --allow-root install
    udocker --allow-root pull nicklashansen/tdmpc2:1.0.0
    udocker --allow-root create --name=td nicklashansen/tdmpc2:1.0.0
    udocker --allow-root setup --nvidia --force td
    [ -d "$TDMPC2_DIR" ] || git clone -q https://github.com/nicklashansen/tdmpc2.git "$TDMPC2_DIR"
    git -C "$TDMPC2_DIR" checkout -q 02b18a48b1c4e4eae0425eada444fafcb43bca0a
    while read -r checksum path; do
        target="$DATA/tdmpc2_checkpoints/$path"
        if [ -f "$target" ] && echo "$checksum  $target" | sha256sum --check --status; then echo "have  $path"; continue; fi
        mkdir -p "$(dirname "$target")"
        curl --fail --location --silent --show-error --retry 3 --output "$target.part" \
            "https://huggingface.co/nicklashansen/tdmpc2/resolve/8fb2a82efb3bae96941da440128fe1332e4394fd/$path"
        echo "$checksum  $target.part" | sha256sum --check --status || { echo "checksum mismatch: $path" >&2; exit 1; }
        mv "$target.part" "$target"
        echo "got   $path"
    done < "$REPO/tdmpc2Probe/checkpoints.sha256"
    echo "set up"
    ;;
start)
    key=${2:?usage: host.sh start KEY}
    bash "$0" stop
    # In a loop, so that a crash in the middle of a showing costs a few seconds and not the demo.
    nohup bash -c "while true; do
        udocker --allow-root run --volume=/workspace --workdir=$REPO/demo --env=MUJOCO_PY_FORCE_CPU=1 \
            --env=TDMPC2_DIR=$TDMPC2_DIR td python3 live.py --checkpoints $DATA/tdmpc2_checkpoints/metaworld \
            --host 0.0.0.0 --port 8000 --key '$key' --size ${SIZE:-400} --quality ${QUALITY:-60} --rate ${RATE:-6}
        echo \"stopped \$(date -u +%FT%TZ); starting again\"; sleep 3
    done" > "$WORK/demo.log" 2>&1 &
    echo $! > "$WORK/demo.pid"
    echo "started; the log is $WORK/demo.log"
    ;;
stop)
    if [ -f "$WORK/demo.pid" ]; then
        kill "$(cat "$WORK/demo.pid")" 2>/dev/null || true
        rm -f "$WORK/demo.pid"
    fi
    pkill -f "live.py --checkpoints" 2>/dev/null || true
    ;;
*)
    sed -n '2,13p' "$0"
    ;;
esac
