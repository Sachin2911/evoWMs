#!/bin/bash
# Downloads the released SafeDreamer checkpoints listed in checkpoints.sha256 to /datasets.
#
# Submit from this folder:
#   ../msCli/ms submit -t 2h -n get-checkpoints -- bash download_checkpoints.sh
#
# Each line of checkpoints.sha256 is a file's checksum and its path in the Hugging Face
# repo. A file that is already there with the right checksum is skipped, so the job can
# be rerun or requeued safely.
set -euo pipefail

DEST=/datasets/$USER/safedreamer_checkpoints
# Pinned to one revision of the Hugging Face repo, so the files cannot change under us.
SOURCE=https://huggingface.co/Weidong-Huang/SafeDreamer/resolve/80ef19ee7a7a8a105489100c1dc6e1681df2c97b

matches() {  # matches CHECKSUM FILE: true if FILE exists and has that checksum
    [ -f "$2" ] && echo "$1  $2" | sha256sum --check --status
}

while read -r checksum path; do
    target="$DEST/$path"
    if matches "$checksum" "$target"; then
        echo "have  $path"
        continue
    fi
    mkdir -p "$(dirname "$target")"
    # Download under a temporary name, and only rename once the checksum is right,
    # so a half-finished download is never mistaken for a checkpoint.
    curl --fail --location --silent --show-error --retry 3 --output "$target.part" "$SOURCE/$path"
    if ! matches "$checksum" "$target.part"; then
        echo "checksum mismatch: $path" >&2
        exit 1
    fi
    mv "$target.part" "$target"
    echo "got   $path"
done < checkpoints.sha256

echo "$(find "$DEST" -name '*.ckpt' | wc -l) checkpoints, $(du -sh "$DEST" | cut -f1) in $DEST"
