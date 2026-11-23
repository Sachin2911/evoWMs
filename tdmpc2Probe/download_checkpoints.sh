#!/bin/bash
# Downloads the released TD-MPC2 checkpoints listed in checkpoints.sha256 to /datasets.
#
# Submit from this folder:
#   ../msCli/ms submit -t 30m -n get-tdmpc2-checkpoints -- bash download_checkpoints.sh
#
# Each line of checkpoints.sha256 is a file's checksum and its path in the Hugging Face
# repo. A file that is already there with the right checksum is skipped, so the job can
# be rerun or requeued safely.
#
# The repo holds three agents for each of the 50 Meta-World tasks, as
# metaworld/mw-TASK-SEED.pt with SEED 1, 2 or 3. To add one, add a line with its checksum,
# which the file's page on Hugging Face shows as "SHA256".
set -euo pipefail

DEST=/datasets/$USER/tdmpc2_checkpoints
# Pinned to one revision of the Hugging Face repo, so the files cannot change under us.
SOURCE=https://huggingface.co/nicklashansen/tdmpc2/resolve/8fb2a82efb3bae96941da440128fe1332e4394fd

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

echo "$(find "$DEST" -name '*.pt' | wc -l) checkpoints, $(du -sh "$DEST" | cut -f1) in $DEST"
