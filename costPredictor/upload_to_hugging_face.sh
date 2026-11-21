#!/bin/bash
# Uploads the experiment records in results/ to a private Hugging Face dataset: the rendering
# check, the cost-predictor test and, once they exist, the adapter pilot's runs.
#
#   bash costPredictor/upload_to_hugging_face.sh        (from the top folder of the repo)
#
# It copies files and runs nothing else, so it can run on the laptop. It needs git and
# git-lfs, and a Hugging Face token with write access in the file $HF_TOKEN_PATH. The token
# is only ever read from that file: it is not put on a command line, in a URL or in git's
# settings.
set -euo pipefail

DATASET=${DATASET:-Sachioster/evowm-experiment-records}
HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/credentials/hf_token}
RESULTS=${RESULTS:-results}
STAGE=${STAGE:-$RESULTS/huggingface}

[ -s "$HF_TOKEN_PATH" ] || { echo "No token in $HF_TOKEN_PATH" >&2; exit 1; }
[ -d "$RESULTS/renderingCheck" ] && [ -d "$RESULTS/costPredictor/recordings" ] || { echo "Run from the top folder of the repo" >&2; exit 1; }
export HF_TOKEN_PATH

# Create the dataset, private. If it already exists Hugging Face answers 409, which is fine.
status=$(curl --silent --output /dev/null --write-out '%{http_code}' \
    --header @<(printf 'Authorization: Bearer %s' "$(tr -d '[:space:]' < "$HF_TOKEN_PATH")") \
    --header 'Content-Type: application/json' \
    --data "{\"type\": \"dataset\", \"organization\": \"${DATASET%%/*}\", \"name\": \"${DATASET##*/}\", \"private\": true}" \
    https://huggingface.co/api/repos/create)
case "$status" in
    200) echo "created $DATASET (private)" ;;
    409) echo "$DATASET already exists" ;;
    *) echo "Could not create $DATASET: HTTP $status" >&2; exit 1 ;;
esac

# git asks this helper for the password each time, so the token is never stored.
helper='!f() { echo username=hf_user; echo "password=$(tr -d "[:space:]" < "$HF_TOKEN_PATH")"; }; f'
hf_git() { git -c credential.helper= -c credential.helper="$helper" "$@"; }

rm -rf "$STAGE"
hf_git clone --quiet "https://huggingface.co/datasets/$DATASET" "$STAGE"
git -C "$STAGE" lfs install --local >/dev/null
git -C "$STAGE" lfs track '*.tar' '*.npz' '*.npy' >/dev/null

cp costPredictor/dataset_card.md "$STAGE/README.md"
mkdir -p "$STAGE/rendering_check" "$STAGE/cost_predictor/recordings" "$STAGE/cost_predictor/results"
cp -r "$RESULTS"/renderingCheck/. "$STAGE/rendering_check/"
cp "$RESULTS/costPredictor/summary.txt" "$STAGE/cost_predictor/"
for folder in "$RESULTS"/costPredictor/recordings/chunk0/*/; do
    name=$(basename "$folder")
    # One archive per checkpoint, holding its four folders of 50 episodes. Sorted and with
    # fixed dates, so packing the same files again gives the same archive.
    tar --sort=name --mtime='2026-11-21 00:00Z' --owner=0 --group=0 --numeric-owner \
        -cf "$STAGE/cost_predictor/recordings/$name.tar" \
        -C "$RESULTS/costPredictor/recordings" chunk0/"$name" chunk1/"$name" chunk2/"$name" chunk3/"$name"
    mkdir -p "$STAGE/cost_predictor/results/$name"
    cp "$RESULTS/costPredictor/$name"/results.json "$RESULTS/costPredictor/$name"/latent_*.npz "$STAGE/cost_predictor/results/$name/"
done
# How the predictors do with fewer episodes, and the adapter pilot: copied as they are.
for pair in costPredictor/efficiency:cost_predictor/efficiency adapterPilot:adapter_pilot; do
    if [ -d "$RESULTS/${pair%%:*}" ]; then
        mkdir -p "$STAGE/${pair##*:}"
        cp -r "$RESULTS/${pair%%:*}/." "$STAGE/${pair##*:}/"
    fi
done

git -C "$STAGE" add --all
if git -C "$STAGE" diff --cached --quiet; then
    echo "nothing new to upload"
else
    git -C "$STAGE" -c user.name="${DATASET%%/*}" -c user.email="${DATASET%%/*}@users.noreply.huggingface.co" \
        commit --quiet -m "Experiment records (code commit $(git rev-parse --short HEAD))"
    hf_git -C "$STAGE" push --quiet origin HEAD:main
    echo "uploaded to https://huggingface.co/datasets/$DATASET"
fi
du -sh --exclude=.git "$STAGE"
