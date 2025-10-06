#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <RUN_ID> <TAG>   e.g. $0 20251001-113045 v1"
  exit 1
fi
RUN_ID="$1"
USER="$2"
TAG="$3"

IMG="${USER}/smollm2-135m-merged:${TAG}"

# Safety checks
[ -d "artifacts/train/${RUN_ID}/merged-model" ] || { echo "Merged model folder not found for RUN_ID=${RUN_ID}"; exit 1; }

# Create a temp Dockerfile with RUN_ID patched
TMP_DF=$(mktemp)
sed "s|REPLACE_RUN_ID|${RUN_ID}|g" training/Dockerfile.model-asset > "$TMP_DF"

echo "==> Building model asset image: ${IMG}"
docker build -f "$TMP_DF" -t "${IMG}" .

#echo "==> Pushing to local registry ${IMG}"
docker push "${IMG}"

echo "Done. Image: ${IMG}"
