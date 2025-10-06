#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <RUN_ID>"
  exit 1
fi
RUN_ID="$1"

# Patch the manifest with the RUN_ID and apply
tmp=$(mktemp)
sed "s/REPLACE_WITH_RUN_ID/${RUN_ID}/g" k8s/20-train/job-merge-model.yaml > "$tmp"
kubectl apply -f "$tmp"
kubectl -n atharva-ml wait --for=condition=complete job/atharva-merge-model --timeout=60m
kubectl -n atharva-ml logs job/atharva-merge-model

echo "Merged model at artifacts/train/${RUN_ID}/merged-model"
echo "Tarball at artifacts/train/${RUN_ID}/model.tgz"
