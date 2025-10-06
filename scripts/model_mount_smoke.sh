#!/usr/bin/env bash
set -euo pipefail
kubectl apply -f k8s/30-model/model-mount-check.yaml
echo "Waiting for Pod to complete..."
kubectl -n atharva-ml wait --for=condition=Ready pod/model-mount-check --timeout=120s || true
kubectl -n atharva-ml logs pod/model-mount-check
