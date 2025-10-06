#!/usr/bin/env bash
set -euo pipefail
kubectl apply -f k8s/10-data/job-build-index.yaml
kubectl -n atharva-ml wait --for=condition=complete job/atharva-build-index --timeout=300s
kubectl -n atharva-ml logs job/atharva-build-index
