#!/usr/bin/env bash
set -euo pipefail
kubectl apply -f k8s/10-data/job-generate-data.yaml
kubectl -n atharva-ml wait --for=condition=complete job/atharva-generate-data --timeout=300s
kubectl -n atharva-ml logs job/atharva-generate-data
