#!/usr/bin/env bash
set -euo pipefail
kubectl apply -f k8s/20-train/job-train-lora.yaml
kubectl -n atharva-ml wait --for=condition=complete job/atharva-train-lora --timeout=12h
kubectl -n atharva-ml logs job/atharva-train-lora
echo "Artifacts under artifacts/train/<run-id>/"
