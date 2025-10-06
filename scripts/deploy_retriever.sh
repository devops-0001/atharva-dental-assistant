#!/usr/bin/env bash
set -euo pipefail
kubectl apply -f k8s/40-serve/deploy-retriever.yaml
kubectl apply -f k8s/40-serve/svc-retriever.yaml
kubectl -n atharva-ml rollout status deploy/atharva-retriever --timeout=180s
kubectl -n atharva-ml get svc/atharva-retriever
