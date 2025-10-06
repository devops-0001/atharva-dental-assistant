#!/usr/bin/env bash
set -euo pipefail
kubectl apply -f k8s/40-serve/rawdeployment-vllm.yaml
kubectl apply -f k8s/40-serve/svc-vllm.yaml
echo "Waiting for vLLM Service endpoints..."
kubectl -n atharva-ml rollout status deploy/atharva-vllm-predictor --timeout=300s || true
kubectl -n atharva-ml get pods -l app=vllm -o wide
kubectl -n atharva-ml get svc atharva-vllm
