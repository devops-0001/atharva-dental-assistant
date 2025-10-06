#!/usr/bin/env bash
set -euo pipefail
kubectl apply -f k8s/40-serve/cm-chat-api.yaml
kubectl apply -f k8s/40-serve/deploy-chat-api.yaml
kubectl apply -f k8s/40-serve/svc-chat-api.yaml
kubectl -n atharva-app rollout status deploy/atharva-chat-api --timeout=180s
kubectl -n atharva-app get svc atharva-chat-api
