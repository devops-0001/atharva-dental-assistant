#!/usr/bin/env bash
set -euo pipefail

# Apply ServiceMonitors + Alerts + Dashboard
kubectl apply -f k8s/50-observability/sm-vllm.yaml
kubectl apply -f k8s/50-observability/sm-chat-api.yaml
kubectl apply -f k8s/50-observability/sm-retriever.yaml
kubectl apply -f k8s/50-observability/cm-grafana-dashboard.yaml
kubectl apply -f k8s/50-observability/pr-alerts.yaml || true

echo "Waiting a bit for Prometheus to pick up targets..."
sleep 10

echo "List ServiceMonitors:"
kubectl -n monitoring get servicemonitors
echo "Prometheus targets (check Up status via UI)."

echo -e "Access Prometheus and Grafana using\n\
  * Prometheus : http://localhost:30500/\n\
  * Grafana : http://localhost:30400/\n\
    * user: admin\n\
    * pass: prom-operator"
