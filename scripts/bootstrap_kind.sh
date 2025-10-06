#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME="llmops-kind"
KIND_CONFIG="setup/kind-config.yaml"
MON_NS="monitoring"

echo "==> Preflight checks"
command -v kind >/dev/null 2>&1 || { echo "Please install kind"; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo "Please install kubectl"; exit 1; }
command -v helm >/dev/null 2>&1 || { echo "Please install helm"; exit 1; }

mkdir -p project

echo "==> Creating KIND cluster (${CLUSTER_NAME}) with ImageVolume feature-gate enabled"
kind create cluster --name "${CLUSTER_NAME}" --config "${KIND_CONFIG}"

echo "==> Verifying Kubernetes server version"
SERVER_MINOR=$(kubectl version -o json | jq -r '.serverVersion.minor' | sed 's/[^0-9].*//')
SERVER_MAJOR=$(kubectl version -o json | jq -r '.serverVersion.major')
echo "Server version: ${SERVER_MAJOR}.${SERVER_MINOR}"
# ImageVolumes need >= 1.31; KServe Quickstart needs >= 1.29
if [ "${SERVER_MAJOR}" -lt 1 ] || [ "${SERVER_MINOR}" -lt 31 ]; then
  echo "ERROR: Kubernetes >=1.31 required for ImageVolumes. Current: ${SERVER_MAJOR}.${SERVER_MINOR}"
  exit 1
fi

echo "==> Creating namespaces"
kubectl apply -f - <<EOF
apiVersion: v1
kind: Namespace
metadata: { name: atharva-ml }
---
apiVersion: v1
kind: Namespace
metadata: { name: atharva-app }
---
apiVersion: v1
kind: Namespace
metadata: { name: ${MON_NS} }
EOF

echo "==> (Optional) Check container runtime inside nodes for ImageVolume support"
CONTROL_NODE=$(kubectl get nodes -o name | head -n1 | sed 's|node/||')
docker exec "${CLUSTER_NAME}-control-plane" containerd --version || true


echo "==> All set!
Namespaces:
  - atharva-ml      (ML training & model artifacts)
  - atharva-app     (chat API / frontend)
  - monitoring      (Prometheus + Grafana)

Next:
  • Lab 1 will generate synthetic data and build the FAISS index.
"
