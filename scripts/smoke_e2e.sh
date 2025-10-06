#!/usr/bin/env bash
set -euo pipefail

CHAT_HOST="${CHAT_HOST:-127.0.0.1}"
CHAT_PORT="${CHAT_PORT:-30300}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-30200}"

chat_url="http://${CHAT_HOST}:${CHAT_PORT}"
vllm_url="http://${VLLM_HOST}:${VLLM_PORT}"

echo "=== End-to-End Test: Chat API -> Retriever -> vLLM ==="
echo "Chat API: ${chat_url}    vLLM: ${vllm_url}"
echo

# 1) Health checks
echo "[1/4] Health: Chat API"
curl -sf "${chat_url}/health" | jq . || { echo "Chat API health failed"; exit 1; }
echo

echo "[2/4] Health: vLLM (OpenAI models)"
curl -sf "${vllm_url}/v1/models" | jq '.data[0] // {}' || { echo "vLLM models failed"; exit 1; }
echo

# 2) Helper to run a chat and extract key fields
ask() {
  local q="$1"; local k="${2:-4}"; local max_tokens="${3:-256}"; local temp="${4:-0.3}"
  echo "Q: $q"
  resp="$(curl -s -X POST "${chat_url}/chat" \
    -H 'content-type: application/json' \
    -d "{\"question\":\"${q}\",\"k\":${k},\"max_tokens\":${max_tokens},\"temperature\":${temp}}")"

  # Pretty summary
  echo "$resp" | jq -r '
    . as $r |
    "─ Answer ─\n" +
    ($r.answer // "<no answer>") + "\n\n" +
    "─ Citations ─\n" + ((($r.citations // [])|join("\n")) // "<none>") + "\n\n" +
    "─ Stats ─\n" +
    ("latency_seconds: " + (($r.latency_seconds // 0)|tostring)) + "\n" +
    ("prompt_tokens: "   + (($r.usage.prompt_tokens // 0)|tostring)) + "\n" +
    ("completion_tokens:" + (($r.usage.completion_tokens // 0)|tostring)) + "\n"
  '
  echo "-------------------------------------------"
}

echo "[3/4] Functional E2E prompts"
ask "Do you accept UPI payments ?"
ask "How long does scaling take and what aftercare is needed?"
ask "Are you open on Sundays ?"
ask "Which city is Atharva Dental Clinic?"
ask "What is the typical cost range for a root canal and crown?" 4 256 0.2
ask "My face is badly swollen and I have a high fever after an extraction. What should I do?" 4 192 0.1

# 3) Optional: short latency/tokens smoke loop
echo
echo "[4/4] Throughput smoke (3 quick runs)"
for i in 1 2 3; do
  curl -s -X POST "${chat_url}/chat" \
    -H 'content-type: application/json' \
    -d '{"question":"Is next-day pain after RCT normal? Suggest aftercare.","k":3,"max_tokens":192}' \
    | jq -r '"run=\($i) lat=\(.latency_seconds)s tokens=(p:\(.usage.prompt_tokens // 0), c:\(.usage.completion_tokens // 0))"' --arg i "$i"
done

echo
echo "✅ E2E complete."
