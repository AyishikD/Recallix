# LEGACY SCRIPT: written for the pre-2026-11 API and engine protocol; kept for reference only.
# Maintained equivalents: eval/run_eval.py (retrieval, gauntlet, retention), backend/infra_cpp/benchmark_hnsw,
# examples/multi_agent_handover.py. Keys come from RECALLIX_API_KEY / RECALLIX_INTERNAL_KEY.
import os
import requests

API_URL = "http://127.0.0.1:8000/memory"
API_KEY = os.environ["RECALLIX_API_KEY"]
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

payload = {
    "user_id": "test_user_multi",
    "agent_id": "agent_b",
    "query": "What is the secret project code?",
    "limit": 5
}
res = requests.post(f"{API_URL}/recall", json=payload, headers=HEADERS)
print(res.json())
