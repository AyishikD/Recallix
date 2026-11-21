# LEGACY SCRIPT: written for the pre-2026-11 API and engine protocol; kept for reference only.
# Maintained equivalents: eval/run_eval.py (retrieval, gauntlet, retention), backend/infra_cpp/benchmark_hnsw,
# examples/multi_agent_handover.py. Keys come from RECALLIX_API_KEY / RECALLIX_INTERNAL_KEY.
import os
import requests
import numpy as np

INFRA_URL = "http://localhost:8080"
VECTOR_DIM = 128
headers = {"X-Internal-Key": os.environ["RECALLIX_INTERNAL_KEY"]}

scale = 1000
raw_vectors = np.random.randn(scale, VECTOR_DIM).astype('float32')
raw_vectors /= np.linalg.norm(raw_vectors, axis=1)[:, np.newaxis]
python_ids = [f"vec_{j}" for j in range(scale)]

requests.post(f"{INFRA_URL}/clear", headers=headers)

chunk = []
for j in range(scale):
    chunk.append({"id": python_ids[j], "vector": raw_vectors[j].tolist()})
res = requests.post(f"{INFRA_URL}/bulk_add", json={"vectors": chunk}, headers=headers)
print("Bulk add response:", res.status_code, res.text)
