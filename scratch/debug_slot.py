# LEGACY SCRIPT: written for the pre-2026-11 API and engine protocol; kept for reference only.
# Maintained equivalents: eval/run_eval.py (retrieval, gauntlet, retention), backend/infra_cpp/benchmark_hnsw,
# examples/multi_agent_handover.py. Keys come from RECALLIX_API_KEY / RECALLIX_INTERNAL_KEY.
import sys
from backend.brain.memory.manager import MemoryManager

mm = MemoryManager()
res1 = mm.store("test_user", "The secret project code is OMEGA-VOID", skip_llm=True)
print("STORE 1:", res1)
res2 = mm.store("test_user", "The secret project code is SolarFlameOcean", skip_llm=True)
print("STORE 2:", res2)

recall = mm.retrieve("test_user", "What is the secret project code?")
print("RECALL:", recall)
