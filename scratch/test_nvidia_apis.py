# LEGACY SCRIPT: written for the pre-2026-11 API and engine protocol; kept for reference only.
# Maintained equivalents: eval/run_eval.py (retrieval, gauntlet, retention), backend/infra_cpp/benchmark_hnsw,
# examples/multi_agent_handover.py. Keys come from RECALLIX_API_KEY / RECALLIX_INTERNAL_KEY.
from backend.brain.models.model_router import ModelRouter
import os

def test_nvidia_apis():
    print("🚀 Testing NVIDIA APIs...")
    router = ModelRouter()
    
    # 1. Test Nemotron VL (Fast Model)
    print(f"\n1️⃣ Testing Nemotron VL (Fast Model: {router.fast_model})...")
    vl_resp = router.route("compression", "Describe the concept of 'Universal Memory Substrate' in one sentence.")
    if "Error" in vl_resp:
        print(f"   ❌ Nemotron VL failed: {vl_resp}")
    else:
        print(f"   ✅ Nemotron VL Response: \"{vl_resp.strip()}\"")
    
    # 2. Test Nemotron Nano (Smart Model with Thinking)
    print(f"\n2️⃣ Testing Nemotron Nano (Smart/Thinking Model: {router.smart_model})...")
    nano_resp = router.route("reasoning", "Explain why persistent memory is essential for multi-agent systems.")
    if "Error" in nano_resp:
        print(f"   ❌ Nemotron Nano failed: {nano_resp}")
    else:
        print(f"   ✅ Nemotron Nano Response:")
        print(f"      {nano_resp.strip()}")

if __name__ == "__main__":
    test_nvidia_apis()
