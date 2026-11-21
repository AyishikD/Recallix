"""
Multi-agent handover through Recallix.

Agent A (an LLM) invents a fact and stores it as a *shared* memory; agent B (an
LLM) recalls it through Recallix and answers from it. Distractors with the same
shape are stored first. Recallix itself makes no LLM calls; only the two agents
do, via the provider in config.yaml `models.provider`.

Environment:
  RECALLIX_API_URL   default http://127.0.0.1:8000
  RECALLIX_API_KEY   a service key (python -m backend.tools.keys create --user demo --role service)
  <PROVIDER>_API_KEY for the agents' LLM provider (e.g. NVIDIA_API_KEY)

Run:  python examples/multi_agent_handover.py [--trials 5]
"""
import argparse
import os
import sys
import time
import uuid
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.brain.models.model_router import ModelRouter  # noqa: E402

API = os.environ.get("RECALLIX_API_URL", "http://127.0.0.1:8000")
KEY = os.environ.get("RECALLIX_API_KEY")


def call(path, payload):
    r = requests.post(f"{API}{path}", json=payload, headers={"X-API-Key": KEY}, timeout=60)
    r.raise_for_status()
    return r.json()["data"]


def trial(router, user):
    secret = router.route("compression", "Invent a project code name of exactly three random English words. "
                                         "Output only the three words.").strip().strip("'\"")
    for d in ["ALPHA-9", "GHOST-PROTO", "ZEPHYR-X", "OMEGA-VOID"]:
        call("/agents/store", {"user_id": user, "agent_id": "system", "memory_type": "shared",
                               "content": f"An old, retired project code was {d}."})
    call("/agents/store", {"user_id": user, "agent_id": "agent_a", "memory_type": "shared",
                           "content": f"The current secret project code is {secret}."})
    t0 = time.time()
    out = call("/agents/recall", {"user_id": user, "agent_id": "agent_b", "query": "What is the current secret project code?"})
    recall_ms = (time.time() - t0) * 1000
    context = out["memories"][0]["content"] if out["memories"] else ""
    answer = router.route("reasoning", f"Using only this memory: '{context}', what is the current secret project "
                                       f"code? Output only the code.").strip()
    return secret, answer, secret.lower() in answer.lower(), recall_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5)
    args = ap.parse_args()
    if not KEY:
        raise SystemExit("Set RECALLIX_API_KEY to a service key.")
    router = ModelRouter()
    ok = 0
    for i in range(args.trials):
        user = f"handover-{uuid.uuid4().hex[:8]}"
        secret, answer, success, ms = trial(router, user)
        ok += success
        print(f"trial {i + 1}: secret={secret!r} answer={answer[-80:]!r} {'OK' if success else 'MISS'} (recall {ms:.0f} ms)")
    print(f"\nexact handover: {ok}/{args.trials} trials")


if __name__ == "__main__":
    main()
