"""
Mem0 side of the comparison (PROTOCOL.md "Systems"). Runs in its own environment with mem0ai 2.2.1.

    python run_mem0.py --data data.json --conv conv-44 --out OUT --ledger ledger.json [--max-turns 40] [--phase store|search|both]

- Stores every message of one conversation, one `add()` each, with inference on, as Mem0 ships.
- Then answers each question with a top-10 search.
- Resumes from its checkpoint, and stops cleanly when the budget guard trips.
- Every OpenAI call is priced by budget.py.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MEM0_TELEMETRY", "False")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import budget  # noqa: E402


def memory_for(conv, out_dir):
    from mem0 import Memory
    base = Path(out_dir) / f"mem0_{conv}"
    base.mkdir(parents=True, exist_ok=True)
    return Memory.from_config({
        "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini"}},
        "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
        "vector_store": {"provider": "qdrant", "config": {"collection_name": conv, "path": str(base / "qdrant"),
                                                          "embedding_model_dims": 1536}},
        "history_db_path": str(base / "history.db"),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--conv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--max-turns", type=int, default=None)
    ap.add_argument("--phase", default="both", choices=["store", "search", "both"])
    args = ap.parse_args()

    ledger = budget.install(args.ledger)
    conv = next(c for c in json.load(open(args.data)) if c["conv"] == args.conv)
    turns = conv["turns"][: args.max_turns]
    ckpt_path = Path(args.out) / f"mem0_{args.conv}.json"
    ckpt = json.load(open(ckpt_path)) if ckpt_path.exists() else {"conv": args.conv, "stored": 0, "store_ms": [],
                                                                   "store_errors": 0, "search": {}}
    m = memory_for(args.conv, args.out)

    def save():
        tmp = ckpt_path.with_suffix(".tmp")
        json.dump(ckpt, open(tmp, "w")); os.replace(tmp, ckpt_path)

    if args.phase in ("store", "both"):
        budget.PURPOSE["value"] = "mem0-store"
        for i in range(ckpt["stored"], len(turns)):
            t0 = time.perf_counter()
            try:
                m.add([{"role": "user", "content": turns[i]["text"]}], user_id=args.conv)
            except budget.BudgetExceeded as e:
                print(f"STOPPED by budget at message {i}: {e}"); save(); return 2
            except Exception as e:  # a failed add is counted, not retried (Mem0 as shipped)
                if isinstance(getattr(e, "__cause__", None), budget.BudgetExceeded) or "budget stop" in str(e):
                    print(f"STOPPED by budget at message {i}: {e}"); save(); return 2
                ckpt["store_errors"] += 1
                print(f"add error at {i}: {type(e).__name__}: {str(e)[:120]}")
            ckpt["store_ms"].append(round((time.perf_counter() - t0) * 1000, 1))
            ckpt["stored"] = i + 1
            if (i + 1) % 20 == 0:
                save(); print(f"{args.conv}: stored {i + 1}/{len(turns)}  spent ${ledger.spent():.3f}", flush=True)
        save()

    if args.phase in ("search", "both") and args.max_turns is None:
        budget.PURPOSE["value"] = "mem0-search"
        for qi, qa in enumerate(conv["qa"]):
            if str(qi) in ckpt["search"]:
                continue
            t0 = time.perf_counter()
            try:
                res = m.search(qa["question"], filters={"user_id": args.conv}, top_k=10)
            except budget.BudgetExceeded as e:
                print(f"STOPPED by budget at question {qi}: {e}"); save(); return 2
            hits = res.get("results", res) if isinstance(res, dict) else res
            ckpt["search"][str(qi)] = {"ms": round((time.perf_counter() - t0) * 1000, 1),
                                       "memories": [h.get("memory", "") for h in hits][:10]}
            if (qi + 1) % 50 == 0:
                save()
        save()
    print(f"{args.conv}: done. stored {ckpt['stored']}, searched {len(ckpt['search'])}, spent ${ledger.spent():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
