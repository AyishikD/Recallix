"""
Recallix side of the comparison (PROTOCOL.md "Systems", amendment 1). Runs in the Recallix environment, no LLM.

    python run_recallix.py --data data.json --out OUT/recallix.json

Uses the shipped config (bge-small embeddings, answer check, date resolution) on fresh storage, one user per conversation.
"""
import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "eval")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import run_eval
    run_eval.Env("st:BAAI/bge-small-en-v1.5")   # fresh temporary storage, in-process engine
    from backend.api.deps import get_memory_manager, reset
    reset()
    with contextlib.redirect_stdout(io.StringIO()):
        mm = get_memory_manager()
    result = {}
    for conv in json.load(open(args.data)):
        store_ms, dia_of = [], {}
        for t in conv["turns"]:
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                r = mm.store(conv["conv"], t["text"], client_metadata={"speaker": t["speaker"]}, timestamp=t["ts"])
            store_ms.append(round((time.perf_counter() - t0) * 1000, 1))
            dia_of.setdefault(r["memory_id"], []).append(t["dia_id"])
        search = {}
        for qi, qa in enumerate(conv["qa"]):
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                out = mm.retrieve(conv["conv"], qa["question"], limit=10)
            search[str(qi)] = {"ms": round((time.perf_counter() - t0) * 1000, 1), "status": out["status"],
                               "memories": [m.get("content_resolved") or m.get("content", "") for m in out["memories"]],
                               "dia_ids": [d for m in out["memories"] for d in dia_of.get(m["id"], [])]}
        result[conv["conv"]] = {"stored": len(conv["turns"]), "store_ms": store_ms, "search": search}
        print(f"{conv['conv']}: stored {len(conv['turns'])}, recalled {len(search)}", flush=True)
    json.dump(result, open(args.out, "w"))


if __name__ == "__main__":
    main()
