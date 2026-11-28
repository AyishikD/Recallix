"""
Recallic side of the comparison (PROTOCOL.md "Systems", amendment 1). Runs in the Recallic environment, no LLM.

    python run_recallix.py --data data.json --out OUT/recallix.json [--deep-out OUT/recallix_deep.json] [--convs conv-47]

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
    ap.add_argument("--deep-out", default=None,
                    help="also recall every question with deep recall (deep=True) from the same stored memories, written here")
    ap.add_argument("--convs", default=None, help="comma-separated conversations to run (default: all in --data)")
    ap.add_argument("--context-out", default=None,
                    help="also write the default recall's results with each memory followed by the message stored "
                         "just before it (same retrieval; only what the answering model reads changes)")
    args = ap.parse_args()
    import run_eval
    run_eval.Env("st:BAAI/bge-small-en-v1.5")   # fresh temporary storage, in-process engine
    from backend.api.deps import get_memory_manager, reset
    reset()
    with contextlib.redirect_stdout(io.StringIO()):
        mm = get_memory_manager()
    result, result_deep, result_ctx = {}, {}, {}
    wanted = set(args.convs.split(",")) if args.convs else None
    for conv in json.load(open(args.data)):
        if wanted and conv["conv"] not in wanted:
            continue
        # each variant gets its own freshly stored copy: a recall reinforces what it returns, so a
        # second pass over the same memories would not start from the same state
        for deep, res in ((None, result), (True, result_deep)):
            if deep and not args.deep_out:
                continue
            user = conv["conv"] + ("#deep" if deep else "")
            store_ms, dia_of = [], {}
            for t in conv["turns"]:
                t0 = time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    r = mm.store(user, t["text"], client_metadata={"speaker": t["speaker"]}, timestamp=t["ts"])
                store_ms.append(round((time.perf_counter() - t0) * 1000, 1))
                dia_of.setdefault(r["memory_id"], []).append(t["dia_id"])
            search, search_ctx = {}, {}
            for qi, qa in enumerate(conv["qa"]):
                t0 = time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    out = mm.retrieve(user, qa["question"], limit=10, deep=deep)
                texts = [m.get("content_resolved") or m.get("content", "") for m in out["memories"]]
                search[str(qi)] = {"ms": round((time.perf_counter() - t0) * 1000, 1), "status": out["status"],
                                   "memories": texts,
                                   "dia_ids": [d for m in out["memories"] for d in dia_of.get(m["id"], [])]}
                if args.context_out and not deep:
                    before = mm.episodic.previous_of(user, [m["id"] for m in out["memories"]], None, 3600.0)
                    search_ctx[str(qi)] = {**search[str(qi)], "memories": [
                        f"{t} [the message before it: {before[m['id']][:300]}]" if m["id"] in before else t
                        for t, m in zip(texts, out["memories"])]}
            res[conv["conv"]] = {"stored": len(conv["turns"]), "store_ms": store_ms, "search": search, "deep": bool(deep)}
            if search_ctx:
                result_ctx[conv["conv"]] = {**res[conv["conv"]], "search": search_ctx, "context": True}
            print(f"{conv['conv']}{' (deep)' if deep else ''}: stored {len(conv['turns'])}, recalled {len(search)}", flush=True)
    json.dump(result, open(args.out, "w"))
    if args.deep_out:
        json.dump(result_deep, open(args.deep_out, "w"))
    if args.context_out:
        json.dump(result_ctx, open(args.context_out, "w"))


if __name__ == "__main__":
    main()
