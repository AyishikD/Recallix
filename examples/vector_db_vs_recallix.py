"""
Plain vector search vs Recallix, on the same messages and the same embedding model.

    python examples/vector_db_vs_recallix.py [--json out.json]

One person texts their assistant over a year. A "vector database" here is exactly what most RAG
setups do: embed every message with the same model Recallix uses (bge-small) and return the
closest ones by cosine similarity. Recallix stores the same messages through its normal pipeline
(no LLM anywhere) and answers the same questions. Runs locally on a temporary store; nothing
leaves the machine. Illustrative examples, not a benchmark: the benchmarks are in eval/results.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MESSAGES = [  # (when it was said, what was said), texting style
    ("2026-01-10 09:12", "i live in delhi with my parents, near lajpat nagar"),
    ("2026-02-02 21:40", "my brother arjun is visiting next week"),
    ("2026-03-15 18:05", "work is crazy busy these days"),
    ("2026-06-03 08:30", "update: moved to pune last month for the new job"),
    ("2026-06-20 13:15", "allergic to penicillin, found out the hard way lol"),
    ("2026-11-21 19:45", "went trekking at sinhagad with friends today"),
]
QUESTIONS = [
    ("A fact that changed", "Where do I live now?"),
    ("Something never said", "What's my sister's name?"),
    ("A plain fact", "Am I allergic to any medicine?"),
]


def ts(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M").timestamp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="also write every result here")
    args = ap.parse_args()
    import numpy as np
    from backend.tools.sandbox import Sandbox

    results = {"messages": [{"said_at": w, "text": t} for w, t in MESSAGES], "questions": []}
    with Sandbox() as box:
        mm, user = box.mm, "demo"
        with box.silenced():
            for when, text in MESSAGES:
                mm.store(user, text, timestamp=ts(when))
            vecs = np.array([mm.semantic.embed(t) for _, t in MESSAGES], dtype=np.float32)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            mm.retrieve(user, "warm up", limit=3)  # first call loads the models; not timed below
        for label, q in QUESTIONS:
            qv = np.asarray(mm.semantic.embed(q), dtype=np.float32)
            qv /= np.linalg.norm(qv)
            sims = vecs @ qv
            top = [{"text": MESSAGES[i][1], "said_at": MESSAGES[i][0], "similarity": round(float(sims[i]), 3)}
                   for i in np.argsort(-sims)[:3]]
            t0 = time.perf_counter()
            with box.silenced():
                out = mm.retrieve(user, q, limit=3)
            ms = (time.perf_counter() - t0) * 1000
            rx = {"status": out["status"], "answer": (out.get("answer") or {}).get("text"),
                  "memories": [m.get("content_resolved") or m.get("content") for m in out["memories"]],
                  "reason": (out.get("receipt") or {}).get("reason"), "ms": round(ms, 1)}
            results["questions"].append({"label": label, "question": q, "vector_db_top3": top, "recallix": rx})

    for r in results["questions"]:
        print(f"\n=== {r['label']}: \"{r['question']}\"")
        print("  Vector database (top 3 by similarity):")
        for x in r["vector_db_top3"]:
            print(f"    {x['similarity']:.3f}  {x['said_at'][:10]}  {x['text']}")
        rx = r["recallix"]
        print(f"  Recallix: {rx['status']}  ({rx['ms']:.0f} ms)")
        if rx["answer"]:
            print(f"    answer: {rx['answer']}")
        for m in rx["memories"][:2]:
            print(f"    memory: {m}")
        print(f"    why: {rx['reason']}")
    if args.json:
        json.dump(results, open(args.json, "w"), indent=2)
        print("\nwrote", args.json)


if __name__ == "__main__":
    main()
