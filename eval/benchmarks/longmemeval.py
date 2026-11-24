"""
LongMemEval retrieval + abstention benchmark (no LLM needed).

LongMemEval (Wu et al., 2024; huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
has 500 questions, each with its own chat history (~500 user/assistant turns in the
"_s" variant). Turns that contain the answer are labelled (has_answer), so retrieval
is scored directly. 30 questions (ids ending in "_abs") are unanswerable: the right
behaviour is to decline.

This benchmark was not used for any fitting, tuning or design decision; it is run
once with the frozen configuration.

Systems (identical text per turn: "[date] User|Assistant: content"):
  recallix          full store + recall pipeline as configured (answer check, abstention)
  recallix_ranked   Recallix's ranking before truth arbitration / abstention
  dense             cosine top-k over the same embeddings
  bm25              Okapi BM25

Usage:
  python -c "from huggingface_hub import hf_hub_download as d; print(d('xiaowu0162/longmemeval-cleaned', \\
      'longmemeval_s_cleaned.json', repo_type='dataset', local_dir='/tmp/lme'))"
  python eval/benchmarks/longmemeval.py --data /tmp/lme/longmemeval_s_cleaned.json \\
      --embedder st:sentence-transformers/all-MiniLM-L6-v2 --engine-url URL --json eval/results/longmemeval_s.json
"""
import argparse
import contextlib
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # eval/
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_eval  # noqa: E402
from locomo import BM25, score, _auc  # noqa: E402


def session_time(date_text):
    """'2023/05/20 (Sat) 02:21' -> a UTC timestamp (None if it does not parse)."""
    import re
    from datetime import datetime, timezone
    m = re.match(r"\s*(\d{4})/(\d{2})/(\d{2})(?:\s*\(\w+\))?\s*(\d{1,2}):(\d{2})", date_text or "")
    if not m:
        return None
    y, mo, d, h, mi = map(int, m.groups())
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc).timestamp()


def turns_of(item, with_time=False):
    """[(turn_key, text, speaker, has_answer)] over the question's haystack, in order
    (plus the session's timestamp when with_time, so stored memories carry their real date)."""
    out = []
    for s_idx, (sid, date, session) in enumerate(zip(item["haystack_session_ids"], item["haystack_dates"],
                                                     item["haystack_sessions"])):
        ts = session_time(date)
        for t_idx, turn in enumerate(session):
            speaker = "User" if turn["role"] == "user" else "Assistant"
            row = (f"{sid}#{t_idx}", f"[{date}] {speaker}: {turn['content']}", speaker, bool(turn.get("has_answer")))
            out.append(row + (ts,) if with_time else row)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--embedder", default="hash")
    ap.add_argument("--engine-url", default=None)
    ap.add_argument("--k", default="5,10")
    ap.add_argument("--limit", type=int, default=None, help="first N questions (quick runs)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    ks = [int(x) for x in args.k.split(",")]
    kmax = max(ks)

    env = run_eval.Env(args.embedder, args.engine_url)
    from backend.api.deps import get_memory_manager, reset
    reset()
    with contextlib.redirect_stdout(io.StringIO()):
        mm = get_memory_manager()

    data = json.load(open(args.data))[: args.limit]
    systems = ["recallix", "recallix_ranked", "dense", "bm25"]
    per_type = {s: defaultdict(lambda: defaultdict(list)) for s in systems}
    abstain = {"answerable": [], "unanswerable": []}
    decision = []  # (top_relevance, best answer margin or None, unanswerable)
    timing = defaultdict(float)
    n_turns = 0

    for qi, item in enumerate(data):
        user = f"lme_{item['question_id']}"
        timed = turns_of(item, with_time=True)
        turns = [t[:4] for t in timed]
        n_turns += len(turns)
        mem_to_turn = defaultdict(list)
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            for key, text, speaker, _, ts in timed:
                res = mm.store(user, text, client_metadata={"speaker": speaker}, timestamp=ts)
                mem_to_turn[res["memory_id"]].append(key)
        timing["recallix_ingest_s"] += time.time() - t0

        keys = [k for k, *_ in turns]
        texts = [t for _, t, _, _ in turns]
        evidence = {k for k, _, _, has in turns if has}
        unanswerable = item["question_id"].endswith("_abs")
        q = item["question"]

        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            out = mm.retrieve(user, q, limit=kmax, return_candidates=True)
        timing["recallix_query_s"] += time.time() - t0
        served = [k for m in out["memories"] for k in mem_to_turn[m["id"]]]
        ranked = [k for c in sorted(out["_candidates"], key=lambda c: -c["relevance"]) for k in mem_to_turn[c["id"]]]
        declined = out["status"] in ("ABSTAINED", "NO_MATCH")
        chk = out["debug"].get("answer_check")
        decision.append((out["debug"].get("top_relevance") or 0.0, chk["best_margin"] if chk else None, unanswerable))

        t0 = time.time()
        vecs = np.vstack([mm.semantic.embed(t) for t in texts])
        dense = [keys[i] for i in np.argsort(-(vecs @ mm.semantic.embed(q)))[:kmax]]
        timing["dense_s"] += time.time() - t0
        t0 = time.time()
        lexical = [keys[i] for i in BM25(texts).top(q, kmax)]
        timing["bm25_s"] += time.time() - t0

        (abstain["unanswerable"] if unanswerable else abstain["answerable"]).append(declined)
        if not unanswerable and evidence:
            for name, ids in (("recallix", served), ("recallix_ranked", ranked), ("dense", dense), ("bm25", lexical)):
                for metric, v in score(ids, evidence, ks).items():
                    per_type[name][item["question_type"]][metric].append(v)
        if (qi + 1) % 25 == 0:
            print(f"  {qi + 1}/{len(data)} questions", flush=True)

    results = {"dataset": "LongMemEval_S (cleaned)", "embedder": args.embedder,
               "engine": "real" if args.engine_url else "in-process brute force",
               "questions": len(data), "turns": n_turns,
               "answer_check": (out["debug"].get("answer_check") or {}).get("model"),
               "ranker_model": __import__("backend.brain.memory_ranking.ranker", fromlist=["Ranker"]).Ranker.model().get("source"),
               "systems": {}, "timing_seconds": {k: round(v, 1) for k, v in timing.items()}, "llm_calls": 0, "api_cost_usd": 0.0}
    for name in systems:
        overall = defaultdict(list)
        res = {}
        for qt, metrics in sorted(per_type[name].items()):
            res[qt] = {m: round(float(np.mean(v)), 4) for m, v in metrics.items()}
            res[qt]["n"] = len(next(iter(metrics.values())))
            for m, v in metrics.items():
                overall[m] += v
        res["overall"] = {m: round(float(np.mean(v)), 4) for m, v in overall.items()}
        res["overall"]["n"] = len(next(iter(overall.values()))) if overall else 0
        results["systems"][name] = res
    rel = [d[0] for d in decision]
    lab = [not d[2] for d in decision]
    margins = [(d[1], not d[2]) for d in decision if d[1] is not None]
    results["abstention"] = {
        "unanswerable_declined": float(np.mean(abstain["unanswerable"])) if abstain["unanswerable"] else None,
        "answerable_wrongly_declined": float(np.mean(abstain["answerable"])) if abstain["answerable"] else None,
        "n_unanswerable": len(abstain["unanswerable"]), "n_answerable": len(abstain["answerable"]),
        "auc_top_relevance": _auc([r for r, a in zip(rel, lab) if a], [r for r, a in zip(rel, lab) if not a]),
        "auc_answer_margin": _auc([m for m, a in margins if a], [m for m, a in margins if not a]) if margins else None,
    }

    print(f"LongMemEval_S: {len(data)} questions, {n_turns} turns | embedder={args.embedder} engine={results['engine']} "
          f"answer_check={results['answer_check']}")
    print(" | ".join(f"{h:>16}" for h in ["system"] + [f"recall@{k}" for k in ks] + [f"hit@{k}" for k in ks]))
    for name in systems:
        o = results["systems"][name]["overall"]
        print(" | ".join([f"{name:>16}"] + [f"{o.get(f'recall@{k}', 0):>16.3f}" for k in ks] + [f"{o.get(f'hit@{k}', 0):>16.3f}" for k in ks]))
    for qt in sorted(per_type["recallix"]):
        print(f"  {qt:>26}: " + "  ".join(f"{s}={results['systems'][s][qt][f'recall@{kmax}']:.3f}" for s in systems))
    a = results["abstention"]
    fmt = lambda x, spec: "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, spec)  # noqa: E731
    print(f"abstention: declined {fmt(a['unanswerable_declined'], '.1%')} of {a['n_unanswerable']} unanswerable; "
          f"wrongly declined {fmt(a['answerable_wrongly_declined'], '.1%')} of {a['n_answerable']} answerable | "
          f"AUC top relevance {fmt(a['auc_top_relevance'], '.3f')}, answer margin {fmt(a['auc_answer_margin'], '.3f')}")
    print(f"timing (s): {results['timing_seconds']} | LLM calls: 0 | API cost: $0")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
