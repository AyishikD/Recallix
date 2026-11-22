"""
Open memory benchmark harness: any memory system, same data, same scoring.

    python eval/harness.py --dataset locomo --data locomo10.json --split dev \\
        --systems recallix,recallix-ranked,vector-search,bm25 --embedder st:BAAI/bge-small-en-v1.5 \\
        --json results.json

    python eval/harness.py --dataset longmemeval --data longmemeval_s_cleaned.json --limit 50 ...

Systems implement eval/adapters/base.py (four methods). Built in: recallix (what it
serves, after its decline decision), recallix-ranked (its full ranking), vector-search,
bm25. Optional: mem0 (eval/adapters/mem0_adapter.py, unverified — see that file).

Scoring (identical for every system; the same functions as eval/benchmarks/*):
  * evidence recall@k / hit@k over answerable questions: did the turns that contain the
    answer reach the top k? (no LLM judge, so no judge bias)
  * declines: share of unanswerable questions for which the system returned nothing, and
    share of answerable questions it wrongly returned nothing for.

Every run records each system's configuration and a fingerprint of the dataset file, so
a published number can be reproduced exactly. Benchmark hygiene still applies: tune on
dev, report test once.
"""
import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval" / "benchmarks"))
sys.path.insert(0, str(REPO / "eval"))


def conversations(dataset, data, split, limit):
    """Yields (user, turns, questions); turns [(turn_id, text, speaker, ts)], questions [(text, evidence_set, answerable)]."""
    if dataset == "locomo":
        import locomo
        for sample in locomo.split_samples(data, split)[:limit]:
            turns = [(tid, text, sp, 1_700_000_000.0 + i) for i, (tid, text, sp) in enumerate(locomo.turns_of(sample))]
            qs = [(qa["question"], locomo.norm_evidence(qa.get("evidence")) if qa["category"] != 5 else set(),
                   qa["category"] != 5) for qa in sample["qa"]]
            yield f"locomo_{sample['sample_id']}", turns, qs
    elif dataset == "longmemeval":
        import longmemeval
        for item in data[:limit]:
            raw = longmemeval.turns_of(item)
            turns = [(key, text, sp, 1_700_000_000.0 + i) for i, (key, text, sp, _) in enumerate(raw)]
            evidence = {key for key, _, _, has in raw if has}
            unanswerable = item["question_id"].endswith("_abs")
            yield f"lme_{item['question_id']}", turns, [(item["question"], evidence, not unanswerable)]
    else:
        raise SystemExit(f"unknown dataset {dataset!r}")


def make_system(name, embedder):
    from adapters import builtin
    if name == "recallix":
        return builtin.Recallix(embedder, mode="served")
    if name == "recallix-ranked":
        return builtin.Recallix(embedder, mode="ranked")
    if name == "vector-search":
        return builtin.Dense(embedder)
    if name == "bm25":
        return builtin.BM25()
    if name == "mem0":
        from adapters.mem0_adapter import Mem0
        return Mem0()
    raise SystemExit(f"unknown system {name!r}")


def run(dataset, data, split, limit, systems, ks):
    import locomo
    kmax = max(ks)
    scores = {s.name: defaultdict(list) for s in systems}
    declines = {s.name: {"unanswerable": [], "answerable": []} for s in systems}
    timing = defaultdict(float)
    n_conv = n_q = 0
    for user, turns, questions in conversations(dataset, data, split, limit):
        n_conv += 1
        for s in systems:
            t = time.time()
            s.add_conversation(user, turns)
            timing[f"{s.name}_ingest_s"] += time.time() - t
        for text, evidence, answerable in questions:
            n_q += 1
            for s in systems:
                t = time.time()
                got = s.search(user, text, kmax)
                timing[f"{s.name}_query_s"] += time.time() - t
                declines[s.name]["answerable" if answerable else "unanswerable"].append(not got)
                if answerable and evidence:
                    for metric, v in locomo.score(got, evidence, ks).items():
                        scores[s.name][metric].append(v)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None  # noqa: E731
    return {
        "conversations": n_conv, "questions": n_q,
        "systems": {name: {"config": next(s.config for s in systems if s.name == name),
                           **{m: mean(v) for m, v in sc.items()},
                           "declined_unanswerable": mean(declines[name]["unanswerable"]),
                           "wrongly_declined_answerable": mean(declines[name]["answerable"])}
                    for name, sc in scores.items()},
        "timing_seconds": {k: round(v, 2) for k, v in timing.items()},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["locomo", "longmemeval"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"], help="LoCoMo only")
    ap.add_argument("--limit", type=int, default=None, help="first N conversations / questions")
    ap.add_argument("--systems", default="recallix,recallix-ranked,vector-search,bm25")
    ap.add_argument("--embedder", default="hash")
    ap.add_argument("--k", default="5,10")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    raw = Path(args.data).read_bytes()
    data = json.loads(raw)
    ks = [int(x) for x in args.k.split(",")]
    systems = [make_system(n.strip(), args.embedder) for n in args.systems.split(",") if n.strip()]
    try:
        result = run(args.dataset, data, args.split, args.limit, systems, ks)
    finally:
        for s in systems:
            s.close()
    result.update({"dataset": args.dataset, "split": args.split if args.dataset == "locomo" else None,
                   "limit": args.limit, "data_sha256": hashlib.sha256(raw).hexdigest()[:16], "ks": ks})
    cols = [f"recall@{k}" for k in ks] + [f"hit@{k}" for k in ks] + ["declined_unanswerable", "wrongly_declined_answerable"]
    print(f"{args.dataset} ({result['split'] or 'all'}): {result['conversations']} conversations, {result['questions']} questions")
    print(" | ".join(f"{c:>16}" for c in ["system"] + cols))
    for name, r in result["systems"].items():
        print(" | ".join([f"{name:>16}"] + [f"{r[c]:>16.3f}" if r.get(c) is not None else f"{'—':>16}" for c in cols]))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))
        print("wrote", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
