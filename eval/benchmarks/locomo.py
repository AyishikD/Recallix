"""
LoCoMo retrieval benchmark (no LLM needed).

LoCoMo (Maharana et al., 2024; github.com/snap-research/locomo) has 10 long
two-person conversations (~5.9k turns) and ~2k questions. Every question lists
the dialogue turns ("evidence") that contain its answer, so retrieval can be
scored directly, without an answer-generating LLM or an LLM judge:

  evidence recall@k  share of a question's evidence turns among the top k retrieved
  hit@k              at least one evidence turn in the top k

Categories 1-4 are answerable. Category 5 is adversarial (the question presumes
something the conversation never says); the right behaviour is to decline, so
for Recallic we report how often it abstains there, and how often it wrongly
abstains on answerable questions. Baselines cannot abstain.

Systems, all fed the identical text for every turn ("[date] Speaker: text"):
  recallix   full store + recall pipeline (as served, with abstention)
  recallix_ranked  Recallic's ranking before truth arbitration and abstention
  dense      cosine top-k over the same embeddings (brute force)
  bm25       Okapi BM25 (k1, b configurable; defaults are the standard 1.2 / 0.75)

Usage:
  curl -sSfL -o /tmp/locomo/locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
  python eval/benchmarks/locomo.py --data /tmp/locomo/locomo10.json \
      --embedder st:sentence-transformers/all-MiniLM-L6-v2 [--engine-url URL] --json eval/results/locomo.json
"""
import argparse
import contextlib
import io
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # eval/
import run_eval  # noqa: E402

TOKEN = re.compile(r"[a-z0-9]+")
CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


DEV_CONVERSATIONS = 5  # conversations [0, 5) may be used for fitting; [5, 10) are held out for reporting


def split_samples(data, split):
    return {"dev": data[:DEV_CONVERSATIONS], "test": data[DEV_CONVERSATIONS:], "all": data}[split]


def turns_of(sample):
    """[(dia_id, text, speaker)] in conversation order, text = '[date] Speaker: utterance [+ image caption]'.
    turn_times(sample) gives each turn's real time."""
    conv = sample["conversation"]
    sessions = sorted((int(k.split("_")[1]), k) for k in conv
                      if re.fullmatch(r"session_\d+", k) and isinstance(conv[k], list))
    out = []
    for _, key in sessions:
        date = conv.get(f"{key}_date_time", "")
        for turn in conv[key]:
            text = f"[{date}] {turn['speaker']}: {turn['text']}"
            if turn.get("blip_caption"):
                text += f" [shares an image: {turn['blip_caption']}]"
            out.append((turn["dia_id"].strip(), text, turn["speaker"]))
    return out


def session_time(date_text):
    """'1:56 pm on 8 May, 2023' -> a UTC timestamp (None if it does not parse)."""
    import calendar
    from datetime import datetime
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return calendar.timegm(datetime.strptime(date_text.strip(), fmt).timetuple())
        except ValueError:
            continue
    return None


def turn_times(sample):
    """Each turn's time: its session's date and time, plus one second per turn so order within a session is kept."""
    conv = sample["conversation"]
    sessions = sorted((int(k.split("_")[1]), k) for k in conv
                      if re.fullmatch(r"session_\d+", k) and isinstance(conv[k], list))
    out = []
    for _, key in sessions:
        base = session_time(conv.get(f"{key}_date_time", ""))
        for i, _turn in enumerate(conv[key]):
            out.append(None if base is None else base + i)
    return out


def ingest(mm, sample):
    """Stores every turn (with its labelled speaker as metadata). Returns {memory_id: [dia_id, ...]}."""
    user = f"locomo_{sample['sample_id']}"
    mem_to_dia = defaultdict(list)
    with contextlib.redirect_stdout(io.StringIO()):
        for (dia_id, text, speaker), ts in zip(turns_of(sample), turn_times(sample)):
            res = mm.store(user, text, client_metadata={"speaker": speaker}, timestamp=ts)
            mem_to_dia[res["memory_id"]].append(dia_id)
    return user, mem_to_dia


def labelled_queries(mm, samples):
    """
    Training data for fit_ranker: {conversation: [[(features, label), ...] per question]}.
    Label 1 = the candidate holds an evidence turn of an answerable question;
    adversarial (category 5) questions have only negatives (the right answer is to abstain).
    """
    groups = {}
    for sample in samples:
        user, mem_to_dia = ingest(mm, sample)
        queries = []
        for qa in sample["qa"]:
            evidence = norm_evidence(qa.get("evidence")) if qa["category"] != 5 else set()
            with contextlib.redirect_stdout(io.StringIO()):
                out = mm.retrieve(user, qa["question"], limit=10, return_candidates=True)
            queries.append([(c["features"], int(bool(set(mem_to_dia[c["id"]]) & evidence)))
                            for c in out["_candidates"]])
        groups[f"locomo_{sample['sample_id']}"] = queries
    return groups


def norm_evidence(ev):
    ids = []
    for e in ev or []:
        ids += [x.strip() for x in re.split(r"[;,]", str(e)) if x.strip()]
    return set(ids)


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(TOKEN.findall(d.lower())) for d in docs]
        self.lens = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg = self.lens.mean() if len(self.lens) else 0.0
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def top(self, query, k):
        q = TOKEN.findall(query.lower())
        scores = np.zeros(len(self.docs))
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in q:
                tf = d.get(t)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
            scores[i] = s
        return list(np.argsort(-scores)[:k])


def _auc(pos, neg):
    """P(score of a random positive > score of a random negative), ties count half."""
    if not pos or not neg:
        return float("nan")
    neg_sorted = np.sort(np.asarray(neg))
    wins = sum(np.searchsorted(neg_sorted, p, side="left") + 0.5 * (np.searchsorted(neg_sorted, p, side="right")
               - np.searchsorted(neg_sorted, p, side="left")) for p in pos)
    return float(wins / (len(pos) * len(neg)))


def abstention_analysis(rows, budgets=(0.05, 0.10, 0.20)):
    """
    How well a score separates answerable from adversarial questions, independent of
    any threshold: AUC, and the share of adversarial questions declined when at most
    `budget` of answerable ones are declined. Scores: the answer check's best margin
    (when enabled) and the top relevance.
    """
    out = {}
    for name, idx in (("answer_margin", 2), ("top_relevance", 0)):
        vals = [(r[idx], r[1]) for r in rows if r[idx] is not None]
        if not vals:
            continue
        ans = np.sort([v for v, adv in vals if not adv])
        adv = np.asarray([v for v, a in vals if a])
        res = {"auc": _auc(list(ans), list(adv)), "operating_points": []}
        for b in budgets:
            t = ans[int(np.floor(b * len(ans)))] if len(ans) else 0.0
            res["operating_points"].append({"max_false_decline": b, "adversarial_declined": float(np.mean(adv < t)) if len(adv) else float("nan")})
        out[name] = res
    return out


def score(retrieved_ids, evidence, ks):
    out = {}
    for k in ks:
        top = set(retrieved_ids[:k])
        out[f"recall@{k}"] = len(top & evidence) / len(evidence)
        out[f"hit@{k}"] = float(bool(top & evidence))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--embedder", default="hash")
    ap.add_argument("--engine-url", default=None)
    ap.add_argument("--k", default="5,10")
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"],
                    help="dev = first 5 conversations (may be used for fitting); test = last 5 (held out)")
    ap.add_argument("--conversations", type=int, default=None, help="limit within the split (quick runs)")
    ap.add_argument("--bm25-k1", type=float, default=1.2)
    ap.add_argument("--bm25-b", type=float, default=0.75)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    ks = [int(x) for x in args.k.split(",")]
    kmax = max(ks)

    env = run_eval.Env(args.embedder, args.engine_url)
    from backend.api.deps import get_memory_manager, reset
    reset()
    with contextlib.redirect_stdout(io.StringIO()):
        mm = get_memory_manager()

    data = split_samples(json.load(open(args.data)), args.split)[: args.conversations]
    systems = ["recallix", "recallix_ranked", "dense", "bm25"]
    per_cat = {s: defaultdict(lambda: defaultdict(list)) for s in systems}
    abstain = {"answerable": [], "adversarial": []}
    # (top_relevance, is_adversarial) per question, for threshold-free analysis
    decision_scores = []
    timing = defaultdict(float)
    n_turns = n_questions = 0
    embed_calls_ingest = 0

    for sample in data:
        turns = turns_of(sample)
        n_turns += len(turns)
        texts = [t for _, t, _ in turns]

        # ---- ingest
        before = env.embed_calls
        t0 = time.time()
        user, mem_to_dia = ingest(mm, sample)
        timing["recallix_ingest_s"] += time.time() - t0
        embed_calls_ingest += env.embed_calls - before

        t0 = time.time()
        vecs = np.vstack([mm.semantic.embed(t) for t in texts])  # cached: same embeddings Recallic stored
        timing["dense_ingest_s"] += time.time() - t0
        t0 = time.time()
        bm25 = BM25(texts, args.bm25_k1, args.bm25_b)
        timing["bm25_ingest_s"] += time.time() - t0
        dia_ids = [d for d, _, _ in turns]

        # ---- questions
        for qa in sample["qa"]:
            cat = qa["category"]
            evidence = norm_evidence(qa.get("evidence")) & set(dia_ids)
            q = qa["question"]
            n_questions += 1

            t0 = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                out = mm.retrieve(user, q, limit=kmax, return_candidates=True)
            timing["recallix_query_s"] += time.time() - t0
            served = [d for m in out["memories"] for d in mem_to_dia[m["id"]]]
            ranked = [d for c in out["_candidates"] for d in mem_to_dia[c["id"]]]
            abstained = out["status"] in ("ABSTAINED", "NO_MATCH")
            chk = out["debug"].get("answer_check")
            decision_scores.append((out["debug"].get("top_relevance") or 0.0, cat == 5,
                                    chk["best_margin"] if chk else None))

            t0 = time.time()
            qv = mm.semantic.embed(q)
            dense = [dia_ids[i] for i in np.argsort(-(vecs @ qv))[:kmax]]
            timing["dense_query_s"] += time.time() - t0
            t0 = time.time()
            lexical = [dia_ids[i] for i in bm25.top(q, kmax)]
            timing["bm25_query_s"] += time.time() - t0

            if cat == 5:
                abstain["adversarial"].append(abstained)
                continue
            abstain["answerable"].append(abstained)
            if not evidence:
                continue
            for name, ids in (("recallix", served), ("recallix_ranked", ranked), ("dense", dense), ("bm25", lexical)):
                for metric, v in score(ids, evidence, ks).items():
                    per_cat[name][cat][metric].append(v)

    # ---- report
    results = {"dataset": "LoCoMo (locomo10.json)", "split": args.split, "embedder": args.embedder,
               "engine": "real" if args.engine_url else "in-process brute force",
               "conversations": len(data), "turns": n_turns, "questions": n_questions,
               "ranker_model": __import__("backend.brain.memory_ranking.ranker", fromlist=["Ranker"]).Ranker.model().get("source"),
               "systems": {}, "abstention": {
                   "adversarial_declined": float(np.mean(abstain["adversarial"])) if abstain["adversarial"] else None,
                   "answerable_wrongly_declined": float(np.mean(abstain["answerable"])) if abstain["answerable"] else None,
                   "n_adversarial": len(abstain["adversarial"]), "n_answerable": len(abstain["answerable"])},
               "abstention_analysis": abstention_analysis(decision_scores),
               "timing_seconds": {k: round(v, 2) for k, v in timing.items()},
               "embedding_calls_during_ingest": embed_calls_ingest,
               "llm_calls": 0, "api_cost_usd": 0.0}
    for name in systems:
        cats = per_cat[name]
        overall = defaultdict(list)
        sys_res = {}
        for cat in sorted(cats):
            sys_res[CATEGORY_NAMES.get(cat, str(cat))] = {m: round(float(np.mean(v)), 4) for m, v in cats[cat].items()}
            sys_res[CATEGORY_NAMES.get(cat, str(cat))]["n"] = len(next(iter(cats[cat].values())))
            for m, v in cats[cat].items():
                overall[m] += v
        sys_res["overall"] = {m: round(float(np.mean(v)), 4) for m, v in overall.items()}
        sys_res["overall"]["n"] = len(next(iter(overall.values()))) if overall else 0
        results["systems"][name] = sys_res

    print(f"LoCoMo [{args.split}]: {len(data)} conversations, {n_turns} turns, {n_questions} questions | embedder={args.embedder} engine={results['engine']}")
    head = ["system"] + [f"recall@{k}" for k in ks] + [f"hit@{k}" for k in ks]
    print(" | ".join(f"{h:>16}" for h in head))
    for name in systems:
        o = results["systems"][name]["overall"]
        print(" | ".join([f"{name:>16}"] + [f"{o.get(f'recall@{k}', 0):>16.3f}" for k in ks] + [f"{o.get(f'hit@{k}', 0):>16.3f}" for k in ks]))
    print("per category recall@%d:" % kmax)
    for cat in sorted({c for s in systems for c in per_cat[s]}):
        name = CATEGORY_NAMES.get(cat, str(cat))
        print(f"  {name:>12}: " + "  ".join(f"{s}={results['systems'][s][name][f'recall@{kmax}']:.3f}" for s in systems))
    a = results["abstention"]
    print(f"Recallic abstention: declined {a['adversarial_declined']:.2%} of {a['n_adversarial']} adversarial questions; "
          f"wrongly declined {a['answerable_wrongly_declined']:.2%} of {a['n_answerable']} answerable")
    for name, aa in results["abstention_analysis"].items():
        print(f"answerable-vs-adversarial AUC of {name}: {aa['auc']:.3f}  ("
              + ", ".join(f"{r['adversarial_declined']:.1%} of traps declined at {r['max_false_decline']:.0%} false declines"
                          for r in aa["operating_points"]) + ")")
    print(f"timing (s): {results['timing_seconds']} | LLM calls: 0 | API cost: $0")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
