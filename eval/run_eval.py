"""
Recallix offline evaluation harness.

Measures the LLM-free pipeline against hand-labelled data in eval/data/{dev,test}/:

  facts.jsonl       fact (slot) extraction      -> precision / recall / F1
  schemas.jsonl     schema classification       -> top-1 accuracy
  importance.jsonl  importance scoring (stream) -> ROC AUC, accuracy at 5.0
  retrieval.jsonl   end-to-end store + recall   -> hit@1 / hit@3 on answerable questions,
                                                   stale-top, abstention on unanswerable ones

Rules: fit and tune on `dev` only; report `test` once changes are final.

Environment (hermetic by default):
  * storage    -> a fresh temporary directory
  * embeddings -> --embedder hash (deterministic hashed bag-of-words) or
                  --embedder st:<sentence-transformers model>
  * engine     -> in-process brute-force fake speaking the engine protocol, or
                  --engine-url URL to use a running C++ engine (needs RECALLIX_INTERNAL_KEY)
  * LLM        -> any call raises

Usage:
  python eval/run_eval.py --split dev
  python eval/run_eval.py --split test --embedder st:sentence-transformers/all-MiniLM-L6-v2
  python eval/run_eval.py --fit-ranker --embedder st:...      # fit on dev, write shipped default model
  python eval/run_eval.py --fit-importance --embedder st:...  # same for importance weights
  python eval/run_eval.py --split dev --json results.json     # machine-readable results
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
DATA = REPO / "eval" / "data"

import numpy as np  # noqa: E402


def load(split, name):
    with open(DATA / split / f"{name}.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def hash_embedder(dim=384):
    from backend.brain.memory_ranking.importance_ranker import content_terms

    def embed(text):
        v = np.zeros(dim)
        for t in content_terms(text):
            v[int(hashlib.md5(t.encode()).hexdigest(), 16) % dim] += 1
        return v.tolist() if v.any() else []
    return embed


class FakeEngine:
    """Brute-force, namespaced implementation of the engine HTTP protocol."""

    def __init__(self):
        self.ns = {}
        self.params = {"M": 16, "efConstruction": 100, "efSearch": 50}

    def post(self, path, body=None):
        body = body or {}
        if path == "/configure":
            self.params.update({k: body[k] for k in ("M", "efConstruction", "efSearch") if k in body})
            return {"status": "ok"}
        if path in ("/add", "/bulk_add"):
            items = body["items"] if path == "/bulk_add" else [{"id": body["id"], "vector": body["vector"]}]
            space = self.ns.setdefault(body["namespace"], {})
            for it in items:
                v = np.asarray(it["vector"], dtype=np.float32)
                space[it["id"]] = v / (np.linalg.norm(v) or 1.0)
            return {"status": "ok", "added": len(items)}
        if path == "/search":
            space = self.ns.get(body["namespace"], {})
            if not space:
                return {"results": []}
            ids = list(space)
            q = np.asarray(body["query"], dtype=np.float32)
            sims = np.vstack([space[i] for i in ids]) @ (q / (np.linalg.norm(q) or 1.0))
            order = np.argsort(-sims)[: body.get("top_k", 10)]
            return {"results": [{"id": ids[i], "score": float(sims[i])} for i in order]}
        if path == "/remove":
            return {"removed": self.ns.get(body["namespace"], {}).pop(body["id"], None) is not None}
        if path == "/drop":
            return {"dropped": self.ns.pop(body["namespace"], None) is not None}
        if path == "/drop_all":
            n = len(self.ns)
            self.ns.clear()
            return {"dropped": n}
        raise ValueError(path)

    def get(self, path):
        if path == "/namespaces":
            return {k: {"live": len(v), "M": self.params["M"], "efConstruction": self.params["efConstruction"],
                        "dim": len(next(iter(v.values()))) if v else 0} for k, v in self.ns.items()}
        if path == "/status":
            return {"status": "online", "instance": "fake"}
        raise ValueError(path)


class Env:
    def __init__(self, embedder="hash", engine_url=None):
        self.tmp = tempfile.mkdtemp(prefix="recallix_eval_")
        os.environ["RECALLIX_STORAGE_ROOT"] = self.tmp
        os.environ.setdefault("RECALLIX_INTERNAL_KEY", "eval-only-key-not-a-secret")
        if engine_url:
            os.environ["RECALLIX_ENGINE_URL"] = engine_url
        from backend import settings
        settings.reload()

        self.embed_calls = 0
        raw = self._embedder(embedder)
        import backend.brain.models.model_router as mr
        env = self

        def get_embedding(_self, text):
            env.embed_calls += 1
            return raw(text)

        def no_llm(_self, *a, **k):
            raise AssertionError("LLM was called on the memory pipeline")
        mr.ModelRouter.get_embedding = get_embedding
        mr.ModelRouter.route = no_llm

        if not engine_url:
            fake = FakeEngine()
            import backend.brain.memory.semantic.semantic_memory as sm
            import backend.brain.memory.index_sync as isync
            sm.engine_post = fake.post
            isync.engine_post, isync.engine_get = fake.post, fake.get

    @staticmethod
    def _embedder(spec):
        if spec == "hash":
            return hash_embedder()
        if spec.startswith("st:"):
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(spec[3:], cache_folder=os.environ.get("HF_CACHE"))
            return lambda text: model.encode([text])[0].tolist()
        raise SystemExit(f"unknown embedder {spec!r}")


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _tokens(s):
    import re
    return re.findall(r"[a-z0-9$£€.,@+-]+", s.lower().replace(",", ""))


def value_match(pred, gold):
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return p == g
    common = sum(min(p.count(t), g.count(t)) for t in set(p))
    if common == 0:
        return False
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec) >= 0.5


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
def eval_facts(split, verbose):
    from backend.brain.features.facts import extract_triads, to_slots
    tp = n_pred = n_gold = 0
    errors = []
    for item in load(split, "facts"):
        pred = {s["slot_id"]: s["value"] for s in to_slots(extract_triads(item["text"]))}
        gold = item["slots"]
        hits = sum(1 for k, v in gold.items() if k in pred and value_match(pred[k], v))
        tp, n_pred, n_gold = tp + hits, n_pred + len(pred), n_gold + len(gold)
        if hits != len(gold) or len(pred) != hits:
            errors.append((item["text"], gold, pred))
    p = tp / n_pred if n_pred else 1.0
    r = tp / n_gold if n_gold else 1.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    print(f"facts      precision={p:.2f} recall={r:.2f} F1={f1:.2f}  (tp={tp}, pred={n_pred}, gold={n_gold})")
    if verbose:
        for text, gold, pred in errors:
            print(f"   x {text!r}\n       gold={gold}\n       pred={pred}")
    return {"precision": p, "recall": r, "f1": f1}


def eval_schemas(env, split, verbose):
    from backend.brain.memory.semantic.semantic_memory import SemanticMemory
    from backend.brain.features.schemas import SchemaClassifier
    sem = SemanticMemory()
    clf = SchemaClassifier(sem.embed, sem.model_name, cache_path=os.path.join(env.tmp, "protos.json"))
    items = load(split, "schemas")
    errors = [(i["text"], i["label"], p) for i in items
              for p in [clf.classify(i["text"], sem.embed(i["text"]))] if p[0] != i["label"]]
    acc = 1 - len(errors) / len(items)
    print(f"schemas    top-1 accuracy={acc:.2f}  ({len(items) - len(errors)}/{len(items)})")
    if verbose:
        for text, gold, pred in errors:
            print(f"   x {text!r}  gold={gold} pred={pred}")
    return {"accuracy": acc}


def importance_features(split):
    """Replays a conversation stream, computing features exactly as the store path does."""
    from backend.brain.memory.semantic.semantic_memory import SemanticMemory
    from backend.brain.memory_ranking.importance_ranker import ImportanceRanker, content_terms
    sem = SemanticMemory()
    seen, vecs, doc_freq, rows = set(), [], {}, []
    for i, item in enumerate(load(split, "importance")):
        text = item["text"]
        if text in seen:  # exact duplicates are skipped by the store path
            rows.append((None, item["keep"], text))
            continue
        seen.add(text)
        v = sem.embed(text)
        max_sim = max((float(np.dot(v, u)) for u in vecs), default=None) if v is not None else None
        rows.append((ImportanceRanker.features(text, max_sim, doc_freq, total_docs=i), item["keep"], text))
        if v is not None:
            vecs.append(v)
        for t in set(content_terms(text)):
            doc_freq[t] = doc_freq.get(t, 0) + 1
    return rows


def score_importance(rows, verbose, label, weights=None):
    from backend.brain.memory_ranking.importance_ranker import ImportanceRanker
    scores = [0.0 if f is None else ImportanceRanker.score_features(f, weights) for f, _, _ in rows]
    labels = [y for _, y, _ in rows]
    a = auc(scores, labels)
    acc = sum((s >= 5.0) == bool(y) for s, y in zip(scores, labels)) / len(labels)
    print(f"importance {label:12} AUC={a:.2f} accuracy@5.0={acc:.2f}")
    if verbose:
        for s, (_, y, text) in sorted(zip(scores, rows), key=lambda x: x[0]):
            print(f"   {s:5.2f} keep={y} {text!r}{'' if (s >= 5.0) == bool(y) else '   <- wrong'}")
    return {"auc": a, "accuracy": acc}


def _label(content, q):
    exp = q.get("expect")
    if exp is None:
        return 0
    c = content.lower()
    return int(exp.lower() in c and not (q.get("not_top") and q["not_top"].lower() in c))


def retrieval_queries(env, split, collect=False, verbose=False):
    """Stores each scenario for a fresh user and runs its queries. Returns per-query records."""
    from backend.api.deps import get_memory_manager, reset
    reset()
    with quiet():
        mm = get_memory_manager()
    records = []
    stage_ms, store_ms = {}, []
    for n, scenario in enumerate(load(split, "retrieval")):
        user = f"eval_{split}_{n}"
        with quiet():
            for m in scenario["memories"]:
                res = mm.store(user, m)
                store_ms.append(res["total_ms"])
                for entry in res["processing_log"]:
                    stage_ms.setdefault(entry["stage"], []).append(entry["duration_ms"])
        for q in scenario["queries"]:
            before = env.embed_calls
            with quiet():
                out = mm.retrieve(user, q["q"], limit=3, return_candidates=collect)
            contents = [m.get("content", "") for m in out["memories"]]
            rec = {"scenario": scenario["name"], "q": q, "contents": contents, "status": out["status"],
                   "embed_calls": env.embed_calls - before, "top_relevance": out["debug"]["top_relevance"],
                   "latency_ms": out["latency_ms"], "store_stage_ms": stage_ms, "store_ms": store_ms}
            if collect:
                rec["candidates"] = [(c["features"], _label(c.get("content", ""), q)) for c in out["_candidates"]]
            records.append(rec)
    return records


def eval_retrieval(env, split, verbose):
    recs = retrieval_queries(env, split)
    ans = [r for r in recs if r["q"].get("expect") is not None]
    unans = [r for r in recs if r["q"].get("expect") is None]
    hit1 = sum(bool(r["contents"]) and r["q"]["expect"].lower() in r["contents"][0].lower() for r in ans)
    hit3 = sum(any(r["q"]["expect"].lower() in c.lower() for c in r["contents"][:3]) for r in ans)
    stale = sum(bool(r["q"].get("not_top")) and bool(r["contents"]) and r["q"]["not_top"].lower() in r["contents"][0].lower()
                for r in ans)
    false_abstain = sum(not r["contents"] for r in ans)
    abstained = sum(not r["contents"] for r in unans)
    res = {"hit@1": hit1 / len(ans), "hit@3": hit3 / len(ans), "stale_top": stale,
           "false_abstain_rate": false_abstain / len(ans),
           "abstain_accuracy": (abstained / len(unans)) if unans else float("nan"),
           "embed_calls_per_query": float(np.mean([r["embed_calls"] for r in recs])),
           # wall-clock timings include the embedding model; see `embedder` and `engine` in the results
           "recall_ms_p50": float(np.percentile([r["latency_ms"] for r in recs], 50)),
           "recall_ms_p95": float(np.percentile([r["latency_ms"] for r in recs], 95)),
           "store_ms_p50": float(np.percentile(recs[-1]["store_ms"], 50)),
           "store_stage_ms_mean": {k: round(float(np.mean(v)), 3) for k, v in recs[-1]["store_stage_ms"].items()}}
    print(f"retrieval  hit@1={res['hit@1']:.2f} hit@3={res['hit@3']:.2f} stale-top={stale} "
          f"false-abstain={res['false_abstain_rate']:.2f} | unanswerable abstained={abstained}/{len(unans)} "
          f"| {len(ans)}+{len(unans)} queries, embed calls/query={res['embed_calls_per_query']:.1f}")
    if verbose:
        for r in recs:
            exp = r["q"].get("expect")
            ok = (not r["contents"]) if exp is None else (bool(r["contents"]) and exp.lower() in r["contents"][0].lower())
            if not ok:
                print(f"   x [{r['scenario']}] {r['q']['q']!r} expect={exp!r} status={r['status']} top={r['top_relevance']}")
                for c in r["contents"]:
                    print(f"       - {c}")
    return res


EXAMPLE_MEMORIES = ["I'm allergic to peanuts and prefer dark chocolate.", "My name is Priya and I live in Pune.",
                    "Need to renew my passport before 1 June.", "The weather is nice today."]
EXAMPLE_QUERY = "What am I allergic to?"


def run_example(env):
    """Stores the website's example messages and recalls one question; returns the real outputs."""
    from backend.api.deps import get_memory_manager, reset
    reset()
    with quiet():
        mm = get_memory_manager()
        stores = [mm.store("example_user", m) for m in EXAMPLE_MEMORIES]
        recall = mm.retrieve("example_user", EXAMPLE_QUERY, limit=3)
    from backend.brain.features.facts import extract_triads, to_slots
    return {
        "stores": [{"content": m, "importance": round(r["importance"], 2), "schemas": r["schemas"],
                    "promoted": r["promoted"], "slots": [x["slot_id"] + " = " + x["value"] for x in to_slots(extract_triads(m))],
                    "processing_log": r["processing_log"]} for m, r in zip(EXAMPLE_MEMORIES, stores)],
        "query": EXAMPLE_QUERY,
        "recall": {"status": recall["status"], "latency_ms": recall["latency_ms"], "threshold": recall["debug"]["threshold"],
                   "memories": [{"content": m["content"], "relevance": m["relevance"], "explain": m["explain"]}
                                for m in recall["memories"]]},
    }


def eval_legacy(env, verbose):
    """
    Pre-existing benchmarks ported from scratch/ (gauntlet v1-v3, 500-turn
    retention). Never used for tuning; reported as an independent check.
    """
    import random
    from backend.api.deps import get_memory_manager, reset
    reset()
    with quiet():
        mm = get_memory_manager()

    passed, total, lines = 0, 0, []
    for n, g in enumerate(json.loads(line) for line in open(DATA / "legacy" / "gauntlet.jsonl") if line.strip()):
        user = f"legacy_g{n}"
        for step in g["steps"]:
            if "store" in step:
                with quiet():
                    mm.store(user, step["store"])
                continue
            with quiet():
                out = mm.retrieve(user, step["query"], limit=5)
            contents = [m["content"] for m in out["memories"]]
            exp, mode = step["expect"], step["mode"]
            ok = (not contents) if mode == "abstain" else \
                any(exp.lower() in c.lower() for c in contents) if mode == "any" else \
                bool(contents) and exp.lower() in contents[0].lower()
            passed, total = passed + ok, total + 1
            lines.append(f"   {'PASS' if ok else 'FAIL'} [{g['name']}] {step['query']!r} -> "
                         f"{contents[0] if contents else '(abstained)'}")
    print(f"legacy     gauntlet v1-v3: {passed}/{total} passed")

    cfg = json.load(open(DATA / "legacy" / "retention.json"))
    rng = random.Random(cfg["seed"])
    user = "legacy_retention"
    with quiet():
        for i in range(cfg["turns"]):
            imp = cfg["important"].get(str(i))
            text = imp[0] if imp else cfg["filler_template"].format(run=cfg["run_id"], i=i, topic=rng.choice(cfg["filler_topics"]))
            mm.store(user, text.format(run=cfg["run_id"]))
    hits = 0
    for turn, (_, query, expect) in cfg["important"].items():
        with quiet():
            out = mm.retrieve(user, query, limit=5)
        top = out["memories"][0]["content"] if out["memories"] else ""
        ok = expect.lower() in top.lower()
        hits += ok
        lines.append(f"   {'PASS' if ok else 'FAIL'} [retention t={turn}] {query!r} -> {top or '(abstained)'}")
    print(f"legacy     retention ({cfg['turns']} turns): recall@1 {hits}/{len(cfg['important'])}")
    if verbose:
        print("\n".join(lines))
    return {"gauntlet_passed": passed, "gauntlet_total": total, "retention_hits": hits,
            "retention_total": len(cfg["important"])}


# ---------------------------------------------------------------------------
def fit_ranker(env, embedder):
    from backend.brain.memory_ranking import fit_ranker as fr
    from backend.brain.memory_ranking.ranker import DEFAULT_MODEL_PATH, Ranker
    recs = retrieval_queries(env, "dev", collect=True)
    groups = {}
    for r in recs:
        groups.setdefault(r["scenario"], []).append(r["candidates"])
    model = fr.fit_model_cv(groups, source=f"eval/data/dev/retrieval.jsonl, embedder={embedder}")
    with open(DEFAULT_MODEL_PATH, "w") as f:
        json.dump(model, f, indent=2)
    print(f"fitted ranker on {model['n_queries']} dev queries / {model['n_candidates']} candidates; "
          f"lam={model['lam']} threshold={model['threshold']:.3f}; "
          f"leave-one-scenario-out CV query accuracy={model['cv_query_accuracy']:.2f} (by lam: {model['cv_by_lam']})")
    print(f"wrote {DEFAULT_MODEL_PATH}")


def fit_importance(embedder):
    from backend.brain.memory_ranking import logistic
    from backend.brain.memory_ranking.importance_ranker import FEATURES, DEFAULT_WEIGHTS_PATH, PRIOR_WEIGHTS_PATH
    with open(PRIOR_WEIGHTS_PATH) as f:
        prior = json.load(f)["weights"]
    with quiet():
        rows = importance_features("dev")
    feats = [f for f, _, _ in rows if f is not None]
    labels = [y for f, y, _ in rows if f is not None]
    w = logistic.fit(feats, labels, prior, FEATURES, lam=0.5)
    with open(DEFAULT_WEIGHTS_PATH, "w") as f:
        json.dump({"weights": w, "source": f"eval/data/dev/importance.jsonl, embedder={embedder}",
                   "n_examples": len(labels), "lam": 0.5}, f, indent=2)
    print("fitted importance weights:", {k: round(v, 2) for k, v in w.items()})
    print(f"wrote {DEFAULT_WEIGHTS_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--embedder", default="hash")
    ap.add_argument("--engine-url", default=None)
    ap.add_argument("--only", default="facts,schemas,importance,retrieval")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--json", default=None, help="write results to this file")
    ap.add_argument("--fit-ranker", action="store_true")
    ap.add_argument("--fit-importance", action="store_true")
    args = ap.parse_args()

    env = Env(args.embedder, args.engine_url)
    if args.fit_importance:
        fit_importance(args.embedder)
    if args.fit_ranker:
        fit_ranker(env, args.embedder)
    if args.fit_ranker or args.fit_importance:
        return

    print(f"== split={args.split} embedder={args.embedder} engine={'real ' + args.engine_url if args.engine_url else 'fake'} ==")
    results = {"split": args.split, "embedder": args.embedder}
    tasks = args.only.split(",")
    if "facts" in tasks:
        results["facts"] = eval_facts(args.split, args.verbose)
    if "schemas" in tasks:
        with quiet():
            from backend.brain.memory.semantic.semantic_memory import SemanticMemory  # noqa: F401
        results["schemas"] = eval_schemas(env, args.split, args.verbose)
    if "importance" in tasks:
        with quiet():
            rows = importance_features(args.split)
        results["importance"] = score_importance(rows, args.verbose, "")
    if "retrieval" in tasks:
        results["retrieval"] = eval_retrieval(env, args.split, args.verbose)
    if "legacy" in tasks:
        results["legacy"] = eval_legacy(env, args.verbose)
    if "example" in tasks:
        results["example"] = run_example(env)
    results["engine"] = "real" if args.engine_url else "in-process brute force"
    results["ranker_model"] = __import__("backend.brain.memory_ranking.ranker", fromlist=["Ranker"]).Ranker.model().get("source")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
