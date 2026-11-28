"""
Compare answer verifiers (recall.answer_check.verify) on LoCoMo DEV, at the same strictness.

    python eval/verifier_compare.py LOCOMO10.json OUT.json [--models a,b,c] [--keep 0.923]
    python eval/verifier_compare.py LOCOMO10.json OUT.json --split test --frozen DEV_OUT.json   (confirmation)

Recall runs once (balanced, the verifier on with a threshold that never drops anything, so the memories the reader
accepted are listed). Each candidate then scores those (question, "<speaker> said: <message>") pairs. Per model the
threshold is set so that `keep` of the real questions answered from an evidence memory keep their answer (0.923 =
350/379, the current setting); the number of answered trick questions it then declines is the comparison. A question
keeps its answer when any accepted memory scores at least the threshold. With --frozen, each model's threshold is
taken from an earlier (dev) output instead of being set on the data being scored.
"""
import argparse
import contextlib
import io
import json
import os
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "eval"), str(ROOT / "eval" / "benchmarks")]

DEFAULT_MODELS = "cross-encoder/qnli-electra-base,cross-encoder/qnli-distilroberta-base,cross-encoder/ms-marco-MiniLM-L-6-v2"


def collect(locomo_path, split="dev"):
    """[(is_trap, from_evidence, question, [premise, ...])] for every question recall answered."""
    cfg = (ROOT / "config.yaml").read_text().replace(
        "    verify:\n      enabled: false\n      model: cross-encoder/qnli-electra-base\n      threshold: 0.001",
        "    verify:\n      enabled: true\n      model: cross-encoder/qnli-electra-base\n      threshold: -1000000000.0")
    assert "threshold: -1000000000.0" in cfg, "config.yaml verify block changed; update verifier_compare.py"
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    tmp.write(cfg)
    tmp.close()
    os.environ["RECALLIC_CONFIG"] = tmp.name
    import run_eval
    import locomo
    env = run_eval.Env("st:BAAI/bge-small-en-v1.5")  # noqa: F841
    from backend.api.deps import get_memory_manager, reset
    from backend.brain.features.answer_verify import premise
    from backend.brain.memory.recall_engine import memory_speaker
    reset()
    with contextlib.redirect_stdout(io.StringIO()):
        mm = get_memory_manager()
    items = []
    for sample in locomo.split_samples(json.load(open(locomo_path)), split):
        user, m2d = locomo.ingest(mm, sample)
        for qa in sample["qa"]:
            with contextlib.redirect_stdout(io.StringIO()):
                out = mm.retrieve(user, qa["question"], limit=10)
            v = ((out.get("debug") or {}).get("answer_check") or {}).get("verify") or {}
            accepted = {int(k) for k in (v.get("scores") or {})}
            if out.get("status") != "CONSISTENT" or not accepted:
                continue
            mems = [m for m in out["memories"] if m["id"] in accepted]
            ev = locomo.norm_evidence(qa.get("evidence"))
            from_ev = any(set(m2d[m["id"]]) & ev for m in mems)
            items.append((str(qa.get("category")) == "5", from_ev, qa["question"],
                          [premise(m.get("content", ""), memory_speaker(m)) for m in mems]))
    os.unlink(tmp.name)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("locomo")
    ap.add_argument("out")
    ap.add_argument("--models", default=DEFAULT_MODELS)
    ap.add_argument("--keep", type=float, default=350 / 379)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--frozen", default=None, help="an earlier output whose per-model thresholds are reused")
    args = ap.parse_args()
    frozen = json.load(open(args.frozen))["results"] if args.frozen else {}
    items = collect(args.locomo, args.split)
    real = [it for it in items if not it[0] and it[1]]
    traps = [it for it in items if it[0]]
    from sentence_transformers import CrossEncoder
    import numpy as np
    results = {}
    for name in args.models.split(","):
        model = CrossEncoder(name, max_length=256)
        best = []
        for _, _, q, prem in items:
            s = model.predict([(q, p) for p in prem], show_progress_bar=False)
            best.append(float(np.max(np.asarray(s).reshape(-1))))
        by = dict(zip(map(id, items), best))
        real_s = sorted(by[id(it)] for it in real)
        t = (frozen[name]["threshold"] if name in frozen
             else real_s[int(round((1 - args.keep) * len(real_s)))])  # keeps `keep` of the real evidence answers
        kept = sum(by[id(it)] >= t for it in real)
        caught = sum(by[id(it)] < t for it in traps)
        lost_other = sum(by[id(it)] < t for it in items if not it[0] and not it[1])
        results[name] = {"threshold": t, "real_from_evidence_kept": f"{kept}/{len(real)}",
                         "answered_traps_declined": f"{caught}/{len(traps)}",
                         "other_real_answers_declined": f"{lost_other}/{sum(1 for it in items if not it[0] and not it[1])}"}
        print(name, results[name], flush=True)
    json.dump({"split": args.split, "frozen_from": args.frozen, "keep": None if frozen else args.keep,
               "answered_questions": len(items), "results": results}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
