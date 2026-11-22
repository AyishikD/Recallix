"""
Check the frozen trust score on data it was never fitted on. Run once per dataset; never refit from it.

    python -m backend.tools.check_trust --dataset longmemeval --data longmemeval_s_cleaned.json [--sample 100]
    python -m backend.tools.check_trust --dataset locomo-test --data locomo10.json

The trust model (backend/brain/features/trust.default.json) was fitted on LoCoMo dev
conversations. This tool scores it, unchanged, on
  longmemeval   a different dataset (different authors, style and length); a fixed random sample
                of questions (seed 0) to keep it cheap. Turns holding the answer are labelled
                (has_answer), so "the top memory is right" is exact, as on LoCoMo.
  locomo-test   LoCoMo's held-out conversations (5-9) — same dataset, unseen conversations.
For unanswerable questions (LongMemEval "_abs", LoCoMo adversarial) any served memory is wrong.

Pass bar, fixed before the first run (PASS_BAR below): the trust score must be well calibrated
(expected calibration error <= 0.10) and beat raw relevance on both Brier score and AUC.
The result is written to eval/results/trust_check_<dataset>.json with the model's SHA-256, and
the outcome is reported whatever it is.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PASS_BAR = {"max_ece": 0.10, "brier_better_than_relevance": True, "auc_better_than_relevance": True}


def _lme_indices(n_items, sample, seed, calibrate_on=0):
    """The evaluation sample (unchanged by --recalibrate-on) and a disjoint calibration sample."""
    order = random.Random(seed).sample(range(n_items), n_items)
    return sorted(order[:sample]), sorted(order[sample:sample + calibrate_on])


def _longmemeval(path, indices):
    sys.path.insert(0, str(REPO / "eval" / "benchmarks"))
    sys.path.insert(0, str(REPO / "eval"))
    from longmemeval import turns_of
    data = json.load(open(path))
    for i in indices:
        item = data[i]
        turns = turns_of(item)
        yield (f"lme_{item['question_id']}", [(k, text, speaker) for k, text, speaker, _ in turns],
               [(item["question"], {k for k, *_, has in turns if has}, item["question_id"].endswith("_abs"))])


def collect(args, indices=None):
    from backend.tools.sandbox import Sandbox
    from backend.brain.features import trust
    rows = []  # (group, trust log-odds, relevance, label)
    with Sandbox(embedder=args.embedder, answer_check=True) as box:
        if args.dataset == "locomo-test":
            sys.path.insert(0, str(REPO / "eval" / "benchmarks"))
            import locomo
            for sample in locomo.split_samples(json.load(open(args.data)), "test"):
                user, mem_to_dia = locomo.ingest(box.mm, sample)
                for qa in sample["qa"]:
                    evidence = locomo.norm_evidence(qa.get("evidence")) if qa["category"] != 5 else set()
                    with box.silenced():
                        out = box.mm.retrieve(user, qa["question"], limit=5)
                    if out["memories"]:
                        top = out["memories"][0]
                        label = int(bool(evidence) and bool(set(mem_to_dia.get(top["id"], [])) & evidence))
                        rows.append((sample["sample_id"], trust.logit(top), float(top.get("relevance") or 0.0), label))
                print(f"  {sample['sample_id']}: {len(rows)} served", flush=True)
        else:
            for n, (user, turns, questions) in enumerate(_longmemeval(args.data, indices), 1):
                mem_to_turn = {}
                with box.silenced():
                    for key, text, speaker in turns:
                        res = box.mm.store(user, text, client_metadata={"speaker": speaker})
                        mem_to_turn.setdefault(res["memory_id"], []).append(key)
                for q, evidence, unanswerable in questions:
                    with box.silenced():
                        out = box.mm.retrieve(user, q, limit=5)
                    if out["memories"]:
                        top = out["memories"][0]
                        label = int(not unanswerable and bool(set(mem_to_turn.get(top["id"], [])) & evidence))
                        rows.append((user, trust.logit(top), float(top.get("relevance") or 0.0), label))
                if n % 10 == 0:
                    print(f"  {n}/{len(indices)} questions", flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["longmemeval", "locomo-test"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--embedder", default="st:BAAI/bge-small-en-v1.5")
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recalibrate-on", type=int, default=0,
                    help="longmemeval: also fit the deployment recalibration on this many OTHER questions "
                         "(disjoint from the evaluation sample) and score the evaluation sample with it")
    ap.add_argument("--out", default=None, help="report path (default eval/results/trust_check_<dataset>.json)")
    args = ap.parse_args(argv)

    from backend.brain.features import trust
    from backend.tools.fit_trust import calibration
    model_path = Path(trust.DEFAULT_PATH)
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    fitted = json.loads(model_path.read_text())
    if args.embedder != fitted.get("embedder"):
        print(f"note: model was fitted with {fitted.get('embedder')}, checking with {args.embedder}")

    t0 = time.time()
    indices = calib_idx = None
    if args.dataset == "longmemeval":
        indices, calib_idx = _lme_indices(len(json.load(open(args.data))), args.sample, args.seed, args.recalibrate_on)
    rows = collect(args, indices)
    rows = [r for r in rows if r[1] is not None]
    y = [r[3] for r in rows]
    t = calibration([trust.recalibrated(r[1], 1.0, 0.0) for r in rows], y)
    rel = calibration([r[2] for r in rows], y)
    checks = {"ece": t["ece"] <= PASS_BAR["max_ece"], "brier": t["brier"] < rel["brier"], "auc": t["auc"] > rel["auc"]}
    report = {
        "dataset": args.dataset, "data": Path(args.data).name, "embedder": args.embedder,
        "sample": args.sample if args.dataset == "longmemeval" else None, "seed": args.seed,
        "trust_model_sha256": model_sha, "fitted_on": fitted.get("fitted_on"),
        "questions_with_a_served_memory": len(rows), "correct_rate": round(sum(y) / len(y), 4) if y else None,
        "mean_trust": round(sum(trust.recalibrated(r[1], 1.0, 0.0) for r in rows) / len(rows), 4) if rows else None,
        "trust": t, "relevance_as_probability": rel,
        "in_sample_reference": fitted.get("validation"),
        "pass_bar": PASS_BAR, "checks": checks, "passed": all(checks.values()),
        "seconds": round(time.time() - t0, 1),
    }
    if calib_idx:
        cal_rows = [r for r in collect(args, calib_idx) if r[1] is not None]
        a, b = trust.fit_recalibration([r[1] for r in cal_rows], [r[3] for r in cal_rows])
        rc = calibration([trust.recalibrated(r[1], a, b) for r in rows], y)
        report["recalibrated"] = {
            "fitted_on_questions": len(calib_idx), "fitted_on_served": len(cal_rows),
            "fitted_correct_rate": round(sum(r[3] for r in cal_rows) / len(cal_rows), 4) if cal_rows else None,
            "a": round(a, 4), "b": round(b, 4), "trust": rc,
            "mean_trust": round(sum(trust.recalibrated(r[1], a, b) for r in rows) / len(rows), 4),
            "checks": {"ece": rc["ece"] <= PASS_BAR["max_ece"], "brier": rc["brier"] < rel["brier"],
                       "auc": rc["auc"] > rel["auc"]}}
        report["recalibrated"]["passed"] = all(report["recalibrated"]["checks"].values())
        report["seconds"] = round(time.time() - t0, 1)
    suffix = "_recalibrated" if calib_idx else ""
    out = Path(args.out) if args.out else REPO / "eval" / "results" / f"trust_check_{args.dataset.replace('-', '_')}{suffix}.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("questions_with_a_served_memory", "correct_rate", "mean_trust",
                                             "trust", "relevance_as_probability", "checks", "passed", "recalibrated") if k in report}, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
