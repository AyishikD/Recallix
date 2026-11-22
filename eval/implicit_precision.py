"""
Precision of implicit-fact inference on the labelled set (eval/data/implicit).

    RECALLIX_STORAGE_ROOT=<with built rule index> python eval/implicit_precision.py

Runs the current pipeline over sentences.txt, scores every verified inference against
labels.jsonl (correct = implied by the sentence, about the right person, usable English),
and lists inferences that have no label yet — label those before trusting the number.
Shipping on by default requires precision >= 0.90 (config.yaml recall.implicit).
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    from backend import settings
    from backend.brain.features import implicit
    settings.config().setdefault("recall", {}).setdefault("implicit", {})["enabled"] = True
    inf = implicit.ImplicitInference.from_config()
    if inf is None:
        raise SystemExit("rule index or models unavailable (see backend/tools/build_inference_rules.py)")
    cfg = implicit.rules_config()
    data = REPO / "eval" / "data" / "implicit"
    labels = {(r["sentence"], r["statement"]): r["correct"] for r in map(json.loads, open(data / "labels.jsonl"))}
    sents = [s.strip() for s in open(data / "sentences.txt") if s.strip() and not s.startswith("#")]
    for name, sim, ent in (("question-time", cfg["pattern_similarity"], cfg["question_min_entailment"]),
                           ("store-time", cfg["store_pattern_similarity"], cfg["store_min_entailment"])):
        scored, unlabelled = [], []
        for s in sents:
            for r in inf.infer(s, speaker="Priya", min_entailment=ent, min_similarity=sim):
                key = (s, r["statement"])
                (scored if key in labels else unlabelled).append(labels.get(key))
        n = len(scored)
        print(f"{name:13s} precision {sum(scored)}/{n} = {sum(scored) / n:.0%}" if n else f"{name}: no labelled output",
              f"| {len(unlabelled)} unlabelled inferences" if unlabelled else "")


if __name__ == "__main__":
    main()
