"""
Compare the rule grammar, the trained student(s) and their union on the held-out messages.

    python phase6/evaluate.py

Gold = teacher labels on the 300 held-out messages (never trained on), checked by the
Recallix author (single annotator). A predicted fact matches a gold fact when the relation
agrees and the values match after normalisation (exact) or overlap (relaxed: one contains
the other). Reports precision / recall / F1 per system, value-only scores (relation ignored)
— fairer to the rules, whose relation names differ — and entity accuracy.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, labels as all_labels, norm, read_jsonl  # noqa: E402

RULE_TO_RELATION = {  # rule-grammar relation names -> the closed set (others count as mismatches)
    "location": "location", "name": "name", "age": "age", "birthday": "birthday", "job": "job", "employer": "employer",
    "email": "email", "phone": "phone", "allergy": "allergy", "deadline": "deadline", "when": "event_time",
    "budget": "budget", "dog": "pet", "cat": "pet", "pet": "pet", "address": "address", "nickname": "name",
}


def _match(p, g, relaxed):
    pv, gv = norm(p["value"]), norm(g["value"])
    same = pv == gv or (relaxed and (pv in gv or gv in pv))
    return same


def score(pred, gold, use_relation=True, relaxed=False):
    tp = fp = fn = 0
    for mid, g_facts in gold.items():
        p_facts = list(pred.get(mid, []))
        used = set()
        for p in p_facts:
            hit = next((k for k, g in enumerate(g_facts) if k not in used and _match(p, g, relaxed)
                        and (not use_relation or p["relation"] == g["relation"])), None)
            if hit is None:
                fp += 1
            else:
                tp += 1
                used.add(hit)
        fn += len(g_facts) - len(used)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0}


def rules(texts):
    from backend.brain.features.facts import extract_triads
    out = {}
    for mid, text in texts.items():
        out[mid] = [{"relation": RULE_TO_RELATION.get(t["relation"].lower().replace(" ", "_"), "other:" + t["relation"]),
                     "value": t["value"], "entity": t["entity"]} for t in extract_triads(text)]
    return out


def student(texts, variant):
    from backend.brain.features.learned_facts import LearnedExtractor
    ex = LearnedExtractor(DATA / "models" / variant)
    return {mid: ex.extract(text) for mid, text in texts.items()}


def union(a, b):
    out = {}
    for mid in set(a) | set(b):
        facts = list(a.get(mid, []))
        for f in b.get(mid, []):
            if not any(norm(f["value"]) == norm(x["value"]) or norm(f["value"]) in norm(x["value"]) for x in facts):
                facts.append(f)
        out[mid] = facts
    return out


def main():
    pool = {r["id"]: r for r in read_jsonl(DATA / "pool.jsonl")}
    held = json.loads((DATA / "splits.json").read_text())["held_out"]
    labels = all_labels()
    reviewed_path = DATA / "labels_reviewed.jsonl"
    reviewed = {r["id"] for r in read_jsonl(reviewed_path)} if reviewed_path.exists() else set()
    print(f"answer key: {len(set(held) & reviewed)}/{len(held)} held-out messages reviewed")
    gold = {mid: labels[mid] for mid in held if mid in labels}
    texts = {mid: pool[mid]["text"] for mid in gold}
    print(f"held-out: {len(gold)} messages, {sum(len(v) for v in gold.values())} gold facts")
    systems = {"rules": rules(texts)}
    for variant in ("all", "no-sa"):
        if (DATA / "models" / variant).exists():
            systems[f"student-{variant}"] = student(texts, variant)
            systems[f"rules+student-{variant}"] = union(systems["rules"], systems[f"student-{variant}"])
    report = {}
    print(f"{'system':24s} {'strict P/R/F1':>22s} {'relaxed F1':>11s} {'value-only F1':>14s} {'entity acc':>11s}")
    for name, pred in systems.items():
        strict, relaxed = score(pred, gold), score(pred, gold, relaxed=True)
        value_only = score(pred, gold, use_relation=False, relaxed=True)
        ent_ok = ent_n = 0
        for mid, g_facts in gold.items():
            for p in pred.get(mid, []):
                g = next((g for g in g_facts if _match(p, g, True)), None)
                if g:
                    ent_n += 1
                    ent_ok += norm(p.get("entity")) == norm(g["entity"])
        ent = ent_ok / ent_n if ent_n else 0.0
        report[name] = {"strict": strict, "relaxed": relaxed, "value_only": value_only, "entity_accuracy": round(ent, 3)}
        print(f"{name:24s} {strict['precision']:.2f}/{strict['recall']:.2f}/{strict['f1']:.2f}".ljust(48)
              + f"{relaxed['f1']:>11.2f} {value_only['f1']:>14.2f} {ent:>11.2f}")
    (DATA / "eval_report.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
