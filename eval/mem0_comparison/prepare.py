"""Export the LoCoMo test split (conversations 5-9) to one JSON file both runners read (PROTOCOL.md "Data")."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "eval" / "benchmarks")]
import locomo  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
out = []
for sample in locomo.split_samples(json.load(open(src)), "test"):
    turns = [{"dia_id": d, "text": t, "speaker": sp, "ts": ts}
             for (d, t, sp), ts in zip(locomo.turns_of(sample), locomo.turn_times(sample))]
    qa = [{"question": q["question"], "category": q["category"], "answer": q.get("answer"),
           "evidence": sorted(locomo.norm_evidence(q.get("evidence")))} for q in sample["qa"]]
    out.append({"conv": sample["sample_id"], "turns": turns, "qa": qa})
json.dump(out, open(dst, "w"))
print(f"{len(out)} conversations, {sum(len(c['turns']) for c in out)} messages, {sum(len(c['qa']) for c in out)} questions -> {dst}")
