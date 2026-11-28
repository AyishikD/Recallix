"""
Curate the ATOMIC rule index for implicit facts (the Phase 4 follow-up).

    RECALLIC_STORAGE_ROOT=<with inference_rules/ built> python phase6/curate_atomic.py

ATOMIC effects are crowd-sourced *likely* consequences; implicit facts need *certain* ones.
For the event patterns that real messages reach (top matches of the phase-6 message pool —
never the Phase 4 evaluation sentences), the teacher marks which effects are strictly implied:
true in essentially every case where the event happened as stated. Only those are kept, in a
separate index (inference_rules_curated/) that recall.implicit.rules_dir can point to.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, Teacher, append_jsonl, items_of, read_jsonl  # noqa: E402

MAX_HEADS = 2500
PER_REQUEST = 16  # keeps the whole Phase 6 run under the free tier's 1,000 requests/day
SIMILARITY = 0.55

SYSTEM = """You judge commonsense inferences for a memory system that must never state guesses as facts.
For each event and each listed effect, decide: if the event happened exactly as stated, is the effect
CERTAINLY true (true in essentially every case, a definitional or unavoidable consequence)?
Likely, typical, emotional, intentional or future consequences are NOT certain.
Examples (none of these come from Recallic's evaluation sentences):
"PersonX gets a puppy" -> "has a dog": yes.   "PersonX books a flight" -> "flies": no (not yet / may cancel).
"PersonX wins the lottery" -> "receives money": yes.   "PersonX orders a pizza" -> "enjoys the pizza": no.
"PersonX breaks PersonX's arm" -> "is injured": yes.   "PersonX joins a choir" -> "becomes a good singer": no.
Return JSON {"keep": [[event number, effect number], ...]} listing only the certain effects."""


def main():
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from backend.brain.features.implicit import rule_paths

    heads_path, rules_path = rule_paths()
    meta = json.loads(Path(rules_path).read_text())
    rules = meta["rules"]
    heads = np.load(heads_path).astype(np.float32)
    model = SentenceTransformer(meta["embedder"])

    pool = [r["text"] for r in read_jsonl(DATA / "pool.jsonl")]
    vecs = model.encode(pool, batch_size=128, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    score = {}
    for sims in vecs @ heads.T:
        for i in np.argsort(-sims)[:3]:
            if sims[i] >= SIMILARITY:
                score[int(i)] = max(score.get(int(i), 0.0), float(sims[i]))
    chosen = sorted(score, key=lambda i: -score[i])[:MAX_HEADS]
    print(f"{len(score)} event patterns reached by the pool; curating {len(chosen)}", flush=True)

    out_path = DATA / "atomic_curation.jsonl"
    done = {r["head_index"] for r in read_jsonl(out_path)}
    todo = [i for i in chosen if i not in done]
    teacher = Teacher()
    for b in range(0, len(todo), PER_REQUEST):
        batch = todo[b:b + PER_REQUEST]
        lines, effects = [], {}
        for e, i in enumerate(batch):
            items = [(rel, tail) for rel, tails in rules[i]["rels"].items() for tail in tails]
            effects[e] = items
            lines.append(f"Event {e}: {rules[i]['head']}\n" + "\n".join(f"  {k}. [{rel}] {tail}" for k, (rel, tail) in enumerate(items)))
        res = teacher.chat(SYSTEM, "\n".join(lines), max_tokens=800) or {}
        keep = {(int(x[0]), int(x[1])) for x in items_of(res, "keep") if isinstance(x, list) and len(x) == 2
                and str(x[0]).isdigit() and str(x[1]).isdigit()}
        append_jsonl(out_path, [{"head_index": i, "head": rules[i]["head"],
                                 "kept": [list(effects[e][k]) for k in range(len(effects[e])) if (e, k) in keep],
                                 "total": len(effects[e]), "judged": res != {}} for e, i in enumerate(batch)])
        if (b // PER_REQUEST) % 25 == 0:
            print(f"  {len(done) + b + len(batch)}/{len(chosen)} patterns, {teacher.used} tokens", flush=True)

    # write the curated index: only patterns with at least one certain effect
    curated = [r for r in read_jsonl(out_path) if r["kept"]]
    total = sum(r["total"] for r in read_jsonl(out_path))
    kept = sum(len(r["kept"]) for r in curated)
    out_dir = Path(heads_path).parent.parent / "inference_rules_curated"
    out_dir.mkdir(parents=True, exist_ok=True)
    rules_out = []
    for r in curated:
        rels = {}
        for rel, tail in r["kept"]:
            rels.setdefault(rel, []).append(tail)
        rules_out.append({"head": r["head"], "rels": rels})
    np.save(out_dir / "heads.npy", heads[[r["head_index"] for r in curated]].astype(np.float16))
    (out_dir / "rules.json").write_text(json.dumps({**{k: v for k, v in meta.items() if k != "rules"},
                                                    "curated_by": f"teacher {teacher.model} (phase6/curate_atomic.py)",
                                                    "rules": rules_out}))
    print(f"kept {kept} of {total} effects ({len(curated)} patterns) -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
