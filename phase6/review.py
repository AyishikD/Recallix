"""
Second labelling pass: the teacher reviews and completes its own first-pass labels.

    python phase6/review.py                 # resumable; writes labels_reviewed.jsonl
    python phase6/review.py --only held_out # just the held-out answer key (first)

Why: a check of the held-out labels found good precision (~87%) but missed facts (recall
~73%), mostly hobbies, pets, likes, what an appointment is, and locations. This pass shows
each message with its first-pass facts and asks for the complete, corrected list. Values
must still be verbatim spans (checked here, as in label.py).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, Teacher, append_jsonl, indexed, items_of, read_jsonl  # noqa: E402
from label import SYSTEM as LABEL_SYSTEM, RELATIONS  # noqa: E402

BATCH = 10

REVIEW = LABEL_SYSTEM + """
You are now REVIEWING first-pass labels. For each message you get the message and the facts found so far.
Return the COMPLETE corrected fact list for every message (same JSON format). Check especially:
- ADD missed facts: hobbies and regular activities ("cycling on weekends" -> hobby "cycling"), pets (including "my kitten"),
  likes/dislikes and food preferences, possessions, what a plan or appointment is ("pet grooming appointment" -> plan),
  places the user is or wants to be ("I was thinking in Petaluma" -> location), course or school subjects.
- FIX relations: times and dates of plans are event_time (not plan); cars and bikes are vehicle (not owns);
  a venue for an event is NOT the user's location; a one-off order ("bacon and onions on my pizza") is a plan, not likes.
- FIX entities: the entity is who/what the fact is about ("the deadline for submissions" -> entity "submissions").
- REMOVE facts that are not stated or are passing states.
"""


def review_batch(teacher, items):
    """items: [(id, text, facts)] -> {id: corrected facts or None}."""
    blocks = "\n".join(f"{k}. {text}\n   found so far: {json.dumps([{x: f[x] for x in ('entity', 'relation', 'value')} for f in facts])}"
                       for k, (_, text, facts) in enumerate(items))
    res = teacher.chat(REVIEW, f"Messages:\n{blocks}", max_tokens=300 + 180 * len(items))
    if res is None:
        if len(items) == 1:
            return {items[0][0]: None}
        half = len(items) // 2
        return {**review_batch(teacher, items[:half]), **review_batch(teacher, items[half:])}
    by_index = indexed(items_of(res, "items"))
    out = {}
    for k, (mid, text, old) in enumerate(items):
        if k not in by_index:
            out[mid] = old  # not returned: keep the first pass
            continue
        facts = []
        for f in by_index[k] or []:
            if not isinstance(f, dict):
                continue
            rel, val, ent = f.get("relation"), str(f.get("value", "")).strip(), str(f.get("entity", "user")).strip() or "user"
            start = text.find(val) if val else -1
            if rel in RELATIONS and start >= 0:
                facts.append({"entity": ent, "relation": rel, "value": val, "start": start, "end": start + len(val)})
        out[mid] = facts
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["held_out"], default=None)
    args = ap.parse_args()
    pool = {r["id"]: r for r in read_jsonl(DATA / "pool.jsonl")}
    first = read_jsonl(DATA / "labels.jsonl")
    out_path = DATA / "labels_reviewed.jsonl"
    done = {r["id"] for r in read_jsonl(out_path)}
    todo = [r for r in first if r["id"] not in done]
    if args.only:
        keep = set(json.loads((DATA / "splits.json").read_text())[args.only])
        todo = [r for r in todo if r["id"] in keep]
    teacher = Teacher()
    print(f"{len(done)} reviewed, {len(todo)} to go", flush=True)
    for b in range(0, len(todo), BATCH):
        batch = todo[b:b + BATCH]
        fixed = review_batch(teacher, [(r["id"], pool[r["id"]]["text"], r["facts"]) for r in batch])
        append_jsonl(out_path, [{"id": r["id"], "facts": fixed[r["id"]] if fixed.get(r["id"]) is not None else r["facts"],
                                 "first_pass": len(r["facts"])} for r in batch])
        if (b // BATCH) % 10 == 0:
            print(f"  {len(done) + b + len(batch)}/{len(first)} reviewed, {teacher.used} tokens", flush=True)
    print("done; tokens used:", teacher.used)


if __name__ == "__main__":
    main()
