"""
Label the pool with the teacher: durable facts, each value an exact span of the message.

    python phase6/label.py              # resumable; writes labels.jsonl next to pool.jsonl
    python phase6/label.py --trial      # 10 messages, printed, nothing saved

A fact is kept only if its value is found verbatim in the message, so every training span
is exact (and the student's predictions can be quoted in receipts).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RELATIONS, Teacher, append_jsonl, indexed, items_of, read_jsonl  # noqa: E402

BATCH = 10

SYSTEM = f"""You label facts in chat messages for a long-term memory system.
Return JSON: {{"items": [{{"i": <message number>, "facts": [{{"entity": "...", "relation": "...", "value": "..."}}]}}]}}
with one item per message (facts may be an empty list).

What counts as a fact: durable or useful-later information the user STATES — identity, where they live,
work and education, family and friends, pets, health, allergies, diet, likes and dislikes, possessions,
plans and appointments with concrete details (what, when, where, how many), deadlines, budgets.
Do NOT label:
- passing states or activities right now ("I'm tired", "my cat is sleeping", "I'm eating lunch");
- opinions about the conversation, greetings, thanks, questions;
- anything implied but not stated, or guessed.

relation: exactly one of {", ".join(RELATIONS)}.
  likes/dislikes = preferences; plan = what is planned or booked; event_time = when something planned happens;
  owns = possessions other than vehicles and pets.
entity: "user" for the speaker; otherwise the person or thing the fact is about, as written
  ("My sister lives in Delhi" -> entity "sister"; "Sam works at Google" -> entity "Sam").
value: copied EXACTLY from the message (same characters), the shortest span that carries the value
  ("I live in Pune" -> "Pune"; "table for 4 at 7pm on Friday" -> party_size "4", event_time "7pm on Friday").

Examples:
"My name is Priya and I live in Pune. My cat is sleeping." ->
  [{{"entity": "user", "relation": "name", "value": "Priya"}}, {{"entity": "user", "relation": "location", "value": "Pune"}}]
"I'm allergic to peanuts but I love dark chocolate." ->
  [{{"entity": "user", "relation": "allergy", "value": "peanuts"}}, {{"entity": "user", "relation": "likes", "value": "dark chocolate"}}]
"Ugh, what a long day." -> []
"""


def label_batch(teacher, messages):
    """messages: [(id, text)] -> {id: [facts with exact spans]}."""
    numbered = "\n".join(f"{k}. {text}" for k, (_, text) in enumerate(messages))
    res = teacher.chat(SYSTEM, f"Messages:\n{numbered}", max_tokens=200 + 150 * len(messages))
    if res is None:  # invalid JSON from the teacher: split the batch and retry the halves
        if len(messages) == 1:
            return {messages[0][0]: None}  # unlabelled (not "no facts"); skipped in training
        half = len(messages) // 2
        return {**label_batch(teacher, messages[:half]), **label_batch(teacher, messages[half:])}
    by_index = indexed(items_of(res, "items"))
    out = {}
    for k, (mid, text) in enumerate(messages):
        facts = []
        for f in by_index.get(k, []) or []:
            if not isinstance(f, dict):
                continue
            rel, val, ent = f.get("relation"), str(f.get("value", "")).strip(), str(f.get("entity", "user")).strip() or "user"
            start = text.find(val) if val else -1
            if rel not in RELATIONS or start < 0:
                continue  # unknown relation, or value not quoted verbatim: drop
            facts.append({"entity": ent, "relation": rel, "value": val, "start": start, "end": start + len(val)})
        out[mid] = facts
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", action="store_true")
    args = ap.parse_args(argv)
    teacher = Teacher()
    if args.trial:
        msgs = [("t0", "My name is Priya and I live in Pune. My cat is sleeping."),
                ("t1", "I need a table for 4 at an Italian place this Friday at 7pm."),
                ("t2", "My sister just moved to Toronto for her new job at Shopify."),
                ("t3", "I'm so tired today."),
                ("t4", "I'm vegetarian and allergic to shellfish."),
                ("t5", "Actually the dentist appointment moved to Thursday at 3."),
                ("t6", "We're flying to Lisbon on March 3rd, budget around 2000 euros."),
                ("t7", "My son Leo turns six on May 12."),
                ("t8", "Thanks, that's great!"),
                ("t9", "I drive a 2015 Honda Civic and I hate traffic.")]
        for mid, facts in label_batch(teacher, msgs).items():
            print(mid, dict(msgs)[mid])
            for f in facts:
                print(f"     {f['entity']}::{f['relation']} = {f['value']!r}")
        print("tokens used:", teacher.used)
        return 0

    pool = read_jsonl(DATA / "pool.jsonl")
    out_path = DATA / "labels.jsonl"
    done = {r["id"] for r in read_jsonl(out_path)}
    todo = [r for r in pool if r["id"] not in done]
    print(f"{len(done)} labelled, {len(todo)} to go", flush=True)
    for b in range(0, len(todo), BATCH):
        batch = todo[b:b + BATCH]
        labels = label_batch(teacher, [(r["id"], r["text"]) for r in batch])
        append_jsonl(out_path, [{"id": r["id"], "facts": labels.get(r["id"], [])} for r in batch
                                if labels.get(r["id"], []) is not None])
        if (b // BATCH) % 10 == 0:
            print(f"  {len(done) + b + len(batch)}/{len(pool)} labelled, {teacher.used} tokens", flush=True)
    print("done; tokens used:", teacher.used)
    return 0


if __name__ == "__main__":
    sys.exit(main())
