"""
Build the message pool: user turns from public dialogue datasets + synthetic personal messages.

    PHASE6_DATA=<dir with taskmaster/ multiwoz/ sgd/> python phase6/build_pool.py

Writes pool.jsonl ({id, text, source, licence}) and splits.json (held-out ids). Benchmark
text (LoCoMo, LongMemEval) is never used, so those evaluations stay clean.
"""
import json
import random
import re
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import DATA, SOURCES, Teacher, append_jsonl, norm, read_jsonl  # noqa: E402

PER_SOURCE = {"taskmaster": 900, "multiwoz": 500, "sgd": 500}
SYNTHETIC_BATCHES = 60          # x 25 messages
HELD_OUT = 300
SEED = 13

TOPICS = [
    "family and relationships", "work, jobs and career changes", "where they live, moving house", "health, allergies, diet",
    "pets", "hobbies, sports and music", "food and drink preferences", "travel plans and trips", "school, courses, exams",
    "appointments and upcoming events with dates", "money, budgets and purchases", "friends and social life",
    "corrections of something said earlier (e.g. 'actually it moved to Friday')", "birthdays and anniversaries",
    "small talk and passing states that are NOT facts (tired, weather, what they're doing right now)",
]


def _ok(text):
    t = text.strip()
    return 20 <= len(t) <= 220 and not t.endswith("?") and re.search(r"[a-zA-Z]", t)


def from_taskmaster():
    for d in json.load(open(DATA / "taskmaster" / "self-dialogs.json")):
        for u in d["utterances"]:
            if u.get("speaker") == "USER" and _ok(u["text"]):
                yield u["text"].strip()


def from_turns(folder):
    for path in sorted((DATA / folder).glob("dialogues_*.json")):
        for d in json.load(open(path)):
            for t in d["turns"]:
                if t.get("speaker") == "USER" and _ok(t["utterance"]):
                    yield t["utterance"].strip()


SYNTH_SYSTEM = ("You write realistic chat messages that people send to their personal AI assistant. "
                "Return JSON {\"messages\": [\"...\"]}. Vary length (5-35 words), names, places, cultures and style; "
                "first person; no numbering; no questions.")


def synthetic(teacher, n_batches):
    """Generated in batches, each saved immediately (synthetic.jsonl), so an interrupted run resumes."""
    path = DATA / "synthetic.jsonl"
    done = read_jsonl(path)
    for b in range(len(done), n_batches):
        topic = TOPICS[b % len(TOPICS)]
        res = teacher.chat(SYNTH_SYSTEM, f"Write 25 different messages about: {topic}.", max_tokens=2500, temperature=1.0) or {}
        msgs = [m.strip() for m in res.get("messages", []) if isinstance(m, str) and _ok(m)]
        append_jsonl(path, [{"batch": b, "topic": topic, "messages": msgs}])
        print(f"  synthetic batch {b + 1}/{n_batches}: {len(msgs)} messages ({topic})", flush=True)
    out = [m for row in read_jsonl(path) for m in row["messages"]]
    random.Random(SEED).shuffle(out)
    return out


def main():
    rng = random.Random(SEED)
    pool_path = DATA / "pool.jsonl"
    if pool_path.exists():
        print("pool exists:", pool_path)
        return
    rows, seen = [], set()

    def add(texts, source, k):
        texts = list(dict.fromkeys(texts))
        rng.shuffle(texts)
        taken = 0
        for t in texts:
            if norm(t) in seen:
                continue
            seen.add(norm(t))
            rows.append({"id": f"{source}-{taken}", "text": t, "source": source, "licence": SOURCES[source]})
            taken += 1
            if taken >= k:
                break
        print(f"{source}: {taken}")

    add(from_taskmaster(), "taskmaster", PER_SOURCE["taskmaster"])
    add(from_turns("multiwoz"), "multiwoz", PER_SOURCE["multiwoz"])
    add(from_turns("sgd"), "sgd", PER_SOURCE["sgd"])
    batches = int(__import__("os").environ.get("SYNTHETIC_BATCHES", SYNTHETIC_BATCHES))
    add(synthetic(Teacher(), batches), "synthetic", batches * 25)

    rng.shuffle(rows)
    append_jsonl(pool_path, rows)
    held = [r["id"] for r in rows[:HELD_OUT]]
    (DATA / "splits.json").write_text(json.dumps({"held_out": held, "seed": SEED}, indent=1))
    print(f"pool: {len(rows)} messages; held out {len(held)}")


if __name__ == "__main__":
    main()
