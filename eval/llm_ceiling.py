"""
LLM reference on the update scenarios: what a large language model answers when it reads all of a scenario's messages.

    python eval/llm_ceiling.py OUT.json --data eval/data/fresh/updates/gemini_upd1.jsonl [--data ...] [--model M]

Each scenario's messages are given in the order sent (day 1, day 2, ...) with the question; the model answers with a
short phrase from the messages, or "unknown". Scored like eval/updates_eval.py (correct / wrong / declined / other), so
the numbers compare with Recallic's. Uses the Phase 6 teacher client (Groq by default, TEACHER_* in .env).
"""
import argparse
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]

from phase6.common import Teacher  # noqa: E402
from updates_eval import score  # noqa: E402

SYSTEM = """You answer questions for a personal AI assistant from the user's own chat messages.
The messages were written by the user (I, my, we = the user) and are listed in the order they were sent.
Answer the question as it is true now, using only the messages. Answer with the shortest exact phrase from the
messages that answers it (a name, place, date, amount or item), not a sentence. If the messages do not answer it,
answer "unknown". Reply as JSON: {"answer": "..."}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--model", default=None, help="default: TEACHER_MODEL from .env")
    args = ap.parse_args()
    teacher = Teacher(model=args.model)
    rows = []
    for path in args.data:
        batch = pathlib.Path(path).stem
        for line in open(path):
            s = json.loads(line)
            msgs = "\n".join(f"Day {d}: {m}" for d, m in enumerate(s["messages"], 1))
            res = teacher.chat(SYSTEM, f"Messages:\n{msgs}\n\nQuestion: {s['question']}", max_tokens=600) or {}
            answer = str(res.get("answer") or "").strip()
            if answer.lower() in ("unknown", "none", "n/a", ""):
                answer = None
            rows.append({"batch": batch, "name": s["name"], "group": s["group"], "question": s["question"],
                         "answer": answer, "result": score(answer, s["answer_contains_any"], s["answer_not_contains"])})
            print(f"{rows[-1]['result']:8} {s['name']:32} {answer!r}", flush=True)
    summary = {}
    for g in ("update", "both_true", "unrelated", "all"):
        rs = [r for r in rows if g == "all" or r["group"] == g]
        summary[g] = dict(collections.Counter(r["result"] for r in rs), n=len(rs))
    json.dump({"model": teacher.model, "base_url": teacher.base, "tokens": teacher.used, "summary": summary,
               "rows": rows}, open(args.out, "w"), indent=1)
    print(json.dumps({"model": teacher.model, "tokens": teacher.used, **summary["all"]}))


if __name__ == "__main__":
    main()
