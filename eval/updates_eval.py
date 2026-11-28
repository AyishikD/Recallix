"""
Updates without cue words (development data, eval/data/dev/updates/scenarios.jsonl).

    python eval/updates_eval.py OUT.json [--embedder st:BAAI/bge-small-en-v1.5]

Each scenario stores a few first-person messages a day apart under its own user and asks one question. The
top-level `answer` is scored:
  correct   contains an expected value and none of the forbidden ones
  wrong     contains a forbidden value (the outdated fact, or someone else's)
  declined  no top-level answer (or recall declined)
  other     an answer with neither
Groups: update (the later message replaces the earlier fact; the old value is forbidden), both_true (a later,
different fact must not replace the earlier one) and unrelated (a later fact about someone else must not replace
the user's own). Wrong answers are the number to keep at zero; declines are the cost of caution.
"""
import argparse
import collections
import contextlib
import io
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]

DAY0 = 1_767_225_600


def score(answer, must, must_not):
    a = (answer or "").lower()
    if not a:
        return "declined"
    if any(x in a for x in must_not):
        return "wrong"
    return "correct" if any(x in a for x in must) else "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--embedder", default="st:BAAI/bge-small-en-v1.5")
    ap.add_argument("--data", default=str(ROOT / "eval/data/dev/updates/scenarios.jsonl"))
    args = ap.parse_args()
    import run_eval
    env = run_eval.Env(args.embedder)  # noqa: F841 (keeps the engine and storage for the run)
    from backend.api.deps import get_memory_manager, reset
    reset()
    with contextlib.redirect_stdout(io.StringIO()):
        mm = get_memory_manager()
    rows = []
    for i, line in enumerate(open(args.data)):
        s = json.loads(line)
        user = f"upd-{i}"
        for d, msg in enumerate(s["messages"], 1):
            mm.store(user, msg, timestamp=DAY0 + d * 86400)
        with contextlib.redirect_stdout(io.StringIO()):
            out = mm.retrieve(user, s["question"], limit=5)
        top = (out.get("answer") or {}).get("text") if out.get("status") == "CONSISTENT" else None
        first = ((out.get("memories") or [{}])[0] or {}).get("answer") if out.get("status") == "CONSISTENT" else None
        rows.append({"name": s["name"], "group": s["group"], "question": s["question"], "status": out.get("status"),
                     "answer": top, "first_memory_answer": first,
                     "result": score(top, s["answer_contains_any"], s["answer_not_contains"]),
                     "first_memory_result": score(first, s["answer_contains_any"], s["answer_not_contains"])})
    summary = {}
    for g in ("update", "both_true", "unrelated", "all"):
        rs = [r for r in rows if g == "all" or r["group"] == g]
        summary[g] = dict(collections.Counter(r["result"] for r in rs), n=len(rs))
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
