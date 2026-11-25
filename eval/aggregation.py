"""
Aggregation dev set: questions whose answer is spread over several dated messages.

  counting      "How many weddings did I go to this year?"        (every attended one is evidence)
  time_between  "How many days after the marathon did I fly out?"  (two dated events)
  ordering      "Which did I visit first, Hampi or Gokarna?"
  preference    "Any podcast you'd recommend for me?"              (stated tastes, some changed)

Each scenario also has traps: questions that look answerable but are not (never mentioned, only planned,
someone else's event), which should be declined.

Why: on LongMemEval, most answerable questions Recallix declined with the evidence already in its top 10
were counting or ordering questions. Fixes are designed on THIS set (dev) and LoCoMo dev; LongMemEval is
run once at the end. The scenarios are written by an open-weight model (Groq, openai/gpt-oss-120b,
Apache 2.0) from a task description only: it never sees Recallix's code, rules or results.

  python eval/aggregation.py draft  [--per-category 18] [--only counting]   # needs GROQ_API_KEY in .env
  python eval/aggregation.py check                                         # answer-key checks, writes a report
  python eval/aggregation.py run    --embedder st:BAAI/bge-small-en-v1.5 [--keep-on-decline] [--deep] [--json out.json]

Files (eval/data/dev/aggregation/): drafts.jsonl (raw model output, never edited), review.jsonl (one
verdict per scenario: accept / fix / drop, with the reason), scenarios.jsonl (the set that is scored).
"""
import argparse
import contextlib
import io
import json
import re
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))
FOLDER = REPO / "eval" / "data" / "dev" / "aggregation"
DRAFTS, REVIEW, SCENARIOS = FOLDER / "drafts.jsonl", FOLDER / "review.jsonl", FOLDER / "scenarios.jsonl"

CATEGORIES = ["counting", "time_between", "ordering", "preference"]

# One persona per scenario. Varied on purpose: names, countries, ages, jobs, writing styles.
PERSONAS = [
    "a 29-year-old software engineer in Bengaluru who texts in lowercase with abbreviations",
    "a 41-year-old secondary-school teacher in Leeds, two kids, writes full sentences",
    "a 34-year-old nurse in Manila working rotating shifts, short messages",
    "a 23-year-old design student in Lisbon, enthusiastic, lots of exclamation marks",
    "a 52-year-old small-restaurant owner in Mexico City, practical and brief",
    "a 37-year-old freelance photographer in Cape Town who travels often",
    "a 30-year-old product manager in Toronto, busy, writes in fragments",
    "a 45-year-old civil engineer in Pune, formal tone, careful with dates",
    "a 27-year-old PhD student in Berlin, dry humour",
    "a 60-year-old retired bank manager in Kolkata, takes up new hobbies",
    "a 33-year-old marketing consultant in Dubai, frequent flyer",
    "a 26-year-old junior doctor in Nairobi, long hours, terse",
    "a 39-year-old stay-at-home parent of three in Austin",
    "a 31-year-old data analyst in Singapore who is training for races",
    "a 48-year-old architect in Melbourne renovating an old house",
    "a 25-year-old barista and musician in Seoul",
    "a 36-year-old HR manager in Hyderabad planning a wedding season",
    "a 43-year-old pharmacist in Dublin who volunteers on weekends",
    "a 28-year-old game developer in Warsaw, writes in a casual, joking way",
    "a 55-year-old farmer in Punjab who has recently started using a phone assistant",
]

TOPICS = {
    "counting": ["weddings and family functions attended", "books finished", "doctor and dentist visits",
                 "runs and races completed", "trips taken", "concerts and gigs", "job interviews",
                 "plants bought for the balcony", "online courses completed", "restaurants tried",
                 "home repairs done", "museum visits", "board-game nights hosted", "flights taken",
                 "times the car went to the mechanic", "birthday gifts bought", "volunteer shifts",
                 "new recipes cooked"],
    "time_between": ["a race and the recovery run after it", "buying a car and its first service", "a job offer and the start date",
                     "a flight out and the flight home", "planting seeds and the first harvest", "an injury and going back to the gym",
                     "adopting a pet and its first vet visit", "signing a lease and moving in", "an exam and getting the results",
                     "booking a holiday and leaving for it", "ordering furniture and its delivery", "two dentist appointments",
                     "starting a course and finishing it", "a first date and meeting the parents", "a phone breaking and the new one arriving",
                     "applying for a visa and receiving it", "starting a diet and the next check-up", "two haircuts"],
    "ordering": ["cities visited on a long trip", "books read this year", "jobs held", "courses taken", "concerts attended",
                 "home renovation steps", "pets adopted", "cars owned", "places lived", "hobbies started", "races run",
                 "doctors seen about one problem", "restaurants tried with friends", "gadgets bought", "flats viewed",
                 "languages started", "festivals attended", "projects shipped at work"],
    "preference": ["podcasts", "cooking and diet", "holiday destinations", "books and genres", "workout style",
                   "coffee and tea", "music for working", "films and series", "restaurants", "gifts for a partner",
                   "phone and laptop brands", "weekend activities", "clothing style", "board games", "hotels and stays",
                   "skincare", "commute options", "learning resources"],
}

SYSTEM = ("You write realistic test data for evaluating a personal assistant's long-term memory. "
          "Answer with one JSON object only, exactly in the requested shape.")

COMMON = """Write the chat history of ONE person with their personal assistant app, plus test questions about it.

Person: {persona}
Theme: {topic}

The messages (14 to 22):
- Things this person told the assistant over 3 to 9 months, in their own voice, one message each, oldest first.
- Every message has the date it was sent (YYYY-MM-DD, between 2025-01-01 and 2026-10-31).
- When a message describes something that happened, it happened ON the day the message was sent
  ("just got back from...", "today I..."). Do not write relative dates like "last Tuesday" or "two weeks ago".
- Mix in ordinary chatter that is not about the theme (work, weather, errands, family), about a third of the messages.
- Include near-misses that a careless reader would confuse with the real answer: {near_misses}

The questions (4 to 6), each asked on a date after the last message ("asked_on"):
- At least 3 answerable questions of this kind: {question_kind}
- At least 1 trap: a question that sounds like the others but cannot be answered from the messages
  (the thing was never mentioned, or only planned and never done, or it happened to someone else).
  A trap has "answer": null and "evidence": [].
- "evidence" lists the indexes (0-based) of EVERY message needed to answer. Nothing else.
- "answer" is short: a number with its unit, a name, or an ordered list "A, B, C".
- Phrase questions the way a real person asks; do not quote the messages word for word.

Shape:
{{"messages": [{{"date": "YYYY-MM-DD", "text": "..."}}],
  "questions": [{{"q": "...", "asked_on": "YYYY-MM-DD", "answer": "..." or null, "evidence": [0, 3],
                 "trap": false, "why": "one line: how the answer follows from the evidence"}}]}}"""

KIND = {
    "counting": ("How many times / how many X did I ...? Count only what this person actually did; the answer is "
                 "the number of evidence messages, each describing a different occurrence.",
                 "an occurrence that was planned but cancelled, one done by a friend or relative, one of a "
                 "similar but different kind, and one occurrence mentioned twice (a follow-up message about the "
                 "same event, which must be counted once and is NOT evidence)."),
    "time_between": ("How many days / weeks / months between two of this person's events, or how long after X did Y "
                     "happen. The answer follows from the two messages' dates.",
                     "a similar event of a friend, a plan for the event that is later rescheduled to another "
                     "date (only the day it actually happened counts), and a third event of the same kind."),
    "ordering": ("Which came first, A or B? / In what order did I ...? The answer follows from the messages' dates.",
                 "an item that was considered but never done, someone else's item, and an item from a "
                 "different category with a similar name."),
    "preference": ("Recommend / suggest something for me, where a good answer must respect this person's stated "
                   "likes, dislikes and constraints. The answer is a short description of what fits "
                   "(e.g. 'vegetarian, mild spice, near Koramangala').",
                   "a preference that later changed (the newer one wins; the old message is NOT evidence), a "
                   "friend's taste, and a one-off exception (\"I don't usually like X but this one was ok\")."),
}


# ---------------------------------------------------------------- draft
def draft(args):
    from phase6.common import Teacher, append_jsonl, read_jsonl
    FOLDER.mkdir(parents=True, exist_ok=True)
    done = {r["name"] for r in read_jsonl(DRAFTS)}
    teacher = Teacher()
    cats = args.only.split(",") if args.only else CATEGORIES
    made = 0
    for cat in cats:
        for i in range(args.per_category):
            name = f"agg_{cat}_{i:02d}"
            if name in done:
                continue
            persona = PERSONAS[(i * 7 + CATEGORIES.index(cat) * 3) % len(PERSONAS)]
            topic = TOPICS[cat][i % len(TOPICS[cat])]
            question_kind, near = KIND[cat]
            prompt = COMMON.format(persona=persona, topic=topic, question_kind=question_kind, near_misses=near)
            res = teacher.chat(SYSTEM, prompt, max_tokens=6000, temperature=0.8)
            if not isinstance(res, dict) or not res.get("messages"):
                print(f"  {name}: no usable answer, skipped (run again to retry)", flush=True)
                continue
            append_jsonl(DRAFTS, [{"name": name, "category": cat, "persona": persona, "topic": topic,
                                   "model": teacher.model, "drafted_at": time.strftime("%Y-%m-%d"), **res}])
            made += 1
            print(f"  {name}: {len(res['messages'])} messages, {len(res.get('questions', []))} questions "
                  f"({teacher.used} tokens so far)", flush=True)
    print(f"drafted {made} new scenarios; {len(done) + made} in {DRAFTS.relative_to(REPO)}")


# ---------------------------------------------------------------- check
def _day(s):
    try:
        return date.fromisoformat(str(s)[:10])
    except ValueError:
        return None


_NUM_WORDS = {w: n for n, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}


def _number(text):
    """First number in an answer ("3", "three times", "about 12 days") and its unit, or (None, None)."""
    t = str(text or "").lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*([a-z]*)", t)
    if m:
        return float(m.group(1)), m.group(2)
    for w, n in _NUM_WORDS.items():
        m = re.search(rf"\b{w}\b\s*([a-z]*)", t)
        if m:
            return float(n), m.group(1)
    return None, None


def problems(s):
    """Machine-checkable problems with one scenario's answer key. A human still reads every scenario."""
    out = []
    msgs, qs = s.get("messages") or [], s.get("questions") or []
    days = [_day(m.get("date")) for m in msgs]
    if not 10 <= len(msgs) <= 30:
        out.append(f"{len(msgs)} messages")
    if any(d is None for d in days):
        out.append("a message date does not parse")
    elif days != sorted(days):
        out.append("messages are not in date order")
    if not any(q.get("trap") for q in qs):
        out.append("no trap question")
    if sum(not q.get("trap") for q in qs) < 3:
        out.append("fewer than 3 answerable questions")
    for qi, q in enumerate(qs):
        tag = f"q{qi} {q.get('q', '')[:50]!r}"
        ev = q.get("evidence") or []
        if any(not isinstance(i, int) or not 0 <= i < len(msgs) for i in ev):
            out.append(f"{tag}: evidence index out of range")
            continue
        asked = _day(q.get("asked_on"))
        if asked is None or (days and all(days) and asked < max(days)):
            out.append(f"{tag}: asked_on missing or before the last message")
        if q.get("trap"):
            if q.get("answer") not in (None, "") or ev:
                out.append(f"{tag}: trap with an answer or evidence")
            continue
        if not ev or q.get("answer") in (None, ""):
            out.append(f"{tag}: answerable but no answer or evidence")
            continue
        n, unit = _number(q["answer"])
        cat = s.get("category")
        if cat == "counting":
            if n is None:
                out.append(f"{tag}: counting answer {q['answer']!r} has no number")
            elif int(n) != len(ev):
                out.append(f"{tag}: answer {q['answer']!r} but {len(ev)} evidence messages")
        elif cat == "time_between" and len(ev) == 2 and all(days[i] for i in ev):
            gap = abs((days[ev[1]] - days[ev[0]]).days)
            per = {"day": 1, "days": 1, "week": 7, "weeks": 7, "month": 30.44, "months": 30.44}.get(unit)
            if n is None or per is None:
                out.append(f"{tag}: time answer {q['answer']!r} has no number with day/week/month")
            elif abs(n * per - gap) > max(1.0, per / 2):
                out.append(f"{tag}: answer {q['answer']!r} but the dates are {gap} days apart")
        elif cat == "time_between" and len(ev) != 2:
            out.append(f"{tag}: time-between question with {len(ev)} evidence messages (expected 2)")
        elif cat == "ordering":
            ev_days = [days[i] for i in ev]
            if len(set(ev_days)) < len(ev_days):
                out.append(f"{tag}: two evidence messages on the same day, so the order is ambiguous")
    return out


def check(args):
    from phase6.common import read_jsonl
    rows = read_jsonl(args.drafts)
    if not rows:
        raise SystemExit(f"no drafts yet in {args.drafts}")
    by_cat = defaultdict(lambda: [0, 0])
    report = []
    for s in rows:
        p = problems(s)
        by_cat[s["category"]][0] += 1
        by_cat[s["category"]][1] += not p
        report.append({"name": s["name"], "problems": p})
        if p and args.verbose:
            print(f"{s['name']}:")
            for x in p:
                print(f"   - {x}")
    for cat, (n, ok) in sorted(by_cat.items()):
        print(f"{cat:13s} {ok}/{n} scenarios pass the automatic checks")
    out = Path(args.drafts).with_name("check_report.json")
    out.write_text(json.dumps(report, indent=1))
    print(f"report: {out} (automatic checks only; every scenario is also read by hand)")


# ---------------------------------------------------------------- run
def _ts(day):
    return datetime.combine(_day(day), datetime.min.time(), tzinfo=timezone.utc).timestamp() + 12 * 3600


def run(args):
    import numpy as np
    import run_eval
    from phase6.common import read_jsonl
    path = Path(args.scenarios)
    rows = read_jsonl(path)
    if not rows:
        raise SystemExit(f"no scenarios in {path}")
    run_eval.Env(args.embedder, args.engine_url)
    if args.config:
        import os
        os.environ["RECALLIX_CONFIG"] = str(Path(args.config).resolve())
        from backend import settings
        settings.reload()
    from backend.api.deps import get_memory_manager, reset
    reset()
    with contextlib.redirect_stdout(io.StringIO()):
        mm = get_memory_manager()

    recs = []
    for n, s in enumerate(rows):
        user = f"agg_{n}"
        mem_to_msg = defaultdict(list)
        with contextlib.redirect_stdout(io.StringIO()):
            for i, m in enumerate(s["messages"]):
                res = mm.store(user, m["text"], timestamp=_ts(m["date"]))
                mem_to_msg[res["memory_id"]].append(i)
        for q in s["questions"]:
            with contextlib.redirect_stdout(io.StringIO()):
                out = mm.retrieve(user, q["q"], limit=args.k, return_candidates=True,
                                  deep=True if args.deep else None, keep_on_decline=True if args.keep_on_decline else None)
            declined = out["status"] in ("ABSTAINED", "NO_MATCH")
            served = [i for m in out["memories"] for i in mem_to_msg[m["id"]]]
            ranked = [i for c in sorted(out["_candidates"], key=lambda c: -c["relevance"]) for i in mem_to_msg[c["id"]]][:args.k]
            ev = set(q.get("evidence") or [])
            chk = out["debug"].get("answer_check") or {}
            recs.append({"scenario": s["name"], "category": s["category"], "q": q["q"], "trap": bool(q.get("trap")),
                         "status": out["status"], "declined": declined,
                         "evidence": sorted(ev), "served": served, "ranked": ranked,
                         # share of the evidence messages among what was served (0 when declined), and in the ranked top k
                         "served_cover": (len(ev & set(served)) / len(ev) if ev and not declined else 0.0),
                         "ranked_cover": (len(ev & set(ranked)) / len(ev) if ev else 0.0),
                         "margin": chk.get("counted_margin"), "latency_ms": out["latency_ms"]})

    def summary(sel):
        ans = [r for r in sel if not r["trap"]]
        traps = [r for r in sel if r["trap"]]
        return {"answerable": len(ans), "traps": len(traps),
                "answerable_declined": round(float(np.mean([r["declined"] for r in ans])), 3) if ans else None,
                "all_evidence_served": round(float(np.mean([r["served_cover"] == 1.0 for r in ans])), 3) if ans else None,
                "evidence_served": round(float(np.mean([r["served_cover"] for r in ans])), 3) if ans else None,
                "all_evidence_ranked": round(float(np.mean([r["ranked_cover"] == 1.0 for r in ans])), 3) if ans else None,
                "traps_declined": round(float(np.mean([r["declined"] for r in traps])), 3) if traps else None}

    from backend.settings import get as settings_get
    result = {"dataset": str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path),
              "scenarios": len(rows), "embedder": args.embedder, "k": args.k,
              "deep_recall": bool(args.deep), "keep_on_decline": bool(args.keep_on_decline), "config": args.config,
              "recall_settings": {k: settings_get(f"recall.{k}") for k in ("second_pass", "rerank", "answer_check", "keep_results_on_decline")},
              "overall": summary(recs), "by_category": {c: summary([r for r in recs if r["category"] == c])
                                                         for c in CATEGORIES if any(r["category"] == c for r in recs)},
              "rows": recs}
    print(f"{'category':13s} {'ans':>4s} {'declined':>9s} {'all-ev served':>14s} {'ev served':>10s} {'all-ev top' + str(args.k):>12s} {'traps':>6s} {'traps declined':>15s}")
    for c, v in list(result["by_category"].items()) + [("overall", result["overall"])]:
        f = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
        print(f"{c:13s} {v['answerable']:4d} {f(v['answerable_declined']):>9s} {f(v['all_evidence_served']):>14s} "
              f"{f(v['evidence_served']):>10s} {f(v['all_evidence_ranked']):>12s} {v['traps']:6d} {f(v['traps_declined']):>15s}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=1))
        print(f"wrote {args.json}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draft")
    d.add_argument("--per-category", type=int, default=18)
    d.add_argument("--only", default=None, help="comma-separated categories")
    c = sub.add_parser("check")
    c.add_argument("--drafts", default=str(DRAFTS))
    c.add_argument("-v", "--verbose", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--scenarios", default=str(SCENARIOS))
    r.add_argument("--embedder", default="st:BAAI/bge-small-en-v1.5")
    r.add_argument("--engine-url", default=None)
    r.add_argument("--config", default=None, help="a config.yaml to use instead of the default one")
    r.add_argument("--k", type=int, default=10)
    r.add_argument("--deep", action="store_true")
    r.add_argument("--keep-on-decline", action="store_true")
    r.add_argument("--json", default=None)
    args = ap.parse_args()
    {"draft": draft, "check": check, "run": run}[args.cmd](args)


if __name__ == "__main__":
    main()
