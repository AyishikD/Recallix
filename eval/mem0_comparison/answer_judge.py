"""
Answers and judges both systems identically (PROTOCOL.md "Answering and judging"). Mem0 environment (has openai).

    python answer_judge.py --data data.json --out OUT --ledger ledger.json --convs conv-44,conv-47,...

Reads OUT/recallix.json and OUT/mem0_<conv>.json; writes OUT/answers.json (resumable) and OUT/report.json.
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import budget  # noqa: E402

MODEL = "gpt-4o-mini"
ANSWER_SYSTEM = ("You answer questions about a user's conversations using only the memories provided. If the memories "
                 "do not contain the answer, reply exactly: I don't know. Otherwise answer in a short phrase.")
JUDGE = ("Question: {q}\nGold answer: {gold}\nGenerated answer: {gen}\n\nIs the generated answer correct, i.e. it states "
         "the same fact as the gold answer (dates and numbers may be written differently)? Reply CORRECT or WRONG.")
IDK = "I don't know"
CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def declines(ans):
    return bool(re.match(r"\s*i (?:don'?t|do not) know", str(ans or ""), re.I))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--convs", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    budget.install(args.ledger)
    from openai import OpenAI
    client = OpenAI()

    convs = args.convs.split(",")
    alldata = {c["conv"]: c for c in json.load(open(args.data))}
    # Same scope for both systems: only conversations Mem0 fully stored and searched (PROTOCOL.md amendment 4).
    def mem0_complete(c):
        p = Path(args.out) / f"mem0_{c}.json"
        if not p.exists():
            return False
        ck = json.load(open(p))
        return ck["stored"] == len(alldata[c]["turns"]) and len(ck["search"]) == len(alldata[c]["qa"])
    skipped = [c for c in convs if not mem0_complete(c)]
    convs = [c for c in convs if c not in skipped]
    if skipped:
        print(f"not judged (Mem0 incomplete): {skipped}")
    if not convs:
        print("nothing to judge"); return
    data = {c: alldata[c] for c in convs}
    rx = json.load(open(Path(args.out) / "recallix.json"))
    systems = {"recallix": {c: rx[c]["search"] for c in convs},
               "mem0": {c: json.load(open(Path(args.out) / f"mem0_{c}.json"))["search"] for c in convs}}
    deep_path = Path(args.out) / "recallix_deep.json"   # optional third system: Recallix with deep recall
    if deep_path.exists():
        rxd = json.load(open(deep_path))
        systems["recallix_deep"] = {c: rxd[c]["search"] for c in convs}
    ans_path = Path(args.out) / "answers.json"
    answers = json.load(open(ans_path)) if ans_path.exists() else {}

    def chat(purpose, messages):
        budget.PURPOSE["value"] = purpose
        r = client.chat.completions.create(model=MODEL, temperature=0, messages=messages)
        return r.choices[0].message.content.strip()

    def one(key, system, conv, qi):
        qa = data[conv]["qa"][int(qi)]
        hit = systems[system][conv].get(qi)
        if hit is None:
            return key, None
        mems = hit["memories"] if hit.get("status", "CONSISTENT") not in ("ABSTAINED", "NO_MATCH") else []
        if not mems:
            gen = IDK
        else:
            user = "Memories:\n" + "\n".join(f"- {m}" for m in mems) + f"\n\nQuestion: {qa['question']}\nAnswer:"
            gen = chat("answer", [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user}])
        if qa["category"] == 5:
            correct = declines(gen)
        elif declines(gen):
            correct = False
        else:
            verdict = chat("judge", [{"role": "user", "content": JUDGE.format(q=qa["question"], gold=qa["answer"], gen=gen)}])
            correct = verdict.upper().startswith("CORRECT")
        return key, {"system": system, "conv": conv, "qi": qi, "category": qa["category"], "answer": gen, "correct": correct}

    jobs = [(f"{s}|{c}|{qi}", s, c, qi) for s in systems for c in convs for qi in map(str, range(len(data[c]["qa"])))
            if f"{s}|{c}|{qi}" not in answers]
    stopped = None
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(one, *j) for j in jobs]
        for n, f in enumerate(futures, 1):
            try:
                key, row = f.result()
                if row is not None:
                    answers[key] = row
            except budget.BudgetExceeded as e:
                stopped = str(e)
            except Exception as e:   # one failed question is left unanswered (retried on the next run), not fatal
                print(f"question failed, will retry on resume: {type(e).__name__}: {str(e)[:120]}")
            if n % 100 == 0:
                json.dump(answers, open(ans_path, "w")); print(f"answered {len(answers)}", flush=True)
    json.dump(answers, open(ans_path, "w"))

    report = {"convs": convs, "not_judged_mem0_incomplete": skipped, "complete": stopped is None, "stopped": stopped, "systems": {}}
    for s in systems:
        rows = [r for r in answers.values() if r["system"] == s]
        by = lambda f: [r for r in rows if f(r)]
        ansable = by(lambda r: r["category"] != 5); adv = by(lambda r: r["category"] == 5)
        report["systems"][s] = {
            "questions": len(rows),
            "accuracy_overall": round(float(np.mean([r["correct"] for r in rows])), 4) if rows else None,
            "accuracy_answerable": round(float(np.mean([r["correct"] for r in ansable])), 4) if ansable else None,
            "adversarial_declined": round(float(np.mean([r["correct"] for r in adv])), 4) if adv else None,
            "answerable_said_idk": round(float(np.mean([declines(r["answer"]) for r in ansable])), 4) if ansable else None,
            "by_category": {CATS[c]: round(float(np.mean([r["correct"] for r in rows if r["category"] == c])), 4)
                            for c in CATS if any(r["category"] == c for r in rows)},
        }
    report["ledger"] = json.load(open(args.ledger))
    json.dump(report, open(Path(args.out) / "report.json", "w"), indent=1)
    print(json.dumps(report["systems"], indent=1)); print("spent $", report["ledger"]["usd"])


if __name__ == "__main__":
    main()
