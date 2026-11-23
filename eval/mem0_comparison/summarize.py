"""
Builds eval/results/mem0_comparison/summary.json (read by the website) from the raw comparison outputs:
report.json, answers.json, retrieval_conv-44.json and openai_spend_ledger.json. No number is typed by hand.

    python eval/mem0_comparison/summarize.py
"""
import json
import math
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "results" / "mem0_comparison"


def pctl(xs, f):
    xs = sorted(xs)
    return xs[int(f * (len(xs) - 1))]


def mcnemar(pairs):
    b = sum(1 for r, m in pairs if r and not m)
    c = sum(1 for r, m in pairs if m and not r)
    n = b + c
    p = min(1.0, sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) * 2 / 2 ** n) if n else 1.0
    return {"only_recallix_right": b, "only_mem0_right": c, "p": round(p, 3)}


def main():
    report = json.load(open(D / "report.json"))
    answers = json.load(open(D / "answers.json"))
    retr = json.load(open(D / "retrieval_conv-44.json"))
    ledger = json.load(open(D / "openai_spend_ledger.json"))

    pairs = {}
    for key, row in answers.items():
        system, conv, qi = key.split("|")
        pairs.setdefault((conv, qi), {})[system] = row
    both = [v for v in pairs.values() if len(v) == 2]
    cats = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
    significance = {"all": mcnemar([(v["recallix"]["correct"], v["mem0"]["correct"]) for v in both]),
                    "answerable": mcnemar([(v["recallix"]["correct"], v["mem0"]["correct"]) for v in both if v["recallix"]["category"] != 5])}
    counts = {cats[c]: sum(1 for v in both if v["recallix"]["category"] == c) for c in cats}

    rx, m0 = retr["recallix"], retr["mem0"]
    summary = {
        "date": "2026-11-24",
        "protocol": "eval/mem0_comparison/PROTOCOL.md",
        "scope": {"dataset": "LoCoMo test split", "conversations": report["convs"], "messages": rx["stored"],
                  "questions": len(both), "answerable": len(both) - counts["adversarial"], "adversarial": counts["adversarial"],
                  "per_category": counts, "not_judged": report.get("not_judged_mem0_incomplete", [])},
        "setup": {"mem0": "mem0ai 2.2.1 (nlp, extras), gpt-4o-mini, text-embedding-3-small",
                  "recallix": "as shipped 2026-11-24: bge-small embeddings, answer check, date resolution; no LLM",
                  "answer_and_judge": "gpt-4o-mini, temperature 0, identical for both"},
        "accuracy": {s: {k: report["systems"][s][k] for k in ("accuracy_overall", "accuracy_answerable", "adversarial_declined")}
                     | {"by_category": report["systems"][s]["by_category"]} for s in ("recallix", "mem0")},
        "significance": significance,
        "store": {"recallix": {"llm_calls": 0, "usd": 0.0, "ms_p50": pctl(rx["store_ms"], .5), "ms_p95": pctl(rx["store_ms"], .95)},
                  "mem0": {"llm_calls": ledger["by_purpose"]["mem0-store"]["calls"], "usd": round(ledger["by_purpose"]["mem0-store"]["usd"], 2),
                           "ms_p50": pctl(m0["store_ms"], .5), "ms_p95": pctl(m0["store_ms"], .95)}},
        "recall_ms": {"recallix": {"p50": pctl([v["ms"] for v in rx["search"].values()], .5), "p95": pctl([v["ms"] for v in rx["search"].values()], .95)},
                      "mem0": {"p50": pctl([v["ms"] for v in m0["search"].values()], .5), "p95": pctl([v["ms"] for v in m0["search"].values()], .95)}},
        "notes": ["Recallix timed in-process on an Apple M4; Mem0 times include OpenAI network calls.",
                  "Mem0 store cost is estimated from the API's token usage at list prices plus 10%.",
                  "A second conversation (conv-47) was stopped at 80 of 689 messages to stay under a $3 budget."],
        "openai_spend_total_usd": round(ledger["usd"], 2),
    }
    json.dump(summary, open(D / "summary.json", "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("accuracy", "significance", "store", "recall_ms")}, indent=1))


if __name__ == "__main__":
    main()
