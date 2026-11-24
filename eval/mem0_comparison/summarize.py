"""
Builds eval/results/mem0_comparison/summary.json (read by the website) from the raw comparison outputs.
No number is typed by hand.

    python eval/mem0_comparison/summarize.py

Inputs (eval/results/mem0_comparison/):
  conv-47/report.json, conv-47/answers.json, conv-47/retrieval_conv-47.json
      the held-out run (PROTOCOL.md amendment 5): Recallix, Recallix with deep recall, Mem0. The headline.
  report.json, answers.json, retrieval_conv-44.json
      the first run on conv-44. conv-44 was inspected question by question afterwards while designing the
      2026-11-24 retrieval fixes, so it is development data now and is reported as the earlier run.
  openai_spend_ledger_before_conv-47.json, openai_spend_ledger.json
      the spend ledger before and after the conv-47 run. Mem0's storage cost is the difference: messages 81-689
      of conv-47, stored in one uninterrupted run with nothing else calling the API. (The earlier ledger total
      also covers a calibration run, an aborted start and conv-47's first 80 messages, so it is not a cost for
      conv-44 alone.)
"""
import json
import math
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "results" / "mem0_comparison"
CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
ACC_KEYS = ("accuracy_overall", "accuracy_answerable", "adversarial_declined")


def pctl(xs, f):
    xs = sorted(xs)
    return xs[int(f * (len(xs) - 1))]


def mcnemar(pairs):
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    n = b + c
    p = min(1.0, sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) * 2 / 2 ** n) if n else 1.0
    return {"only_recallix_right": b, "only_mem0_right": c, "p": round(p, 3)}


def run_summary(folder, conv, systems):
    report = json.load(open(folder / "report.json"))
    answers = json.load(open(folder / "answers.json"))
    retr = json.load(open(folder / f"retrieval_{conv}.json"))
    rows = {}
    for key, row in answers.items():
        system, c, qi = key.split("|")
        if c == conv and system in systems:
            rows.setdefault(qi, {})[system] = row
    full = [v for v in rows.values() if len(v) == len(systems)]
    counts = {CATS[c]: sum(1 for v in full if v["mem0"]["category"] == c) for c in CATS}

    def sig(x, filt=lambda v: True):
        return mcnemar([(v[x]["correct"], v["mem0"]["correct"]) for v in full if filt(v)])
    answerable = lambda v: v["mem0"]["category"] != 5  # noqa: E731
    return {
        "scope": {"dataset": "LoCoMo test split", "conversations": [conv], "messages": retr["recallix"]["stored"],
                  "questions": len(full), "answerable": len(full) - counts["adversarial"],
                  "adversarial": counts["adversarial"], "per_category": counts},
        "accuracy": {s: {k: report["systems"][s][k] for k in ACC_KEYS} | {"by_category": report["systems"][s]["by_category"]}
                     for s in systems},
        "significance": {s: {"all": sig(s), "answerable": sig(s, answerable)} for s in systems if s != "mem0"},
        "recall_ms": {s: {"p50": pctl([v["ms"] for v in retr[s]["search"].values()], .5),
                          "p95": pctl([v["ms"] for v in retr[s]["search"].values()], .95)} for s in systems},
        "store_ms": {s: {"p50": pctl(retr[s]["store_ms"], .5), "p95": pctl(retr[s]["store_ms"], .95)}
                     for s in ("recallix", "mem0")},
        "_retr": retr,
    }


def main():
    head = run_summary(D / "conv-47", "conv-47", ("recallix", "recallix_deep", "mem0"))
    early = run_summary(D, "conv-44", ("recallix", "mem0"))
    before = json.load(open(D / "openai_spend_ledger_before_conv-47.json"))["by_purpose"]["mem0-store"]
    ledger = json.load(open(D / "openai_spend_ledger.json"))
    after = ledger["by_purpose"]["mem0-store"]
    retr = head.pop("_retr"); early.pop("_retr")
    resumed = retr["mem0"].get("store_ms_resumed_from", 0)
    measured = retr["mem0"]["stored"] - resumed
    usd, calls = after["usd"] - before["usd"], after["calls"] - before["calls"]
    mem0_store_ms = retr["mem0"]["store_ms"][resumed:]
    msgs = head["scope"]["messages"]
    cost = {"measured_on": f"conv-47 messages {resumed + 1}-{retr['mem0']['stored']}, one uninterrupted run",
            "messages": measured, "llm_calls": calls, "usd": round(usd, 3),
            "usd_per_message": round(usd / measured, 5), "llm_calls_per_message": round(calls / measured, 2)}
    summary = {
        "date": "2026-11-24",
        "protocol": "eval/mem0_comparison/PROTOCOL.md",
        "headline_conversation": "conv-47",
        "scope": head["scope"],
        "setup": {"mem0": "mem0ai 2.2.1 (nlp, extras), gpt-4o-mini, text-embedding-3-small",
                  "recallix": "as shipped on the evening of 2026-11-24: bge-small embeddings, answer check, date resolution, "
                              "second pass; no LLM. Deep recall: the same with deep=True",
                  "answer_and_judge": "gpt-4o-mini, temperature 0, identical for all systems"},
        "accuracy": head["accuracy"],
        "significance": head["significance"],
        # Recallix: measured. Mem0: its measured per-message rate (mem0_store_cost) times this conversation's messages.
        "store": {"recallix": {"llm_calls": 0, "usd": 0.0, "ms_p50": head["store_ms"]["recallix"]["p50"],
                               "ms_p95": head["store_ms"]["recallix"]["p95"]},
                  "mem0": {"llm_calls": round(calls / measured * msgs), "usd": round(usd / measured * msgs, 2),
                           "ms_p50": pctl(mem0_store_ms, .5), "ms_p95": pctl(mem0_store_ms, .95), "estimated_for_messages": msgs}},
        "mem0_store_cost": cost,
        "recall_ms": head["recall_ms"],
        "earlier_run": early | {"ledger_total_then": {"calls": before["calls"], "usd": round(before["usd"], 3)}, "note": "conv-44, the first run (Recallix as shipped on the morning of 2026-11-24). conv-44 was "
                                        "inspected question by question afterwards while designing that day's retrieval fixes, "
                                        "so it is development data now."},
        "notes": ["Recallix timed in-process on an Apple M4; Mem0 times include OpenAI network calls.",
                  "Mem0 store cost is estimated from the API's token usage at list prices plus 10%.",
                  "A decline counts as \"I don't know\": correct on adversarial questions, wrong on answerable ones."],
        "openai_spend_total_usd": round(ledger["usd"], 2),
    }
    json.dump(summary, open(D / "summary.json", "w"), indent=1)
    (D / "RESULTS.md").write_text(results_md(summary))
    print(json.dumps({k: summary[k] for k in ("accuracy", "significance", "store", "mem0_store_cost", "recall_ms")}, indent=1))
    print("earlier run:", json.dumps({k: summary["earlier_run"][k] for k in ("accuracy", "significance")})[:600])


def results_md(s):
    """The human-readable write-up, generated from summary.json (no number typed by hand)."""
    pc = lambda x: f"{x * 100:.1f}%"  # noqa: E731
    a, sc, st, c, e = s["accuracy"], s["scope"], s["store"], s["mem0_store_cost"], s["earlier_run"]
    rows = [("Accuracy, all questions", "accuracy_overall"), ("Accuracy, answerable", "accuracy_answerable"),
            ("Adversarial questions declined", "adversarial_declined")]
    out = [f"# Recallix vs Mem0 on LoCoMo: results ({s['date']})", "",
           f"Protocol: {s['protocol']} (written before any run; every amendment logged before the answers it concerns).",
           "Generated by eval/mem0_comparison/summarize.py from the raw files in this folder.", "",
           f"## Held-out run: {', '.join(sc['conversations'])}", "",
           f"{sc['messages']} messages, {sc['questions']} questions ({sc['answerable']} answerable, {sc['adversarial']} adversarial). "
           "Never inspected while developing Recallix.", "",
           f"- **Mem0:** {s['setup']['mem0']}", f"- **Recallix:** {s['setup']['recallix']}",
           f"- **Answering and judging:** {s['setup']['answer_and_judge']}", "",
           "| | Recallix | Recallix, deep recall | Mem0 |", "|---|---|---|---|"]
    out += [f"| {label} | {pc(a['recallix'][k])} | {pc(a['recallix_deep'][k])} | {pc(a['mem0'][k])} |" for label, k in rows]
    out += [f"| {cat} ({sc['per_category'][cat]}) | {pc(a['recallix']['by_category'][cat])} | {pc(a['recallix_deep']['by_category'][cat])} "
            f"| {pc(a['mem0']['by_category'][cat])} |" for cat in ("single-hop", "temporal", "multi-hop", "open-domain")]
    out += [f"| LLM calls to store the conversation | 0 | 0 | ≈ {st['mem0']['llm_calls']:,} |",
            f"| Cost to store the conversation | $0 | $0 | ≈ ${st['mem0']['usd']:.2f} |",
            f"| Store, median / p95 per message | {st['recallix']['ms_p50']:.0f} / {st['recallix']['ms_p95']:.0f} ms | same | "
            f"{st['mem0']['ms_p50']:,.0f} / {st['mem0']['ms_p95']:,.0f} ms |",
            "| Recall, median / p95 per question | " + " | ".join(
                f"{s['recall_ms'][x]['p50']:.0f} / {s['recall_ms'][x]['p95']:.0f} ms" for x in ("recallix", "recallix_deep", "mem0")) + " |", ""]
    sig = s["significance"]
    out += ["**Significance (paired, exact McNemar):**",
            *[f"- {name}: right where Mem0 was wrong on {sig[x]['all']['only_recallix_right']}, the reverse on "
              f"{sig[x]['all']['only_mem0_right']}, p = {sig[x]['all']['p']} (answerable only: p = {sig[x]['answerable']['p']})."
              for name, x in (("Recallix", "recallix"), ("Recallix, deep recall", "recallix_deep"))],
            "- Neither difference is significant on one conversation: default recall ties, deep recall leads.", "",
            f"**Mem0's storage cost:** measured on {c['measured_on']}: {c['llm_calls']:,} LLM calls and ${c['usd']:.3f} for "
            f"{c['messages']} messages (${c['usd_per_message']:.5f} and {c['llm_calls_per_message']} calls per message, "
            "estimated from token usage at list prices plus 10%). The conversation figures above are that rate times its messages.", "",
            f"## Earlier run: {', '.join(e['scope']['conversations'])}", "", e["note"], "",
            "| | Recallix | Mem0 |", "|---|---|---|"]
    out += [f"| {label} | {pc(e['accuracy']['recallix'][k])} | {pc(e['accuracy']['mem0'][k])} |" for label, k in rows]
    es = e["significance"]["recallix"]["all"]
    out += ["", f"{e['scope']['questions']} questions; Recallix alone right on {es['only_recallix_right']}, Mem0 alone on "
                f"{es['only_mem0_right']}, p = {es['p']}. An earlier version of this file gave Mem0's storage cost as the "
                f"ledger's running total at the end of this run ({e['ledger_total_then']['calls']:,} calls, ${e['ledger_total_then']['usd']:.2f}); that total also covered a calibration "
                "run, an aborted start and conv-47's first 80 messages, so it overstated the cost of this conversation.", "",
            "## Notes", "", *[f"- {x}" for x in s["notes"]], "",
            f"**Total OpenAI spend for the comparison:** ${s['openai_spend_total_usd']:.2f} (openai_spend_ledger.json).", ""]
    return "\n".join(out)


if __name__ == "__main__":
    main()
