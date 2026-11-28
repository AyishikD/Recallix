"""
Collects measured results into eval/results/summary.json (read by the website's
benchmark page) and eval/results/RESULTS.md (linked from the README).

Inputs (all produced by tools in this repo):
  eval/results/{dev,test}_<embedder>.json   python eval/run_eval.py --split S --embedder E --only facts,schemas,importance,retrieval,legacy --json ...
  eval/results/hnsw.jsonl                    RESULT lines of backend/infra_cpp/build/benchmark_hnsw
  eval/results/hnsw_reference.jsonl          python eval/hnsw_reference.py ...

  python eval/publish_results.py
"""
import glob
import json
import os
import platform
import subprocess
import time
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def machine():
    info = {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version()}
    try:
        if platform.system() == "Darwin":
            info["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        else:
            info["cpu"] = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "")
    except Exception:
        info["cpu"] = platform.processor()
    return info


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def main():
    evals = {}
    for path in sorted(glob.glob(str(RESULTS / "*_*.json"))):
        name = Path(path).stem
        if name in ("summary",) or name.startswith(("locomo", "longmemeval", "gate_")):
            continue
        with open(path) as f:
            evals[name] = json.load(f)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=RESULTS).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "backend", "eval/run_eval.py", "config.yaml"],
                                capture_output=True, text=True, cwd=RESULTS.parent.parent).stdout.strip())
    if dirty:
        commit += " + uncommitted changes"
    summary = {"generated_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "git_commit": commit,
               "machine": machine(), "eval": evals,
               "hnsw": read_jsonl(RESULTS / "hnsw.jsonl"), "hnsw_reference": read_jsonl(RESULTS / "hnsw_reference.jsonl"),
               "locomo": [{**json.load(open(p)), "file": Path(p).stem} for p in sorted(glob.glob(str(RESULTS / "locomo_*.json")))],
               "longmemeval": [{**json.load(open(p)), "file": Path(p).stem}
                               for p in sorted(glob.glob(str(RESULTS / "longmemeval_*.json")))]}
    with open(RESULTS / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    md = [f"# Measured results\n", f"Generated {summary['generated_at']} from commit `{commit}` on "
          f"{summary['machine'].get('cpu') or summary['machine']['machine']} ({summary['machine']['system']}). "
          "Regenerate with the commands in `eval/README.md`.\n"]
    md.append("## Memory pipeline (eval harness)\n")
    md.append("| file | split | embedder | facts F1 | schema acc | importance AUC / acc@5 | retrieval hit@1 / hit@3 | false abstain | unanswerable declined | gauntlet | retention@1 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")

    def f2(x):  # "–" where a run did not measure something (e.g. retrieval-only runs)
        return "–" if x is None or (isinstance(x, float) and x != x) else f"{x:.2f}"
    for name, r in evals.items():
        if not r.get("split"):
            continue  # not a pipeline eval (trust checks are listed below)
        ret, leg = r.get("retrieval", {}), r.get("legacy", {})
        imp = r.get("importance", {})
        md.append(f"| `{name}` | {r.get('split')} | {r.get('embedder')} | {f2(r.get('facts', {}).get('f1'))} | "
                  f"{f2(r.get('schemas', {}).get('accuracy'))} | {f2(imp.get('auc'))} / {f2(imp.get('accuracy'))} | "
                  f"{f2(ret.get('hit@1'))} / {f2(ret.get('hit@3'))} | {f2(ret.get('false_abstain_rate'))} | "
                  f"{f2(ret.get('abstain_accuracy'))} | "
                  + (f"{leg['gauntlet_passed']}/{leg['gauntlet_total']} | " if "gauntlet_passed" in leg else "– | ")
                  + (f"{leg['retention_hits']}/{leg['retention_total']} |" if "retention_hits" in leg else "– |"))
    others = [name for name, r in evals.items() if not r.get("split")]
    if others:
        md.append("\n`fresh` rows are retrieval-only runs on the practice batches. Other result files (not pipeline evals): "
                  + ", ".join(f"`{n}.json`" for n in others) + ".")
    md.append("\n## Vector engine (single thread, recall vs exact search)\n")
    md.append("| impl | n | dim | data | M | efC | efS | recall@10 | search avg ms | p99 ms | build s |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in summary["hnsw"]:
        md.append(f"| recallix ({r['simd']}) | {r['n']:,} | {r['dim']} | {r['data']} | {r['M']} | {r['efConstruction']} | "
                  f"{r['efSearch']} | {r['recall_at_10']:.3f} | {r['search_avg_ms']:.3f} | {r['search_p99_ms']:.3f} | {r['build_seconds']:.1f} |")
    for r in summary["hnsw_reference"]:
        md.append(f"| hnswlib | {r['n']:,} | {r['dim']} | {r['data']} | {r['M']} | {r['efConstruction']} | "
                  f"{r['efSearch']} | {r['recall_at_10']:.3f} | {r['search_avg_ms']:.3f} | – | {r['build_seconds']:.1f} |")
    for lm in summary["longmemeval"]:
        md.append(f"\n## LongMemEval_S (fresh held-out: {lm['questions']} questions, {lm['turns']:,} turns; embedder "
                  f"{lm['embedder']}, {lm['engine']} engine, answer check {lm['answer_check']}; no LLM)"
                  + (" — deep recall" if lm.get("deep_recall") else "")
                  + (" — results kept on decline (`recall.keep_results_on_decline`)"
                     if lm.get("keep_on_decline") or (lm.get("recall_settings") or {}).get("keep_results_on_decline") else "")
                  + f" — `{lm['file']}.json`\n")
        types = [t for t in lm["systems"]["recallix"] if t != "overall"]
        md.append("| system | recall@5 | recall@10 | hit@10 | " + " | ".join(f"{t} recall@10" for t in types) + " |")
        md.append("|---" * (4 + len(types)) + "|")
        for name, r in lm["systems"].items():
            md.append(f"| {name} | {r['overall']['recall@5']:.3f} | {r['overall']['recall@10']:.3f} | {r['overall']['hit@10']:.3f} | "
                      + " | ".join(f"{r[t]['recall@10']:.3f}" for t in types) + " |")
        a = lm["abstention"]
        md.append(f"\nRecallic declined {a['unanswerable_declined']:.1%} of {a['n_unanswerable']} unanswerable questions and wrongly "
                  f"declined {a['answerable_wrongly_declined']:.1%} of {a['n_answerable']} answerable ones "
                  f"(AUC: answer margin {a['auc_answer_margin']:.3f}, top relevance {a['auc_top_relevance']:.3f}).")
    for lc in summary["locomo"]:
        split = lc.get("split", "all")
        md.append(f"\n## LoCoMo retrieval, {split} split ({lc['conversations']} conversations, {lc['turns']} turns, {lc['questions']} questions; "
                  f"embedder {lc['embedder']}, {lc['engine']} engine; no LLM)"
                  + (" — dev conversations were used for fitting" if split == "dev" else
                     " — held out from fitting, but evaluated repeatedly during development" if split == "test" else "") + "\n")
        cats = [c for c in ("multi-hop", "temporal", "open-domain", "single-hop") if c in lc["systems"]["recallix"]]
        md.append("| system | recall@5 | recall@10 | hit@10 | " + " | ".join(f"{c} recall@10" for c in cats) + " |")
        md.append("|---" * (4 + len(cats)) + "|")
        for name, r in lc["systems"].items():
            md.append(f"| {name} | {r['overall']['recall@5']:.3f} | {r['overall']['recall@10']:.3f} | {r['overall']['hit@10']:.3f} | "
                      + " | ".join(f"{r[c]['recall@10']:.3f}" for c in cats) + " |")
        a = lc["abstention"]
        md.append(f"\nRecallic declined {a['adversarial_declined']:.1%} of {a['n_adversarial']} adversarial (unanswerable) questions "
                  f"and wrongly declined {a['answerable_wrongly_declined']:.1%} of {a['n_answerable']} answerable ones.")
    with open(RESULTS / "RESULTS.md", "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {RESULTS / 'summary.json'} and {RESULTS / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
