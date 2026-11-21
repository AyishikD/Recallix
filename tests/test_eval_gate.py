"""
Regression gate: the hermetic eval (hash embedder, dev split, answer check off so
it runs identically without torch) must not fall below eval/results/gate_dev_hash.json,
which is produced by the same command. Thresholds are the measured numbers
themselves (minus float tolerance), never hand-picked.
"""
import json
import os
import subprocess
import sys

from tests.conftest import ROOT

TOL = 1e-6


def test_dev_metrics_do_not_regress(tmp_path):
    baseline = json.load(open(ROOT / "eval" / "results" / "gate_dev_hash.json"))
    out = tmp_path / "now.json"
    env = {k: v for k, v in os.environ.items() if not k.startswith("RECALLIX_")}
    env["RECALLIX_ANSWER_CHECK"] = "0"
    proc = subprocess.run([sys.executable, str(ROOT / "eval" / "run_eval.py"), "--split", "dev", "--embedder", "hash",
                           "--only", "facts,schemas,importance,retrieval,legacy", "--json", str(out)],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-2000:]
    now = json.load(open(out))

    checks = {
        ("facts", "f1"): "higher", ("schemas", "accuracy"): "higher",
        ("importance", "auc"): "higher", ("importance", "accuracy"): "higher",
        ("retrieval", "hit@1"): "higher", ("retrieval", "hit@3"): "higher",
        ("retrieval", "abstain_accuracy"): "higher", ("retrieval", "false_abstain_rate"): "lower",
        ("retrieval", "stale_top"): "lower",
        ("legacy", "gauntlet_passed"): "higher", ("legacy", "retention_hits"): "higher",
    }
    failures = []
    for (task, metric), direction in checks.items():
        b, n = baseline[task][metric], now[task][metric]
        if (direction == "higher" and n < b - TOL) or (direction == "lower" and n > b + TOL):
            failures.append(f"{task}.{metric}: {n} vs published {b}")
    assert failures == []
