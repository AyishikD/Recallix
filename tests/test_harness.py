"""The open benchmark harness runs any adapter on the same data with the same scoring."""
# eval keeps the system names it used when the recorded results were made (recallix, recallix-ranked).
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tiny_locomo(path):
    sample = {"sample_id": "t1", "conversation": {
        "speaker_a": "Priya", "speaker_b": "Sam",
        "session_1_date_time": "1 May 2023", "session_1": [
            {"speaker": "Priya", "dia_id": "D1:1", "text": "I just moved to Pune for my new job."},
            {"speaker": "Sam", "dia_id": "D1:2", "text": "Nice! I am still in Mumbai, training for a marathon."},
            {"speaker": "Priya", "dia_id": "D1:3", "text": "I'm allergic to peanuts, so no satay for me."}]},
        "qa": [{"question": "Where did Priya move?", "evidence": ["D1:1"], "category": 4},
               {"question": "What is Priya allergic to?", "evidence": ["D1:3"], "category": 4},
               {"question": "What is Sam allergic to?", "evidence": [], "category": 5}]}
    path.write_text(json.dumps([sample] * 6))  # 6 copies: conversations 0-4 are the dev split


def test_harness_scores_every_system_identically(tmp_path):
    sys.path.insert(0, str(ROOT / "eval"))
    import harness
    data = tmp_path / "tiny.json"
    _tiny_locomo(data)
    out = tmp_path / "result.json"
    assert harness.main(["--dataset", "locomo", "--data", str(data), "--split", "test",
                         "--systems", "recallix,recallix-ranked,vector-search,bm25", "--json", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["conversations"] == 1 and result["questions"] == 3 and result["data_sha256"]
    for name in ("recallix", "recallix-ranked", "vector-search", "bm25"):
        r = result["systems"][name]
        assert 0.0 <= r["recall@10"] <= 1.0 and r["config"] is not None
    assert result["systems"]["recallix-ranked"]["recall@10"] == 1.0  # three turns: everything is in the top 10
