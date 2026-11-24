"""
Shared pieces for Phase 6 (a small, deterministic fact extractor distilled from a teacher LLM).

The teacher is used OFFLINE only, to label training data; Recallix never calls an LLM at
runtime. Teacher: any OpenAI-compatible endpoint serving an open-weight model whose licence
allows training on its outputs (default: Groq, openai/gpt-oss-120b, Apache 2.0), configured
in .env (GROQ_API_KEY, TEACHER_BASE_URL, TEACHER_MODEL). Data stays outside the repo
(PHASE6_DATA, default the session scratchpad) because some sources are CC BY-SA.
"""
import json
import os
import socket
import re
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("PHASE6_DATA", REPO / ".phase6_data"))

# Closed relation set the student learns (a value span is tagged with one of these).
RELATIONS = [
    "name", "nickname", "age", "birthday", "location", "hometown", "address", "nationality", "language",
    "job", "employer", "school", "education", "email", "phone",
    "allergy", "diet", "health", "medication",
    "likes", "dislikes", "hobby",
    "pet", "spouse", "partner", "child", "sibling", "parent", "friend", "relative",
    "vehicle", "owns",
    "plan", "event_time", "deadline", "destination", "travel_date", "party_size", "budget",
    "goal",
]

SOURCES = {  # dataset -> licence (recorded on every message for the no-share-alike variant)
    "taskmaster": "CC BY 4.0", "multiwoz": "MIT", "sgd": "CC BY-SA 4.0", "synthetic": "teacher output (Apache 2.0 model)",
}


def env():
    vals = {}
    path = REPO / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                vals[k.strip()] = v.strip().strip('"').strip("'")
    vals.update({k: v for k, v in os.environ.items() if k in ("GROQ_API_KEY", "TEACHER_BASE_URL", "TEACHER_MODEL")})
    return vals


socket.setdefaulttimeout(90)  # a dead connection fails within 90 s instead of hanging (seen: CLOSE_WAIT stall)


class Teacher:
    """OpenAI-compatible chat client that stays under a tokens-per-minute budget."""

    def __init__(self, tpm=None, rpd=None):
        e = env()
        self.key = e.get("TEACHER_API_KEY") or e.get("GROQ_API_KEY")
        if not self.key:
            raise SystemExit("no teacher API key: set GROQ_API_KEY in .env")
        self.base = e.get("TEACHER_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
        self.model = e.get("TEACHER_MODEL", "openai/gpt-oss-120b")
        self.tpm = int(tpm or os.environ.get("TEACHER_TPM", 7500))  # a little under the 8000 limit
        self.window = deque()  # (time, tokens)
        self.used = 0

    def _wait_for(self, tokens):
        while True:
            now = time.time()
            while self.window and now - self.window[0][0] > 60:
                self.window.popleft()
            if sum(t for _, t in self.window) + tokens <= self.tpm:
                return
            time.sleep(max(0.5, 60 - (now - self.window[0][0]) + 0.1))

    def chat(self, system, user, max_tokens=1500, temperature=0.0, json_mode=True):
        estimate = (len(system) + len(user)) // 3 + max_tokens
        body = {"model": self.model, "temperature": temperature, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if "gpt-oss" in self.model:
            body["reasoning_effort"] = os.environ.get("TEACHER_REASONING", "low")  # ~30% fewer tokens, same labels
        for attempt in range(35):  # rides out ~30 minutes of network or provider outage
            self._wait_for(min(estimate, self.tpm))
            req = urllib.request.Request(self.base + "/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                                                  "User-Agent": "recallix-phase6/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=90) as r:
                    out = json.load(r)
                tokens = int(out.get("usage", {}).get("total_tokens", estimate))
                self.window.append((time.time(), tokens))
                self.used += tokens
                text = out["choices"][0]["message"]["content"]
                return json.loads(text) if json_mode else text
            except urllib.error.HTTPError as e:
                if e.code == 400 and json_mode:
                    detail = e.read()[:400]
                    if b"json_validate_failed" in detail:  # the model produced invalid JSON: try again
                        print(f"  teacher returned invalid JSON (attempt {attempt + 1})", flush=True)
                        self.window.append((time.time(), estimate))
                        if attempt >= 2:
                            return None  # give up on this request; the caller decides
                        continue
                    raise SystemExit(f"teacher error 400: {detail!r}")
                if e.code == 429 or e.code >= 500:
                    wait = float(e.headers.get("retry-after", 0) or 0) or min(60, 5 * 2 ** attempt)
                    try:
                        why = json.loads(e.read() or b"{}").get("error", {}).get("message", "")[:160]
                    except (ValueError, AttributeError):
                        why = ""
                    print(f"  teacher {e.code}; waiting {wait:.0f}s (attempt {attempt + 1}) {why}", flush=True)
                    self.window.append((time.time(), self.tpm))  # back off the whole window
                    time.sleep(wait)
                    continue
                raise SystemExit(f"teacher error {e.code}: {e.read()[:200]!r}")
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError, KeyError) as e:
                print(f"  teacher unreachable ({type(e).__name__}); retrying in {min(60, 5 * 2 ** attempt)}s", flush=True)
                time.sleep(min(60, 5 * 2 ** attempt))
        raise SystemExit("teacher kept failing; try again later")


def read_jsonl(path):
    p = Path(path)
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def append_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def labels():
    """{message id: facts}: the reviewed label where one exists (review.py), else the first pass.
    Lets the held-out answer key be reviewed before the rest of the pool."""
    out = {r["id"]: r["facts"] for r in read_jsonl(DATA / "labels.jsonl")}
    reviewed = DATA / "labels_reviewed.jsonl"
    if reviewed.exists():
        out.update({r["id"]: r["facts"] for r in read_jsonl(reviewed)})
    return out


def labels_path():
    """Reviewed labels (review.py) when complete, else the first pass."""
    reviewed, first = DATA / "labels_reviewed.jsonl", DATA / "labels.jsonl"
    if reviewed.exists() and len(read_jsonl(reviewed)) >= len(read_jsonl(first)):
        return reviewed
    return first


def items_of(res, key):
    """
    The list under `key` in a teacher's JSON answer. The model is asked for {key: [...]} but sometimes
    answers with the bare list, or with something else entirely; anything that is not a list gives [].
    """
    found = res.get(key, []) if isinstance(res, dict) else res
    return found if isinstance(found, list) else []


def indexed(items):
    """{i: facts} from [{"i": .., "facts": [..]}] items; items without a usable index keep their position."""
    out = {}
    for pos, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        try:
            i = int(it.get("i", pos))
        except (TypeError, ValueError):
            i = pos
        facts = it.get("facts", [])
        out[i] = facts if isinstance(facts, list) else []
    return out
