"""
Learned fact extractor: a small token tagger distilled from a teacher LLM (phase6/).

Complements the rule grammar (facts.py): it recognises phrasings the rules were not
written for. No LLM at runtime — a ~140M-parameter encoder on CPU, argmax decoding, so the
same message always yields the same facts (on the same hardware and model file).

Every value is an exact span of the message and every fact records the model version,
so receipts can quote it like a rule-extracted fact. Configuration: facts.learned_extractor
{enabled, path}. Off unless a trained model is installed and its evaluation justifies it.
"""
import hashlib
import json
import re
import threading
from pathlib import Path

from backend import settings

_KIN = (r"sister|brother|mother|mom|mum|father|dad|wife|husband|partner|girlfriend|boyfriend|son|daughter|"
        r"friend|boss|manager|colleague|cousin|aunt|uncle|grandma|grandmother|grandpa|grandfather|niece|nephew|roommate")
_KIN_SUBJECT = re.compile(rf"\b(?:my|our)\s+(?:(?:best|older|younger|little|big)\s+)?({_KIN})\b", re.I)
_NAME_SUBJECT = re.compile(r"\b([A-Z][a-z]+)\s+(?:is|was|lives|lived|works|worked|has|had|turns|moved|loves|likes|hates|"
                           r"studies|got|started|joined|plays|drives)\b")


# Relations that can hold several values at once ("likes tea" must not supersede "likes chess"):
# each value becomes its own slot, so the TruthEngine never treats them as replacements.
MULTI_VALUED = {"likes", "dislikes", "allergy", "hobby", "owns", "friend", "child", "sibling", "relative", "pet",
                "language", "plan", "goal", "medication", "health"}


def config():
    c = settings.get("facts.learned_extractor", {}) or {}
    return {"enabled": bool(c.get("enabled", False)), "path": c.get("path", "models/fact_extractor")}


_FIRST_PERSON = re.compile(r"\b(?:I|I'm|I've|I'd|I'll|me|we)\b")


def entity_for(text, start):
    """Who a value at `start` is about: the nearest subject before it in the same sentence —
    a kinship phrase ("my sister"), a name followed by a verb ("Sam works"), or the speaker ("I")."""
    sentence = re.split(r"[.!?;]\s+", text[:start])[-1]
    found = [(m.start(), m.group(1).lower()) for m in _KIN_SUBJECT.finditer(sentence)]
    found += [(m.start(), m.group(1)) for m in _NAME_SUBJECT.finditer(sentence)]
    found += [(m.start(), "user") for m in _FIRST_PERSON.finditer(sentence)]
    if not found:
        return "user"
    return max(found)[1]


class LearnedExtractor:
    _instance = None
    _loaded = False
    _lock = threading.Lock()

    def __init__(self, path):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        self.path = Path(path)
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForTokenClassification.from_pretrained(self.path).eval()
        self.labels = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]
        weights = next(self.path.glob("*.safetensors"), None) or next(self.path.glob("*.bin"), None)
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()[:12] if weights else "unknown"
        manifest = json.loads((self.path / "manifest.json").read_text()) if (self.path / "manifest.json").exists() else {}
        self.version = f"model-{manifest.get('variant', 'x')}-{digest}"
        self._infer_lock = threading.Lock()
        torch.set_num_threads(max(1, torch.get_num_threads()))

    @classmethod
    def from_config(cls):
        cfg = config()
        if not cfg["enabled"]:
            return None
        with cls._lock:
            if not cls._loaded:
                cls._loaded = True
                path = Path(cfg["path"])
                path = path if path.is_absolute() else Path(settings.storage_root()) / path
                try:
                    cls._instance = cls(path)
                except Exception as e:
                    print(f"[LearnedExtractor] disabled: cannot load {path}: {e}")
            return cls._instance

    def extract(self, text):
        """[{"entity", "relation", "value", "rule", "span"}] — same shape as facts.extract_triads()."""
        import torch
        enc = self.tok(text, truncation=True, max_length=256, return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        with self._infer_lock, torch.no_grad():
            pred = self.model(**enc).logits[0].argmax(-1).tolist()
        spans, cur = [], None
        for (a, b), p in zip(offsets, pred):
            if a == b:
                continue
            while a < b and text[a].isspace():
                a += 1
            tag = self.labels[p]
            if tag == "O":
                cur = None
                continue
            kind, rel = tag.split("-", 1)
            if kind == "B" or cur is None or cur["relation"] != rel:
                cur = {"relation": rel, "start": a, "end": b}
                spans.append(cur)
            else:
                cur["end"] = b
        out = []
        for s in spans:
            value = text[s["start"]:s["end"]].strip().rstrip(".,;:!")
            if not value:
                continue
            rel = s["relation"]
            if rel in MULTI_VALUED:
                rel = f"{rel}_{re.sub(r'[^a-z0-9]+', '_', value.lower()).strip('_')[:40]}"
            out.append({"entity": entity_for(text, s["start"]), "relation": rel, "value": value,
                        "rule": self.version, "extractor": self.version, "span": [s["start"], s["start"] + len(value)]})
        return out
