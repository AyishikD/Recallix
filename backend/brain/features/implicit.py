"""
Implicit facts: what a memory implies but does not say (no LLM).

"I just got back from my sister's wedding in Goa" does not say "my sister got married",
but it implies it. Recallix finds such implications in two steps:

  1. propose  ATOMIC 2020 commonsense (backend/tools/build_inference_rules.py) maps events
              to their effects. The memory is matched to the closest event patterns by
              meaning, and their effects are written out about the actual people
              ("PersonY gets married" -> "Priya's sister gets married").
  2. verify   a natural-language-inference model (DeBERTa-v3 trained on MNLI/FEVER/ANLI)
              scores P(memory entails the statement). Only strong entailments survive.

ATOMIC alone over-generates (it is crowd-sourced; many effects are guesses about feelings
or plans), and similar wording can mislead ("my sister runs a wedding planning business"
does not imply she married); the verifier is what makes an inference trustworthy.

Use (config recall.implicit): at store time only very confident inferences are kept with
the memory (metadata.implied); at question time, when no memory answers directly, the
top candidate memories are expanded and the answer check reads the verified inferences.
Every inferred statement carries its chain: source memory -> ATOMIC pattern and relation
-> entailment probability.
"""
import json
import re
import threading

import numpy as np

from backend import settings
from .query import resolve_first_person

_KIN = (r"sister|brother|mother|mom|mum|father|dad|wife|husband|partner|girlfriend|boyfriend|fianc[eé]e?|son|"
        r"daughter|kids?|children|friend|boss|manager|colleague|coworker|cousin|aunt|uncle|grandma|grandmother|"
        r"grandpa|grandfather|niece|nephew|roommate|neighbou?r|team|teacher|doctor")
_PERSON_NP = re.compile(
    rf"\b((?:[A-Z][\w-]*'s|my|his|her|their|our)\s+(?:(?:best|older|younger|little|big|elder|step|new|old)\s+)?(?:{_KIN}))\b")
_NAME = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b")
_NOT_NAMES = {"I", "The", "A", "An", "My", "Our", "We", "He", "She", "They", "It", "This", "That", "Today", "Yesterday",
              "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "January", "February",
              "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"}


def rules_config():
    c = settings.get("recall.implicit", {}) or {}
    return {
        "enabled": bool(c.get("enabled", False)),
        "rules_dir": c.get("rules_dir", "inference_rules"),
        "rule_embedder": c.get("rule_embedder", "sentence-transformers/all-MiniLM-L6-v2"),
        "nli_model": c.get("nli_model", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"),
        "max_patterns": int(c.get("max_patterns", 3)),
        "pattern_similarity": float(c.get("pattern_similarity", 0.55)),
        "store_pattern_similarity": float(c.get("store_pattern_similarity", 0.7)),
        "store_min_entailment": float(c.get("store_min_entailment", 0.95)),
        "question_min_entailment": float(c.get("question_min_entailment", 0.9)),
        "max_restatement_similarity": float(c.get("max_restatement_similarity", 0.8)),
    }


def rule_paths():
    from pathlib import Path
    d = Path(settings.storage_path(rules_config()["rules_dir"], is_dir=True))
    return d / "heads.npy", d / "rules.json"


_AUX = {"is", "was", "are", "were", "has", "had", "can", "could", "will", "would", "should", "may", "might", "must",
        "does", "did", "gets", "got", "becomes", "became"}


def _third_person(word):
    """Present-tense verb for a third-person subject ("get" -> "gets"); other words unchanged."""
    w = word.lower()
    if w in _AUX or w.endswith("ing") or w.endswith("ed") or w.endswith("s"):
        return word
    irregular = {"have": "has", "be": "is", "do": "does", "go": "goes"}
    if w in irregular:
        return irregular[w]
    if re.search(r"(sh|ch|x|z|o)$", w):
        return word + "es"
    if re.search(r"[^aeiou]y$", w):
        return word[:-1] + "ies"
    return word + "s"


def _possessive(x):
    return "my" if x == "I" else f"{x}'s"


def instantiate(tail, relation, x, y):
    """An ATOMIC effect written as a sentence about real people, or None when it would be vague."""
    t = " ".join(tail.strip().rstrip(".").split())
    if not t or t.isupper() or len(t) < 3:
        return None  # e.g. "INFIDELITY": noise
    if "PersonY" in t and not y:
        return None
    if not re.search(r"Person[XY]", t):
        subject = y if relation == "oEffect" else x
        if not subject:
            return None  # "others" with nobody named: too vague to state
        t = re.sub(r"^(he|she|they|i|it)\s+", "", t, flags=re.I)
        first, _, rest = t.partition(" ")
        if subject != "I":
            first = _third_person(first)
        t = f"{subject} {first[0].lower() + first[1:]}" + (f" {rest}" if rest else "")
    t = (t.replace("PersonX's", _possessive(x)).replace("PersonY's", _possessive(y) if y else "")
          .replace("PersonX", x).replace("PersonY", y or ""))
    return t[0].upper() + t[1:] + "."


def people(premise, subject):
    """Candidate fillers for PersonY: kinship phrases ("Priya's sister") and other names."""
    found = []
    for m in _PERSON_NP.finditer(premise):
        found.append(m.group(1))
    for m in _NAME.finditer(premise):
        name = m.group(1)
        if name not in _NOT_NAMES and name != subject and not any(name in f for f in found):
            found.append(name)
    return list(dict.fromkeys(found))[:3]


class ImplicitInference:
    _instance = None
    _loaded = False
    _lock = threading.Lock()

    def __init__(self):
        import torch  # noqa: F401  (the NLI model is a torch model)
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        cfg = rules_config()
        heads_path, rules_path = rule_paths()
        with open(rules_path) as f:
            meta = json.load(f)
        self.rules = meta["rules"]
        self.source = meta.get("source", "")
        self.heads = np.load(heads_path).astype(np.float32)
        self.embedder = SentenceTransformer(meta.get("embedder", cfg["rule_embedder"]))
        self.tok = AutoTokenizer.from_pretrained(cfg["nli_model"])
        self.nli = AutoModelForSequenceClassification.from_pretrained(cfg["nli_model"], use_safetensors=True).eval()
        self.entail = self.nli.config.label2id.get("entailment", 0)
        self.nli_model = cfg["nli_model"]
        self.max_restatement = float(cfg["max_restatement_similarity"])
        self._infer_lock = threading.Lock()

    @classmethod
    def from_config(cls):
        """The shared instance, or None if disabled, the rule index is not built, or models cannot load."""
        if not rules_config()["enabled"]:
            return None
        with cls._lock:
            if not cls._loaded:
                cls._loaded = True
                heads_path, rules_path = rule_paths()
                if not (heads_path.exists() and rules_path.exists()):
                    print("[ImplicitInference] disabled: rule index not built "
                          "(python -m backend.tools.build_inference_rules --atomic <ATOMIC 2020 folder>)")
                else:
                    try:
                        cls._instance = cls()
                    except Exception as e:
                        print(f"[ImplicitInference] disabled: {e}")
            return cls._instance

    def entailment(self, premise, hypotheses):
        """P(premise entails h) for each hypothesis."""
        if not hypotheses:
            return []
        import torch
        enc = self.tok([premise] * len(hypotheses), hypotheses, return_tensors="pt", padding=True, truncation=True,
                       max_length=256)
        with self._infer_lock, torch.no_grad():
            probs = self.nli(**enc).logits.softmax(-1)[:, self.entail]
        return [float(p) for p in probs]

    def patterns(self, text):
        """[(similarity, rule)] for the event patterns closest in meaning to `text`, best first."""
        q = self.embedder.encode([text], normalize_embeddings=True)[0].astype(np.float32)
        sims = self.heads @ q
        top = np.argsort(-sims)[: rules_config()["max_patterns"]]
        return [(float(sims[i]), self.rules[i]) for i in top]

    def infer(self, text, speaker=None, min_entailment=0.9, min_similarity=0.55):
        """Verified implications of `text`: [{"statement", "entailment", "pattern", "relation", "similarity"}]."""
        premise = resolve_first_person(text, speaker) if speaker else text
        x = speaker or "I"
        candidates = {}
        for sim, rule in self.patterns(text):
            if sim < min_similarity:
                continue
            fillers = people(premise, x) if "PersonY" in rule["head"] else [None]
            for relation, tails in rule["rels"].items():
                for tail in tails:
                    for y in fillers or [None]:
                        s = instantiate(tail, relation, x, y)
                        if s and s.lower() not in premise.lower() and s not in candidates:
                            candidates[s] = {"pattern": rule["head"], "relation": relation, "similarity": round(sim, 3)}
        if not candidates:
            return []
        # drop restatements: an inference must add something the memory does not already say
        statements = list(candidates)
        vecs = self.embedder.encode([premise] + statements, normalize_embeddings=True)
        statements = [s for s, v in zip(statements, vecs[1:]) if float(v @ vecs[0]) < self.max_restatement]
        out = []
        for s, p in zip(statements, self.entailment(premise, statements)):
            if p >= min_entailment:
                out.append({"statement": s, "entailment": round(p, 3), **candidates[s]})
        return sorted(out, key=lambda r: -r["entailment"])
