"""
Importance scoring without an LLM.

importance = 1 + 9 * sigmoid(z), with

    z = b + w_n * novelty + w_i * information + w_s * specificity
          + cue_bonus - small_talk_penalty - brevity_penalty

Features (all in [0, 1] unless noted):
  novelty      1 - max cosine similarity to the user's nearest existing memories.
               A near-duplicate of something already stored is worth little.
  information  mean normalised IDF of the message's terms over the FTS corpus.
               Rare words carry more information (Shannon / tf-idf intuition).
  specificity  1 - exp(-k / 2), k = count of concrete tokens (numbers, dates,
               times, proper nouns, codes, emails, URLs, money).
  cue_bonus    additive prior for explicit goals, preferences, "remember this".

z is linear in the features, z = bias + sum_k w_k * f_k, so sigmoid(z) is a
logistic-regression probability that the memory is worth keeping. WEIGHTS are
hand-set priors (a neutral message scores ~5). `fit_importance.py` refits them
(MAP logistic regression around these priors) from labelled data or from real
usage, and writes backend/storage/importance_weights.json, which overrides the
defaults at load time.
"""
import json
import math
import os
import re
import time

# Shipped default weights, produced by `python eval/run_eval.py --fit-importance`
# (provenance inside the file). Novelty depends on the embedding model's
# similarity scale, so refit from real usage once available:
#   python -m backend.brain.memory_ranking.fit_importance --from-db
DEFAULT_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "importance_weights.default.json")
PRIOR_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "importance_weights.prior.json")
with open(DEFAULT_WEIGHTS_PATH) as _f:
    WEIGHTS = json.load(_f)["weights"]
FEATURES = [k for k in WEIGHTS if k != "bias"]


def weights_path():
    from backend.utils.paths import get_db_path
    return get_db_path("backend/storage/importance_weights.json")


def load_weights():
    """Fitted weights if present (and complete), else the hand-set priors."""
    try:
        with open(weights_path()) as f:
            data = json.load(f)
        w = data.get("weights", {})
        if set(WEIGHTS) <= set(w):
            return {k: float(w[k]) for k in WEIGHTS}
    except (OSError, ValueError):
        pass
    return dict(WEIGHTS)

NEUTRAL = 0.5          # value used for a feature that cannot be measured
MIN_CORPUS_DOCS = 20   # below this, IDF statistics are too noisy to trust

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

STOPWORDS = frozenset("""
a an the and or but if of to in on at by for with from as is are was were be been being
i me my mine we our you your he she it its they them their this that these those there here
do does did have has had will would can could should may might must not no yes so than then
too very just also about into over under again what which who whom when where why how all any
""".split())

_GOAL = re.compile(r"\b(goal|task|need to|have to|must|plan|scheduled|deadline|due|todo|remind me)\b", re.I)
_PREF = re.compile(r"\b(like|love|hate|prefer|allergic|favou?rite|enjoy|dislike)\b", re.I)
_EXPLICIT = re.compile(r"\b(important|remember this|remember that|don't forget|do not forget|note that|never forget)\b", re.I)
_SMALL_TALK = re.compile(r"^\s*(hello|hi|hey|how are you|thanks|thank you|bye|ok|okay|cool|lol|good (morning|night))\b", re.I)

_SPECIFIC = [
    re.compile(r"\b\d+(?:[.,:/-]\d+)*\b"),                                  # numbers, dates, times
    re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b", re.I),
    re.compile(r"\b(?:mon|tues|wednes|thurs|fri|satur|sun)day\b", re.I),
    re.compile(r"\b[A-Z]{2,}(?:-\w+)?\b"),                                 # codes / acronyms
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),                               # emails
    re.compile(r"https?://\S+"),                                           # urls
    re.compile(r"[$€£₹]\s?\d"),                                            # money
]
_PROPER = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b")              # capitalised, not sentence-initial


def tokenize(text):
    """Tokenise roughly like SQLite FTS5's unicode61 tokenizer (lowercase alnum runs)."""
    return _TOKEN_RE.findall(text.lower())


def content_terms(text):
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


class ImportanceRanker:
    @staticmethod
    def novelty(max_similarity):
        """max_similarity: highest cosine similarity to an existing memory, or None if none exist."""
        if max_similarity is None:
            return 1.0
        return min(max(1.0 - float(max_similarity), 0.0), 1.0)

    @staticmethod
    def information(terms, doc_freqs, total_docs):
        """Mean normalised IDF: log(1 + N/(1+df)) / log(1 + N)."""
        if not terms or total_docs < MIN_CORPUS_DOCS:
            return NEUTRAL
        denom = math.log(1 + total_docs)
        idfs = [math.log(1 + total_docs / (1 + doc_freqs.get(t, 0))) / denom for t in terms]
        return min(sum(idfs) / len(idfs), 1.0)

    @staticmethod
    def specificity(text):
        k = sum(len(p.findall(text)) for p in _SPECIFIC) + len(_PROPER.findall(text))
        return 1.0 - math.exp(-k / 2.0)

    @staticmethod
    def features(content, max_similarity=None, doc_freqs=None, total_docs=0):
        terms = content_terms(content)
        words = content.split()
        return {
            "novelty": ImportanceRanker.novelty(max_similarity),
            "information": ImportanceRanker.information(terms, doc_freqs or {}, total_docs),
            "specificity": ImportanceRanker.specificity(content),
            "goal": 1.0 if _GOAL.search(content) else 0.0,
            "preference": 1.0 if _PREF.search(content) else 0.0,
            "explicit": 1.0 if _EXPLICIT.search(content) else 0.0,
            "small_talk": 1.0 if _SMALL_TALK.search(content) else 0.0,
            "brevity": 1.0 if len(words) < 3 else 0.0,
        }

    _weights = None
    _weights_mtime = None

    @classmethod
    def active_weights(cls):
        """Current weights, reloaded when importance_weights.json changes."""
        path = weights_path()
        mtime = (path, os.path.getmtime(path) if os.path.exists(path) else None)
        if cls._weights is None or mtime != cls._weights_mtime:
            cls._weights, cls._weights_mtime = load_weights(), mtime
        return cls._weights

    @staticmethod
    def score_features(f, weights=None):
        w = weights or ImportanceRanker.active_weights()
        z = w["bias"] + sum(w[k] * f[k] for k in FEATURES)
        return round(1.0 + 9.0 * _sigmoid(z), 2)

    @staticmethod
    def calculate(content, max_similarity=None, doc_freqs=None, total_docs=0):
        """Returns (importance in [1, 10], feature dict)."""
        f = ImportanceRanker.features(content, max_similarity, doc_freqs, total_docs)
        return ImportanceRanker.score_features(f), f

    @staticmethod
    def decay_importance(base_score, last_accessed_time):
        elapsed_days = (time.time() - last_accessed_time) / 86400
        decay_factor = math.pow(0.5, elapsed_days / 30)
        return base_score * decay_factor
