"""
Prototype schema classifier (no LLM).

Each schema, including "general", is defined by a handful of seed sentences.
For a memory embedding v (computed by the store path anyway):

    proto_k  = cos(v, centroid_k)                 # mean of the class's seeds
    nearest_k = max_j cos(v, seed_kj)             # closest single seed
    score_k  = ALPHA * proto_k + (1 - ALPHA) * nearest_k + CUE_WEIGHT * min(cues_k, 3)

The label is argmax_k score_k; additional labels are returned when within TIE
of the top. Because "general" is an explicit class it competes on equal
terms, so no model-specific absolute threshold is needed.

Seed embeddings are cached on disk per embedding model.
"""
import json
import re
import threading

import numpy as np

from backend.utils.paths import get_db_path, ensure_dir

SEEDS = {
    "identity": [
        "My name is Alex.",
        "I am a software engineer.",
        "I was born in Delhi.",
        "I'm originally from Kenya.",
        "I'm 27 years old.",
        "I am a vegetarian and a night owl.",
        "I have been a night-shift worker for years.",
        "I'm a mother of two.",
    ],
    "preference": [
        "I like spicy food.",
        "I prefer dark mode in every editor.",
        "I hate waking up early.",
        "My favorite color is blue.",
        "I'd rather take the train than fly.",
        "I don't enjoy crowded places.",
        "Pineapple does not belong on pizza for me.",
        "I love listening to podcasts while cooking.",
    ],
    "calendar": [
        "The meeting is on Monday at 3pm.",
        "My flight leaves on March 3.",
        "Dentist appointment next Tuesday morning.",
        "The workshop runs from 9 to 5 on Saturday.",
        "Interview scheduled for the 12th at noon.",
        "The review was moved to next week.",
        "Concert tickets are for Friday night.",
        "Holiday starts on 1 August.",
    ],
    "learning": [
        "I am learning Rust.",
        "Explain how recursion works.",
        "I studied linear algebra today.",
        "Reading a paper about neural networks.",
        "Taking an online course in economics.",
        "Memorising new vocabulary every morning.",
        "How does photosynthesis work?",
        "I want to understand quantum computing better.",
    ],
    "task": [
        "I need to finish the quarterly report.",
        "Remind me to call mom tonight.",
        "Todo: fix the login bug.",
        "I have to pick up the dry cleaning.",
        "Don't forget to buy milk.",
        "I should reply to the landlord's email.",
        "Next, deploy the hotfix to production.",
        "Must renew the car insurance.",
    ],
    "security": [
        "My password is hunter2.",
        "The API key is stored in the vault.",
        "Enable two-factor authentication on the account.",
        "The secret access code is ALPHA-9.",
        "Change the router admin password.",
        "My debit card PIN is 4321.",
        "Shred old bank statements before binning them.",
        "Never share the SSH private key.",
    ],
    "social": [
        "My friend Sam is visiting this weekend.",
        "Had dinner with my family.",
        "My sister's baby shower is in May.",
        "Met a colleague at the conference.",
        "Going to a party at Jake's place.",
        "Catching up with college friends at a reunion.",
        "Played football with the neighbours.",
        "Video called my grandmother today.",
    ],
    "general": [
        "The sky was clear last night.",
        "Cats sleep a lot during the day.",
        "The river flooded after the storm.",
        "There was a long queue at the shop.",
        "Vegetable prices went up again.",
        "The football match went to extra time.",
        "Some trees lose their leaves in winter.",
        "The new phone has a bigger screen.",
    ],
}

CUES = {
    "identity": r"\b(my name|i am|i'm|call me|born|i live|my age|years old|i grew up)\b",
    "preference": r"\b(like|love|hate|prefer|favou?rite|enjoy|dislike|rather)\b",
    "calendar": r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|tonight|deadline|meeting|appointment|schedule[d]?|january|february|march|april|june|july|august|september|october|november|december|\d{1,2}(:\d{2})?\s?(am|pm))\b",
    "learning": r"\b(learn|learning|study|studying|explain|understand|course|tutorial|practi[sc]ing|how does)\b",
    "task": r"\b(todo|to-do|need to|have to|must|remind me|don't let me forget|finish|should|next step|renew)\b",
    "security": r"\b(password|passcode|secret|api key|token|credential|2fa|two-factor|pin|access keys?)\b",
    "social": r"\b(friend|family|mom|mum|dad|mother|father|sister|brother|cousin|wife|husband|partner|colleague|neighbou?rs?|wedding|party)\b",
    "general": r"$^",  # no lexical cues: general wins on embedding similarity alone
}

ALPHA = 0.5
CUE_WEIGHT = 0.04
TIE = 0.01

_CUE_RE = {k: re.compile(v, re.I) for k, v in CUES.items()}


class SchemaClassifier:
    def __init__(self, embed_fn, model_name, cache_path=None):
        """
        embed_fn: text -> normalised np.ndarray, or None if embedding failed.
        model_name: used to invalidate cached seed embeddings when the model changes.
        """
        self.embed_fn = embed_fn
        self.model_name = model_name
        self.cache_path = cache_path or get_db_path("backend/storage/schema_prototypes.json")
        self.labels = list(SEEDS.keys())
        # (seeds [n_seeds, dim], seed_labels [n_seeds], centroids [n_labels, dim]); replaced atomically
        self._state = None
        self._lock = threading.Lock()

    # -- prototypes -------------------------------------------------------
    def _seed_signature(self):
        return {k: list(v) for k, v in SEEDS.items()}

    def _load_cache(self):
        try:
            with open(self.cache_path) as f:
                data = json.load(f)
            if data.get("model") == self.model_name and data.get("seeds") == self._seed_signature():
                return np.asarray(data["vectors"], dtype=np.float32), np.asarray(data["labels"])
        except (OSError, ValueError, KeyError):
            pass
        return None

    def _build(self):
        vecs, labels = [], []
        for i, label in enumerate(self.labels):
            for s in SEEDS[label]:
                v = self.embed_fn(s)
                if v is None:
                    return None
                vecs.append(v)
                labels.append(i)
        vecs = np.vstack(vecs).astype(np.float32)
        labels = np.asarray(labels)
        try:
            ensure_dir(self.cache_path)
            with open(self.cache_path, "w") as f:
                json.dump({"model": self.model_name, "seeds": self._seed_signature(),
                           "vectors": vecs.tolist(), "labels": labels.tolist()}, f)
        except OSError:
            pass
        return vecs, labels

    def _prepare(self, dim):
        """
        Returns (seeds, seed_labels, centroids) for this dimension, or None.
        Built under a lock and published as one immutable tuple, so concurrent
        requests never see seeds without centroids.
        """
        state = self._state
        if state is not None and state[0].shape[1] == dim:
            return state
        with self._lock:
            state = self._state
            if state is not None and state[0].shape[1] == dim:
                return state
            loaded = self._load_cache()
            if loaded is None or loaded[0].shape[1] != dim:
                loaded = self._build()
            if loaded is None:
                return None
            seeds, labels = loaded
            cents = np.vstack([seeds[labels == i].mean(axis=0) for i in range(len(self.labels))])
            self._state = (seeds, labels, cents / np.linalg.norm(cents, axis=1, keepdims=True))
            return self._state

    # -- classification ---------------------------------------------------
    def cue_hits(self, text):
        return np.array([len(_CUE_RE[k].findall(text)) for k in self.labels], dtype=np.float32)

    def scores(self, text, vec):
        """Per-label scores, or None when no embedding is available."""
        state = self._prepare(len(vec)) if vec is not None else None
        if state is None:
            return None
        v = np.asarray(vec, dtype=np.float32)
        seeds, seed_labels, centroids = state
        proto = centroids @ v
        seed_sims = seeds @ v
        nearest = np.array([seed_sims[seed_labels == i].max() for i in range(len(self.labels))])
        return ALPHA * proto + (1 - ALPHA) * nearest + CUE_WEIGHT * np.minimum(self.cue_hits(text), 3)

    def classify(self, text, vec):
        s = self.scores(text, vec)
        if s is None:
            # No embeddings available: fall back to cue counts alone.
            cues = self.cue_hits(text)
            if cues.max() == 0:
                return ["general"]
            return [self.labels[i] for i in range(len(self.labels)) if cues[i] == cues.max()]
        top = float(s.max())
        return [self.labels[i] for i in np.argsort(-s) if s[i] >= top - TIE]
