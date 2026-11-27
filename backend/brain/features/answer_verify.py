"""
Answer verification: does the memory the reader answered from actually answer the question?

The extractive reader (features/answer_check.py) finds the most answer-like span in a memory. On trick questions
it often finds one that does not answer at all: "Why did Joanna choose the hair color she did?" answered from
Joanna saying "Wow, your new hair color looks amazing!" (the other person's hair), or "What is Caroline's reason
for getting into running?" answered "purple color" from "Love that purple color! For walking or running?".
On LoCoMo dev, 154 of the 160 trick questions recall answered were answered from a memory spoken by the person
asked about, so speaker filters cannot catch them.

A second model reads the question and the memory, written as "<speaker> said: <message>", and scores whether the
memory answers the question. It is a cross-encoder trained on QNLI (cross-encoder/qnli-electra-base, 110M
parameters, Apache 2.0, CPU): the GLUE task "does this sentence contain the answer to this question?". No LLM.
A reader answer stands only if one of the memories the reader accepted scores at least `threshold`.

Configuration: config.yaml recall.answer_check.verify (enabled, model, threshold). Applied in
RecallEngine.multi_stage_recall right after the reader.
"""
import re
import threading

from backend import settings

_DEFAULT_MODEL = "cross-encoder/qnli-electra-base"


def config():
    c = ((settings.get("recall.answer_check", {}) or {}).get("verify") or {})
    return {"enabled": bool(c.get("enabled", False)), "model": c.get("model") or _DEFAULT_MODEL,
            "threshold": float(c.get("threshold", 0.001)), "max_length": int(c.get("max_length", 256)),
            "device": c.get("device")}


def premise(content, speaker=None):
    """The memory as the verifier reads it: "<speaker> said: <message>" (a leading "[date]" is dropped)."""
    text = re.sub(r"^\s*\[[^\]]*\]\s*", "", str(content or ""))
    m = re.match(r"^([A-Z][\w'’\-]*(?: [A-Z][\w'’\-]*)?):\s+(.*)$", text, re.S)  # "John: ..." as stored by some clients
    if m:
        speaker, text = speaker or m.group(1), m.group(2)
    return f"{speaker} said: {text}" if speaker else text


class AnswerVerifier:
    _instances = {}
    _lock = threading.Lock()

    def __init__(self, model_name, max_length=256, device=None):
        from sentence_transformers import CrossEncoder
        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)
        self._infer_lock = threading.Lock()

    @classmethod
    def from_config(cls):
        """The configured verifier, or None if disabled or unavailable (logged once)."""
        cfg = config()
        if not cfg["enabled"]:
            return None
        key = (cfg["model"], cfg["max_length"], cfg["device"])
        with cls._lock:
            if key not in cls._instances:
                try:
                    cls._instances[key] = cls(*key)
                except Exception as e:  # missing sentence-transformers/torch, no network for the first download, ...
                    print(f"[AnswerVerifier] disabled: cannot load {cfg['model']}: {e}")
                    cls._instances[key] = None
            return cls._instances[key]

    def scores(self, question, premises):
        """P(premise answers question) for each premise."""
        if not premises:
            return []
        with self._infer_lock:
            out = self.model.predict([(question, p) for p in premises], show_progress_bar=False)
        return [float(x) for x in (out.reshape(-1) if hasattr(out, "reshape") else out)]
