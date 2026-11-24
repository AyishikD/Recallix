"""
Answer-level check: does a retrieved memory actually answer the question?

A small extractive question-answering model trained on SQuAD 2.0 reads each
served memory together with the question. SQuAD 2.0 contains questions that
are deliberately *unanswerable* from a plausible-looking passage (often about
the wrong entity), so the model learned to compare

    best answer span score   max over spans (start_logit[i] + end_logit[j])
    no-answer score          start_logit[CLS] + end_logit[CLS]

margin = best span - no answer. A margin above the model's decision boundary
means the memory contains an answer. Recall abstains when no served memory has
one. No LLM: the model is an encoder (tens of millions of parameters) on CPU.

Configuration (config.yaml recall.answer_check): enabled, model, max_length,
stride, max_answer_tokens, threshold. threshold=null uses, in order: this
deployment's certified threshold (backend/tools/calibrate.py, see calibration.py),
the value recorded in answer_check.default.json (chosen on dev data) for that
model, else the model's native boundary (margin 0).
"""
import json
import os
import re
import threading

import numpy as np

from backend import settings

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "answer_check.default.json")


def _cfg():
    c = settings.get("recall.answer_check", {}) or {}
    enabled = bool(c.get("enabled", False))
    env = os.environ.get("RECALLIX_ANSWER_CHECK")  # "0"/"1" overrides config (CI, eval gate)
    if env is not None:
        enabled = env.strip() not in ("0", "false", "no", "")
    return {"enabled": enabled, "model": c.get("model"),
            "max_length": int(c.get("max_length", 384)), "stride": int(c.get("stride", 128)),
            "max_answer_tokens": int(c.get("max_answer_tokens", 30)), "threshold": c.get("threshold")}


def restore_case(text):
    """
    Texting-style memories ("moving to an apartment in bansko bulgaria instead") are read as if
    written normally: sentence starts and a standalone "i" are capitalised and a final full
    stop is added. Only text with no capital letter at all is touched, so ordinary prose (and
    every LoCoMo turn) is read exactly as before. Letter case is the only in-place change, so
    answer offsets still index the original text. The SQuAD readers are cased models trained
    on edited prose; measured on fresh, independently written texting data (eval/data/fresh,
    development batches 1-2): right memory recognised as answering 44/50 -> 49/50, trap questions
    wrongly answered 1/20 -> 2/20; unseen batch 3: one more question answered, traps wrongly answered
    0/10 -> 2/10. Off by default (recall.answer_check.restore_case) because a wrong answer costs more
    than a refusal.
    """
    if not text or text != text.lower() or not re.search(r"[a-z]", text):
        return text
    out = re.sub(r"\bi\b", "I", text)
    out = re.sub(r"(^\s*|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), out)
    return out if re.search(r"[.!?]\s*$", out) else out + "."


def _restore_case_enabled():
    return bool((settings.get("recall.answer_check", {}) or {}).get("restore_case", False))


class AnswerChecker:
    _instances = {}
    _lock = threading.Lock()

    def __init__(self, model_name, max_length=384, stride=128, max_answer_tokens=30):
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.eval()
        self.max_length, self.stride, self.max_answer_tokens = max_length, stride, max_answer_tokens
        self._infer_lock = threading.Lock()

    @classmethod
    def from_config(cls):
        """The configured checker, or None if disabled or unavailable (logged once)."""
        cfg = _cfg()
        if not cfg["enabled"] or not cfg["model"]:
            return None
        key = (cfg["model"], cfg["max_length"], cfg["stride"], cfg["max_answer_tokens"])
        with cls._lock:
            if key not in cls._instances:
                try:
                    cls._instances[key] = cls(cfg["model"], cfg["max_length"], cfg["stride"], cfg["max_answer_tokens"])
                except Exception as e:  # missing transformers/torch, no network for the first download, ...
                    print(f"[AnswerChecker] disabled: cannot load {cfg['model']}: {e}")
                    cls._instances[key] = None
            return cls._instances[key]

    def certificate(self):
        """This deployment's wrong-refusal certificate for this model (calibration.py), or None."""
        from .calibration import load
        return load(self.model_name)

    def threshold(self):
        """
        Resolution order: config override > this deployment's certified threshold >
        the threshold learned from this deployment's feedback (answer_check_learning.py) >
        the dev-fitted default for this model > the model's native boundary (0).
        A certificate without enough data means "never decline": -inf.
        """
        override = _cfg()["threshold"]
        if override is not None:
            return float(override)
        cert = self.certificate()
        if cert is not None:
            return float(cert["threshold"]) if cert.get("certified") else float("-inf")
        from .answer_check_learning import load as learned
        t = learned(self.model_name)
        if t is not None:
            return t
        try:
            with open(DEFAULT_PATH) as f:
                fitted = json.load(f)
            if fitted.get("model") == self.model_name:
                return float(fitted["threshold"])
        except (OSError, ValueError, KeyError):
            pass
        return 0.0  # the model's own answer / no-answer boundary

    def check(self, question, contexts):
        """[{"margin", "answer"}] per context: best span vs no-answer, over all windows of long contexts."""
        if not contexts:
            return []
        import torch
        originals = contexts
        if _restore_case_enabled():
            contexts = [restore_case(c) for c in contexts]
        enc = self.tokenizer([question] * len(contexts), contexts, truncation="only_second",
                             max_length=self.max_length, stride=self.stride, return_overflowing_tokens=True,
                             return_offsets_mapping=True, padding=True, return_tensors="pt")
        sample_map = enc.pop("overflow_to_sample_mapping").tolist()
        offsets = enc.pop("offset_mapping").tolist()
        with self._infer_lock, torch.no_grad():
            out = self.model(**{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")})
        starts, ends = out.start_logits.numpy(), out.end_logits.numpy()
        results = [{"margin": -np.inf, "answer": ""} for _ in contexts]
        for w in range(len(sample_map)):
            seq_ids = enc.sequence_ids(w)
            ctx = np.array([sid == 1 for sid in seq_ids])
            if not ctx.any():
                continue
            s, e = starts[w], ends[w]
            null = s[0] + e[0]
            idx = np.where(ctx)[0]
            # best (i, j) with i <= j < i + max_answer_tokens, both in the context
            span = s[idx][:, None] + e[idx][None, :]
            gap = idx[None, :] - idx[:, None]
            span[(gap < 0) | (gap >= self.max_answer_tokens)] = -np.inf
            i, j = np.unravel_index(np.argmax(span), span.shape)
            margin = float(span[i, j] - null)
            k = sample_map[w]
            if margin > results[k]["margin"]:
                a, b = offsets[w][idx[i]][0], offsets[w][idx[j]][1]
                results[k] = {"margin": margin, "answer": originals[k][a:b]}  # same offsets: case-only rewrite
        return results
