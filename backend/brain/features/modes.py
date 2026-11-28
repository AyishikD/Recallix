"""
Recall modes: one choice of how readily recall answers, instead of separate switches.

  balanced  the default: the answer check as configured, nothing extra.
  cautious  also verify that the memory the reader answered from actually answers the question
            (features/answer_verify.py). Declines far more trick questions, and refuses more real ones.
  helpful   also rescue declined counting / ordering / time-between questions whose every presupposed event
            the memories support (features/aggregation_check.py, mode "rescue"), and accept a slightly less
            confident reader answer (recall.helpful_threshold_offset). Fewer refusals; more trick questions answered.

Measured trade-offs (eval/results/answer_verify, eval/results/aggregation_check/rescue):
  cautious  LoCoMo test (held out): trick questions declined 66 -> 122 of 209, real questions wrongly declined
            70 -> 224 of 772. LongMemEval_S: 16 -> 20 of 30 and 78 -> 115 of 470.
  helpful   LongMemEval_S: real questions wrongly declined 78 -> 55 of 470, unanswerable declined 16 -> 17 of 30.
            LoCoMo dev: 60 -> 56 of 760, traps unchanged. The threshold offset (0.8) on top, development data
            (eval/results/helpful_threshold): LoCoMo dev 56 -> 45 of 760 wrongly declined, trick questions
            declined 77 -> 68 of 237.

Set in config.yaml (recall.mode) or per request (mode in /memory/recall; recall(..., mode=...) in the SDKs). The
underlying switches (recall.answer_check.verify, recall.aggregation_check) still work on their own; a mode only
turns its own feature on, never off. Deep recall and keep_on_decline stay per request in every mode: they change
coverage and speed, not the answer-or-decline decision.
"""
import contextvars

from backend import settings

MODES = ("cautious", "balanced", "helpful")
_current = contextvars.ContextVar("recall_mode", default=None)


def resolve(mode=None):
    """The mode for this recall: the request's, else config recall.mode, else balanced. Unknown values -> balanced."""
    m = mode if mode is not None else settings.get("recall.mode", "balanced")
    m = str(m or "balanced").strip().lower()
    return m if m in MODES else "balanced"


def current():
    """The mode of the recall being served (balanced outside one)."""
    return _current.get() or resolve()


def use(mode):
    """Make `mode` current for this recall; returns a token for `reset`."""
    return _current.set(resolve(mode))


def reset(token):
    _current.reset(token)
