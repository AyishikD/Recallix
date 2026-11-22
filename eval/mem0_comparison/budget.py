"""
Hard OpenAI spending cap for the Mem0 comparison (PROTOCOL.md "Budget").

install(ledger_path) patches the OpenAI SDK so that every chat, responses and embeddings call (Mem0's
internal ones included) is priced from the API's own token usage, added to a shared ledger file (with a
lock, so parallel processes share one budget), and refused once the ledger reaches the stop line. A model
without a known price is refused before it is called.

Prices: USD per 1M tokens (OpenAI's published list prices, 2026-11-23) plus a 10% margin.
"""
import fcntl
import json
import os

MARGIN = 1.10
PRICES = {  # model prefix: (input, output) per 1M tokens
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
}
CAP_USD = float(os.environ.get("OPENAI_BUDGET_USD", "3"))
STOP_USD = round(CAP_USD * 0.95, 2)  # stop with margin under the cap


class BudgetExceeded(RuntimeError):
    pass


def _price(model):
    for prefix, p in PRICES.items():
        if str(model).startswith(prefix):
            return p
    raise BudgetExceeded(f"no known price for model {model!r}; refusing to call it")


class Ledger:
    def __init__(self, path):
        self.path = path
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump({"usd": 0.0, "calls": 0, "by_purpose": {}}, f)

    def _update(self, fn):
        with open(self.path, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            d = json.load(f)
            out = fn(d)
            f.seek(0); f.truncate(); json.dump(d, f, indent=1)
            fcntl.flock(f, fcntl.LOCK_UN)
        return out

    def spent(self):
        with open(self.path) as f:
            return json.load(f)["usd"]

    def check(self):
        if self.spent() >= STOP_USD:
            raise BudgetExceeded(f"budget stop: estimated spend reached ${STOP_USD} (cap ${CAP_USD})")

    def add(self, purpose, model, tokens_in, tokens_out):
        pin, pout = _price(model)
        usd = (tokens_in * pin + tokens_out * pout) * MARGIN / 1e6

        def fn(d):
            d["usd"] = round(d["usd"] + usd, 6); d["calls"] += 1
            b = d["by_purpose"].setdefault(purpose, {"usd": 0.0, "calls": 0, "tokens_in": 0, "tokens_out": 0})
            b["usd"] = round(b["usd"] + usd, 6); b["calls"] += 1
            b["tokens_in"] += tokens_in; b["tokens_out"] += tokens_out
            return d["usd"]
        return self._update(fn)


PURPOSE = {"value": "unlabelled"}  # set by the caller: "mem0-store", "mem0-search", "answer", "judge"


def install(ledger_path):
    """Patch the OpenAI SDK in this process. Returns the Ledger."""
    from openai.resources.chat.completions import Completions
    from openai.resources.embeddings import Embeddings
    ledger = Ledger(ledger_path)

    def wrap(cls, kind):
        original = cls.create

        def create(self, *args, **kwargs):
            model = kwargs.get("model", "")
            _price(model)          # unknown model: refuse before calling
            ledger.check()         # over the stop line: refuse before calling
            resp = original(self, *args, **kwargs)
            u = getattr(resp, "usage", None)
            if kind == "chat":
                tin, tout = (u.prompt_tokens, u.completion_tokens) if u else (0, 0)
            else:
                tin, tout = (getattr(u, "prompt_tokens", 0) or getattr(u, "total_tokens", 0), 0) if u else (0, 0)
            ledger.add(PURPOSE["value"], model, tin, tout)
            return resp
        cls.create = create

    wrap(Completions, "chat")
    wrap(Embeddings, "embed")
    try:  # the Responses API (used by some models): refuse outright, nothing here should use it
        from openai.resources.responses import Responses

        def refuse(self, *a, **k):
            raise BudgetExceeded("Responses API is not priced in this harness")
        Responses.create = refuse
    except ImportError:
        pass
    return ledger
