"""
Train the student: a token tagger that marks fact values and their relation.

    python phase6/train.py --variant all        # every source
    python phase6/train.py --variant no-sa      # without CC BY-SA data (Schema-Guided Dialogue)

Model: microsoft/deberta-v3-small (MIT). Tags: BIO per relation (B-location, I-location, ...).
Held-out messages (splits.json) are never trained on. Output: DATA/models/<variant>/ with
the model, tokenizer, label map and a manifest of exactly what it was trained on.
Deterministic training (fixed seed); inference is deterministic on CPU (argmax).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RELATIONS, labels, read_jsonl  # noqa: E402

BASE = "microsoft/deberta-v3-small"
LABELS = ["O"] + [f"{p}-{r}" for r in RELATIONS for p in ("B", "I")]
MAX_LEN = 96


def examples(variant):
    pool = {r["id"]: r for r in read_jsonl(DATA / "pool.jsonl")}
    held = set(json.loads((DATA / "splits.json").read_text())["held_out"])
    out = []
    for mid, facts in labels().items():
        msg = pool.get(mid)
        if msg is None or mid in held or (variant == "no-sa" and msg["source"] == "sgd"):
            continue
        out.append((msg["text"], facts))
    return out


def encode(tok, text, facts):
    """Token tags from character spans. A token belongs to a fact if its first non-space character
    lies inside the value span; the first such token is B-, the rest I-."""
    enc = tok(text, truncation=True, max_length=MAX_LEN, return_offsets_mapping=True)
    tags, started = [], set()
    for (a, b) in enc["offset_mapping"]:
        if a == b:
            tags.append(-100)  # special tokens
            continue
        while a < b and text[a].isspace():
            a += 1
        tag = "O"
        for k, f in enumerate(facts):
            if f["start"] <= a < f["end"]:
                tag = ("I-" if k in started else "B-") + f["relation"]
                started.add(k)
                break
        tags.append(LABELS.index(tag))
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": tags}


def main(argv=None):
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["all", "no-sa"], default="all")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    args = ap.parse_args(argv)

    random.seed(0)
    torch.manual_seed(0)
    data = examples(args.variant)
    tok = AutoTokenizer.from_pretrained(BASE)
    enc = [encode(tok, t, f) for t, f in data]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForTokenClassification.from_pretrained(
        BASE, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)}).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(enc) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, 1 - s / steps))
    print(f"variant {args.variant}: {len(enc)} training messages, {steps} steps on {device}", flush=True)

    def pad(batch):
        n = max(len(e["input_ids"]) for e in batch)
        ids = [e["input_ids"] + [tok.pad_token_id] * (n - len(e["input_ids"])) for e in batch]
        att = [e["attention_mask"] + [0] * (n - len(e["attention_mask"])) for e in batch]
        lab = [e["labels"] + [-100] * (n - len(e["labels"])) for e in batch]
        return {k: torch.tensor(v, device=device) for k, v in (("input_ids", ids), ("attention_mask", att), ("labels", lab))}

    model.train()
    t0, step = time.time(), 0
    for epoch in range(args.epochs):
        random.shuffle(enc)
        total = 0.0
        for i in range(0, len(enc), args.batch):
            loss = model(**pad(enc[i:i + args.batch])).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += float(loss)
            step += 1
        print(f"  epoch {epoch + 1}: loss {total / max(1, len(enc) // args.batch):.4f} ({time.time() - t0:.0f}s)", flush=True)

    out = DATA / "models" / args.variant
    out.mkdir(parents=True, exist_ok=True)
    model.to("cpu").save_pretrained(out)
    tok.save_pretrained(out)
    sources = sorted({r["source"] for r in read_jsonl(DATA / "pool.jsonl")} - ({"sgd"} if args.variant == "no-sa" else set()))
    (out / "manifest.json").write_text(json.dumps({
        "base": BASE, "base_licence": "MIT", "variant": args.variant, "training_messages": len(enc),
        "sources": sources, "labels": LABELS, "epochs": args.epochs, "lr": args.lr, "seed": 0,
        "teacher": "see phase6/common.py (open-weight model, Apache 2.0)"}, indent=1))
    print("saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
