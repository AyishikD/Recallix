"""
Build the commonsense rule index for implicit facts from ATOMIC 2020.

    python -m backend.tools.build_inference_rules --atomic path/to/atomic2020_data-feb2021

ATOMIC 2020 (Hwang et al., AAAI 2021; CC BY 4.0) is a knowledge graph of everyday
if-then knowledge: "PersonX attends PersonY's wedding" -> oEffect -> "PersonY gets
married". Download it from the official link in https://github.com/allenai/comet-atomic-2020
(the data is not redistributed with Recallic; this tool builds a derived index locally).

Kept: event knowledge that can make a stated memory imply another fact —
  xEffect / oEffect   what follows for the people involved
  HasSubEvent         parts of an event
  isAfter / isBefore  what happened around it
  Causes              consequences
Dropped: tails "none", heads with blanks ("___"), and relations that infer intentions,
feelings or personality (xIntent, xReact, xAttr, ...): speculation memory should not store.

Output (under the storage root, recall.implicit.rules_dir): heads.npy (float16 unit
embeddings of each event pattern), rules.json (per head: relation -> tails, plus source
attribution). Embedding model: recall.implicit.rule_embedder (fixed, independent of the
deployment's own embedding model, so the index is portable).
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict

RELATIONS = ("xEffect", "oEffect", "HasSubEvent", "isAfter", "isBefore", "Causes")
MAX_TAILS = 4  # most frequent tails per (head, relation)
ATTRIBUTION = ("ATOMIC 2020 — Hwang, Bhagavatula, Le Bras, Da, Sakaguchi, Bosselut, Choi (AAAI 2021). "
               "Licensed CC BY 4.0. https://github.com/allenai/comet-atomic-2020")


def read_atomic(folder):
    tails = defaultdict(lambda: defaultdict(Counter))
    for split in ("train", "dev", "test"):
        path = os.path.join(folder, f"{split}.tsv")
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                if len(row) != 3:
                    continue
                head, rel, tail = (x.strip() for x in row)
                if rel not in RELATIONS or not head or not tail or tail.lower() in ("none", "") or "___" in head:
                    continue
                if "PersonX" not in head and "PersonY" not in head:
                    continue  # keep events involving people: memories are about people
                tails[head][rel][tail] += 1
    rules = []
    for head, by_rel in tails.items():
        rels = {rel: [t for t, _ in c.most_common(MAX_TAILS)] for rel, c in by_rel.items()}
        rules.append({"head": head, "rels": rels})
    rules.sort(key=lambda r: r["head"])
    return rules


def display(head):
    """Embedding text for a head: placeholders become plain words ("PersonX" -> "someone")."""
    return head.replace("PersonX's", "my").replace("PersonY's", "their").replace("PersonX", "I").replace("PersonY", "someone")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atomic", required=True, help="folder with ATOMIC 2020 train.tsv / dev.tsv / test.tsv")
    args = ap.parse_args(argv)

    import numpy as np
    from backend import settings
    from backend.brain.features.implicit import rule_paths, rules_config

    cfg = rules_config()
    t0 = time.time()
    rules = read_atomic(args.atomic)
    if not rules:
        raise SystemExit("no usable ATOMIC tuples found (expected train.tsv etc. in that folder)")
    print(f"{len(rules)} event patterns from ATOMIC 2020 ({sum(len(r['rels']) for r in rules)} relation groups)")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(cfg["rule_embedder"])
    vecs = model.encode([display(r["head"]) for r in rules], batch_size=256, normalize_embeddings=True,
                        show_progress_bar=False).astype(np.float16)
    heads_path, rules_path = rule_paths()
    np.save(heads_path, vecs)
    with open(rules_path, "w") as f:
        json.dump({"source": ATTRIBUTION, "embedder": cfg["rule_embedder"], "relations": list(RELATIONS),
                   "rules": rules}, f)
    print(f"wrote {heads_path} and {rules_path} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
