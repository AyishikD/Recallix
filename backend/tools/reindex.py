"""
Rebuild vector indexes from SQLite.

  python -m backend.tools.reindex              # rebuild namespaces that don't match SQLite
  python -m backend.tools.reindex --force      # rebuild everything
  python -m backend.tools.reindex --backfill   # first embed memories lacking a current-model embedding
                                               # (after changing the embedding model)
"""
import argparse

from backend.brain.memory import index_sync
from backend.brain.memory.episodic.episodic_memory import EpisodicMemory
from backend.brain.memory.semantic.semantic_memory import SemanticMemory


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--backfill", action="store_true")
    args = ap.parse_args()
    episodic, semantic = EpisodicMemory(), SemanticMemory()
    if args.backfill:
        index_sync.backfill(episodic, semantic)
    index_sync.sync(episodic, semantic, force=args.force)


if __name__ == "__main__":
    main()
