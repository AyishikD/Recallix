#!/bin/sh
# Starts the vector engine (loopback only) and the API. All state lives in /data:
#   /data/.internal_key   engine secret (generated on first start)
#   /data/...             SQLite databases, fitted files, downloaded models (/data/models)
set -e
cd /app
if [ -z "$RECALLIC_INTERNAL_KEY" ]; then
  [ -s /data/.internal_key ] || python -c "import secrets; print(secrets.token_hex(32))" > /data/.internal_key
  chmod 600 /data/.internal_key
  RECALLIC_INTERNAL_KEY=$(cat /data/.internal_key); export RECALLIC_INTERNAL_KEY
fi

# first start: create the admin API key and print it once
python - <<'PY'
from backend.api import keys
if not keys.list_keys():
    key = keys.create("admin", role="admin", label="docker first start")
    print("\n" + "=" * 72 + "\n Recallic admin API key (shown ONCE - store it now):\n   " + key +
          "\n Create user keys with:  docker compose exec recallic python -m backend.tools.keys create --user <name>\n" +
          "=" * 72 + "\n", flush=True)
PY

# download (first start only, into /data/models) and load the models before taking requests
echo "Preparing models (first start downloads ~700 MB into the data volume)..."
python - <<'PY'
from backend.brain.models.model_router import ModelRouter
from backend.brain.features.answer_check import AnswerChecker
from backend.brain.features.reranker import Reranker
assert ModelRouter().get_embedding("warm up"), "embedding model failed to load"
AnswerChecker.from_config()
from backend.brain.features.reranker import config as rerank_config
if rerank_config()["preload"]:
    Reranker.from_config(force=True)   # deep recall (recall.rerank)
print("models ready", flush=True)
PY

memory_service &
ENGINE=$!
trap 'kill $ENGINE 2>/dev/null' EXIT INT TERM
exec python server.py
