"""
Episodic store: the system of record for every memory (SQLite, WAL mode).

The vector engine's in-memory indexes are derived from the `embedding` column
here and can always be rebuilt from it (see backend/brain/memory/index_sync.py).

Visibility: a memory is visible to an agent if it is `shared`, or `private`
and written by that agent. Every read that takes an agent_id applies this rule
in SQL; agent_id=None means an internal, unscoped read.
"""
import hashlib
import json
import re
import sqlite3
import time

import numpy as np

from backend.utils.paths import get_db_path

COLUMNS = ("id", "user_id", "agent_id", "memory_type", "content", "timestamp", "importance",
           "reinforcement_score", "retrieval_count", "metadata", "entity_id", "relation_type", "status")
_SELECT = ", ".join(f"e.{c}" for c in COLUMNS)
_VISIBLE = "(e.memory_type = 'shared' OR e.agent_id = ?)"


def content_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def user_token(user_id):
    """
    A single full-text token per user ("u" + hex digest), indexed in its own FTS column so
    a search can be restricted to one user *inside* the index. Without it, FTS matched and
    BM25-scored every user's rows before filtering, so recall slowed down as other tenants
    added data (5.4 s per query on a 247k-memory, 500-user database).
    """
    return "u" + hashlib.sha1(str(user_id).encode("utf-8")).hexdigest()[:20]


_FTS_DDL = "CREATE VIRTUAL TABLE IF NOT EXISTS episodic_fts USING fts5(content, content_id UNINDEXED, user_id UNINDEXED, utok)"
# BM25 over the content column only: the user token restricts the match but adds nothing to the score.
_BM25 = "bm25(episodic_fts, 1.0, 0.0, 0.0, 0.0)"


class EpisodicMemory:
    def __init__(self, db_path=None):
        self.db_path = db_path or get_db_path("backend/storage/sqlite_db/memory.db")
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS episodic_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                agent_id TEXT,
                memory_type TEXT,
                content TEXT,
                timestamp REAL,
                importance REAL DEFAULT 5.0,
                reinforcement_score REAL DEFAULT 1.0,
                retrieval_count INTEGER DEFAULT 0,
                metadata TEXT,
                entity_id TEXT,
                relation_type TEXT,
                status TEXT DEFAULT 'ACTIVE'
            )
        ''')
        conn.create_function("rx_user_token", 1, user_token, deterministic=True)
        fts_cols = None
        if cursor.execute("SELECT 1 FROM sqlite_master WHERE name = 'episodic_fts'").fetchone():
            fts_cols = [d[0] for d in cursor.execute("SELECT * FROM episodic_fts LIMIT 0").description]
        if fts_cols is not None and "utok" not in fts_cols:
            # Migration: rebuild the full-text index with the per-user token column. Rows are
            # keyed by the memory id (rowid) so deletes are index lookups, not table scans.
            cursor.execute("DROP TABLE IF EXISTS episodic_vocab")
            cursor.execute("DROP TABLE episodic_fts")
            fts_cols = None
        cursor.execute(_FTS_DDL)
        cursor.execute("CREATE VIRTUAL TABLE IF NOT EXISTS episodic_vocab USING fts5vocab(episodic_fts, 'row')")
        if fts_cols is None:
            cursor.execute("INSERT INTO episodic_fts (rowid, content_id, content, user_id, utok) "
                           "SELECT id, id, content, user_id, rx_user_token(user_id) FROM episodic_events")

        # Migrations (idempotent)
        columns = {row[1] for row in cursor.execute("PRAGMA table_info(episodic_events)")}
        for name, ddl in (("embedding", "BLOB"), ("embedding_model", "TEXT"), ("content_hash", "TEXT")):
            if name not in columns:
                cursor.execute(f"ALTER TABLE episodic_events ADD COLUMN {name} {ddl}")
        for (mid, content) in cursor.execute(
                "SELECT id, content FROM episodic_events WHERE content_hash IS NULL").fetchall():
            cursor.execute("UPDATE episodic_events SET content_hash = ? WHERE id = ?", (content_hash(content or ""), mid))

        # Questions each memory answers (facts.question_keys), embedded; see recall.question_keys.
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS question_keys (
                memory_id INTEGER, user_id TEXT, text TEXT, embedding BLOB, embedding_model TEXT
            )
        ''')
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_qk_memory ON question_keys(memory_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_relation_status ON episodic_events(entity_id, relation_type, status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_hash ON episodic_events(user_id, content_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_time ON episodic_events(user_id, timestamp)")
        conn.commit()
        conn.close()

    def _connect(self):
        return sqlite3.connect(self.db_path, timeout=30)

    @staticmethod
    def _row(r):
        return dict(zip(COLUMNS, r))

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def has_speaker(self, user_id, name):
        """True if any of this user's memories was recorded as said by `name` (metadata.client.speaker)."""
        conn = self._connect()
        try:
            return conn.execute(
                "SELECT 1 FROM episodic_events WHERE user_id = ? AND lower(json_extract(metadata, '$.client.speaker')) = ? "
                "LIMIT 1", (user_id, str(name).lower())).fetchone() is not None
        finally:
            conn.close()

    def latest_fact_values(self, user_id, slot_ids, agent_id, before):
        """
        {slot_id: (value, memory_id, timestamp)} — the most recent earlier value of each fact
        this agent can see (used to flag a new message that contradicts it).
        """
        if not slot_ids:
            return {}
        conn = self._connect()
        try:
            ph = ",".join("?" * len(slot_ids))
            rows = conn.execute(
                f"SELECT json_extract(j.value, '$.slot_id'), json_extract(j.value, '$.value'), e.id, e.timestamp "
                f"FROM episodic_events e, json_each(e.metadata, '$.slots') j "
                f"WHERE e.user_id = ? AND e.status = 'ACTIVE' AND {_VISIBLE} AND e.timestamp < ? "
                f"AND json_extract(j.value, '$.slot_id') IN ({ph}) ORDER BY e.timestamp, e.id",
                (user_id, agent_id, before, *slot_ids)).fetchall()
        except sqlite3.Error:
            return {}
        finally:
            conn.close()
        return {slot: (value, mid, ts) for slot, value, mid, ts in rows}  # later rows overwrite: latest wins

    def latest_entity_for_relation(self, user_id, relation, agent_id, before):
        """
        Entity of this user's most recent fact with `relation` stored before `before` and
        visible to `agent_id`, or None (used to resolve "the deadline" across messages).
        """
        conn = self._connect()
        try:
            row = conn.execute(
                f"SELECT json_extract(j.value, '$.entity') FROM episodic_events e, json_each(e.metadata, '$.slots') j "
                f"WHERE e.user_id = ? AND e.status = 'ACTIVE' AND {_VISIBLE} AND e.timestamp < ? "
                f"AND json_extract(j.value, '$.relation') = ? ORDER BY e.timestamp DESC, e.id DESC LIMIT 1",
                (user_id, agent_id, before, relation)).fetchone()
            return row[0] if row else None
        except sqlite3.Error:
            return None
        finally:
            conn.close()

    def find_duplicate(self, user_id, content, agent_id):
        """Id of an identical memory already visible to this agent, else None."""
        conn = self._connect()
        try:
            row = conn.execute(
                f"SELECT e.id FROM episodic_events e WHERE e.user_id = ? AND e.content_hash = ? "
                f"AND e.status = 'ACTIVE' AND {_VISIBLE} LIMIT 1",
                (user_id, content_hash(content), agent_id)).fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def store(self, event):
        metadata = event.get("metadata", {}) or {}
        slots = metadata.get("slots") or []
        entity_id = slots[0].get("entity") if slots else None
        relation_type = slots[0].get("relation") if slots else None
        embedding = event.get("embedding")
        blob = np.asarray(embedding, dtype=np.float32).tobytes() if embedding is not None else None

        conn = self._connect()
        try:
            cur = conn.execute("""
                INSERT INTO episodic_events
                (user_id, agent_id, memory_type, content, timestamp, importance, reinforcement_score,
                 retrieval_count, metadata, entity_id, relation_type, status, embedding, embedding_model, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, 'ACTIVE', ?, ?, ?)
            """, (event["user_id"], event.get("agent_id", "default_agent"), event.get("memory_type", "private"),
                  event["content"], event.get("timestamp", time.time()), event.get("importance", 5.0),
                  event.get("reinforcement_score", 1.0), json.dumps(metadata), entity_id, relation_type,
                  blob, event.get("embedding_model") if blob else None, content_hash(event["content"])))
            memory_id = cur.lastrowid
            conn.execute("INSERT INTO episodic_fts (rowid, content_id, content, user_id, utok) VALUES (?, ?, ?, ?, ?)",
                         (memory_id, memory_id, event["content"], event["user_id"], user_token(event["user_id"])))
            conn.commit()
            return memory_id
        finally:
            conn.close()

    def delete(self, user_id, memory_id):
        """Deletes one memory of this user. Returns True if it existed."""
        conn = self._connect()
        try:
            cur = conn.execute("DELETE FROM episodic_events WHERE id = ? AND user_id = ?", (memory_id, user_id))
            if cur.rowcount:
                conn.execute("DELETE FROM episodic_fts WHERE rowid = ?", (memory_id,))
                conn.execute("DELETE FROM question_keys WHERE memory_id = ?", (memory_id,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def clear_user(self, user_id):
        conn = self._connect()
        try:
            conn.execute("DELETE FROM episodic_fts WHERE rowid IN (SELECT id FROM episodic_events WHERE user_id = ?)",
                         (user_id,))
            conn.execute("DELETE FROM question_keys WHERE user_id = ?", (user_id,))
            n = conn.execute("DELETE FROM episodic_events WHERE user_id = ?", (user_id,)).rowcount
            conn.commit()
            return n
        finally:
            conn.close()

    def clear_all(self):
        conn = self._connect()
        try:
            conn.execute("DELETE FROM episodic_fts")
            conn.execute("DELETE FROM question_keys")
            n = conn.execute("DELETE FROM episodic_events").rowcount
            conn.commit()
            return n
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def search(self, user_id, query, agent_id=None, limit=200):
        """
        Lexical candidates: (A) memories whose primary fact is about an entity
        named in the query, (B) FTS5 BM25 matches. Both bounded by `limit`.
        """
        from backend.brain.features.query import capitalised_entities
        vis = f"AND {_VISIBLE}" if agent_id is not None else ""
        vis_args = (agent_id,) if agent_id is not None else ()
        results = {}
        conn = self._connect()
        try:
            for ent in capitalised_entities(query)[:5]:
                if len(ent) < 2:
                    continue
                for r in conn.execute(
                        f"SELECT {_SELECT} FROM episodic_events e WHERE e.user_id = ? AND e.status = 'ACTIVE' "
                        f"AND (e.entity_id = ? OR e.entity_id LIKE ?) {vis} LIMIT ?",
                        (user_id, ent, f"%{ent}%", *vis_args, limit)):
                    m = self._row(r)
                    m["symbolic_match"] = True
                    m["bm25_score"] = None  # filled in below if the words match too
                    results[m["id"]] = m

            words = [re.sub(r"[^\w]+", " ", w).strip() for w in query.replace('"', " ").split()]
            tokens = [f'"{w}"' for w in words if len(w) > 2]
            if tokens:
                for r in conn.execute(f"""
                        SELECT {_SELECT}, f.score FROM (
                            SELECT content_id, {_BM25} AS score FROM episodic_fts
                            WHERE episodic_fts MATCH ? ORDER BY score LIMIT ?
                        ) f JOIN episodic_events e ON e.id = f.content_id
                        WHERE e.status = 'ACTIVE' AND e.user_id = ? {vis}
                        """, (f"utok:{user_token(user_id)} AND ({' OR '.join(tokens)})", limit, user_id, *vis_args)):
                    m = self._row(r[:-1])
                    if m["id"] in results:
                        results[m["id"]]["bm25_score"] = r[-1]
                    else:
                        m["bm25_score"] = r[-1]
                        m["symbolic_match"] = False
                        results[m["id"]] = m
        finally:
            conn.close()
        return list(results.values())

    def get_many(self, user_id, memory_ids, agent_id=None):
        """Batch fetch (one query per 500 ids) of visible ACTIVE memories, as {id: memory}."""
        ids = [int(i) for i in memory_ids if i is not None]
        out = {}
        if not ids:
            return out
        vis = f"AND {_VISIBLE}" if agent_id is not None else ""
        conn = self._connect()
        try:
            for start in range(0, len(ids), 500):
                chunk = ids[start:start + 500]
                ph = ",".join("?" * len(chunk))
                args = (user_id, *chunk) + ((agent_id,) if agent_id is not None else ())
                for r in conn.execute(
                        f"SELECT {_SELECT} FROM episodic_events e WHERE e.user_id = ? AND e.id IN ({ph}) "
                        f"AND e.status = 'ACTIVE' {vis}", args):
                    m = self._row(r)
                    out[m["id"]] = m
        finally:
            conn.close()
        return out

    def get_by_id(self, user_id, memory_id, agent_id=None):
        return self.get_many(user_id, [memory_id], agent_id).get(int(memory_id))

    def neighbours(self, user_id, memory_id, agent_id=None, window=2, max_gap_seconds=3600.0):
        """
        The visible ACTIVE memories stored just after and just before `memory_id` in the same
        conversation: at most `window` on each side, each within `max_gap_seconds` of it.
        Nearest first (next, previous, second next, second previous). recall.second_pass uses it.
        """
        vis = f"AND {_VISIBLE}" if agent_id is not None else ""
        tail = (agent_id,) if agent_id is not None else ()
        conn = self._connect()
        try:
            row = conn.execute("SELECT timestamp FROM episodic_events WHERE id = ? AND user_id = ?",
                               (int(memory_id), user_id)).fetchone()
            if not row or row[0] is None:
                return []
            t, mid, gap, w = float(row[0]), int(memory_id), float(max_gap_seconds), int(window)
            after = [self._row(r) for r in conn.execute(
                f"SELECT {_SELECT} FROM episodic_events e WHERE e.user_id = ? AND e.status = 'ACTIVE' {vis} "
                f"AND (e.timestamp > ? OR (e.timestamp = ? AND e.id > ?)) AND e.timestamp <= ? "
                f"ORDER BY e.timestamp ASC, e.id ASC LIMIT ?", (user_id, *tail, t, t, mid, t + gap, w))]
            before = [self._row(r) for r in conn.execute(
                f"SELECT {_SELECT} FROM episodic_events e WHERE e.user_id = ? AND e.status = 'ACTIVE' {vis} "
                f"AND (e.timestamp < ? OR (e.timestamp = ? AND e.id < ?)) AND e.timestamp >= ? "
                f"ORDER BY e.timestamp DESC, e.id DESC LIMIT ?", (user_id, *tail, t, t, mid, t - gap, w))]
        finally:
            conn.close()
        out = []
        for i in range(w):
            out.extend(side[i] for side in (after, before) if i < len(side))
        return out

    def list_recent(self, user_id, limit=200):
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT {_SELECT} FROM episodic_events e WHERE e.user_id = ? AND e.status = 'ACTIVE' "
                f"ORDER BY e.timestamp DESC LIMIT ?", (user_id, limit)).fetchall()
            return [self._row(r) for r in rows]
        finally:
            conn.close()

    def active_users(self, since_ts):
        conn = self._connect()
        try:
            return [r[0] for r in conn.execute(
                "SELECT DISTINCT user_id FROM episodic_events WHERE timestamp >= ?", (since_ts,))]
        finally:
            conn.close()

    def count(self, user_id=None):
        conn = self._connect()
        try:
            if user_id is None:
                return conn.execute("SELECT COUNT(*) FROM episodic_events").fetchone()[0]
            return conn.execute("SELECT COUNT(*) FROM episodic_events WHERE user_id = ?", (user_id,)).fetchone()[0]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Embedding cache (source of truth for the vector indexes)
    # ------------------------------------------------------------------
    def add_question_keys(self, memory_id, user_id, items, model):
        """items: [(question text, vector)]."""
        conn = self._connect()
        try:
            with conn:
                conn.executemany("INSERT INTO question_keys VALUES (?, ?, ?, ?, ?)",
                                 [(int(memory_id), user_id, text, np.asarray(vec, dtype=np.float32).tobytes(), model)
                                  for text, vec in items if vec is not None])
        finally:
            conn.close()

    def get_question_keys(self, memory_ids, model):
        """{memory id: [(question text, vector)]} for keys embedded with `model`."""
        ids = [int(i) for i in memory_ids if i is not None]
        out = {}
        conn = self._connect()
        try:
            for start in range(0, len(ids), 500):
                chunk = ids[start:start + 500]
                ph = ",".join("?" * len(chunk))
                for mid, text, blob in conn.execute(
                        f"SELECT memory_id, text, embedding FROM question_keys WHERE memory_id IN ({ph}) "
                        f"AND embedding_model = ?", (*chunk, model)):
                    out.setdefault(mid, []).append((text, np.frombuffer(blob, dtype=np.float32)))
        finally:
            conn.close()
        return out

    def get_embeddings(self, memory_ids, model):
        """{id: vector} for ids whose stored embedding was made by `model`."""
        ids = [int(i) for i in memory_ids if i is not None]
        out = {}
        conn = self._connect()
        try:
            for start in range(0, len(ids), 500):
                chunk = ids[start:start + 500]
                ph = ",".join("?" * len(chunk))
                for mid, blob in conn.execute(
                        f"SELECT id, embedding FROM episodic_events WHERE id IN ({ph}) "
                        f"AND embedding IS NOT NULL AND embedding_model = ?", (*chunk, model)):
                    out[mid] = np.frombuffer(blob, dtype=np.float32)
        finally:
            conn.close()
        return out

    def set_embedding(self, memory_id, vector, model):
        if vector is None:
            return
        conn = self._connect()
        try:
            conn.execute("UPDATE episodic_events SET embedding = ?, embedding_model = ? WHERE id = ?",
                         (np.asarray(vector, dtype=np.float32).tobytes(), model, int(memory_id)))
            conn.commit()
        finally:
            conn.close()

    def embedding_counts(self, model):
        """{user_id: number of ACTIVE memories with an embedding from `model`}."""
        conn = self._connect()
        try:
            return dict(conn.execute(
                "SELECT user_id, COUNT(*) FROM episodic_events WHERE status = 'ACTIVE' "
                "AND embedding IS NOT NULL AND embedding_model = ? GROUP BY user_id", (model,)).fetchall())
        finally:
            conn.close()

    def embedding_dims(self, model):
        """{user_id: vector dimension} of the stored embeddings from `model`."""
        conn = self._connect()
        try:
            return {u: int(n) // 4 for u, n in conn.execute(
                "SELECT user_id, MAX(length(embedding)) FROM episodic_events WHERE status = 'ACTIVE' "
                "AND embedding IS NOT NULL AND embedding_model = ? GROUP BY user_id", (model,)).fetchall()}
        finally:
            conn.close()

    def iter_embeddings(self, user_id, model, batch=1000):
        """Yields lists of (id, vector) for this user's indexable memories."""
        conn = self._connect()
        try:
            cur = conn.execute(
                "SELECT id, embedding FROM episodic_events WHERE user_id = ? AND status = 'ACTIVE' "
                "AND embedding IS NOT NULL AND embedding_model = ? ORDER BY id", (user_id, model))
            while True:
                rows = cur.fetchmany(batch)
                if not rows:
                    return
                yield [(mid, np.frombuffer(blob, dtype=np.float32)) for mid, blob in rows]
        finally:
            conn.close()

    def rows_needing_embedding(self, model, limit=1000):
        """ACTIVE memories with no embedding, or one from a different model."""
        conn = self._connect()
        try:
            return conn.execute(
                "SELECT id, user_id, content FROM episodic_events WHERE status = 'ACTIVE' AND "
                "(embedding IS NULL OR embedding_model IS NULL OR embedding_model != ?) LIMIT ?",
                (model, limit)).fetchall()
        finally:
            conn.close()

    def stale_embedding_count(self, model):
        conn = self._connect()
        try:
            return conn.execute(
                "SELECT COUNT(*) FROM episodic_events WHERE status = 'ACTIVE' AND "
                "(embedding IS NULL OR embedding_model IS NULL OR embedding_model != ?)", (model,)).fetchone()[0]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Corpus statistics
    # ------------------------------------------------------------------
    def term_stats(self, terms):
        """(total_docs, {term: document_frequency}) over the whole FTS corpus."""
        conn = self._connect()
        try:
            total = conn.execute("SELECT COUNT(*) FROM episodic_events").fetchone()[0]
            unique = list(set(terms))
            dfs = {}
            if unique:
                ph = ",".join("?" * len(unique))
                dfs = dict(conn.execute(f"SELECT term, doc FROM episodic_vocab WHERE term IN ({ph})", unique).fetchall())
            return total, dfs
        except sqlite3.Error:
            return 0, {}
        finally:
            conn.close()
