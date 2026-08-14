"""Single-file SQLite store: episodes, traces, FTS5, durable job queue."""

import json
import sqlite3
import time
from pathlib import Path

import numpy as np

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS episodes(
  id INTEGER PRIMARY KEY,
  ns TEXT DEFAULT 'default',
  session_id TEXT,
  user_text TEXT,
  assistant_text TEXT,
  model TEXT,
  ts REAL,
  sha TEXT UNIQUE,
  status TEXT DEFAULT 'done'
);

CREATE TABLE IF NOT EXISTS traces(
  id INTEGER PRIMARY KEY,
  ns TEXT DEFAULT 'default',
  episode_id INT,
  title TEXT,
  gist TEXT,
  status TEXT DEFAULT 'active',        -- forming|active|silent|superseded
  quality TEXT DEFAULT 'llm',          -- llm|extractive
  created_ts REAL,
  n_access REAL DEFAULT 1.0,
  last8 TEXT DEFAULT '[]',             -- JSON [[ts, weight], ...] newest last
  session_id TEXT,
  embedding BLOB,
  superseded_by INT,                   -- correction that replaced this trace
  beta REAL DEFAULT 0,                 -- write-time importance (novelty+salience)
  source TEXT DEFAULT 'user',
  pinned INT DEFAULT 0                 -- never decays, outranks at recall
);

CREATE VIRTUAL TABLE IF NOT EXISTS trace_fts USING fts5(title, gist, tokenize='porter unicode61');

CREATE TABLE IF NOT EXISTS jobs(
  id INTEGER PRIMARY KEY,
  episode_id INT,
  attempts INT DEFAULT 0,
  enqueued REAL
);

CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS links(
  a INT, b INT,                -- undirected, a < b
  weight REAL,                 -- hebbian, saturates toward 1
  last_ts REAL,
  PRIMARY KEY (a, b)
);

CREATE TABLE IF NOT EXISTS schemas(
  id INTEGER PRIMARY KEY,
  kind TEXT,                 -- 'profile' (singleton) | future: 'topic'
  gist TEXT,
  source_ids TEXT,           -- JSON list of trace ids it distills
  created_ts REAL,
  updated_ts REAL
);
"""


class Store:
    def __init__(self, db_path: Path, embed_dim: int = 256):
        db_path = Path(db_path).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.embed_dim = embed_dim
        for ddl in (
            "ALTER TABLE traces ADD COLUMN superseded_by INT",
            "ALTER TABLE traces ADD COLUMN beta REAL DEFAULT 0",
            "ALTER TABLE traces ADD COLUMN source TEXT DEFAULT 'user'",
            "ALTER TABLE traces ADD COLUMN pinned INT DEFAULT 0",
            "ALTER TABLE episodes ADD COLUMN source TEXT DEFAULT 'user'",
        ):
            try:
                self.db.execute(ddl)
            except sqlite3.OperationalError:
                pass  # column already exists
        self._migrate_fts()
        self._migrate_provenance()
        # In-memory dense index: disposable cache over the traces table.
        self._ids: list[int] = []
        self._mat = np.zeros((0, embed_dim), dtype=np.float32)
        self._rebuild_index()

    def _migrate_fts(self):
        """v2: porter stemming (peanut matches peanuts). Rebuild from traces
        when the stored FTS version predates it."""
        row = self.db.execute("SELECT value FROM meta WHERE key='fts_v'").fetchone()
        if row and json.loads(row["value"]) == 2:
            return
        self.db.executescript(
            "DROP TABLE IF EXISTS trace_fts;"
            "CREATE VIRTUAL TABLE trace_fts USING fts5(title, gist, tokenize='porter unicode61');"
        )
        for r in self.db.execute(
            "SELECT t.id, t.title, t.gist, e.user_text FROM traces t"
            " LEFT JOIN episodes e ON e.id=t.episode_id WHERE t.status='active'"
        ).fetchall():
            extra = (r["user_text"] or "")[:400]
            self.db.execute(
                "INSERT INTO trace_fts(rowid, title, gist) VALUES(?,?,?)",
                (r["id"], r["title"], f"{r['gist']}\n{extra}".strip()),
            )
        self.db.execute(
            "INSERT INTO meta(key, value) VALUES('fts_v', '2')"
            " ON CONFLICT(key) DO UPDATE SET value='2'"
        )
        self.db.commit()

    def _migrate_provenance(self):
        """One-time backfill: classify existing episodes by whose words they
        were, propagate to traces, and silence traces formed from framework
        plumbing (approval notices, memory-flush prompts) that had been
        masquerading as user facts."""
        row = self.db.execute("SELECT value FROM meta WHERE key='prov_v'").fetchone()
        if row and json.loads(row["value"]) == 2:
            return
        from .formation import classify_source

        for e in self.db.execute("SELECT id, user_text FROM episodes").fetchall():
            src = classify_source(e["user_text"] or "")
            if src != "user":
                self.db.execute(
                    "UPDATE episodes SET source=? WHERE id=?", (src, e["id"])
                )
                self.db.execute(
                    "UPDATE traces SET source=? WHERE episode_id=?", (src, e["id"])
                )
        for t in self.db.execute(
            "SELECT id FROM traces WHERE source='system' AND status='active'"
        ).fetchall():
            self.db.execute(
                "UPDATE traces SET status='silent' WHERE id=?", (t["id"],)
            )
            self.db.execute("DELETE FROM trace_fts WHERE rowid=?", (t["id"],))
        self.db.execute(
            "INSERT INTO meta(key, value) VALUES('prov_v','2')"
            " ON CONFLICT(key) DO UPDATE SET value='2'"
        )
        self.db.commit()

    # ---------- dense index ----------

    def _rebuild_index(self):
        rows = self.db.execute(
            "SELECT id, embedding FROM traces WHERE status='active' AND embedding IS NOT NULL"
        ).fetchall()
        ids, vecs = [], []
        for r in rows:
            v = np.frombuffer(r["embedding"], dtype=np.float32)
            if v.shape[0] == self.embed_dim:
                ids.append(r["id"])
                vecs.append(v)
        self._ids = ids
        self._mat = (
            np.vstack(vecs) if vecs else np.zeros((0, self.embed_dim), dtype=np.float32)
        )

    def dense_search(self, qvec: np.ndarray, k: int = 40) -> list[tuple[int, float]]:
        if self._mat.shape[0] == 0:
            return []
        sims = self._mat @ qvec.astype(np.float32)
        top = np.argsort(-sims)[:k]
        return [(self._ids[i], float(sims[i])) for i in top]

    # ---------- episodes ----------

    def add_episode(
        self, session_id, user_text, assistant_text, model, sha, ns="default",
        source="user",
    ) -> int | None:
        """Insert episode + summarize job. Returns episode id, or None if duplicate."""
        try:
            cur = self.db.execute(
                "INSERT INTO episodes(ns, session_id, user_text, assistant_text, model, ts, sha, source)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (ns, session_id, user_text, assistant_text, model, time.time(), sha, source),
            )
        except sqlite3.IntegrityError:
            return None  # byte-identical repeat (agent retry loop)
        eid = cur.lastrowid
        self.db.execute(
            "INSERT INTO jobs(episode_id, enqueued) VALUES(?,?)", (eid, time.time())
        )
        self.db.commit()
        return eid

    def repeat_presentation(self, sha: str, weight: float):
        """A byte-identical episode re-occurred: strengthen its trace at half weight."""
        row = self.db.execute(
            "SELECT t.id FROM traces t JOIN episodes e ON t.episode_id=e.id WHERE e.sha=?",
            (sha,),
        ).fetchone()
        if row:
            self.add_access(row["id"], weight)

    # ---------- traces ----------

    def add_trace(
        self, episode_id, title, gist, session_id, embedding, quality,
        ns="default", fts_extra="", beta=0.0, source="user",
    ) -> int:
        now = time.time()
        blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
        cur = self.db.execute(
            "INSERT INTO traces(ns, episode_id, title, gist, status, quality, created_ts,"
            " n_access, last8, session_id, embedding, beta, source)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                ns, episode_id, title, gist, "active", quality, now,
                1.0, json.dumps([[now, 1.0]]), session_id, blob, beta, source,
            ),
        )
        tid = cur.lastrowid
        # FTS also indexes a verbatim slice of the episode: literal tokens
        # (error codes, names, values) must survive a lossy small-model gist.
        self.db.execute(
            "INSERT INTO trace_fts(rowid, title, gist) VALUES(?,?,?)",
            (tid, title, f"{gist}\n{fts_extra}".strip()),
        )
        self.db.commit()
        if embedding is not None:
            self._ids.append(tid)
            self._mat = np.vstack([self._mat, embedding.astype(np.float32)[None, :]])
        return tid

    def supersede(self, old_id: int, new_id: int):
        """A correction replaces a stale trace: silence it, never delete -
        the history stays auditable, but it can no longer be recalled."""
        self.db.execute(
            "UPDATE traces SET status='superseded', superseded_by=? WHERE id=?",
            (new_id, old_id),
        )
        self.db.execute("DELETE FROM trace_fts WHERE rowid=?", (old_id,))
        self.db.commit()
        if old_id in self._ids:
            i = self._ids.index(old_id)
            self._ids.pop(i)
            self._mat = np.delete(self._mat, i, axis=0)

    def silence(self, trace_id: int):
        """Forgetting is retrieval failure, not deletion: kept on disk,
        excluded from recall and both indexes."""
        self.db.execute("UPDATE traces SET status='silent' WHERE id=?", (trace_id,))
        self.db.execute("DELETE FROM trace_fts WHERE rowid=?", (trace_id,))
        self.db.commit()
        if trace_id in self._ids:
            i = self._ids.index(trace_id)
            self._ids.pop(i)
            self._mat = np.delete(self._mat, i, axis=0)

    def delete_trace(self, trace_id: int):
        """Hard deletion - privacy path only."""
        self.db.execute("DELETE FROM traces WHERE id=?", (trace_id,))
        self.db.execute("DELETE FROM trace_fts WHERE rowid=?", (trace_id,))
        self.db.commit()
        if trace_id in self._ids:
            i = self._ids.index(trace_id)
            self._ids.pop(i)
            self._mat = np.delete(self._mat, i, axis=0)

    def edit_trace(self, trace_id: int, title: str, gist: str, embedding=None):
        blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
        if blob is not None:
            self.db.execute(
                "UPDATE traces SET title=?, gist=?, embedding=? WHERE id=?",
                (title, gist, blob, trace_id),
            )
        else:
            self.db.execute(
                "UPDATE traces SET title=?, gist=? WHERE id=?", (title, gist, trace_id)
            )
        self.db.execute("DELETE FROM trace_fts WHERE rowid=?", (trace_id,))
        self.db.execute(
            "INSERT INTO trace_fts(rowid, title, gist) VALUES(?,?,?)",
            (trace_id, title, gist),
        )
        self.db.commit()
        self._rebuild_index()

    def set_pinned(self, trace_id: int, pinned: bool):
        self.db.execute(
            "UPDATE traces SET pinned=? WHERE id=?", (1 if pinned else 0, trace_id)
        )
        self.db.commit()

    def supersedes_something(self, trace_id: int) -> bool:
        return (
            self.db.execute(
                "SELECT 1 FROM traces WHERE superseded_by=? LIMIT 1", (trace_id,)
            ).fetchone()
            is not None
        )

    def add_access(self, trace_id: int, weight: float):
        row = self.db.execute(
            "SELECT last8, n_access FROM traces WHERE id=?", (trace_id,)
        ).fetchone()
        if not row:
            return
        last8 = json.loads(row["last8"])
        last8.append([time.time(), weight])
        last8 = last8[-8:]
        self.db.execute(
            "UPDATE traces SET last8=?, n_access=? WHERE id=?",
            (json.dumps(last8), row["n_access"] + weight, trace_id),
        )
        self.db.commit()

    def get_traces(self, ids: list[int]) -> dict[int, sqlite3.Row]:
        if not ids:
            return {}
        q = ",".join("?" * len(ids))
        rows = self.db.execute(
            f"SELECT * FROM traces WHERE id IN ({q}) AND status='active'", ids
        ).fetchall()
        return {r["id"]: r for r in rows}

    def fts_search(self, tokens: list[str], k: int = 40) -> list[tuple[int, float, int]]:
        """Returns (trace_id, score, nmatch): score = -bm25 (bigger is better),
        nmatch = how many distinct query tokens the row matched. bm25 magnitudes
        collapse under FTS5's idf clamp in small stores, so admission uses
        nmatch - raw literal evidence - instead."""
        if not tokens:
            return []
        match = " OR ".join(f'"{t}"' for t in tokens)
        try:
            rows = self.db.execute(
                "SELECT rowid, bm25(trace_fts) AS b FROM trace_fts WHERE trace_fts MATCH ?"
                " ORDER BY b LIMIT ?",
                (match, k),
            ).fetchall()
            counts: dict[int, int] = {}
            candidate_ids = {r["rowid"] for r in rows}
            for t in tokens:
                for m in self.db.execute(
                    "SELECT rowid FROM trace_fts WHERE trace_fts MATCH ? LIMIT 200",
                    (f'"{t}"',),
                ).fetchall():
                    if m["rowid"] in candidate_ids:
                        counts[m["rowid"]] = counts.get(m["rowid"], 0) + 1
        except sqlite3.OperationalError:
            return []
        return [(r["rowid"], -r["b"], counts.get(r["rowid"], 0)) for r in rows]

    def max_cosine(self, vec: np.ndarray) -> tuple[int | None, float]:
        if self._mat.shape[0] == 0:
            return None, 0.0
        sims = self._mat @ vec.astype(np.float32)
        i = int(np.argmax(sims))
        return self._ids[i], float(sims[i])

    # ---------- jobs ----------

    def next_job(self):
        return self.db.execute(
            "SELECT j.id AS job_id, j.attempts, e.* FROM jobs j"
            " JOIN episodes e ON e.id=j.episode_id ORDER BY j.id LIMIT 1"
        ).fetchone()

    def finish_job(self, job_id: int):
        self.db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        self.db.commit()

    def bump_job(self, job_id: int):
        self.db.execute("UPDATE jobs SET attempts=attempts+1 WHERE id=?", (job_id,))
        self.db.commit()

    def queue_depth(self) -> int:
        return self.db.execute("SELECT COUNT(*) c FROM jobs").fetchone()["c"]

    # ---------- links (the memory graph) ----------

    def add_link(self, a: int, b: int, w: float):
        """Hebbian: co-active traces wire together, saturating toward 1."""
        if a == b:
            return
        a, b = (a, b) if a < b else (b, a)
        row = self.db.execute(
            "SELECT weight FROM links WHERE a=? AND b=?", (a, b)
        ).fetchone()
        if row:
            nw = row["weight"] + w * (1.0 - row["weight"])
            self.db.execute(
                "UPDATE links SET weight=?, last_ts=? WHERE a=? AND b=?",
                (nw, time.time(), a, b),
            )
        else:
            self.db.execute(
                "INSERT INTO links(a, b, weight, last_ts) VALUES(?,?,?,?)",
                (a, b, min(w, 1.0), time.time()),
            )
        self.db.commit()

    def neighbors(self, ids: list[int]) -> dict[int, list[tuple[int, float]]]:
        """id -> [(neighbor_id, weight), ...] for each requested id."""
        out: dict[int, list[tuple[int, float]]] = {i: [] for i in ids}
        if not ids:
            return out
        q = ",".join("?" * len(ids))
        for r in self.db.execute(
            f"SELECT a, b, weight FROM links WHERE a IN ({q}) OR b IN ({q})",
            ids + ids,
        ).fetchall():
            if r["a"] in out:
                out[r["a"]].append((r["b"], r["weight"]))
            if r["b"] in out:
                out[r["b"]].append((r["a"], r["weight"]))
        return out

    # ---------- schemas ----------

    def set_profile(self, gist: str, source_ids: list[int]):
        now = time.time()
        row = self.db.execute("SELECT id FROM schemas WHERE kind='profile'").fetchone()
        if row:
            self.db.execute(
                "UPDATE schemas SET gist=?, source_ids=?, updated_ts=? WHERE id=?",
                (gist, json.dumps(source_ids), now, row["id"]),
            )
        else:
            self.db.execute(
                "INSERT INTO schemas(kind, gist, source_ids, created_ts, updated_ts)"
                " VALUES('profile',?,?,?,?)",
                (gist, json.dumps(source_ids), now, now),
            )
        self.db.commit()

    def get_profile(self):
        return self.db.execute(
            "SELECT * FROM schemas WHERE kind='profile'"
        ).fetchone()

    # ---------- meta ----------

    def set_meta(self, key: str, value):
        self.db.execute(
            "INSERT INTO meta(key, value) VALUES(?,?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )
        self.db.commit()

    def get_meta(self, key: str):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else None

    def stats(self) -> dict:
        return {
            "episodes": self.db.execute("SELECT COUNT(*) c FROM episodes").fetchone()["c"],
            "traces": self.db.execute(
                "SELECT COUNT(*) c FROM traces WHERE status='active'"
            ).fetchone()["c"],
            "queue_depth": self.queue_depth(),
            "degraded_recalls": self.get_meta("degraded_recalls") or 0,
        }
