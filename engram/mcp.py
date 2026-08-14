"""engram mcp: stdio MCP server exposing memory verbs to agents.

Transparent injection serves models that can't call tools; this server is the
explicit surface for agents that can. Both share one store, so a memory formed
while chatting via the proxy is searchable here and vice versa. Tool
descriptions deliberately teach the agent WHEN to reach for each verb.

Corrections supersede (old trace silenced, auditable, never fed); edits rewrite
in place; forgetting silences by default and only deletes when hard=True.
"""

import json
import sys
import urllib.request

import numpy as np

from . import formation
from .config import Config
from .recall import cue_tokens
from .store import Store

QUERY_INSTRUCT = (
    "Instruct: Given a user message, retrieve past conversation memories"
    " relevant to answering it\nQuery: "
)


class MemoryTools:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.store = Store(cfg.db_path, cfg.embed_dim)

    # ---------- helpers ----------

    def _embed(self, text: str, role: str = "doc"):
        model = self.cfg.embed_model
        if "qwen3-embedding" in model and role == "query":
            text = QUERY_INSTRUCT + text
        elif "nomic" in model:
            text = f"search_{'query' if role == 'query' else 'document'}: {text}"
        try:
            req = urllib.request.Request(
                f"{self.cfg.services_url}/api/embed",
                data=json.dumps({"model": model, "input": [text[:2000]]}).encode(),
                headers={"Content-Type": "application/json"},
            )
            d = json.load(urllib.request.urlopen(req, timeout=20))
            v = np.array(d["embeddings"][0], dtype=np.float32)[: self.cfg.embed_dim]
            n = float(np.linalg.norm(v))
            return v / n if n > 0 else None
        except Exception:
            return None

    def _fmt(self, row, extra="") -> str:
        src = row["source"] if "source" in row.keys() and row["source"] else "user"
        tag = "" if src == "user" else f", {src}"
        pin = ", pinned" if ("pinned" in row.keys() and row["pinned"]) else ""
        return (
            f"[#{row['id']}] ({row['status']}{tag}{pin}, n={row['n_access']:.2f}) {row['title']}\n"
            f"    {row['gist']}{extra}"
        )

    def _make_trace(self, fact: str) -> tuple[int, str]:
        """Shared by save/correct: embed, near-dup check, insert."""
        fact = " ".join(fact.split())
        title = " ".join(fact.split()[:10])
        gist = " ".join(fact.split()[:100])
        emb = self._embed(f"{title}. {gist}", role="doc")
        near_id, near_cos = (None, 0.0)
        if emb is not None:
            near_id, near_cos = self.store.max_cosine(emb)
            if near_id is not None and near_cos >= self.cfg.near_dup_cosine:
                self.store.add_access(near_id, self.cfg.w_create)
                return near_id, f"Already known - reinforced existing memory #{near_id}."
        novelty = 1.0 - near_cos if near_id is not None else 1.0
        tid = self.store.add_trace(
            episode_id=None, title=title, gist=gist, session_id="mcp",
            embedding=emb, quality="explicit", fts_extra=fact[:400],
            beta=formation.beta(novelty, fact),
        )
        return tid, f"Saved as memory #{tid}."

    # ---------- tools ----------

    def search(self, query: str, limit: int = 8, include_silent: bool = False) -> str:
        limit = max(1, min(int(limit or 8), 25))
        qvec = self._embed(query, role="query")
        ids = []
        if qvec is not None:
            ids += [tid for tid, _ in self.store.dense_search(qvec, limit * 2)]
        ids += [tid for tid, _, _ in self.store.fts_search(cue_tokens(query), limit * 2)]
        seen, ordered = set(), []
        for tid in ids:
            if tid not in seen:
                seen.add(tid)
                ordered.append(tid)
        rows = self.store.get_traces(ordered[: limit])
        out = [self._fmt(rows[t]) for t in ordered[:limit] if t in rows]
        if include_silent:
            like = f"%{query.strip()[:60]}%"
            for r in self.store.db.execute(
                "SELECT * FROM traces WHERE status IN ('silent','superseded')"
                " AND (title LIKE ? OR gist LIKE ?) ORDER BY id DESC LIMIT ?",
                (like, like, limit),
            ).fetchall():
                out.append(self._fmt(r, extra="  (not injected)"))
        return "\n".join(out) if out else "No matching memories."

    def recent(self, limit: int = 10) -> str:
        rows = self.store.db.execute(
            "SELECT * FROM traces WHERE status='active' ORDER BY id DESC LIMIT ?",
            (max(1, min(int(limit or 10), 50)),),
        ).fetchall()
        return "\n".join(self._fmt(r) for r in rows) or "No memories yet."

    def save(self, fact: str) -> str:
        if not fact or len(cue_tokens(fact)) < 1:
            return "Refusing to save: no substantive content."
        _, msg = self._make_trace(fact)
        return msg

    def correct(self, trace_id: int, corrected_fact: str) -> str:
        old = self.store.get_traces([int(trace_id)]).get(int(trace_id))
        if old is None:
            return f"No active memory #{trace_id}. Use memory_search first."
        if not corrected_fact or len(cue_tokens(corrected_fact)) < 1:
            return "Refusing: corrected fact has no substantive content."
        new_id, _ = self._make_trace(corrected_fact)
        if new_id != int(trace_id):
            self.store.supersede(int(trace_id), new_id)
        return (
            f"Corrected. Memory #{trace_id} ('{old['title']}') is superseded by"
            f" #{new_id}; the old version stays on record but is never injected."
        )

    def edit(self, trace_id: int, title: str = "", gist: str = "") -> str:
        row = self.store.get_traces([int(trace_id)]).get(int(trace_id))
        if row is None:
            return f"No active memory #{trace_id}."
        new_title = " ".join(title.split()) or row["title"]
        new_gist = " ".join(gist.split()) or row["gist"]
        emb = self._embed(f"{new_title}. {new_gist}", role="doc")
        self.store.edit_trace(int(trace_id), new_title, new_gist, emb)
        return f"Memory #{trace_id} updated: {new_title}"

    def forget(self, trace_id: int, hard: bool = False) -> str:
        row = self.store.db.execute(
            "SELECT id, title FROM traces WHERE id=?", (int(trace_id),)
        ).fetchone()
        if row is None:
            return f"No memory #{trace_id}."
        if hard:
            self.store.delete_trace(int(trace_id))
            return f"Memory #{trace_id} ('{row['title']}') permanently deleted."
        self.store.silence(int(trace_id))
        return (
            f"Memory #{trace_id} ('{row['title']}') silenced - kept on record,"
            " never injected. Use hard=true only for privacy-sensitive removal."
        )

    def pin(self, trace_id: int, pinned: bool = True) -> str:
        row = self.store.get_traces([int(trace_id)]).get(int(trace_id))
        if row is None:
            return f"No active memory #{trace_id}."
        self.store.set_pinned(int(trace_id), pinned)
        if pinned:
            return (
                f"Memory #{trace_id} ('{row['title']}') pinned: it now outranks"
                " everything at recall and never decays. It can still be"
                " superseded by a correction."
            )
        return f"Memory #{trace_id} ('{row['title']}') unpinned: normal decay resumes."

    def profile(self, refresh: bool = False) -> str:
        from . import profile as prof

        if refresh:
            rows = prof.select_sources(self.store)
            if not rows:
                return "No user-source memories to build a profile from yet."
            try:
                req = urllib.request.Request(
                    f"{self.cfg.services_url}/api/chat",
                    data=json.dumps({
                        "model": self.cfg.profile_model or self.cfg.summarizer_model,
                        "stream": False, "think": False, "keep_alive": "2m",
                        "options": {"temperature": 0.2, "top_p": 0.8, "num_predict": 400},
                        "messages": prof.build_messages(rows),
                    }).encode(),
                    headers={"Content-Type": "application/json"},
                )
                d = json.load(urllib.request.urlopen(req, timeout=300))
                text = prof.validate(d["message"]["content"])
            except Exception as e:
                return f"Profile rebuild failed: {e}"
            if text is None:
                return "Profile rebuild produced invalid output; kept the previous profile."
            self.store.set_profile(text, [r["id"] for r in rows])
            self.store.set_meta(
                "profile_high_water", max(r["id"] for r in rows)
            )
        row = self.store.get_profile()
        if row is None:
            return ("No profile yet - it forms automatically once enough"
                    " user-stated memories accumulate, or call with refresh=true.")
        ids = json.loads(row["source_ids"] or "[]")
        return f"User profile (distilled from {len(ids)} memories):\n{row['gist']}"

    def stats(self) -> str:
        return json.dumps(self.store.stats())


TOOLS = [
    {
        "name": "memory_search",
        "description": (
            "Search the user's long-term memory (Engram). ALWAYS use this before"
            " answering questions about the user's past, preferences, people,"
            " projects, or prior decisions - do not guess from conversation"
            " context alone. Returns memories with their #ids for use with the"
            " other memory tools. include_silent also shows superseded/forgotten"
            " versions (history)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "description": "default 8"},
                "include_silent": {"type": "boolean", "description": "include superseded/forgotten history"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_recent",
        "description": "List the most recently formed memories with their #ids.",
        "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer"}}},
    },
    {
        "name": "memory_save",
        "description": (
            "Explicitly save a fact to long-term memory. Use when the user asks"
            " you to remember something, or states an important durable fact"
            " (identity, preference, deadline, credential-adjacent detail)."
            " State the fact plainly with exact names and values."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"fact": {"type": "string"}},
            "required": ["fact"],
        },
    },
    {
        "name": "memory_correct",
        "description": (
            "Correct a WRONG or OUTDATED memory. Use when the user says a"
            " remembered fact is no longer true ('actually it's X now')."
            " Finds nothing to correct? memory_search first to get the #id."
            " The old memory is superseded (kept for history, never injected);"
            " the corrected fact replaces it."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "trace_id": {"type": "integer", "description": "#id of the wrong memory"},
                "corrected_fact": {"type": "string", "description": "the fact as it should be"},
            },
            "required": ["trace_id", "corrected_fact"],
        },
    },
    {
        "name": "memory_edit",
        "description": (
            "Rewrite an existing memory's title/gist in place (fix garbled"
            " wording, add a missed detail). For facts that CHANGED, use"
            " memory_correct instead so history is preserved."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "trace_id": {"type": "integer"},
                "title": {"type": "string"},
                "gist": {"type": "string"},
            },
            "required": ["trace_id"],
        },
    },
    {
        "name": "memory_forget",
        "description": (
            "Forget a memory. Default silences it (kept on record, never"
            " injected, reversible). hard=true permanently deletes - only for"
            " privacy-sensitive content the user wants gone."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "trace_id": {"type": "integer"},
                "hard": {"type": "boolean"},
            },
            "required": ["trace_id"],
        },
    },
    {
        "name": "memory_pin",
        "description": (
            "Pin a memory the user marks as permanently important (hard"
            " constraints, allergies, standing rules): it outranks everything"
            " at recall and never decays. pinned=false unpins. Corrections"
            " still supersede pinned memories."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "trace_id": {"type": "integer"},
                "pinned": {"type": "boolean", "description": "default true"},
            },
            "required": ["trace_id"],
        },
    },
    {
        "name": "memory_profile",
        "description": (
            "The consolidated USER PROFILE - distilled durable facts about the"
            " user (identity, preferences, constraints, people, projects)."
            " Use for broad questions like 'what do you know about me'."
            " refresh=true rebuilds it from current memories first."
        ),
        "inputSchema": {"type": "object", "properties": {"refresh": {"type": "boolean"}}},
    },
    {
        "name": "memory_stats",
        "description": "Memory store health: episode/trace counts, queue depth.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def handle(req: dict, tools: MemoryTools):
    method, rid = req.get("method"), req.get("id")
    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": rid,
            "result": {
                "protocolVersion": req.get("params", {}).get("protocolVersion", "2024-11-05"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "engram-memory", "version": "0.3.0"},
            },
        }
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        p = req.get("params", {})
        args = p.get("arguments") or {}
        fn = {
            "memory_search": lambda: tools.search(args.get("query", ""), args.get("limit", 8), args.get("include_silent", False)),
            "memory_recent": lambda: tools.recent(args.get("limit", 10)),
            "memory_save": lambda: tools.save(args.get("fact", "")),
            "memory_correct": lambda: tools.correct(args.get("trace_id", 0), args.get("corrected_fact", "")),
            "memory_edit": lambda: tools.edit(args.get("trace_id", 0), args.get("title", ""), args.get("gist", "")),
            "memory_forget": lambda: tools.forget(args.get("trace_id", 0), args.get("hard", False)),
            "memory_pin": lambda: tools.pin(args.get("trace_id", 0), args.get("pinned", True)),
            "memory_profile": lambda: tools.profile(args.get("refresh", False)),
            "memory_stats": lambda: tools.stats(),
        }.get(p.get("name"))
        try:
            text = fn() if fn else f"Unknown tool: {p.get('name')}"
            return {"jsonrpc": "2.0", "id": rid,
                    "result": {"content": [{"type": "text", "text": text}]}}
        except Exception as e:
            return {"jsonrpc": "2.0", "id": rid,
                    "result": {"content": [{"type": "text", "text": f"error: {e}"}], "isError": True}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": rid, "result": {}}
    if rid is not None:
        return {"jsonrpc": "2.0", "id": rid,
                "error": {"code": -32601, "message": f"unknown method {method}"}}
    return None


def main():
    tools = MemoryTools(Config())
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle(req, tools)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
