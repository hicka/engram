"""Recall path: cue -> hybrid candidates -> admission gate -> activation rank -> block.

Invariants (from the design spec):
  - No LLM on this path, ever. One capped embedder HTTP call at most.
  - Admission uses RAW thresholds (cosine / bm25), never normalized scores,
    so "inject nothing" is the default on unrelated turns.
  - Activation orders admitted candidates; it never admits anything by itself.
  - Traces from the current session are excluded (already in context).
"""

import asyncio
import datetime
import json
import re
import time

import aiohttp
import numpy as np

from .activation import base_level, sigmoid
from .config import Config, SENTINEL_CLOSE, SENTINEL_OPEN

_SENTINEL_RE = re.compile(
    re.escape(SENTINEL_OPEN).replace(r"v=1", r"v=\d+") + r".*?" + re.escape(SENTINEL_CLOSE),
    re.DOTALL,
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-\.]{3,}")

# Identity/aggregate questions have no specific cue to match - they route to
# the consolidated profile instead of (only) point traces.
AGG_RE = re.compile(
    r"\b(about me|who am i|what do you know|know about me|my preferences"
    r"|my profile|everything you (?:know|remember)|remind me everything"
    r"|list everything)\b",
    re.IGNORECASE,
)
_STOP = frozenset(
    "the and for you your are this that with have has was were what when where which "
    "who how can could should would will just like about from not any all out get "
    "want need know think make going really there here they them then than "
    "thanks thank okay yes sure great cool nice good hello hey please help".split()
)


def strip_sentinel(text: str) -> str:
    return _SENTINEL_RE.sub("", text)


def est_tokens(text: str) -> int:
    return int(len(text) / 3.5 * 1.1) + 1


def cue_tokens(cue: str, limit: int = 12) -> list[str]:
    seen, out = set(), []
    for t in _TOKEN_RE.findall(cue.lower()):
        if t in _STOP or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= limit:
            break
    return out


class Recall:
    def __init__(self, store, cfg: Config, http: aiohttp.ClientSession):
        self.store = store
        self.cfg = cfg
        self.http = http
        # per-session injection-reinforcement cap: session_id -> set(trace_id)
        self._injected: dict[str, set[int]] = {}
        # per-session priming state: what was shown last turn spreads
        # activation to its graph neighbors this turn
        self._last_shown: dict[str, list[int]] = {}

    def _prefixed(self, text: str, role: str) -> str:
        """Embedding models need task-specific prefixes for asymmetric retrieval."""
        model = self.cfg.embed_model
        if "qwen3-embedding" in model:
            if role == "query":
                return (
                    "Instruct: Given a user message, retrieve past conversation"
                    f" memories relevant to answering it\nQuery: {text}"
                )
            return text  # documents take no instruction
        if "nomic" in model:
            return f"search_{'query' if role == 'query' else 'document'}: {text}"
        return text

    async def embed(
        self, text: str, timeout: float | None = None, role: str = "query"
    ) -> np.ndarray | None:
        try:
            async with asyncio.timeout(timeout or self.cfg.embed_timeout_s):
                async with self.http.post(
                    f"{self.cfg.services_url}/api/embed",
                    json={
                        "model": self.cfg.embed_model,
                        "input": [self._prefixed(text[:2000], role)],
                    },
                ) as resp:
                    data = await resp.json()
            vec = np.array(data["embeddings"][0], dtype=np.float32)[: self.cfg.embed_dim]
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else None
        except Exception:
            return None

    async def recall(
        self, cue: str, session_id: str, dry_run: bool = False,
        limits: tuple | None = None,
    ) -> tuple[str | None, list[dict]]:
        """Returns (memory_block_or_None, candidate_debug_list).
        dry_run skips reinforcement and last_recall bookkeeping (CLI/bench)."""
        t0 = time.time()
        cue = cue.strip()[:512]
        if not cue:
            return None, []

        tokens = cue_tokens(cue)
        if len(tokens) < self.cfg.min_content_tokens:
            return None, []  # smalltalk ("thanks!") - never worth an injection

        qvec = await self.embed(cue)
        if qvec is None and not dry_run:
            n = (self.store.get_meta("degraded_recalls") or 0) + 1
            self.store.set_meta("degraded_recalls", n)

        dense = self.store.dense_search(qvec, 40) if qvec is not None else []
        lex = self.store.fts_search(tokens, 40)

        dense_rank = {tid: i for i, (tid, _) in enumerate(dense)}
        lex_rank = {tid: i for i, (tid, _, _) in enumerate(lex)}
        cos = dict(dense)
        bm25 = {tid: b for tid, b, _ in lex}
        nmatch = {tid: n for tid, _, n in lex}

        candidates = set(dense_rank) | set(lex_rank)
        rows = self.store.get_traces(list(candidates))

        quarantine = [q.strip() for q in self.cfg.quarantine_sources.split(",") if q.strip()]

        # Spreading activation: last turn's recalled traces prime their graph
        # neighbors (ACT-R S = W * (mas - ln(1+fan)); fan damping keeps hub
        # traces from dominating - structural spam resistance).
        spread: dict[int, float] = {}
        sources = self._last_shown.get(session_id, [])
        if sources:
            import math as _math

            for src, lst in self.store.neighbors(sources).items():
                if not lst:
                    continue
                damp = max(0.0, 2.0 - _math.log(1 + len(lst)))
                for other, w in lst:
                    spread[other] = spread.get(other, 0.0) + w * damp * 0.5

        now = time.time()
        scored = []
        for tid in candidates:
            row = rows.get(tid)
            if row is None or row["session_id"] == session_id:
                continue
            src = row["source"] if "source" in row.keys() else "user"
            if any(src == q or src.startswith(q + ":") or (q == "observed" and src.startswith("observed")) for q in quarantine):
                continue  # quarantined provenance: searchable, never injected
            c = cos.get(tid, 0.0)
            b = bm25.get(tid, 0.0)
            nm = nmatch.get(tid, 0)
            # Admission is raw evidence only: semantic proximity, a strong
            # bm25 hit (large stores), or >=N distinct cue tokens literally
            # present (store-size-independent; carries the degraded path).
            admitted = (
                c >= self.cfg.admit_cosine
                or b >= self.cfg.admit_bm25
                or nm >= self.cfg.admit_lex_tokens
            )
            # RRF (k=10) orders relevance among admitted; activation modulates.
            rrf = 0.0
            if tid in dense_rank:
                rrf += 1.0 / (10 + dense_rank[tid])
            if tid in lex_rank:
                rrf += 1.0 / (10 + lex_rank[tid])
            act = base_level(
                json.loads(row["last8"]), row["n_access"], row["created_ts"],
                now, self.cfg.decay_d,
            ) + (row["beta"] or 0.0)
            if row["pinned"]:
                act += self.cfg.beta_pin
            act += spread.get(tid, 0.0)
            score = rrf * 10.0 + self.cfg.act_weight * sigmoid(act / self.cfg.act_scale)
            scored.append(
                {
                    "id": tid, "title": row["title"], "gist": row["gist"],
                    "created_ts": row["created_ts"], "cos": round(c, 3),
                    "bm25": round(b, 2), "nmatch": nm, "activation": round(act, 2),
                    "beta": round(row["beta"] or 0.0, 2), "source": src,
                    "score": round(score, 4), "admitted": admitted,
                    "_emb": row["embedding"],
                }
            )
        scored.sort(key=lambda x: -x["score"])

        admitted = [c for c in scored if c["admitted"]]
        # near-duplicate suppression among admitted (keep the higher-scored)
        kept = []
        for c in admitted:
            dup = False
            if c["_emb"] is not None:
                v = np.frombuffer(c["_emb"], dtype=np.float32)
                for k in kept:
                    if k["_emb"] is not None:
                        w = np.frombuffer(k["_emb"], dtype=np.float32)
                        if float(v @ w) >= self.cfg.near_dup_cosine:
                            dup = True
                            break
            if not dup:
                kept.append(c)

        profile_text = None
        if AGG_RE.search(cue):
            prow = self.store.get_profile()
            if prow is not None:
                profile_text = prow["gist"]

        if kept or profile_text:
            block, shown = self._render(kept, profile_text, limits)
        else:
            block, shown = None, []
        if block is not None and not dry_run:
            self._reinforce(shown, session_id)
            # priming state + hebbian co-retrieval wiring among what was shown
            if session_id not in self._last_shown and len(self._last_shown) >= 512:
                self._last_shown.pop(next(iter(self._last_shown)))
            self._last_shown[session_id] = shown
            for i in range(min(len(shown), 4)):
                for j in range(i + 1, min(len(shown), 4)):
                    self.store.add_link(shown[i], shown[j], 0.1)

        debug = [{k: v for k, v in c.items() if k != "_emb"} for c in scored[:12]]
        if not dry_run:
            self.store.set_meta(
                "last_recall",
                {
                    "ts": now, "cue": cue, "ms": round((time.time() - t0) * 1000, 1),
                    "semantic": qvec is not None, "block": block, "candidates": debug,
                },
            )
        return block, debug

    def _reinforce(self, trace_ids: list[int], session_id: str):
        if session_id not in self._injected and len(self._injected) >= 512:
            self._injected.pop(next(iter(self._injected)))  # evict oldest session
        seen = self._injected.setdefault(session_id, set())
        for tid in trace_ids:
            if tid not in seen:
                seen.add(tid)
                self.store.add_access(tid, self.cfg.w_inject)

    def _render(
        self, kept: list[dict], profile_text: str | None = None,
        limits: tuple | None = None,
    ) -> tuple[str | None, list[int]]:
        """Fit admitted traces to the token budget: gist tier first, overflow
        demotes to the title tier (never silently dropped). Returns the block
        and the ids actually shown - only those earn injection reinforcement.
        For aggregate cues, the consolidated profile leads the block. `limits`
        (max_gists, max_titles, token_budget) overrides the tier-S defaults
        when the upstream model can exploit more."""
        max_gists, max_titles, token_budget = limits or (
            self.cfg.max_gists, self.cfg.max_titles, self.cfg.token_budget
        )
        header = "Background notes from past sessions (may be stale):"
        budget = token_budget - est_tokens(header) - 12
        profile_part = []
        if profile_text:
            profile_part = ["User profile (distilled from memory):", profile_text]
            budget -= est_tokens(profile_text) + 10
        lines, titles, shown = [], [], []
        for c in kept:
            if len(lines) < max_gists:
                date = datetime.datetime.fromtimestamp(c["created_ts"]).strftime("%Y-%m-%d")
                if self.store.supersedes_something(c["id"]):
                    date = f"updated {date}"
                src = c.get("source", "user")
                attribution = (
                    f" (reported by {src.split(':', 1)[1]}, unverified)"
                    if src.startswith("observed:") else ""
                )
                line = f"- [{date}]{attribution} {c['gist']}"
                t = est_tokens(line)
                if t <= budget:
                    lines.append(line)
                    budget -= t
                    shown.append(c["id"])
                    continue
            if len(titles) < max_titles:
                t = est_tokens(c["title"]) + 3
                if t <= budget:
                    titles.append(c["title"])
                    budget -= t
                    shown.append(c["id"])
        if not lines and not titles and not profile_part:
            return None, []
        parts = [SENTINEL_OPEN, *profile_part]
        if lines or titles:
            parts.append(header)
            parts.extend(lines)
        if titles:
            parts.append("Also on file: " + " | ".join(titles))
        parts.append(SENTINEL_CLOSE)
        return "\n".join(parts), shown


def inject_block(messages: list[dict], block: str, at_end: bool = False) -> list[dict]:
    """Idempotently place the memory block.

    Default: top of the system message (positional primacy - best for small
    models with small prompts). at_end: append to the newest user message -
    for big agent prompts (Hermes/OpenClaw), a block at the top invalidates
    the entire llama.cpp prompt-prefix cache on every request (measured:
    ~90s re-prefill of a 23k-token prompt vs 0.1s cached), while a block on
    the newest message leaves the whole cached prefix intact."""
    out = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        content = m.get("content")
        if isinstance(content, str) and SENTINEL_OPEN.split(" ")[0] in content:
            m = {**m, "content": strip_sentinel(content).strip()}
        out.append(m)
    if at_end:
        for i in range(len(out) - 1, -1, -1):
            m = out[i]
            if (
                isinstance(m, dict)
                and m.get("role") == "user"
                and isinstance(m.get("content"), str)
            ):
                out[i] = {**m, "content": f"{m['content']}\n\n{block}"}
                return out
        # no user message found - fall through to system placement
    if (
        out
        and isinstance(out[0], dict)
        and out[0].get("role") == "system"
        and isinstance(out[0].get("content"), str)
    ):
        out[0] = {**out[0], "content": f"{block}\n\n{out[0]['content']}".strip()}
    else:
        out.insert(0, {"role": "system", "content": block})
    return out
