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


# A terminator only ends a sentence when followed by whitespace or the end
# of text: "2.5 mg", "v2.5", "$1.5M" must never be split into misquoted
# fragments presented as the verbatim record.
_SENT_RE = re.compile(r"(?:[^.!?\n]|[.!?](?!\s|$))+[.!?]*")
# Verbatim secret material never rides to an upstream (SALIENCE_RE makes
# these episodes the MOST recallable, which is exactly why the raw sentence
# must not travel; the gist's summary already answers).
_SECRET_RE = re.compile(
    r"\b(password|passcode|secret|credential|api[_ ]?key|token)\b", re.IGNORECASE
)


def evidence_sentences(
    text: str, gist: str, cue_toks: list[str], k: int = 2, max_chars: int = 220
) -> list[str]:
    """Cue-relevant verbatim sentences from a source episode that the gist
    does not already carry. The gist is a lossy index over a lossless
    record; this is how a big-budget block reaches the record. Pure token
    arithmetic - nothing here may cost more than microseconds."""
    scored = []
    gist_lower = (gist or "").lower()
    for m in _SENT_RE.finditer(text or ""):
        sent = m.group().strip()
        if len(sent) < 15:
            continue
        if "<engram:memory" in sent or "</engram:memory" in sent:
            continue  # an echoed injection must never nest inside the fence
        if _SECRET_RE.search(sent):
            continue
        # dedup on the WHOLE sentence: a prefix probe would suppress exactly
        # the boundary-straddling sentence the extractive gist cut mid-way,
        # whose truncated tail is the detail expansion exists to recover
        probe = sent.lower().rstrip(".!? ")
        if probe and probe in gist_lower:
            continue
        toks = set(_TOKEN_RE.findall(sent.lower()))
        hits = sum(1 for t in cue_toks if t in toks)
        if hits:
            scored.append((hits, _cue_window(sent, cue_toks, max_chars)))
    scored.sort(key=lambda x: (-x[0], len(x[1])))
    return [sent for _, sent in scored[:k]]


def _cue_window(sent: str, cue_toks: list[str], max_chars: int) -> str:
    """A long unpunctuated message is one giant 'sentence'; truncating its
    head reproduces exactly the coverage the extractive gist already has
    and loses the tail detail again. Center the excerpt on the cue hits."""
    if len(sent) <= max_chars:
        return sent
    low = sent.lower()
    hits = [m.start() for t in cue_toks for m in re.finditer(re.escape(t), low)]
    if not hits:
        return sent[:max_chars]
    mid = sum(hits) // len(hits)
    start = max(0, min(mid - max_chars // 2, len(sent) - max_chars))
    frag = sent[start:start + max_chars]
    if start > 0:
        cut = frag.find(" ")
        frag = "..." + (frag[cut + 1:] if 0 <= cut < 40 else frag)
    if start + max_chars < len(sent):
        cut = frag.rfind(" ")
        frag = (frag[:cut] if cut > len(frag) - 40 else frag) + " ..."
    return frag


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
                    "episode_id": row["episode_id"],
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

        # Silent resurrection: when nothing active answered but a silent
        # memory matches the cue strongly, wake it. Only fires on an
        # otherwise-empty recall, so it can never add distractors.
        if not kept and qvec is not None:
            for sid, scos in self.store.silent_search(qvec, 2):
                if scos < self.cfg.resurrect_cosine:
                    break
                if dry_run:
                    scored.append({"id": sid, "title": "(silent match)", "gist": "",
                                   "created_ts": time.time(), "cos": round(scos, 3),
                                   "bm25": 0.0, "nmatch": 0, "activation": 0.0,
                                   "beta": 0.0, "source": "silent",
                                   "score": 0.0, "admitted": False, "_emb": None})
                    continue
                self.store.resurrect(sid)
                self.store.add_access(sid, self.cfg.w_create)
                row = self.store.get_traces([sid]).get(sid)
                if row is not None:
                    kept.append({
                        "id": sid, "title": row["title"], "gist": row["gist"],
                        "created_ts": row["created_ts"], "cos": round(scos, 3),
                        "bm25": 0.0, "nmatch": 0, "activation": 0.0,
                        "beta": round(row["beta"] or 0.0, 2),
                        "source": row["source"] if "source" in row.keys() else "user",
                        "score": round(scos, 3), "admitted": True,
                        "resurrected": True, "_emb": row["embedding"],
                    })

        profile_text = None
        if AGG_RE.search(cue):
            prow = self.store.get_profile()
            if prow is not None:
                profile_text = prow["gist"]

        # Topic schemas: a broad cue near a cluster's centroid injects the
        # cluster's synthesis alongside (or instead of) point traces.
        topic_text = None
        if qvec is not None:
            best = 0.0
            for t in self.store.topics():
                if t["embedding"] is None:
                    continue
                tv = np.frombuffer(t["embedding"], dtype=np.float32)
                if tv.shape[0] != qvec.shape[0]:
                    continue
                c = float(tv @ qvec)
                if c >= self.cfg.topic_admit_cosine and c > best:
                    best, topic_text = c, t["gist"]

        if kept or profile_text or topic_text:
            block, shown = self._render(kept, profile_text, limits, topic_text, tokens)
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

        # Debug carries a "shown" flag per candidate: True = this trace was
        # actually rendered into the block (render can drop admitted traces
        # via near-dup suppression or budget; scoring top-12 is NOT the block).
        # Any shown trace outside the top 12 is appended so the debug list is
        # always a superset of what the model saw - eval harnesses depend on
        # this being exact.
        shown_set = set(shown)
        debug_rows = scored[:12]
        seen = {c["id"] for c in debug_rows}
        debug_rows = debug_rows + [
            c for c in scored[12:] if c["id"] in shown_set and c["id"] not in seen
        ]
        debug = [
            {**{k: v for k, v in c.items() if k != "_emb"}, "shown": c["id"] in shown_set}
            for c in debug_rows
        ]
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
        limits: tuple | None = None, topic_text: str | None = None,
        cue_toks: list[str] | None = None,
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
        # evidence expansion: only tiers with room for it, and only when the
        # cue gives us something to select by
        expand = (
            cue_toks
            and self.cfg.expand_min_budget
            and token_budget >= self.cfg.expand_min_budget
        )
        episodes = self.store.get_episodes(
            [c.get("episode_id") for c in kept[:max_gists]
             if c.get("source", "user") == "user"]
        ) if expand else {}
        header = "Background notes from past sessions (may be stale):"
        budget = token_budget - est_tokens(header) - 12
        profile_part = []
        if profile_text:
            profile_part = ["User profile (distilled from memory):", profile_text]
            budget -= est_tokens(profile_text) + 10
        if topic_text and est_tokens(topic_text) + 8 <= budget:
            profile_part.append(f"Topic summary: {topic_text}")
            budget -= est_tokens(topic_text) + 8
        lines, titles, shown = [], [], []
        gist_slots = []  # (index in lines, candidate) for the evidence pass
        gist_count = 0
        for c in kept:
            if gist_count < max_gists:
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
                    gist_count += 1
                    if expand:
                        gist_slots.append((len(lines) - 1, c))
                    continue
            if len(titles) < max_titles:
                t = est_tokens(c["title"]) + 3
                if t <= budget:
                    titles.append(c["title"])
                    budget -= t
                    shown.append(c["id"])
        # Evidence pass, strictly from leftover budget: every admitted gist
        # and title has already been placed, so verbatim detail can only add,
        # never displace. User-source episodes only (allowlist: system
        # plumbing and observed third-party text never travel verbatim), and
        # only the user's own words - assistant text can embed relayed or
        # generated content and stays summarized-only, per the provenance
        # design. The 4KB cap bounds the scan (episodes are uncapped;
        # a pasted 8MB blob must not eat the recall deadline).
        inserts = []
        for idx, c in gist_slots:
            if c.get("source", "user") != "user":
                continue
            ep = episodes.get(c.get("episode_id"))
            if ep is None:
                continue
            sents = evidence_sentences(
                (ep["user_text"] or "")[:4000], c["gist"], cue_toks,
                k=self.cfg.expand_sentences,
            )
            if not sents:
                continue
            ev = "  > " + " · ".join(f'"{x}"' for x in sents)
            te = est_tokens(ev)
            if te <= budget:
                inserts.append((idx, ev))
                budget -= te
        for idx, ev in sorted(inserts, key=lambda x: -x[0]):
            lines.insert(idx + 1, ev)

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
