"""Write-behind summarizer: drains the job queue only when the proxy is idle.

Asks the summarizer model (non-thinking sampling) for two plain lines TITLE:/GIST: parsed by
regex - never JSON, which small models reliably botch. Deterministic extractive
fallback means correctness requires zero LLM competence.
"""

import asyncio
import re
import time

import aiohttp

from . import formation, profile
from .config import Config
from .recall import cue_tokens

_TITLE_RE = re.compile(r"^\s*TITLE:\s*(.+)$", re.MULTILINE)
_GIST_RE = re.compile(r"^\s*GIST:\s*(.+)$", re.MULTILINE)
# Unterminated <think> (truncated generation) swallows the rest - correctly,
# since everything after an unclosed tag IS reasoning, not answer.
_THINK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)

SYSTEM_PROMPT = (
    "You compress one chat exchange into memory notes about what the USER"
    " stated. Copy the user's names, numbers, and facts EXACTLY as written -"
    " never generalize. Ignore the assistant's advice, opinions, and"
    " elaboration; the assistant text is only context. Reply with exactly two"
    " lines:\n"
    "TITLE: <at most 10 words naming the user's key fact>\n"
    "GIST: <at most 100 words restating the user's facts with exact names and values>"
)

# One-shot example: small models follow demonstrations far better than rules.
# The demonstration also shows assistant content being ignored.
ONE_SHOT_USER = (
    "USER said: My staging deploy keeps failing with ECONNRESET until I set"
    " PGSSLMODE=require, super annoying\n"
    "ASSISTANT said: That points to the Postgres TLS handshake being dropped."
    " You could also consider connection pooling, or upgrading your driver..."
)
ONE_SHOT_REPLY = (
    "TITLE: Staging deploys need PGSSLMODE=require to avoid ECONNRESET\n"
    "GIST: The user's staging deploy fails with ECONNRESET unless"
    " PGSSLMODE=require is set."
)
# Second demonstration: WHO said/decided/wants something is part of the fact,
# and nothing gets added that the user did not say.
TWO_SHOT_USER = (
    "USER said: Priya needs the audit report finished by March 3rd, that's a"
    " hard deadline\n"
    "ASSISTANT said: Understood! A few tips for hitting deadlines: break the"
    " work into milestones, leave buffer time..."
)
TWO_SHOT_REPLY = (
    "TITLE: Priya set a hard March 3rd deadline for the audit report\n"
    "GIST: Priya needs the audit report finished by March 3rd; the user called"
    " it a hard deadline."
)


def extractive_fallback(user_text: str, assistant_text: str) -> tuple[str, str]:
    words = user_text.split()
    title = " ".join(words[:10]) or "(empty)"
    gist = " ".join(words[:60])
    a = " ".join(assistant_text.split()[:40])
    if a:
        gist = f"{gist} / assistant: {a}"
    return title, gist


class Summarizer:
    """Runs as a single asyncio task inside the proxy process."""

    def __init__(self, store, recall, cfg: Config, http: aiohttp.ClientSession, activity):
        self.store = store
        self.recall = recall
        self.cfg = cfg
        self.http = http
        self.activity = activity  # object with .in_flight and .last_done

    async def run_forever(self):
        while True:
            try:
                worked = await self.tick()
                if not worked:
                    await self.maybe_consolidate()
                    await self.maybe_topics()
                    self.maybe_decay()
            except Exception:
                worked = False
            await asyncio.sleep(0.5 if worked else 1.5)

    def maybe_decay(self):
        """Forgetting as retrieval failure: traces whose activation stays
        below the silence line past the minimum age drift to silent - still
        on disk, recoverable via MCP, never injected. Pinned traces are
        exempt. Pure arithmetic, no LLM."""
        if time.time() - (self.store.get_meta("decay_last_sweep") or 0) < self.cfg.decay_sweep_cooldown_s:
            return
        self.store.set_meta("decay_last_sweep", time.time())
        import json as _json

        from .activation import base_level

        now = time.time()
        min_age = self.cfg.silence_min_age_days * 86400
        silenced = 0
        for r in self.store.db.execute(
            "SELECT id, last8, n_access, created_ts, beta FROM traces"
            " WHERE status='active' AND pinned=0 AND created_ts < ?",
            (now - min_age,),
        ).fetchall():
            act = base_level(
                _json.loads(r["last8"]), r["n_access"], r["created_ts"],
                now, self.cfg.decay_d,
            ) + (r["beta"] or 0.0)
            if act < self.cfg.silence_activation:
                self.store.silence(r["id"])
                silenced += 1
        if silenced:
            total = (self.store.get_meta("decayed_total") or 0) + silenced
            self.store.set_meta("decayed_total", total)

    async def maybe_consolidate(self):
        """Replay: rebuild the user profile from user-source traces when
        enough new material accumulated - same idle discipline as summaries."""
        if self.activity.in_flight > 0:
            return
        if time.time() - self.activity.last_done < self.cfg.idle_before_summarize_s:
            return
        if self.store.queue_depth() > 0:
            return
        last_try = self.store.get_meta("profile_last_attempt") or 0
        if time.time() - last_try < self.cfg.profile_attempt_cooldown_s:
            return
        hw = self.store.get_meta("profile_high_water") or 0
        fresh = self.store.db.execute(
            "SELECT COUNT(*) c, MAX(id) m FROM traces WHERE status='active' AND"
            " (source='user' OR source IS NULL) AND id > ?", (hw,)
        ).fetchone()
        threshold = (
            self.cfg.profile_min_traces
            if self.store.get_profile() is None
            else self.cfg.profile_refresh_traces
        )
        if (fresh["c"] or 0) < threshold:
            return
        if await self._big_model_resident():
            return
        self.store.set_meta("profile_last_attempt", time.time())

        rows = profile.select_sources(self.store)
        if not rows:
            return
        try:
            async with asyncio.timeout(240):
                async with self.http.post(
                    f"{self.cfg.services_url}/api/chat",
                    json={
                        "model": self.cfg.profile_model or self.cfg.summarizer_model,
                        "stream": False, "think": False, "keep_alive": "2m",
                        "options": {"temperature": 0.2, "top_p": 0.8, "num_predict": 400},
                        "messages": profile.build_messages(rows),
                    },
                ) as resp:
                    data = await resp.json()
            text = profile.validate(data["message"]["content"])
        except Exception:
            return
        if text is None:
            return
        self.store.set_profile(text, [r["id"] for r in rows])
        self.store.set_meta("profile_high_water", fresh["m"] or hw)

    async def maybe_topics(self):
        """Thematic consolidation: cluster related user memories and keep one
        topic summary per cluster. Rebuilt whole from traces each time
        (derived cache, single-level, no compounding drift)."""
        if self.activity.in_flight > 0 or self.store.queue_depth() > 0:
            return
        if time.time() - self.activity.last_done < self.cfg.idle_before_summarize_s:
            return
        if time.time() - (self.store.get_meta("topic_last_attempt") or 0) < self.cfg.topic_cooldown_s:
            return
        hw = self.store.get_meta("topic_high_water") or 0
        fresh = self.store.db.execute(
            "SELECT COUNT(*) c, MAX(id) m FROM traces WHERE status='active'"
            " AND (source='user' OR source IS NULL) AND id > ?", (hw,)
        ).fetchone()
        if (fresh["c"] or 0) < self.cfg.topic_refresh_traces:
            return
        if await self._big_model_resident():
            return
        self.store.set_meta("topic_last_attempt", time.time())

        clusters = profile.cluster_topics(self.store)
        if not clusters:
            self.store.set_meta("topic_high_water", fresh["m"] or hw)
            return
        results = []
        for rows, centroid in clusters:
            try:
                async with asyncio.timeout(120):
                    async with self.http.post(
                        f"{self.cfg.services_url}/api/chat",
                        json={
                            "model": self.cfg.profile_model or self.cfg.summarizer_model,
                            "stream": False, "think": False, "keep_alive": "2m",
                            "options": {"temperature": 0.2, "top_p": 0.8, "num_predict": 220},
                            "messages": profile.build_topic_messages(rows),
                        },
                    ) as resp:
                        data = await resp.json()
                text = profile.validate_topic(data["message"]["content"])
            except Exception:
                text = None
            if text is not None:
                results.append((text, [r["id"] for r in rows], centroid))
        if results:
            self.store.replace_topics(results)
            self.store.set_meta("topic_high_water", fresh["m"] or hw)

    async def tick(self) -> bool:
        """Process one job if the proxy is idle. Returns True if a job ran."""
        if self.activity.in_flight > 0:
            return False
        if time.time() - self.activity.last_done < self.cfg.idle_before_summarize_s:
            return False
        job = self.store.next_job()
        if job is None:
            return False
        if self.activity.in_flight > 0:
            return False

        user_text = (job["user_text"] or "").strip()
        assistant_text = (job["assistant_text"] or "").strip()
        source = job["source"] if "source" in job.keys() else "user"
        # Formation gates: episodes stay logged, but not everything becomes
        # a memory. Framework plumbing is never memory; pure questions produce
        # derived content, not user knowledge; content-free turns are noise.
        if (
            source == "system"
            or len(user_text) < self.cfg.min_episode_chars
            or len(cue_tokens(user_text)) < self.cfg.min_content_tokens
            or formation.is_pure_question(user_text)
        ):
            self.store.finish_job(job["job_id"])
            return True

        title, gist, quality = None, None, "llm"
        # If a big chat model is resident (an agent mid-conversation), loading
        # the summarizer model would evict it and thrash the GPU - take the
        # extractive path instead; facts survive verbatim.
        big = await self._big_model_resident()
        if not big and job["attempts"] < 2:
            title, gist = await self._llm_summarize(user_text, assistant_text)
        if title is None:
            if not big and job["attempts"] < 2:
                self.store.bump_job(job["job_id"])
                return True
            title, gist = extractive_fallback(user_text, assistant_text)
            quality = "extractive"

        # Embed gist + a verbatim slice (off-path, so a long timeout is fine):
        # the embedding should match future cues even when the gist is lossy.
        emb = await self.recall.embed(
            f"{title}. {gist}\n{user_text[:300]}", timeout=10.0, role="doc"
        )

        # ADD vs REINFORCE vs SUPERSEDE, decided with arithmetic:
        # a repeat strengthens the existing trace (co-allocation), but a
        # CORRECTION must supersede the stale trace, never feed it.
        near_id, near_cos, conflict = None, 0.0, False
        if emb is not None:
            near_id, near_cos = self.store.max_cosine(emb)
            if near_id is not None and near_cos >= self.cfg.conflict_cosine:
                old = self.store.get_traces([near_id]).get(near_id)
                if old is not None:
                    conflict = formation.is_conflicting(
                        f"{old['title']}. {old['gist']}", user_text, near_cos
                    )
                    # A third party's words never supersede a fact the USER
                    # stated - both live; the user's outranks via beta.
                    old_src = old["source"] if "source" in old.keys() else "user"
                    if conflict and old_src == "user" and source != "user":
                        conflict = False
            if (
                near_id is not None
                and near_cos >= self.cfg.near_dup_cosine
                and not conflict
            ):
                self.store.add_access(near_id, self.cfg.w_repeat)
                self.store.finish_job(job["job_id"])
                return True

        novelty = 1.0 - near_cos if near_id is not None else 1.0
        tid = self.store.add_trace(
            episode_id=job["id"], title=title, gist=gist,
            session_id=job["session_id"], embedding=emb, quality=quality,
            fts_extra=user_text[:400],
            beta=formation.beta(novelty, user_text, source), source=source,
        )
        if conflict and near_id is not None:
            self.store.supersede(near_id, tid)
        self._form_links(tid, job["session_id"], emb)
        self.store.finish_job(job["job_id"])
        return True

    def _form_links(self, tid: int, session_id: str, emb):
        """Wire the new trace into the memory graph: temporal co-allocation
        (same session, within 6h - the empirical linking window) and semantic
        neighbors. Recalling one trace will prime these at retrieval."""
        try:
            for r in self.store.db.execute(
                "SELECT id FROM traces WHERE session_id=? AND id!=? AND"
                " status='active' AND created_ts > ? ORDER BY id DESC LIMIT 5",
                (session_id, tid, time.time() - 6 * 3600),
            ).fetchall():
                self.store.add_link(tid, r["id"], 0.3)
            if emb is not None:
                for oid, cos_ in self.store.dense_search(emb, 6):
                    if oid != tid and 0.60 <= cos_ < self.cfg.near_dup_cosine:
                        self.store.add_link(tid, oid, 0.5 * cos_)
        except Exception:
            pass  # links are an enhancement; formation must never fail on them

    async def _big_model_resident(self) -> bool:
        try:
            async with asyncio.timeout(2):
                async with self.http.get(f"{self.cfg.services_url}/api/ps") as resp:
                    data = await resp.json()
            return any(
                m.get("size", 0) >= self.cfg.big_model_bytes
                and m.get("name") != self.cfg.summarizer_model
                for m in data.get("models", [])
            )
        except Exception:
            return False

    async def _llm_summarize(self, user_text, assistant_text) -> tuple[str | None, str | None]:
        prompt = (
            f"USER said: {user_text[:1200]}\n"
            f"ASSISTANT said: {assistant_text[:600]}"
        )
        try:
            async with asyncio.timeout(30):
                async with self.http.post(
                    f"{self.cfg.services_url}/api/chat",
                    json={
                        "model": self.cfg.summarizer_model,
                        "stream": False,
                        "think": False,
                        "keep_alive": "2m",
                        "options": {"temperature": 0.2, "top_p": 0.8, "num_predict": 220},
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": ONE_SHOT_USER},
                            {"role": "assistant", "content": ONE_SHOT_REPLY},
                            {"role": "user", "content": TWO_SHOT_USER},
                            {"role": "assistant", "content": TWO_SHOT_REPLY},
                            {"role": "user", "content": prompt},
                        ],
                    },
                ) as resp:
                    data = await resp.json()
            text = _THINK_RE.sub("", data["message"]["content"]).strip()
        except Exception:
            return None, None

        # Take the LAST match of each: a re-echoed one-shot demonstration
        # precedes the real answer.
        gists = _GIST_RE.findall(text)
        if not gists:
            return None, None
        gist = " ".join(gists[-1].split())
        titles = _TITLE_RE.findall(text)
        # Lenient: a GIST-only reply is usable - synthesize the title from it.
        title = " ".join(titles[-1].split()) if titles else " ".join(gist.split()[:10])
        if not (1 <= len(title.split()) <= 15) or not (3 <= len(gist.split()) <= 130):
            return None, None
        # One-shot leak guard: example-specific tokens appearing in the output
        # but not in the source exchange mean the model replayed the demo.
        source = f"{user_text} {assistant_text}".lower()
        for marker in ("econnreset", "pgsslmode", "priya", "march 3", "audit report"):
            if marker in f"{title} {gist}".lower() and marker not in source:
                return None, None
        return title, gist

    async def keep_warm(self):
        while True:
            try:
                # Pinning the embedder is a warm-latency optimization, never
                # worth evicting an agent's resident model for.
                if not await self._big_model_resident():
                    await self.recall.embed("keep-warm ping", timeout=5.0)
            except Exception:
                pass
            await asyncio.sleep(self.cfg.keep_warm_interval_s)
