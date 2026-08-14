"""engramd: transparent dual-protocol memory proxy in front of Ollama.

Memory-enabled routes: POST /api/chat (NDJSON), POST /v1/chat/completions (SSE).
Everything else - /api/generate, /api/tags, /api/show, /v1/models, embeddings -
is byte passthrough so client auto-discovery and raw completions keep working.

The response is never re-serialized: bytes stream to the client verbatim while a
shadow accumulator builds the episode log. Any recall failure or deadline breach
fails open to an uninjected passthrough.
"""

import asyncio
import hashlib
import json
import re
import time
import uuid
from pathlib import Path

import aiohttp
from aiohttp import web
from multidict import CIMultiDict

# The shadow accumulator only needs enough of the response to build the
# episode log; a 100k-token agent generation must not hold tens of MB.
MAX_SHADOW_BYTES = 8 * 1024 * 1024

from . import formation
from .activation import base_level
from .config import Config
from .recall import Recall, est_tokens, inject_block, strip_sentinel
from .store import Store
from .summarizer import Summarizer, _THINK_RE

# Hop-controlled headers, dropped in every direction. content-encoding is here
# because aiohttp transcodes bodies on both sides (auto-decompress), so the
# wire-encoding header must never survive the hop. x-engram-* are consumed by
# the proxy itself.
HOP_HEADERS = {
    "host", "content-length", "transfer-encoding", "connection", "keep-alive",
    "accept-encoding", "content-encoding", "te", "upgrade",
    "proxy-authorization", "proxy-authenticate",
    "x-engram", "x-engram-session", "x-engram-budget", "x-engram-agent",
}


class Activity:
    def __init__(self):
        self.in_flight = 0
        self.last_done = 0.0


async def _index_refresher(store: Store):
    """The MCP server (a separate process) writes to the shared store; rebuild
    the in-memory dense index periodically so agent-made memories and edits
    become recallable without a daemon restart. Milliseconds at this scale."""
    while True:
        await asyncio.sleep(60)
        try:
            store._rebuild_index()
        except Exception:
            pass


async def _reembed_if_model_changed(store: Store, recall, cfg: Config):
    """Stored vectors are only comparable within one embedding model - on a
    model switch, re-embed every trace and rebuild the dense index.

    All embeds are buffered before any write, then vectors + the model stamp
    commit in one synchronous transaction with no await in between: a timeout,
    crash, or config revert always leaves the store self-consistent (all-old
    under the old stamp, or all-new under the new)."""
    stamp = f"{cfg.embed_model}@{cfg.embed_dim}"
    if store.get_meta("embed_model") == stamp:
        return
    rows = store.db.execute(
        "SELECT t.id, t.title, t.gist, e.user_text FROM traces t"
        " LEFT JOIN episodes e ON e.id=t.episode_id WHERE t.status='active'"
    ).fetchall()
    pending = []
    for r in rows:
        text = f"{r['title']}. {r['gist']}\n{(r['user_text'] or '')[:300]}"
        emb = await recall.embed(text, timeout=30.0, role="doc")
        if emb is None:
            return  # embedder unavailable; DB untouched, retry next boot
        pending.append((emb.astype("float32").tobytes(), r["id"]))
    try:
        store.db.executemany("UPDATE traces SET embedding=? WHERE id=?", pending)
        store.set_meta("embed_model", stamp)  # commits vectors + stamp together
    except Exception:
        store.db.rollback()
        raise
    store._rebuild_index()


def make_app(cfg: Config) -> web.Application:
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app["cfg"] = cfg
    app["sessions"] = {}  # first-message-hash -> [session_uuid, last_seen]

    async def on_startup(app):
        store = Store(cfg.db_path, cfg.embed_dim)
        http = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, connect=5, sock_read=600)
        )
        recall = Recall(store, cfg, http)
        activity = Activity()
        summarizer = Summarizer(store, recall, cfg, http, activity)
        app["store"], app["http"], app["recall"], app["activity"] = (
            store, http, recall, activity,
        )
        app["tasks"] = [
            asyncio.create_task(summarizer.run_forever()),
            asyncio.create_task(summarizer.keep_warm()),
            asyncio.create_task(_reembed_if_model_changed(store, recall, cfg)),
            asyncio.create_task(_index_refresher(store)),
        ]

    async def on_cleanup(app):
        for t in app["tasks"]:
            t.cancel()
        await asyncio.gather(*app["tasks"], return_exceptions=True)
        await app["http"].close()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    app.router.add_post("/api/chat", handle_chat_native)
    app.router.add_post("/v1/chat/completions", handle_chat_openai)
    # Anthropic Messages protocol (e.g. MiniMax's /anthropic/v1/messages) -
    # cloud clients mount it under arbitrary prefixes, so match both forms.
    app.router.add_post("/v1/messages", handle_chat_anthropic)
    app.router.add_post("/{prefix:.+}/v1/messages", handle_chat_anthropic)
    # Companion Messages-API endpoints (count_tokens, batches - GET included)
    # must follow the same upstream, or the client goes split-brained: chat to
    # the cloud, token counting 404ing against local Ollama. Prefixed /v1/models
    # is unambiguously Anthropic-surface too; bare /v1/models stays local for
    # OpenAI-protocol clients.
    app.router.add_route("*", "/v1/messages/{rest:.+}", handle_anthropic_passthrough)
    app.router.add_route("*", "/{prefix:.+}/v1/messages/{rest:.+}", handle_anthropic_passthrough)
    app.router.add_route("*", "/{prefix:.+}/v1/models{rest:(/.*)?}", handle_anthropic_passthrough)
    app.router.add_get("/engram/stats", handle_stats)
    app.router.add_get("/engram/why", handle_why)
    app.router.add_get("/engram/memories", handle_memories)
    app.router.add_get("/engram/ui", handle_ui)
    app.router.add_route("*", "/{tail:.*}", handle_passthrough)
    return app


# ---------------- shared helpers ----------------

def _last_user_text(messages) -> str | None:
    if not messages or not isinstance(messages, list):
        return None
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        return None
    content = last.get("content")
    return content if isinstance(content, str) else None


def _session_id(request, body) -> str:
    """Session identity: X-Engram-Session header wins; otherwise the first
    user message keys an in-memory registry that mints a fresh uuid per
    conversation, TTL-scoped - so a habitual opener ("hi") tomorrow is a NEW
    session and yesterday's traces are not self-excluded from recall."""
    hdr = request.headers.get("X-Engram-Session")
    if hdr:
        return hdr[:64]
    first_user = ""
    for m in body.get("messages") or []:
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            # Flatten block lists to their text: volatile non-text keys must
            # not fork the session identity between turns.
            first_user = c if isinstance(c, str) else (_anthropic_text(c) or json.dumps(c)[:200])
            break
    seed = hashlib.sha1(f"{body.get('model', '')}\x00{first_user}".encode()).hexdigest()

    sessions = request.app["sessions"]
    now = time.time()
    ttl = request.app["cfg"].session_ttl_s
    entry = sessions.get(seed)
    if entry is None or now - entry[1] > ttl:
        entry = [uuid.uuid4().hex[:16], now]
        sessions[seed] = entry
    entry[1] = now
    if len(sessions) > 1024:  # opportunistic eviction of stale conversations
        for k in [k for k, v in sessions.items() if now - v[1] > 2 * ttl]:
            del sessions[k]
    return entry[0]


def _parse_param_size(ps: str) -> float:
    """'1.7B' / '752.75M' -> parameter count; 0 when unknown."""
    m = re.match(r"([\d.]+)\s*([MB])", (ps or "").strip())
    if not m:
        return 0.0
    return float(m.group(1)) * (1e6 if m.group(2) == "M" else 1e9)


async def _tier_limits(app, model: str, cloud: bool) -> tuple | None:
    """Injection budget scaled to the upstream model. None = tier-S defaults.
    Cloud upstreams are frontier-class: tier L. Local models are probed once
    via /api/show and cached; probe failure degrades to tier S (safe)."""
    cfg: Config = app["cfg"]
    if cloud:
        return cfg.tier_l_budget
    cache = app.setdefault("tier_cache", {})
    if model not in cache:
        params = 0.0
        try:
            async with asyncio.timeout(2):
                async with app["http"].post(
                    f"{cfg.upstream}/api/show", json={"model": model}
                ) as r:
                    d = await r.json()
            params = _parse_param_size((d.get("details") or {}).get("parameter_size", ""))
        except Exception:
            params = 0.0
        cache[model] = params
    params = cache[model]
    if params >= cfg.tier_l_params:
        return cfg.tier_l_budget
    if params >= cfg.tier_m_params:
        return cfg.tier_m_budget
    return None


async def _recall_block(
    app, cue: str, session_id: str, limits: tuple | None = None
) -> str | None:
    cfg: Config = app["cfg"]
    try:
        async with asyncio.timeout(cfg.recall_deadline_s):
            block, _ = await app["recall"].recall(cue, session_id, limits=limits)
            return block
    except Exception:
        return None


async def _forward_and_tee(
    request, upstream_url: str, payload: dict
) -> tuple[web.StreamResponse, int, bytes]:
    """POST payload upstream, stream response bytes verbatim to the client,
    return (prepared_response, status, accumulated_bytes)."""
    http: aiohttp.ClientSession = request.app["http"]
    # Forward client headers (Authorization etc.) minus hop-controlled ones;
    # content-type is re-set by json= since the body is re-serialized.
    # CIMultiDict preserves repeated headers (e.g. multiple Set-Cookie).
    fwd = CIMultiDict(
        (k, v) for k, v in request.headers.items()
        if k.lower() not in HOP_HEADERS and k.lower() != "content-type"
    )
    chunks, shadow_size = [], 0
    async with http.post(upstream_url, json=payload, headers=fwd) as up:
        headers = CIMultiDict(
            (k, v) for k, v in up.headers.items() if k.lower() not in HOP_HEADERS
        )
        resp = web.StreamResponse(status=up.status, headers=headers)
        await resp.prepare(request)
        async for chunk in up.content.iter_any():
            if shadow_size < MAX_SHADOW_BYTES:
                chunks.append(chunk)
                shadow_size += len(chunk)
            await resp.write(chunk)
        await resp.write_eof()
    return resp, up.status, b"".join(chunks)


def _parse_native(raw: bytes) -> str:
    """Accumulate assistant content from NDJSON chunks (or a single JSON object)."""
    out = []
    for line in raw.split(b"\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = obj.get("message") or {}
        c = msg.get("content")
        if isinstance(c, str):
            out.append(c)
    return "".join(out)


def _parse_openai(raw: bytes, streamed: bool) -> str:
    if not streamed:
        try:
            obj = json.loads(raw)
            return (obj.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        except json.JSONDecodeError:
            return ""
    out = []
    for line in raw.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        delta = (obj.get("choices") or [{}])[0].get("delta") or {}
        c = delta.get("content")
        if isinstance(c, str):
            out.append(c)
    return "".join(out)


def _store_episode(app, session_id, user_text, assistant_text, model):
    cfg: Config = app["cfg"]
    store: Store = app["store"]
    user_text = strip_sentinel(user_text).strip()
    assistant_text = _THINK_RE.sub("", assistant_text).strip()
    if len(user_text) < cfg.min_episode_chars or not assistant_text:
        return
    sha = hashlib.sha256(f"{user_text}\x00{assistant_text}".encode()).hexdigest()
    try:
        eid = store.add_episode(
            session_id, user_text, assistant_text, model, sha,
            source=formation.classify_source(user_text),
        )
        if eid is None:
            store.repeat_presentation(sha, cfg.w_repeat)
    except Exception:
        pass  # the client already has its response; losing one episode beats a 500


# ---------------- memory-enabled chat routes ----------------

async def _handle_chat(request: web.Request, native: bool) -> web.StreamResponse:
    app = request.app
    activity: Activity = app["activity"]
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)

    path = "/api/chat" if native else "/v1/chat/completions"
    upstream_url = f"{app['cfg'].upstream}{path}"
    # Shape guards: anything unexpected forwards untouched, memory off.
    if not isinstance(body, dict):
        body, messages = {"__raw__": body}, None
        payload = body["__raw__"]
    else:
        payload = body
        messages = body.get("messages")
        if not isinstance(messages, list):
            messages = None
    user_text = _last_user_text(messages)
    memory_on = user_text is not None and request.headers.get("X-Engram", "") != "off"
    session_id = _session_id(request, body) if memory_on else ""

    # in_flight covers recall too: the summarizer must not grab the GPU while
    # a request is in its embed phase.
    activity.in_flight += 1
    try:
        if memory_on:
            limits = await _tier_limits(app, str(body.get("model", "")), cloud=False)
            block = await _recall_block(app, user_text, session_id, limits)
            if block:
                prompt_tokens = sum(
                    est_tokens(m["content"])
                    for m in messages
                    if isinstance(m, dict) and isinstance(m.get("content"), str)
                )
                payload = {
                    **body,
                    "messages": inject_block(
                        list(messages), block,
                        at_end=prompt_tokens > app["cfg"].big_prompt_tokens,
                    ),
                }
        resp, status, raw = await _forward_and_tee(request, upstream_url, payload)
    finally:
        activity.in_flight -= 1
        activity.last_done = time.time()

    if memory_on and status == 200:
        streamed = body.get("stream", True) if native else bool(body.get("stream"))
        assistant = _parse_native(raw) if native else _parse_openai(raw, streamed)
        _store_episode(app, session_id, user_text, assistant, body.get("model", ""))

    return resp


async def handle_chat_native(request):
    return await _handle_chat(request, native=True)


async def handle_chat_openai(request):
    return await _handle_chat(request, native=False)


# ---------------- Anthropic Messages protocol ----------------

def _est_any(content) -> int:
    return est_tokens(content if isinstance(content, str) else json.dumps(content))


def _anthropic_text(content) -> str:
    """Flatten Anthropic content (str or block list) to its text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _anthropic_last_user_text(messages) -> str | None:
    """Text of the newest user turn. Pure tool-result rounds (agent plumbing)
    return None - no recall, no memory - but a round that mixes tool results
    with genuine user text is user speech and keeps its text."""
    if not isinstance(messages, list) or not messages:
        return None
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        return None
    text = _anthropic_text(last.get("content")).strip()
    return text or None


def _anthropic_strip_sentinels(messages: list) -> list:
    out = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        c = m.get("content")
        if isinstance(c, str) and "<engram:memory" in c:
            m = {**m, "content": strip_sentinel(c).strip()}
        elif isinstance(c, list):
            blocks = []
            for b in c:
                if isinstance(b, dict) and b.get("type") == "text" and "<engram:memory" in b.get("text", ""):
                    b = {**b, "text": strip_sentinel(b["text"]).strip()}
                    if not b["text"]:
                        continue  # a stripped-empty block (our old injection) is invalid upstream
                blocks.append(b)
            m = {**m, "content": blocks if blocks else c}
        out.append(m)
    return out


def _anthropic_inject(body: dict, block: str, at_end: bool) -> dict:
    """Place the memory block in an Anthropic-shaped request. Small prompts:
    top of the system param (primacy). Big prompts: appended to the newest
    user message, preserving the provider's prompt cache (and its cache-read
    discount - 5x cheaper than fresh tokens on MiniMax)."""
    messages = _anthropic_strip_sentinels(list(body.get("messages") or []))
    out = {**body, "messages": messages}
    if at_end:
        for i in range(len(messages) - 1, -1, -1):
            m = messages[i]
            if not (isinstance(m, dict) and m.get("role") == "user"):
                continue
            c = m.get("content")
            if isinstance(c, str):
                messages[i] = {**m, "content": f"{c}\n\n{block}"}
                return out
            if isinstance(c, list):
                messages[i] = {**m, "content": c + [{"type": "text", "text": block}]}
                return out
        # no injectable user message - fall through to system placement
    system = out.get("system")
    if isinstance(system, str) or system is None:
        out["system"] = f"{block}\n\n{system}".strip() if system else block
    elif isinstance(system, list):
        out["system"] = [{"type": "text", "text": block}] + system
    return out


def _parse_anthropic(raw: bytes, streamed: bool) -> str:
    if not streamed:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return ""
        return "".join(
            b.get("text", "") for b in obj.get("content") or []
            if isinstance(b, dict) and b.get("type") == "text"
        )
    out = []
    for line in raw.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        try:
            obj = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        if obj.get("type") == "error":
            return ""  # truncated/errored stream must not become a memory
        if obj.get("type") == "content_block_delta":
            delta = obj.get("delta") or {}
            if delta.get("type") == "text_delta":
                out.append(delta.get("text", ""))
    return "".join(out)


async def handle_chat_anthropic(request: web.Request) -> web.StreamResponse:
    app = request.app
    cfg: Config = app["cfg"]
    activity: Activity = app["activity"]
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)

    base = cfg.anthropic_upstream or cfg.upstream
    upstream_url = f"{base}{request.rel_url}"

    if not isinstance(body, dict):
        payload, messages = body, None
    else:
        payload = body
        messages = body.get("messages") if isinstance(body.get("messages"), list) else None

    user_text = _anthropic_last_user_text(messages) if messages else None
    memory_on = user_text is not None and request.headers.get("X-Engram", "") != "off"
    session_id = _session_id(request, body) if memory_on else ""

    # Cloud chat doesn't occupy the local GPU - tracking it in the activity
    # gate would starve the summarizer forever under a busy bot.
    is_local = base == cfg.upstream
    if is_local:
        activity.in_flight += 1
    try:
        if memory_on:
            limits = await _tier_limits(
                app, str(body.get("model", "")), cloud=not is_local
            )
            block = await _recall_block(app, user_text, session_id, limits)
            if block:
                # Size by SERIALIZED content: tool_result/tool_use/thinking
                # blocks and the tools array are usually the bulk of an agent
                # prompt, and text-only counting misrouted them to the
                # cache-destroying system-top placement. json.dumps overcounts
                # slightly, which errs toward at_end - the safe direction.
                prompt_tokens = _est_any(body.get("system") or "") + sum(
                    _est_any(m.get("content") or "")
                    for m in messages
                    if isinstance(m, dict)
                )
                if body.get("tools"):
                    prompt_tokens += _est_any(body["tools"])
                payload = _anthropic_inject(
                    body, block, at_end=prompt_tokens > cfg.big_prompt_tokens
                )
        resp, status, raw = await _forward_and_tee(request, upstream_url, payload)
    finally:
        if is_local:
            activity.in_flight -= 1
            activity.last_done = time.time()

    if memory_on and status == 200:
        assistant = _parse_anthropic(raw, streamed=bool(body.get("stream")))
        _store_episode(app, session_id, user_text, assistant, body.get("model", ""))

    return resp


# ---------------- passthrough + local endpoints ----------------

async def handle_passthrough(request: web.Request) -> web.StreamResponse:
    return await _passthrough_to(request, request.app["cfg"].upstream)


async def handle_anthropic_passthrough(request: web.Request) -> web.StreamResponse:
    cfg: Config = request.app["cfg"]
    return await _passthrough_to(request, cfg.anthropic_upstream or cfg.upstream)


async def _passthrough_to(request: web.Request, base: str) -> web.StreamResponse:
    app = request.app
    activity: Activity = app["activity"]
    http: aiohttp.ClientSession = app["http"]
    url = f"{base}{request.rel_url}"
    headers = CIMultiDict(
        (k, v) for k, v in request.headers.items() if k.lower() not in HOP_HEADERS
    )
    data = await request.read() if request.can_read_body else None

    activity.in_flight += 1
    try:
        async with http.request(
            request.method, url, headers=headers, data=data
        ) as up:
            out_headers = CIMultiDict(
                (k, v) for k, v in up.headers.items() if k.lower() not in HOP_HEADERS
            )
            resp = web.StreamResponse(status=up.status, headers=out_headers)
            await resp.prepare(request)
            async for chunk in up.content.iter_any():
                await resp.write(chunk)
            await resp.write_eof()
            return resp
    finally:
        activity.in_flight -= 1
        activity.last_done = time.time()


async def handle_memories(request):
    """Trace snapshot for the UI: activation computed now, all statuses."""
    store: Store = request.app["store"]
    cfg: Config = request.app["cfg"]
    now = time.time()
    rows = store.db.execute(
        "SELECT id, title, gist, status, quality, created_ts, n_access, last8,"
        " beta, superseded_by, source, pinned FROM traces ORDER BY id DESC LIMIT 500"
    ).fetchall()
    traces = []
    for r in rows:
        last8 = json.loads(r["last8"])
        act = base_level(last8, r["n_access"], r["created_ts"], now, cfg.decay_d)
        traces.append({
            "id": r["id"], "title": r["title"], "gist": r["gist"],
            "status": r["status"], "quality": r["quality"],
            "source": r["source"] or "user",
            "pinned": bool(r["pinned"]),
            "created_ts": r["created_ts"], "n_access": round(r["n_access"], 2),
            "beta": round(r["beta"] or 0.0, 2),
            "last_ts": last8[-1][0] if last8 else r["created_ts"],
            "activation": round(act + (r["beta"] or 0.0), 3),
        })
    # graph edges among the returned traces, for the observatory
    ids = [t["id"] for t in traces]
    links = []
    if ids:
        q = ",".join("?" * len(ids))
        links = [
            [r["a"], r["b"], round(r["weight"], 3)]
            for r in store.db.execute(
                f"SELECT a, b, weight FROM links WHERE a IN ({q}) AND b IN ({q})"
                " AND weight >= 0.12 ORDER BY weight DESC LIMIT 400",
                ids + ids,
            ).fetchall()
        ]
    last = store.get_meta("last_recall") or {}
    return web.json_response({
        "now": now, "traces": traces, "links": links,
        "last_cue": last.get("cue", ""), "last_ms": last.get("ms"),
    })


async def handle_ui(request):
    return web.FileResponse(Path(__file__).parent / "ui.html")


async def handle_stats(request):
    stats = request.app["store"].stats()
    stats["in_flight"] = request.app["activity"].in_flight
    return web.json_response(stats)


async def handle_why(request):
    return web.json_response(request.app["store"].get_meta("last_recall") or {})


def run(cfg: Config):
    web.run_app(make_app(cfg), host="127.0.0.1", port=cfg.port, print=None)
