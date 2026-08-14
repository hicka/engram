"""CLI: engram up | recall | list | why | stats | bench"""

import argparse
import asyncio
import datetime
import json
import statistics
import time

from .config import Config


def _fmt_ts(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def cmd_up(cfg: Config, args):
    from .proxy import run

    print(f"engram v0.1 · proxy http://127.0.0.1:{cfg.port} -> {cfg.upstream}")
    print(f"store {cfg.db_path} · summarizer {cfg.summarizer_model} · embed {cfg.embed_model}")
    print(f'try:  export OLLAMA_HOST=127.0.0.1:{cfg.port}')
    run(cfg)


def cmd_list(cfg: Config, args):
    from .store import Store

    store = Store(cfg.db_path, cfg.embed_dim)
    rows = store.db.execute(
        "SELECT * FROM traces WHERE status='active' ORDER BY id DESC LIMIT ?",
        (args.n,),
    ).fetchall()
    if not rows:
        print("no traces yet")
        return
    for r in rows:
        print(f"[{r['id']:>4}] {_fmt_ts(r['created_ts'])}  n={r['n_access']:.2f}  ({r['quality']})")
        print(f"       {r['title']}")
        print(f"       {r['gist'][:160]}")


def cmd_recall(cfg: Config, args):
    import aiohttp

    from .recall import Recall
    from .store import Store

    async def go():
        store = Store(cfg.db_path, cfg.embed_dim)
        async with aiohttp.ClientSession() as http:
            r = Recall(store, cfg, http)
            r.cfg.embed_timeout_s = 5.0  # CLI has no latency budget
            block, debug = await r.recall(args.query, session_id="__cli__", dry_run=True)
        print(f"cue: {args.query!r}\n")
        if not debug:
            print("no candidates")
        for c in debug:
            mark = "✓" if c["admitted"] else "·"
            print(
                f" {mark} [{c['id']:>4}] score={c['score']:.3f} cos={c['cos']:.2f}"
                f" bm25={c['bm25']:.1f} act={c['activation']:.2f}  {c['title']}"
            )
        print("\n--- injected block ---")
        print(block or "(nothing admitted - inject nothing)")

    asyncio.run(go())


def cmd_why(cfg: Config, args):
    from .store import Store

    store = Store(cfg.db_path, cfg.embed_dim)
    last = store.get_meta("last_recall")
    if not last:
        print("no recall recorded yet")
        return
    print(f"at {_fmt_ts(last['ts'])} · {last['ms']}ms · semantic={last['semantic']}")
    print(f"cue: {last['cue'][:120]!r}")
    for c in last["candidates"]:
        mark = "✓" if c["admitted"] else "·"
        print(
            f" {mark} [{c['id']:>4}] score={c['score']:.3f} cos={c['cos']:.2f}"
            f" bm25={c['bm25']:.1f} act={c['activation']:.2f}  {c['title']}"
        )
    print("\n--- block ---")
    print(last["block"] or "(nothing injected)")


def cmd_stats(cfg: Config, args):
    from .store import Store

    print(json.dumps(Store(cfg.db_path, cfg.embed_dim).stats(), indent=2))


def cmd_bench(cfg: Config, args):
    import aiohttp

    from .recall import Recall
    from .store import Store

    async def go():
        store = Store(cfg.db_path, cfg.embed_dim)
        async with aiohttp.ClientSession() as http:
            r = Recall(store, cfg, http)
            r.cfg.embed_timeout_s = 5.0
            await r.embed("warmup")  # eat the cold start
            embed_ms, recall_ms = [], []
            for i in range(args.n):
                t0 = time.perf_counter()
                await r.embed(f"benchmark query {i} peanut allergy thai food deploy")
                embed_ms.append((time.perf_counter() - t0) * 1000)
                t0 = time.perf_counter()
                await r.recall(
                    f"benchmark query {i} peanut allergy thai food", "__bench__",
                    dry_run=True,
                )
                recall_ms.append((time.perf_counter() - t0) * 1000)

        def pct(xs, p):
            return statistics.quantiles(xs, n=100)[p - 1] if len(xs) >= 10 else max(xs)

        print(f"traces in store: {store.stats()['traces']}")
        print(f"embed  p50 {statistics.median(embed_ms):6.1f}ms  p95 {pct(embed_ms, 95):6.1f}ms")
        print(f"recall p50 {statistics.median(recall_ms):6.1f}ms  p95 {pct(recall_ms, 95):6.1f}ms")

    asyncio.run(go())


def cmd_profile(cfg: Config, args):
    from .store import Store

    row = Store(cfg.db_path, cfg.embed_dim).get_profile()
    if row is None:
        print("no profile yet - forms automatically during idle consolidation")
        return
    print(f"updated {_fmt_ts(row['updated_ts'])}, from {len(json.loads(row['source_ids'] or '[]'))} memories:\n")
    print(row["gist"])


def cmd_lapse(cfg: Config, args):
    """Advance the decay clock: shift every memory timestamp into the past.
    Makes forgetting testable in minutes instead of weeks."""
    import re as _re

    from .store import Store

    m = _re.fullmatch(r"(\d+(?:\.\d+)?)([mhd])", args.duration.strip())
    if not m:
        print("duration like 30m, 72h, or 45d")
        return
    secs = float(m.group(1)) * {"m": 60, "h": 3600, "d": 86400}[m.group(2)]
    store = Store(cfg.db_path, cfg.embed_dim)
    n = 0
    for r in store.db.execute("SELECT id, last8, created_ts FROM traces").fetchall():
        last8 = [[ts - secs, w] for ts, w in json.loads(r["last8"])]
        store.db.execute(
            "UPDATE traces SET created_ts=?, last8=? WHERE id=?",
            (r["created_ts"] - secs, json.dumps(last8), r["id"]),
        )
        n += 1
    store.db.execute("UPDATE episodes SET ts = ts - ?", (secs,))
    store.db.execute("UPDATE schemas SET created_ts=created_ts-?, updated_ts=updated_ts-?", (secs, secs))
    store.set_meta("decay_last_sweep", 0)  # let the next sweep run immediately
    store.db.commit()
    print(f"lapsed {args.duration}: {n} traces aged; next idle sweep may silence the faded")


def cmd_import(cfg: Config, args):
    """Bootstrap memories from a markdown notes file (e.g. an agent's
    MEMORY.md). Entries are backdated with a synthetic spaced presentation so
    they neither flood retrieval nor arrive dead on the decay line."""
    import time as _time
    from pathlib import Path as _P

    from .mcp import MemoryTools

    text = _P(args.file).expanduser().read_text(errors="ignore")
    entries = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-•*").strip()
        line = __import__("re").sub(r"^#?\d+[.)]?\s*", "", line)
        if len(line) >= 15 and not line.startswith("#"):
            entries.append(line)
    if not entries:
        print("no importable entries found")
        return
    tools = MemoryTools(cfg)
    now = _time.time()
    imported = dup = 0
    for i, fact in enumerate(entries):
        tid, msg = tools._make_trace(fact)
        if "Already known" in msg:
            dup += 1
            continue
        created = now - 30 * 86400 - i * 3600
        recent = now - 3 * 86400 - i * 600
        tools.store.db.execute(
            "UPDATE traces SET created_ts=?, last8=?, quality='imported' WHERE id=?",
            (created, json.dumps([[created, 1.0], [recent, 0.5]]), tid),
        )
        imported += 1
    tools.store.db.commit()
    print(f"imported {imported} entries ({dup} already known) from {args.file}")


def cmd_mcp(cfg: Config, args):
    from .mcp import main as mcp_main

    mcp_main()


def cmd_eval(cfg: Config, args):
    import sys

    from .eval import DEFAULT_MODEL, run_eval

    sys.exit(asyncio.run(run_eval(cfg, model=args.model or DEFAULT_MODEL)))


def main():
    p = argparse.ArgumentParser(prog="engram", description="local memory sidecar for Ollama")
    p.add_argument("--port", type=int, help="proxy port (default 11435)")
    p.add_argument("--upstream", help="Ollama URL (default http://127.0.0.1:11434)")
    p.add_argument("--db", help="database path (default ~/.engram/engram.db)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("up", help="run the proxy")
    sp = sub.add_parser("list", help="show recent traces")
    sp.add_argument("-n", type=int, default=10)
    sp = sub.add_parser("recall", help="dry-run recall for a query")
    sp.add_argument("query")
    sub.add_parser("why", help="explain the last injection")
    sub.add_parser("stats", help="store stats")
    sp = sub.add_parser("bench", help="measure recall latency")
    sp.add_argument("-n", type=int, default=20)
    sp = sub.add_parser("eval", help="run the scripted memory regression scenario")
    sp.add_argument("--model", default=None, help="answering model to wrap (default qwen3:0.6b)")
    sub.add_parser("mcp", help="run the memory-verbs MCP server on stdio")
    sub.add_parser("profile", help="show the consolidated user profile")
    sp = sub.add_parser("lapse", help="advance the decay clock (testing/demos)")
    sp.add_argument("duration", help="e.g. 30m, 72h, 45d")
    sp = sub.add_parser("import", help="import memories from a markdown notes file")
    sp.add_argument("file")

    args = p.parse_args()
    cfg = Config()
    if args.port:
        cfg.port = args.port
    if args.upstream:
        cfg.upstream = args.upstream
    if args.db:
        from pathlib import Path

        cfg.db_path = Path(args.db)

    {
        "up": cmd_up, "list": cmd_list, "recall": cmd_recall,
        "why": cmd_why, "stats": cmd_stats, "bench": cmd_bench,
        "eval": cmd_eval, "mcp": cmd_mcp, "profile": cmd_profile,
        "lapse": cmd_lapse, "import": cmd_import,
    }[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
