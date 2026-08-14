"""engram eval: the field-test scenario as a regression suite.

Boots an in-process proxy on a throwaway store, replays a scripted
multi-session conversation, then probes memory in fresh sessions. Each probe
is scored twice: RETRIEVAL (did the right facts reach the injected block -
deterministic-ish) and ANSWER (did the small model actually use them -
stochastic, informational). Also reports injection precision, junk-trace
count, and recall latency.
"""

import asyncio
import json
import statistics
import tempfile
import time
from pathlib import Path

import aiohttp
from aiohttp import web

from .config import Config
from .proxy import make_app

PORT = 11499
BASE = f"http://127.0.0.1:{PORT}"
DEFAULT_MODEL = "qwen3:0.6b"  # the model being wrapped; override with --model

SEED_SESSIONS = [
    ("personal", [
        "Hi! I'm Sam. I work at Meridian Labs as a backend engineer.",
        "My cat Miso knocked my coffee onto my keyboard this morning, chaos.",
        "Important: I'm badly allergic to peanuts. Even trace amounts.",
    ]),
    ("work", [
        "Our staging deploy keeps failing with ECONNRESET until I set PGSSLMODE=require. Annoying but that's the fix.",
        "Team decision from today: the new invoicing service uses Postgres 16, not MySQL. Final.",
    ]),
    ("editor", ["By the way, my favorite editor is vim, been using it for years."]),
    ("correction", ["Update: I actually switched from vim to Neovim last month, so Neovim is my editor now."]),
    ("long", [
        "Hey, I'm planning a trip to Japan in October with my sister Ana.",
        "We'll do 5 days in Tokyo and 3 days in Kyoto. Total budget is $3000.",
        "Also work stuff: the API gateway kept timing out at 30 seconds, so I bumped the timeout to 120 seconds yesterday.",
        "Ana is vegetarian, so restaurant picks need to account for that.",
        "Back to code: we renamed the auth service to 'gatekeeper' last sprint, everyone keeps forgetting.",
        "My boss Elena wants the database migration finished by November 15th, hard deadline.",
        "Given our budget, what should we prioritize in Kyoto?",
        "Remind me, what's my sister's dietary restriction?",
        "Good. We'll also add a day trip to Nara to see the deer park.",
        "thanks, this was helpful",
    ]),
]

# block: needles that must appear in the injected memory block (retrieval).
# reply: needles for the model's answer (informational - 0.6b is stochastic).
# any_of: any single needle passes instead of all.
PROBES = [
    dict(name="P1-semantic-hop", q="I'm ordering Thai food for dinner tonight, anything I should avoid?",
         block=["peanut"], reply=["peanut"]),
    dict(name="P2-literal-token", q="The deploy is failing with ECONNRESET again. What was the fix?",
         block=["pgsslmode"], reply=["pgsslmode"]),
    dict(name="P3-deadline", q="When is my database migration due, and who set the deadline?",
         block=[["november 15", "nov 15"], "elena"], reply=[["november 15", "nov 15"]]),
    dict(name="P4-rename", q="What did we rename the auth service to?",
         block=["gatekeeper"], reply=["gatekeeper"]),
    dict(name="P5-trip-constraints", q="I'm booking restaurants for the Japan trip. Any constraints I should remember?",
         block=["vegetarian"], reply=["vegetarian"]),
    dict(name="P6-number-recall", q="What's the API gateway timeout set to now?",
         block=["120"], reply=["120"]),
    dict(name="P7-contradiction", q="Which editor do I use these days?",
         block=["neovim"], reply=["neovim"], block_must_not=["vim is a great productivity"]),
    dict(name="P8-unrelated", q="Explain briefly how rainbows form.", expect_no_injection=True),
    dict(name="P9-smalltalk", q="thanks!", expect_recall_skipped=True),
    dict(name="P10-aggregate", q="What do you know about me? List everything.",
         block=["sam", "peanut", "meridian", "miso"], any_of=True, known_gap="needs v0.2 consolidation"),
]


class _Session:
    def __init__(self, model=DEFAULT_MODEL):
        self.msgs = []
        self.model = model

    async def say(self, http, text):
        self.msgs.append({"role": "user", "content": text})
        out = []
        async with http.post(
            f"{BASE}/api/chat",
            json={"model": self.model, "think": False, "stream": True,
                  "options": {"num_predict": 220}, "messages": self.msgs},
        ) as r:
            async for line in r.content:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line).get("message", {}).get("content", ""))
                    except json.JSONDecodeError:
                        pass
        reply = "".join(out)
        self.msgs.append({"role": "assistant", "content": reply})
        async with http.get(f"{BASE}/engram/why") as r:
            why = await r.json()
        recall_ran = why.get("cue", "") == text.strip()[:512]
        return reply, (why if recall_ran else None)


async def _drain(http, max_s=240):
    t0 = time.time()
    while time.time() - t0 < max_s:
        async with http.get(f"{BASE}/engram/stats") as r:
            s = await r.json()
        if s["queue_depth"] == 0 and s["in_flight"] == 0:
            return s
        await asyncio.sleep(2)
    return s


async def run_eval(cfg: Config, model: str = DEFAULT_MODEL):
    tmp = tempfile.mkdtemp(prefix="engram-eval-")
    cfg.db_path = Path(tmp) / "eval.db"
    cfg.port = PORT

    runner = web.AppRunner(make_app(cfg))
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", PORT)
    await site.start()
    print(f"eval store {cfg.db_path}")
    print(f"answering model {model} · summarizer {cfg.summarizer_model}"
          f" · embedder {cfg.embed_model}\n")

    results, recall_ms = [], []
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_read=300)
        ) as http:
            for name, turns in SEED_SESSIONS:
                s = _Session(model)
                for t in turns:
                    _, why = await s.say(http, t)
                    if why:
                        recall_ms.append(why["ms"])
                await _drain(http)
                print(f"  seeded [{name}] ({len(turns)} turns)")

            for p in PROBES:
                s = _Session(model)
                reply, why = await s.say(http, p["q"])
                if why:
                    recall_ms.append(why["ms"])
                block = (why or {}).get("block") or ""
                bl, rl = block.lower(), reply.lower()

                if p.get("expect_no_injection"):
                    retrieval = why is not None and not block
                    answer = retrieval
                    note = "no injection" if retrieval else "INJECTED on unrelated cue"
                elif p.get("expect_recall_skipped"):
                    retrieval = why is None
                    answer = retrieval
                    note = "recall skipped" if retrieval else "recall RAN on smalltalk"
                else:
                    # a needle may be a list of alternates (any one counts)
                    def hit(n, hay):
                        alts = n if isinstance(n, list) else [n]
                        return any(a.lower() in hay for a in alts)
                    needles = p["block"]
                    hits = [n for n in needles if hit(n, bl)]
                    retrieval = bool(hits) if p.get("any_of") else len(hits) == len(needles)
                    if p.get("block_must_not") and any(n.lower() in bl for n in p["block_must_not"]):
                        retrieval = False
                    answer = all(hit(n, rl) for n in p.get("reply", []))
                    note = f"block hit {len(hits)}/{len(needles)}"
                    if p.get("known_gap"):
                        note += f" · known gap: {p['known_gap']}"
                results.append(dict(p=p, retrieval=retrieval, answer=answer, note=note,
                                    block=block, reply=reply[:200]))
                await asyncio.sleep(0.5)

            final = await _drain(http)
            # junk metric: traces formed from probe questions / smalltalk
            async with http.get(f"{BASE}/engram/stats") as r:
                stats = await r.json()
    finally:
        await runner.cleanup()

    print(f"\n{'probe':<22} {'retrieval':<10} {'answer':<8} note")
    r_pass = a_pass = scored = 0
    for r in results:
        p = r["p"]
        hard = not p.get("known_gap")
        if hard:
            scored += 1
            r_pass += r["retrieval"]
            a_pass += r["answer"]
        rv = "PASS" if r["retrieval"] else ("gap" if not hard else "FAIL")
        av = "PASS" if r["answer"] else ("gap" if not hard else "fail")
        print(f"{p['name']:<22} {rv:<10} {av:<8} {r['note']}")

    print(f"\nretrieval {r_pass}/{scored} · answer {a_pass}/{scored} (excl. known gaps)")
    print(f"traces formed: {stats['traces']} · episodes {stats['episodes']}"
          f" · degraded recalls {stats['degraded_recalls']}")
    if recall_ms:
        print(f"recall p50 {statistics.median(recall_ms):.0f}ms · max {max(recall_ms):.0f}ms")
    for r in results:
        if not r["retrieval"] and not r["p"].get("known_gap"):
            print(f"\n--- {r['p']['name']} failed retrieval ---")
            print(f"block: {r['block'] or '(none)'}")
    return 0 if r_pass == scored else 1
