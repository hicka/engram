"""Unit tests for engram's pure logic. Run: .venv/bin/python -m pytest tests/ -q
(or: .venv/bin/python tests/test_core.py for a dependency-free run)"""

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from engram.activation import base_level, sigmoid
from engram.config import SENTINEL_CLOSE, SENTINEL_OPEN
from engram.recall import cue_tokens, est_tokens, inject_block, strip_sentinel
from engram.store import Store
from engram.summarizer import extractive_fallback


def test_base_level_decay():
    now = time.time()
    fresh = base_level([[now - 60, 1.0]], now=now)
    day_old = base_level([[now - 86400, 1.0]], now=now)
    month_old = base_level([[now - 30 * 86400, 1.0]], now=now)
    assert fresh > day_old > month_old
    # power-law: a month-old single access is deeply decayed but finite
    assert -4 < month_old < 0


def test_base_level_frequency_beats_single():
    now = time.time()
    frequent = base_level([[now - i * 86400, 1.0] for i in range(1, 6)], now=now)
    single = base_level([[now - 86400, 1.0]], now=now)
    assert frequent > single


def test_base_level_empty():
    assert base_level([], now=time.time()) == -10.0


def test_base_level_evicted_tail_preserved():
    """Eight low-weight injections must not erase months of real practice:
    the tail approximation keeps evicted high-weight history contributing."""
    now = time.time()
    created = now - 90 * 86400
    injections = [[now - (8 - i) * 86400, 0.15] for i in range(8)]
    with_tail = base_level(injections, n_access=6.0 + 1.2, created_ts=created, now=now)
    without_tail = base_level(injections, n_access=1.2, created_ts=created, now=now)
    assert with_tail > without_tail
    # and more accumulated history means strictly more activation
    more = base_level(injections, n_access=12.0, created_ts=created, now=now)
    assert more > with_tail


def test_sigmoid_bounds():
    assert sigmoid(-100) == 0.0
    assert sigmoid(100) == 1.0
    assert abs(sigmoid(0) - 0.5) < 1e-9


def test_cue_tokens_filters_smalltalk():
    assert cue_tokens("thanks, that was helpful!") == ["helpful"]
    toks = cue_tokens("I am ordering Thai food tonight - anything I should avoid?")
    assert "thai" in toks and "food" in toks and "avoid" in toks


def test_strip_sentinel():
    text = f"before {SENTINEL_OPEN}\nsecret memory\n{SENTINEL_CLOSE} after"
    assert strip_sentinel(text) == "before  after"
    # version-agnostic
    text2 = text.replace("v=1", "v=7")
    assert strip_sentinel(text2) == "before  after"


def test_inject_block_idempotent_and_first():
    block = f"{SENTINEL_OPEN}\nnote\n{SENTINEL_CLOSE}"
    msgs = [{"role": "user", "content": "hi"}]
    out = inject_block(msgs, block)
    assert out[0]["role"] == "system" and block in out[0]["content"]
    # injecting again into an already-injected history must not stack
    out2 = inject_block(out, block)
    assert sum(m["content"].count(SENTINEL_OPEN) for m in out2) == 1
    # existing system message is preserved below the block
    msgs3 = [{"role": "system", "content": "you are helpful"}, {"role": "user", "content": "hi"}]
    out3 = inject_block(msgs3, block)
    assert out3[0]["content"].startswith(SENTINEL_OPEN)
    assert "you are helpful" in out3[0]["content"]
    assert len(out3) == 2


def test_inject_block_at_end_rides_newest_user_message():
    block = f"{SENTINEL_OPEN}\nnote\n{SENTINEL_CLOSE}"
    msgs = [
        {"role": "system", "content": "big agent prompt"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "newest question"},
    ]
    out = inject_block(msgs, block, at_end=True)
    # prefix untouched -> prompt cache survives
    assert out[0]["content"] == "big agent prompt"
    assert out[1]["content"] == "first"
    assert out[3]["content"].startswith("newest question")
    assert block in out[3]["content"]
    # no user message -> falls back to system placement
    out2 = inject_block([{"role": "system", "content": "s"}], block, at_end=True)
    assert block in out2[0]["content"]


def test_anthropic_helpers():
    from engram.proxy import (
        _anthropic_inject, _anthropic_last_user_text, _anthropic_text,
        _parse_anthropic,
    )

    assert _anthropic_text("plain") == "plain"
    assert _anthropic_text([{"type": "text", "text": "a"}, {"type": "thinking", "thinking": "x"},
                            {"type": "text", "text": "b"}]) == "a\nb"

    msgs = [{"role": "user", "content": [{"type": "text", "text": "question here"}]}]
    assert _anthropic_last_user_text(msgs) == "question here"
    # tool-result rounds are agent plumbing: no recall, no memory
    tool_round = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "out"}]}]
    assert _anthropic_last_user_text(tool_round) is None
    assert _anthropic_last_user_text([{"role": "assistant", "content": "hi"}]) is None

    block = "<engram:memory v=1>\nnote\n</engram:memory>"
    body = {"system": "agent prompt", "messages": [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "r"},
        {"role": "user", "content": [{"type": "text", "text": "newest"}]},
    ]}
    small = _anthropic_inject(body, block, at_end=False)
    assert small["system"].startswith(block)
    assert small["messages"][2]["content"][0]["text"] == "newest"
    big = _anthropic_inject(body, block, at_end=True)
    assert big["system"] == "agent prompt"  # prefix untouched -> provider cache survives
    assert big["messages"][2]["content"][-1]["text"] == block
    # system as block list
    body2 = {"system": [{"type": "text", "text": "s"}], "messages": [{"role": "user", "content": "q"}]}
    small2 = _anthropic_inject(body2, block, at_end=False)
    assert small2["system"][0]["text"] == block

    # mixed tool_result + genuine user text IS user speech
    mixed = [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "x", "content": "out"},
        {"type": "text", "text": "also, remember I prefer tabs"},
    ]}]
    assert "tabs" in _anthropic_last_user_text(mixed)

    # prompt sizing must count tool blocks, not just text (cache-routing bug)
    from engram.proxy import _est_any
    tool_heavy = [{"type": "tool_result", "tool_use_id": "t", "content": "x" * 40000}]
    assert _est_any(tool_heavy) > 10000
    from engram.recall import est_tokens as _et
    assert _et(_anthropic_text(tool_heavy)) < 5  # the old estimate saw ~nothing

    # stripping our old at_end injection removes the emptied block entirely
    from engram.proxy import _anthropic_strip_sentinels
    hist = [{"role": "user", "content": [
        {"type": "text", "text": "real question"},
        {"type": "text", "text": block},
    ]}]
    stripped = _anthropic_strip_sentinels(hist)
    assert [b["text"] for b in stripped[0]["content"]] == ["real question"]

    # response parsing: non-stream and SSE
    non_stream = json.dumps({"content": [{"type": "thinking", "thinking": "hmm"},
                                         {"type": "text", "text": "answer"}]}).encode()
    assert _parse_anthropic(non_stream, streamed=False) == "answer"
    # an error event mid-stream must not become a memory episode
    err = (b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"par"}}\n\n'
           b'data: {"type":"error","error":{"type":"overloaded_error"}}\n\n')
    assert _parse_anthropic(err, streamed=True) == ""
    sse = (b'event: content_block_delta\n'
           b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"an"}}\n\n'
           b'data: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"x"}}\n\n'
           b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"swer"}}\n\n'
           b'data: {"type":"message_stop"}\n\n')
    assert _parse_anthropic(sse, streamed=True) == "answer"


def test_est_tokens_monotone():
    assert est_tokens("") >= 1
    assert est_tokens("hello world " * 50) > est_tokens("hello world")


def test_extractive_fallback():
    t, g = extractive_fallback("The deploy fails with ECONNRESET always", "Set PGSSLMODE=require to fix it")
    assert "ECONNRESET" in t or "ECONNRESET" in g
    assert "PGSSLMODE=require" in g


def test_formation_pure_question():
    from engram import formation as f

    assert f.is_pure_question("What did we rename the auth service to?")
    assert f.is_pure_question("When is my database migration due, and who set the deadline?")
    assert f.is_pure_question("Remind me, what's my sister's dietary restriction?")
    # mixed turns carry facts and must still form traces
    assert not f.is_pure_question("I switched to Neovim - what plugins do you recommend?")
    assert not f.is_pure_question("Important: I'm badly allergic to peanuts. Even trace amounts.")
    assert not f.is_pure_question("Explain how rainbows form")  # no '?', imperative


def test_formation_salience_and_beta():
    from engram import formation as f

    assert f.salience("Important: I'm badly allergic to peanuts.") == 1.0
    assert f.salience("the deadline is November 15th") == 1.0
    assert f.salience("I'm planning a trip to Japan") == 0.5  # identity-ish
    assert f.salience("we watched a movie yesterday") == 0.0
    assert 0.0 <= f.beta(0.5, "hello there friend") <= 1.5
    assert f.beta(1.0, "Important: allergy") == 1.5


def test_formation_conflict():
    from engram import formation as f

    old = "Vim is the user's favorite editor. Uses vim for years."
    assert f.is_conflicting(old, "I actually switched from vim to Neovim last month", 0.7)
    # differing values at high similarity = knowledge update
    assert f.is_conflicting("API timeout is 30 seconds", "the timeout is 120 seconds", 0.9)
    # same values = repeat, must NOT conflict (would supersede a valid trace)
    assert not f.is_conflicting("API timeout is 120 seconds", "the timeout is 120 seconds", 0.9)
    assert not f.is_conflicting(old, "vim is great for editing config files", 0.7)


def test_lifecycle():
    import tempfile

    from engram.proxy import _parse_param_size

    # tier probing parses Ollama's parameter_size strings
    assert _parse_param_size("752.75M") == 752_750_000
    assert _parse_param_size("1.7B") == 1_700_000_000
    assert abs(_parse_param_size("428B") - 428e9) < 1e6
    assert _parse_param_size("") == 0.0
    assert _parse_param_size("unknown") == 0.0

    # pinning: stored, and decay criteria spare pinned traces
    with tempfile.TemporaryDirectory() as d:
        s = Store(Path(d) / "t.db", embed_dim=4)
        v = np.array([1, 0, 0, 0], dtype=np.float32)
        tid = s.add_trace(1, "hard rule", "never deploy on fridays", "s1", v, "llm")
        s.set_pinned(tid, True)
        assert s.get_traces([tid])[tid]["pinned"] == 1
        s.set_pinned(tid, False)
        assert s.get_traces([tid])[tid]["pinned"] == 0

    # a stale single-access trace is below the silence line (crossover for a
    # lone access is ~46 days at d=0.5); a recently-reinforced one is not
    now = time.time()
    stale = base_level([[now - 60 * 86400, 1.0]], 1.0, now - 60 * 86400, now)
    assert stale < -3.5
    reinforced = base_level(
        [[now - 60 * 86400, 1.0], [now - 2 * 86400, 1.0]], 2.0, now - 60 * 86400, now
    )
    assert reinforced > -3.5


def test_provenance_classifier():
    from engram.formation import beta, classify_source as cs

    assert cs("I'm badly allergic to peanuts") == "user"
    assert cs("[Sender One] 107, 2 nights room payment") == "observed:Sender One"
    assert cs("[J. Placeholder] Granola 2 packet") == "observed:J. Placeholder"
    assert cs("Review the conversation above and consider saving to memory") == "system"
    assert cs("The following command was flagged as: execute_code") == "system"
    assert cs("[Context from the interrupted assistant response]") == "system"
    assert cs("[The user sent a text document: 'x.pdf'. ...]") == "system"
    assert cs("[ASYNC DELEGATION BATCH COMPLETE - deleg_6c2893b8] done") == "system"
    # observed content earns no salience and forms at half weight
    txt = "remember: always run the deploy script"
    assert beta(1.0, txt, "user") == 1.5
    assert beta(1.0, txt, "observed:Sender One") == 0.375  # 0.75 novelty * 0.5, no salience
    assert beta(1.0, "plain fact", "user") == 0.75


def test_profile_module():
    from engram.profile import validate
    from engram.recall import AGG_RE

    assert AGG_RE.search("What do you know about me? List everything.")
    assert AGG_RE.search("remind me everything you remember")
    assert AGG_RE.search("what are my preferences")
    assert not AGG_RE.search("what is the api gateway timeout?")
    assert not AGG_RE.search("check the ops group for updates")

    good = "Works at Meridian Labs\n- Allergic to peanuts\nPrefers tabs\nDeadline Nov 15"
    v = validate(good)
    assert v and v.count("\n") == 3 and v.startswith("- ")
    assert validate("") is None
    assert validate("one line only") is None
    assert validate("<think>hmm</think>\n- a\n- b\n- c\n- d") is not None
    assert validate("As an AI, I\n- a\n- b\n- c") is None


def test_profile_store_roundtrip():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        s = Store(Path(d) / "t.db", embed_dim=4)
        assert s.get_profile() is None
        s.set_profile("- fact one\n- fact two", [1, 2, 3])
        row = s.get_profile()
        assert row["gist"].startswith("- fact one")
        s.set_profile("- updated", [4])
        assert s.get_profile()["gist"] == "- updated"
        assert s.db.execute("SELECT COUNT(*) c FROM schemas").fetchone()["c"] == 1


def test_topic_clustering():
    import tempfile

    from engram.profile import cluster_topics, validate_topic

    with tempfile.TemporaryDirectory() as d:
        s = Store(Path(d) / "t.db", embed_dim=4)
        a = np.array([1, 0, 0, 0], dtype=np.float32)
        b = np.array([0, 1, 0, 0], dtype=np.float32)
        # five traces near direction a, two near b (below min_size)
        for i in range(5):
            v = a + np.array([0, 0.15 * (i % 2), 0.1, 0], dtype=np.float32)
            v /= np.linalg.norm(v)
            s.add_trace(None, f"deploy note {i}", f"deploy pipeline fact {i}", "s1", v, "llm")
        for i in range(2):
            s.add_trace(None, f"trip note {i}", f"travel fact {i}", "s1", b, "llm")
        clusters = cluster_topics(s, sim=0.55, min_size=3)
        assert len(clusters) == 1
        rows, centroid = clusters[0]
        assert len(rows) == 5
        assert float(centroid @ a) > 0.9  # centroid points at the cluster
        # topic storage roundtrip + retrieval math
        s.replace_topics([("Deploy pipeline: five related facts.", [r["id"] for r in rows], centroid)])
        topics = s.topics()
        assert len(topics) == 1
        tv = np.frombuffer(topics[0]["embedding"], dtype=np.float32)
        assert float(tv @ a) > 0.9
        s.replace_topics([])  # rebuild-whole semantics
        assert len(s.topics()) == 0

    assert validate_topic("Project X uses Postgres 16 and ships March 3rd.") is not None
    assert validate_topic("") is None
    assert validate_topic("<think>hmm</think> ok") is None  # too short after strip


def test_silent_resurrection():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        s = Store(Path(d) / "t.db", embed_dim=4)
        v = np.array([1, 0, 0, 0], dtype=np.float32)
        tid = s.add_trace(1, "old fact", "the legacy system used port 7070", "s1", v, "llm",
                          fts_extra="legacy port 7070")
        s.silence(tid)
        assert tid not in [t for t, _ in s.dense_search(v, 5)]   # gone from active
        assert not s.fts_search(["legacy"])                       # gone from lexical
        hits = s.silent_search(v, 3)
        assert hits and hits[0][0] == tid and hits[0][1] > 0.99   # shadow index finds it
        s.resurrect(tid)
        assert s.get_traces([tid])[tid]["status"] == "active"
        assert s.dense_search(v, 5)[0][0] == tid                  # back in both indexes
        assert s.fts_search(["legacy"])[0][0] == tid


def test_store_supersede():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        s = Store(Path(d) / "t.db", embed_dim=4)
        v = np.array([1, 0, 0, 0], dtype=np.float32)
        old = s.add_trace(1, "old fact", "server is 10.0.0.5", "s1", v, "llm")
        new = s.add_trace(2, "new fact", "server is 10.0.0.9", "s2", v, "llm")
        s.supersede(old, new)
        assert old not in s.get_traces([old, new])  # inactive -> not recallable
        assert new in s.get_traces([old, new])
        assert s.supersedes_something(new)
        assert not s.fts_search(["10.0.0.5"])  # removed from lexical index too
        assert old not in [t for t, _ in s.dense_search(v, 5)]


def test_store_roundtrip(tmp_path=None):
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        s = Store(Path(d) / "t.db", embed_dim=4)
        eid = s.add_episode("sess1", "user text here", "assistant text", "m", "sha1")
        assert eid is not None
        assert s.add_episode("sess1", "user text here", "assistant text", "m", "sha1") is None
        v = np.array([1, 0, 0, 0], dtype=np.float32)
        tid = s.add_trace(eid, "a title", "a gist about peanuts", "sess1", v, "llm",
                          fts_extra="verbatim ECONNRESET slice")
        assert s.dense_search(v, 5)[0][0] == tid
        # FTS finds gist terms AND verbatim-slice terms; porter stemming
        # bridges singular/plural (peanut <-> peanuts)
        assert s.fts_search(["peanuts"])[0][0] == tid
        assert s.fts_search(["peanut"])[0][0] == tid
        assert s.fts_search(["econnreset"])[0][0] == tid
        # nmatch counts distinct matched tokens (store-size-independent admission)
        _, _, nm = s.fts_search(["peanut", "econnreset", "zebra"])[0]
        assert nm == 2
        s.add_access(tid, 0.15)
        row = s.get_traces([tid])[tid]
        assert abs(row["n_access"] - 1.15) < 1e-9
        assert s.queue_depth() == 1


def test_ann_index():
    import tempfile

    from engram.ann import DenseIndex

    rng = np.random.default_rng(3)
    dim, n = 32, 4000

    def normed(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    centers = normed(rng.standard_normal((20, dim)).astype(np.float32))
    noise = normed(rng.standard_normal((n, dim)).astype(np.float32))
    mat = normed(centers[np.arange(n) % 20] + 0.55 * noise)

    # below ann_min: exact brute force
    idx = DenseIndex(dim, ann_min=10_000)
    idx.build(list(range(1, n + 1)), mat)
    assert idx.describe().startswith("exact")
    q = normed(centers[3] + 0.55 * normed(rng.standard_normal(dim).astype(np.float32)))
    assert idx.search(q, 10) == idx.exact_search(q, 10)

    # past ann_min: IVF active, high recall parity on clustered data
    idx = DenseIndex(dim, ann_min=1000)
    idx.build(list(range(1, n + 1)), mat)
    assert "ivf" in idx.describe()
    hits = total = 0
    for i in range(20):
        q = normed(centers[i % 20] + 0.55 * normed(rng.standard_normal(dim).astype(np.float32)))
        exact = {t for t, _ in idx.exact_search(q, 10)}
        got = {t for t, _ in idx.search(q, 10)}
        hits += len(exact & got)
        total += len(exact)
    assert hits / total >= 0.95, f"ivf recall parity {hits / total}"

    # mutations between rebuilds: append findable, tombstone hidden,
    # in-place replace, exact_max skips the dead
    probe = normed(np.ones(dim).astype(np.float32))
    idx.add(n + 1, probe)
    assert idx.search(probe, 1)[0][0] == n + 1
    idx.remove(n + 1)
    assert (n + 1) not in {t for t, _ in idx.search(probe, 5)}
    assert (n + 1) not in idx
    idx.add(7, probe)  # replaces id 7's vector in place
    assert idx.search(probe, 1)[0][0] == 7
    assert idx.exact_max(probe)[0] == 7


def test_store_incremental_index():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        s = Store(Path(d) / "t.db", embed_dim=4)
        va = np.array([1, 0, 0, 0], dtype=np.float32)
        vb = np.array([0, 1, 0, 0], dtype=np.float32)
        a = s.add_trace(1, "fact a", "gist a", "s1", va, "llm")
        b = s.add_trace(2, "fact b", "gist b", "s2", vb, "llm")
        # every mutation below must maintain the indexes incrementally -
        # a full rebuild here means the O(N)-per-write bug is back
        s._rebuild_index = lambda: (_ for _ in ()).throw(
            AssertionError("full rebuild on the write path")
        )
        assert s.dense_search(va, 1)[0][0] == a
        s.silence(a)
        assert a not in {t for t, _ in s.dense_search(va, 2)}
        assert s.silent_search(va, 1)[0][0] == a  # shadow index, immediately
        s.resurrect(a)
        assert s.dense_search(va, 1)[0][0] == a
        assert s.silent_search(va, 1) == []
        s.supersede(b, a)
        assert b not in {t for t, _ in s.dense_search(vb, 2)}
        s.delete_trace(a)
        assert s.dense_search(va, 2) == [] or s.dense_search(va, 2)[0][0] != a
        assert s.index_stale()  # refresher will compact the tombstones


def test_fts_nmatch_at_scale():
    import tempfile

    # the pre-0.5 nmatch used an unordered scan capped at 200 rows per token:
    # past ~200 matching rows it silently undercounted, killing lexical
    # admission in big stores. The rowid-constrained MATCH must stay exact.
    with tempfile.TemporaryDirectory() as d:
        s = Store(Path(d) / "t.db", embed_dim=4)
        rows = [(i, f"filler {i}", "alpha common filler text") for i in range(1, 301)]
        rows.append((301, "target", "alpha bravo the one that matters"))
        s.db.executemany(
            "INSERT INTO traces(id, title, gist, status) VALUES(?,?,?, 'active')",
            [(i, t, g) for i, t, g in rows],
        )
        s.db.executemany(
            "INSERT INTO trace_fts(rowid, title, gist) VALUES(?,?,?)", rows
        )
        s.db.commit()
        got = {tid: nm for tid, _, nm in s.fts_search(["alpha", "bravo"])}
        assert got[301] == 2, f"target nmatch {got.get(301)} (undercounted)"


def test_lex_prune_big_store():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        s = Store(Path(d) / "t.db", embed_dim=4, lex_prune_traces=100)
        blob = np.array([1, 0, 0, 0], dtype=np.float32).tobytes()
        rows = [(i, f"note {i}", "deploy pipeline routine") for i in range(1, 1101)]
        rows.append((1101, "target", "deploy zanzibar incident"))
        s.db.executemany(
            "INSERT INTO traces(id, title, gist, status, embedding)"
            " VALUES(?,?,?, 'active', ?)",
            [(i, t, g, blob) for i, t, g in rows],
        )
        s.db.executemany(
            "INSERT INTO trace_fts(rowid, title, gist) VALUES(?,?,?)", rows
        )
        s.db.commit()
        s._rebuild_index()
        assert s._active.live == 1101
        # "deploy" is in every row (df 1101 > cap 1000): pruned, so the rare
        # token still finds the target fast; an all-common cue finds nothing
        res = s.fts_search(["deploy", "zanzibar"])
        assert res and res[0][0] == 1101 and res[0][2] == 1
        assert s.fts_search(["deploy"]) == []
        # small stores never prune
        s.lex_prune_traces = 20_000
        assert s.fts_search(["deploy"]) != []


def test_index_staleness_and_snapshot():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.db"
        s = Store(p, embed_dim=4)
        assert s._active_idx is None  # lazy: no build until dense access
        assert not s.index_stale()
        v = np.array([1, 0, 0, 0], dtype=np.float32)
        tid = s.add_trace(1, "fact", "gist", "s1", v, "llm")
        assert s.index_stale()  # own write; pure check, not consumed
        assert s.index_stale()
        # snapshot rebuild via a private read-only connection; a clean adopt
        # installs it and clears staleness
        gen, tg = s.index_marks()
        pair = s.rebuild_indexes_snapshot()
        assert pair[0].live == 1
        assert s.adopt_indexes(pair, gen, tg)
        assert not s.index_stale()
        assert s.dense_search(v, 1)[0][0] == tid
        # a trace mutation from ANOTHER process's Store (the MCP server)
        # moves the shared traces_gen counter - detected without any
        # in-process signal. Bookkeeping writes (meta, access, episodes)
        # deliberately do NOT: the MCP process must not rebuild per chat turn
        s2 = Store(p, embed_dim=4)
        vb = np.array([0, 1, 0, 0], dtype=np.float32)
        s2.add_trace(2, "from mcp", "gist2", "s2", vb, "llm")
        assert s.index_stale()
        s.set_meta("last_recall", {"cue": "x"})  # bookkeeping: no signal
        s.add_access(tid, 0.15)
        gen, tg = s.index_marks()
        assert s.adopt_indexes(s.rebuild_indexes_snapshot(), gen, tg)
        assert not s.index_stale()
        assert s.dense_search(vb, 1)[0][0] == 2
        # raw-SQL embedding rewrites (the re-embed path) must announce
        # themselves: mark_index_write moves both staleness signals
        gen, tg = s.index_marks()
        pair = s.rebuild_indexes_snapshot()
        s.db.execute("UPDATE traces SET embedding=? WHERE id=?",
                     (vb.tobytes(), tid))
        s.mark_index_write()
        assert not s.adopt_indexes(pair, gen, tg), "pre-rewrite snapshot adopted"
        assert s.index_stale()
        # the recall-path superseded_by lookup is indexed
        names = {r[0] for r in s.db.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )}
        assert "idx_traces_superseded_by" in names


def test_snapshot_adopt_race():
    import tempfile

    # a write landing between the snapshot read and the adopt must NOT be
    # clobbered: installing the stale snapshot would resurrect silenced
    # traces in the live index and drop fresh ones, and near-dup formation
    # decisions made in that window are permanent
    with tempfile.TemporaryDirectory() as d:
        s = Store(Path(d) / "t.db", embed_dim=4)
        va = np.array([1, 0, 0, 0], dtype=np.float32)
        vb = np.array([0, 1, 0, 0], dtype=np.float32)
        a = s.add_trace(1, "stale fact", "old", "s1", va, "llm")
        gen, dv = s.index_marks()
        pair = s.rebuild_indexes_snapshot()  # snapshot still contains a
        b = s.add_trace(2, "correction", "new", "s2", vb, "llm")
        s.supersede(a, b)                    # races the rebuild
        assert not s.adopt_indexes(pair, gen, dv), "stale snapshot adopted"
        assert a not in {t for t, _ in s.dense_search(va, 2)}
        assert s.dense_search(vb, 1)[0][0] == b
        assert s.index_stale()  # next cycle retries with a fresh snapshot
        gen, dv = s.index_marks()
        assert s.adopt_indexes(s.rebuild_indexes_snapshot(), gen, dv)
        assert not s.index_stale()
        assert a not in {t for t, _ in s.dense_search(va, 2)}
        assert s.max_cosine(va)[0] != a


def test_openai_upstream_routing():
    import asyncio
    import tempfile

    import aiohttp
    from aiohttp import web as aioweb

    from engram.config import Config
    from engram.proxy import make_app

    async def go():
        hits = {"cloud": [], "local": []}

        def stub(name):
            async def h(request):
                hits[name].append(request.path)
                return aioweb.json_response(
                    {"choices": [{"message": {"role": "assistant", "content": "ok"}}],
                     "data": [], "message": {"content": "ok"}, "done": True}
                )
            app = aioweb.Application()
            app.router.add_route("*", "/{tail:.*}", h)
            return app

        async def serve(app):
            runner = aioweb.AppRunner(app)
            await runner.setup()
            site = aioweb.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            return runner, site._server.sockets[0].getsockname()[1]

        cloud_r, cloud_port = await serve(stub("cloud"))
        local_r, local_port = await serve(stub("local"))
        with tempfile.TemporaryDirectory() as d:
            cfg = Config()
            cfg.db_path = Path(d) / "t.db"
            cfg.upstream = f"http://127.0.0.1:{local_port}"
            cfg.openai_upstream = f"http://127.0.0.1:{cloud_port}"
            cfg.services_url = f"http://127.0.0.1:{local_port}"
            proxy_r, proxy_port = await serve(make_app(cfg))
            base = f"http://127.0.0.1:{proxy_port}"
            async with aiohttp.ClientSession() as http:
                # OpenAI chat + companions follow the cloud upstream
                await http.post(f"{base}/v1/chat/completions", json={
                    "model": "m", "stream": False,
                    "messages": [{"role": "user", "content": "hello there friend"}],
                }, headers={"X-Engram": "off"})
                await http.get(f"{base}/v1/models")
                # embeddings deliberately stay local even with a cloud upstream
                await http.post(f"{base}/v1/embeddings", json={"input": "x"})
                # Ollama-native chat stays local no matter what
                await http.post(f"{base}/api/chat", json={
                    "model": "m", "stream": False,
                    "messages": [{"role": "user", "content": "hello there friend"}],
                }, headers={"X-Engram": "off"})
            await proxy_r.cleanup()
        await cloud_r.cleanup()
        await local_r.cleanup()
        assert "/v1/chat/completions" in hits["cloud"], hits
        assert "/v1/models" in hits["cloud"], hits
        assert "/v1/embeddings" in hits["local"], hits
        assert "/api/chat" in hits["local"], hits
        assert "/v1/chat/completions" not in hits["local"], hits

    asyncio.run(go())


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
