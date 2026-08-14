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
