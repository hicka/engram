"""LongMemEval-S subset harness.

Protocol, per question: a FRESH store ingests that question's haystack
sessions through Engram's real write path (formation gates, extractive
summaries, near-dup reinforcement, conflict supersession), backdated to
the dataset's session timestamps so activation and recency mean what
they meant in the transcript. The probe is then answered by a chat
model whose only knowledge of the haystack is Engram's injected block.

RETRIEVAL is scored deterministically without any judge: the dataset
names which sessions contain the evidence; the harness tracks every
session that formed OR reinforced each trace, and a hit means a trace
actually rendered into the injected block (recall's per-candidate
"shown" flag, not merely a top-scored candidate) carries an evidence
session. Abstention (_abs) questions are excluded from the retrieval
column: their "evidence" sessions are the topically-nearest sessions
that deliberately lack the answer, so retrieving them is correct
behavior and no retrieval criterion is meaningful - they still count
in the judged ANSWER column, where the model must decline.

ANSWER is scored by an LLM judge with LongMemEval's yes/no correctness
framing. Answering and judging models speak either Ollama (local,
default, free) or an Anthropic-protocol endpoint (--answer-api
anthropic, key from MINIMAX_API_KEY or LME_API_KEY env) so the same
blocks can be answered by a frontier cloud model for comparison.

Honest deviations from the paper's protocol, in the open:
- Stratified subset, not all 500 questions (--n 500 works if you have
  the patience).
- Ingestion uses Engram's extractive summaries, not the LLM summarizer:
  at ~50 sessions per question the LLM path would take days on a
  laptop, and extractive is the documented fallback that preserves
  facts verbatim. Categories needing assistant-side knowledge
  (single-session-assistant) are expected to suffer; that is a real
  property of the current formation policy, published, not hidden.
- Judge quality bounds answer-score quality; the deterministic
  retrieval column is judge-free.

Dataset (278MB, MIT): download longmemeval_s from
https://huggingface.co/datasets/xiaowu0162/longmemeval and place it at
.lme/longmemeval_s.json (path configurable via --data).
"""

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import tempfile
import time
from pathlib import Path

import aiohttp

from . import formation
from .config import Config
from .recall import Recall, cue_tokens
from .store import Store
from .summarizer import extractive_fallback

_DATE_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2}).*?(\d{2}):(\d{2})")

JUDGE_STD = (
    "I will give you a question, the correct answer, and a response from a"
    " model. Reply with exactly yes or no: yes if the response contains the"
    " correct answer or an equivalent statement of it, no otherwise. Extra"
    " detail is fine; contradicting or missing the answer is not.\n\n"
    "Question: {q}\nCorrect answer: {gold}\nModel response: {hyp}\n\n"
    "Reply yes or no only."
)
JUDGE_ABS = (
    "I will give you a question and a response from a model. The question"
    " is NOT answerable from the information the model was given, so the"
    " correct behavior is to say it does not know or lacks that"
    " information. Reply with exactly yes or no: yes if the model declined"
    " or said it does not know, no if it invented an answer.\n\n"
    "Question: {q}\nModel response: {hyp}\n\n"
    "Reply yes or no only."
)


def _parse_ts(s: str) -> float:
    m = _DATE_RE.search(s or "")
    if not m:
        return time.time()
    y, mo, d, h, mi = (int(x) for x in m.groups())
    return time.mktime((y, mo, d, h, mi, 0, 0, 0, -1))


def _sample(questions: list[dict], n: int, seed: int = 11) -> list[dict]:
    """Stratified by question_type, deterministic."""
    if n >= len(questions):
        return questions
    rng = random.Random(seed)
    by_type: dict[str, list] = {}
    for q in questions:
        by_type.setdefault(q["question_type"], []).append(q)
    out = []
    for qtype, qs in sorted(by_type.items()):
        k = max(1, round(n * len(qs) / len(questions)))
        out.extend(rng.sample(qs, min(k, len(qs))))
    rng.shuffle(out)
    return out[:n]


def _pair_exchanges(session: list[dict]) -> list[tuple[str, str]]:
    """Collapse a role-turn list into (user, assistant) exchanges."""
    out, user = [], None
    for turn in session:
        role, content = turn.get("role"), (turn.get("content") or "").strip()
        if role == "user":
            if user:
                out.append((user, ""))
            user = content
        elif role == "assistant" and user is not None:
            out.append((user, content))
            user = None
    if user:
        out.append((user, ""))
    return out


def _backdated_access(store: Store, tid: int, ts: float, weight: float):
    """add_access with the transcript's clock, not 2026's: a wall-clock
    reinforcement would hand every near-dup a ~+0.5 recency boost over the
    purely backdated evidence traces and invert the ranking the backdating
    exists to preserve."""
    row = store.db.execute(
        "SELECT last8, n_access FROM traces WHERE id=?", (tid,)
    ).fetchone()
    if row is None:
        return
    last8 = (json.loads(row["last8"]) + [[ts, weight]])[-8:]
    store.db.execute(
        "UPDATE traces SET last8=?, n_access=? WHERE id=?",
        (json.dumps(last8), row["n_access"] + weight, tid),
    )


async def _ingest(
    store: Store, rec: Recall, cfg: Config, q: dict
) -> tuple[dict, dict[int, set]]:
    """Replay one question's haystack through the write path, backdated.
    Returns (stats, trace_id -> set of session_ids that formed OR
    reinforced it) - session attribution must survive near-dup merging or
    a merged evidence fact scores as a retrieval miss."""
    formed = reinforced = superseded = skipped = embeds_failed = 0
    sessions_of: dict[int, set] = {}
    for sid, date, session in zip(
        q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"]
    ):
        ts = _parse_ts(date)
        for user_text, assistant_text in _pair_exchanges(session):
            # the same gates the daemon applies (summarizer.tick)
            if (
                len(user_text) < cfg.min_episode_chars
                or len(cue_tokens(user_text)) < cfg.min_content_tokens
                or formation.is_pure_question(user_text)
            ):
                skipped += 1
                continue
            title, gist = extractive_fallback(user_text, assistant_text)
            emb = await rec.embed(
                f"{title}. {gist}\n{user_text[:300]}", timeout=10.0, role="doc"
            )
            if emb is None:
                embeds_failed += 1
                continue
            # ADD vs REINFORCE vs SUPERSEDE, same arithmetic as the daemon
            near_id, near_cos = store.max_cosine(emb)
            conflict = False
            if near_id is not None and near_cos >= cfg.conflict_cosine:
                old = store.get_traces([near_id]).get(near_id)
                if old is not None:
                    conflict = formation.is_conflicting(
                        f"{old['title']}. {old['gist']}", user_text, near_cos
                    )
            if near_id is not None and not conflict and near_cos >= cfg.near_dup_cosine:
                _backdated_access(store, near_id, ts, cfg.w_repeat)
                sessions_of.setdefault(near_id, set()).add(sid)
                reinforced += 1
                continue
            novelty = 1.0 - near_cos if near_id is not None else 1.0
            tid = store.add_trace(
                None, title, gist, sid, emb, "extractive",
                fts_extra=user_text[:400],
                beta=formation.beta(novelty, user_text, "user"),
            )
            sessions_of[tid] = {sid}
            if conflict:
                store.supersede(near_id, tid)
                superseded += 1
            # backdate to the transcript's clock so decay and recency are real
            store.db.execute(
                "UPDATE traces SET created_ts=?, last8=? WHERE id=?",
                (ts, json.dumps([[ts, cfg.w_create]]), tid),
            )
            formed += 1
    store.db.commit()
    store._rebuild_index()
    stats = {
        "formed": formed, "reinforced": reinforced, "superseded": superseded,
        "skipped": skipped, "embeds_failed": embeds_failed,
    }
    return stats, sessions_of


def _api_key() -> str:
    key = os.environ.get("LME_API_KEY") or os.environ.get("MINIMAX_API_KEY", "")
    if not key:
        raise SystemExit(
            "anthropic API selected but neither LME_API_KEY nor"
            " MINIMAX_API_KEY is set"
        )
    return key


async def _chat(http, api: str, base: str, model: str, system: str, user: str,
                n_predict: int = 220) -> str:
    if api == "anthropic":
        async with http.post(
            f"{base}/v1/messages",
            headers={"Authorization": f"Bearer {_api_key()}",
                     "anthropic-version": "2023-06-01"},
            json={"model": model, "max_tokens": n_predict, "system": system,
                  "messages": [{"role": "user", "content": user}]},
        ) as r:
            text = await r.text()
        if r.status != 200:
            raise RuntimeError(f"anthropic upstream {r.status}: {text[:200]}")
        data = json.loads(text)
        return " ".join(
            b.get("text", "") for b in data.get("content", [])
            if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
    out = []
    async with http.post(
        f"{base}/api/chat",
        json={
            "model": model, "think": False, "stream": True,
            "options": {"num_predict": n_predict, "temperature": 0.1},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
    ) as r:
        async for line in r.content:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line).get("message", {}).get("content", ""))
                except json.JSONDecodeError:
                    pass
    return "".join(out).strip()


def _write_atomic(path: Path, data: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)


def _load_data(args) -> list[dict]:
    p = Path(args.data)
    if not p.exists():
        raise SystemExit(
            f"dataset not found at {p}\n"
            "Download longmemeval_s (278MB, MIT) from"
            " https://huggingface.co/datasets/xiaowu0162/longmemeval :\n"
            f"  mkdir -p {p.parent} && curl -L -o {p} "
            "'https://huggingface.co/datasets/xiaowu0162/longmemeval/"
            "resolve/main/longmemeval_s'"
        )
    return json.loads(p.read_text())


async def run(cfg: Config, args) -> None:
    picked = _sample(_load_data(args), args.n)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows, done_ids = [], set()
    if out_path.exists():  # resume: keep completed questions, skip them
        try:
            rows = json.loads(out_path.read_text())["rows"]
            done_ids = {r["question_id"] for r in rows}
            if done_ids:
                print(f"resuming: {len(done_ids)} questions already done")
        except Exception:
            rows, done_ids = [], set()
    t_start = time.time()
    async with aiohttp.ClientSession() as http:
        for i, q in enumerate(picked, 1):
            qid = str(q["question_id"])
            if qid in done_ids:
                continue
            abstention = qid.endswith("_abs")
            with tempfile.TemporaryDirectory(prefix="engram-lme-") as d:
                store = Store(Path(d) / "lme.db", cfg.embed_dim)
                rec = Recall(store, cfg, http)
                rec.cfg.embed_timeout_s = 10.0
                ing, sessions_of = await _ingest(store, rec, cfg, q)

                t0 = time.perf_counter()
                block, debug = await rec.recall(
                    q["question"], "__lme__", dry_run=True,
                    limits=cfg.tier_l_budget,
                )
                recall_ms = (time.perf_counter() - t0) * 1000

                # retrieval: only traces actually rendered into the block,
                # scored against every session that contributed to them
                evidence = set(q.get("answer_session_ids") or [])
                shown_sessions: set = set()
                for c in debug:
                    if c.get("shown"):
                        shown_sessions |= sessions_of.get(c["id"], set())
                retrieval_ok = (
                    None if abstention else bool(shown_sessions & evidence)
                )

                system = (
                    "You are an assistant with a memory of past conversations"
                    " with this user."
                    + (f"\n\n{block}" if block else "")
                    + "\nAnswer from these memories. If they do not contain"
                    " the answer, say you do not know. Be brief."
                )
                user = f"Today is {q.get('question_date', '')}. {q['question']}"
                hyp = await _chat(http, args.answer_api, args.answer_base
                                  or cfg.services_url, args.answer_model,
                                  system, user)
                rows.append({
                    "question_id": qid, "type": q["question_type"],
                    "abstention": abstention, "question": q["question"],
                    "gold": str(q["answer"]), "hypothesis": hyp,
                    "block": block, "block_present": block is not None,
                    "retrieval_ok": retrieval_ok, "recall_ms": recall_ms,
                    **ing,
                })
            print(f"[{i}/{len(picked)}] {qid} formed={ing['formed']} "
                  f"retr={'n/a' if retrieval_ok is None else ('PASS' if retrieval_ok else 'MISS')} "
                  f"({time.time() - t_start:.0f}s elapsed)", flush=True)
            _write_atomic(out_path, {
                "config": {"n": args.n, "answer_api": args.answer_api,
                           "answer_model": args.answer_model,
                           "data": str(args.data)}, "rows": rows})

    if not args.no_judge:
        await judge(cfg, args)
    _report(json.loads(out_path.read_text()))


async def judge(cfg: Config, args) -> None:
    out_path = Path(args.out)
    data = json.loads(out_path.read_text())
    async with aiohttp.ClientSession() as http:
        for row in data["rows"]:
            if "judged_correct" in row and not args.rejudge:
                continue
            tpl = JUDGE_ABS if row["abstention"] else JUDGE_STD
            prompt = tpl.format(q=row["question"], gold=row["gold"],
                                hyp=row["hypothesis"][:1200])
            verdict = await _chat(http, args.judge_api, args.judge_base
                                  or cfg.services_url, args.judge_model,
                                  "You are a strict grader.", prompt,
                                  n_predict=8)
            row["judged_correct"] = verdict.strip().lower().startswith("yes")
            row["judge_model"] = args.judge_model
            _write_atomic(out_path, data)  # a dead judge loses nothing


def _report(data: dict) -> None:
    rows = data["rows"]
    if not rows:
        print("no rows")
        return
    by_type: dict[str, list] = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r)
    judged = any("judged_correct" in r for r in rows)
    print(f"\n{'category':26s} {'n':>3s}  {'retrieval':>9s}  {'answer':>7s}")
    for t, rs in sorted(by_type.items()):
        scored = [r for r in rs if r["retrieval_ok"] is not None]
        retr = f"{sum(r['retrieval_ok'] for r in scored)}/{len(scored)}" if scored else "n/a"
        ans = sum(r.get("judged_correct", False) for r in rs)
        ans_s = f"{ans}/{len(rs)}" if judged else "-"
        print(f"{t:26s} {len(rs):3d}  {retr:>9s}  {ans_s:>7s}")
    scored = [r for r in rows if r["retrieval_ok"] is not None]
    retr = sum(r["retrieval_ok"] for r in scored)
    ans = sum(r.get("judged_correct", False) for r in rows)
    ms = statistics.median([r["recall_ms"] for r in rows])
    n_abs = sum(1 for r in rows if r["abstention"])
    print(f"{'TOTAL':26s} {len(rows):3d}  {retr:>5d}/{len(scored):<3d}  "
          f"{(f'{ans}/{len(rows)}' if judged else '-'):>7s}   "
          f"recall p50 {ms:.0f}ms")
    if n_abs:
        print(f"({n_abs} abstention questions: excluded from retrieval,"
              f" judged on declining to answer)")


def main(cfg: Config, args) -> None:
    if args.judge_only:
        asyncio.run(judge(cfg, args))
        _report(json.loads(Path(args.out).read_text()))
    else:
        asyncio.run(run(cfg, args))


def add_parser(sub) -> None:
    sp = sub.add_parser(
        "lme", help="LongMemEval-S subset run (see BENCHMARKS.md)")
    sp.add_argument("--data", default=".lme/longmemeval_s.json")
    sp.add_argument("--n", type=int, default=60)
    sp.add_argument("--out", default=".lme/results.json")
    sp.add_argument("--answer-api", choices=("ollama", "anthropic"),
                    default="ollama")
    sp.add_argument("--answer-base", default="",
                    help="answer endpoint base URL (default: local Ollama)")
    sp.add_argument("--answer-model", default="qwen3:4b")
    sp.add_argument("--judge-api", choices=("ollama", "anthropic"),
                    default="ollama")
    sp.add_argument("--judge-base", default="")
    sp.add_argument("--judge-model", default="qwen3:4b")
    sp.add_argument("--no-judge", action="store_true")
    sp.add_argument("--judge-only", action="store_true",
                    help="re-score an existing results file")
    sp.add_argument("--rejudge", action="store_true")
