# Benchmarks

Every number here was measured, not estimated. Reproduce any of them with the
commands shown. Hardware for the reference numbers: Apple M2 Pro, 16 GB,
macOS, Ollama 0.12, embedder `qwen3-embedding:0.6b` truncated to 256d.

## Recall latency vs store size

`engram bench --synthetic N` seeds a throwaway store with N synthetic traces
and measures the full recall path (embed, hybrid search, scoring, render)
with real embedder calls. `--clustered` draws the vectors around topic
centers, which is what real embedding stores look like; without it they are
uniform random, the worst case for any locality structure. Run it yourself;
seeds are fixed.

| store size | embed p50 / p95 | full recall p50 / p95 | v0.4 recall p50 / p95 |
|---|---|---|---|
| 136 (a real store) | 68.3 / 74.7 ms | 72.7 / 80.1 ms | same |
| 1,000 | 66.9 / 73.8 ms | 69.0 / 77.5 ms | 69.9 / 72.4 ms |
| 10,000 | 70.0 / 73.6 ms | 71.0 / 77.5 ms | 71.2 / 75.0 ms |
| 100,000 | 68.0 / 73.0 ms | 70.6 / 80.4 ms | 152.8 / **497.5 ms** |
| 250,000 | 69.1 / 78.7 ms | 75.4 / 85.3 ms | not runnable |

The table uses uniform-random vectors (worst case for the IVF index);
`--clustered` runs land within noise of the same rows (70.3 / 78.0 ms at
100k, 73.3 / 80.4 ms at 250k).

Reading:

- The pipeline is embed-bound at every measured size: search, scoring, and
  rendering add ~4 ms at 100k and ~9 ms at 250k. The v0.4 ceiling (p95
  crossing the 300 ms fail-open deadline at 100k) is gone.
- Three things removed it, all measured before being built: an index on the
  recall path's supersession lookup (34 ms per injected gist at 100k, now
  microseconds), document-frequency pruning of lexical tokens present in
  more than ~2% of a big store (they carry no discriminative evidence but
  forced FTS5 to rank 81k postings, ~49 ms), and an IVF dense index past
  50k traces (below).
- The same profiling found a correctness bug the latency numbers never
  showed: lexical match counting used a capped unordered scan that silently
  undercounted past ~10k rows, disabling lexical admission in big stores.
  It is now computed exactly at any size (a rowid-constrained FTS query),
  and a regression test pins it.
- Every failure mode is still fail-open: a slow recall forwards the request
  untouched rather than delaying it past the deadline.

## The dense index past 50k traces

Below 50k traces (every personal store we have measured; months of heavy
production use produced ~140) dense search is a single exact matmul - no
approximation, nothing to tune. Past `ENGRAM_ANN_TRACES` (default 50k) the
store builds an IVF index: spherical k-means in pure numpy (no new
dependencies), queries probe the nearest 1/16th of the inverted lists (at
1/8th the gather cost erased the win over the exact matmul at 100k - the
probe fraction was tuned by measurement, like everything else here).
Deletions tombstone, additions land in an always-scanned overflow, and the
daemon compacts in a background thread only when something changed - and a
compaction that a write raced is discarded, never installed, so a silenced
memory can not reappear in the index for even one cycle. The request path
never waits on an index build.

Measured recall@40 against exact brute force on the same store
(`engram bench --synthetic 100000 [--clustered]`; exact search with
top-k selection costs 2.4 ms p50 at 100k for comparison):

| vector distribution | IVF recall@40 | search cost |
|---|---|---|
| clustered (realistic for embeddings) | 1.000 | 1.3 ms p50 |
| uniform random (pathological) | 0.227 | 1.3 ms p50 |

The uniform number is published because it is the honest worst case, and it
is also moot in practice: uniform random 256-d vectors have a mean best
cosine of 0.27 against 100k candidates, at Engram's 0.25 admission
threshold - when vectors carry no cluster structure, almost nothing passes
admission anyway, approximate or not. Real embedding stores are heavily
clustered by construction; at 0.88 mean within-cluster cosine the index
loses nothing. Write-path near-duplicate detection stays exact regardless
(a missed near-dup would pollute the store).

## Memory quality: the built-in regression scenario

`engram eval` replays a scripted 24-turn, 8-session conversation against a
throwaway store, then asks 10 probe questions in fresh sessions. It scores
RETRIEVAL (did the right facts reach the injected block, checked
deterministically against expected content) separately from ANSWER (did the
chat model use them), because those failures have different owners.

| answering model | retrieval | answer |
|---|---|---|
| qwen3:1.7b | 9/9 | 9/9 |
| qwen3:4b | 9/9 | 9/9 |

Probes cover: cross-session semantic recall, literal token recall
(error codes, config values), names and dates, knowledge updates
(vim -> Neovim style corrections), aggregate identity questions, plus two
negative controls (an unrelated question must inject nothing; smalltalk must
skip recall entirely). Retrieval is model-independent by construction. Sub-1B answering models were
tested and dropped from support: they fumble context they are correctly
given (retrieval still scored 9/9; their answers did not).

This is our own scenario, so treat it as a regression suite, not an
independent benchmark. Runs are fully local and free.

## Context against published competitor numbers

Different hardware, different workloads, their numbers from their papers;
directional context only, not a controlled comparison.

| system | retrieval path | published latency |
|---|---|---|
| Engram (this doc) | local hybrid + activation, no LLM | 72 ms p50 / 80 ms p95 at real store size |
| Mem0 (arXiv 2504.19413) | vector search + LLM pipeline | 148 ms p50 / 200 ms p95 search; 0.71 s / 1.44 s end-to-end |
| Zep (arXiv 2501.13956) | temporal knowledge graph | ~300 ms p95 claimed |
| full-context baseline (Mem0 paper) | none | 17.1 s p95 at ~26k tokens |

Structural differences that explain the gap: Engram never calls an LLM on
the request path, runs no network hop (localhost), and bounds every stage
with a deadline. The write path is also different: competitors spend seconds
of LLM tool-calling per exchange; Engram summarizes in the background only
when the GPU is idle, and correctness does not depend on the LLM (extractive
fallback preserves facts verbatim). Past 50k memories the dense leg switches
to the IVF index above and the whole path stays under ~86 ms p95 at 250k.

## LongMemEval-S: the full 500 questions

`engram lme` runs LongMemEval-S (Wu et al., the standard long-term-memory
benchmark) end to end: for every question a FRESH store ingests that
question's ~50 haystack sessions through Engram's real write path
(formation gates, extractive summaries, near-dup reinforcement, conflict
supersession), backdated to the transcript's timestamps so activation and
recency mean what they meant in the conversation. The probe is then
answered by a chat model whose only knowledge of the haystack is Engram's
injected block: 6 gists + 6 titles, at most 2000 tokens, against a ~115k
token haystack. `--watch PORT` serves a live dashboard while it runs.

RETRIEVAL is deterministic and judge-free: a hit means a trace actually
rendered into the injected block traces back (by formation or
reinforcement) to one of the dataset's evidence sessions. ANSWER is scored
by an LLM judge (yes/no correctness). The 30 abstention questions are
excluded from retrieval (their "evidence" sessions deliberately lack the
answer, so no retrieval verdict is meaningful) and judged on declining.

Full run, all 500 questions, both answering models over byte-identical
blocks (the two runs' retrieval columns matched exactly):

| category | retrieval | answer: MiniMax-M3 | answer: qwen3:1.7b |
|---|---|---|---|
| knowledge-update | 72/72 | 56/78 | 31/78 |
| multi-session | 121/121 | 68/133 | 33/133 |
| single-session-user | 63/64 | 58/70 | 45/70 |
| temporal-reasoning | 124/127 | 91/133 | 32/133 |
| single-session-assistant | 53/56 | 21/56 | 20/56 |
| single-session-preference | 28/30 | 20/30 | 8/30 |
| **total** | **461/470 (98.1%)** | **314/500 (62.8%)** | **169/500 (33.8%)** |

Abstention: MiniMax-M3 correctly declined 24/30; qwen3:1.7b 16/30.
Recall p50 was 96ms across five hundred ~210-trace stores.

Reading, honestly:

- Retrieval is near-ceiling, with PERFECT scores on knowledge-update and
  multi-session - the categories built on supersession and cross-session
  accumulation, which is what Engram's write path is for. All 9 misses are
  published: 3 assistant-side facts (user-fact-primacy formation records
  assistant prose weakly, by design), 3 relative-date cues ("last Tuesday"
  - embeddings cannot do date arithmetic; date-aware cue expansion is now
  on the roadmap), 2 vague preference cues (production's profile
  consolidation targets exactly these; the harness does not exercise it),
  and 1 paraphrase that fell below the admission threshold - the only
  empty block in 500 questions.
- The answer column is bounded by what the answering model can do with a
  correct block: same blocks, 63% for a frontier-class model vs 34% for a
  1.7B. Retrieval is model-independent; reading comprehension is not.
  For context, the LongMemEval paper reports full-context frontier models
  around 60% with the entire 115k-token haystack in the window - Engram
  reaches comparable accuracy through a 2000-token block.
- The rest of the answer gap decomposes into four measured weaknesses,
  ranked below.

### Where the lost answers go, ranked

1. **Lossy gists under extractive summarization** (the big one). The
   harness ingests with the extractive fallback: a verbatim truncation of
   ~60 user words plus ~40 assistant words. In 21 of 37 analyzed wrong
   answers, the needed detail was truncated inside the gist - the right
   memory reached the block without the fact the question wanted. The
   budget ablation proves the diagnosis: a 12-gist / 3500-token block
   scored 59/100 vs 61/100 for the standard 6 / 2000 budget. If block
   size were the constraint, doubling it would recover answers; instead
   more gists just add more truncated gists. The loss happens at WRITE
   time. The daemon's primary path is LLM summarization, which distills
   ("completed their 20th course") instead of truncating around the fact;
   the harness cannot afford it at 500 x ~50 sessions on a laptop, so the
   published number is a floor for the production write path. Quantifying
   that gap on a small subset is the next planned ablation.
2. **The answering model's reading ability.** In the other 16 of 37
   misses, the answer was present in the block and the model fumbled it.
   Same blocks, both columns: 62.8% frontier vs 33.8% for a 1.7B.
   Retrieval is model-independent by construction; comprehension is not.
3. **User-fact primacy.** Knowledge that only ever appeared in the
   assistant's words forms weakly, by design: imperatives and facts from
   anyone but the user are what memory poisoning looks like, so the
   asymmetry is deliberate. It caps single-session-assistant (21/56) and
   we publish it as a property rather than quietly special-casing the
   benchmark.
4. **Consolidation is not exercised.** Aggregate and preference questions
   are what the idle-time profile and topic schemas exist for; a
   benchmark run never idles, so those categories rely on point traces
   alone.

Deviations and limitations, in the open: ingestion uses extractive
summaries (the documented fallback), not the LLM summarizer - at 500 x ~50
sessions the LLM path would take days on a laptop. The judge for the
published numbers is MiniMax-M3, which also answered one column: an
element of self-judging we note rather than hide. Dataset: the official
`longmemeval_s` release (MIT), unfiltered. Reproduce with
`engram lme --n 500` (the CLI prints download instructions), roughly 4
hours per answering leg on an M2 Pro, watchable live with `--watch`.

## Not yet benchmarked

LoCoMo remains unrun; its published scores are disputed between vendors
and we would rather publish nothing than unverifiable numbers. The
LLM-summarizer ingestion ablation (small subset, quantifying the
extractive-gist gap) is planned.
