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

## Not yet benchmarked

LongMemEval and LoCoMo runs require an answering model plus an LLM judge and
careful protocol to be meaningful; the field's published scores are already
disputed between vendors. We would rather publish nothing than unverifiable
numbers. A LongMemEval-subset harness is planned; when it lands, the losing
categories will be published alongside the winning ones.
