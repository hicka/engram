# Benchmarks

Every number here was measured, not estimated. Reproduce any of them with the
commands shown. Hardware for the reference numbers: Apple M2 Pro, 16 GB,
macOS, Ollama 0.12, embedder `qwen3-embedding:0.6b` truncated to 256d.

## Recall latency vs store size

`engram bench --synthetic N` seeds a throwaway store with N synthetic traces
and measures the full recall path (embed, hybrid search, scoring, render)
with real embedder calls. Run it yourself; seeds are fixed.

| store size | embed p50 / p95 | full recall p50 / p95 |
|---|---|---|
| 136 (a real store) | 68.3 / 74.7 ms | 72.7 / 80.1 ms |
| 1,000 | 67.9 / 72.5 ms | 69.9 / 72.4 ms |
| 10,000 | 68.4 / 72.0 ms | 71.2 / 75.0 ms |
| 100,000 | 77.4 / 92.6 ms | 152.8 / **497.5 ms** |

Reading:

- Up to ~10k memories the pipeline is embed-bound: search and scoring add
  single-digit milliseconds. This is the realistic personal-store regime
  (months of heavy use produced ~140 traces on our production instance;
  formation gates reject most raw material by design).
- At 100k the brute-force design shows its ceiling: p95 crosses Engram's
  300 ms fail-open deadline, which means some recalls degrade to uninjected
  passthrough. That is the honest current limit. An ANN index and a cheaper
  lexical-count path activate past ~50k in the roadmap; until then Engram is
  correctly sized for personal and per-team stores, not corpus-scale RAG.
- Every failure mode is fail-open: a slow recall forwards the request
  untouched rather than delaying it past the deadline.

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
fallback preserves facts verbatim).

## Not yet benchmarked

LongMemEval and LoCoMo runs require an answering model plus an LLM judge and
careful protocol to be meaningful; the field's published scores are already
disputed between vendors. We would rather publish nothing than unverifiable
numbers. A LongMemEval-subset harness is planned; when it lands, the losing
categories will be published alongside the winning ones.
