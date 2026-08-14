# Engram

Give your local model a memory. One process, one SQLite file, zero code changes.

Engram is a transparent proxy that sits between any client and Ollama. It recalls
relevant memories from past sessions and injects them before the model answers
(~22 ms overhead, no LLM on the request path), then summarizes each exchange in
the background - using the same small model you're chatting with.

```
                 ┌─────────────────────────────┐
  client ──────► │ engramd :11435              │ ──────► ollama :11434
  (curl,         │  recall → inject → forward  │
   openclaw,     │  response tee → episode log │
   hermes)       │  idle-time summarizer       │
                 └──────────┬──────────────────┘
                            └── engram.db (SQLite: FTS5 + vectors + queue)
```

## Install

```bash
pip install git+https://github.com/hicka/engram.git    # or: uvx --from git+https://github.com/hicka/engram engram up
ollama pull qwen3:1.7b qwen3-embedding:0.6b
```

## Quick start

```bash
engram up

# in another terminal - stock ollama, now with memory:
export OLLAMA_HOST=127.0.0.1:11435
ollama run qwen3:0.6b "I'm Sam. My cat is Miso and I'm badly allergic to peanuts."
# ... later, in a NEW session:
ollama run qwen3:0.6b "Ordering Thai tonight - anything I should avoid?"
```

Requires Ollama with `qwen3:1.7b` (summarizer) and `qwen3-embedding:0.6b`
(retrieval) - `ollama pull qwen3:1.7b qwen3-embedding:0.6b`, ~2 GB total.
Field-tested defaults: a 0.6b summarizer garbles facts ("allergic to peanuts"
became "a brave statement"), so 1.7b is the default; without it, summaries
fall back to extractive, which also preserves facts. Set
`ENGRAM_SUMMARIZER_MODEL` / `ENGRAM_EMBED_MODEL` to override
(`nomic-embed-text` is ~3x faster to embed but measurably worse at admission).

## CLI

```
engram up               run the proxy (default :11435 -> :11434)
engram list             recent traces (title + gist + access stats)
engram recall "query"   dry-run recall with per-candidate scores
engram why              explain the last injection, score by score
engram stats            store counts, queue depth, degraded recalls
engram bench            measure embed + recall latency on your machine
```

## Design in one paragraph

Every exchange becomes an **episode** (verbatim, SHA-deduped). A write-behind
worker - running only when the GPU is idle - asks qwen3:0.6b for two plain
lines (`TITLE:` / `GIST:`; never JSON, with a deterministic extractive fallback)
and embeds the gist (qwen3-embedding, Matryoshka-truncated to 256d). Recall is
hybrid BM25 + brute-force dense with **admission on raw thresholds** (unrelated
turns inject nothing), ranked by relevance + ACT-R activation
`B = ln(Σ w·t^-0.5)` - frequent-and-recent memories surface first, unused ones
sink, and injection itself reinforces at only 0.15 weight (capped per session)
so the rich-get-richer loop stays closed. Injection is budgeted for small
models: ≤2 gists + 2 titles, ≤350 tokens, placed first in the system message.
Failures fail open: a slow or broken pipeline degrades to a plain proxy.

Full architecture spec: see the published design document.

## Cloud models

Engram can give a cloud model the same memory, with background work staying
local and free:

```bash
# route Anthropic-protocol traffic (/v1/messages) to a cloud endpoint;
# OpenAI/Ollama routes keep hitting local Ollama
ENGRAM_ANTHROPIC_UPSTREAM=https://api.minimax.io .venv/bin/python -m engram up

# point your client's Anthropic base_url at Engram, e.g. Hermes Agent + MiniMax:
#   MINIMAX_BASE_URL=http://127.0.0.1:11435/anthropic
```

Auth headers pass through untouched; embeddings and summarization always run
on local Ollama (`ENGRAM_SERVICES`, default `http://127.0.0.1:11434`), so the
only added cloud cost is the injected block itself (&le;350 tokens, ~1% of a
typical agent prompt). Big-prompt injection rides the newest message, so
provider prompt caches - and their cache-read discounts - survive.

## Endpoints

- `POST /api/chat` - Ollama-native NDJSON, memory-enabled
- `POST /v1/chat/completions` - OpenAI-compatible SSE, memory-enabled
- `POST [/prefix]/v1/messages` - Anthropic Messages (local or cloud upstream), memory-enabled
- everything else (`/api/generate`, `/api/tags`, `/api/show`, `/v1/models`, embeddings) - byte passthrough
- `GET /engram/stats`, `GET /engram/why` - local introspection
- header `X-Engram: off` bypasses memory per-request; `X-Engram-Session` pins session identity
