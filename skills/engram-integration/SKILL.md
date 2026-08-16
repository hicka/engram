---
name: engram-integration
description: Install and wire Engram, a transparent memory proxy, into a user's LLM stack so their model remembers across sessions. Use when the user asks to add memory to their model or agent, make their assistant remember things, persist conversation knowledge, or set up engram. Covers the Ollama CLI, OpenAI-compatible clients, OpenClaw, Hermes Agent, and Anthropic-protocol cloud endpoints.
---

# Integrating Engram into a user's LLM workflow

Engram (https://hicka.github.io/engram/) is a transparent proxy that sits
between any client and a model server. It injects relevant memories before the
model answers and forms new memories in the background. One process, one
SQLite file at ~/.engram/engram.db, zero client code changes.

Follow the phases in order. Never edit a user's config without writing a
timestamped backup first. Never echo API keys or tokens into output.

## Phase 1: preflight

Run these and act on failures:

    ollama --version && curl -s --max-time 3 http://127.0.0.1:11434/api/version
    python3 -c "import sys; assert sys.version_info >= (3, 11)" && echo py-ok
    lsof -i :11435 -sTCP:LISTEN || echo port-free

- No Ollama: install it first (https://ollama.com). Engram needs local Ollama
  for embeddings and summarization even when the chat model is in the cloud.
- Port 11435 taken: an engram daemon may already run. Check
  `curl -s http://127.0.0.1:11435/engram/stats`; if it answers, skip to Phase 3.

## Phase 2: install

    pip install 'engram-proxy==0.6.0'  # pin, then bump deliberately: pypi.org/project/engram-proxy
    ollama pull qwen3:1.7b qwen3-embedding:0.6b

The two models (~2 GB) are the background summarizer and the retrieval
embedder. Missing summarizer degrades to extractive summaries (still correct);
missing embedder degrades recall to lexical-only. Pull both.

## Phase 3: start the daemon

Prefer a user-owned long-running process (their terminal, launchd, systemd):

    engram up          # proxy on :11435 -> Ollama on :11434

For cloud chat models over the Anthropic protocol, start it with a cloud
route instead (chat goes to the cloud, memory stays local):

    ENGRAM_ANTHROPIC_UPSTREAM=https://api.minimax.io engram up

## Phase 4: wire the client (pick the one that matches)

Back up any config file before editing: `cp CONFIG CONFIG.bak-YYYYMMDD` (use today's date)

**Stock Ollama CLI**

    export OLLAMA_HOST=127.0.0.1:11435    # add to shell profile to persist

**OpenAI-compatible SDK or agent**: change the base URL only.

    base_url = "http://127.0.0.1:11435/v1"

**OpenClaw**: keep its mandated native Ollama API. Add a provider:

    { "models": { "mode": "merge", "providers": {
      "engram": { "baseUrl": "http://127.0.0.1:11435",
                  "api": "ollama", "apiKey": "ollama-local" } } } }

**Hermes Agent** (~/.hermes/config.yaml):

    model:
      default: <ollama model tag>
      provider: ollama
      base_url: http://localhost:11435/v1
      ollama_num_ctx: 65536
      context_length: 65536

  Hermes verifies a real 64k context. Ollama's /v1 ignores per-request
  num_ctx, so bake it into a model variant:
  `printf 'FROM <tag>\nPARAMETER num_ctx 65536\n' > Modelfile && ollama create <tag>-64k -f Modelfile`
  On 16 GB machines, enable KV quantization first or the model will not fit
  comfortably: set env OLLAMA_FLASH_ATTENTION=1 and OLLAMA_KV_CACHE_TYPE=q8_0
  for the Ollama server, then restart it (halves the KV cache).
  Restart the gateway after config changes: `hermes gateway restart`

**Anthropic-protocol cloud client** (requires the Phase 3 cloud route):
point the client's anthropic base URL at `http://127.0.0.1:11435/anthropic`.
Auth headers pass through; do not copy keys into engram config.

## Phase 5: verify before declaring success

1. `curl -s http://127.0.0.1:11435/engram/stats` returns JSON counts.
2. Send one chat through the client, then run `engram why`: the cue shown
   must match the message sent. That proves traffic flows through Engram.
3. Cross-session test: tell the model a distinctive fact, end the session,
   start a new one, ask about the fact obliquely. Memories form during idle
   about 30 seconds AFTER an exchange, so wait before the second session.
4. Show the user the observatory: http://127.0.0.1:11435/engram/ui

## Phase 6 (optional): explicit memory tools for tool-capable agents

Register Engram's MCP server so the agent can search, save, correct, pin,
and forget memories directly (same store as the proxy):

    # Hermes
    hermes mcp add engram-memory --command engram --args mcp
    # any MCP host: stdio server, command "engram", args ["mcp"]

Use the absolute path to the engram binary if the host launches with a
minimal PATH (`which engram`).

## Gotchas that cost real debugging time

- Memory formation is intentionally delayed (GPU-idle gate). A fact is
  recallable in the NEXT session, not instantly. Do not diagnose this as
  a failure.
- Only chat-shaped turns get memory. /api/generate and FIM completions pass
  through uninjected by design.
- Backups of the store must use `sqlite3 ~/.engram/engram.db ".backup f.db"`,
  never bare cp (WAL mode makes cp copies stale).
- Two daemons cannot share port 11435. "address already in use" means one is
  running: `pkill -f "engram up"` before starting another.
- Sub-2B chat models sometimes ignore injected memory even when retrieval is
  correct. Verify retrieval with `engram why` (block content) separately from
  the model's answer.
- Per-request bypass for testing: send header `X-Engram: off`.

## Rollback

Restore the config backup made in Phase 4, `pkill -f "engram up"`, and
optionally `pip uninstall engram-proxy`. The memory store at
~/.engram/engram.db is the user's data; ask before deleting it.
