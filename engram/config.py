import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    port: int = int(os.environ.get("ENGRAM_PORT", "11435"))
    upstream: str = os.environ.get("ENGRAM_UPSTREAM", "http://127.0.0.1:11434")
    # Anthropic-protocol traffic (/v1/messages) can route to a different
    # upstream - e.g. MiniMax's Anthropic-compatible cloud endpoint - while
    # OpenAI/Ollama traffic keeps hitting the local upstream. Empty = use
    # `upstream` for everything.
    anthropic_upstream: str = os.environ.get("ENGRAM_ANTHROPIC_UPSTREAM", "")
    # Embeddings, summarization, and /api/ps always run against local Ollama,
    # so cloud chat never adds background API cost.
    services_url: str = os.environ.get("ENGRAM_SERVICES", "http://127.0.0.1:11434")
    db_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("ENGRAM_DB", Path.home() / ".engram" / "engram.db")
        )
    )
    embed_model: str = os.environ.get("ENGRAM_EMBED_MODEL", "qwen3-embedding:0.6b")
    # Field-tested 2026-08-14: qwen3:0.6b as summarizer destroys facts
    # ("badly allergic to peanuts" -> "a brave statement about courage");
    # qwen3:1.7b keeps them verbatim. If the model isn't pulled, summarization
    # falls back to extractive, which also preserves facts - so 1.7b is a
    # strictly safe default.
    summarizer_model: str = os.environ.get("ENGRAM_SUMMARIZER_MODEL", "qwen3:1.7b")
    embed_dim: int = 256  # Matryoshka truncation (qwen3-embedding: 1024 native)

    # Measured on M2 Pro (2026-08-14): qwen3-embedding:0.6b separates related
    # from unrelated cues where nomic-embed-text ties them; warm ~65ms/query.
    # nomic-embed-text (~19ms) works with prefixes but needs admit_cosine ~0.46
    # and has far thinner margins.
    # Warm p50 ~65ms, but generation on the shared GPU pushes spikes past
    # 100ms - the cap must absorb contention, not just the warm path.
    embed_timeout_s: float = 0.250  # embed leg cap; breach -> lexical-only
    recall_deadline_s: float = 0.300  # whole-recall cap; breach -> no injection
    # Calibrated on-machine: related cues 0.27-0.42, unrelated max ~0.23.
    # The smalltalk filter (min_content_tokens) guards content-free cues
    # independently of cosine.
    admit_cosine: float = 0.25
    min_content_tokens: int = 2
    # Lexical admission: N distinct non-stopword cue tokens literally present
    # in the trace. Store-size-independent, unlike bm25 magnitudes, which
    # FTS5's idf clamp drives to ~1e-6 in small/homogeneous stores.
    admit_lex_tokens: int = 2
    session_ttl_s: float = 1800.0  # same-opener conversations merge within this
    # Comma-separated source labels whose traces are never injected (still
    # stored and searchable via MCP), e.g. "observed:Ops Team".
    # "observed" alone quarantines ALL third-party content.
    quarantine_sources: str = os.environ.get("ENGRAM_QUARANTINE", "")

    # profile consolidation (idle-time replay)
    profile_min_traces: int = 3       # first profile once this many user traces exist
    profile_refresh_traces: int = 8   # rebuild after this many new user traces
    profile_attempt_cooldown_s: float = 600.0
    # Empty = fall back to summarizer_model (1.7b non-thinking, field-tested:
    # faithful to notes once sources are filtered and the prompt forbids
    # invention). qwen3:4b thinking mode was tried and never converges on
    # this task; override via env if a stronger non-qwen model is available.
    profile_model: str = os.environ.get("ENGRAM_PROFILE_MODEL", "")
    admit_bm25: float = 1.2  # -bm25() score floor for lexical admission
    max_gists: int = 2  # tier S (<4B upstream)
    max_titles: int = 2
    token_budget: int = 350
    # Prompts above this size get the memory block appended to the newest
    # user message instead of the system top: preserves the llama.cpp
    # prompt-prefix cache (measured 90s -> ~1s per message for a 23k prompt).
    big_prompt_tokens: int = 4000

    # scoring
    decay_d: float = 0.5
    act_weight: float = 0.5
    act_scale: float = 0.4
    w_create: float = 1.0
    w_inject: float = 0.15
    w_repeat: float = 0.5
    beta_pin: float = 2.0  # pinned traces outrank everything and never decay out

    # lifecycle: forgetting as retrieval failure. A single-access trace's
    # base level reaches ~-3.3 at 30 days; sustained reinforcement or high
    # beta keeps a trace above the line indefinitely.
    silence_activation: float = -3.5
    silence_min_age_days: float = 30.0
    decay_sweep_cooldown_s: float = 21600.0  # 6h between sweeps

    # adaptive injection: budget scales with what the upstream model can
    # actually exploit (sub-7B peaks at 2-4 passages; big/cloud models more).
    tier_m_params: float = 4e9    # >= this -> tier M
    tier_l_params: float = 14e9   # >= this -> tier L (cloud upstreams too)
    tier_m_budget: tuple = (4, 4, 1000)   # (max_gists, max_titles, tokens)
    tier_l_budget: tuple = (6, 6, 2000)

    def __post_init__(self):
        self.upstream = self.upstream.rstrip("/")
        self.anthropic_upstream = self.anthropic_upstream.rstrip("/")
        self.services_url = self.services_url.rstrip("/")

    # write path
    min_episode_chars: int = 15
    near_dup_cosine: float = 0.92
    conflict_cosine: float = 0.60  # similarity floor for correction detection
    # Agent clients (Hermes/OpenClaw) pause 2-20s between tool iterations; a
    # short idle gate made the summarizer load its model mid-run and evict the
    # agent's (7.5GB at 64k ctx), causing a reload-thrash loop on 16GB Macs.
    idle_before_summarize_s: float = 30.0
    keep_warm_interval_s: float = 240.0
    # When a resident chat model is at least this big, background LLM work
    # would evict it - prefer extractive summaries and skip keep-warm pings.
    big_model_bytes: int = 4_000_000_000


SENTINEL_OPEN = "<engram:memory v=1>"
SENTINEL_CLOSE = "</engram:memory>"
