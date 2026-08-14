"""Formation policy: what deserves to become a memory trace, and how.

Field-test findings (2026-08-14) this module answers:
- Probe answers became traces and later competed with real memories
  ("experience-following") -> pure questions form no trace.
- "thanks, this was helpful" became a trace -> content-token gate.
- Corrections reinforced nothing / relied on lucky phrasing -> explicit
  conflict detection drives supersession.
- "badly allergic to peanuts" ranked below travel chit-chat -> write-time
  importance (beta) from novelty + lexical salience, no LLM judge.
"""

import re

# A turn whose first word is interrogative AND that ends in '?' is a pure
# question: the answer is derived content, not user knowledge. Mixed turns
# ("I switched to Neovim - what plugins do you recommend?") start with a
# fact-bearing word and still form traces.
_QWORDS = frozenset(
    "what whats when where who whom whose why how which can could do does did "
    "is are was were should would will any remind tell explain suggest "
    "recommend list show give".split()
)

# Update markers: the user is changing a previously stated fact.
UPDATE_RE = re.compile(
    r"\b(actually|instead|switched|no longer|not .{0,12} anymore|changed to"
    r"|renamed|moved to|updated?|correction|now uses?|is now|from now on)\b",
    re.IGNORECASE,
)

# High-salience content: things that must never lose to chit-chat.
SALIENCE_RE = re.compile(
    r"\b(important|critical|remember|always|never|allerg\w*|deadline|must"
    r"|warning|password|secret|credential|token|hard requirement"
    r"|my name is|(?:the )?user'?s name is)\b",
    re.IGNORECASE,
)

# Identity/preference statements: mildly salient.
IDENTITY_RE = re.compile(
    r"\b(i am|i'm|i work|i use|i live|i prefer|my \w+ is)\b", re.IGNORECASE
)

_NUM_RE = re.compile(r"\d[\d.,:]*")

# ---------------- provenance ----------------
# Whose words is this turn? 'user' (the human, full trust), 'system' (agent
# framework plumbing arriving as user-role turns - never memory-worthy), or
# 'observed:<name>' (third-party content relayed by a gateway, e.g. WhatsApp
# group members - remembered at reduced weight, never salience-boosted).

SYSTEM_RES = [re.compile(p, re.IGNORECASE) for p in (
    r"^review the conversation above",
    r"^the following command was flagged",
    r"^\[context from",
    r"^\[delivered from",
    r"^\[the user sent",
    r"^\[note:",
    r"^\[async delegation",
    r"^you are being (compacted|resumed)",
)]

# Hermes gateway prefixes third-party messages as "[Sender Name] text".
OBSERVED_RE = re.compile(r"^\[([^\[\]]{1,48})\]\s+\S")


def classify_source(text: str) -> str:
    t = (text or "").strip()
    for rx in SYSTEM_RES:
        if rx.search(t):
            return "system"
    m = OBSERVED_RE.match(t)
    if m:
        name = m.group(1).strip()
        # A bracket tag that shouts (mostly caps, long) is framework framing,
        # not a person's name.
        letters = [c for c in name if c.isalpha()]
        if letters and len(name) > 24 and sum(c.isupper() for c in letters) / len(letters) > 0.7:
            return "system"
        return f"observed:{name[:40]}"
    return "user"


def is_pure_question(text: str) -> bool:
    t = text.strip()
    if not t.endswith("?"):
        return False
    first = re.split(r"[\s,]+", t.lower(), maxsplit=1)[0].strip("'\"")
    return first in _QWORDS


def salience(user_text: str) -> float:
    """0..1 lexical salience of USER-typed text (never assistant/tool text -
    imperative phrasing is also the grammar of injection attacks)."""
    if SALIENCE_RE.search(user_text):
        return 1.0
    if UPDATE_RE.search(user_text) or IDENTITY_RE.search(user_text):
        return 0.5
    return 0.0


def beta(novelty: float, user_text: str, source: str = "user") -> float:
    """Write-time importance, ACT-R base-level constant style, range [0, 1.5].
    Salience is earned by the USER's words only - imperative phrasing from
    observed third parties is also the grammar of injection attacks - and
    observed content forms at half weight overall."""
    s = salience(user_text) if source == "user" else 0.0
    b = 0.75 * max(0.0, min(1.0, novelty)) + 0.75 * s
    if source.startswith("observed"):
        b *= 0.5
    return b


def is_conflicting(old_text: str, new_user_text: str, cos: float) -> bool:
    """Does the new statement contradict (rather than repeat) the old trace?
    A correction must supersede the stale trace, never strengthen it."""
    if UPDATE_RE.search(new_user_text):
        return True
    if cos >= 0.85:
        old_nums = set(_NUM_RE.findall(old_text))
        new_nums = set(_NUM_RE.findall(new_user_text))
        if old_nums and new_nums and old_nums.isdisjoint(new_nums):
            return True
    return False
