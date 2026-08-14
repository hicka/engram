"""User-profile consolidation: the replay step, scoped to the highest-value
schema. Point traces answer point questions; identity/aggregate questions
("what do you know about me?") have no cue to match, so they need a distilled
profile - rebuilt off the hot path from USER-source traces only (observed
third-party content never enters the profile as fact).

Single-level by design: the profile is built from traces, never from a
previous profile - no compounding summary drift."""

import re

SYSTEM_PROMPT = (
    "You maintain a factual profile of the USER from their memory notes."
    " Write AT MOST 10 short lines, one durable fact per line: identity, work,"
    " preferences, hard constraints (allergies, deadlines), key people,"
    " projects, and standing instructions. SKIP trivia, arithmetic, greetings,"
    " test messages, and one-off questions - durable facts only. Copy names,"
    " numbers, and values EXACTLY. Only include facts present in the notes."
    " No advice, no speculation, no introduction - just the lines."
    " NEVER infer or invent: if a fact is not written in a note, it does not"
    " go in the profile. When unsure, omit the line. Note: you are the"
    " assistant; lines about the assistant itself do not belong in the USER's"
    " profile."
)

_THINK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)


def select_sources(store, limit: int = 24):
    """User-stated, active, substantive traces: the most reinforced-and-
    important plus the most recent. Thin gists (arithmetic, one-liners) are
    excluded - novelty-driven beta makes trivia look important otherwise."""
    where = (
        "status='active' AND (source='user' OR source IS NULL)"
        " AND length(gist) - length(replace(gist, ' ', '')) >= 5"
    )
    by_weight = store.db.execute(
        f"SELECT id, gist, beta FROM traces WHERE {where}"
        " ORDER BY beta + n_access DESC, id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    recent = store.db.execute(
        f"SELECT id, gist, beta FROM traces WHERE {where} ORDER BY id DESC LIMIT ?",
        (limit // 2,),
    ).fetchall()
    seen, rows = set(), []
    for r in list(by_weight) + list(recent):
        if r["id"] not in seen:
            seen.add(r["id"])
            rows.append(r)
    return rows[:limit]


def build_messages(rows) -> list[dict]:
    notes = "\n".join(f"- {r['gist']}" for r in rows)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Memory notes:\n{notes}"},
    ]


TOPIC_PROMPT = (
    "You summarize one cluster of related memory notes into a single topic"
    " summary. Write ONE paragraph, at most 80 words, naming the topic and its"
    " durable facts with exact names, numbers, and values. Only facts present"
    " in the notes; no advice, no speculation, no introduction."
)


def cluster_topics(store, sim: float = 0.55, min_size: int = 3, max_clusters: int = 4):
    """Greedy embedding clustering over active user-source traces. Returns up
    to max_clusters lists of rows, largest first. Pure arithmetic."""
    import numpy as np

    rows = store.db.execute(
        "SELECT id, gist, embedding FROM traces WHERE status='active'"
        " AND (source='user' OR source IS NULL) AND embedding IS NOT NULL"
    ).fetchall()
    items = []
    for r in rows:
        v = np.frombuffer(r["embedding"], dtype=np.float32)
        if v.shape[0] == store.embed_dim:
            items.append((r, v))
    unassigned = list(range(len(items)))
    clusters = []
    while unassigned:
        seed = unassigned.pop(0)
        members = [seed]
        rest = []
        for j in unassigned:
            if float(items[seed][1] @ items[j][1]) >= sim:
                members.append(j)
            else:
                rest.append(j)
        unassigned = rest
        if len(members) >= min_size:
            clusters.append(members)
    clusters.sort(key=len, reverse=True)
    out = []
    for c in clusters[:max_clusters]:
        rows_c = [items[i][0] for i in c]
        centroid = np.mean([items[i][1] for i in c], axis=0)
        n = np.linalg.norm(centroid)
        out.append((rows_c, centroid / n if n > 0 else centroid))
    return out


def build_topic_messages(rows) -> list[dict]:
    notes = "\n".join(f"- {r['gist']}" for r in rows)
    return [
        {"role": "system", "content": TOPIC_PROMPT},
        {"role": "user", "content": f"Memory notes:\n{notes}"},
    ]


def validate_topic(text: str) -> str | None:
    text = " ".join(_THINK_RE.sub("", text or "").split())
    if not (20 <= len(text) <= 700) or "as an ai" in text.lower():
        return None
    return text


def validate(text: str) -> str | None:
    """Clean and sanity-check the model's profile. None = reject."""
    text = _THINK_RE.sub("", text or "").strip()
    lines = [ln.strip().lstrip("-•* ").strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    if not (3 <= len(lines) <= 14):
        return None
    joined = "\n".join(f"- {ln}" for ln in lines)
    if len(joined) > 1400 or "as an ai" in joined.lower():
        return None
    return joined
