"""ACT-R base-level activation. B = ln(sum w_j * t_j^-d), d = 0.5.

Anderson & Schooler: this quantity tracks the log-odds a memory is needed now.
Accesses are weighted re-presentations (testing effect): creation 1.0,
user re-reference 1.0, injection 0.15 (capped once per session), repeat 0.5.

Only the 8 most recent events are stored exactly; the evicted tail is
approximated by the ACT-R hybrid optimized-learning form (Petrov 2006) using
the running weight total (n_access) and the trace's creation time, so old
high-weight history is never erased by recent low-weight injections.
"""

import math
import time


def base_level(
    last8: list[list[float]],
    n_access: float = 0.0,
    created_ts: float | None = None,
    now: float | None = None,
    d: float = 0.5,
) -> float:
    now = time.time() if now is None else now
    s = 0.0
    for ts, w in last8:
        dt_h = max((now - ts) / 3600.0, 0.01)
        s += w * dt_h ** -d
    # Evicted-tail approximation: weight mass beyond the exact window
    # contributes its integral-average decayed strength over the interval
    # [creation, oldest retained event].
    tail_w = max(n_access - sum(w for _, w in last8), 0.0)
    if tail_w > 0 and created_ts is not None:
        t_life = max((now - created_ts) / 3600.0, 0.01)
        t_old = (
            max((now - last8[0][0]) / 3600.0, 0.01) if last8 else 0.01
        )
        if t_life > t_old + 1e-9:
            s += tail_w * (t_life ** (1 - d) - t_old ** (1 - d)) / (
                (1 - d) * (t_life - t_old)
            )
        else:
            s += tail_w * t_life ** -d
    return math.log(s) if s > 0 else -10.0


def sigmoid(x: float) -> float:
    if x < -30:
        return 0.0
    if x > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))
