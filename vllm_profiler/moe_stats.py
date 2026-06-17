# SPDX-License-Identifier: Apache-2.0
"""MoE transfer statistics by batch type (prefill / decode).

Reports dispatch/combine bytes, transferred/received tokens, and padding -- each
as mean / p50 / p90 / p99 -- split by `batch_type`. Reuses summarize's recursive
loader (so multi-node merges + per-file init skip work the same).

    python -m vllm_profiler.moe_stats ./vllm_prof_out
    python -m vllm_profiler.moe_stats ./merged --skip=0 --include-dummy
"""

from __future__ import annotations

import sys
from collections import defaultdict

from .summarize import _DEFAULT_SKIP, _load

# (label, record kind, value fn, unit divisor, unit label)
_METRICS = [
    # send = data this rank pushes into the collective, recv = data it gets out.
    # dispatch(all-gather): send small (my tokens) -> recv big (all tokens).
    # combine(reduce-scatter): send big (my expert outputs) -> recv small (my results).
    ("dispatch_send", "moe_dispatch_size", lambda r: r.get("bytes_in"), 1e6, "MB"),
    ("dispatch_recv", "moe_dispatch_size", lambda r: r.get("bytes_recv"), 1e6, "MB"),
    ("combine_send",  "moe_combine_size", lambda r: r.get("bytes_in"), 1e6, "MB"),
    ("combine_recv",  "moe_combine_size", lambda r: r.get("bytes_out"), 1e6, "MB"),
    ("tokens_sent",   "moe_dispatch_size", lambda r: r.get("tokens_in"), 1, "tok"),
    ("tokens_recv",   "moe_dispatch_size", lambda r: r.get("tokens_recv"), 1, "tok"),
    ("amplification", "moe_dispatch_size",
     lambda r: (r.get("tokens_recv") / r["tokens_in"]) if r.get("tokens_in") else None,
     1, "x"),
    ("routing_slots", "moe_dispatch_size", lambda r: r.get("routing_slots_sent"), 1, "-"),
    ("per_token_bytes", "moe_dispatch_size", lambda r: r.get("per_token_bytes"), 1, "B"),
    ("num_tokens",    "moe_call", lambda r: r.get("num_tokens"), 1, "tok"),
    ("tokens_before_chunk", "moe_call", lambda r: r.get("tokens_before_chunk"), 1, "tok"),
    ("pad_tokens_this_rank", "moe_call", lambda r: r.get("pad_tokens_this_rank"), 1, "tok"),
    ("pad_total",     "moe_call", lambda r: r.get("pad_total"), 1, "tok"),
]


def _percentile(sorted_xs: list[float], q: float) -> float:
    """Linear-interpolation percentile (numpy default) on a pre-sorted list."""
    n = len(sorted_xs)
    if n == 1:
        return sorted_xs[0]
    idx = (q / 100.0) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac


def _agg(xs: list[float]) -> dict | None:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    s = sorted(xs)
    return {
        "n": len(s),
        "mean": sum(s) / len(s),
        "p50": _percentile(s, 50),
        "p90": _percentile(s, 90),
        "p99": _percentile(s, 99),
    }


def moe_stats(path: str, skip: int | None = None, include_dummy: bool = False) -> None:
    if skip is None:
        skip = _DEFAULT_SKIP
    rows = _load(path, skip=skip)
    if not rows:
        print(f"No records under {path!r} (after skipping {skip} lines/file; "
              f"try --skip=0)")
        return
    n_dummy = sum(1 for r in rows if r.get("dummy"))
    if not include_dummy:
        rows = [r for r in rows if not r.get("dummy")]

    # batch_type -> kind -> [records]
    grouped: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        grouped[r.get("batch_type") or "unknown"][r.get("kind")].append(r)

    print(f"\n=== MoE transfer stats by batch_type "
          f"(skip={skip}/file, dummy {'kept' if include_dummy else 'dropped'}"
          f"{f': {n_dummy}' if n_dummy else ''}) ===")

    order = [bt for bt in ("prefill", "decode", "mixed", "unknown") if bt in grouped]
    for bt in order:
        kinds = grouped[bt]
        print(f"\n[{bt}]")
        print(f"  {'metric':<22}{'unit':>5}{'n':>9}{'mean':>12}"
              f"{'p50':>12}{'p90':>12}{'p99':>12}")
        for label, kind, fn, div, unit in _METRICS:
            st = _agg([fn(r) for r in kinds.get(kind, [])])
            if st is None:
                continue
            print(f"  {label:<22}{unit:>5}{st['n']:>9}"
                  f"{st['mean'] / div:>12.3f}{st['p50'] / div:>12.3f}"
                  f"{st['p90'] / div:>12.3f}{st['p99'] / div:>12.3f}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    include_dummy = "--include-dummy" in argv
    skip = None
    for a in argv:
        if a.startswith("--skip="):
            skip = int(a.split("=", 1)[1])
    pos = [a for a in argv if not a.startswith("--")]
    moe_stats(pos[0] if pos else "./vllm_prof_out", skip=skip,
              include_dummy=include_dummy)
