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
    mean = sum(s) / len(s)
    var = sum((x - mean) ** 2 for x in s) / len(s)
    return {
        "n": len(s),
        "mean": mean,
        "p50": _percentile(s, 50),
        "p90": _percentile(s, 90),
        "p99": _percentile(s, 99),
        "min": s[0],
        "max": s[-1],
        "cov": (var ** 0.5) / mean if mean else 0.0,
    }


def _histogram(xs: list[float], edges: list[float]) -> None:
    """Bucketed count histogram with bar, for distribution inspection."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return
    counts = [0] * (len(edges) + 1)
    for x in xs:
        placed = False
        for i, e in enumerate(edges):
            if x < e:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    total = len(xs)
    labels = ([f"<{edges[0]:g}"]
              + [f"{edges[i]:g}-{edges[i + 1]:g}" for i in range(len(edges) - 1)]
              + [f">={edges[-1]:g}"])
    mx = max(counts) or 1
    for lab, c in zip(labels, counts):
        if c == 0:
            continue
        bar = "#" * int(40 * c / mx)
        print(f"      {lab:>12} | {c:>7} ({100 * c / total:4.1f}%) {bar}")


def _deep(grouped: dict) -> None:
    """Dig into the dispatch amplification: distribution + histograms, to tell
    whether a high mean is systematic or outlier/idle-driven, and whether
    tokens_recv is a near-constant floor (group-load / dummy dominated) or scales
    with real load."""
    AMP_EDGES = [1.5, 2, 5, 10, 20, 50, 100, 200, 500]
    TOK_EDGES = [2, 5, 10, 50, 100, 500, 1000, 5000]
    for bt in [b for b in ("prefill", "decode", "mixed", "unknown") if b in grouped]:
        disp = grouped[bt].get("moe_dispatch_size", [])
        if not disp:
            continue
        tin = [r.get("tokens_in") for r in disp]
        trecv = [r.get("tokens_recv") for r in disp]
        amp = [(r.get("tokens_recv") / r["tokens_in"]) if r.get("tokens_in") else None
               for r in disp]
        print(f"\n[{bt}] amplification deep-dive  (n={len(disp)})")
        for name, xs, edges in (("tokens_sent", tin, TOK_EDGES),
                                ("tokens_recv", trecv, TOK_EDGES),
                                ("amplification", amp, AMP_EDGES)):
            st = _agg(xs)
            if not st:
                continue
            print(f"  {name}: min={st['min']:.2f} p50={st['p50']:.2f} "
                  f"p90={st['p90']:.2f} p99={st['p99']:.2f} max={st['max']:.2f} "
                  f"mean={st['mean']:.2f} CoV={st['cov']:.2f}")
            _histogram(xs, edges)
        # Interpretation hint.
        st_recv = _agg(trecv)
        if st_recv:
            floor = st_recv["min"]
            print(f"  -> tokens_recv CoV={st_recv['cov']:.2f}, floor(min)={floor:.0f}: "
                  f"{'≈constant → group-load/dummy-dominated' if st_recv['cov'] < 0.3 else 'variable → real-load driven'}")


def moe_stats(path: str, skip: int | None = None, include_dummy: bool = False,
              deep: bool = False) -> None:
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

    if deep:
        print("\n=== amplification deep-dive ===")
        _deep(grouped)


if __name__ == "__main__":
    argv = sys.argv[1:]
    include_dummy = "--include-dummy" in argv
    deep = "--deep" in argv
    skip = None
    for a in argv:
        if a.startswith("--skip="):
            skip = int(a.split("=", 1)[1])
    pos = [a for a in argv if not a.startswith("--")]
    moe_stats(pos[0] if pos else "./vllm_prof_out", skip=skip,
              include_dummy=include_dummy, deep=deep)
