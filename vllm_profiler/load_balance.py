# SPDX-License-Identifier: Apache-2.0
"""Expert load-balance analysis.

Aggregates the per-call routed-expert histograms (`moe_expert_load.counts`) into
the long-run expert load distribution -- so the per-batch sparsity that makes
decode noisy is averaged out -- and reports:

  * EP-rank load (experts grouped into rank blocks): which rank is the straggler
    that gates every MoE step, and the over-perfect-balance overhead.
  * Expert-level imbalance (CoV, max/mean) + hottest / coldest experts.
  * Split by batch_type (prefill / decode) and optionally per layer.

Streams the files (does not hold all rows in memory), so it scales to the large
decode logs.

    python -m vllm_profiler.load_balance ./vllm_prof_out
    python -m vllm_profiler.load_balance ./merged --by-layer --top=15
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

from .summarize import _DEFAULT_SKIP, _find_files


def _imbalance(loads: list[float]) -> dict:
    n = len(loads)
    tot = sum(loads)
    if n == 0 or tot == 0:
        return {"n": n, "total": tot}
    mean = tot / n
    var = sum((x - mean) ** 2 for x in loads) / n
    return {
        "n": n, "total": tot, "mean": mean,
        "max": max(loads), "min": min(loads),
        "max_over_mean": max(loads) / mean,
        "min_over_mean": min(loads) / mean,
        "cov": (var ** 0.5) / mean,
    }


def _report(tag: str, counts: list[float], experts_per_rank: int | None,
            top: int) -> None:
    ex = _imbalance(counts)
    if not ex.get("total"):
        return
    print(f"\n[{tag}]  total_routings={int(ex['total'])}  experts={ex['n']}")
    print(f"  expert-level: CoV={ex['cov']:.3f}  max/mean={ex['max_over_mean']:.2f}"
          f"  min/mean={ex['min_over_mean']:.3f}  (mean={ex['mean']:.1f}/expert)")
    ranked = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)
    hot = ", ".join(f"E{i}={int(counts[i])}({counts[i]/ex['mean']:.1f}x)"
                    for i in ranked[:top])
    print(f"  hot  experts (top {top}): {hot}")
    n_zero = sum(1 for c in counts if c == 0)
    print(f"  cold experts: {n_zero} experts got 0 routings "
          f"({100 * n_zero / len(counts):.0f}%)")

    if experts_per_rank and experts_per_rank > 0 and len(counts) % experts_per_rank == 0:
        ranks = [sum(counts[i:i + experts_per_rank])
                 for i in range(0, len(counts), experts_per_rank)]
        rk = _imbalance(ranks)
        hottest = max(range(len(ranks)), key=lambda i: ranks[i])
        coldest = min(range(len(ranks)), key=lambda i: ranks[i])
        print(f"  -- EP-rank level ({len(ranks)} ranks, {experts_per_rank} experts/rank) --")
        print(f"  rank load: max/mean={rk['max_over_mean']:.2f}  CoV={rk['cov']:.3f}"
              f"  hottest=rank{hottest}({ranks[hottest]/rk['mean']:.2f}x)"
              f"  coldest=rank{coldest}({ranks[coldest]/rk['mean']:.2f}x)")
        overhead = (rk["max_over_mean"] - 1) * 100
        print(f"  -> straggler overhead vs perfect balance: +{overhead:.0f}% "
              f"(slowest rank does {rk['max_over_mean']:.2f}x the average work)")


def load_balance(path: str, skip: int | None = None, by_layer: bool = False,
                 top: int = 10, include_dummy: bool = False) -> None:
    if skip is None:
        skip = _DEFAULT_SKIP
    # phase -> summed counts (list grows to n_experts); and per (phase, layer).
    agg: dict = defaultdict(lambda: None)
    agg_layer: dict = defaultdict(lambda: None)
    epr: dict = {}          # phase -> experts_per_rank
    ncalls: dict = defaultdict(int)

    def add(acc_key, acc_dict, counts):
        cur = acc_dict[acc_key]
        if cur is None:
            acc_dict[acc_key] = list(counts)
        else:
            for i, c in enumerate(counts):
                if i < len(cur):
                    cur[i] += c

    files = _find_files(path)
    if not files:
        print(f"No prof_rank*.jsonl under {path!r}")
        return
    for fp in files:
        n = 0
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            n += 1
            if n <= skip:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") != "moe_expert_load":
                continue
            if r.get("dummy") and not include_dummy:
                continue
            counts = r.get("counts")
            if not counts:
                continue
            bt = r.get("batch_type") or "unknown"
            add(bt, agg, counts)
            ncalls[bt] += 1
            if r.get("experts_per_rank"):
                epr[bt] = r["experts_per_rank"]
            if by_layer:
                add((bt, r.get("moe_layer")), agg_layer, counts)

    if not any(v for v in agg.values()):
        print(f"No moe_expert_load records found (after skip={skip}). "
              f"Was the run profiled with 'moe' enabled? Try --skip=0.")
        return

    print(f"=== Expert load balance (skip={skip}/file, "
          f"dummy {'kept' if include_dummy else 'dropped'}) ===")
    for bt in ("prefill", "decode", "mixed", "unknown"):
        if agg.get(bt):
            print(f"\n#### {bt}  ({ncalls[bt]} calls aggregated) ####")
            _report(bt, agg[bt], epr.get(bt), top)
            if by_layer:
                layers = sorted(l for (p, l) in agg_layer if p == bt)
                for layer in layers:
                    _report(f"{bt} L{layer}", agg_layer[(bt, layer)],
                            epr.get(bt), min(top, 5))


if __name__ == "__main__":
    argv = sys.argv[1:]
    by_layer = "--by-layer" in argv
    include_dummy = "--include-dummy" in argv
    skip = None
    top = 10
    for a in argv:
        if a.startswith("--skip="):
            skip = int(a.split("=", 1)[1])
        elif a.startswith("--top="):
            top = int(a.split("=", 1)[1])
    pos = [a for a in argv if not a.startswith("--")]
    load_balance(pos[0] if pos else "./vllm_prof_out", skip=skip,
                 by_layer=by_layer, top=top, include_dummy=include_dummy)
