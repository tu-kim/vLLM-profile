# SPDX-License-Identifier: Apache-2.0
"""Offline summarizer for vllm_profiler JSONL output.

Reads ``prof_rank*.jsonl`` from a directory and prints the three MoE metrics and
the attention compute/comm/memory breakdown.  Pure stdlib -- run it anywhere,
including the dev box.

    python -m vllm_profiler.summarize ./vllm_prof_out
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict


def _load(path: str):
    rows = []
    for fp in sorted(glob.glob(os.path.join(path, "prof_rank*.jsonl"))):
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return rows


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _p(label, val, unit=""):
    print(f"    {label:<34} {val:>12.3f} {unit}")


def summarize(path: str) -> None:
    rows = _load(path)
    if not rows:
        print(f"No records found under {path!r}")
        return
    by_kind = defaultdict(list)
    for r in rows:
        by_kind[r.get("kind")].append(r)

    print(f"\n=== vllm_profiler summary: {path} "
          f"({len(rows)} records, {len({r['rank'] for r in rows})} ranks) ===")

    # ---- MoE: transfer method (item 2) ----
    calls = by_kind.get("moe_call", [])
    if calls:
        methods = defaultdict(int)
        for c in calls:
            methods[(c.get("pf_class"), c.get("grouping"), c.get("act_format"))] += 1
        print("\n[MoE] token transfer method (item 2):")
        for (cls, grp, fmt), n in sorted(methods.items(), key=lambda x: -x[1]):
            print(f"    {n:>6}x  class={cls}  grouping={grp}  act_format={fmt}")

    # ---- MoE: dispatch/combine transfer size (item 1) ----
    disp = by_kind.get("moe_dispatch_size", [])
    comb = by_kind.get("moe_combine_size", [])
    if disp:
        print("\n[MoE] dispatch transfer size (item 1):")
        _p("avg bytes sent (prepare in)", _mean([d.get("bytes_in") for d in disp]), "B")
        _p("avg bytes recv (local experts)", _mean([d.get("bytes_recv") for d in disp]), "B")
        _p("avg per-token bytes", _mean([d.get("per_token_bytes") for d in disp]), "B/tok")
    if comb:
        print("\n[MoE] combine transfer size (item 1):")
        _p("avg bytes in (expert out)", _mean([c.get("bytes_in") for c in comb]), "B")
        _p("avg bytes out (combined)", _mean([c.get("bytes_out") for c in comb]), "B")
    for k, name in (("moe_dispatch", "dispatch"), ("moe_combine", "combine")):
        if by_kind.get(k):
            _p(f"avg {name} time", _mean([r.get("ms") for r in by_kind[k]]), "ms")

    # ---- MoE: expert load balance (item 3) ----
    loads = by_kind.get("moe_expert_load", [])
    if loads:
        per_layer = defaultdict(lambda: None)
        for l in loads:
            counts = l.get("counts") or []
            agg = per_layer[l.get("moe_layer")]
            if agg is None:
                per_layer[l.get("moe_layer")] = list(counts)
            else:
                for i, v in enumerate(counts):
                    if i < len(agg):
                        agg[i] += v
        print("\n[MoE] expert load balance (item 3), per layer:")
        for layer in sorted(per_layer):
            c = per_layer[layer]
            if not c:
                continue
            tot = sum(c) or 1
            mx, mn = max(c), min(c)
            # Coefficient of variation = imbalance indicator.
            mean = tot / len(c)
            var = sum((x - mean) ** 2 for x in c) / len(c)
            cov = (var ** 0.5) / mean if mean else 0.0
            print(f"    layer {layer:>3}: experts={len(c)} tot_routes={tot} "
                  f"max={mx} min={mn} max/mean={mx/mean:.2f} CoV={cov:.3f}")

    # ---- Attention: total time GQA vs MLA (item 1) ----
    tot = by_kind.get("attn_total", [])
    if tot:
        by_cls = defaultdict(list)
        for t in tot:
            by_cls[t.get("attn_cls")].append(t.get("ms"))
        print("\n[Attn] total time by structure (item 1):")
        for cls, ms in by_cls.items():
            print(f"    {cls:<36} calls={len(ms):>6} avg={_mean(ms):.4f} ms")

    # ---- Attention: compute/comm breakdown (item 2) ----
    steps = by_kind.get("attn_step", [])
    if steps:
        by_role = defaultdict(lambda: defaultdict(list))
        for s in steps:
            by_role[s.get("attn_cls")][s.get("role")].append(s.get("ms"))
        print("\n[Attn] per-role time breakdown (item 2):")
        for cls, roles in by_role.items():
            print(f"  {cls}:")
            for role, ms in sorted(roles.items()):
                _p(f"{role}", _mean(ms), "ms")
    ar = by_kind.get("tp_all_reduce", [])
    if ar:
        print("\n[Attn] pure TP all-reduce (communication):")
        _p("avg all-reduce time", _mean([r.get("ms") for r in ar]), "ms")
        _p("avg all-reduce bytes", _mean([r.get("bytes") for r in ar]), "B")

    # ---- Attention: memory load/store (item 3) ----
    io = by_kind.get("attn_step_io", [])
    if steps or io:
        print("\n[Attn] memory load/store (item 3), effective bandwidth by role:")
        bytes_by_role = defaultdict(float)
        for s in steps:
            bytes_by_role[(s.get("attn_cls"), s.get("role"))] += (s.get("bytes_in") or 0)
        for i in io:
            bytes_by_role[(i.get("attn_cls"), i.get("role"))] += (i.get("bytes_out") or 0)
        time_by_role = defaultdict(float)
        for s in steps:
            time_by_role[(s.get("attn_cls"), s.get("role"))] += (s.get("ms") or 0)
        for key in sorted(bytes_by_role):
            b, t = bytes_by_role[key], time_by_role.get(key, 0)
            gbps = (b / (t / 1e3)) / 1e9 if t else 0.0
            print(f"    {key[0]:<34} {str(key[1]):<14} "
                  f"{b/1e6:>10.1f} MB  {gbps:>8.1f} GB/s")


if __name__ == "__main__":
    summarize(sys.argv[1] if len(sys.argv) > 1 else "./vllm_prof_out")
