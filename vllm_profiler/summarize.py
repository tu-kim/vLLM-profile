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

# Init/warmup records sit at the front of each rank file and are hard to tag
# reliably, so we just drop a fixed number of leading lines per file and average
# the steady state. Override with --skip=N or VLLM_PROFILER_SKIP.
_DEFAULT_SKIP = int(os.environ.get("VLLM_PROFILER_SKIP", "30000"))


def _load(path: str, skip: int = 0):
    """Load all rank files. ``skip`` drops the first N non-empty lines *per file*
    (init/warmup records are written first in each rank's file)."""
    rows = []
    for fp in sorted(glob.glob(os.path.join(path, "prof_rank*.jsonl"))):
        n = 0
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n += 1
                if n <= skip:
                    continue
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


def _by_seqlen(rows, bin_width: int | None = None) -> None:
    """Bucket key timing metrics by sequence length for a length sweep.

    Buckets on ``seq_len`` (total sequence length = context + this step's
    tokens, set per forward) so it works for BOTH prefill (length = prompt len)
    and decode (length = context + 1). Falls back to num_tokens if seq_len is
    absent. ``bin_width`` groups nearby lengths into [k*w, (k+1)*w) bins -- useful
    when a benchmark sends variable lengths (e.g. random dataset)."""
    kinds = ["attn_total", "moe_total", "moe_dispatch", "moe_combine"]
    by_kind = defaultdict(list)
    for r in rows:
        if r.get("kind") in kinds and r.get("ms") is not None:
            by_kind[r["kind"]].append(r)
    key = "seq_len" if any(r.get("seq_len") is not None
                           for rs in by_kind.values() for r in rs) else "num_tokens"

    def bucket_of(r):
        v = r.get(key)
        if v is None:
            return None
        return (v // bin_width) * bin_width if bin_width else v

    label = f"{key} (bin={bin_width})" if bin_width else key
    print(f"\n(bucketing by {label})")
    for kind in kinds:
        recs = by_kind.get(kind)
        if not recs:
            continue
        buckets = defaultdict(list)
        for r in recs:
            buckets[bucket_of(r)].append(r["ms"])
        print(f"\n[by seqlen] {kind} (ms):")
        for nt in sorted(buckets, key=lambda x: (x is None, x)):
            ms = buckets[nt]
            tag = (f"{nt}-{nt + bin_width - 1}" if bin_width and nt is not None
                   else str(nt))
            print(f"    {key}={tag:<12} n={len(ms):>7} avg={_mean(ms):.4f} ms")


def summarize(path: str, phase: str | None = None, skip: int | None = None,
              by_seqlen: bool = False, bin_width: int | None = None) -> None:
    if skip is None:
        skip = _DEFAULT_SKIP
    rows = _load(path, skip=skip)
    if not rows:
        print(f"No records found under {path!r}" +
              (f" (after skipping first {skip} lines/file)" if skip else ""))
        return
    if skip:
        print(f"(skipped first {skip} lines per rank file; --skip=0 to keep)")

    # Per-forward phase (prefill / decode / mixed) breakdown + optional filter.
    bt_counts = defaultdict(int)
    for r in rows:
        bt_counts[r.get("batch_type") or "unknown"] += 1
    if any(k != "unknown" for k in bt_counts):
        print("records by phase: " + ", ".join(
            f"{k}={v}" for k, v in sorted(bt_counts.items(), key=lambda x: -x[1])))
    if phase:
        rows = [r for r in rows if r.get("batch_type") == phase]
        print(f"(filtered to batch_type == {phase!r}: {len(rows)} records)")

    if by_seqlen:
        _by_seqlen(rows, bin_width=bin_width)
        return

    if not rows:
        print("No matching records left after filtering.")
        return
    by_kind = defaultdict(list)
    for r in rows:
        by_kind[r.get("kind")].append(r)

    print(f"\n=== vllm_profiler summary: {path} "
          f"({len(rows)} records, {len({r['rank'] for r in rows})} ranks) ===")

    # ---- MoE: total time (full MoE layer, analogue of attn_total) ----
    mt = by_kind.get("moe_total", [])
    if mt:
        print("\n[MoE] total time (full MoE layer):")
        _p("avg moe_total", _mean([r.get("ms") for r in mt]), "ms")
        by_bt = defaultdict(list)
        for r in mt:
            by_bt[r.get("batch_type") or "all"].append(r.get("ms"))
        if len(by_bt) > 1 or "all" not in by_bt:
            for bt, ms in sorted(by_bt.items()):
                print(f"    {bt:<12} calls={len(ms):>6} avg={_mean(ms):.4f} ms")

    # ---- MoE: transfer method (item 2) ----
    calls = by_kind.get("moe_call", [])
    if calls:
        methods = defaultdict(int)
        for c in calls:
            methods[(c.get("pf_class"), c.get("grouping"), c.get("act_format"))] += 1
        print("\n[MoE] token transfer method (item 2):")
        for (cls, grp, fmt), n in sorted(methods.items(), key=lambda x: -x[1]):
            print(f"    {n:>6}x  class={cls}  grouping={grp}  act_format={fmt}")

        # Sequence-parallel chunk padding overhead.
        sp = [c for c in calls if c.get("is_sequence_parallel")]
        if sp:
            before = sum(c.get("tokens_before_chunk", 0) for c in sp)
            padded = sum(c.get("padded_len", 0) for c in sp)
            pad_tot = sum(c.get("pad_total", 0) for c in sp)
            pad_here = sum(c.get("pad_tokens_this_rank", 0) for c in sp)
            chunk_tok = sum(c.get("chunk_tokens", 0) for c in sp)
            tp = sp[0].get("tp_size")
            print(f"\n[MoE] sequence-parallel chunk padding (TP={tp}, {len(sp)} calls):")
            _p("total tokens before chunk", before, "tok")
            _p("total padded length", padded, "tok")
            _p("total padding (global)", pad_tot, "tok")
            if padded:
                _p("padding overhead (global)", 100.0 * pad_tot / padded, "%")
            _p("padding tokens on THIS rank", pad_here, "tok")
            if chunk_tok:
                _p("padding share of this rank's MoE work",
                   100.0 * pad_here / chunk_tok, "%")

    # ---- MoE: dispatch/combine transfer size (item 1) ----
    disp = by_kind.get("moe_dispatch_size", [])
    comb = by_kind.get("moe_combine_size", [])
    if disp:
        print("\n[MoE] dispatch transfer size (item 1):")
        _p("avg bytes sent (prepare in)", _mean([d.get("bytes_in") for d in disp]), "B")
        _p("avg bytes recv (local experts)", _mean([d.get("bytes_recv") for d in disp]), "B")
        _p("avg per-token bytes", _mean([d.get("per_token_bytes") for d in disp]), "B/tok")
        # Batched token counts -- how many tokens went into one dispatch.
        print("    -- batched token counts --")
        _p("avg local tokens sent", _mean([d.get("tokens_in") for d in disp]), "tok")
        _p("avg routing slots sent (tok*topk)",
           _mean([d.get("routing_slots_sent") for d in disp]), "slots")
        recv2d = [d.get("tokens_recv") for d in disp if d.get("tokens_recv") is not None]
        if recv2d:
            _p("avg tokens recv (Standard 2D)", _mean(recv2d), "tok")
        recv3d = [d for d in disp if d.get("layout") == "batched_experts_3d"]
        if recv3d:
            _p("avg local experts (E)", _mean([d.get("n_local_experts") for d in recv3d]), "")
            _p("avg max tokens/expert (pad cap)",
               _mean([d.get("max_tokens_per_expert") for d in recv3d]), "tok")
            _p("avg recv padded (E*max)",
               _mean([d.get("tokens_recv_padded") for d in recv3d]), "tok")
    dtok = by_kind.get("moe_dispatch_tokens", [])
    if dtok:
        # Real (unpadded) per-local-expert batched counts.
        reals, maxs = [], []
        for d in dtok:
            ent = d.get("expert_num_tokens") or []
            if ent:
                reals.append(sum(ent))
                maxs.append(max(ent))
        if reals:
            _p("avg real tokens recv (unpadded)", _mean(reals), "tok")
            _p("avg busiest local expert", _mean(maxs), "tok")
    if comb:
        print("\n[MoE] combine transfer size (item 1):")
        _p("avg bytes in (expert out)", _mean([c.get("bytes_in") for c in comb]), "B")
        _p("avg bytes out (combined)", _mean([c.get("bytes_out") for c in comb]), "B")
    for k, name in (("moe_dispatch", "dispatch"), ("moe_combine", "combine")):
        if by_kind.get(k):
            _p(f"avg {name} time", _mean([r.get("ms") for r in by_kind[k]]), "ms")

    # ---- MoE: per-batch load imbalance (item 3) ----
    loads = by_kind.get("moe_expert_load", [])
    batched = [l for l in loads if l.get("cov") is not None]
    if batched:
        covs = [l["cov"] for l in batched]
        moms = [l["max_over_mean"] for l in batched]
        print("\n[MoE] per-batch load imbalance (item 3):")
        _p("batches measured", len(batched), "")
        _p("avg expert CoV", _mean(covs))
        _p("worst-batch expert CoV", max(covs))
        _p("avg expert max/mean", _mean(moms))
        _p("worst-batch expert max/mean", max(moms))
        rcov = [l["rank_cov"] for l in batched if l.get("rank_cov") is not None]
        rmom = [l["rank_max_over_mean"] for l in batched
                if l.get("rank_max_over_mean") is not None]
        if rcov:
            ranks_n = batched[0].get("n_ep_ranks")
            print(f"    -- EP-rank level (across {ranks_n} ranks; slowest gates the step) --")
            _p("avg rank CoV", _mean(rcov))
            _p("worst-batch rank CoV", max(rcov))
            _p("avg rank max/mean", _mean(rmom))
            _p("worst-batch rank max/mean", max(rmom))

    # ---- MoE: expert load balance aggregate (item 3) ----
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
    argv = list(sys.argv[1:])
    phase = None
    skip = None  # None -> default fixed skip
    by_seqlen = "--by-seqlen" in argv
    bin_width = None
    for a in argv:
        if a.startswith("--phase="):
            phase = a.split("=", 1)[1]
        elif a.startswith("--skip="):
            skip = int(a.split("=", 1)[1])
        elif a.startswith("--bin="):
            bin_width = int(a.split("=", 1)[1])
    pos = [a for a in argv if not a.startswith("--")]
    summarize(pos[0] if pos else "./vllm_prof_out", phase=phase, skip=skip,
              by_seqlen=by_seqlen, bin_width=bin_width)
