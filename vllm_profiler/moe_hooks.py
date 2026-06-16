# SPDX-License-Identifier: Apache-2.0
"""MoE dispatch/combine + expert-routing profiling hooks.

Covers the three MoE items requested for DeepSeek V3 (EP16 / DP8 / TP2):

  1. Per-token Expert *Dispatch / Combine* transfer sizes
       -> wrap ``FusedMoEKernelModularImpl._prepare`` (dispatch / all-to-all in)
          and ``_finalize`` (combine / all-to-all out); record tensor bytes and
          per-token payload (hidden_dim * itemsize).
  2. Token transfer *method* (one batched buffer vs per-token)
       -> the prepare/finalize class + its ``activation_format``
          (Standard = contiguous batched buffer, BatchedExperts = padded
          [E, max_tokens, H] per-expert batch) tells us how tokens are grouped.
  3. Expert *load balance* (per-token routed-expert distribution)
       -> histogram ``topk_ids`` over the global expert space, accumulated on
          GPU and moved to host lazily at flush time.

Everything is attached by monkey-patching the *class* methods, so the model
source is never touched and no model instance is required at enable() time.
"""

from __future__ import annotations

import threading
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from . import timing
from .recorder import get_recorder
from .timing import Region, nbytes, register_resolver

# --- per-MoE-call context (thread-local so _prepare/_finalize can find it) ----
_ctx = threading.local()

# Stable small integer id per distinct kernel-impl object (== per MoE layer).
_layer_ids: dict[int, int] = {}
_layer_lock = threading.Lock()

_call_seq = 0  # global monotonic MoE-call counter (proxy for decode step order)

_originals: dict[str, Any] = {}


def _layer_idx(obj: Any) -> int:
    key = id(obj)
    idx = _layer_ids.get(key)
    if idx is None:
        with _layer_lock:
            idx = _layer_ids.get(key)
            if idx is None:
                idx = len(_layer_ids)
                _layer_ids[key] = idx
    return idx


def _arg(args: tuple, kwargs: dict, pos: int, name: str, default=None):
    if name in kwargs:
        return kwargs[name]
    if len(args) > pos:
        return args[pos]
    return default


# --- transfer method classification -------------------------------------------
def _transfer_method(pf: Any) -> dict[str, Any]:
    """Describe how a prepare/finalize object moves tokens between EP ranks."""
    cls = type(pf).__name__
    fmt = None
    try:
        fmt = pf.activation_format.name  # FusedMoEActivationFormat enum
    except Exception:
        try:
            fmt = str(pf.activation_format())
        except Exception:
            fmt = None
    # Heuristic label keyed on the well-known vLLM EP backends.
    low = cls.lower()
    fmt_low = (fmt or "").lower()
    if "deepepht" in low or "deepep_ht" in low or "highthroughput" in low:
        grouping = "batched_grouped_by_rank"      # DeepEP high-throughput
    elif "deepepll" in low or "deepep_ll" in low or "lowlatency" in low:
        grouping = "per_token_low_latency"        # DeepEP low-latency
    elif "pplx" in low:
        grouping = "batched_grouped_by_rank"
    elif "batched" in fmt_low:
        grouping = "batched_per_expert_padded"
    else:
        grouping = "contiguous_batched"
    return {"pf_class": cls, "act_format": fmt, "grouping": grouping}


# --- expert load-balance histogram buffer -------------------------------------
class _HistBuffer:
    """Accumulates topk_ids histograms on GPU; drains to host at flush."""

    def __init__(self) -> None:
        self._pending: list[tuple[dict, Any]] = []
        register_resolver(self.drain)

    def add(self, meta: dict, hist_gpu: Any) -> None:
        self._pending.append((meta, hist_gpu))

    def drain(self) -> None:
        if not self._pending:
            return
        rec = get_recorder()
        pending, self._pending = self._pending, []
        for meta, hist in pending:
            counts = hist.tolist() if hasattr(hist, "tolist") else list(hist)
            rec.record("moe_expert_load", counts=counts, **meta)


_hist: _HistBuffer | None = None


def _record_load_balance(topk_ids: Any, global_num_experts: int, meta: dict) -> None:
    if torch is None or topk_ids is None or _hist is None:
        return
    try:
        flat = topk_ids.flatten()
        ne = global_num_experts if global_num_experts and global_num_experts > 0 else None
        if ne is None:
            ne = int(flat.max().item()) + 1
        hist = torch.bincount(flat.to(torch.int64), minlength=ne)
        _hist.add(meta, hist)
    except Exception:
        pass


# --- patched methods ----------------------------------------------------------
def _wrap_apply(orig):
    def apply(self, *args, **kwargs):
        global _call_seq
        topk_ids = _arg(args, kwargs, 3, "topk_ids")
        topk_weights = _arg(args, kwargs, 4, "topk_weights")
        hidden_states = _arg(args, kwargs, 0, "hidden_states")
        gne = _arg(args, kwargs, 6, "global_num_experts", -1)

        seq = _call_seq
        _call_seq += 1
        layer = _layer_idx(self)

        num_tokens = int(topk_ids.shape[0]) if topk_ids is not None else -1
        top_k = int(topk_ids.shape[1]) if topk_ids is not None and topk_ids.dim() > 1 else -1
        hidden_dim = int(hidden_states.shape[-1]) if hidden_states is not None else -1

        meta = {
            "moe_layer": layer,
            "call_seq": seq,
            "num_tokens": num_tokens,
            "top_k": top_k,
            "hidden_dim": hidden_dim,
            "global_num_experts": int(gne),
        }
        # Item 2: how are tokens transferred? (class + activation format)
        pf = getattr(self, "prepare_finalize", None)
        method = _transfer_method(pf) if pf is not None else {}
        get_recorder().record("moe_call", **meta, **method)

        # Item 3: expert load balance.
        _record_load_balance(topk_ids, int(gne), {"moe_layer": layer, "call_seq": seq})

        # Expose context so the _prepare / _finalize wrappers can tag their
        # transfer-size records to this exact call.
        prev = getattr(_ctx, "cur", None)
        _ctx.cur = meta
        try:
            return orig(self, *args, **kwargs)
        finally:
            _ctx.cur = prev

    return apply


def _wrap_prepare(orig):
    def _prepare(self, *args, **kwargs):
        hidden_states = _arg(args, kwargs, 0, "hidden_states")
        meta = dict(getattr(_ctx, "cur", None) or {})
        bytes_in = nbytes(hidden_states)  # tokens handed to dispatch (local)
        with Region("moe_dispatch", phase="prepare", bytes_in=bytes_in, **meta):
            out = orig(self, *args, **kwargs)
        # out == (a1q, a1q_scale, expert_tokens_meta, topk_ids, topk_weights)
        try:
            a1q = out[0]
            get_recorder().record(
                "moe_dispatch_size",
                phase="prepare",
                bytes_in=bytes_in,                 # sent out from this rank
                bytes_recv=nbytes(a1q),            # tokens received for local experts
                recv_shape=list(a1q.shape) if a1q is not None else None,
                per_token_bytes=(bytes_in // meta["num_tokens"])
                if meta.get("num_tokens", 0) > 0 else None,
                **meta,
            )
        except Exception:
            pass
        return out

    return _prepare


def _wrap_finalize(orig):
    def _finalize(self, *args, **kwargs):
        fused_out = _arg(args, kwargs, 1, "fused_out")
        meta = dict(getattr(_ctx, "cur", None) or {})
        with Region("moe_combine", phase="finalize", bytes_in=nbytes(fused_out), **meta):
            out = orig(self, *args, **kwargs)
        try:
            get_recorder().record(
                "moe_combine_size",
                phase="finalize",
                bytes_in=nbytes(fused_out),         # expert outputs to be combined
                bytes_out=nbytes(out),              # combined result returned
                **meta,
            )
        except Exception:
            pass
        return out

    return _finalize


def install() -> list[str]:
    """Monkey-patch the modular MoE kernel.  Returns list of patched targets."""
    global _hist
    if torch is None:
        return []
    from vllm.model_executor.layers.fused_moe import modular_kernel as mk

    _hist = _HistBuffer()
    patched = []
    cls = mk.FusedMoEKernelModularImpl
    for name, wrapper in (
        ("apply", _wrap_apply),
        ("_prepare", _wrap_prepare),
        ("_finalize", _wrap_finalize),
    ):
        if name in _originals:
            continue
        orig = getattr(cls, name)
        _originals[name] = orig
        setattr(cls, name, wrapper(orig))
        patched.append(f"FusedMoEKernelModularImpl.{name}")
    return patched


def uninstall() -> None:
    if torch is None or not _originals:
        return
    from vllm.model_executor.layers.fused_moe import modular_kernel as mk

    cls = mk.FusedMoEKernelModularImpl
    for name, orig in _originals.items():
        setattr(cls, name, orig)
    _originals.clear()
