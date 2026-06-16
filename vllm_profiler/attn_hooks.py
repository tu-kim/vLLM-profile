# SPDX-License-Identifier: Apache-2.0
"""Attention profiling hooks for Llama (GQA) and DeepSeek V3 (MLA).

Covers the three attention items requested:

  1. GQA vs MLA structural time difference
       -> total attention time recorded per call, tagged with the concrete
          class (``LlamaAttention`` vs ``MultiHeadLatentAttentionWrapper``) so
          the two structures can be compared directly.
  2. Compute vs communication breakdown
       -> per-submodule timing (qkv/q-down/kv-down projections, RoPE, core
          attention, output projection).  The output projection is a
          RowParallelLinear and carries the TP all-reduce; an optional hook on
          ``tensor_model_parallel_all_reduce`` isolates the pure comm time.
  3. Memory load/store overhead
       -> every submodule region records input/output bytes, so effective
          bandwidth = bytes / time can be derived offline.

Implementation trick (kept simple, version-robust): we never reimplement any
``forward``.  We patch the class ``forward`` to (a) lazily wrap the relevant
child modules' ``.forward`` with timing on first call, and (b) wrap the whole
call in a "total" timing region.
"""

from __future__ import annotations

import threading
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from .recorder import get_recorder
from .timing import Region, nbytes

_originals: dict[str, Any] = {}
_INSTRUMENTED = "_vllm_prof_done"

# Thread-local "current attention region" so the all-reduce hook can attribute
# a comm call to the attention layer that triggered it.
_ctx = threading.local()

_layer_ids: dict[int, int] = {}
_call_seq = 0

# Submodules to instrument, keyed by the role used in the comp/comm breakdown.
# Only those present on a given instance are wrapped.
_ROLES = {
    # projections (column/replicated -> compute only, no all-reduce in forward)
    "qkv_proj": "proj",
    "fused_qkv_a_proj": "proj_down",   # MLA: fused q+kv latent down-projection
    "q_proj": "proj",
    "q_a_proj": "proj_down",
    "q_b_proj": "proj_up",
    "kv_a_proj_with_mqa": "proj_down",
    "kv_b_proj": "proj_up",
    "kv_a_layernorm": "norm",
    "q_a_layernorm": "norm",
    "rotary_emb": "rope",
    # core attention (KV-cache load/store happens here)
    "attn": "core_attn",
    "mla_attn": "core_attn",
    # output projection: RowParallelLinear -> compute + TP all-reduce (comm)
    "o_proj": "out_proj_comm",
}


def _layer_idx(obj: Any) -> int:
    key = id(obj)
    idx = _layer_ids.get(key)
    if idx is None:
        idx = len(_layer_ids)
        _layer_ids[key] = idx
    return idx


def _rep_tensor(x: Any):
    """Pick a representative tensor from an arg/return (handles tuples)."""
    if torch is not None and isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)):
        for e in x:
            if torch is not None and isinstance(e, torch.Tensor):
                return e
    return None


def _wrap_submodule(mod: Any, role: str, sub_name: str, layer: int, attn_cls: str):
    """Replace ``mod.forward`` with a timed version (idempotent per instance)."""
    if getattr(mod, _INSTRUMENTED, False):
        return
    orig = mod.forward

    def timed_forward(*args, **kwargs):
        in_t = _rep_tensor(args[0]) if args else None
        meta = {
            "attn_cls": attn_cls,
            "layer": layer,
            "role": role,
            "submodule": sub_name,
        }
        prev = getattr(_ctx, "region", None)
        _ctx.region = meta  # let all-reduce hook attribute itself here
        try:
            with Region("attn_step", bytes_in=nbytes(in_t),
                        in_shape=list(in_t.shape) if in_t is not None else None,
                        **meta) as _r:
                out = orig(*args, **kwargs)
        finally:
            _ctx.region = prev
        out_t = _rep_tensor(out)
        if out_t is not None:
            # Second tiny record carries output bytes for bandwidth analysis.
            get_recorder().record("attn_step_io", bytes_out=nbytes(out_t),
                                   out_shape=list(out_t.shape), **meta)
        return out

    mod.forward = timed_forward
    setattr(mod, _INSTRUMENTED, True)


def _instrument(instance: Any, attn_cls: str, layer: int) -> None:
    for name, role in _ROLES.items():
        sub = getattr(instance, name, None)
        if sub is not None and hasattr(sub, "forward"):
            _wrap_submodule(sub, role, name, layer, attn_cls)


def _wrap_forward(orig, attn_cls: str):
    def forward(self, *args, **kwargs):
        global _call_seq
        layer = _layer_idx(self)
        if not getattr(self, _INSTRUMENTED, False):
            _instrument(self, attn_cls, layer)
            setattr(self, _INSTRUMENTED, True)
        seq = _call_seq
        _call_seq += 1
        hs = None
        for a in args:
            t = _rep_tensor(a)
            if t is not None and t.dim() >= 2:
                hs = t
                break
        with Region("attn_total", attn_cls=attn_cls, layer=layer, call_seq=seq,
                    num_tokens=int(hs.shape[0]) if hs is not None else -1):
            return orig(self, *args, **kwargs)

    return forward


# --- optional pure-communication timing ---------------------------------------
def _wrap_all_reduce(orig):
    def all_reduce(*args, **kwargs):
        t = _rep_tensor(args[0]) if args else None
        region = getattr(_ctx, "region", None)
        tag = dict(region) if region else {"role": "other_comm"}
        with Region("tp_all_reduce", bytes=nbytes(t), **tag):
            return orig(*args, **kwargs)

    return all_reduce


def install(comm: bool = True) -> list[str]:
    """Patch Llama + DeepSeek attention forwards. ``comm`` adds all-reduce timing."""
    if torch is None:
        return []
    patched: list[str] = []

    # Llama GQA
    try:
        from vllm.model_executor.models.llama import LlamaAttention
        if "llama" not in _originals:
            _originals["llama"] = LlamaAttention.forward
            LlamaAttention.forward = _wrap_forward(LlamaAttention.forward, "LlamaAttention")
            patched.append("LlamaAttention.forward")
    except Exception:
        pass

    # DeepSeek MLA (wrapper holds the submodules)
    try:
        from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper as W
        if "mla" not in _originals:
            _originals["mla"] = W.forward
            W.forward = _wrap_forward(W.forward, "MultiHeadLatentAttentionWrapper")
            patched.append("MultiHeadLatentAttentionWrapper.forward")
    except Exception:
        pass

    if comm:
        # RowParallelLinear (o_proj) calls the name bound into linear.py's
        # namespace at import time, so we must patch it *there* to intercept it.
        try:
            import vllm.model_executor.layers.linear as lin
            if "all_reduce" not in _originals:
                _originals["all_reduce"] = lin.tensor_model_parallel_all_reduce
                lin.tensor_model_parallel_all_reduce = _wrap_all_reduce(
                    lin.tensor_model_parallel_all_reduce
                )
                patched.append("linear.tensor_model_parallel_all_reduce")
        except Exception:
            pass

    return patched


def uninstall() -> None:
    if torch is None:
        return
    try:
        from vllm.model_executor.models.llama import LlamaAttention
        if "llama" in _originals:
            LlamaAttention.forward = _originals.pop("llama")
    except Exception:
        pass
    try:
        from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper as W
        if "mla" in _originals:
            W.forward = _originals.pop("mla")
    except Exception:
        pass
    try:
        import vllm.model_executor.layers.linear as lin
        if "all_reduce" in _originals:
            lin.tensor_model_parallel_all_reduce = _originals.pop("all_reduce")
    except Exception:
        pass
