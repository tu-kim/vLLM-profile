# SPDX-License-Identifier: Apache-2.0
"""Deferred GPU timing primitives.

GPU work is asynchronous, so reading ``event.elapsed_time`` right after a region
would force a ``cuda.synchronize()`` on every call and wreck the very timings we
are trying to measure.  Instead we *record* a pair of CUDA events per region and
resolve them lazily: a single ``synchronize()`` at flush time converts a whole
batch of event pairs into milliseconds.

On a CUDA-less box (the dev machine) we transparently fall back to
``time.perf_counter`` so the addon stays importable and unit-testable.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import torch

    _CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _CUDA = False

from .recorder import get_recorder


@dataclass
class _Pending:
    kind: str
    fields: dict[str, Any]
    start: Any  # cuda.Event or float (perf_counter)
    end: Any


_PENDING: list[_Pending] = []
# Resolve in batches so the pending list (and its CUDA events) stay bounded.
# Higher -> fewer cuda.synchronize() calls -> lower measurement overhead, at the
# cost of holding more CUDA Event objects and fresher-data latency. Lower it only
# if you need timing (ms) records to land sooner.
_RESOLVE_EVERY = int(os.environ.get("VLLM_PROFILER_RESOLVE_EVERY", "2048"))

# Extra resolvers (e.g. MoE histogram buffers) that also need the post-sync
# window to move GPU tensors to host without paying a per-call synchronize.
_RESOLVERS: list[Any] = []


def register_resolver(fn: Any) -> None:
    """Register a callable invoked (after sync) on every :func:`resolve_pending`."""
    if fn not in _RESOLVERS:
        _RESOLVERS.append(fn)


class Region:
    """Context manager timing a single GPU (or CPU-fallback) region.

    Usage:
        with Region("attn.qkv_proj", layer="model.layers.0.self_attn",
                     bytes_in=..., bytes_out=...):
            qkv = self.qkv_proj(x)

    The elapsed time is not known until :func:`resolve_pending` runs, so extra
    metadata (shapes, byte counts) is captured eagerly and stored alongside the
    event pair.
    """

    __slots__ = ("kind", "fields", "_start", "_end")

    def __init__(self, kind: str, **fields: Any) -> None:
        self.kind = kind
        self.fields = fields

    def __enter__(self) -> "Region":
        if _CUDA:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record()
        else:
            self._start = time.perf_counter()
            self._end = None
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if _CUDA:
            self._end.record()
            _PENDING.append(_Pending(self.kind, self.fields, self._start, self._end))
        else:
            elapsed_ms = (time.perf_counter() - self._start) * 1e3
            get_recorder().record(self.kind, ms=elapsed_ms, **self.fields)
        if len(_PENDING) >= _RESOLVE_EVERY:
            resolve_pending()


def resolve_pending() -> None:
    """Synchronize once and emit a timing record for every pending region."""
    if not _PENDING:
        return
    if _CUDA:
        torch.cuda.synchronize()
        rec = get_recorder()
        for p in _PENDING:
            ms = p.start.elapsed_time(p.end)  # milliseconds
            rec.record(p.kind, ms=ms, **p.fields)
    _PENDING.clear()
    for fn in _RESOLVERS:
        try:
            fn()
        except Exception:
            pass


def dtype_size(t: Any) -> int:
    """Bytes-per-element for a tensor, robust to fp8 / quantized dtypes."""
    if torch is None or t is None:
        return 0
    try:
        return t.element_size()
    except Exception:
        return 0


def nbytes(t: Any) -> int:
    """Total bytes backing a tensor (numel * element_size)."""
    if torch is None or t is None or not hasattr(t, "numel"):
        return 0
    try:
        return int(t.numel()) * dtype_size(t)
    except Exception:
        return 0
