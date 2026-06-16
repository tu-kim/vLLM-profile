# SPDX-License-Identifier: Apache-2.0
"""Rank-aware JSONL recorder for the vLLM internal profiler addon.

One output file per (rank) process.  Records are buffered in memory and flushed
periodically and at interpreter exit so the hot path never blocks on disk I/O.

Design goals (kept intentionally simple):
  * No dependency on vLLM internals -- only ``os`` / ``json`` / optional torch.
  * Safe to import on a CUDA-less box (the H100 run does the real work).
  * Every record is a flat JSON dict with a ``kind`` discriminator so the
    different profiling categories can share a single sink and be split later.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from typing import Any

try:  # torch is present on the GPU servers, absent on the dev box.
    import torch
except Exception:  # pragma: no cover - import guard
    torch = None  # type: ignore


_FLUSH_EVERY = int(os.environ.get("VLLM_PROFILER_FLUSH_EVERY", "1024"))

# --- run-phase tracking -------------------------------------------------------
# vLLM runs many dummy/warmup/cudagraph-capture/DP-lockstep forward passes during
# initialization (all via GPUModelRunner._dummy_run). Those fire our hooks too.
# We track a depth counter while inside a dummy run so records can be tagged
# `dummy: True` and filtered out, leaving only real inference data.
_dummy_depth = 0
_dummy_lock = threading.Lock()


def enter_dummy() -> None:
    global _dummy_depth
    with _dummy_lock:
        _dummy_depth += 1


def exit_dummy() -> None:
    global _dummy_depth
    with _dummy_lock:
        if _dummy_depth > 0:
            _dummy_depth -= 1


def in_dummy() -> bool:
    return _dummy_depth > 0


# Current batch phase ("prefill" / "decode" / "mixed"), set per forward pass by
# the execute_model wrapper (runphase.py). None outside a real forward.
_batch_type: str | None = None


def set_batch_type(bt: str | None) -> None:
    global _batch_type
    _batch_type = bt


def current_batch_type() -> str | None:
    return _batch_type


# Everything before the first *real* execute_model is initialization: weight
# load, memory profiling, kernel warmup (DeepGEMM / FlashInfer), cudagraph
# capture. Those fire our hooks from many different code paths, so rather than
# chase each one we treat the whole pre-serving window as dummy.
_serving_started = False


def mark_serving_started() -> None:
    global _serving_started
    _serving_started = True


def stamp_tags(d: dict) -> dict:
    """Add run-phase tags (dummy, batch_type) captured at *this* moment.

    Used at capture time by record()/Region/_DeferredBuffer so deferred records
    carry the phase that was active when the work actually ran, not at flush.
    setdefault keeps any tag the caller already stamped.
    """
    if _dummy_depth > 0 or not _serving_started:
        d.setdefault("dummy", True)
    if _batch_type is not None:
        d.setdefault("batch_type", _batch_type)
    return d


def _detect_rank() -> int:
    """Best-effort global rank detection without forcing torch.distributed init."""
    if torch is not None:
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return torch.distributed.get_rank()
        except Exception:
            pass
    for key in ("RANK", "VLLM_DP_RANK", "LOCAL_RANK"):
        val = os.environ.get(key)
        if val is not None and val.lstrip("-").isdigit():
            return int(val)
    return 0


class Recorder:
    """Thread-safe buffered JSONL writer, one file per process/rank."""

    def __init__(self, out_dir: str | None = None) -> None:
        self.out_dir = out_dir or os.environ.get(
            "VLLM_PROFILER_DIR", "./vllm_prof_out"
        )
        self.rank = _detect_rank()
        os.makedirs(self.out_dir, exist_ok=True)
        self.path = os.path.join(self.out_dir, f"prof_rank{self.rank}.jsonl")
        self._buf: list[str] = []
        self._lock = threading.Lock()
        self._closed = False
        # Truncate any stale file from a previous run for this rank.
        with open(self.path, "w"):
            pass
        atexit.register(self.close)

    def record(self, kind: str, **fields: Any) -> None:
        rec = {"kind": kind, "rank": self.rank, "t_wall": time.time(), **fields}
        # Tag with run phase (dummy / prefill-decode) unless the caller already
        # stamped it at capture time (see Region / _DeferredBuffer).
        stamp_tags(rec)
        line = json.dumps(rec, default=_jsonable)
        with self._lock:
            if self._closed:
                return
            self._buf.append(line)
            if len(self._buf) >= _FLUSH_EVERY:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buf:
            return
        with open(self.path, "a") as f:
            f.write("\n".join(self._buf))
            f.write("\n")
        self._buf.clear()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._flush_locked()
            self._closed = True


def _jsonable(obj: Any) -> Any:
    """Fallback serializer for the odd torch scalar / tensor that slips through."""
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    try:
        return obj.item()  # numpy / torch 0-dim scalars
    except Exception:
        return str(obj)


# Process-wide singleton -- created lazily on first use.
_RECORDER: Recorder | None = None
_REC_LOCK = threading.Lock()


def get_recorder() -> Recorder:
    global _RECORDER
    if _RECORDER is None:
        with _REC_LOCK:
            if _RECORDER is None:
                _RECORDER = Recorder()
    return _RECORDER
