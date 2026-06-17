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
# Dummy-run depth. GPUModelRunner._dummy_run is, by definition, ALWAYS a dummy
# forward (warmup / cudagraph capture / DP idle-rank lockstep) -- real inference
# never goes through it. So tagging records produced inside it as dummy=True is
# unambiguous and safe (this is the part that was correct before; the buggy bit
# was the separate "serving not started" gate, which is NOT reintroduced).
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


# Representative total sequence length (context + this step's tokens) of the
# active request, set per forward by the execute_model wrapper. With
# max-num-seqs 1 this is exact and is the length axis for both prefill (= prompt
# length) and decode (= context length + 1).
_seq_len: int | None = None


def set_seq_len(n: int | None) -> None:
    global _seq_len
    _seq_len = n


def stamp_tags(d: dict) -> dict:
    """Stamp dummy flag + batch phase + seq length, captured now (deferred
    records emit later, so the value must be read at capture time)."""
    if _dummy_depth > 0:
        d.setdefault("dummy", True)
    if _batch_type is not None:
        d.setdefault("batch_type", _batch_type)
    if _seq_len is not None:
        d.setdefault("seq_len", _seq_len)
    return d


def _detect_rank() -> int:
    """Best-effort global rank detection. Called lazily at first record(), by
    which time a forward is running and torch.distributed is initialized."""
    if torch is not None:
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return torch.distributed.get_rank()
        except Exception:
            pass
    # vLLM's own global rank (set even when the default pg differs).
    try:
        from vllm.distributed.parallel_state import get_world_group

        return get_world_group().rank_in_group
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
        # IMPORTANT: do NOT detect the rank or open the file here. This object is
        # created at enable()/plugin-load time, *before* torch.distributed is
        # initialized in the worker (and mp/Ray launchers don't set RANK env), so
        # detecting now would label every worker rank 0 -> all write one file.
        # Defer rank detection + file open to the first record(), by which time a
        # forward is running and torch.distributed is up -> correct global rank.
        self.rank: int | None = None
        self.path: str | None = None
        self._buf: list[str] = []
        self._lock = threading.Lock()
        self._closed = False
        atexit.register(self.close)

    def _ensure_open(self) -> None:
        if self.path is not None:
            return
        self.rank = _detect_rank()
        os.makedirs(self.out_dir, exist_ok=True)
        # Hostname suffix so primary + headless node logs never collide even if
        # rank detection still falls back to local rank. Transparent to the
        # summarizer (it globs prof_rank*.jsonl, keys on the "rank" field).
        import socket
        host = socket.gethostname().split(".")[0]
        self.path = os.path.join(self.out_dir, f"prof_rank{self.rank}_{host}.jsonl")
        with open(self.path, "w"):  # truncate any stale file
            pass

    def record(self, kind: str, **fields: Any) -> None:
        with self._lock:
            if self._closed:
                return
            self._ensure_open()
            rec = {"kind": kind, "rank": self.rank, "t_wall": time.time(), **fields}
            # Tag with batch phase (prefill/decode) unless the caller already
            # stamped it at capture time (see Region / _DeferredBuffer).
            stamp_tags(rec)
            self._buf.append(json.dumps(rec, default=_jsonable))
            if len(self._buf) >= _FLUSH_EVERY:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buf or self.path is None:
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
