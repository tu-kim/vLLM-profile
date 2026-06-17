# SPDX-License-Identifier: Apache-2.0
"""Per-forward prefill/decode tagging.

Wraps ``GPUModelRunner.execute_model`` to classify each real forward as
prefill / decode / mixed (from the per-request scheduled token counts) and stamp
it on every record via recorder.set_batch_type.

Init/warmup separation is NOT done here -- it is handled bluntly by summarize's
fixed leading-line skip (init records sit at the front of each rank file).
"""

from __future__ import annotations

from typing import Any

from .recorder import enter_dummy, exit_dummy, set_batch_type, set_seq_len

_originals: dict[str, Any] = {}


def _seq_len(scheduler_output: Any) -> int | None:
    """Max total sequence length (computed context + this step's tokens) across
    requests. With max-num-seqs 1 this is the single sequence's length, the
    unified length axis for prefill (prompt len) and decode (context+1)."""
    try:
        nst = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
        if not nst:
            return None
        computed: dict[str, int] = {}
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached is not None:
            ids = getattr(cached, "req_ids", None)
            nct = getattr(cached, "num_computed_tokens", None)
            if ids and nct:
                computed.update(zip(ids, nct))
        for r in getattr(scheduler_output, "scheduled_new_reqs", []) or []:
            rid = getattr(r, "req_id", None)
            if rid is not None:
                computed[rid] = getattr(r, "num_computed_tokens", 0)
        best = max(computed.get(rid, 0) + sched for rid, sched in nst.items())
        return int(best) or None
    except Exception:
        return None


def _classify(runner: Any, scheduler_output: Any) -> str | None:
    """prefill / decode / mixed for a forward, from per-request token counts.

    A request scheduled for more than ``uniform_decode_query_len`` tokens
    (1, or 1+num_spec with spec decoding) is a prefill; otherwise a decode.
    Without chunked prefill a batch is cleanly all-prefill or all-decode.
    """
    try:
        nst = getattr(scheduler_output, "num_scheduled_tokens", None)
        if not nst:
            return None
        q = getattr(runner, "uniform_decode_query_len", 1) or 1
        vals = list(nst.values())
        n_prefill = sum(1 for v in vals if v > q)
        if n_prefill == 0:
            return "decode"
        if n_prefill == len(vals):
            return "prefill"
        return "mixed"
    except Exception:
        return None


def install() -> list[str]:
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception:
        return []
    patched = []

    # _dummy_run is always a dummy forward (warmup / cudagraph capture / DP
    # idle-rank lockstep). Tag everything it produces as dummy=True. No serving
    # gate -- real inference never goes through _dummy_run.
    if "dummy_run" not in _originals:
        orig_dummy = GPUModelRunner._dummy_run

        def _dummy_run(self, *args, **kwargs):
            enter_dummy()
            try:
                return orig_dummy(self, *args, **kwargs)
            finally:
                exit_dummy()

        _originals["dummy_run"] = orig_dummy
        GPUModelRunner._dummy_run = _dummy_run
        patched.append("GPUModelRunner._dummy_run")

    if "execute_model" not in _originals:
        orig_exec = GPUModelRunner.execute_model

        def execute_model(self, scheduler_output, *args, **kwargs):
            set_batch_type(_classify(self, scheduler_output))
            set_seq_len(_seq_len(scheduler_output))
            try:
                return orig_exec(self, scheduler_output, *args, **kwargs)
            finally:
                set_batch_type(None)
                set_seq_len(None)

        _originals["execute_model"] = orig_exec
        GPUModelRunner.execute_model = execute_model
        patched.append("GPUModelRunner.execute_model")

    return patched


def uninstall() -> None:
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        if "dummy_run" in _originals:
            GPUModelRunner._dummy_run = _originals.pop("dummy_run")
        if "execute_model" in _originals:
            GPUModelRunner.execute_model = _originals.pop("execute_model")
    except Exception:
        _originals.pop("dummy_run", None)
        _originals.pop("execute_model", None)
