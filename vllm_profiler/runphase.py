# SPDX-License-Identifier: Apache-2.0
"""Run-phase guard: mark records produced during dummy/warmup forward passes.

vLLM fires many forward passes during initialization that are NOT real
inference: memory profiling (`profile_run`), CUDA-graph capture
(`capture_model`), backend warm-up, and DP idle-rank lockstep
(`_dummy_run(1)`). They all funnel through `GPUModelRunner._dummy_run`, whose
own docstring says it runs a dummy pass "to warm up/profile run or capture the
CUDA graph". Real requests go through `execute_model` and never touch it.

By wrapping `_dummy_run` to bump a depth counter, every record our MoE/attn
hooks emit while inside it gets tagged `dummy: True` (see recorder.in_dummy()),
so the summarizer can drop warmup noise and keep only real inference data.
"""

from __future__ import annotations

from typing import Any

from .recorder import (
    enter_dummy,
    exit_dummy,
    mark_serving_started,
    set_batch_type,
)

_originals: dict[str, Any] = {}


def _is_warmup_batch(scheduler_output: Any) -> bool:
    """True for the Triton/sampler warmup (gpu/warmup.py), which runs a synthetic
    prefill+decode through the *real* execute_model path using request ids
    prefixed ``_warmup_``. Real requests never use that prefix, so this cleanly
    separates the last init step from genuine serving."""
    try:
        nst = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
        return bool(nst) and any(str(k).startswith("_warmup_") for k in nst)
    except Exception:
        return False


def _classify(runner: Any, scheduler_output: Any) -> str | None:
    """prefill / decode / mixed for a real forward.

    Uses per-request scheduled token counts. A request scheduled for more than
    ``uniform_decode_query_len`` tokens (1, or 1+num_spec with spec decoding) is
    a prefill; otherwise it's a decode. Chunked-prefill batches that mix both
    are labelled "mixed".
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
            # The Triton/sampler warmup (gpu/warmup.py) runs a synthetic
            # prefill+decode through this very path; treat it as dummy so it
            # neither pollutes data nor prematurely flips "serving started".
            if _is_warmup_batch(scheduler_output):
                enter_dummy()
                try:
                    return orig_exec(self, scheduler_output, *args, **kwargs)
                finally:
                    exit_dummy()
            bt = _classify(self, scheduler_output)
            if bt is not None:
                # First real forward with scheduled work -> serving has begun;
                # everything captured before this was init/warmup.
                mark_serving_started()
            set_batch_type(bt)
            try:
                return orig_exec(self, scheduler_output, *args, **kwargs)
            finally:
                set_batch_type(None)

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
