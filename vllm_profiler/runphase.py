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

from .recorder import set_batch_type

_originals: dict[str, Any] = {}


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
    if "execute_model" in _originals:
        return []
    orig_exec = GPUModelRunner.execute_model

    def execute_model(self, scheduler_output, *args, **kwargs):
        set_batch_type(_classify(self, scheduler_output))
        try:
            return orig_exec(self, scheduler_output, *args, **kwargs)
        finally:
            set_batch_type(None)

    _originals["execute_model"] = orig_exec
    GPUModelRunner.execute_model = execute_model
    return ["GPUModelRunner.execute_model"]


def uninstall() -> None:
    if "execute_model" not in _originals:
        return
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        GPUModelRunner.execute_model = _originals.pop("execute_model")
    except Exception:
        _originals.pop("execute_model", None)
