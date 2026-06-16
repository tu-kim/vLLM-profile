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

from .recorder import enter_dummy, exit_dummy

_originals: dict[str, Any] = {}


def install() -> list[str]:
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception:
        return []
    if "dummy_run" in _originals:
        return []
    orig = GPUModelRunner._dummy_run

    def _dummy_run(self, *args, **kwargs):
        enter_dummy()
        try:
            return orig(self, *args, **kwargs)
        finally:
            exit_dummy()

    _originals["dummy_run"] = orig
    GPUModelRunner._dummy_run = _dummy_run
    return ["GPUModelRunner._dummy_run"]


def uninstall() -> None:
    if "dummy_run" not in _originals:
        return
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        GPUModelRunner._dummy_run = _originals.pop("dummy_run")
    except Exception:
        _originals.pop("dummy_run", None)
