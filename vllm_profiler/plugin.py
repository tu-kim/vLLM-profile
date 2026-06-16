# SPDX-License-Identifier: Apache-2.0
"""vLLM general-plugin entry point.

vLLM calls ``load_general_plugins()`` once **inside every worker process** (see
``vllm/v1/worker/worker_base.py``) and the engine core, *before* the model is
built.  Registering :func:`register` under the ``vllm.general_plugins`` entry
point group therefore makes the profiler activate identically whether the run is
launched via ``vllm serve``, the OpenAI api_server, or the in-process ``LLM``
API -- including multi-process EP/DP/TP fan-out where the model actually runs in
spawned subprocesses.

Activation stays gated on the ``VLLM_PROFILER`` env var, so merely having the
package installed never forces profiling on.

Entry point (declared in pyproject.toml)::

    [project.entry-points."vllm.general_plugins"]
    vllm_profiler = "vllm_profiler.plugin:register"
"""

from __future__ import annotations

import os


def register() -> None:
    """Called once per worker/engine process by vLLM's plugin loader.

    Idempotent: :func:`vllm_profiler.enable` guards against double-patching, and
    vLLM may invoke plugins more than once per process.
    """
    val = os.environ.get("VLLM_PROFILER")
    if not val or val.lower() in ("0", "off", "false", "none"):
        return
    cats = [c.strip() for c in val.replace(";", ",").split(",") if c.strip()]
    comm = os.environ.get("VLLM_PROFILER_COMM", "1") not in ("0", "off", "false")

    from . import enable

    enable(*cats, comm=comm)
