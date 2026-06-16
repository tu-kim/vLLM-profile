# SPDX-License-Identifier: Apache-2.0
"""vllm_profiler -- a non-invasive, monkey-patch profiling addon for vLLM 0.18.0.

Targets DeepSeek V3 (MLA + fine-grained MoE) and Llama 3.1 405B (GQA + dense
MLP) running on 2x8 H100 (DeepSeek: EP16/DP8/TP2, Llama: TP8/DP2).

Quick start
-----------
Programmatic (call once, after vLLM is importable, before / right after the
engine builds the model)::

    import vllm_profiler
    vllm_profiler.enable("moe", "attn")     # or just "moe", or just "attn"

Environment-variable (no code change to the launch script)::

    VLLM_PROFILER=moe,attn  python -m vllm.entrypoints.openai.api_server ...

Each rank writes ``./vllm_prof_out/prof_rank{R}.jsonl`` (override dir with
``VLLM_PROFILER_DIR``).  Not every category has to run at once -- enable only
what you need to keep overhead low.
"""

from __future__ import annotations

import atexit
import os

from . import attn_hooks, moe_hooks, runphase, timing
from .recorder import get_recorder

_ENABLED: set[str] = set()

_ALIASES = {
    "moe": "moe",
    "expert": "moe",
    "attn": "attn",
    "attention": "attn",
}


def enable(*categories: str, comm: bool = True) -> dict[str, list[str]]:
    """Install the requested profiling categories.

    Parameters
    ----------
    categories : str
        Any of ``"moe"`` / ``"attn"`` (aliases: ``"expert"``, ``"attention"``).
        Empty -> enable everything.
    comm : bool
        For ``"attn"``: also time the TP all-reduce to isolate pure
        communication from the output-projection compute.

    Returns a dict mapping category -> list of patched targets.
    """
    cats = {_ALIASES.get(c.lower(), c.lower()) for c in categories} or {"moe", "attn"}
    patched: dict[str, list[str]] = {}
    # Always guard run phase so warmup/dummy records get tagged & filterable.
    if "phase" not in _ENABLED:
        patched["phase"] = runphase.install()
        _ENABLED.add("phase")
    if "moe" in cats and "moe" not in _ENABLED:
        patched["moe"] = moe_hooks.install()
        _ENABLED.add("moe")
    if "attn" in cats and "attn" not in _ENABLED:
        patched["attn"] = attn_hooks.install(comm=comm)
        _ENABLED.add("attn")
    if patched:
        get_recorder()  # create sink + register atexit
        atexit.register(flush)
    return patched


def flush() -> None:
    """Resolve outstanding GPU timers and flush all records to disk."""
    timing.resolve_pending()
    get_recorder().flush()


def disable() -> None:
    """Remove all installed hooks and flush."""
    flush()
    moe_hooks.uninstall()
    attn_hooks.uninstall()
    runphase.uninstall()
    _ENABLED.clear()


def _auto_enable_from_env() -> None:
    val = os.environ.get("VLLM_PROFILER")
    if not val:
        return
    cats = [c.strip() for c in val.replace(";", ",").split(",") if c.strip()]
    if cats and cats[0].lower() not in ("0", "off", "false", "none"):
        comm = os.environ.get("VLLM_PROFILER_COMM", "1") not in ("0", "off", "false")
        enable(*cats, comm=comm)


_auto_enable_from_env()
