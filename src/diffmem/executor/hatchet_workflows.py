"""
Hatchet workflow registrations for the DiffMem Hatchet backend.

Imported by BOTH:
  - the API process (submit-side, M3): to register workflows and submit runs.
  - the worker process (execute-side, M4): to attach @workflow.task() handlers.

Task handlers are NOT registered here — they live in hatchet_worker.py (M4).
This keeps the submit-side importable without any worker-specific imports.

Usage (API process):
    from diffmem.executor.hatchet_workflows import (
        build_hatchet_client, register_workflows, WriteInput, ConsolidateInput,
    )
    hatchet = build_hatchet_client()
    write_wf, consolidate_wf = register_workflows(hatchet)
    ref = write_wf.run(WriteInput(...), wait_for_result=False)

Usage (worker, M4):
    from diffmem.executor.hatchet_workflows import (
        build_hatchet_client, register_workflows,
    )
    hatchet = build_hatchet_client()
    write_wf, consolidate_wf = register_workflows(hatchet)

    @write_wf.task()
    def run_write(input: WriteInput, ctx) -> dict:
        ...  # reconstitute DiffMemory and run the write
"""

from __future__ import annotations

import os
from typing import List, Optional


def _get_sdk():
    """Return the hatchet_sdk components needed for workflow definitions."""
    try:
        from hatchet_sdk import (  # type: ignore[import-untyped]
            Hatchet,
            ConcurrencyExpression,
            ConcurrencyLimitStrategy,
        )
        from pydantic import BaseModel
        return Hatchet, ConcurrencyExpression, ConcurrencyLimitStrategy, BaseModel
    except ImportError as e:
        raise ImportError(
            "Hatchet SDK is not installed. Run: pip install diffmem[hatchet] "
            "(or `poetry install --extras hatchet`)"
        ) from e


# ---------------------------------------------------------------------------
# Input models
# These are defined lazily (inside functions) because pydantic's BaseModel
# requires pydantic to be importable.  Callers should import WriteInput and
# ConsolidateInput via get_input_models() rather than importing them directly
# at module level.
# ---------------------------------------------------------------------------

def get_input_models():
    """Return (WriteInput, ConsolidateInput) Pydantic model classes.

    Called by HatchetExecutor.submit_*() and by the worker in M4.
    Returns new class objects on each call (Python does not deduplicate class
    objects across calls), which is fine for correctness; the only overhead
    is a pydantic class creation which happens once per process import.

    To avoid repeated class creation, call this once and cache the result.
    """
    _, _, _, BaseModel = _get_sdk()

    class WriteInput(BaseModel):
        user_id: str
        memory_input: str
        session_id: str
        session_date: Optional[str] = None
        callback_url: Optional[str] = None

    class ConsolidateInput(BaseModel):
        user_id: str
        memory_input: Optional[str] = None
        session_id: Optional[str] = None
        session_date: Optional[str] = None
        tools: Optional[List[str]] = None
        window: int = 3
        soft_cap_tokens: int = 32000
        callback_url: Optional[str] = None

    return WriteInput, ConsolidateInput


def build_hatchet_client():
    """Construct and return a Hatchet client.

    Reads env vars automatically (HATCHET_CLIENT_TOKEN, HATCHET_CLIENT_HOST_PORT,
    HATCHET_CLIENT_TLS_STRATEGY, HATCHET_NAMESPACE).

    Raises:
        ImportError: if hatchet-sdk is not installed.
        RuntimeError: if HATCHET_CLIENT_TOKEN is not set.
    """
    Hatchet, _, _, _ = _get_sdk()
    token = os.getenv("HATCHET_CLIENT_TOKEN")
    if not token:
        raise RuntimeError(
            "HATCHET_CLIENT_TOKEN is not set. "
            "Set HATCHET_CLIENT_TOKEN env var; for Hatchet Cloud, copy from "
            "https://cloud.onhatchet.run dashboard"
        )
    return Hatchet()


def register_workflows(hatchet) -> tuple:
    """Register and return (write_workflow, consolidate_workflow).

    Both the API process and the worker call this.  The worker ALSO attaches
    @workflow.task() handlers after calling this function — see hatchet_worker.py.

    Args:
        hatchet: A Hatchet client instance (from build_hatchet_client()).

    Returns:
        tuple[Workflow, Workflow]: (write_workflow, consolidate_workflow)
    """
    _, ConcurrencyExpression, ConcurrencyLimitStrategy, _ = _get_sdk()
    WriteInput, ConsolidateInput = get_input_models()

    write_workflow = hatchet.workflow(
        name="diffmem-write",
        input_validator=WriteInput,
        concurrency=ConcurrencyExpression(
            expression="input.user_id",
            max_runs=1,
            limit_strategy=ConcurrencyLimitStrategy.GROUP_ROUND_ROBIN,
        ),
    )

    consolidate_workflow = hatchet.workflow(
        name="diffmem-consolidate",
        input_validator=ConsolidateInput,
        concurrency=ConcurrencyExpression(
            expression="input.user_id",
            max_runs=1,
            limit_strategy=ConcurrencyLimitStrategy.GROUP_ROUND_ROBIN,
        ),
    )

    # Task handlers (@write_workflow.task(), @consolidate_workflow.task()) are
    # added in hatchet_worker.py (M4).  Not defining them here keeps the
    # submit-side importable and runnable without any worker-specific code.

    return write_workflow, consolidate_workflow
