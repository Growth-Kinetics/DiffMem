"""
Long-running worker process that consumes Hatchet workflow runs and executes
them against DiffMemory.

Companion to:
  - hatchet_workflows.py  — workflow registrations (shared by API + worker)
  - hatchet.py            — submit-side HatchetExecutor (API process)

This module is the execute-side: it attaches @workflow.task() handlers for
`diffmem-write` and `diffmem-consolidate`, then calls worker.start() which
blocks forever, pulling and executing jobs from the Hatchet engine.

Entrypoints:
  diffmem-worker               (console script, pip-installed package)
  python -m diffmem.executor.hatchet_worker   (development)

Required env vars (same as the API process):
  HATCHET_CLIENT_TOKEN  — Hatchet auth token (required)
  OPENROUTER_API_KEY    — LLM API key (required)
  DEFAULT_MODEL         — model name (required unless set in DiffMemory init)
  STORAGE_PATH          — base path for user worktrees (read by RepoManager)
  WORKTREE_ROOT         — worktree root override (optional, read by RepoManager)
  HATCHET_WORKER_SLOTS  — max concurrent jobs on this worker (default 10)

Optional env vars:
  HATCHET_NAMESPACE
  HATCHET_CLIENT_HOST_PORT
  HATCHET_CLIENT_TLS_STRATEGY
"""

# NOTE: do NOT add `from __future__ import annotations` here.
# Hatchet's @workflow.task() introspects handler annotations via
# typing.get_type_hints() at decoration time; with PEP-563 lazy annotations
# the input model class names (resolved from get_input_models()) would not be
# in module globalns at that point and resolution would fail with NameError.

import logging
import os
import threading
import time
from datetime import timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .hatchet_workflows import WriteInput, ConsolidateInput  # noqa: F401

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-worker singletons (one per worker process, not per job)
# ---------------------------------------------------------------------------

_repo_manager_singleton = None
_memory_cache: dict = {}
_memory_cache_lock = threading.Lock()


def _get_memory(user_id: str):
    """Return a cached DiffMemory instance for user_id.

    RepoManager is constructed once per worker process; DiffMemory instances
    are memoised per user_id.  This mirrors the server.py memory_instances
    pattern inside the worker process.
    """
    from ..api import DiffMemory
    from ..repo_manager import RepoManager

    global _repo_manager_singleton
    if _repo_manager_singleton is None:
        _repo_manager_singleton = RepoManager()

    with _memory_cache_lock:
        if user_id not in _memory_cache:
            path = _repo_manager_singleton.get_user_worktree(user_id)
            _memory_cache[user_id] = DiffMemory(
                str(path),
                user_id,
                os.environ["OPENROUTER_API_KEY"],
                os.getenv("DEFAULT_MODEL"),
            )
        return _memory_cache[user_id]


# ---------------------------------------------------------------------------
# Task handler attachment
# ---------------------------------------------------------------------------

def _attach_write_handler(write_workflow) -> None:
    """Attach the process_and_commit task handler to write_workflow.

    The handler is sync (not async) — simpler and matches our blocking
    git+LLM workload.

    retries=0: write commits are git-transactional and idempotent on session_id,
    but LLM output is not deterministic.  A Hatchet retry could produce a
    different commit.  We surface failures immediately; revisit in M6 ADR.
    """
    from .hatchet_workflows import get_input_models

    WriteInput, _ = get_input_models()

    @write_workflow.task(execution_timeout=timedelta(minutes=15), retries=0)
    def process_and_commit(input: WriteInput, ctx) -> dict:  # type: ignore[valid-type]
        run_id = getattr(ctx, "workflow_run_id", "unknown")
        logger.info(
            f"JOB_PICKED_UP: workflow=diffmem-write"
            f" user_id={input.user_id}"
            f" run_id={run_id}"
            f" session_id={input.session_id}"
        )
        t0 = time.monotonic()
        try:
            memory = _get_memory(input.user_id)
            memory.process_and_commit_session(
                input.memory_input,
                input.session_id,
                input.session_date,
            )
        except Exception as exc:
            logger.error(
                f"JOB_FAILED: workflow=diffmem-write"
                f" user_id={input.user_id}"
                f" run_id={run_id}"
                f" error={exc!r}"
            )
            raise

        duration_ms = int((time.monotonic() - t0) * 1000)
        result = {
            "session_id": input.session_id,
            "user_id": input.user_id,
            "status": "completed",
        }
        logger.info(
            f"JOB_COMPLETED: workflow=diffmem-write"
            f" user_id={input.user_id}"
            f" run_id={run_id}"
            f" session_id={input.session_id}"
            f" duration_ms={duration_ms}"
        )

        # Best-effort callback — never raise on failure.
        if input.callback_url:
            try:
                import httpx
                httpx.post(
                    input.callback_url,
                    json={"job_id": run_id, "status": "completed", "result": result},
                    timeout=10,
                )
            except Exception as cb_exc:
                logger.warning(
                    f"CALLBACK_FAILED: workflow=diffmem-write"
                    f" url={input.callback_url}"
                    f" error={cb_exc!r}"
                )

        return result


def _attach_consolidate_handler(consolidate_workflow) -> None:
    """Attach the consolidate task handler to consolidate_workflow."""
    from .hatchet_workflows import get_input_models

    _, ConsolidateInput = get_input_models()

    @consolidate_workflow.task(execution_timeout=timedelta(minutes=15), retries=0)
    def consolidate(input: ConsolidateInput, ctx) -> dict:  # type: ignore[valid-type]
        run_id = getattr(ctx, "workflow_run_id", "unknown")
        logger.info(
            f"JOB_PICKED_UP: workflow=diffmem-consolidate"
            f" user_id={input.user_id}"
            f" run_id={run_id}"
        )
        t0 = time.monotonic()
        try:
            memory = _get_memory(input.user_id)
            result = memory.consolidate(
                tools=input.tools,
                window=input.window,
                soft_cap_tokens=input.soft_cap_tokens,
            )
        except Exception as exc:
            logger.error(
                f"JOB_FAILED: workflow=diffmem-consolidate"
                f" user_id={input.user_id}"
                f" run_id={run_id}"
                f" error={exc!r}"
            )
            raise

        duration_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            f"JOB_COMPLETED: workflow=diffmem-consolidate"
            f" user_id={input.user_id}"
            f" run_id={run_id}"
            f" duration_ms={duration_ms}"
        )

        # Best-effort callback.
        if input.callback_url:
            try:
                import httpx
                httpx.post(
                    input.callback_url,
                    json={"job_id": run_id, "status": "completed", "result": result},
                    timeout=10,
                )
            except Exception as cb_exc:
                logger.warning(
                    f"CALLBACK_FAILED: workflow=diffmem-consolidate"
                    f" url={input.callback_url}"
                    f" error={cb_exc!r}"
                )

        return result


# ---------------------------------------------------------------------------
# Worker construction
# ---------------------------------------------------------------------------

def build_worker():
    """Build and return a configured Hatchet worker (not yet started).

    Calls register_workflows to get workflow objects, attaches task handlers,
    then constructs the hatchet.worker(...) object.
    """
    from .hatchet_workflows import build_hatchet_client, register_workflows

    hatchet = build_hatchet_client()
    write_workflow, consolidate_workflow = register_workflows(hatchet)

    _attach_write_handler(write_workflow)
    _attach_consolidate_handler(consolidate_workflow)

    slots = int(os.getenv("HATCHET_WORKER_SLOTS", "10"))
    worker = hatchet.worker(
        "diffmem-worker",
        slots=slots,
        workflows=[write_workflow, consolidate_workflow],
    )
    return worker


def main() -> None:
    """Start the DiffMem Hatchet worker.  Blocks until the process is killed."""
    logging.basicConfig(level=logging.INFO)
    logger.info("WORKER_STARTING: building worker...")
    worker = build_worker()
    logger.info("WORKER_STARTED: entering blocking event loop")
    worker.start()


if __name__ == "__main__":
    main()
