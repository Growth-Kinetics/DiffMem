"""
HatchetExecutor — submit-side Hatchet backend for DiffMem (M3).

This module covers the submit side only: enqueue runs, query status, and
optionally wait for results.  The actual workflow execution is done by the
worker process added in M4 (hatchet_worker.py).

Architecture:
  API process (this file)           Worker process (M4)
  ────────────────────────          ─────────────────────────────────
  HatchetExecutor.submit_*()        hatchet_worker.py registers
    → workflow.run(input, ...)        @workflow.task() handlers and
    → returns WorkflowRunRef          calls worker.start()
  HatchetExecutor.get_job()
    → hatchet.runs.get_status()
  HatchetExecutor.wait_for()
    → ref.result() (blocking, thread)

Per-user serialisation is guaranteed by the ConcurrencyExpression on each
workflow (expression="input.user_id", max_runs=1) — Hatchet queues the second
run until the first completes, even across worker restarts.

Install the optional extras to use this backend:
    pip install diffmem[hatchet]
    poetry install --extras hatchet
"""

from __future__ import annotations

import concurrent.futures as _cf
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional

from .base import (
    ConsolidatePayload,
    JobHandle,
    JobResult,
    JobStatus,
    TaskExecutor,
    WritePayload,
)
from .jobstore import JobStore

logger = logging.getLogger(__name__)

# Map Hatchet V1TaskStatus string names → our JobStatus literals.
# We normalise to lowercase ".name" or ".value" strings for robustness across
# SDK versions.
_STATUS_MAP: dict[str, JobStatus] = {
    "pending": "queued",
    "queued": "queued",
    "running": "running",
    "completed": "completed",
    "succeeded": "completed",
    "failed": "failed",
    "cancelled": "failed",
    "cancelling": "failed",
}

# Single-step workflow → task handler function name.
# Must match the @workflow.task()-decorated function names in
# `hatchet_worker.py` (source of truth).  Hatchet wraps task return values
# under the handler function name; we unwrap for single-step workflows so
# `JobResult.result` matches `InlineExecutor`'s contract (handler return value
# directly, no wrapper layer).
_TASK_NAMES: dict[str, str] = {
    "diffmem-write": "process_and_commit",
    "diffmem-consolidate": "consolidate",
}


def _unwrap_task_output(workflow_name: str, output: object) -> object:
    """Unwrap a Hatchet single-step workflow output dict.

    Hatchet returns workflow outputs as ``{<task_name>: <handler_return>}``.
    For DiffMem's single-step workflows we strip that wrapper so callers see
    the handler's direct return value (matches InlineExecutor's contract).

    Defensive: returns ``output`` unchanged on shape mismatch and logs a
    WARNING so future Hatchet SDK shape changes are visible.
    """
    expected = _TASK_NAMES.get(workflow_name)
    if not isinstance(output, dict) or not expected:
        return output
    if list(output.keys()) == [expected] and isinstance(output[expected], dict):
        return output[expected]
    logger.warning(
        f"HATCHET_OUTPUT_SHAPE_UNEXPECTED: workflow={workflow_name} "
        f"expected_key={expected} actual_keys={list(output.keys())[:5]}"
    )
    return output


def _map_hatchet_status(raw_status: object) -> JobStatus:
    """Map a Hatchet V1TaskStatus enum value to our JobStatus literal."""
    if hasattr(raw_status, "name"):
        key = raw_status.name.lower()  # type: ignore[union-attr]
    elif hasattr(raw_status, "value"):
        key = str(raw_status.value).lower()  # type: ignore[union-attr]
    else:
        key = str(raw_status).lower()
    return _STATUS_MAP.get(key, "queued")


class HatchetExecutor(TaskExecutor):
    """TaskExecutor backed by Hatchet for durable, per-user-serialised execution.

    Required env vars:
      HATCHET_CLIENT_TOKEN  — auth token.
      HATCHET_NAMESPACE     — namespace scoping (default: "diffmem").

    Optional env vars:
      HATCHET_CLIENT_HOST_PORT    — for self-hosted Hatchet (e.g. host:7077).
      HATCHET_CLIENT_TLS_STRATEGY — tls | mtls | none (default: tls for Cloud).
    """

    def __init__(self, pool: ThreadPoolExecutor) -> None:
        # Validate the SDK is installed — lazy import to give a clean error.
        try:
            import hatchet_sdk as _sdk  # noqa: F401  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "Hatchet SDK is not installed. Run: pip install diffmem[hatchet] "
                "(or `poetry install --extras hatchet`)"
            ) from e

        self._pool = pool
        self._jobstore = JobStore()

        # Thread-safe cache of (WorkflowRunRef, workflow_name) tuples keyed by
        # workflow_run_id.  We track the workflow name so wait_for / get_job
        # know which task name to unwrap in `_unwrap_task_output`.
        self._refs: dict[str, tuple[object, str]] = {}
        self._refs_lock = threading.Lock()

        # Build client + register workflows (also validates HATCHET_CLIENT_TOKEN).
        from .hatchet_workflows import build_hatchet_client, register_workflows, get_input_models
        self._hatchet = build_hatchet_client()
        self._write_workflow, self._consolidate_workflow = register_workflows(self._hatchet)
        self._WriteInput, self._ConsolidateInput = get_input_models()

        namespace = os.getenv("HATCHET_NAMESPACE", "diffmem")
        server = os.getenv("HATCHET_CLIENT_HOST_PORT", "cloud.onhatchet.run")
        logger.info(f"HATCHET_INITIALIZED: namespace={namespace} server={server}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _store_ref(self, job_id: str, ref: object, workflow_name: str) -> None:
        with self._refs_lock:
            self._refs[job_id] = (ref, workflow_name)

    def _get_ref_entry(self, job_id: str) -> tuple[object, str] | None:
        """Return the (ref, workflow_name) tuple for a job, or None."""
        with self._refs_lock:
            return self._refs.get(job_id)

    def _get_workflow_name(self, job_id: str) -> str | None:
        entry = self._get_ref_entry(job_id)
        return entry[1] if entry is not None else None

    def _submit_run(
        self,
        workflow: object,
        input_obj: object,
        user_id: str,
        workflow_name: str,
    ) -> str:
        """Call workflow.run() and return the workflow_run_id (our job_id)."""
        ref = workflow.run(  # type: ignore[union-attr]
            input_obj,
            wait_for_result=False,
            additional_metadata={"source": "diffmem-api", "user_id": user_id},
        )
        run_id: str = ref.workflow_run_id
        self._store_ref(run_id, ref, workflow_name)
        return run_id

    def _enrich_from_details(
        self,
        job_id: str,
        workflow_name: Optional[str] = None,
    ) -> None:
        """Fetch run details from Hatchet and update jobstore with started_at,
        completed_at (from finished_at), and unwrapped result.

        Best-effort: Hatchet API errors are logged at WARNING but do not raise.
        This is how we populate ``JobResult.started_at`` in hatchet mode
        — the submit-side process has no direct signal of worker pickup time,
        so we query the engine on demand (only when transitioning to terminal
        state, to keep REST calls cheap).
        """
        try:
            details = self._hatchet.runs.get(job_id)
        except Exception as exc:
            logger.warning(f"HATCHET_DETAILS_ERROR: job_id={job_id} error={exc!r}")
            return

        run = getattr(details, "run", None)
        if run is None:
            return

        # Timestamp enrichment.
        updates: dict[str, object] = {}
        started_at = getattr(run, "started_at", None)
        finished_at = getattr(run, "finished_at", None)
        if started_at is not None:
            updates["started_at"] = started_at
        if finished_at is not None:
            updates["completed_at"] = finished_at

        if updates:
            current = self._jobstore.get(job_id)
            if current is not None:
                # update_status preserves status; we re-pass current.status and
                # only override the timestamp fields.
                self._jobstore.update_status(job_id, current.status, **updates)

        # Result enrichment (only if jobstore doesn't already have one).
        # The primary unwrap happens in wait_for() right after ref.result()
        # returns; this is the fallback for jobs reaching terminal state via
        # get_job() without anyone calling wait_for() (e.g. submitted by
        # another process, status-polled to completion here).
        current = self._jobstore.get(job_id)
        if current is None or current.result is None:
            output = getattr(run, "output", None)
            if isinstance(output, dict):
                wf_name = workflow_name or self._get_workflow_name(job_id)
                if wf_name is not None:
                    unwrapped = _unwrap_task_output(wf_name, output)
                    if isinstance(unwrapped, dict):
                        self._jobstore.set_result(job_id, unwrapped)

    # ------------------------------------------------------------------
    # TaskExecutor interface
    # ------------------------------------------------------------------

    def submit_write(
        self,
        user_id: str,
        work: Optional[object],
        payload: Optional[WritePayload] = None,
        callback_url: Optional[str] = None,
    ) -> JobHandle:
        """Enqueue a write job via Hatchet.  ``payload`` is required; ``work`` is ignored."""
        if payload is None:
            raise ValueError(
                "HatchetExecutor.submit_write requires a WritePayload. "
                "Pass payload= when constructing the job."
            )
        input_obj = self._WriteInput(
            **asdict(payload),
            callback_url=callback_url,
        )
        now = datetime.now(timezone.utc)
        run_id = self._submit_run(self._write_workflow, input_obj, user_id, "diffmem-write")
        self._jobstore.put(JobResult(job_id=run_id, status="queued", submitted_at=now))
        logger.info(f"JOB_SUBMITTED: kind=write job_id={run_id} user_id={user_id}")
        return JobHandle(job_id=run_id, status="queued", submitted_at=now)

    def submit_consolidate(
        self,
        user_id: str,
        work: Optional[object],
        payload: Optional[ConsolidatePayload] = None,
        callback_url: Optional[str] = None,
    ) -> JobHandle:
        """Enqueue a consolidate job via Hatchet.  ``payload`` is required; ``work`` is ignored."""
        if payload is None:
            raise ValueError(
                "HatchetExecutor.submit_consolidate requires a ConsolidatePayload. "
                "Pass payload= when constructing the job."
            )
        input_obj = self._ConsolidateInput(
            **asdict(payload),
            callback_url=callback_url,
        )
        now = datetime.now(timezone.utc)
        run_id = self._submit_run(
            self._consolidate_workflow, input_obj, user_id, "diffmem-consolidate"
        )
        self._jobstore.put(JobResult(job_id=run_id, status="queued", submitted_at=now))
        logger.info(f"JOB_SUBMITTED: kind=consolidate job_id={run_id} user_id={user_id}")
        return JobHandle(job_id=run_id, status="queued", submitted_at=now)

    def get_job(self, job_id: str) -> JobResult | None:
        """Return current JobResult, refreshing status from Hatchet if non-terminal.

        On transition to a terminal state (completed/failed), enriches the
        JobResult with started_at, completed_at, and unwrapped result via
        ``_enrich_from_details``.  Cheap-path: terminal cached entries skip the
        REST call entirely.
        """
        cached = self._jobstore.get(job_id)
        if cached is not None and cached.status in {"completed", "failed"}:
            # Already at terminal state — no need to poll Hatchet.
            return cached

        try:
            raw_status = self._hatchet.runs.get_status(job_id)
            new_status = _map_hatchet_status(raw_status)
        except Exception as exc:
            logger.warning(
                f"HATCHET_STATUS_ERROR: job_id={job_id} error={exc!r}; "
                "returning cached status"
            )
            return cached  # may be None if this process never submitted it

        now = datetime.now(timezone.utc)
        if cached is None:
            cached = JobResult(job_id=job_id, status=new_status, submitted_at=now)
            self._jobstore.put(cached)
        else:
            self._jobstore.update_status(job_id, new_status)

        # On transition to terminal, enrich with started_at/completed_at/result.
        # Best-effort — errors are swallowed by _enrich_from_details.
        if new_status in {"completed", "failed"}:
            self._enrich_from_details(job_id)

        return self._jobstore.get(job_id)

    def wait_for(self, job_id: str, timeout: float | None = None) -> JobResult:
        """Block until terminal state or timeout.

        Strategy:
        - If a WorkflowRunRef is cached (this process submitted the job), call
          ref.result() in a dedicated thread with a timeout wrapper — this is
          Hatchet's efficient server-push mechanism.
        - Otherwise, poll get_job() every 0.5 s.

        Raises TimeoutError if ``timeout`` seconds elapse first.
        """
        ref_entry = self._get_ref_entry(job_id)
        if ref_entry is not None:
            ref, wf_name = ref_entry
            deadline = None if timeout is None else time.monotonic() + timeout

            def _call_result() -> object:
                return ref.result()  # type: ignore[union-attr]

            with _cf.ThreadPoolExecutor(max_workers=1) as _tmp:
                future = _tmp.submit(_call_result)
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                try:
                    result_dict = future.result(timeout=remaining)
                except _cf.TimeoutError as exc:
                    raise TimeoutError(
                        f"wait_for timed out after {timeout}s for job_id={job_id}"
                    ) from exc

            # Unwrap Hatchet's single-step task-name wrapper to match
            # InlineExecutor's contract (JobResult.result is the handler's
            # direct return value, not a {task_name: ...} dict).
            if isinstance(result_dict, dict):
                result_dict = _unwrap_task_output(wf_name, result_dict)

            completed_at = datetime.now(timezone.utc)
            self._jobstore.update_status(job_id, "completed", completed_at=completed_at)
            if isinstance(result_dict, dict):
                self._jobstore.set_result(job_id, result_dict)

            # Enrich with started_at from Hatchet's REST API (the ref.result()
            # path doesn't give it to us natively).  Best-effort.
            self._enrich_from_details(job_id, workflow_name=wf_name)

            final = self._jobstore.get(job_id)
            if final is not None:
                return final
            # Fallback (shouldn't happen — jobstore.put was called in submit_*).
            return JobResult(
                job_id=job_id,
                status="completed",
                submitted_at=completed_at,
                result=result_dict if isinstance(result_dict, dict) else None,
                completed_at=completed_at,
            )

        # Polling fallback (job submitted by a different process).
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            result = self.get_job(job_id)
            if result is not None and result.status in {"completed", "failed"}:
                return result
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"wait_for timed out after {timeout}s for job_id={job_id}"
                )
            time.sleep(0.5)

    @property
    def supports_async_api(self) -> bool:
        # Hatchet's native execution model is async — the worker handles work
        # out-of-band.  Default endpoint behaviour with EXECUTOR=hatchet is
        # async (return job_id immediately), opposite of InlineExecutor.
        return True
