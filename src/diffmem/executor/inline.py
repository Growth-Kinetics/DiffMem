"""
InlineExecutor: runs jobs in the existing ThreadPoolExecutor with per-user
serialization via per-user threading.Lock objects.

This is the default (and only) executor in M1/M2. It requires zero extra
infrastructure — just wrap the existing _writer_pool.
"""

from __future__ import annotations

import logging
import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Callable
from uuid import uuid4

try:
    import httpx as _http_lib
    _USE_HTTPX = True
except ImportError:  # pragma: no cover
    import requests as _http_lib  # type: ignore[no-redef]
    _USE_HTTPX = False

from .base import JobHandle, JobResult, JobStatus, TaskExecutor
from .jobstore import JobStore

logger = logging.getLogger(__name__)


class InlineExecutor(TaskExecutor):
    """Runs jobs inline in a ThreadPoolExecutor with per-user write locks.

    Per-user lock guarantee: two jobs for the same user_id are never
    executed concurrently — the second waits until the first finishes.
    Jobs for *different* users run in parallel (bounded by pool size).

    Known limitation: _user_locks dict grows monotonically (one entry per
    unique user_id seen). Fine for typical deployments; M3/M4 may add LRU
    eviction if thousands of distinct users are expected.
    """

    def __init__(self, pool: ThreadPoolExecutor) -> None:
        self._pool = pool
        self._jobstore = JobStore()
        self._user_locks: dict[str, threading.Lock] = {}
        self._user_locks_guard = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_user_lock(self, user_id: str) -> threading.Lock:
        """Atomically get-or-create the per-user lock."""
        with self._user_locks_guard:
            if user_id not in self._user_locks:
                self._user_locks[user_id] = threading.Lock()
            return self._user_locks[user_id]

    def _fire_callback(self, callback_url: str, result: JobResult | None) -> None:
        """POST the job result to callback_url. Never raises."""
        if result is None:
            return
        try:
            if _USE_HTTPX:
                _http_lib.post(callback_url, json=result.to_dict(), timeout=10.0)
            else:  # pragma: no cover
                _http_lib.post(callback_url, json=result.to_dict(), timeout=10.0)
        except Exception as exc:
            logger.warning(
                f"CALLBACK_FAILED: job_id={result.job_id} url={callback_url} error={exc!r}"
            )

    def _run_job(
        self,
        job_id: str,
        user_id: str,
        work: Callable[[], dict],
        callback_url: str | None,
    ) -> None:
        """Worker function submitted to the thread pool."""
        user_lock = self._get_user_lock(user_id)
        with user_lock:
            started = datetime.now(timezone.utc)
            self._jobstore.update_status(job_id, "running", started_at=started)
            try:
                result_dict = work()
                self._jobstore.set_result(job_id, result_dict)
                self._jobstore.update_status(
                    job_id, "completed", completed_at=datetime.now(timezone.utc)
                )
            except Exception as exc:
                logger.error(
                    f"JOB_FAILED: job_id={job_id} user_id={user_id} "
                    f"error={exc!r}\n{traceback.format_exc()}"
                )
                self._jobstore.set_error(job_id, repr(exc))
                self._jobstore.update_status(
                    job_id, "failed", completed_at=datetime.now(timezone.utc)
                )

        if callback_url:
            self._fire_callback(callback_url, self._jobstore.get(job_id))

    def _submit(
        self,
        kind: str,
        user_id: str,
        work: Callable[[], dict],
        callback_url: str | None,
    ) -> JobHandle:
        job_id = uuid4().hex
        now = datetime.now(timezone.utc)
        initial = JobResult(job_id=job_id, status="queued", submitted_at=now)
        self._jobstore.put(initial)

        self._pool.submit(self._run_job, job_id, user_id, work, callback_url)

        logger.info(f"JOB_SUBMITTED: kind={kind} job_id={job_id} user_id={user_id}")
        return JobHandle(job_id=job_id, status="queued", submitted_at=now)

    # ------------------------------------------------------------------
    # TaskExecutor interface
    # ------------------------------------------------------------------

    def submit_write(
        self,
        user_id: str,
        work: Callable[[], dict],
        callback_url: str | None = None,
    ) -> JobHandle:
        return self._submit("write", user_id, work, callback_url)

    def submit_consolidate(
        self,
        user_id: str,
        work: Callable[[], dict],
        callback_url: str | None = None,
    ) -> JobHandle:
        return self._submit("consolidate", user_id, work, callback_url)

    def get_job(self, job_id: str) -> JobResult | None:
        return self._jobstore.get(job_id)

    def wait_for(self, job_id: str, timeout: float | None = None) -> JobResult:
        """Poll the jobstore every 50 ms until terminal state or timeout."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            result = self._jobstore.get(job_id)
            if result is not None and result.status in {"completed", "failed"}:
                return result
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"wait_for timed out after {timeout}s for job_id={job_id}"
                )
            time.sleep(0.05)

    @property
    def supports_async_api(self) -> bool:
        # False: the default response mode is sync (block-until-done), which
        # preserves the pre-M2 API contract for callers that don't pass ?sync.
        # Pass ?sync=false explicitly to get async (queued) behaviour.
        return False
