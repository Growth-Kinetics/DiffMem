"""
In-process job store backed by an OrderedDict for FIFO eviction.

Thread-safe via a single Lock. Evicts oldest entries when capacity (1000)
is exceeded so the store never grows unbounded in long-running deployments.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict

from .base import JobResult, JobStatus

logger = logging.getLogger(__name__)

_MAX_ENTRIES = 1000


class JobStore:
    """Thread-safe ordered job store with FIFO eviction at 1 000 entries."""

    def __init__(self) -> None:
        self._store: OrderedDict[str, JobResult] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def put(self, result: JobResult) -> None:
        """Insert or replace a job record, evicting the oldest if needed."""
        with self._lock:
            # Evict BEFORE inserting if we are already at capacity.
            if len(self._store) >= _MAX_ENTRIES:
                evicted_id, _ = self._store.popitem(last=False)
                logger.info(f"JOBSTORE_EVICTED: job_id={evicted_id}")
            self._store[result.job_id] = result

    def update_status(self, job_id: str, status: JobStatus, **kw) -> None:
        """Update the status field (and any extra fields passed as kw).

        Accepted kw keys: started_at, completed_at.
        """
        with self._lock:
            job = self._store.get(job_id)
            if job is None:
                return
            job.status = status
            if "started_at" in kw:
                job.started_at = kw["started_at"]
            if "completed_at" in kw:
                job.completed_at = kw["completed_at"]

    def set_result(self, job_id: str, result_dict: dict) -> None:
        """Store the successful return value of a work thunk."""
        with self._lock:
            job = self._store.get(job_id)
            if job is not None:
                job.result = result_dict

    def set_error(self, job_id: str, err: str) -> None:
        """Store the error string for a failed job."""
        with self._lock:
            job = self._store.get(job_id)
            if job is not None:
                job.error = err

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get(self, job_id: str) -> JobResult | None:
        """Return the live JobResult object, or None if unknown.

        Callers must treat the returned object as read-only; mutations should
        go through update_status / set_result / set_error.  The object is not
        copied — reads are cheap and a copy would not protect against the
        underlying dataclass fields being mutated by another thread anyway.
        """
        with self._lock:
            return self._store.get(job_id)
