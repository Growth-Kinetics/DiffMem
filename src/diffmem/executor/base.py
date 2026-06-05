"""
Base types and abstract interface for the pluggable task executor.

All concrete executor implementations (InlineExecutor, HatchetExecutor, …)
implement TaskExecutor. Endpoints construct a thunk (Callable[[], dict]) that
closes over the actual writer/consolidator call and hand it to submit_write /
submit_consolidate — keeping the executor decoupled from DiffMemory internals.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Literal

JobStatus = Literal["queued", "running", "completed", "failed"]


@dataclass
class JobHandle:
    """Lightweight receipt returned immediately after submit_*()."""
    job_id: str
    status: JobStatus
    submitted_at: datetime


@dataclass
class JobResult:
    """Full job record stored in JobStore and returned by get_job / wait_for."""
    job_id: str
    status: JobStatus
    submitted_at: datetime
    result: dict | None = field(default=None)
    error: str | None = field(default=None)
    started_at: datetime | None = field(default=None)
    completed_at: datetime | None = field(default=None)

    def to_dict(self) -> dict:
        """Serialize to a plain dict; datetimes are ISO 8601 strings."""

        def _iso(dt: datetime | None) -> str | None:
            return dt.isoformat() if dt is not None else None

        return {
            "job_id": self.job_id,
            "status": self.status,
            "submitted_at": _iso(self.submitted_at),
            "result": self.result,
            "error": self.error,
            "started_at": _iso(self.started_at),
            "completed_at": _iso(self.completed_at),
        }


class TaskExecutor(abc.ABC):
    """Pluggable task executor interface.

    Implementations are responsible for:
    - Sequentializing writes *per user* (per-user serialization guarantee).
    - Tracking job lifecycle via JobHandle / JobResult.
    - Optionally calling a callback URL when a job finishes.
    """

    @abc.abstractmethod
    def submit_write(
        self,
        user_id: str,
        work: Callable[[], dict],
        callback_url: str | None = None,
    ) -> JobHandle:
        """Queue a writer-agent thunk for user_id. Returns immediately."""

    @abc.abstractmethod
    def submit_consolidate(
        self,
        user_id: str,
        work: Callable[[], dict],
        callback_url: str | None = None,
    ) -> JobHandle:
        """Queue a consolidator thunk for user_id. Returns immediately."""

    @abc.abstractmethod
    def get_job(self, job_id: str) -> JobResult | None:
        """Return current JobResult, or None if unknown."""

    @abc.abstractmethod
    def wait_for(self, job_id: str, timeout: float | None = None) -> JobResult:
        """Block until the job reaches a terminal state (completed / failed).

        Raises TimeoutError if *timeout* seconds elapse first.
        If timeout is None, waits indefinitely.
        """

    @property
    @abc.abstractmethod
    def supports_async_api(self) -> bool:  # noqa: D401
        """True if this backend supports async job submission (job_id polling)."""
