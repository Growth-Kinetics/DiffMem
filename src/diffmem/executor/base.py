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
from typing import Callable, List, Literal, Optional

JobStatus = Literal["queued", "running", "completed", "failed"]


@dataclass
class WritePayload:
    """Structured payload for a write job.

    Used by HatchetExecutor (which ignores the ``work`` thunk and serialises
    this payload as a Pydantic model into the Hatchet workflow input).
    InlineExecutor ignores this and uses the thunk instead.

    At least one of ``work`` or ``payload`` must be supplied to any
    ``submit_*`` call; the executor's choice which to use.
    """
    user_id: str
    memory_input: str
    session_id: str
    session_date: Optional[str] = None


@dataclass
class ConsolidatePayload:
    """Structured payload for a consolidate job.

    Same contract as WritePayload but for consolidation operations.
    Fields match the union of ConsolidateRequest and
    ProcessCommitAndConsolidateRequest in server.py.
    """
    user_id: str
    # Write fields (present for process-commit-and-consolidate)
    memory_input: Optional[str] = None
    session_id: Optional[str] = None
    session_date: Optional[str] = None
    # Consolidate-specific fields
    tools: Optional[List[str]] = None
    window: int = 3
    soft_cap_tokens: int = 32000


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

    Contract for ``work`` / ``payload`` parameters:
    - At least one of ``work`` or ``payload`` must be supplied.
    - InlineExecutor requires ``work`` (the thunk) and ignores ``payload``.
    - HatchetExecutor requires ``payload`` (serialisable struct) and ignores
      ``work`` (the worker process has no access to Python closures).
    - If both are supplied, the executor decides which to use.
    """

    @abc.abstractmethod
    def submit_write(
        self,
        user_id: str,
        work: Optional[Callable[[], dict]],
        payload: Optional[WritePayload] = None,
        callback_url: Optional[str] = None,
    ) -> JobHandle:
        """Queue a writer-agent job for user_id. Returns immediately."""

    @abc.abstractmethod
    def submit_consolidate(
        self,
        user_id: str,
        work: Optional[Callable[[], dict]],
        payload: Optional[ConsolidatePayload] = None,
        callback_url: Optional[str] = None,
    ) -> JobHandle:
        """Queue a consolidator job for user_id. Returns immediately."""

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
