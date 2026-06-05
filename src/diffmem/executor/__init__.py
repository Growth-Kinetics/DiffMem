"""executor — pluggable task execution capability for DiffMem.

Public surface (re-exported here so callers import from `diffmem.executor`):
  TaskExecutor   — abstract base; implement to add a new backend.
  JobHandle      — lightweight receipt returned by submit_*().
  JobResult      — full job record (status, result, error, timestamps).
  JobStatus      — Literal["queued", "running", "completed", "failed"].
  build_executor — factory; reads EXECUTOR env var, returns TaskExecutor.
"""

from .base import JobHandle, JobResult, JobStatus, TaskExecutor
from .factory import build_executor

__all__ = [
    "TaskExecutor",
    "JobHandle",
    "JobResult",
    "JobStatus",
    "build_executor",
]
