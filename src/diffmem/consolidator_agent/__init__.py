"""Consolidator agent: out-of-band repair pass over a user's memory worktree.

Three tools — dedupe, redistribute, link — runnable independently or chained.
Designed to run outside the writer's session hot path; produces its own commits
with a `consolidate:` prefix so retrieval agents can tell repair commits apart
from session-formation commits.
"""
from .agent import ConsolidatorAgent
from .lock import ConsolidatorLock, LockBusyError

__all__ = ["ConsolidatorAgent", "ConsolidatorLock", "LockBusyError"]
