"""
Abstract base classes for storage and backup backends.

A StorageBackend owns the local git repository and the per-user worktrees.
It must always be a real directory on the local filesystem -- the retrieval
agent shells out to `grep`, `git log`, etc., which need a checked-out tree.

A BackupBackend is an *optional* mirror. It is invoked out-of-band (by a
scheduled task or on explicit demand), never in the request hot path. A
backup failure must never break a write.
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class StorageBackend(ABC):
    """Manages the local git storage repo and per-user worktrees."""

    @abstractmethod
    def init(self) -> None:
        """Initialize the storage repo. Idempotent."""

    @abstractmethod
    def get_user_worktree(self, user_id: str) -> str:
        """
        Return an absolute path to a checked-out worktree for this user,
        creating the branch and worktree if they do not yet exist.
        """

    @abstractmethod
    def list_active_users(self) -> List[str]:
        """Return user_ids that currently have a mounted worktree."""

    @abstractmethod
    def user_branch_exists(self, user_id: str) -> bool:
        """True if a branch exists locally for this user."""

    @abstractmethod
    def commit_user(self, user_id: str, message: str) -> bool:
        """
        Stage and commit any uncommitted changes in the user's worktree.
        Returns True if a commit was made, False if the tree was clean.
        """

    @abstractmethod
    def wipe_user(self, user_id: str) -> None:
        """Remove the user's worktree and local branch. Right-to-be-forgotten."""

    @abstractmethod
    def install_post_commit_hook(self, api_url: str, api_key: Optional[str]) -> bool:
        """
        Install a post-commit hook that pings an HTTP webhook after each
        commit. Used to trigger out-of-band work (e.g. backup). Optional --
        implementations may choose to no-op.
        """


class BackupBackend(ABC):
    """
    Optional mirror of user branches to an external location.

    Must be safe to call from a background task. Must never raise into the
    request path; log and swallow errors instead.
    """

    #: Human-readable name, used in logs and /health.
    name: str = "base"

    #: When False, RepoManager skips restore_all() and periodic-backup wiring.
    #: Concrete backends override to True.
    enabled: bool = False

    @abstractmethod
    def configure(self, storage: StorageBackend) -> None:
        """Wire the backup backend to its storage source. Called once at startup."""

    @abstractmethod
    def sync_user(self, user_id: str) -> bool:
        """Push the user's branch to the remote. Returns True on success."""

    @abstractmethod
    def restore_all(self) -> int:
        """
        Pull user branches from the remote into the local storage repo.
        Called once at startup (best-effort; idempotent). Returns number of
        branches newly restored.
        """


class NoopBackupBackend(BackupBackend):
    """Default backup backend: does nothing. For volume-only self-hosters."""

    name = "none"
    enabled = False

    def configure(self, storage: StorageBackend) -> None:
        return

    def sync_user(self, user_id: str) -> bool:
        return True

    def restore_all(self) -> int:
        return 0
