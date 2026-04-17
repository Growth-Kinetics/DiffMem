"""
Construct backend instances from environment variables.

The factory is the *only* place that reads env vars for backend
configuration, so adding a new backend is a purely local change.
"""

import logging
import os
from typing import Optional

from .base import BackupBackend, NoopBackupBackend, StorageBackend
from .github_backup import GitHubBackupBackend
from .local_storage import LocalStorageBackend

logger = logging.getLogger(__name__)


# Default on-disk locations. These are inside a single /data volume so a
# self-hoster only needs to mount one directory.
DEFAULT_STORAGE_PATH = "/data/storage"
DEFAULT_WORKTREE_ROOT = "/data/worktrees"


def build_storage_backend() -> StorageBackend:
    """Build the storage backend from env."""
    backend_type = os.getenv("STORAGE_BACKEND", "local").lower()
    storage_path = os.getenv("STORAGE_PATH", DEFAULT_STORAGE_PATH)
    worktree_root = os.getenv("WORKTREE_ROOT", DEFAULT_WORKTREE_ROOT)

    if backend_type == "local":
        logger.info(f"STORAGE: local (storage={storage_path}, worktrees={worktree_root})")
        return LocalStorageBackend(storage_path=storage_path, worktree_root=worktree_root)

    raise ValueError(
        f"Unknown STORAGE_BACKEND={backend_type!r}. Supported: local"
    )


def build_backup_backend() -> BackupBackend:
    """
    Build the backup backend from env.

    Backwards compatibility: if GITHUB_REPO_URL and GITHUB_TOKEN are set but
    BACKUP_BACKEND is unset, we implicitly select `github`. This keeps
    existing deployments working without any env changes.
    """
    explicit = os.getenv("BACKUP_BACKEND")
    github_url = os.getenv("GITHUB_REPO_URL")
    github_token = os.getenv("GITHUB_TOKEN")

    if explicit is None:
        backend_type = "github" if (github_url and github_token) else "none"
        if backend_type == "github":
            logger.info("BACKUP: auto-selected 'github' (legacy GITHUB_* env vars set)")
    else:
        backend_type = explicit.lower()

    if backend_type == "none":
        logger.info("BACKUP: disabled")
        return NoopBackupBackend()

    if backend_type == "github":
        if not (github_url and github_token):
            logger.warning(
                "BACKUP: 'github' selected but GITHUB_REPO_URL/GITHUB_TOKEN missing; "
                "falling back to 'none'"
            )
            return NoopBackupBackend()
        logger.info(f"BACKUP: github ({github_url})")
        return GitHubBackupBackend(github_url=github_url, github_token=github_token)

    raise ValueError(
        f"Unknown BACKUP_BACKEND={backend_type!r}. Supported: none, github"
    )


def backup_interval_minutes(default: int = 30) -> int:
    """Minutes between periodic backup runs. 0 disables periodic backups."""
    # Prefer new name, fall back to legacy SYNC_INTERVAL_MINUTES.
    for var in ("BACKUP_INTERVAL_MINUTES", "SYNC_INTERVAL_MINUTES"):
        raw = os.getenv(var)
        if raw is not None:
            try:
                return max(0, int(raw))
            except ValueError:
                logger.warning(f"BACKUP: invalid {var}={raw!r}, using default {default}")
                return default
    return default
