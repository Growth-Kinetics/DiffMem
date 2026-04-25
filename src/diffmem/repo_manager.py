"""
RepoManager: thin composition of a StorageBackend and a BackupBackend.

This class used to conflate three concerns (local git ops, worktree
lifecycle, GitHub sync). It is now just a coordinator -- all substantive
work happens inside the backends in `diffmem.storage`.

Writes are *never* blocked on the backup backend. Commits are local and
instant; backups happen out-of-band (triggered by the post-commit webhook
or the periodic scheduler in `server.py`).
"""

import logging
from typing import List, Optional

from .storage import (
    BackupBackend,
    StorageBackend,
    build_backup_backend,
    build_storage_backend,
)

logger = logging.getLogger(__name__)


class RepoManager:
    """Coordinates a StorageBackend and an optional BackupBackend."""

    def __init__(
        self,
        storage: Optional[StorageBackend] = None,
        backup: Optional[BackupBackend] = None,
    ):
        self.storage: StorageBackend = storage or build_storage_backend()
        self.backup: BackupBackend = backup if backup is not None else build_backup_backend()

        self.storage.init()
        self.backup.configure(self.storage)

        # Cold-start restore: only when the backup is enabled AND the local
        # storage has no user branches yet. Otherwise restore_all is a waste
        # of network on every restart.
        if self.backup.enabled and self._storage_is_empty():
            try:
                restored = self.backup.restore_all()
                if restored:
                    logger.info(f"REPO_MANAGER: Restored {restored} branches from backup")
                else:
                    logger.info("REPO_MANAGER: Backup enabled but nothing to restore")
            except Exception as e:
                logger.warning(f"REPO_MANAGER: restore_all failed (non-fatal): {e}")

    def _storage_is_empty(self) -> bool:
        """True if the local storage has no user branches (fresh volume)."""
        try:
            for user_id in self.storage.list_active_users():
                # Any mounted worktree means we've been here before.
                return False
            # No mounted worktrees -- check the branch cache directly.
            try:
                cache = self.storage._branch_cache  # type: ignore[attr-defined]
                return not any(ref.startswith("user/") for ref in cache)
            except AttributeError:
                return True
        except Exception:
            return True

    # ---- Pass-through to storage ----

    def get_user_worktree(self, user_id: str) -> str:
        return self.storage.get_user_worktree(user_id)

    def list_active_users(self) -> List[str]:
        return self.storage.list_active_users()

    def install_post_commit_hook(self, api_url: str, api_key: Optional[str] = None) -> bool:
        return self.storage.install_post_commit_hook(api_url, api_key)

    def wipe_user(self, user_id: str) -> None:
        """Remove a user locally, then best-effort delete their backup."""
        self.storage.wipe_user(user_id)
        # Best-effort remote deletion. Never raises.
        try:
            self.backup.delete_user(user_id)
        except Exception as e:
            logger.warning(f"WIPE: backup delete failed (non-fatal): {e}")

    # ---- Commit + backup ----

    def sync_user(self, user_id: str, message: str = "Auto-sync") -> None:
        """
        Commit any uncommitted changes for `user_id`, then trigger a backup.

        The backup runs synchronously here (caller is already on a background
        task / scheduler), but failures are swallowed so writes stay durable
        locally even if the remote is unreachable.
        """
        try:
            committed = self.storage.commit_user(user_id, message)
            if not committed:
                logger.debug(f"SYNC: no local changes for {user_id}")
        except Exception as e:
            logger.error(f"SYNC_COMMIT_ERROR for {user_id}: {e}")
            return

        try:
            self.backup.sync_user(user_id)
        except Exception as e:
            logger.error(f"SYNC_BACKUP_ERROR for {user_id}: {e}")
