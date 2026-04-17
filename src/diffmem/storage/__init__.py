"""
Storage and backup backends for DiffMem.

This package decouples three previously-conflated concerns:

1. StorageBackend  -- where the bare git repo and worktrees live (always local
   disk, but configurable for future remote-storage backends).

2. BackupBackend   -- an *optional*, out-of-band mirror of user branches to an
   external location (GitHub today; S3/GCS later). Backups run on a scheduler,
   never in the request hot path.

3. RepoManager     -- thin composition of the two. See diffmem.repo_manager.

The default self-hosting configuration is:

    STORAGE_BACKEND=local
    BACKUP_BACKEND=none

which requires no external credentials and stores everything on a mounted
volume. The `github` backup backend is a first-class citizen for users who
want an offsite mirror "for free" using a private GitHub repo.
"""

from .base import StorageBackend, BackupBackend, NoopBackupBackend
from .local_storage import LocalStorageBackend
from .github_backup import GitHubBackupBackend
from .factory import build_storage_backend, build_backup_backend

__all__ = [
    "StorageBackend",
    "BackupBackend",
    "NoopBackupBackend",
    "LocalStorageBackend",
    "GitHubBackupBackend",
    "build_storage_backend",
    "build_backup_backend",
]
