"""
GitHub backup backend.

Mirrors user branches to a GitHub repository. Runs out-of-band -- failures
are logged and swallowed, never propagated into the request path.

Each user gets a dedicated orphan branch `user/{user_id}` in the remote,
mirroring the local storage layout.

Authentication: we *do not* embed the token in the remote URL (that would
persist it to .git/config on the mounted volume and leak it into ps output).
Instead we inject it at call time via GIT_ASKPASS + GIT_TERMINAL_PROMPT=0.
"""

import logging
import os
import stat
import tempfile
from pathlib import Path
from typing import Dict, Optional

import git

from .base import BackupBackend, StorageBackend
from .local_storage import LocalStorageBackend

logger = logging.getLogger(__name__)


class GitHubBackupBackend(BackupBackend):
    """Pushes user branches to a GitHub repo on demand."""

    name = "github"
    enabled = True

    def __init__(self, github_url: str, github_token: str, branch_prefix: str = "user/"):
        if not github_url:
            raise ValueError("GitHubBackupBackend requires github_url")
        if not github_token:
            raise ValueError("GitHubBackupBackend requires github_token")
        self.github_url = github_url
        self.github_token = github_token
        self.branch_prefix = branch_prefix
        self.storage: Optional[LocalStorageBackend] = None
        self._askpass_path: Optional[Path] = None

    def _ensure_askpass_script(self) -> Path:
        """
        Write a tiny shell script git can call via GIT_ASKPASS to retrieve
        credentials. The token itself is passed through the environment so it
        never appears on any command line or in .git/config.
        """
        if self._askpass_path is not None and self._askpass_path.exists():
            return self._askpass_path

        fd, path = tempfile.mkstemp(prefix="diffmem-askpass-", suffix=".sh")
        try:
            with os.fdopen(fd, "w") as f:
                # GIT_ASKPASS is invoked with a prompt like "Username for ..."
                # or "Password for ...". We answer 'x-access-token' for any
                # username query (GitHub's recommended PAT username) and the
                # token itself for any password query.
                f.write(
                    '#!/bin/sh\n'
                    'case "$1" in\n'
                    '    Username*) echo "x-access-token" ;;\n'
                    '    *)         echo "$DIFFMEM_GH_TOKEN" ;;\n'
                    'esac\n'
                )
            os.chmod(path, stat.S_IRWXU)
            self._askpass_path = Path(path)
            return self._askpass_path
        except Exception:
            os.unlink(path)
            raise

    def _git_env(self) -> Dict[str, str]:
        """Environment overlay for git subprocesses that need auth."""
        return {
            "GIT_ASKPASS": str(self._ensure_askpass_script()),
            "GIT_TERMINAL_PROMPT": "0",
            "DIFFMEM_GH_TOKEN": self.github_token,
        }

    def configure(self, storage: StorageBackend) -> None:
        if not isinstance(storage, LocalStorageBackend):
            raise TypeError("GitHubBackupBackend requires a LocalStorageBackend")
        self.storage = storage

        if self.storage.storage is None:
            raise RuntimeError("Storage backend must be init()'d before configure()")

        repo = self.storage.storage
        try:
            if "origin" in [r.name for r in repo.remotes]:
                repo.remotes.origin.set_url(self.github_url)
            else:
                repo.create_remote("origin", self.github_url)
            # Pre-create the askpass helper so the first push doesn't race.
            self._ensure_askpass_script()
            logger.info("BACKUP_GITHUB: Configured remote 'origin' (credentials via GIT_ASKPASS)")
        except Exception as e:
            logger.error(f"BACKUP_GITHUB: Failed to configure remote: {e}")

    def sync_user(self, user_id: str) -> bool:
        if self.storage is None or self.storage.storage is None:
            logger.warning("BACKUP_GITHUB: not configured, skipping sync")
            return False

        branch = f"{self.branch_prefix}{user_id}"
        try:
            # git.Repo.git.custom_environment() scopes env changes to one call.
            with self.storage.storage.git.custom_environment(**self._git_env()):
                self.storage.storage.remotes.origin.push(f"{branch}:{branch}")
            logger.info(f"BACKUP_GITHUB: Pushed {branch}")
            return True
        except Exception as e:
            logger.error(f"BACKUP_GITHUB: Failed to push {branch}: {e}")
            return False

    def restore_all(self) -> int:
        """
        Fetch all user/* branches from the remote and create matching local
        branches. Best-effort and idempotent -- existing local branches are
        left alone.
        """
        if self.storage is None or self.storage.storage is None:
            return 0

        repo = self.storage.storage
        try:
            with repo.git.custom_environment(**self._git_env()):
                origin = repo.remotes.origin
                fetch_info = origin.fetch(
                    refspec=f"refs/heads/{self.branch_prefix}*:refs/remotes/origin/{self.branch_prefix}*"
                )
            logger.info(f"BACKUP_GITHUB: Fetched {len(fetch_info)} refs")
        except git.exc.GitCommandError as e:
            # Common on a brand-new remote: no refs exist yet.
            logger.info(f"BACKUP_GITHUB: Nothing to restore ({e})")
            return 0
        except Exception as e:
            logger.warning(f"BACKUP_GITHUB: Restore fetch failed: {e}")
            return 0

        self.storage._refresh_branch_cache()

        user_refs = [
            ref for ref in repo.references
            if f"origin/{self.branch_prefix}" in ref.name
        ]
        local_heads = {ref.name for ref in repo.heads}
        created = 0
        for remote_ref in user_refs:
            local_name = remote_ref.name.split("origin/", 1)[-1]
            if local_name in local_heads:
                continue
            try:
                repo.create_head(local_name, remote_ref)
                created += 1
            except Exception as e:
                logger.warning(f"BACKUP_GITHUB: could not create local branch {local_name}: {e}")

        if created:
            logger.info(f"BACKUP_GITHUB: Restored {created} user branches from remote")
            self.storage._refresh_branch_cache()
        return created
