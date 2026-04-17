"""
Local-disk storage backend.

Holds the central git storage repo and per-user worktrees on the filesystem.
Contains none of the backup (push/pull to remote) logic -- that lives in a
BackupBackend.
"""

import logging
import os
import shutil
import stat
from pathlib import Path
from typing import List, Optional

import git

from .base import StorageBackend

logger = logging.getLogger(__name__)


class LocalStorageBackend(StorageBackend):
    """
    Bare-ish central git repo + per-user worktrees, all on local disk.

    Layout:
        <storage_path>/          # central git repo (holds all user/* branches)
        <worktree_root>/<user>/  # active worktree for a user
    """

    def __init__(self, storage_path: str, worktree_root: str):
        self.storage_path = Path(storage_path)
        self.worktree_root = Path(worktree_root)
        self.storage: Optional[git.Repo] = None
        self._branch_cache: set = set()
        self._cache_valid = False

    def init(self) -> None:
        self.worktree_root.mkdir(parents=True, exist_ok=True)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        try:
            self.storage = git.Repo(self.storage_path)
            logger.info(f"STORAGE_LOCAL: Loaded existing storage at {self.storage_path}")
        except git.exc.InvalidGitRepositoryError:
            logger.info(f"STORAGE_LOCAL: Initializing new git storage at {self.storage_path}")
            self.storage = git.Repo.init(self.storage_path)

        self._refresh_branch_cache()

    def _refresh_branch_cache(self) -> None:
        if self.storage is None:
            return
        self._branch_cache = {ref.name for ref in self.storage.references}
        self._cache_valid = True

    def _create_worktree_with_retry(self, worktree_path: str, branch_name: str, orphan: bool) -> None:
        """Create a worktree, retrying with --force if a stale registration exists."""
        args = ["add"]
        if orphan:
            args.extend(["--orphan", "-b", branch_name, worktree_path])
        else:
            args.extend([worktree_path, branch_name])

        try:
            self.storage.git.worktree(*args)
        except git.exc.GitCommandError as e:
            if "already registered" in str(e):
                logger.warning("WORKTREE_STALE: Pruning and retrying with --force")
                try:
                    self.storage.git.worktree("prune")
                except Exception:
                    pass
                args.insert(1, "--force")
                self.storage.git.worktree(*args)
            else:
                raise

    def get_user_worktree(self, user_id: str) -> str:
        user_branch = f"user/{user_id}"
        worktree_path = self.worktree_root / user_id

        if worktree_path.exists():
            try:
                _ = git.Repo(worktree_path)
                return str(worktree_path)
            except git.exc.InvalidGitRepositoryError:
                logger.warning(f"WORKTREE_STALE: {worktree_path} is not valid git. Cleaning up.")
                shutil.rmtree(worktree_path)

        logger.info(f"WORKTREE_CREATE: Mounting {user_id} at {worktree_path}")

        if not self._cache_valid:
            self._refresh_branch_cache()

        local_exists = user_branch in self._branch_cache

        if local_exists:
            self._create_worktree_with_retry(str(worktree_path), user_branch, orphan=False)
        else:
            self._create_worktree_with_retry(str(worktree_path), user_branch, orphan=True)
            self._branch_cache.add(user_branch)

        return str(worktree_path)

    def list_active_users(self) -> List[str]:
        if not self.worktree_root.exists():
            return []
        return [
            d.name for d in self.worktree_root.iterdir()
            if d.is_dir() and (d / ".git").exists()
        ]

    def user_branch_exists(self, user_id: str) -> bool:
        if not self._cache_valid:
            self._refresh_branch_cache()
        return f"user/{user_id}" in self._branch_cache

    def commit_user(self, user_id: str, message: str) -> bool:
        worktree_path = self.worktree_root / user_id
        if not worktree_path.exists():
            raise ValueError(f"Worktree for {user_id} does not exist")

        repo = git.Repo(worktree_path)
        if repo.is_dirty(untracked_files=True):
            repo.git.add(A=True)
            repo.index.commit(message)
            logger.info(f"STORAGE_COMMIT: Committed changes for {user_id}")
            return True
        return False

    def wipe_user(self, user_id: str) -> None:
        user_branch = f"user/{user_id}"
        worktree_path = self.worktree_root / user_id

        if worktree_path.exists():
            try:
                self.storage.git.worktree("remove", "--force", str(worktree_path))
            except Exception as e:
                logger.warning(f"WIPE: worktree remove failed ({e}); falling back to rmtree")
            if worktree_path.exists():
                shutil.rmtree(worktree_path)
            logger.info(f"WIPE: Removed worktree for {user_id}")

        try:
            branches = self.storage.git.branch("--list", user_branch)
            if branches:
                self.storage.git.branch("-D", user_branch)
                self._branch_cache.discard(user_branch)
                logger.info(f"WIPE: Deleted local branch {user_branch}")
        except Exception as e:
            logger.error(f"WIPE_ERROR: Failed to delete local branch {user_branch}: {e}")
            raise

    # Sentinel line we use to detect (and safely replace) our own hook
    # installation without clobbering user-authored hooks.
    _HOOK_SENTINEL = "# >>> diffmem-post-commit (managed) >>>"
    _HOOK_END_SENTINEL = "# <<< diffmem-post-commit (managed) <<<"

    def install_post_commit_hook(self, api_url: str, api_key: Optional[str]) -> bool:
        """
        Install a post-commit hook that curls a webhook on this service after
        every commit, used to trigger an optional out-of-band backup.

        Worktrees share hooks with the storage repo's common dir, so
        installing once here covers every user worktree.

        If a user-authored post-commit hook already exists, we *append* our
        managed block (delimited by sentinel comments) rather than clobber
        it. Re-running the installer replaces the managed block in place.
        """
        if self.storage is None:
            return False

        hooks_dir = Path(self.storage.git_dir) / "hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)
        hook_path = hooks_dir / "post-commit"
        log_file = hooks_dir / "post-commit.log"

        auth_header = f'-H "Authorization: Bearer {api_key}"' if api_key else ""

        managed_block = f"""{self._HOOK_SENTINEL}
# Auto-generated by DiffMem. Do not edit between sentinel markers;
# re-running the server replaces this block in place.
LOG_FILE="{log_file}"
WORK_DIR=$(git rev-parse --show-toplevel 2>&1)
USER_ID=$(basename "$WORK_DIR")
if [ -n "$USER_ID" ]; then
    curl -s -X POST "{api_url}/memory/$USER_ID/webhook/post-commit" \\
         {auth_header} \\
         --max-time 5 \\
         >> "$LOG_FILE" 2>&1 &
else
    echo "$(date): ERROR - no user_id from $WORK_DIR" >> "$LOG_FILE"
fi
{self._HOOK_END_SENTINEL}
"""

        try:
            if hook_path.exists():
                existing = hook_path.read_text(encoding="utf-8")
                if self._HOOK_SENTINEL in existing and self._HOOK_END_SENTINEL in existing:
                    # Replace our previous managed block in place.
                    pre, _, rest = existing.partition(self._HOOK_SENTINEL)
                    _, _, post = rest.partition(self._HOOK_END_SENTINEL)
                    new_content = pre.rstrip() + "\n\n" + managed_block + post.lstrip()
                else:
                    # User-authored hook exists -- append our block.
                    sep = "" if existing.endswith("\n") else "\n"
                    new_content = existing + sep + "\n" + managed_block
            else:
                new_content = "#!/bin/sh\n\n" + managed_block

            with open(hook_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            st = os.stat(hook_path)
            os.chmod(hook_path, st.st_mode | stat.S_IEXEC)
            logger.info(f"HOOK_INSTALLED: {hook_path}")
            return True
        except Exception as e:
            logger.error(f"HOOK_INSTALL_ERROR: {e}")
            return False
