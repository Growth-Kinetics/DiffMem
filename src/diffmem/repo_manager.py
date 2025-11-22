import os
import logging
import shutil
import stat
from pathlib import Path
from typing import Optional
import git

# Structured logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RepoManager:
    """
    Manages Git operations for the Orphan Branch + Worktree architecture.
    
    Concepts:
    - Storage Repo: A bare or hidden git repository that holds all history/objects.
    - Worktree: A checked-out directory for a specific user's branch (active context).
    - User Branch: 'user/{user_id}' - completely isolated orphan branch.
    
    Performance Optimizations (for 100+ users):
    - Branch cache: O(1) lookups instead of scanning all refs
    - Lazy fetch: Only fetch missing branches on-demand
    - Batch operations: Create local tracking branches in bulk on init
    - Consolidated retry logic: DRY for worktree creation
    - Reduced logging: Debug-level for frequent operations
    """
    
    def __init__(self, storage_path: str, worktree_root: str, github_url: str = None, github_token: str = None, lazy_fetch: bool = True):
        """
        Args:
            storage_path: Path to the central git storage (usually hidden/bare)
            worktree_root: Path where user contexts will be mounted/checked out
            github_url: Remote GitHub URL for syncing
            github_token: Auth token for GitHub
        """
        self.storage_path = Path(storage_path)
        self.worktree_root = Path(worktree_root)
        self.github_url = github_url
        self.github_token = github_token
        self.lazy_fetch = lazy_fetch
        
        # Cache for branch existence checks (improves performance at scale)
        self._branch_cache = set()
        self._cache_valid = False
        
        # Ensure roots exist
        self.worktree_root.mkdir(parents=True, exist_ok=True)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load Storage Repo
        is_fresh_init = False
        try:
            # Try to load existing repo
            self.storage = git.Repo(self.storage_path)
            logger.info(f"REPO_MANAGER: Loaded existing storage at {self.storage_path}")
        except git.exc.InvalidGitRepositoryError:
            # Not a git repo yet - initialize it
            logger.info(f"REPO_MANAGER: Initializing new git storage at {self.storage_path}")
            self.storage = git.Repo.init(self.storage_path)
            is_fresh_init = True
            logger.info(f"REPO_MANAGER: Initialized new git storage at {self.storage_path}")

        # Configure Remote if credentials provided
        self._configure_remote()
        
        # Always fetch from remote on startup if configured (not just fresh init)
        # This ensures we have latest branches from GitHub
        if self.github_url:
            if is_fresh_init:
                self._fetch_remote_on_init()
            else:
                # For existing repos, do a lightweight fetch to update references
                self._sync_remote_branches()

    def _configure_remote(self):
        """Configures the 'origin' remote on the storage repository."""
        if self.github_url and self.github_token:
            try:
                # Construct authenticated URL
                if "github.com" in self.github_url:
                    clean_url = self.github_url.replace("https://", "").replace("http://", "")
                    auth_url = f"https://{self.github_token}@{clean_url}"
                else:
                    auth_url = self.github_url

                if "origin" in self.storage.remotes:
                    self.storage.remotes.origin.set_url(auth_url)
                else:
                    self.storage.create_remote("origin", auth_url)
                
                logger.info("REPO_MANAGER: Configured remote 'origin'")
            except Exception as e:
                logger.error(f"REPO_MANAGER: Failed to configure remote: {e}")
    
    def _refresh_branch_cache(self):
        """Refreshes the internal branch cache for fast lookups."""
        self._branch_cache = {ref.name for ref in self.storage.references}
        self._cache_valid = True
        logger.debug(f"BRANCH_CACHE: Refreshed with {len(self._branch_cache)} references")
    
    def _create_worktree_with_retry(self, worktree_path: str, branch_name: str, orphan: bool = False):
        """
        Creates a worktree with automatic retry on stale registration errors.
        
        Args:
            worktree_path: Path where worktree should be created
            branch_name: Branch name to checkout
            orphan: If True, create as orphan branch
        """
        args = ["add"]
        if orphan:
            args.extend(["--orphan", "-b", branch_name, worktree_path])
        else:
            args.extend([worktree_path, branch_name])
        
        try:
            self.storage.git.worktree(*args)
        except git.exc.GitCommandError as e:
            # Retry with --force if stale registration detected
            if "already registered" in str(e):
                logger.warning(f"WORKTREE_STALE: Cleaning up and retrying with --force")
                # First try pruning
                try:
                    self.storage.git.worktree("prune")
                except:
                    pass
                # Retry with force
                args.insert(1, "--force")
                self.storage.git.worktree(*args)
            else:
                raise
    
    def _sync_remote_branches(self):
        """Lightweight sync of remote branch references (for non-fresh startups)."""
        try:
            logger.info("REPO_MANAGER: Syncing remote branch references...")
            origin = self.storage.remotes.origin
            
            # Fetch just the refs without downloading objects we already have
            fetch_info = origin.fetch(refspec='refs/heads/user/*:refs/remotes/origin/user/*')
            logger.info(f"REPO_MANAGER: Synced {len(fetch_info)} remote references")
            
            # Refresh branch cache after fetch
            self._refresh_branch_cache()
            
            # Count user branches (from cache, O(1))
            remote_count = sum(1 for ref in self._branch_cache if 'origin/user/' in ref)
            logger.info(f"REPO_MANAGER: Found {remote_count} user branches on remote")
            
        except Exception as e:
            logger.warning(f"REPO_MANAGER: Could not sync remote branches: {e}")
    
    def _fetch_remote_on_init(self):
        """Fetches all branches from remote on fresh initialization."""
        try:
            logger.info("REPO_MANAGER: Fetching existing branches from remote...")
            origin = self.storage.remotes.origin
            
            # Fetch all branches from remote with refspec to get user branches
            fetch_info = origin.fetch(refspec='refs/heads/user/*:refs/remotes/origin/user/*')
            logger.info(f"REPO_MANAGER: Fetch completed, received {len(fetch_info)} refs")
            
            # Refresh cache after fetch
            self._refresh_branch_cache()
            
            # Find all remote user branches using cache
            user_branches = [ref for ref in self.storage.references if 'origin/user/' in ref.name]
            
            if user_branches:
                logger.info(f"REPO_MANAGER: Found {len(user_branches)} user branches on remote")
                
                # Build set of existing local branch names for O(1) lookups
                local_heads = {ref.name for ref in self.storage.heads}
                
                # Create local tracking branches for each remote user branch
                created_count = 0
                for remote_ref in user_branches:
                    # Extract branch name: refs/remotes/origin/user/alex -> user/alex
                    if 'origin/user/' not in remote_ref.name:
                        continue
                    
                    local_branch_name = remote_ref.name.split('origin/')[-1]
                    
                    try:
                        # Check if local branch already exists (O(1) set lookup)
                        if local_branch_name in local_heads:
                            continue
                        
                        # Create local branch tracking remote
                        self.storage.create_head(local_branch_name, remote_ref)
                        created_count += 1
                        logger.debug(f"REPO_MANAGER: Created local branch {local_branch_name}")
                    except Exception as e:
                        logger.warning(f"REPO_MANAGER: Could not create local branch {local_branch_name}: {e}")
                
                if created_count > 0:
                    logger.info(f"REPO_MANAGER: Created {created_count} local tracking branches")
            else:
                logger.info("REPO_MANAGER: No existing user branches found on remote (fresh deployment)")
                
        except Exception as e:
            # Non-fatal: remote might not exist yet or network issues
            logger.warning(f"REPO_MANAGER: Could not fetch from remote (this is normal for first deployment): {e}")

    def get_user_worktree(self, user_id: str) -> str:
        """
        Ensures the user's branch is checked out into a worktree and returns the path.
        
        Flow:
        1. Check if worktree already exists at active path.
        2. If not, create it from existing branch OR create new orphan branch.
        """
        user_branch = f"user/{user_id}"
        worktree_path = self.worktree_root / user_id
        
        # 1. Check if worktree directory exists and is a valid git repo/worktree
        if worktree_path.exists():
            try:
                # Verify it's a valid git context
                _ = git.Repo(worktree_path)
                logger.info(f"WORKTREE_FOUND: Returning existing worktree for {user_id}")
                return str(worktree_path)
            except git.exc.InvalidGitRepositoryError:
                # Stale directory? Clean up
                logger.warning(f"WORKTREE_STALE: {worktree_path} exists but is not valid git. Cleaning up.")
                shutil.rmtree(worktree_path)
        
        # 2. Create Worktree
        logger.info(f"WORKTREE_CREATE: Mounting {user_id} at {worktree_path}")
        
        # Refresh cache if invalid
        if not self._cache_valid:
            self._refresh_branch_cache()
        
        # Check if branch exists locally or remotely (using cache for O(1) lookup)
        local_branch_exists = user_branch in self._branch_cache
        remote_branch_name = f"origin/{user_branch}"
        remote_branch_exists = remote_branch_name in self._branch_cache
        
        logger.debug(f"BRANCH_CHECK: {user_id} local={local_branch_exists}, remote={remote_branch_exists}")
        
        # If branch doesn't exist locally or remotely, try fetching from remote (lazy fetch)
        if not local_branch_exists and not remote_branch_exists and self.github_url and self.lazy_fetch:
            logger.info(f"LAZY_FETCH: Attempting to fetch {user_branch} from remote")
            try:
                origin = self.storage.remotes.origin
                # Try to fetch this specific branch
                fetch_result = origin.fetch(refspec=f"refs/heads/{user_branch}:refs/remotes/origin/{user_branch}")
                logger.debug(f"FETCH_RESULT: {len(fetch_result)} refs fetched")
                
                # Refresh cache and re-check
                self._refresh_branch_cache()
                local_branch_exists = user_branch in self._branch_cache
                remote_branch_exists = remote_branch_name in self._branch_cache
                logger.info(f"LAZY_FETCH_COMPLETE: Found remote={remote_branch_exists}")
            except Exception as e:
                logger.debug(f"LAZY_FETCH: Branch {user_branch} not found on remote: {e}")
        
        # If remote branch exists but not local, create local tracking branch
        if remote_branch_exists and not local_branch_exists:
            logger.info(f"REMOTE_BRANCH_FOUND: Creating local tracking branch for {user_branch}")
            try:
                remote_ref = self.storage.references[remote_branch_name]
                self.storage.create_head(user_branch, remote_ref)
                local_branch_exists = True
                self._branch_cache.add(user_branch)  # Update cache
                logger.info(f"LOCAL_BRANCH_CREATED: {user_branch}")
            except Exception as e:
                logger.error(f"Failed to create local tracking branch: {e}")
        
        try:
            if local_branch_exists:
                # Create worktree from existing local branch
                logger.info(f"BRANCH_EXISTS: Creating worktree from existing branch {user_branch}")
                self._create_worktree_with_retry(str(worktree_path), user_branch, orphan=False)
            else:
                # Create new orphan branch via worktree
                logger.info(f"NEW_BRANCH: Creating new orphan branch {user_branch}")
                self._create_worktree_with_retry(str(worktree_path), user_branch, orphan=True)
                self._branch_cache.add(user_branch)  # Update cache

            return str(worktree_path)
            
        except Exception as e:
            logger.error(f"WORKTREE_ERROR: Failed to create worktree for {user_id}: {e}")
            raise e

    def install_post_commit_hook(self, api_url: str, api_key: str = None):
        """
        Installs a global post-commit hook in the storage repository.
        This hook will run for ALL worktrees (since worktrees share hooks).
        The hook detects the user from the worktree path and calls the API.
        """
        hooks_dir = Path(self.storage.git_dir) / "hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)
        
        hook_path = hooks_dir / "post-commit"
        
        # Hook script content
        # It detects the USER_ID from the current working directory (which will be the worktree root)
        # The worktree root is extracted using basename
        
        auth_header = f'-H "Authorization: Bearer {api_key}"' if api_key else ""
        
        # Create a log file for debugging hook execution
        log_file = Path(self.storage.git_dir) / "hooks" / "post-commit.log"
        
        script_content = f"""#!/bin/sh
# DiffMem Post-Commit Hook
# Triggers server-side reindex and sync for the active user

# Log file for debugging
LOG_FILE="{log_file}"

# Get the root of the current worktree
WORK_DIR=$(git rev-parse --show-toplevel 2>&1)

# Log the execution
echo "$(date): Hook triggered from $WORK_DIR" >> "$LOG_FILE"

# Extract user_id from worktree path
# Worktree structure: /path/to/worktree_root/{{user_id}}
USER_ID=$(basename "$WORK_DIR")

# Validate we have a user_id
if [ -z "$USER_ID" ]; then
    echo "$(date): ERROR - Could not extract user_id from $WORK_DIR" >> "$LOG_FILE"
    exit 0
fi

echo "$(date): Calling webhook for user_id=$USER_ID" >> "$LOG_FILE"

# Call the webhook (fire and forget, log errors)
# Using a small timeout to avoid blocking git operations
curl -s -X POST "{api_url}/memory/$USER_ID/webhook/post-commit" \\
     {auth_header} \\
     --max-time 5 \\
     >> "$LOG_FILE" 2>&1 &

echo "$(date): Webhook call dispatched" >> "$LOG_FILE"
"""
        
        try:
            with open(hook_path, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            # Make executable (chmod +x)
            st = os.stat(hook_path)
            os.chmod(hook_path, st.st_mode | stat.S_IEXEC)
            
            logger.info(f"HOOK_INSTALLED: Post-commit hook at {hook_path}")
            return True
        except Exception as e:
            logger.error(f"HOOK_INSTALL_ERROR: {e}")
            return False

    def sync_user(self, user_id: str, message: str = "Auto-sync"):
        """
        Commits active changes in the user's worktree and pushes to storage/remote.
        """
        worktree_path = self.worktree_root / user_id
        if not worktree_path.exists():
            raise ValueError(f"Worktree for {user_id} does not exist")
        
        repo = git.Repo(worktree_path)
        user_branch = f"user/{user_id}"
        
        # 1. Commit local changes (triggers post-commit hook)
        if repo.is_dirty(untracked_files=True):
            repo.git.add(A=True)
            repo.index.commit(message)
            logger.info(f"SYNC_COMMIT: Committed changes for {user_id}")
        else:
            logger.debug(f"SYNC_SKIP: No changes to commit for {user_id}")
            
        # 2. Push branch to remote
        self._push_branch_to_remote(user_branch)

    def _push_branch_to_remote(self, branch_name: str):
        """Pushes a specific branch from storage to origin."""
        if not self.github_url:
            logger.info("SYNC_SKIP: No remote URL configured")
            return

        try:
            origin = self.storage.remotes.origin
            # Push specific ref: refs/heads/user/alex:refs/heads/user/alex
            origin.push(f"{branch_name}:{branch_name}")
            logger.info(f"SYNC_PUSH: Pushed {branch_name} to remote")
        except Exception as e:
            logger.error(f"SYNC_PUSH_ERROR: Failed to push {branch_name}: {e}")

    def list_active_users(self):
        """Lists user IDs with active worktrees."""
        if not self.worktree_root.exists():
            return []
        return [d.name for d in self.worktree_root.iterdir() if d.is_dir() and (d / ".git").exists()]

    def cleanup_worktree(self, user_id: str):
        """Removes a user's worktree (unmounts) but keeps data in storage."""
        worktree_path = self.worktree_root / user_id
        if not worktree_path.exists():
            logger.debug(f"CLEANUP_SKIP: Worktree for {user_id} does not exist")
            return
            
        try:
            # Prune from git worktree list
            self.storage.git.worktree("remove", "--force", str(worktree_path))
            # Directory should be gone, but ensure
            if worktree_path.exists():
                shutil.rmtree(worktree_path)
            logger.info(f"WORKTREE_CLEANUP: Removed context for {user_id}")
        except Exception as e:
            logger.error(f"CLEANUP_ERROR: {e}")

    def wipe_user(self, user_id: str):
        """
        Completely removes a user's data (Right to be Forgotten):
        1. Removes the active worktree (if any).
        2. Deletes the local user branch.
        3. Deletes the remote user branch (if configured).
        4. Invalidates branch cache.
        """
        user_branch = f"user/{user_id}"
        logger.info(f"WIPE_USER: Starting deletion for {user_id}...")

        # 1. Remove worktree
        self.cleanup_worktree(user_id)

        # 2. Delete local branch
        try:
            branches = self.storage.git.branch("--list", user_branch)
            if branches:
                self.storage.git.branch("-D", user_branch)
                logger.info(f"WIPE_USER: Deleted local branch {user_branch}")
                # Remove from cache
                self._branch_cache.discard(user_branch)
            else:
                logger.debug(f"WIPE_USER: Local branch {user_branch} not found")
        except Exception as e:
            logger.error(f"WIPE_USER_ERROR: Failed to delete local branch {user_branch}: {e}")
            raise e

        # 3. Delete remote branch
        if self.github_url:
            try:
                self.storage.git.push("origin", "--delete", user_branch)
                logger.info(f"WIPE_USER: Deleted remote branch {user_branch}")
                # Remove from cache
                self._branch_cache.discard(f"origin/{user_branch}")
            except Exception as e:
                logger.warning(f"WIPE_USER: Remote branch deletion skipped (may not exist): {e}")
        
        logger.info(f"WIPE_USER: Completed data wipe for {user_id}")
