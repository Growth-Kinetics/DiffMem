import asyncio
import os
import re
import subprocess
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Runtime config ---

# Public URL of *this* service, used only for the post-commit webhook.
# Defaults to localhost so self-hosters behind a reverse proxy don't need to
# set it; the webhook is an internal loop-back call.
_port = os.getenv("PORT", "8000")
API_URL = os.getenv("API_URL", f"http://localhost:{_port}")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "xiaomi/mimo-v2-omni")

# Authentication is OFF by default: the intended deployment is behind a
# reverse proxy (e.g. Coolify/Traefik) on a private network. Set
# REQUIRE_AUTH=true + API_KEY=... to enable bearer-token auth.
API_KEY = os.getenv("API_KEY")
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "false").lower() == "true"
security = HTTPBearer(auto_error=False)

# CORS: permissive by default (most self-hosters don't need to think about
# this). Override via ALLOWED_ORIGINS (comma-separated) to lock down.
ALLOWED_ORIGINS_RAW = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in ALLOWED_ORIGINS_RAW.split(",") if o.strip()]

from .api import DiffMemory, onboard_new_user
from .repo_manager import RepoManager
from .retrieval_agent import command_router
from .storage.factory import backup_interval_minutes

repo_manager: Optional[RepoManager] = None


async def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not REQUIRE_AUTH:
        return True
    if not API_KEY:
        logger.warning("AUTH_DISABLED: REQUIRE_AUTH=true but no API_KEY configured")
        return True
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True


memory_instances: Dict[str, DiffMemory] = {}


# --- Pydantic models ---

class ContextRequest(BaseModel):
    conversation: List[Dict[str, str]] = Field(..., description="Conversation history")
    max_tokens: int = Field(default=20000, description="Agent's additional context token budget")
    max_turns: int = Field(default=6, description="Max agent exploration turns")
    timeout_seconds: int = Field(default=30, description="Hard timeout for agent loop")
    baseline_only: bool = Field(default=False, description="If true, return only user entity + timeline (no agent, zero LLM cost)")


class ProcessSessionRequest(BaseModel):
    memory_input: str = Field(..., description="Session transcript or memory content")
    session_id: str = Field(..., description="Unique session identifier")
    session_date: Optional[str] = Field(None, description="Session date (YYYY-MM-DD)")


class CommitSessionRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier to commit")


class OnboardUserRequest(BaseModel):
    user_info: str = Field(..., description="Raw information dump about the user")
    session_id: Optional[str] = Field(None, description="Optional session ID for tracking")
    template: Optional[str] = Field(None, description="Pre-filled user entity markdown. If provided, bypasses LLM entity generation.")


class RunCommandRequest(BaseModel):
    command: str = Field(..., description="Sandboxed shell command to execute")


def get_memory_instance(user_id: str, allow_unboarded: bool = False) -> DiffMemory:
    if user_id not in memory_instances:
        try:
            user_repo_path = repo_manager.get_user_worktree(user_id)
            logger.info(f"MEMORY_MOUNT: Mounted worktree for {user_id} at {user_repo_path}")
            memory_instances[user_id] = DiffMemory(
                user_repo_path,
                user_id,
                OPENROUTER_API_KEY,
                DEFAULT_MODEL,
                auto_onboard=allow_unboarded,
            )
            logger.info(f"MEMORY_INSTANCE_CREATED: user={user_id}")
        except Exception as e:
            logger.error(f"Failed to create memory instance for {user_id}: {e}")
            if allow_unboarded:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to initialize memory system for user {user_id}: {str(e)}",
                )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {user_id} not found or memory setup invalid. Use /onboard endpoint to create user.",
            )
    return memory_instances[user_id]


# --- Backup scheduling ---

# Holds references to fire-and-forget backup tasks so the asyncio event loop
# cannot garbage-collect them mid-flight (see Python docs for
# asyncio.create_task -- "important" note about losing task references).
_background_tasks: Set[asyncio.Task] = set()


def _spawn_background(coro) -> asyncio.Task:
    """Create a task, track it, and auto-remove it from the set on completion."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def backup_user(user_id: str) -> None:
    """Run a backup for one user. Safe to call from request handlers."""
    try:
        repo_manager.sync_user(user_id)
    except Exception as e:
        logger.error(f"BACKUP_ERROR for {user_id}: {e}")


async def periodic_backup(interval_minutes: int) -> None:
    """Runs on a loop, backing up every active user every `interval_minutes`."""
    while True:
        await asyncio.sleep(interval_minutes * 60)
        try:
            active_users = repo_manager.list_active_users()
            if active_users:
                logger.info(f"PERIODIC_BACKUP: Backing up {len(active_users)} active users...")
                for user_id in active_users:
                    await backup_user(user_id)
        except Exception as e:
            logger.error(f"Periodic backup error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global repo_manager

    logger.info("DiffMem server starting up...")
    repo_manager = RepoManager()

    # Post-commit hook is a best-effort webhook; enabling it costs nothing
    # if the backup backend is a no-op.
    logger.info(f"HOOK_INSTALL: Configuring post-commit webhook to {API_URL}")
    repo_manager.install_post_commit_hook(API_URL, API_KEY)

    interval = backup_interval_minutes()
    backup_task = None
    if interval > 0:
        logger.info(f"BACKUP_SCHEDULER: Starting periodic backup every {interval} minutes")
        backup_task = asyncio.create_task(periodic_backup(interval))
    else:
        logger.info("BACKUP_SCHEDULER: Periodic backup disabled (interval=0)")

    yield

    logger.info("DiffMem server shutting down...")
    if backup_task is not None:
        backup_task.cancel()

    try:
        active_users = repo_manager.list_active_users()
        for user_id in active_users:
            await backup_user(user_id)
    except Exception as e:
        logger.error(f"Final backup error: {e}")


app = FastAPI(
    title="DiffMem Server",
    description="Git-native memory server with agent-based retrieval",
    version="0.4.0",
    lifespan=lifespan,
)

_effective_origins = ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"]
# Starlette silently ignores allow_credentials=True when origin is "*" (per the
# CORS spec). Set it explicitly so the behavior is obvious and matches spec.
_allow_credentials = "*" not in _effective_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_effective_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Onboarding ---

@app.post("/memory/{user_id}/onboard")
async def onboard_user(user_id: str, request: OnboardUserRequest, authenticated: bool = Depends(verify_api_key)):
    """Onboard a new user to the memory system"""
    try:
        user_repo_path = repo_manager.get_user_worktree(user_id)
        result = onboard_new_user(
            user_repo_path, user_id, request.user_info,
            OPENROUTER_API_KEY, DEFAULT_MODEL, request.session_id,
            template=request.template,
        )
        if user_id in memory_instances:
            del memory_instances[user_id]
        await backup_user(user_id)

        if result.get("success"):
            return {
                "status": "success",
                "message": f"User {user_id} successfully onboarded",
                "result": result,
                "metadata": {"timestamp": datetime.now().isoformat()},
            }
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Onboarding failed: {result.get('error', 'Unknown error')}",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Onboarding error for {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Onboarding failed: {str(e)}",
        )


@app.get("/memory/{user_id}/onboard-status")
async def get_onboard_status(user_id: str, authenticated: bool = Depends(verify_api_key)):
    """Check if a user is onboarded"""
    try:
        memory = get_memory_instance(user_id, allow_unboarded=True)
        is_onboarded = memory.is_onboarded()
        return {
            "status": "success",
            "user_id": user_id,
            "onboarded": is_onboarded,
            "message": f"User {user_id} is {'onboarded' if is_onboarded else 'not onboarded'}",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Onboard status check error for {user_id}: {e}")
        return {
            "status": "success",
            "user_id": user_id,
            "onboarded": False,
            "message": f"User {user_id} is not onboarded or error accessing context",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


# --- Context Retrieval ---

@app.post("/memory/{user_id}/context")
async def get_context(user_id: str, request: ContextRequest, authenticated: bool = Depends(verify_api_key)):
    """Git-native agent retrieval. Agent explores git history to build context."""
    memory = get_memory_instance(user_id)
    try:
        context = memory.get_context(
            request.conversation,
            max_tokens=request.max_tokens,
            max_turns=request.max_turns,
            timeout_seconds=request.timeout_seconds,
            baseline_only=request.baseline_only,
        )
        return {
            "status": "success",
            "context": context,
            "metadata": {"user_id": user_id, "timestamp": datetime.now().isoformat()},
        }
    except Exception as e:
        logger.error(f"Context retrieval error for {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Context retrieval failed: {str(e)}",
        )


# --- Read Endpoints ---

@app.get("/memory/{user_id}/user-entity")
async def get_user_entity(user_id: str, authenticated: bool = Depends(verify_api_key)):
    """Get the complete user entity file"""
    memory = get_memory_instance(user_id)
    try:
        entity = memory.get_user_entity()
        return {"status": "success", "entity": entity,
                "metadata": {"user_id": user_id, "timestamp": datetime.now().isoformat()}}
    except Exception as e:
        logger.error(f"User entity retrieval error for {user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"User entity retrieval failed: {str(e)}")


@app.get("/memory/{user_id}/recent-timeline")
async def get_recent_timeline(user_id: str, days_back: int = 30, authenticated: bool = Depends(verify_api_key)):
    """Get recent timeline entries"""
    memory = get_memory_instance(user_id)
    try:
        timeline = memory.get_recent_timeline(days_back)
        return {"status": "success", "timeline": timeline,
                "metadata": {"days_back": days_back, "user_id": user_id, "timestamp": datetime.now().isoformat()}}
    except Exception as e:
        logger.error(f"Timeline retrieval error for {user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Timeline retrieval failed: {str(e)}")


@app.post("/memory/{user_id}/run-command")
async def run_command(user_id: str, request: RunCommandRequest, authenticated: bool = Depends(verify_api_key)):
    """Execute a sandboxed shell command in the user's worktree. Same whitelist and sandboxing as the retrieval agent."""
    memory = get_memory_instance(user_id)
    try:
        output = command_router.run(request.command, str(memory.repo_path))
        match = re.search(r"\[exit:(\d+) \| (\d+)ms\]", output)
        exit_code = int(match.group(1)) if match else 0
        elapsed_ms = int(match.group(2)) if match else 0
        return {"output": output, "exit_code": exit_code, "elapsed_ms": elapsed_ms}
    except Exception as e:
        logger.error(f"Run command error for {user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Command execution failed: {str(e)}")


@app.get("/memory/{user_id}/entity-version")
async def get_entity_version(user_id: str, authenticated: bool = Depends(verify_api_key)):
    """Return the latest commit hash and timestamp for the user's entity file."""
    memory = get_memory_instance(user_id)
    entity_file = f"{user_id}.md"
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H %at", "--", entity_file],
            cwd=str(memory.repo_path),
            capture_output=True, text=True, timeout=10,
        )
        raw = result.stdout.strip()
        if not raw:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No commits found for entity file: {entity_file}",
            )
        commit_hash, unix_ts = raw.split(" ", 1)
        committed_at = datetime.fromtimestamp(int(unix_ts)).isoformat()
        return {"commit_hash": commit_hash, "committed_at": committed_at, "entity_path": entity_file}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Entity version error for {user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Entity version lookup failed: {str(e)}")


# --- Write Endpoints ---

@app.post("/memory/{user_id}/process-session")
async def process_session(user_id: str, request: ProcessSessionRequest, authenticated: bool = Depends(verify_api_key)):
    """Process session transcript and stage changes"""
    memory = get_memory_instance(user_id)
    try:
        memory.process_session(request.memory_input, request.session_id, request.session_date)
        return {"status": "success", "session_id": request.session_id,
                "message": "Session processed and staged for commit",
                "metadata": {"user_id": user_id, "timestamp": datetime.now().isoformat()}}
    except Exception as e:
        logger.error(f"Session processing error for {user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Session processing failed: {str(e)}")


@app.post("/memory/{user_id}/commit-session")
async def commit_session(user_id: str, request: CommitSessionRequest, authenticated: bool = Depends(verify_api_key)):
    """Commit staged changes for a session. Backup runs out-of-band."""
    memory = get_memory_instance(user_id)
    try:
        memory.commit_session(request.session_id)
        # Fire-and-forget backup; never blocks the response.
        _spawn_background(backup_user(user_id))
        return {"status": "success", "session_id": request.session_id,
                "message": "Session committed",
                "metadata": {"user_id": user_id, "timestamp": datetime.now().isoformat()}}
    except Exception as e:
        logger.error(f"Session commit error for {user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Session commit failed: {str(e)}")


@app.post("/memory/{user_id}/process-and-commit")
async def process_and_commit_session(user_id: str, request: ProcessSessionRequest, authenticated: bool = Depends(verify_api_key)):
    """Process and immediately commit a session. Backup runs out-of-band."""
    memory = get_memory_instance(user_id)
    try:
        memory.process_and_commit_session(request.memory_input, request.session_id, request.session_date)
        # Fire-and-forget backup; never blocks the response.
        _spawn_background(backup_user(user_id))
        return {"status": "success", "session_id": request.session_id,
                "message": "Session processed and committed",
                "metadata": {"user_id": user_id, "timestamp": datetime.now().isoformat()}}
    except Exception as e:
        logger.error(f"Session processing and commit error for {user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Session processing and commit failed: {str(e)}")


# --- Deletion ---

@app.delete("/memory/{user_id}")
async def delete_user(user_id: str, authenticated: bool = Depends(verify_api_key)):
    if user_id in memory_instances:
        del memory_instances[user_id]
    try:
        repo_manager.wipe_user(user_id)
        logger.info(f"DELETE_USER: user={user_id} permanently deleted")
    except Exception as e:
        logger.warning(f"DELETE_USER: wipe raised (idempotent, ignoring): {e}")
    return {
        "status": "success",
        "message": f"User {user_id} and all associated data permanently deleted",
        "metadata": {"user_id": user_id, "timestamp": datetime.now().isoformat()},
    }


# --- Utility Endpoints ---

@app.post("/memory/{user_id}/webhook/post-commit")
async def post_commit_webhook(user_id: str, authenticated: bool = Depends(verify_api_key)):
    """Webhook triggered by git post-commit hook. Triggers an out-of-band backup."""
    logger.info(f"WEBHOOK_RECEIVED: post-commit for {user_id}")
    try:
        await backup_user(user_id)
        return {"status": "success", "message": "Post-commit backup completed",
                "metadata": {"user_id": user_id, "timestamp": datetime.now().isoformat()}}
    except Exception as e:
        logger.error(f"WEBHOOK_ERROR for {user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Post-commit webhook failed: {str(e)}")


@app.get("/memory/{user_id}/status")
async def get_repo_status(user_id: str, authenticated: bool = Depends(verify_api_key)):
    """Get repository status and statistics"""
    memory = get_memory_instance(user_id, allow_unboarded=True)
    try:
        repo_status = memory.get_repo_status()
        return {"status": "success", "repo_status": repo_status,
                "metadata": {"user_id": user_id, "timestamp": datetime.now().isoformat()}}
    except Exception as e:
        logger.error(f"Status retrieval error for {user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Status retrieval failed: {str(e)}")


@app.get("/memory/{user_id}/validate")
async def validate_setup(user_id: str, authenticated: bool = Depends(verify_api_key)):
    """Validate memory setup for user"""
    memory = get_memory_instance(user_id, allow_unboarded=True)
    try:
        validation = memory.validate_setup()
        return {"status": "success", "validation": validation,
                "metadata": {"user_id": user_id, "timestamp": datetime.now().isoformat()}}
    except Exception as e:
        logger.error(f"Validation error for {user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Validation failed: {str(e)}")


# --- Server Management ---

@app.post("/server/sync")
async def manual_sync(authenticated: bool = Depends(verify_api_key)):
    """Manually trigger a backup for all active users."""
    try:
        active_users = repo_manager.list_active_users()
        synced = []
        for user_id in active_users:
            await backup_user(user_id)
            synced.append(user_id)
        return {"status": "success", "message": f"Backed up {len(synced)} active users",
                "synced_users": synced, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Manual sync error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Sync failed: {str(e)}")


@app.get("/server/users")
async def list_users(authenticated: bool = Depends(verify_api_key)):
    """List active users with mounted worktrees"""
    try:
        active_users = repo_manager.list_active_users()
        return {"status": "success", "users": active_users, "count": len(active_users),
                "note": "Lists currently mounted active contexts only",
                "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"User listing error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"User listing failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Liveness + backend info. Safe to call unauthenticated for reverse-proxy healthchecks."""
    backup_name = repo_manager.backup.name if repo_manager else "unknown"
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.4.0",
        "architecture": "git_native_agent",
        "active_contexts": len(memory_instances),
        "storage_backend": "local",
        "backup_backend": backup_name,
    }


@app.get("/")
async def root():
    return {
        "service": "DiffMem Server",
        "version": "0.4.0",
        "description": "Git-native memory server with agent-based retrieval",
        "docs": "/docs",
        "health": "/health",
    }


def main():
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("diffmem.server:app", host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
