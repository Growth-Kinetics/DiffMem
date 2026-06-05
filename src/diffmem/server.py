import asyncio
import os
import re
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Set
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, status, Depends
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
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")

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
from .executor import ConsolidatePayload, TaskExecutor, WritePayload, build_executor
from .ontology.loader import load_ontology
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
    callback_url: Optional[str] = Field(None, description="Optional URL to POST job result to on completion. Best-effort, never retried.")


class CommitSessionRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier to commit")
    callback_url: Optional[str] = Field(None, description="Optional URL to POST job result to on completion. Best-effort, never retried.")


class OnboardUserRequest(BaseModel):
    user_info: str = Field(..., description="Raw information dump about the user")
    session_id: Optional[str] = Field(None, description="Optional session ID for tracking")
    template: Optional[str] = Field(None, description="Pre-filled user entity markdown. If provided, bypasses LLM entity generation.")


class RunCommandRequest(BaseModel):
    command: str = Field(..., description="Sandboxed shell command to execute")


class ConsolidateRequest(BaseModel):
    tools: Optional[List[str]] = Field(
        default=None,
        description="Subset of ['dedupe','redistribute','link']. Order ignored; executed dedupe \u2192 redistribute \u2192 link. Default: all three.",
    )
    window: int = Field(default=3, description="Commit window for the link tool.")
    soft_cap_tokens: int = Field(
        default=32000,
        description="Token cap above which an entity is considered oversized for redistribute.",
    )
    callback_url: Optional[str] = Field(None, description="Optional URL to POST job result to on completion. Best-effort, never retried.")


class ProcessCommitAndConsolidateRequest(BaseModel):
    memory_input: str = Field(..., description="Session transcript or memory content")
    session_id: str = Field(..., description="Unique session identifier")
    session_date: Optional[str] = Field(None, description="Session date (YYYY-MM-DD)")
    consolidate_tools: Optional[List[str]] = Field(
        default=None,
        description="Subset of ['dedupe','redistribute','link']. Default: all three.",
    )
    window: int = Field(default=3, description="Commit window for the link tool.")
    soft_cap_tokens: int = Field(default=32000, description="Soft cap for the redistribute tool.")
    callback_url: Optional[str] = Field(None, description="Optional URL to POST job result to on completion. Best-effort, never retried.")


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

# Thread pool for blocking writer-agent operations. The writer agent runs
# multi-step git + LLM work that can block for 60-600s on large sessions.
# Running it in a thread keeps the uvicorn event loop responsive so health
# probes and read endpoints are never blocked by an in-flight write.
_writer_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="diffmem-writer")


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

    if not DEFAULT_MODEL:
        raise RuntimeError("DEFAULT_MODEL env var is required. Set it to an OpenRouter model slug.")
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY env var is required.")

    repo_manager = RepoManager()
    app.state.executor = build_executor(_writer_pool)

    # Load and validate the active ontology at startup — fail fast on misconfiguration.
    active_ontology = load_ontology()
    app.state.ontology = active_ontology
    logger.info(f"ONTOLOGY_LOADED: name={active_ontology.name} entity_types={[e['name'] for e in active_ontology.entity_types]}")

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

    _writer_pool.shutdown(wait=False)

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


# --- Executor helper ---

async def _submit_and_respond(
    *,
    executor: TaskExecutor,
    submit_fn: Callable,
    user_id: str,
    work: Optional[Callable[[], dict]],
    payload=None,
    callback_url: Optional[str],
    sync: Optional[bool],
    session_id: Optional[str] = None,
) -> dict:
    """Submit work to the executor and return either a sync success or async queued response.

    Sync path: blocks in a thread pool (not the event loop) until the job completes,
    then returns the pre-M2 compatible response shape.
    Async path: returns job_id immediately; backup fires via the post-commit git hook.

    Args:
        work:    Thunk callable (required for InlineExecutor; ignored by HatchetExecutor).
        payload: Structured WritePayload / ConsolidatePayload (required for HatchetExecutor;
                 ignored by InlineExecutor).  Pass both when the executor type is not known
                 at compile time.
    """
    handle = submit_fn(user_id, work, payload=payload, callback_url=callback_url)
    effective_sync = (not executor.supports_async_api) if sync is None else sync

    if effective_sync:
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None, lambda: executor.wait_for(handle.job_id, timeout=900.0)
            )
        except TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Job {handle.job_id} timed out after 900s. Poll GET /memory/{user_id}/jobs/{handle.job_id} for status.",
            )
        if result.status == "failed":
            raise HTTPException(
                status_code=500,
                detail=f"Job {handle.job_id} failed: {result.error}",
            )
        resp: dict = {
            "status": "success",
            "message": "Job completed",
            "metadata": {"user_id": user_id, "timestamp": datetime.now().isoformat(), "job_id": handle.job_id},
        }
        if session_id:
            resp["session_id"] = session_id
        if result.result and isinstance(result.result, dict):
            resp.update(result.result)
        return resp
    else:
        return {
            "status": "queued",
            "job_id": handle.job_id,
            "submitted_at": handle.submitted_at.isoformat(),
            "metadata": {"user_id": user_id, "poll_url": f"/memory/{user_id}/jobs/{handle.job_id}"},
        }


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
    """Git-native agent retrieval. Runs in thread pool (pull + agent are blocking)."""
    memory = get_memory_instance(user_id)
    try:
        loop = asyncio.get_event_loop()
        context = await loop.run_in_executor(
            _writer_pool,
            lambda: memory.get_context(
                request.conversation,
                max_tokens=request.max_tokens,
                max_turns=request.max_turns,
                timeout_seconds=request.timeout_seconds,
                baseline_only=request.baseline_only,
            )
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
async def process_session(
    user_id: str,
    request: ProcessSessionRequest,
    sync: Optional[bool] = Query(None, description="Force sync (block-until-done) or async (return job_id). Default depends on executor."),
    authenticated: bool = Depends(verify_api_key),
):
    """Process session transcript and stage changes.

    Requires an executor that supports staged writes (InlineExecutor only).
    Use process-and-commit for single-call ingestion; it works with all executors.
    Async mode: backup fires via the post-commit git hook when the job commits.
    """
    executor: TaskExecutor = app.state.executor
    if not executor.supports_staged_writes:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                "process-session / commit-session require an executor that supports "
                "in-process staged writes (EXECUTOR=inline). "
                "Use POST /memory/{user_id}/process-and-commit instead, "
                "which works with all executors."
            ),
        )
    memory = get_memory_instance(user_id)

    def work():
        memory.process_session(request.memory_input, request.session_id, request.session_date)
        return {"session_id": request.session_id, "message": "Session processed and staged for commit"}

    payload = WritePayload(
        user_id=user_id,
        memory_input=request.memory_input,
        session_id=request.session_id,
        session_date=request.session_date,
    )
    resp = await _submit_and_respond(
        executor=executor,
        submit_fn=executor.submit_write,
        user_id=user_id,
        work=work,
        payload=payload,
        callback_url=request.callback_url,
        sync=sync,
        session_id=request.session_id,
    )
    return resp


@app.post("/memory/{user_id}/commit-session")
async def commit_session(
    user_id: str,
    request: CommitSessionRequest,
    sync: Optional[bool] = Query(None, description="Force sync (block-until-done) or async (return job_id). Default depends on executor."),
    authenticated: bool = Depends(verify_api_key),
):
    """Commit staged changes for a session.

    Requires an executor that supports staged writes (InlineExecutor only).
    Use process-and-commit for single-call ingestion; it works with all executors.
    Sync mode: backup fires out-of-band after commit completes.
    Async mode: backup fires via the post-commit git hook when the job commits.
    """
    executor: TaskExecutor = app.state.executor
    if not executor.supports_staged_writes:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                "process-session / commit-session require an executor that supports "
                "in-process staged writes (EXECUTOR=inline). "
                "Use POST /memory/{user_id}/process-and-commit instead, "
                "which works with all executors."
            ),
        )
    memory = get_memory_instance(user_id)

    def work():
        memory.commit_session(request.session_id)
        return {"session_id": request.session_id, "message": "Session committed"}

    # commit-session has no memory_input; empty string is a safe placeholder
    # since this endpoint is guarded to inline-only (supports_staged_writes).
    payload = WritePayload(
        user_id=user_id,
        memory_input="",
        session_id=request.session_id,
        session_date=None,
    )
    resp = await _submit_and_respond(
        executor=executor,
        submit_fn=executor.submit_write,
        user_id=user_id,
        work=work,
        payload=payload,
        callback_url=request.callback_url,
        sync=sync,
        session_id=request.session_id,
    )
    # In sync mode, fire-and-forget backup (idempotent; git hook may also fire).
    if resp.get("status") == "success":
        _spawn_background(backup_user(user_id))
    return resp


@app.post("/memory/{user_id}/process-and-commit")
async def process_and_commit_session(
    user_id: str,
    request: ProcessSessionRequest,
    sync: Optional[bool] = Query(None, description="Force sync (block-until-done) or async (return job_id). Default depends on executor."),
    authenticated: bool = Depends(verify_api_key),
):
    """Process and immediately commit a session.

    Sync mode: backup fires out-of-band after commit completes.
    Async mode: backup fires via the post-commit git hook when the job commits.
    """
    memory = get_memory_instance(user_id)
    executor: TaskExecutor = app.state.executor

    def work():
        memory.process_and_commit_session(request.memory_input, request.session_id, request.session_date)
        return {"session_id": request.session_id, "message": "Session processed and committed"}

    payload = WritePayload(
        user_id=user_id,
        memory_input=request.memory_input,
        session_id=request.session_id,
        session_date=request.session_date,
    )
    resp = await _submit_and_respond(
        executor=executor,
        submit_fn=executor.submit_write,
        user_id=user_id,
        work=work,
        payload=payload,
        callback_url=request.callback_url,
        sync=sync,
        session_id=request.session_id,
    )
    # In sync mode, fire-and-forget backup (idempotent; git hook may also fire).
    if resp.get("status") == "success":
        _spawn_background(backup_user(user_id))
    return resp


# --- Consolidation Endpoints ---

@app.post("/memory/{user_id}/consolidate")
async def consolidate(
    user_id: str,
    request: ConsolidateRequest,
    sync: Optional[bool] = Query(None, description="Force sync (block-until-done) or async (return job_id). Default depends on executor."),
    authenticated: bool = Depends(verify_api_key),
):
    """Run the consolidator over this user's worktree.

    Sync mode: backup fires out-of-band after consolidation completes.
    Async mode: backup fires via the post-commit git hook when the job commits.
    """
    memory = get_memory_instance(user_id)
    executor: TaskExecutor = app.state.executor

    # Validate tool names eagerly so unknown tools return 400, not 500.
    _VALID_TOOLS = {"dedupe", "redistribute", "link"}
    if request.tools is not None:
        unknown = [t for t in request.tools if t not in _VALID_TOOLS]
        if unknown:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"Unknown consolidator tool(s): {unknown}")

    def work():
        result = memory.consolidate(
            tools=request.tools,
            window=request.window,
            soft_cap_tokens=request.soft_cap_tokens,
        )
        return {"consolidate": result}

    payload = ConsolidatePayload(
        user_id=user_id,
        tools=request.tools,
        window=request.window,
        soft_cap_tokens=request.soft_cap_tokens,
    )
    resp = await _submit_and_respond(
        executor=executor,
        submit_fn=executor.submit_consolidate,
        user_id=user_id,
        work=work,
        payload=payload,
        callback_url=request.callback_url,
        sync=sync,
    )
    # In sync mode, fire-and-forget backup.
    if resp.get("status") == "success":
        _spawn_background(backup_user(user_id))
    return resp


@app.post("/memory/{user_id}/process-commit-and-consolidate")
async def process_commit_and_consolidate(
    user_id: str,
    request: ProcessCommitAndConsolidateRequest,
    sync: Optional[bool] = Query(None, description="Force sync (block-until-done) or async (return job_id). Default depends on executor."),
    authenticated: bool = Depends(verify_api_key),
):
    """Process + commit a session, then consolidate. Single sequential thunk.

    Sync mode: backup fires out-of-band after both operations complete.
    Async mode: backup fires via the post-commit git hook when the job commits.
    """
    memory = get_memory_instance(user_id)
    executor: TaskExecutor = app.state.executor

    def work():
        result = memory.process_commit_and_consolidate(
            request.memory_input,
            request.session_id,
            request.session_date,
            consolidate_tools=request.consolidate_tools,
            window=request.window,
            soft_cap_tokens=request.soft_cap_tokens,
        )
        return {"session_id": request.session_id, "consolidate": result.get("consolidate")}

    payload = ConsolidatePayload(
        user_id=user_id,
        memory_input=request.memory_input,
        session_id=request.session_id,
        session_date=request.session_date,
        tools=request.consolidate_tools,
        window=request.window,
        soft_cap_tokens=request.soft_cap_tokens,
    )
    resp = await _submit_and_respond(
        executor=executor,
        submit_fn=executor.submit_consolidate,
        user_id=user_id,
        work=work,
        payload=payload,
        callback_url=request.callback_url,
        sync=sync,
        session_id=request.session_id,
    )
    # In sync mode, fire-and-forget backup.
    if resp.get("status") == "success":
        _spawn_background(backup_user(user_id))
    return resp


@app.get("/memory/{user_id}/jobs/{job_id}")
async def get_job_status(user_id: str, job_id: str, authenticated: bool = Depends(verify_api_key)):
    """Poll for job status. user_id is for API symmetry; not enforced against the job record."""
    executor: TaskExecutor = app.state.executor
    result = executor.get_job(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return {
        "status": "success",
        "job": result.to_dict(),
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
    executor = getattr(app.state, "executor", None)
    executor_type = type(executor).__name__ if executor is not None else "unknown"
    ontology = getattr(app.state, "ontology", None)
    ontology_name = ontology.name if ontology is not None else "unknown"
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.4.0",
        "architecture": "git_native_agent",
        "active_contexts": len(memory_instances),
        "storage_backend": "local",
        "backup_backend": backup_name,
        "executor_type": executor_type,
        "ontology": ontology_name,
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
