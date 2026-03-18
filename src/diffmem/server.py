import asyncio
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
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

API_URL = os.getenv("API_URL", "http://localhost:8000")
GITHUB_REPO_URL = os.getenv("GITHUB_REPO_URL")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "google/gemini-2.5-pro")
STORAGE_PATH = Path(os.getenv("STORAGE_PATH", "/app/diffmem_storage"))
WORKTREE_ROOT = Path(os.getenv("WORKTREE_ROOT", "/app/active_contexts"))
SYNC_INTERVAL_MINUTES = int(os.getenv("SYNC_INTERVAL_MINUTES", "5"))
API_KEY = os.getenv("API_KEY")
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "true").lower() == "true"
security = HTTPBearer(auto_error=False)

from .api import DiffMemory, onboard_new_user
from .repo_manager import RepoManager

repo_manager: Optional[RepoManager] = None


async def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not REQUIRE_AUTH:
        return True
    if not API_KEY:
        logger.warning("AUTH_DISABLED: No API_KEY configured")
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


# Pydantic models
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
                auto_onboard=allow_unboarded
            )
            logger.info(f"MEMORY_INSTANCE_CREATED: user={user_id}")
        except Exception as e:
            logger.error(f"Failed to create memory instance for {user_id}: {e}")
            if allow_unboarded:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to initialize memory system for user {user_id}: {str(e)}"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"User {user_id} not found or memory setup invalid. Use /onboard endpoint to create user."
                )
    return memory_instances[user_id]


async def sync_user_to_github(user_id: str, force: bool = False):
    try:
        repo_manager.sync_user(user_id)
        logger.info(f"USER_SYNCED: {user_id}")
    except Exception as e:
        logger.error(f"SYNC_ERROR for {user_id}: {e}")


async def periodic_sync():
    while True:
        await asyncio.sleep(SYNC_INTERVAL_MINUTES * 60)
        try:
            active_users = repo_manager.list_active_users()
            logger.info(f"PERIODIC_SYNC: Syncing {len(active_users)} active users...")
            for user_id in active_users:
                await sync_user_to_github(user_id)
        except Exception as e:
            logger.error(f"Periodic sync error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global repo_manager

    logger.info("DiffMem server starting up...")
    repo_manager = RepoManager(
        storage_path=str(STORAGE_PATH),
        worktree_root=str(WORKTREE_ROOT),
        github_url=GITHUB_REPO_URL,
        github_token=GITHUB_TOKEN
    )
    logger.info(f"REPO_MANAGER: Initialized (Storage: {STORAGE_PATH})")

    logger.info(f"HOOK_INSTALL: Configuring post-commit webhook to {API_URL}")
    repo_manager.install_post_commit_hook(API_URL, API_KEY)

    sync_task = asyncio.create_task(periodic_sync())

    yield

    logger.info("DiffMem server shutting down...")
    sync_task.cancel()

    try:
        active_users = repo_manager.list_active_users()
        for user_id in active_users:
            await sync_user_to_github(user_id)
    except Exception as e:
        logger.error(f"Final sync error: {e}")


app = FastAPI(
    title="DiffMem Server",
    description="Git-native memory server with agent-based retrieval",
    version="0.3.0",
    lifespan=lifespan
)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "https://difmem.kingbarry.cc,http://192.168.60.108:8000").split(",")
ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
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
            template=request.template
        )
        if user_id in memory_instances:
            del memory_instances[user_id]
        await sync_user_to_github(user_id, force=True)

        if result.get('success'):
            return {"status": "success", "message": f"User {user_id} successfully onboarded",
                    "result": result, "metadata": {"timestamp": datetime.now().isoformat()}}
        else:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Onboarding failed: {result.get('error', 'Unknown error')}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Onboarding error for {user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Onboarding failed: {str(e)}")

@app.get("/memory/{user_id}/onboard-status")
async def get_onboard_status(user_id: str, authenticated: bool = Depends(verify_api_key)):
    """Check if a user is onboarded"""
    try:
        memory = get_memory_instance(user_id, allow_unboarded=True)
        is_onboarded = memory.is_onboarded()
        return {"status": "success", "user_id": user_id, "onboarded": is_onboarded,
                "message": f"User {user_id} is {'onboarded' if is_onboarded else 'not onboarded'}",
                "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Onboard status check error for {user_id}: {e}")
        return {"status": "success", "user_id": user_id, "onboarded": False,
                "message": f"User {user_id} is not onboarded or error accessing context",
                "error": str(e), "timestamp": datetime.now().isoformat()}

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
            "metadata": {"user_id": user_id, "timestamp": datetime.now().isoformat()}
        }
    except Exception as e:
        logger.error(f"Context retrieval error for {user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Context retrieval failed: {str(e)}")

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
    """Commit staged changes for a session"""
    memory = get_memory_instance(user_id)
    try:
        memory.commit_session(request.session_id)
        await sync_user_to_github(user_id, force=True)
        return {"status": "success", "session_id": request.session_id,
                "message": "Session committed and synced",
                "metadata": {"user_id": user_id, "timestamp": datetime.now().isoformat()}}
    except Exception as e:
        logger.error(f"Session commit error for {user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Session commit failed: {str(e)}")

@app.post("/memory/{user_id}/process-and-commit")
async def process_and_commit_session(user_id: str, request: ProcessSessionRequest, authenticated: bool = Depends(verify_api_key)):
    """Process and immediately commit a session"""
    memory = get_memory_instance(user_id)
    try:
        memory.process_and_commit_session(request.memory_input, request.session_id, request.session_date)
        await sync_user_to_github(user_id, force=True)
        return {"status": "success", "session_id": request.session_id,
                "message": "Session processed, committed, and synced",
                "metadata": {"user_id": user_id, "timestamp": datetime.now().isoformat()}}
    except Exception as e:
        logger.error(f"Session processing and commit error for {user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Session processing and commit failed: {str(e)}")

# --- Utility Endpoints ---

@app.post("/memory/{user_id}/webhook/post-commit")
async def post_commit_webhook(user_id: str, authenticated: bool = Depends(verify_api_key)):
    """Webhook triggered by git post-commit hook. Syncs to remote."""
    logger.info(f"WEBHOOK_RECEIVED: post-commit for {user_id}")
    try:
        await sync_user_to_github(user_id)
        return {"status": "success", "message": "Post-commit sync completed",
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
    """Manually trigger global sync for all active users"""
    try:
        active_users = repo_manager.list_active_users()
        synced = []
        for user_id in active_users:
            await sync_user_to_github(user_id)
            synced.append(user_id)
        return {"status": "success", "message": f"Synced {len(synced)} active users",
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
    return {"status": "healthy", "timestamp": datetime.now().isoformat(), "version": "0.3.0",
            "architecture": "git_native_agent", "active_contexts": len(memory_instances),
            "storage_path": str(STORAGE_PATH), "github_repo": GITHUB_REPO_URL}

@app.get("/")
async def root():
    return {"service": "DiffMem Server", "version": "0.3.0",
            "description": "Git-native memory server with agent-based retrieval",
            "docs": "/docs", "health": "/health", "github_repo": GITHUB_REPO_URL}

def main():
    import uvicorn
    uvicorn.run("diffmem.server:app", host="0.0.0.0", port=8000, reload=True, log_level="debug")

if __name__ == "__main__":
    main()
