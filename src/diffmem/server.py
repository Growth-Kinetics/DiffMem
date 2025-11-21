# CAPABILITY: FastAPI server for DiffMem - self-contained memory operations
# INPUTS: Pre-configured GitHub repo, user requests via API
# OUTPUTS: Simple REST API wrapping diffmem.api methods
# CONSTRAINTS: Server manages its own repo, minimal API surface

import asyncio
import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import git
from git import Repo
#for local dev
from dotenv import load_dotenv
load_dotenv()
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Configuration from environment
API_URL = os.getenv("API_URL", "http://localhost:8000")
GITHUB_REPO_URL = os.getenv("GITHUB_REPO_URL")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") 
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "google/gemini-2.5-pro")
# Git Paths
STORAGE_PATH = Path(os.getenv("STORAGE_PATH", "/app/diffmem_storage"))
WORKTREE_ROOT = Path(os.getenv("WORKTREE_ROOT", "/app/active_contexts"))
# Configuration
SYNC_INTERVAL_MINUTES = int(os.getenv("SYNC_INTERVAL_MINUTES", "5"))
API_KEY = os.getenv("API_KEY")
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "true").lower() == "true"
security = HTTPBearer(auto_error=False)
#import diffmem relative modules
from .api import DiffMemory, onboard_new_user
from .repo_manager import RepoManager
# Global RepoManager instance
repo_manager: Optional[RepoManager] = None

async def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Verify API key authentication"""
    
    # Skip auth if disabled
    if not REQUIRE_AUTH:
        return True
    
    # Check if API key is configured
    if not API_KEY:
        logger.warning("AUTH_DISABLED: No API_KEY configured")
        return True
    
    # Check credentials
    if not credentials:
        logger.warning("AUTH_MISSING: No authorization header provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify token
    if credentials.credentials != API_KEY:
        logger.warning(f"AUTH_FAILED: Invalid API key attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    logger.debug("AUTH_SUCCESS: Valid API key provided")
    return True


# Global DiffMemory instances (one per user)
# This cache might need careful management if users are inactive for long
memory_instances: Dict[str, DiffMemory] = {}

# Pydantic models
class ContextRequest(BaseModel):
    conversation: List[Dict[str, str]] = Field(..., description="Conversation history")
    depth: str = Field(default="basic", description="Context depth: basic, wide, deep, temporal")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    k: int = Field(default=5, description="Number of results to return")

class ProcessSessionRequest(BaseModel):
    memory_input: str = Field(..., description="Session transcript or memory content")
    session_id: str = Field(..., description="Unique session identifier")
    session_date: Optional[str] = Field(None, description="Session date (YYYY-MM-DD)")

class CommitSessionRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier to commit")

class OnboardUserRequest(BaseModel):
    user_info: str = Field(..., description="Raw information dump about the user")
    session_id: Optional[str] = Field(None, description="Optional session ID for tracking")

def get_memory_instance(user_id: str, allow_unboarded: bool = False) -> DiffMemory:
    """Get or create DiffMemory instance for user, ensuring worktree is mounted."""    
    if user_id not in memory_instances:
        try:
            # 1. Ensure user repo (worktree) is mounted
            user_repo_path = repo_manager.get_user_worktree(user_id)
            logger.info(f"MEMORY_MOUNT: Mounted worktree for {user_id} at {user_repo_path}")
            
            # 2. Initialize DiffMemory with the worktree path
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
    """Push user changes to GitHub via RepoManager"""
    try:
        # Just call the sync method on repo manager

        repo_manager.sync_user(user_id)
        logger.info(f"USER_SYNCED: {user_id}")
    except Exception as e:
        logger.error(f"SYNC_ERROR for {user_id}: {e}")

async def periodic_sync():
    """Periodically sync all active users"""
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
    """Application lifespan management"""
    global repo_manager
    
    # Startup
    logger.info("DiffMem server starting up...")
    
    # Initialize RepoManager
    repo_manager = RepoManager(
        storage_path=str(STORAGE_PATH),
        worktree_root=str(WORKTREE_ROOT),
        github_url=GITHUB_REPO_URL,
        github_token=GITHUB_TOKEN
    )
    logger.info(f"REPO_MANAGER: Initialized (Storage: {STORAGE_PATH})")
    
    # Install global post-commit hook
    logger.info(f"HOOK_INSTALL: Configuring post-commit webhook to {API_URL}")
    repo_manager.install_post_commit_hook(API_URL, API_KEY)
    
    # Start background sync
    sync_task = asyncio.create_task(periodic_sync())
    
    yield
    
    # Shutdown
    logger.info("DiffMem server shutting down...")
    sync_task.cancel()
    
    # Final sync of all active contexts
    try:
        active_users = repo_manager.list_active_users()
        for user_id in active_users:
            await sync_user_to_github(user_id)
    except Exception as e:
        logger.error(f"Final sync error: {e}")

# FastAPI app
app = FastAPI(
    title="DiffMem Server",
    description="Self-contained FastAPI server for DiffMem memory operations",
    version="0.2.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Onboarding endpoints
@app.post("/memory/{user_id}/onboard")
async def onboard_user(user_id: str, request: OnboardUserRequest, authenticated: bool = Depends(verify_api_key)):
    """Onboard a new user to the memory system"""
    try:
        # Ensure worktree exists/is created (RepoManager handles creation)
        user_repo_path = repo_manager.get_user_worktree(user_id)
        
        # Perform onboarding
        result = onboard_new_user(
            user_repo_path, # Pass the worktree path directly
            user_id,
            request.user_info,
            OPENROUTER_API_KEY,
            DEFAULT_MODEL,
            request.session_id
        )
        
        # Clear any cached instances so they get recreated with proper onboarded state
        if user_id in memory_instances:
            del memory_instances[user_id]
        
        # Trigger immediate sync
        await sync_user_to_github(user_id, force=True)
        
        if result.get('success'):
            return {
                "status": "success",
                "message": f"User {user_id} successfully onboarded",
                "result": result,
                "metadata": {
                    "timestamp": datetime.now().isoformat()
                }
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Onboarding failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        logger.error(f"Onboarding error for {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Onboarding failed: {str(e)}"
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
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Onboard status check error for {user_id}: {e}")
        # If we can't get memory instance, they probably aren't onboarded or repo issues
        return {
            "status": "success",
            "user_id": user_id,
            "onboarded": False,
            "message": f"User {user_id} is not onboarded or error accessing context",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Memory read endpoints
@app.post("/memory/{user_id}/context")
async def get_context(user_id: str, request: ContextRequest, authenticated: bool = Depends(verify_api_key)):
    """Get assembled context for a conversation"""
    memory = get_memory_instance(user_id)
    
    try:
        context = memory.get_context(request.conversation, request.depth)
        return {
            "status": "success",
            "context": context,
            "metadata": {
                "depth": request.depth,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Context retrieval error for {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Context retrieval failed: {str(e)}"
        )

@app.post("/memory/{user_id}/search")
async def search_memory(user_id: str, request: SearchRequest, authenticated: bool = Depends(verify_api_key)):
    """Search memory using BM25"""
    memory = get_memory_instance(user_id)
    
    try:
        results = memory.search(request.query, request.k)
        return {
            "status": "success",
            "results": results,
            "metadata": {
                "query": request.query,
                "k": request.k,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Search error for {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@app.post("/memory/{user_id}/orchestrated-search")
async def orchestrated_search(user_id: str, request: ContextRequest, authenticated: bool = Depends(verify_api_key)):
    """LLM-orchestrated search from conversation"""
    memory = get_memory_instance(user_id)
    
    try:
        results = memory.orchestrated_search(request.conversation)
        return {
            "status": "success",
            "results": results,
            "metadata": {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Orchestrated search error for {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Orchestrated search failed: {str(e)}"
        )

@app.get("/memory/{user_id}/user-entity")
async def get_user_entity(user_id: str, authenticated: bool = Depends(verify_api_key)):
    """Get the complete user entity file"""
    memory = get_memory_instance(user_id)
    
    try:
        entity = memory.get_user_entity()
        return {
            "status": "success",
            "entity": entity,
            "metadata": {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"User entity retrieval error for {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"User entity retrieval failed: {str(e)}"
        )

@app.get("/memory/{user_id}/recent-timeline")
async def get_recent_timeline(user_id: str, days_back: int = 30, authenticated: bool = Depends(verify_api_key)):
    """Get recent timeline entries"""
    memory = get_memory_instance(user_id)
    
    try:
        timeline = memory.get_recent_timeline(days_back)
        return {
            "status": "success",
            "timeline": timeline,
            "metadata": {
                "days_back": days_back,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Timeline retrieval error for {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Timeline retrieval failed: {str(e)}"
        )

# Memory write endpoints
@app.post("/memory/{user_id}/process-session")
async def process_session(user_id: str, request: ProcessSessionRequest, authenticated: bool = Depends(verify_api_key)):
    """Process session transcript and stage changes"""
    memory = get_memory_instance(user_id)
    
    try:
        memory.process_session(
            request.memory_input,
            request.session_id,
            request.session_date
        )
        
        return {
            "status": "success",
            "session_id": request.session_id,
            "message": "Session processed and staged for commit",
            "metadata": {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Session processing error for {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Session processing failed: {str(e)}"
        )

@app.post("/memory/{user_id}/commit-session")
async def commit_session(user_id: str, request: CommitSessionRequest, authenticated: bool = Depends(verify_api_key)):
    """Commit staged changes for a session"""
    memory = get_memory_instance(user_id)
    
    try:
        memory.commit_session(request.session_id)
        
        # Trigger immediate sync
        await sync_user_to_github(user_id, force=True)
        
        return {
            "status": "success",
            "session_id": request.session_id,
            "message": "Session committed and synced to GitHub",
            "metadata": {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Session commit error for {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Session commit failed: {str(e)}"
        )

@app.post("/memory/{user_id}/process-and-commit")
async def process_and_commit_session(user_id: str, request: ProcessSessionRequest, authenticated: bool = Depends(verify_api_key)):
    """Process and immediately commit a session"""
    memory = get_memory_instance(user_id)
    
    try:
        memory.process_and_commit_session(
            request.memory_input,
            request.session_id,
            request.session_date
        )
        
        # Trigger immediate sync
        await sync_user_to_github(user_id, force=True)
        
        return {
            "status": "success",
            "session_id": request.session_id,
            "message": "Session processed, committed, and synced to GitHub",
            "metadata": {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Session processing and commit error for {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Session processing and commit failed: {str(e)}"
        )

# Utility endpoints
@app.post("/memory/{user_id}/webhook/post-commit")
async def post_commit_webhook(user_id: str, authenticated: bool = Depends(verify_api_key)):
    """
    Webhook triggered by git post-commit hook.
    Rebuilds indexes and syncs to remote.
    """
    logger.info(f"WEBHOOK_RECEIVED: post-commit for {user_id}")
    memory = get_memory_instance(user_id)
    
    try:
        # Sync to GitHub
        await sync_user_to_github(user_id)
        # Rebuild in-memory indexes to reflect new commit content
        memory.rebuild_index()
        
        return {
            "status": "success",
            "message": "Post-commit actions completed: synced and rebuilt indexes",
            "metadata": {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"WEBHOOK_ERROR for {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Post-commit webhook failed: {str(e)}"
        )

@app.post("/memory/{user_id}/rebuild-index")
async def rebuild_index(user_id: str, authenticated: bool = Depends(verify_api_key)):
    """Force rebuild of BM25 index"""
    memory = get_memory_instance(user_id)
    
    try:
        memory.rebuild_index()
        return {
            "status": "success",
            "message": "Index rebuilt successfully",
            "metadata": {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Index rebuild error for {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Index rebuild failed: {str(e)}"
        )

@app.get("/memory/{user_id}/status")
async def get_repo_status(user_id: str, authenticated: bool = Depends(verify_api_key)):
    """Get repository status and statistics"""
    memory = get_memory_instance(user_id, allow_unboarded=True)
    
    try:
        status = memory.get_repo_status()
        return {
            "status": "success",
            "repo_status": status,
            "metadata": {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Status retrieval error for {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Status retrieval failed: {str(e)}"
        )

@app.get("/memory/{user_id}/validate")
async def validate_setup(user_id: str, authenticated: bool = Depends(verify_api_key)):
    """Validate memory setup for user"""
    memory = get_memory_instance(user_id, allow_unboarded=True)
    
    try:
        validation = memory.validate_setup()
        return {
            "status": "success",
            "validation": validation,
            "metadata": {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Validation error for {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {str(e)}"
        )

# Server management
@app.post("/server/sync")
async def manual_sync(authenticated: bool = Depends(verify_api_key)):
    """Manually trigger global sync (for all active users)"""
    try:
        active_users = repo_manager.list_active_users()
        synced = []
        for user_id in active_users:
            await sync_user_to_github(user_id)
            synced.append(user_id)
            
        return {
            "status": "success",
            "message": f"Synced {len(synced)} active users to GitHub",
            "synced_users": synced,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Manual sync error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sync failed: {str(e)}"
        )

@app.get("/server/users")
async def list_users(authenticated: bool = Depends(verify_api_key)):
    """List active users (with mounted worktrees)"""
    try:
        # In this architecture, 'users' usually refers to active contexts
        # To get all users ever, one would need to query the storage repo branches
        active_users = repo_manager.list_active_users()
        return {
            "status": "success",
            "users": active_users,
            "count": len(active_users),
            "note": "Lists currently mounted active contexts only",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"User listing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"User listing failed: {str(e)}"
        )

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.2.0",
        "architecture": "orphan_branches_worktrees",
        "active_contexts": len(memory_instances),
        "storage_path": str(STORAGE_PATH),
        "github_repo": GITHUB_REPO_URL
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "DiffMem Server",
        "version": "0.2.0",
        "description": "Self-contained FastAPI server for DiffMem memory operations",
        "docs": "/docs",
        "health": "/health",
        "github_repo": GITHUB_REPO_URL
    }

def main():
    """CLI entry point for DiffMem server"""
    import uvicorn
    uvicorn.run(
        "diffmem.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug"
    )

if __name__ == "__main__":
    main()
