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

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import git
from git import Repo

from .api import DiffMemory


#for local dev
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment
GITHUB_REPO_URL = os.getenv("GITHUB_REPO_URL")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") 
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "google/gemini-2.5-pro")
REPO_PATH = Path(os.getenv("REPO_PATH", "/app/memory_repo"))
SYNC_INTERVAL_MINUTES = int(os.getenv("SYNC_INTERVAL_MINUTES", "5"))

# Global DiffMemory instances (one per user)
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

async def setup_repository():
    """Clone/update the configured GitHub repository"""
    if not GITHUB_REPO_URL or not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_REPO_URL and GITHUB_TOKEN must be set")
    
    auth_url = GITHUB_REPO_URL.replace("https://", f"https://{GITHUB_TOKEN}@")
    
    try:
        if REPO_PATH.exists():
            # Update existing repo
            repo = Repo(REPO_PATH)
            origin = repo.remotes.origin
            origin.fetch()
            repo.git.checkout(GITHUB_BRANCH)
            origin.pull(GITHUB_BRANCH)
            logger.info(f"REPO_UPDATED: {GITHUB_REPO_URL} branch={GITHUB_BRANCH}")
        else:
            # Clone new repo
            REPO_PATH.parent.mkdir(parents=True, exist_ok=True)
            repo = Repo.clone_from(auth_url, REPO_PATH, branch=GITHUB_BRANCH)
            logger.info(f"REPO_CLONED: {GITHUB_REPO_URL} branch={GITHUB_BRANCH}")
            
    except git.exc.GitError as e:
        logger.error(f"Git error: {e}")
        raise RuntimeError(f"Failed to setup repository: {str(e)}")

def get_memory_instance(user_id: str) -> DiffMemory:
    """Get or create DiffMemory instance for user"""
    if user_id not in memory_instances:
        try:
            memory_instances[user_id] = DiffMemory(
                str(REPO_PATH), 
                user_id, 
                OPENROUTER_API_KEY,
                DEFAULT_MODEL
            )
            logger.info(f"MEMORY_INSTANCE_CREATED: user={user_id}")
        except Exception as e:
            logger.error(f"Failed to create memory instance for {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {user_id} not found or memory setup invalid"
            )
    
    return memory_instances[user_id]

async def sync_to_github():
    """Push any changes back to GitHub"""
    try:
        repo = Repo(REPO_PATH)
        if repo.is_dirty() or repo.untracked_files:
            repo.git.add(A=True)
            repo.git.commit(m=f"Auto-sync: {datetime.now().isoformat()}")
            
        origin = repo.remotes.origin
        origin.push()
        logger.info("REPO_SYNCED: Changes pushed to GitHub")
    except git.exc.GitError as e:
        logger.error(f"Sync error: {e}")

async def periodic_sync():
    """Periodically sync repository with GitHub"""
    while True:
        try:
            await sync_to_github()
        except Exception as e:
            logger.error(f"Periodic sync error: {e}")
        
        await asyncio.sleep(SYNC_INTERVAL_MINUTES * 60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("DiffMem server starting up...")
    
    # Setup repository
    await setup_repository()
    
    # Start background sync
    sync_task = asyncio.create_task(periodic_sync())
    
    yield
    
    # Shutdown
    logger.info("DiffMem server shutting down...")
    sync_task.cancel()
    
    # Final sync
    try:
        await sync_to_github()
    except Exception as e:
        logger.error(f"Final sync error: {e}")

# FastAPI app
app = FastAPI(
    title="DiffMem Server",
    description="Self-contained FastAPI server for DiffMem memory operations",
    version="0.1.0",
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

# Memory read endpoints
@app.post("/memory/{user_id}/context")
async def get_context(user_id: str, request: ContextRequest):
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
async def search_memory(user_id: str, request: SearchRequest):
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
async def orchestrated_search(user_id: str, request: ContextRequest):
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
async def get_user_entity(user_id: str):
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
async def get_recent_timeline(user_id: str, days_back: int = 30):
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
async def process_session(user_id: str, request: ProcessSessionRequest):
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
async def commit_session(user_id: str, request: CommitSessionRequest):
    """Commit staged changes for a session"""
    memory = get_memory_instance(user_id)
    
    try:
        memory.commit_session(request.session_id)
        
        # Trigger immediate sync
        await sync_to_github()
        
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
async def process_and_commit_session(user_id: str, request: ProcessSessionRequest):
    """Process and immediately commit a session"""
    memory = get_memory_instance(user_id)
    
    try:
        memory.process_and_commit_session(
            request.memory_input,
            request.session_id,
            request.session_date
        )
        
        # Trigger immediate sync
        await sync_to_github()
        
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
@app.post("/memory/{user_id}/rebuild-index")
async def rebuild_index(user_id: str):
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
async def get_repo_status(user_id: str):
    """Get repository status and statistics"""
    memory = get_memory_instance(user_id)
    
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
async def validate_setup(user_id: str):
    """Validate memory setup for user"""
    memory = get_memory_instance(user_id)
    
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
async def manual_sync():
    """Manually trigger GitHub sync"""
    try:
        await sync_to_github()
        return {
            "status": "success",
            "message": "Repository synced to GitHub",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Manual sync error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sync failed: {str(e)}"
        )

@app.get("/server/users")
async def list_users():
    """List available users in the repository"""
    try:
        users_dir = REPO_PATH / "users"
        if not users_dir.exists():
            return {"status": "success", "users": []}
        
        users = [d.name for d in users_dir.iterdir() if d.is_dir()]
        return {
            "status": "success",
            "users": users,
            "count": len(users),
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
        "version": "0.1.0",
        "repo_path": str(REPO_PATH),
        "active_users": len(memory_instances),
        "github_repo": GITHUB_REPO_URL
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "DiffMem Server",
        "version": "0.1.0",
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