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

from .api import DiffMemory, onboard_new_user


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
API_KEY = os.getenv("API_KEY")
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "true").lower() == "true"

security = HTTPBearer(auto_error=False)

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
    """Get or create DiffMemory instance for user"""
    if user_id not in memory_instances:
        try:
            memory_instances[user_id] = DiffMemory(
                str(REPO_PATH), 
                user_id, 
                OPENROUTER_API_KEY,
                DEFAULT_MODEL,
                auto_onboard=allow_unboarded  # Allow unboarded users when requested
            )
            logger.info(f"MEMORY_INSTANCE_CREATED: user={user_id} allow_unboarded={allow_unboarded}")
        except Exception as e:
            logger.error(f"Failed to create memory instance for {user_id}: {e}")
            if allow_unboarded:
                # More specific error for onboarding scenarios
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

def parse_github_repo_slug(github_url):
    """Extract owner/repo from various GitHub URL formats"""
    if github_url.startswith("https://github.com/"):
        return github_url.replace("https://github.com/", "").replace(".git", "")
    elif github_url.startswith("git@github.com:"):
        return github_url.split(":")[1].replace(".git", "")
    else:
        # Assume it's already in owner/repo format
        return github_url.replace(".git", "")
def get_authenticated_url(github_url: str, github_token: str) -> str:
    """Get consistently formatted authenticated GitHub URL"""
    repo_slug = parse_github_repo_slug(github_url)
    # Use the format that works for both fetch and push
    return f"https://{github_token}@github.com/{repo_slug}.git"

async def setup_repository():
    """Setup Git repository with proper remote configuration"""
    repo_path = Path(os.getenv("REPO_PATH", "/app/memory_repo"))
    github_url = os.getenv("GITHUB_REPO_URL")
    github_token = os.getenv("GITHUB_TOKEN")
    branch = os.getenv("GITHUB_BRANCH", "main")
    
    # Initialize repo if it doesn't exist
    if not repo_path.exists():
        repo_path.mkdir(parents=True, exist_ok=True)
    
    try:
        repo = git.Repo(repo_path)
    except git.InvalidGitRepositoryError:
        repo = git.Repo.init(repo_path)
    
    # Configure git settings
    with repo.config_writer() as git_config:
        git_config.set_value("user", "name", "DiffMem")
        git_config.set_value("user", "email", "diffmem@system.local")
        git_config.set_value("http", "postBuffer", "524288000")
        git_config.set_value("http", "version", "HTTP/1.1")
        git_config.set_value("credential", "helper", "")
    
    # Configure remote if GitHub URL is provided
    if github_url and github_token:
        repo_slug = parse_github_repo_slug(github_url)
        
        # Use CONSISTENT authentication format
        auth_url = get_authenticated_url(github_url, github_token)
        
        logger.info(f"Configuring remote for repository: {repo_slug}")
        
        # Add or update remote with authenticated URL
        if "origin" not in [remote.name for remote in repo.remotes]:
            origin = repo.create_remote("origin", auth_url)
        else:
            origin = repo.remotes.origin
            origin.set_url(auth_url)
        
        # Store the authenticated URL in git config
        with repo.config_writer() as git_config:
            git_config.set_value(f'remote "origin"', "url", auth_url)
        
        # Fetch and setup branch
        try:
            logger.info(f"Fetching from origin...")
            origin.fetch()
            
            remote_refs = [ref.name for ref in origin.refs]
            logger.info(f"Available remote refs: {remote_refs}")
            
            if f"origin/{branch}" in remote_refs:
                logger.info(f"Remote branch {branch} exists, checking out...")
                
                if branch not in repo.heads:
                    repo.create_head(branch, f"origin/{branch}")
                
                repo.heads[branch].set_tracking_branch(origin.refs[branch])
                repo.heads[branch].checkout()
                
                # Pull with authenticated URL
                repo.git.pull("origin", branch)
                logger.info(f"Successfully pulled branch {branch}")
                
            else:
                logger.info(f"Remote branch {branch} doesn't exist, creating...")
                
                if branch not in repo.heads:
                    if not repo.heads:
                        readme = repo_path / "README.md"
                        if not readme.exists():
                            readme.write_text("# DiffMem Repository\n\nInitialized by DiffMem system.")
                            repo.index.add([str(readme)])
                            repo.index.commit("Initial commit")
                    
                    if repo.heads:
                        repo.create_head(branch)
                    else:
                        repo.git.checkout("-b", branch)
                
                repo.heads[branch].checkout()
                
                logger.info(f"Pushing new branch {branch} to origin...")
                repo.git.push("--set-upstream", "origin", branch)
                logger.info(f"Successfully created and pushed branch {branch}")
                
        except git.exc.GitCommandError as e:
            if "403" in str(e):
                logger.error(f"Authentication failed (403). Check your PAT permissions: {e}")
                logger.error("Required PAT scopes: repo (full control), workflow (if using Actions)")
                raise
            else:
                logger.warning(f"Git operation failed: {e}")
                raise
                
    else:
        logger.warning("No GitHub URL or token provided, working with local repository only")
    
    return repo

async def sync_to_github(force: bool = False):
    """Push any changes back to GitHub with proper authentication"""
    try:
        repo = Repo(REPO_PATH)
        
        # Ensure we have the authenticated URL for pushing
        if GITHUB_REPO_URL and GITHUB_TOKEN:
            # Use SAME authentication format as setup
            auth_url = get_authenticated_url(GITHUB_REPO_URL, GITHUB_TOKEN)
            
            # Update remote URL with authentication before each push
            origin = repo.remotes.origin
            origin.set_url(auth_url)
            
            # Log without exposing full token
            repo_slug = parse_github_repo_slug(GITHUB_REPO_URL)
            masked_token = f"{GITHUB_TOKEN[:8]}...{GITHUB_TOKEN[-4:]}"
            logger.info(f"REMOTE_URL_SET: https://{masked_token}@github.com/{repo_slug}.git")
        
        # Check for changes
        has_changes = repo.is_dirty() or repo.untracked_files
        
        if has_changes:
            repo.git.add(A=True)
            repo.git.commit(m=f"Auto-sync: {datetime.now().isoformat()}")
            logger.info("CHANGES_COMMITTED: Local changes staged and committed")
            force = True  # We made changes, so push
        
        # Push if forced or if we just committed
        if force:
            current_branch = repo.active_branch.name
            origin.push(refspec=f"{current_branch}:{current_branch}")
            logger.info(f"REPO_SYNCED: Changes pushed to GitHub branch {current_branch}")
        else:
            logger.info("NO_CHANGES: Repository is clean")
        
    except git.exc.GitCommandError as e:
        if "403" in str(e) or "401" in str(e):
            logger.error(f"AUTH_FAILED: {e}")
            logger.error("PAT_CHECK: Verify token has 'repo' scope and hasn't expired")
            logger.error(f"REPO_ACCESS: Confirm access to {GITHUB_REPO_URL}")
        else:
            logger.error(f"GIT_ERROR: {e}")
    except Exception as e:
        logger.error(f"SYNC_ERROR: {e}")

async def pull_latest():
    """Simple pull from GitHub before processing"""
    try:
        repo = Repo(REPO_PATH)
        
        if GITHUB_REPO_URL and GITHUB_TOKEN:
            # Use SAME authentication format
            auth_url = get_authenticated_url(GITHUB_REPO_URL, GITHUB_TOKEN)
            origin = repo.remotes.origin
            origin.set_url(auth_url)
        
        # Simple fetch and pull
        origin.fetch()
        current_branch = repo.active_branch.name
        
        if not (repo.is_dirty() or repo.untracked_files):
            repo.git.pull("origin", current_branch)
            logger.info("PULL_SUCCESS: Latest changes pulled")
        else:
            logger.info("PULL_SKIPPED: Local changes present")
            
    except Exception as e:
        logger.error(f"PULL_ERROR: {e}")
        
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

# Onboarding endpoints
@app.post("/memory/{user_id}/onboard")
async def onboard_user(user_id: str, request: OnboardUserRequest, authenticated: bool = Depends(verify_api_key)):
    """Onboard a new user to the memory system"""
    try:
        # Check if user already exists/is onboarded
        try:
            existing_memory = get_memory_instance(user_id, allow_unboarded=True)
            if existing_memory.is_onboarded():
                return {
                    "status": "error",
                    "message": f"User {user_id} is already onboarded",
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat()
                }
        except:
            # User doesn't exist, which is what we want for onboarding
            pass
        
        # Perform onboarding
        result = onboard_new_user(
            str(REPO_PATH),
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
        await sync_to_github(force=True)
        
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
        return {
            "status": "success",
            "user_id": user_id,
            "onboarded": False,
            "message": f"User {user_id} is not onboarded",
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
        await pull_latest()
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
        await pull_latest()
        memory.commit_session(request.session_id)
        
        # Trigger immediate sync
        await sync_to_github(force=True)
        
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
        await pull_latest()
        memory.process_and_commit_session(
            request.memory_input,
            request.session_id,
            request.session_date
        )
        
        # Trigger immediate sync
        await sync_to_github(force=True)
        
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
    """Manually trigger GitHub sync"""
    try:
        await pull_latest()
        await sync_to_github(force=True)
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
async def list_users(authenticated: bool = Depends(verify_api_key)):
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