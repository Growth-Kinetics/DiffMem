# CAPABILITY: Main API interface for DiffMem - module-driven memory operations
# INPUTS: repo_path, user_id, openrouter_api_key for initialization
# OUTPUTS: Structured memory operations via DiffMemory class
# CONSTRAINTS: No servers/endpoints - direct import and use in chat agents

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from .context_manager.agent import ContextManager, ContextRequest
from .writer_agent.agent import WriterAgent
from .writer_agent.onboarding_agent import OnboardingAgent
from .bm25_indexer.api import build_index, search
from .searcher_agent.agent import orchestrate_query

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiffMemory:
    """
    Main API interface for DiffMem memory operations.
    
    Provides clean read/write access to differential memory without servers or endpoints.
    Can be imported directly into chat agents for immediate use.
    
    Usage:
        memory = DiffMemory("/path/to/repo", "alex", "your-openrouter-key")
        
        # Read operations
        context = memory.get_context(conversation, depth="basic")
        results = memory.search("relationship dynamics")
        
        # Write operations  
        memory.process_session("Had coffee with mom today...", "session-123")
        memory.commit_session("session-123")
    """
    
    def __init__(self, repo_path: str, user_id: str, openrouter_api_key: str, 
                 model: str = "google/gemini-2.5-pro", auto_onboard: bool = False):
        """
        Initialize DiffMemory for a specific user and repository.
        
        Args:
            repo_path: Path to the git repository containing memory files
            user_id: User identifier (must exist in users/ directory unless auto_onboard=True)
            openrouter_api_key: API key for OpenRouter LLM access
            model: Default model for LLM operations
            auto_onboard: If True, will create user structure if it doesn't exist
        """
        self.repo_path = Path(repo_path)
        self.user_id = user_id
        self.openrouter_api_key = openrouter_api_key
        self.model = model
        
        # Validate paths
        self.user_path = self.repo_path / "users" / user_id
        if not self.user_path.exists():
            if auto_onboard:
                logger.info(f"User path not found, auto_onboard enabled: {self.user_path}")
                # Don't raise error, will be handled by onboarding
            else:
                raise FileNotFoundError(f"User path not found: {self.user_path}")
        
        # Initialize components
        self._context_manager = None
        self._writer_agent = None
        self._bm25_index = None
        
        logger.info(f"DIFFMEM_INIT: repo={repo_path} user={user_id}")
    
    @property
    def context_manager(self) -> ContextManager:
        """Lazy initialization of context manager"""
        if self._context_manager is None:
            self._context_manager = ContextManager(
                str(self.repo_path), 
                self.user_id, 
                self.openrouter_api_key,
                self.model
            )
        return self._context_manager
    
    @property
    def writer_agent(self) -> WriterAgent:
        """Lazy initialization of writer agent"""
        if self._writer_agent is None:
            self._writer_agent = WriterAgent(
                str(self.repo_path),
                self.user_id,
                self.openrouter_api_key,
                self.model
            )
        return self._writer_agent
    
    @property
    def bm25_index(self) -> Dict:
        """Lazy initialization of BM25 index"""
        if self._bm25_index is None:
            logger.info("Building BM25 index...")
            self._bm25_index = build_index(str(self.repo_path))
        return self._bm25_index
    
    def rebuild_index(self) -> None:
        """Force rebuild of BM25 index (call after memory updates)"""
        logger.info("Rebuilding BM25 index...")
        self._bm25_index = build_index(str(self.repo_path))
    
    def is_onboarded(self) -> bool:
        """Check if user has been properly onboarded"""
        required_paths = [
            self.user_path,
            self.user_path / f"{self.user_id}.md",
            self.user_path / "memories"
        ]
        return all(path.exists() for path in required_paths)
    
    def onboard_user(self, user_info: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Onboard a new user by creating initial directory structure and files.
        
        Args:
            user_info: Raw information dump about the user
            session_id: Optional session ID for tracking
            
        Returns:
            Dict with onboarding results and metadata
        """
        if self.is_onboarded():
            return {
                'success': False,
                'error': f'User {self.user_id} is already onboarded',
                'user_id': self.user_id,
                'timestamp': datetime.now().isoformat()
            }
        
        onboarding_agent = OnboardingAgent(
            str(self.repo_path),
            self.user_id,
            self.openrouter_api_key,
            self.model
        )
        
        result = onboarding_agent.onboard_user(user_info, session_id)
        
        # Reset components after onboarding
        if result.get('success'):
            self._context_manager = None
            self._writer_agent = None
            self._bm25_index = None
        
        return result
    
    # READ OPERATIONS
    
    def get_context(self, conversation: List[Dict[str, str]], 
                   depth: str = "basic") -> Dict[str, Any]:
        """
        Get assembled context for a conversation.
        
        Args:
            conversation: List of message dicts [{'role': 'user', 'content': '...'}, ...]
            depth: Context depth - "basic", "wide", "deep", or "temporal"
                  - basic: Top entities with ALWAYS_LOAD blocks
                  - wide: Semantic search with ALWAYS_LOAD blocks  
                  - deep: Complete entity files
                  - temporal: Complete files with Git blame history
        
        Returns:
            Dict containing:
            - always_load_blocks: Core memory blocks
            - recent_timeline: Recent timeline entries
            - session_metadata: Context assembly metadata
            - complete_entities: Full entity files (deep/temporal mode)
            - temporal_blame: Git history data (temporal mode)
        """
        if not self.is_onboarded():
            raise ValueError(f"User {self.user_id} has not been onboarded. Call onboard_user() first.")
        
        return self.context_manager.get_session_context(conversation, depth)
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Direct BM25 search over memory blocks.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of dicts with 'score' and 'snippet' keys
        """
        return search(self.bm25_index, query, k)
    
    def orchestrated_search(self, conversation: List[Dict[str, str]], 
                           model: str = None, k: int = 5) -> Dict[str, Any]:
        """
        LLM-orchestrated search from conversation context.
        
        Args:
            conversation: List of message dicts
            model: Override default model for search
            k: Number of results per search term
            
        Returns:
            Dict with 'response', 'snippets', and 'derived_query' keys
        """
        search_model = model or self.model
        return orchestrate_query(conversation, self.bm25_index, search_model, k)
    
    def get_user_entity(self) -> Dict[str, Any]:
        """
        Get the complete user entity file.
        
        Returns:
            Dict with user entity content and metadata
        """
        if not self.is_onboarded():
            raise ValueError(f"User {self.user_id} has not been onboarded. Call onboard_user() first.")
        
        return self.context_manager._load_complete_user_entity()
    
    def get_recent_timeline(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """
        Get recent timeline entries.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            List of timeline entry dicts
        """
        if not self.is_onboarded():
            raise ValueError(f"User {self.user_id} has not been onboarded. Call onboard_user() first.")
        
        return self.context_manager._load_recent_timeline(self.user_path, days_back)
    
    # WRITE OPERATIONS
    
    def process_session(self, memory_input: str, session_id: str, 
                       session_date: str = None) -> None:
        """
        Process a session transcript and stage memory updates.
        
        This analyzes the input, creates/updates entity files, and stages all changes
        in git working directory. No commit is made until commit_session() is called.
        
        Args:
            memory_input: Raw session transcript or memory content
            session_id: Unique session identifier  
            session_date: Date string (YYYY-MM-DD), defaults to today
        """
        if not self.is_onboarded():
            raise ValueError(f"User {self.user_id} has not been onboarded. Call onboard_user() first.")
        
        if session_date is None:
            session_date = datetime.now().strftime('%Y-%m-%d')
            
        self.writer_agent.process_session(memory_input, session_id, session_date)
        
        # Rebuild index after processing to reflect changes
        self.rebuild_index()
    
    def commit_session(self, session_id: str) -> None:
        """
        Commit all staged changes for a session.
        
        Args:
            session_id: Session identifier to commit
        """
        if not self.is_onboarded():
            raise ValueError(f"User {self.user_id} has not been onboarded. Call onboard_user() first.")
        
        self.writer_agent.commit_session(session_id)
    
    def process_and_commit_session(self, memory_input: str, session_id: str,
                                  session_date: str = None) -> None:
        """
        Convenience method to process and immediately commit a session.
        
        Args:
            memory_input: Raw session transcript or memory content
            session_id: Unique session identifier
            session_date: Date string (YYYY-MM-DD), defaults to today
        """
        self.process_session(memory_input, session_id, session_date)
        self.commit_session(session_id)
    
    # UTILITY OPERATIONS
    
    def get_repo_status(self) -> Dict[str, Any]:
        """
        Get current repository status and statistics.
        
        Returns:
            Dict with repo stats, index info, and user metadata
        """
        if not self.is_onboarded():
            return {
                'repo_path': str(self.repo_path),
                'user_id': self.user_id,
                'onboarded': False,
                'error': 'User has not been onboarded'
            }
        
        index_stats = {
            'total_blocks': len(self.bm25_index['corpus']),
            'avg_tokens': sum(len(t) for t in self.bm25_index['tokens']) / len(self.bm25_index['tokens']) if self.bm25_index['tokens'] else 0
        }
        
        # Count memory files
        memories_path = self.user_path / "memories"
        memory_files = list(memories_path.rglob('*.md')) if memories_path.exists() else []
        
        return {
            'repo_path': str(self.repo_path),
            'user_id': self.user_id,
            'user_path': str(self.user_path),
            'onboarded': True,
            'index_stats': index_stats,
            'memory_files_count': len(memory_files),
            'has_timeline': (self.user_path / "timeline").exists(),
            'has_master_index': (self.user_path / "index.md").exists()
        }
    
    def validate_setup(self) -> Dict[str, Any]:
        """
        Validate that the memory setup is correct and complete.
        
        Returns:
            Dict with validation results and any issues found
        """
        issues = []
        warnings = []
        
        # Check if onboarded
        if not self.is_onboarded():
            issues.append("User has not been onboarded")
            return {
                'valid': False,
                'onboarded': False,
                'issues': issues,
                'warnings': warnings,
                'user_id': self.user_id,
                'repo_path': str(self.repo_path)
            }
        
        # Check required paths
        required_paths = [
            self.user_path,
            self.user_path / f"{self.user_id}.md",
            self.user_path / "memories"
        ]
        
        for path in required_paths:
            if not path.exists():
                issues.append(f"Missing required path: {path}")
        
        # Check for master index
        master_index = self.user_path / "index.md"
        if not master_index.exists():
            warnings.append("No master index found - will be created on first write operation")
        
        # Check timeline directory
        timeline_dir = self.user_path / "timeline"
        if not timeline_dir.exists():
            warnings.append("No timeline directory found - will be created on first timeline entry")
        
        # Validate API key
        if not self.openrouter_api_key:
            issues.append("No OpenRouter API key provided")
        
        return {
            'valid': len(issues) == 0,
            'onboarded': True,
            'issues': issues,
            'warnings': warnings,
            'user_id': self.user_id,
            'repo_path': str(self.repo_path)
        }


# Convenience functions for quick access

def create_memory_interface(repo_path: str, user_id: str, 
                          openrouter_api_key: str = None,
                          model: str = "google/gemini-2.5-pro",
                          auto_onboard: bool = False) -> DiffMemory:
    """
    Convenience function to create a DiffMemory interface.
    
    Args:
        repo_path: Path to memory repository
        user_id: User identifier
        openrouter_api_key: API key, or None to use OPENROUTER_API_KEY env var
        model: Default model for operations
        auto_onboard: If True, will allow initialization even if user doesn't exist
        
    Returns:
        Configured DiffMemory instance
    """
    if openrouter_api_key is None:
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError("OpenRouter API key must be provided or set in OPENROUTER_API_KEY env var")
    
    return DiffMemory(repo_path, user_id, openrouter_api_key, model, auto_onboard)


def onboard_new_user(repo_path: str, user_id: str, user_info: str,
                    openrouter_api_key: str = None, 
                    model: str = "google/gemini-2.5-pro",
                    session_id: str = None) -> Dict[str, Any]:
    """
    Onboard a completely new user to the memory system.
    
    Args:
        repo_path: Path to memory repository
        user_id: New user identifier
        user_info: Raw information dump about the user
        openrouter_api_key: API key, or None to use OPENROUTER_API_KEY env var
        model: Model to use for onboarding
        session_id: Optional session ID for tracking
        
    Returns:
        Dict with onboarding results
    """
    if openrouter_api_key is None:
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError("OpenRouter API key must be provided or set in OPENROUTER_API_KEY env var")
    
    # Create memory interface with auto_onboard to allow initialization
    memory = DiffMemory(repo_path, user_id, openrouter_api_key, model, auto_onboard=True)
    
    # Perform onboarding
    return memory.onboard_user(user_info, session_id)


def quick_search(repo_path: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Quick search without full initialization (index-only).
    
    Args:
        repo_path: Path to memory repository
        query: Search query
        k: Number of results
        
    Returns:
        Search results
    """
    index = build_index(repo_path)
    return search(index, query, k) 