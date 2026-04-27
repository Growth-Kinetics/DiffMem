import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from .writer_agent.agent import WriterAgent
from .writer_agent.onboarding_agent import OnboardingAgent
from .retrieval_agent.baseline import load_baseline, load_always_load_for_entities, load_user_entity, load_recent_timeline
from .retrieval_agent.agent import run_retrieval_agent, LLMConfig
from .retrieval_agent.resolver import resolve_pointers

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
        context = memory.get_context(conversation, max_tokens=15000)
        entity = memory.get_user_entity()

        # Write operations
        memory.process_session("Had coffee with mom today...", "session-123")
        memory.commit_session("session-123")
    """

    def __init__(self, repo_path: str, user_id: str, openrouter_api_key: str,
                 model: Optional[str] = None, auto_onboard: bool = False,
                 max_concurrent_llm_calls: int = 8):
        self.repo_path = Path(repo_path)
        self.user_id = user_id
        self.openrouter_api_key = openrouter_api_key
        self.model = model or os.getenv("DEFAULT_MODEL")
        if not self.model:
            raise ValueError("Default model must be provided or set in DEFAULT_MODEL env var")
        self.max_concurrent_llm_calls = max_concurrent_llm_calls

        self.user_path = self.repo_path
        if not self.user_path.exists():
            if auto_onboard:
                logger.info(f"User path not found, auto_onboard enabled: {self.user_path}")
            else:
                raise FileNotFoundError(f"User path not found: {self.user_path}")

        self._writer_agent = None

        logger.info(f"DIFFMEM_INIT: repo={repo_path} user={user_id}")

    @property
    def writer_agent(self) -> WriterAgent:
        if self._writer_agent is None:
            self._writer_agent = WriterAgent(
                str(self.repo_path),
                self.user_id,
                self.openrouter_api_key,
                self.model,
                self.max_concurrent_llm_calls
            )
        return self._writer_agent

    def is_onboarded(self) -> bool:
        required_paths = [
            self.user_path,
            self.user_path / f"{self.user_id}.md",
            self.user_path / "memories"
        ]
        return all(path.exists() for path in required_paths)

    def onboard_user(self, user_info: str, session_id: Optional[str] = None,
                     template: str = None) -> Dict[str, Any]:
        """Onboard a new user by creating initial directory structure and files."""
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

        result = onboarding_agent.onboard_user(user_info, session_id, template=template)

        if result.get('success'):
            self._writer_agent = None

        return result

    # READ OPERATIONS

    def get_context(self, conversation: List[Dict[str, str]],
                    max_tokens: int = 20000,
                    max_turns: int = 6,
                    timeout_seconds: int = 120,
                    baseline_only: bool = False) -> Dict[str, Any]:
        """
        Git-native agent retrieval. The agent explores the git repository
        and builds targeted context for the conversation.

        Args:
            conversation: List of message dicts [{'role': 'user', 'content': '...'}]
            max_tokens: Agent's additional context token budget
            max_turns: Max agent exploration turns
            timeout_seconds: Hard timeout for the agent loop
            baseline_only: If True, skip the agent entirely and return just
                          the user entity + recent timeline. Zero LLM tokens.

        Returns:
            Dict with:
            - user_entity: The user's core profile
            - recent_timeline: Recent timeline entries
            - agent_context: Targeted blocks discovered by the agent
            - always_load_blocks: Core identity blocks for identified entities
            - retrieval_plan: Agent's reasoning and pointer details
            - session_metadata: Token accounting and timing
        """
        if not self.is_onboarded():
            raise ValueError(f"User {self.user_id} has not been onboarded. Call onboard_user() first.")

        baseline = load_baseline(str(self.repo_path), self.user_id)
        baseline_tokens = baseline["total_tokens"]

        if baseline_only:
            return {
                "user_entity": baseline["user_entity"],
                "recent_timeline": baseline["timeline"],
                "agent_context": [],
                "always_load_blocks": [],
                "retrieval_plan": {
                    "synthesis": "Baseline only -- agent skipped",
                    "entities_identified": [],
                    "pointers": [],
                    "agent_turns": 0,
                    "agent_elapsed_ms": 0,
                },
                "session_metadata": {
                    "user_id": self.user_id,
                    "retrieval_version": "baseline_only",
                    "max_tokens": 0,
                    "baseline_tokens": baseline_tokens,
                    "agent_tokens": 0,
                    "always_load_tokens": 0,
                    "total_tokens": baseline_tokens,
                    "agent_ms": 0,
                    "timestamp": datetime.now().isoformat(),
                },
            }

        try:
            plan = run_retrieval_agent(
                worktree_path=str(self.repo_path),
                user_id=self.user_id,
                conversation=conversation,
                max_tokens=max_tokens,
                baseline_tokens=baseline_tokens,
                max_turns=max_turns,
                timeout_seconds=timeout_seconds,
            )

            agent_blocks = resolve_pointers(
                plan=plan,
                worktree_path=str(self.repo_path),
                token_budget=max_tokens,
            )

            agent_tokens = sum(b["tokens"] for b in agent_blocks)
            al_budget = max(1000, max_tokens - agent_tokens)
            always_load = load_always_load_for_entities(
                str(self.repo_path),
                plan.entities_identified,
                max_tokens=al_budget,
            )

            return {
                "user_entity": baseline["user_entity"],
                "recent_timeline": baseline["timeline"],
                "agent_context": agent_blocks,
                "always_load_blocks": always_load,
                "retrieval_plan": {
                    "synthesis": plan.synthesis,
                    "entities_identified": plan.entities_identified,
                    "pointers": [
                        {"type": p.type, "path": p.path, "git_cmd": p.git_cmd,
                         "reason": p.reason, "priority": p.priority, "est_tokens": p.est_tokens}
                        for p in plan.pointers
                    ],
                    "agent_turns": plan.agent_turns,
                    "agent_elapsed_ms": plan.total_elapsed_ms,
                },
                "session_metadata": {
                    "user_id": self.user_id,
                    "retrieval_version": "agent",
                    "max_tokens": max_tokens,
                    "baseline_tokens": baseline_tokens,
                    "agent_tokens": agent_tokens,
                    "always_load_tokens": sum(b["tokens"] for b in always_load),
                    "total_tokens": baseline_tokens + agent_tokens + sum(b["tokens"] for b in always_load),
                    "agent_ms": plan.total_elapsed_ms,
                    "timestamp": datetime.now().isoformat(),
                },
            }

        except Exception as e:
            logger.error(f"CONTEXT_AGENT_FAILED: {e} -- falling back to baseline")
            return {
                "user_entity": baseline["user_entity"],
                "recent_timeline": baseline["timeline"],
                "agent_context": [],
                "always_load_blocks": [],
                "retrieval_plan": {
                    "synthesis": f"Agent failed: {e}",
                    "entities_identified": [],
                    "pointers": [],
                    "agent_turns": 0,
                    "agent_elapsed_ms": 0,
                },
                "session_metadata": {
                    "user_id": self.user_id,
                    "retrieval_version": "fallback",
                    "max_tokens": max_tokens,
                    "baseline_tokens": baseline_tokens,
                    "agent_tokens": 0,
                    "always_load_tokens": 0,
                    "total_tokens": baseline_tokens,
                    "agent_ms": 0,
                    "timestamp": datetime.now().isoformat(),
                },
            }

    def get_user_entity(self) -> Dict[str, Any]:
        """Get the complete user entity file."""
        if not self.is_onboarded():
            raise ValueError(f"User {self.user_id} has not been onboarded. Call onboard_user() first.")
        return load_user_entity(str(self.repo_path), self.user_id)

    def get_recent_timeline(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """Get recent timeline entries."""
        if not self.is_onboarded():
            raise ValueError(f"User {self.user_id} has not been onboarded. Call onboard_user() first.")
        return load_recent_timeline(str(self.repo_path), days_back)

    # WRITE OPERATIONS

    def process_session(self, memory_input: str, session_id: str,
                       session_date: str = None) -> None:
        """
        Process a session transcript and stage memory updates.

        Analyzes input, creates/updates entity files, and stages all changes
        in git working directory. No commit until commit_session() is called.
        """
        if not self.is_onboarded():
            raise ValueError(f"User {self.user_id} has not been onboarded. Call onboard_user() first.")

        if session_date is None:
            session_date = datetime.now().strftime('%Y-%m-%d')

        self.writer_agent.process_session(memory_input, session_id, session_date)

    def commit_session(self, session_id: str) -> None:
        """Commit all staged changes for a session."""
        if not self.is_onboarded():
            raise ValueError(f"User {self.user_id} has not been onboarded. Call onboard_user() first.")
        self.writer_agent.commit_session(session_id)

    def process_and_commit_session(self, memory_input: str, session_id: str,
                                  session_date: str = None) -> None:
        """Convenience method to process and immediately commit a session."""
        self.process_session(memory_input, session_id, session_date)
        self.commit_session(session_id)

    # UTILITY OPERATIONS

    def get_repo_status(self) -> Dict[str, Any]:
        """Get current repository status and statistics."""
        if not self.is_onboarded():
            return {
                'repo_path': str(self.repo_path),
                'user_id': self.user_id,
                'onboarded': False,
                'error': 'User has not been onboarded'
            }

        memories_path = self.user_path / "memories"
        memory_files = list(memories_path.rglob('*.md')) if memories_path.exists() else []

        return {
            'repo_path': str(self.repo_path),
            'user_id': self.user_id,
            'user_path': str(self.user_path),
            'onboarded': True,
            'memory_files_count': len(memory_files),
            'has_timeline': (self.user_path / "timeline").exists(),
            'has_master_index': (self.user_path / "index.md").exists()
        }

    def validate_setup(self) -> Dict[str, Any]:
        """Validate that the memory setup is correct and complete."""
        issues = []
        warnings = []

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

        required_paths = [
            self.user_path,
            self.user_path / f"{self.user_id}.md",
            self.user_path / "memories"
        ]

        for path in required_paths:
            if not path.exists():
                issues.append(f"Missing required path: {path}")

        master_index = self.user_path / "index.md"
        if not master_index.exists():
            warnings.append("No master index found - will be created on first write operation")

        timeline_dir = self.user_path / "timeline"
        if not timeline_dir.exists():
            warnings.append("No timeline directory found - will be created on first timeline entry")

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

    def delete_user(self, repo_manager: Optional["RepoManager"] = None) -> Dict[str, Any]:
        from .repo_manager import RepoManager

        rm = repo_manager or RepoManager()
        rm.wipe_user(self.user_id)
        self._writer_agent = None
        logger.info(f"DELETE_USER: user={self.user_id} permanently deleted")
        return {
            "success": True,
            "user_id": self.user_id,
            "timestamp": datetime.now().isoformat(),
        }


# Convenience functions

def create_memory_interface(repo_path: str, user_id: str,
                          openrouter_api_key: str = None,
                          model: Optional[str] = None,
                          auto_onboard: bool = False) -> DiffMemory:
    """Convenience function to create a DiffMemory interface."""
    if openrouter_api_key is None:
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError("OpenRouter API key must be provided or set in OPENROUTER_API_KEY env var")
    return DiffMemory(repo_path, user_id, openrouter_api_key, model, auto_onboard)


def onboard_new_user(repo_path: str, user_id: str, user_info: str,
                    openrouter_api_key: str = None,
                    model: Optional[str] = None,
                    session_id: str = None,
                    template: str = None) -> Dict[str, Any]:
    """Onboard a completely new user to the memory system."""
    if openrouter_api_key is None:
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError("OpenRouter API key must be provided or set in OPENROUTER_API_KEY env var")
    memory = DiffMemory(repo_path, user_id, openrouter_api_key, model, auto_onboard=True)
    return memory.onboard_user(user_info, session_id, template=template)
