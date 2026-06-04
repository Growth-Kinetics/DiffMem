# CAPABILITY: Out-of-band consolidator over a DiffMem user worktree.
# INPUTS: repo_path (worktree), user_id, OpenRouter key, model.
# OUTPUTS: Three tools (dedupe, redistribute, link), each producing
#          consolidate:-prefixed git commits.
# CONSTRAINTS: Runs in the same _writer_pool as writes (see ADR-D001).
#              Acquires .diffmem/consolidator.lock before any mutation.
#              M1: scaffolding only — public methods are not_implemented stubs.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI

from .lock import ConsolidatorLock

logger = logging.getLogger(__name__)


class ConsolidatorAgent:
    """Repair pass over a user's memory: dedupe, redistribute, link.

    Mirrors WriterAgent's __init__ shape. Tool methods (run_dedupe,
    run_redistribute, run_link) are stubs in M1; M2/M3/M4 implement them.
    """

    def __init__(
        self,
        repo_path: str,
        user_id: str,
        openrouter_api_key: str,
        model: Optional[str] = None,
    ) -> None:
        if not model:
            raise ValueError("model must be set via argument or DEFAULT_MODEL env var")

        self.repo_path = Path(repo_path)
        self.user_id = user_id
        self.user_path = self.repo_path
        self.model = model
        self.prompts_path = Path(__file__).parent / "prompts"

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )

        self.logger = logger
        self.logger.info(
            "CONSOLIDATOR_INIT: repo=%s user=%s model=%s",
            self.repo_path,
            self.user_id,
            self.model,
        )

    # --- public tool surface (stubs in M1) ------------------------------------

    def run_dedupe(self) -> Dict[str, Any]:
        """STUB (M2): identify and merge duplicate entities."""
        self.logger.info("CONSOLIDATOR_DEDUPE_STUB: not_implemented")
        return {
            "status": "not_implemented",
            "tool": "dedupe",
            "commits": [],
            "summary": "run_dedupe is a stub; implemented in M2.",
        }

    def run_redistribute(self, soft_cap_tokens: int = 32000) -> Dict[str, Any]:
        """STUB (M3): redistribute over-large entities and extract orphan themes."""
        self.logger.info(
            "CONSOLIDATOR_REDISTRIBUTE_STUB: not_implemented soft_cap_tokens=%d",
            soft_cap_tokens,
        )
        return {
            "status": "not_implemented",
            "tool": "redistribute",
            "commits": [],
            "summary": f"run_redistribute is a stub; implemented in M3. soft_cap_tokens={soft_cap_tokens}",
        }

    def run_link(self, window: int = 3) -> Dict[str, Any]:
        """STUB (M4): mine commit co-occurrence and weave Obsidian wikilinks."""
        self.logger.info("CONSOLIDATOR_LINK_STUB: not_implemented window=%d", window)
        return {
            "status": "not_implemented",
            "tool": "link",
            "commits": [],
            "summary": f"run_link is a stub; implemented in M4. window={window}",
        }

    # --- helpers --------------------------------------------------------------

    def _lock(self) -> ConsolidatorLock:
        """Returns a ConsolidatorLock context manager for this worktree."""
        return ConsolidatorLock(self.repo_path)
