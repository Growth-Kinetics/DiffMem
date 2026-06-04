# CAPABILITY: Out-of-band consolidator over a DiffMem user worktree.
# INPUTS: repo_path (worktree), user_id, OpenRouter key, model.
# OUTPUTS: Three tools (dedupe, redistribute, link), each producing
#          consolidate:-prefixed git commits.
# CONSTRAINTS: Runs in the same _writer_pool as writes (see ADR-D001).
#              Acquires .diffmem/consolidator.lock before any mutation.

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import git
from openai import OpenAI

from . import _dedupe, _redistribute
from .lock import ConsolidatorLock

logger = logging.getLogger(__name__)


# Type alias: an llm_call(prompt, is_json) -> dict|str hook for testability.
LLMCall = Callable[[str, bool], Any]


class ConsolidatorAgent:
    """Repair pass over a user's memory: dedupe, redistribute, link.

    Mirrors WriterAgent's __init__ shape. Use `llm_call` to inject a fake
    LLM for tests; default is the real OpenRouter client.
    """

    def __init__(
        self,
        repo_path: str,
        user_id: str,
        openrouter_api_key: str,
        model: Optional[str] = None,
        llm_call: Optional[LLMCall] = None,
        validate_paths: bool = True,
    ) -> None:
        if not model:
            raise ValueError("model must be set via argument or DEFAULT_MODEL env var")

        self.repo_path = Path(repo_path)
        self.user_id = user_id
        self.user_path = self.repo_path
        self.model = model
        self.prompts_path = Path(__file__).parent / "prompts"

        if validate_paths and not self.repo_path.exists():
            raise FileNotFoundError(f"Worktree not found: {self.repo_path}")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
        self._llm_call_override = llm_call

        self.logger = logger
        self.logger.info(
            "CONSOLIDATOR_INIT: repo=%s user=%s model=%s",
            self.repo_path,
            self.user_id,
            self.model,
        )

    # --- LLM plumbing ---------------------------------------------------------

    def _call_llm(self, prompt: str, is_json: bool = True) -> Any:
        if self._llm_call_override is not None:
            return self._llm_call_override(prompt, is_json)
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.15,
                response_format={"type": "json_object"} if is_json else None,
            )
            content = response.choices[0].message.content
            if is_json:
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    self.logger.warning("LLM_JSON_DECODE_FAIL: err=%s content=%r", e, content[:200])
                    return {}
            return content
        except Exception as e:
            self.logger.error("LLM_CALL_FAIL: err=%s", e)
            return {} if is_json else ""

    # --- internal helpers -----------------------------------------------------

    def _repo(self) -> git.Repo:
        return git.Repo(self.repo_path)

    def _lock(self) -> ConsolidatorLock:
        return ConsolidatorLock(self.repo_path)

    # --- public tool surface --------------------------------------------------

    def run_dedupe(self) -> Dict[str, Any]:
        """Find duplicate entities and merge them under the higher-strength filename."""
        with self._lock():
            repo = self._repo()
            return _dedupe.run(
                worktree=self.repo_path,
                repo=repo,
                prompts_dir=self.prompts_path,
                llm_call=self._call_llm,
                user_id=self.user_id,
            )

    def run_redistribute(self, soft_cap_tokens: int = 32000) -> Dict[str, Any]:
        """Redistribute oversized entities: move attributed sections to subject
        entities and extract orphan themes into new contexts files."""
        with self._lock():
            repo = self._repo()
            return _redistribute.run(
                worktree=self.repo_path,
                repo=repo,
                prompts_dir=self.prompts_path,
                llm_call=self._call_llm,
                user_id=self.user_id,
                soft_cap_tokens=soft_cap_tokens,
            )

    def run_link(self, window: int = 3) -> Dict[str, Any]:
        """STUB (M4): mine commit co-occurrence and weave Obsidian wikilinks."""
        self.logger.info("CONSOLIDATOR_LINK_STUB: not_implemented window=%d", window)
        return {
            "status": "not_implemented",
            "tool": "link",
            "commits": [],
            "summary": f"run_link is a stub; implemented in M4. window={window}",
        }
