"""
Pointer resolver: takes a RetrievalPlan from the agent and resolves
each ContentPointer to actual content, respecting token budget.
"""

import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any

from .agent import RetrievalPlan, ContentPointer

logger = logging.getLogger(__name__)

COMMAND_TIMEOUT_SECONDS = 10


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _read_file(worktree: Path, pointer: ContentPointer) -> str:
    file_path = worktree / pointer.path
    if not file_path.exists():
        logger.warning(f"RESOLVE_FILE_MISSING: {pointer.path}")
        return ""
    try:
        return file_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"RESOLVE_FILE_ERROR: {pointer.path} {e}")
        return ""


def _read_file_section(worktree: Path, pointer: ContentPointer) -> str:
    file_path = worktree / pointer.path
    if not file_path.exists():
        logger.warning(f"RESOLVE_SECTION_MISSING: {pointer.path}")
        return ""
    try:
        lines = file_path.read_text(encoding="utf-8").split("\n")
        start = max(0, pointer.line_start - 1)
        end = min(len(lines), pointer.line_end) if pointer.line_end > 0 else len(lines)
        return "\n".join(lines[start:end])
    except Exception as e:
        logger.warning(f"RESOLVE_SECTION_ERROR: {pointer.path} {e}")
        return ""


def _execute_git_command(worktree: Path, pointer: ContentPointer) -> str:
    cmd = pointer.git_cmd
    if not cmd:
        logger.warning(f"RESOLVE_GIT_NO_CMD: {pointer.path}")
        return ""

    if not cmd.startswith("git "):
        logger.warning(f"RESOLVE_GIT_INVALID: {cmd}")
        return ""

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=str(worktree),
            capture_output=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
        output = result.stdout.decode("utf-8", errors="replace")
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            logger.warning(f"RESOLVE_GIT_FAIL: {cmd} rc={result.returncode} stderr={stderr[:200]}")
            return output or ""
        return output
    except subprocess.TimeoutExpired:
        logger.warning(f"RESOLVE_GIT_TIMEOUT: {cmd}")
        return ""
    except Exception as e:
        logger.warning(f"RESOLVE_GIT_ERROR: {cmd} {e}")
        return ""


def resolve_pointers(
    plan: RetrievalPlan,
    worktree_path: str,
    token_budget: int = 15000,
) -> List[Dict[str, Any]]:
    """
    Resolve a RetrievalPlan's pointers into actual content blocks.

    Processes must_include pointers first, then if_budget_allows.
    Stops when token budget is exhausted.

    Returns list of resolved content dicts:
        [{"source": str, "type": str, "content": str, "reason": str, "tokens": int}]
    """
    worktree = Path(worktree_path)
    resolved = []
    tokens_used = 0

    must_include = [p for p in plan.pointers if p.priority == "must_include"]
    optional = [p for p in plan.pointers if p.priority != "must_include"]
    ordered = must_include + optional

    for pointer in ordered:
        if tokens_used >= token_budget:
            logger.info(f"RESOLVER_BUDGET_HIT: {tokens_used} >= {token_budget}, stopping")
            break

        content = ""
        source = pointer.path

        if pointer.type == "file":
            content = _read_file(worktree, pointer)
        elif pointer.type == "file_section":
            content = _read_file_section(worktree, pointer)
            source = f"{pointer.path}:{pointer.line_start}-{pointer.line_end}"
        elif pointer.type in ("git_diff", "git_show", "git_log"):
            content = _execute_git_command(worktree, pointer)
            source = pointer.git_cmd or pointer.path
        else:
            logger.warning(f"RESOLVER_UNKNOWN_TYPE: {pointer.type}")
            continue

        if not content.strip():
            continue

        content_tokens = _estimate_tokens(content)

        if tokens_used + content_tokens > token_budget and pointer.priority != "must_include":
            remaining = token_budget - tokens_used
            if remaining > 100:
                truncated_chars = remaining * 4
                content = content[:truncated_chars] + "\n... [truncated to fit budget]"
                content_tokens = remaining
            else:
                logger.info(f"RESOLVER_SKIP: {source} ({content_tokens} tokens) exceeds remaining budget ({token_budget - tokens_used})")
                continue

        resolved.append({
            "source": source,
            "type": pointer.type,
            "content": content,
            "reason": pointer.reason,
            "tokens": content_tokens,
        })

        tokens_used += content_tokens
        logger.debug(f"RESOLVER_LOADED: {source} {content_tokens}tok (total: {tokens_used}/{token_budget})")

    logger.info(f"RESOLVER_COMPLETE: {len(resolved)} blocks, {tokens_used} tokens used of {token_budget} budget")
    return resolved
