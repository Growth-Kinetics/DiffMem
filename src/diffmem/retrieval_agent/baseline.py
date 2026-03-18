"""
Deterministic baseline loader.

Loads content that is ALWAYS included in context regardless of what
the retrieval agent finds: user entity, recent timeline, and all
[ALWAYS_LOAD] blocks across entity files.

No BM25, no embeddings, no LLM calls -- pure file reads.
"""

import re
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def load_user_entity(worktree_path: str, user_id: str) -> Dict[str, Any]:
    """Load the complete user entity file."""
    user_file = Path(worktree_path) / f"{user_id}.md"

    if not user_file.exists():
        logger.warning(f"BASELINE_USER_MISSING: {user_file}")
        return {
            "source": f"{user_id}.md",
            "type": "user_entity",
            "content": f"# {user_id.title()} (User Entity Not Found)",
            "tokens": 10,
        }

    try:
        content = user_file.read_text(encoding="utf-8")
        return {
            "source": f"{user_id}.md",
            "type": "user_entity",
            "content": content,
            "tokens": _estimate_tokens(content),
        }
    except Exception as e:
        logger.error(f"BASELINE_USER_ERROR: {e}")
        return {
            "source": f"{user_id}.md",
            "type": "user_entity",
            "content": f"# {user_id.title()} (Error: {e})",
            "tokens": 10,
        }


def load_recent_timeline(worktree_path: str, days_back: int = 30,
                          max_files: int = 5,
                          max_chars_per_file: int = 1000) -> List[Dict[str, Any]]:
    """Load recent timeline entries."""
    timeline_path = Path(worktree_path) / "timeline"
    blocks = []

    if not timeline_path.exists():
        return blocks

    cutoff = datetime.now() - timedelta(days=days_back)

    for tf in timeline_path.glob("*.md"):
        try:
            stem = tf.stem
            date_part = stem.split("_")[0] if "_" in stem else stem
            if date_part.count("-") == 1:
                file_date = datetime.strptime(date_part, "%Y-%m")
            elif date_part.count("-") >= 2:
                file_date = datetime.strptime(date_part, "%Y-%m-%d")
            else:
                continue

            if file_date >= cutoff:
                content = tf.read_text(encoding="utf-8")
                if len(content) > max_chars_per_file:
                    content = content[:max_chars_per_file] + "\n... [truncated]"

                blocks.append({
                    "source": f"timeline/{tf.name}",
                    "type": "timeline",
                    "content": content,
                    "date": file_date.isoformat(),
                    "tokens": _estimate_tokens(content),
                })
        except (ValueError, Exception) as e:
            logger.warning(f"BASELINE_TIMELINE_ERROR: {tf.name} {e}")

    blocks.sort(key=lambda x: x["date"], reverse=True)
    return blocks[:max_files]


def load_always_load_blocks(worktree_path: str,
                             user_id: str,
                             max_tokens: int = 3000) -> List[Dict[str, Any]]:
    """
    Load [ALWAYS_LOAD] blocks from entity files in memories/,
    capped to a token budget.

    With hundreds of entities, loading ALL blocks would be enormous.
    We sort by [Strength: High] first and stop when the budget is hit.
    The retrieval agent handles the rest via targeted exploration.
    """
    memories_path = Path(worktree_path) / "memories"
    all_blocks = []

    if not memories_path.exists():
        return []

    user_file_stem = user_id.lower()

    for md_file in memories_path.rglob("*.md"):
        if md_file.stem.lower() == user_file_stem:
            continue
        if "index" in md_file.name.lower():
            continue

        try:
            content = md_file.read_text(encoding="utf-8")

            pattern = r"/START\s*(.*?)/END"
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)

            for block_content in matches:
                if "[ALWAYS_LOAD]" in block_content:
                    header_match = re.search(r"###\s*(.+)", block_content, re.MULTILINE)
                    header = header_match.group(1).strip() if header_match else md_file.stem

                    strength_match = re.search(r"\[Strength:\s*(High|Medium|Low)\]", block_content, re.IGNORECASE)
                    strength = strength_match.group(1).lower() if strength_match else "low"
                    strength_score = {"high": 3, "medium": 2, "low": 1}.get(strength, 0)

                    rel_path = md_file.relative_to(Path(worktree_path))

                    all_blocks.append({
                        "source": str(rel_path),
                        "type": "always_load",
                        "header": header,
                        "content": block_content.strip(),
                        "tokens": _estimate_tokens(block_content),
                        "_strength": strength_score,
                    })
        except Exception as e:
            logger.warning(f"BASELINE_ALWAYS_LOAD_ERROR: {md_file.name} {e}")

    # Sort by strength descending, then take blocks until budget is hit
    all_blocks.sort(key=lambda b: b["_strength"], reverse=True)

    selected = []
    tokens_used = 0
    for block in all_blocks:
        if tokens_used + block["tokens"] > max_tokens:
            if tokens_used == 0:
                selected.append(block)
                tokens_used += block["tokens"]
            break
        selected.append(block)
        tokens_used += block["tokens"]

    for b in selected:
        del b["_strength"]

    logger.info(
        f"BASELINE_ALWAYS_LOAD: selected {len(selected)}/{len(all_blocks)} blocks, "
        f"{tokens_used} tokens (budget: {max_tokens})"
    )
    return selected


def load_always_load_for_entities(worktree_path: str,
                                   entity_stems: List[str],
                                   max_tokens: int = 3000) -> List[Dict[str, Any]]:
    """
    Load [ALWAYS_LOAD] blocks ONLY for specific entities identified by the agent.

    This runs AFTER the agent, as a safety net: the agent decides which entities
    matter, then we load their core blocks to ensure stability.
    """
    memories_path = Path(worktree_path) / "memories"
    blocks = []

    if not memories_path.exists() or not entity_stems:
        return blocks

    target_stems = {s.lower().replace(" ", "_").replace(".", "") for s in entity_stems}

    for md_file in memories_path.rglob("*.md"):
        if md_file.stem.lower().replace(" ", "_").replace(".", "") not in target_stems:
            continue
        if "index" in md_file.name.lower():
            continue

        try:
            content = md_file.read_text(encoding="utf-8")

            pattern = r"/START\s*(.*?)/END"
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)

            for block_content in matches:
                if "[ALWAYS_LOAD]" in block_content:
                    header_match = re.search(r"###\s*(.+)", block_content, re.MULTILINE)
                    header = header_match.group(1).strip() if header_match else md_file.stem
                    rel_path = md_file.relative_to(Path(worktree_path))

                    blocks.append({
                        "source": str(rel_path),
                        "type": "always_load",
                        "header": header,
                        "content": block_content.strip(),
                        "tokens": _estimate_tokens(block_content),
                    })
        except Exception as e:
            logger.warning(f"BASELINE_ALWAYS_LOAD_ERROR: {md_file.name} {e}")

    # Cap to token budget
    selected = []
    tokens_used = 0
    for block in blocks:
        if tokens_used + block["tokens"] > max_tokens and selected:
            break
        selected.append(block)
        tokens_used += block["tokens"]

    logger.info(
        f"ALWAYS_LOAD_FOR_ENTITIES: {len(selected)}/{len(blocks)} blocks "
        f"for {len(target_stems)} entities, {tokens_used} tokens"
    )
    return selected


def load_baseline(worktree_path: str, user_id: str) -> Dict[str, Any]:
    """
    Load the deterministic baseline: just the user entity.

    ALWAYS_LOAD blocks are loaded AFTER the agent runs, scoped to
    entities the agent identified (see load_always_load_for_entities).
    """
    user_entity = load_user_entity(worktree_path, user_id)
    timeline = load_recent_timeline(worktree_path)

    total_tokens = (
        user_entity["tokens"]
        + sum(t["tokens"] for t in timeline)
    )

    logger.info(
        f"BASELINE_LOADED: user_entity={user_entity['tokens']}tok "
        f"timeline={sum(t['tokens'] for t in timeline)}tok ({len(timeline)} files) "
        f"total={total_tokens}tok"
    )

    return {
        "user_entity": user_entity,
        "timeline": timeline,
        "total_tokens": total_tokens,
    }
