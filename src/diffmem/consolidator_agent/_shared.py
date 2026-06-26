# CAPABILITY: Shared helpers used across consolidator tools (dedupe, redistribute, link).
# INPUTS: Worktree path + entity files.
# OUTPUTS: Semantic index extraction, fuzzy text find, master-index rebuild,
#          token estimation, canonical-path helpers.
# CONSTRAINTS: No LLM calls here. Pure utilities. No coupling to WriterAgent.

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..frontmatter import (
    parse_frontmatter,
    merge_frontmatter,
    strip_legacy_semantic_index as _strip_legacy_block,
)

logger = logging.getLogger(__name__)


# --- token estimation ---------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Cheap token estimator: ~4 chars per token. Good enough for cap-checks."""
    return len(text) // 4


# --- canonical paths ----------------------------------------------------------


def relative_entity_path(worktree: Path, file_path: Path) -> str:
    """Return forward-slash path relative to the worktree root."""
    if not file_path.is_absolute():
        file_path = (worktree / file_path).resolve()
    try:
        rel = file_path.relative_to(worktree.resolve())
        return str(rel).replace("\\", "/")
    except ValueError:
        return f"memories/unknown/{file_path.name}"


# --- semantic index extraction ------------------------------------------------


SEMANTIC_INDEX_HEADER = "## SEMANTIC INDEX"  # legacy trailing-block header


def extract_semantic_index(content: str) -> Optional[Dict[str, Any]]:
    """Return the entity's structured descriptor (the 'semantic index').

    Prefers YAML frontmatter (the v2 location); falls back to the legacy
    trailing `## SEMANTIC INDEX` JSON block for files not yet migrated.
    Returns None only when NEITHER is present.
    """
    fm, _ = parse_frontmatter(content)
    if fm is not None:
        return fm
    # Legacy trailing-block fallback.
    if SEMANTIC_INDEX_HEADER not in content:
        return None
    after = content.split(SEMANTIC_INDEX_HEADER, 1)[1]
    json_lines: List[str] = []
    for line in after.splitlines():
        s = line.strip()
        if not s:
            if json_lines:
                break
            continue
        if s.startswith("##"):
            break
        json_lines.append(s)
    if not json_lines:
        return None
    try:
        return json.loads("".join(json_lines))
    except json.JSONDecodeError as e:
        logger.warning("SEMANTIC_INDEX_PARSE_FAIL: err=%s", e)
        return None


def strip_semantic_index(content: str) -> str:
    """Return content with the legacy trailing SEMANTIC INDEX section removed.
    Frontmatter is preserved (it is now the primary metadata location)."""
    return _strip_legacy_block(content)


def write_with_semantic_index(content: str, semantic_index: Dict[str, Any]) -> str:
    """Merge the descriptor `semantic_index` into the file's frontmatter.

    Replaces the legacy trailing-block write: structured fields now live only in
    frontmatter. Any trailing SEMANTIC INDEX block is stripped (migration).
    """
    # `file` is a path computed at read time (scan_entities sets it); never store it.
    updates = {k: v for k, v in semantic_index.items() if k != "file"}
    return merge_frontmatter(content, updates)


# --- index.md scanning --------------------------------------------------------


def scan_entities(
    worktree: Path,
    entity_dirs: List[Path] = None,
) -> List[Dict[str, Any]]:
    """Walk entity directories and return list of dicts:
        {file, path (relative), content, semantic_index, tokens}
    Files without a SEMANTIC INDEX are skipped (logged).

    entity_dirs: explicit list of absolute entity folder paths (from
        OntologyProfile.entity_dirs(worktree)). Defaults to [worktree/"memories"]
        for backwards compatibility with personal-ontology worktrees.
    """
    if entity_dirs is None:
        entity_dirs = [worktree / "memories"]
    roots = [d for d in entity_dirs if d.exists()]
    if not roots:
        return []
    out: List[Dict[str, Any]] = []
    for memories in roots:
        for md in memories.rglob("*.md"):
            rel = md.relative_to(worktree)
            if md.name in {"index.md", "episodes_index.md"}:
                continue
            if "/sessions/" in str(rel).replace("\\", "/"):
                continue
            try:
                content = md.read_text(encoding="utf-8")
            except OSError as e:
                logger.warning("ENTITY_READ_FAIL: path=%s err=%s", rel, e)
                continue
            si = extract_semantic_index(content)
            if si is None:
                logger.info("ENTITY_NO_INDEX: path=%s (skipped)", rel)
                continue
            si["file"] = str(rel).replace("\\", "/")  # enforce canonical path
            out.append(
                {
                    "file": md,
                    "path": str(rel).replace("\\", "/"),
                    "content": content,
                    "semantic_index": si,
                    "tokens": estimate_tokens(content),
                }
            )
    return out


# --- fuzzy text find ----------------------------------------------------------


_WS = re.compile(r"\s+")


def find_text(haystack: str, needle: str, fuzzy: bool = True) -> int:
    """Find `needle` in `haystack`. Exact first; fuzzy whitespace-normalised fallback."""
    if not needle:
        return -1
    pos = haystack.find(needle)
    if pos != -1:
        return pos
    if not fuzzy:
        return -1
    norm_h = _WS.sub(" ", haystack)
    norm_n = _WS.sub(" ", needle)
    nrm_pos = norm_h.find(norm_n)
    if nrm_pos == -1:
        return -1
    # Walk haystack and count significant chars to map back.
    significant = 0
    last_was_ws = False
    for i, ch in enumerate(haystack):
        if ch.isspace():
            if not last_was_ws:
                significant += 1
            last_was_ws = True
        else:
            significant += 1
            last_was_ws = False
        if significant > nrm_pos:
            return i
    return -1


def replace_first(content: str, search_text: str, replacement: str) -> Tuple[str, bool]:
    """Replace first exact occurrence. Returns (new_content, did_replace)."""
    if search_text and search_text in content:
        return content.replace(search_text, replacement, 1), True
    return content, False


# --- memory strength ----------------------------------------------------------


def calculate_memory_strength(number_of_edits: int, last_update: str) -> float:
    """Mirrors WriterAgent._calculate_memory_strength but isolated here so the
    consolidator does not import from the writer module."""
    edit_score = math.log(max(1, number_of_edits)) / math.log(10)
    try:
        if last_update and last_update != "Unknown" and last_update != "New File":
            last_date = datetime.strptime(last_update[:19], "%Y-%m-%d %H:%M:%S")
            days_ago = (datetime.now() - last_date).days
            recency_score = math.exp(-max(0, days_ago) / 30.0)
        else:
            recency_score = 0.1
    except Exception:
        recency_score = 0.1
    return round(edit_score * 0.7 + recency_score * 0.3, 3)


# --- git stats ----------------------------------------------------------------


def get_file_git_stats(repo, repo_path: Path, file_path: Path) -> Dict[str, Any]:
    """Last commit date + number of edits for a tracked file. Tolerates fresh repos."""
    try:
        rel_path = file_path.relative_to(repo_path)
    except ValueError:
        return {"last_update": "Unknown", "number_of_edits": 0}
    try:
        last_commit = repo.git.log("-1", "--format=%ci", str(rel_path))
        last_update = last_commit.strip() if last_commit else "Unknown"
        commit_count = repo.git.rev_list("--count", "HEAD", "--", str(rel_path))
        number_of_edits = int(commit_count.strip()) if commit_count.strip() else 0
    except Exception:
        last_update = "New File"
        number_of_edits = 1
    return {"last_update": last_update, "number_of_edits": number_of_edits}


# --- master index rebuild -----------------------------------------------------


def rebuild_master_index(
    worktree: Path,
    user_id: str,
    repo=None,
    entity_dirs: List[Path] = None,
) -> Path:
    """Scan entity dirs, collect SEMANTIC INDEX blocks, write index.md sorted by
    memory_strength. If `repo` is given (gitpython Repo), augments entries with
    live git stats. Returns the index file path.

    entity_dirs: from OntologyProfile.entity_dirs(worktree). Defaults to
        [worktree/"memories"] for personal-ontology backwards compatibility.
    """
    entries: List[Dict[str, Any]] = []
    for ent in scan_entities(worktree, entity_dirs=entity_dirs):
        si = dict(ent["semantic_index"])
        if repo is not None:
            stats = get_file_git_stats(repo, worktree, ent["file"])
            si["last_update"] = stats["last_update"]
            si["number_of_edits"] = stats["number_of_edits"]
            si["memory_strength"] = calculate_memory_strength(
                stats["number_of_edits"], stats["last_update"]
            )
        entries.append(si)
    entries.sort(key=lambda x: x.get("memory_strength", 0), reverse=True)
    lines = [
        f"# Memory Index for {user_id}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total entities: {len(entries)}",
        "",
        "## Entity Index (by memory strength)",
        "",
    ]
    for e in entries:
        lines.append(f"### {e.get('name', 'Unknown')} ({e.get('type', 'unknown')})")
        lines.append(f"- **File**: `{e.get('file', 'unknown')}`")
        lines.append(
            f"- **Strength**: {e.get('strength', 'Low')} "
            f"(Score: {e.get('memory_strength', 0)})"
        )
        lines.append("```" + json.dumps(e, separators=(",", ":")) + "```")
        lines.append("")
    out_path = worktree / "index.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("MASTER_INDEX_REBUILT: path=%s entities=%d", out_path, len(entries))
    return out_path
