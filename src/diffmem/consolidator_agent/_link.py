# CAPABILITY: Link tool — mine commit co-occurrence and weave Obsidian wikilinks.
# INPUTS: ConsolidatorAgent (worktree, repo, LLM, prompts dir).
# OUTPUTS: One `consolidate(link):` commit covering all link edits.
# CONSTRAINTS: Idempotent. Never touches `## SEMANTIC INDEX` blocks.
#              Co-occurrence window is a runtime parameter (default 3 commits).

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from ._shared import (
    estimate_tokens,
    extract_semantic_index,
    replace_first,
    scan_entities,
    strip_semantic_index,
)

logger = logging.getLogger(__name__)


WIKILINK_RE = re.compile(r"\[\[([^\]\|]+)(?:\|[^\]]+)?\]\]")


def _commits_in_window(repo, window: int) -> List[Tuple[str, List[str]]]:
    """Returns [(commit_hash, [file_paths_touched])] for the last `window` commits.

    Filters to .md files under memories/ and timeline/ and the user entity at
    the root. Excludes sessions/.
    """
    out: List[Tuple[str, List[str]]] = []
    try:
        commits = list(repo.iter_commits(max_count=window))
    except Exception as e:
        logger.warning("LINK_COMMITS_FAIL: err=%s", e)
        return out
    for c in commits:
        # parents-aware diff to get files touched in this commit
        if c.parents:
            diffs = c.diff(c.parents[0])
            files = [d.a_path or d.b_path for d in diffs]
        else:
            # root commit — list everything
            files = [b.path for b in c.tree.traverse() if hasattr(b, "path")]
        files = [
            f for f in files
            if f and f.endswith(".md") and "/sessions/" not in f
            and f not in {"repo_guide.md", "index.md", "episodes_index.md"}
        ]
        out.append((c.hexsha, files))
    return out


def _build_cooccurrence(
    commits: List[Tuple[str, List[str]]],
) -> Dict[str, Dict[str, int]]:
    """For each file, count how many times it co-occurred with each other file
    across the windowed commits."""
    co: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _hash, files in commits:
        unique = list(set(files))
        for i, a in enumerate(unique):
            for b in unique[i + 1:]:
                co[a][b] += 1
                co[b][a] += 1
    return co


def _vault_path(rel_path: str) -> str:
    """Strip the .md suffix for Obsidian wikilink. memories/people/maya.md -> memories/people/maya."""
    if rel_path.endswith(".md"):
        return rel_path[:-3]
    return rel_path


def _existing_wikilink_targets(content: str) -> set:
    """Set of vault paths already linked from this file."""
    return {m.group(1) for m in WIKILINK_RE.finditer(content)}


def _load_entity_by_path(worktree: Path, rel_path: str) -> Dict[str, Any]:
    """Load an entity from a path that may live under memories/ OR be the user
    entity at the worktree root. Returns a dict with file/path/content/SI."""
    abs_path = (worktree / rel_path).resolve()
    if not abs_path.exists():
        return {}
    content = abs_path.read_text(encoding="utf-8")
    si = extract_semantic_index(content) or {}
    return {
        "file": abs_path,
        "path": rel_path,
        "content": content,
        "semantic_index": si,
        "tokens": estimate_tokens(content),
    }


def _cooccurrence_block(
    worktree: Path,
    co_for_file: Dict[str, int],
) -> Tuple[str, List[str]]:
    """Render the co-occurring entities block for the link_weave prompt.

    Returns (block_text, list_of_target_rel_paths) sorted by count desc.
    """
    ranked = sorted(co_for_file.items(), key=lambda kv: kv[1], reverse=True)
    lines: List[str] = []
    targets: List[str] = []
    for rel_path, count in ranked:
        ent = _load_entity_by_path(worktree, rel_path)
        if not ent:
            continue
        si = ent.get("semantic_index", {})
        name = si.get("name", Path(rel_path).stem)
        ent_type = si.get("type", "unknown")
        role = si.get("role", "")
        cues = ",".join((si.get("hard_cues") or [])[:5])
        vault = _vault_path(rel_path)
        targets.append(rel_path)
        lines.append(
            f"- count={count} | vault_path={vault} | name={name} | type={ent_type} | role={role} | cues={cues}"
        )
    return ("\n".join(lines) if lines else "(none)"), targets


def _apply_edits(content: str, edits: List[Dict[str, str]]) -> Tuple[str, int]:
    """Apply search-and-replace edits to content. The `## SEMANTIC INDEX`
    header line + the JSON line that follows it are off-limits; everything
    else (including prose AFTER the SI block, e.g. trailing notes that
    accumulated post-write) is editable.

    Returns (new_content, edits_applied_count).
    """
    si_protected = _extract_si_protected_lines(content)
    applied = 0
    new_content = content
    for e in edits:
        s = e.get("search_text", "")
        r = e.get("replacement_text", "")
        if not s or not r or s == r:
            continue
        if "## SEMANTIC INDEX" in s:
            logger.warning("LINK_EDIT_HITS_SI: skipping edit (search_text overlaps SI header)")
            continue
        # Defensive: do not propose to rewrite the SI JSON line itself.
        if any(line in s for line in si_protected if line):
            logger.warning("LINK_EDIT_HITS_SI_JSON: skipping edit")
            continue
        candidate, did = replace_first(new_content, s, r)
        if did:
            new_content = candidate
            applied += 1
    return new_content, applied


def _extract_si_protected_lines(content: str) -> List[str]:
    """Returns the literal lines we must not allow any edit to overlap:
    the `## SEMANTIC INDEX` header AND the JSON line immediately after it."""
    out: List[str] = []
    lines = content.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("## SEMANTIC INDEX"):
            out.append(line)
            # The JSON line is the next non-empty line.
            for j in range(i + 1, len(lines)):
                if lines[j].strip():
                    out.append(lines[j])
                    break
            break
    return out


def run(
    *,
    worktree: Path,
    repo,
    prompts_dir: Path,
    llm_call: Callable[[str, bool], Any],
    user_id: str,
    window: int = 3,
) -> Dict[str, Any]:
    # NOTE: entity_dirs is not needed here — the link tool builds co-occurrence
    # from git log paths (already worktree-relative) and loads files by path,
    # not by scanning a folder tree. It works correctly for any ontology layout.
    commits = _commits_in_window(repo, window)
    cooccurrence = _build_cooccurrence(commits)

    if not cooccurrence:
        return {
            "status": "ok",
            "tool": "link",
            "commits": [],
            "files_touched": 0,
            "links_added": 0,
            "window": window,
            "summary": f"No co-occurrence in last {window} commits.",
        }

    tmpl = (prompts_dir / "link_weave.txt").read_text(encoding="utf-8")
    files_changed = 0
    links_added = 0
    touched_rel: List[str] = []

    # Iterate over files that appear in the cooccurrence map.
    for rel_path, co_for_file in cooccurrence.items():
        ent = _load_entity_by_path(worktree, rel_path)
        if not ent:
            continue
        co_block, targets = _cooccurrence_block(worktree, co_for_file)

        # Filter targets already linked from this file (idempotency).
        existing = _existing_wikilink_targets(ent["content"])
        remaining_targets = [t for t in targets if _vault_path(t) not in existing]
        if not remaining_targets:
            logger.info("LINK_SKIP_ALL_PRESENT: %s", rel_path)
            continue

        prompt = tmpl.format(
            file_path=rel_path,
            file_content=ent["content"],
            cooccurrence_block=co_block,
        )
        resp = llm_call(prompt, True)
        if not isinstance(resp, dict):
            continue
        edits = resp.get("edits", []) or []
        if not edits:
            continue
        new_content, applied = _apply_edits(ent["content"], edits)
        if applied == 0 or new_content == ent["content"]:
            continue

        ent["file"].write_text(new_content, encoding="utf-8")
        rel = str(ent["file"].relative_to(worktree)).replace("\\", "/")
        repo.git.add(rel)
        touched_rel.append(rel)
        files_changed += 1
        links_added += applied
        logger.info("LINK_APPLIED: file=%s links=%d", rel, applied)

    if not touched_rel:
        return {
            "status": "ok",
            "tool": "link",
            "commits": [],
            "files_touched": 0,
            "links_added": 0,
            "window": window,
            "summary": f"No new links proposed (window={window}).",
        }

    msg = f"consolidate(link): wikilinks across {files_changed} files (window={window})"
    repo.index.commit(msg)
    commit_hash = repo.head.commit.hexsha

    return {
        "status": "ok",
        "tool": "link",
        "commits": [commit_hash],
        "files_touched": files_changed,
        "links_added": links_added,
        "window": window,
        "summary": f"Added {links_added} wikilinks across {files_changed} files (window={window}).",
    }
