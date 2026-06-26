# CAPABILITY: Reabsorb tool — folds legacy `entities/commitments/*.md` files into
# their owner entity's `## Open Items` section, then deletes the commitment file.
# This is the v2 migration path: commitments are no longer a top-level entity type;
# deliverables/follow-ups live as in-file Open Items entries inside their owner.
# INPUTS:  worktree (Path) + repo (gitpython) + user_id
# OUTPUTS: Per-batch `consolidate(reabsorb):` commits. Result dict. Idempotent.
# CONSTRAINTS: deterministic (no LLM). Owner resolved via slug match against
# entity files (projects preferred, then people, then the root user entity).
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..status import canonicalize_status

logger = logging.getLogger(__name__)

COMMITMENTS_DIR = "entities/commitments"
OPEN_ITEMS_HEADER = "## Open Items"
COMMIT_PREFIX = "consolidate(reabsorb):"
OPEN_ITEM_ENUM = ["open", "in_progress", "blocked", "done", "cancelled"]
ACTIVE_DEFAULT = "open"  # if a commitment has no recognisable status, assume open


def _parse_commitment(content: str) -> Dict[str, Any]:
    """Extract title, status, assignee, and related-entity slugs from a legacy
    commitment file. Best-effort parsing (LLM-produced formats vary)."""
    title = ""
    m = re.search(r"^#\s*(?:Commitment|Project|Task):\s*(.+)$", content, re.M)
    if m:
        title = m.group(1).strip()

    def _field(label: str) -> str:
        """Match `- **Label:** value` or `- **Label**: value` (colon in or out of bold)."""
        pat = rf"^\s*-\s*\*{{0,2}}\s*(?:{label})\s*\*{{0,2}}\s*[:：]\s*(.+)$"
        mm = re.search(pat, content, re.M | re.I)
        if not mm:
            return ""
        # strip stray markdown bold markers (`**Status:** value` leaves a leading `**`)
        return mm.group(1).strip().lstrip("*").strip().rstrip(".,;")

    status = _field("Status")
    assignee_raw = _field("Assignee|Owner")
    due = _field(r"Due(?:\s*Date)?|Target\s*Date")

    # normalise assignee to a bare slug (strip [[ ]])
    assignee = assignee_raw
    am = re.search(r"\[\[([^\]]+)\]\]", assignee_raw)
    if am:
        assignee = am.group(1).strip()

    # related-entity slugs from wikilinks anywhere in the file
    slugs = re.findall(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]", content)
    seen = set()
    related = [s for s in slugs if not (s in seen or seen.add(s))]
    return {"title": title, "status": status, "assignee": assignee,
            "due": due, "related": related}


def _slug_to_path_candidates(slug: str, entity_dirs: List[Path]) -> List[Path]:
    """Possible file paths for a wikilink slug across entity folders."""
    fname = slug.strip().lower().replace(" ", "_").replace(".md", "") + ".md"
    return [d / fname for d in entity_dirs]


def _resolve_owner(related: List[str], entity_dirs: List[Path], repo_root: Path,
                  user_entity: Path) -> Tuple[Optional[Path], str]:
    """Resolve the owner file for a commitment. Prefer projects, then people,
    then any entity dir, then the root user entity as a last resort."""
    order = entity_dirs  # entity_dirs is already [people, external, projects, decisions] order
    # but prefer projects, then people
    prioritised = []
    for d in entity_dirs:
        if d.name == "projects":
            prioritised.append(d)
    for d in entity_dirs:
        if d.name == "people":
            prioritised.append(d)
    for d in entity_dirs:
        if d not in prioritised:
            prioritised.append(d)
    for slug in related:
        for d in prioritised:
            for cand in _slug_to_path_candidates(slug, [d]):
                if cand.exists():
                    return cand, slug
    return user_entity, "root"


def _append_open_item(owner_path: Path, entry: str) -> bool:
    """Append an Open Items entry to the owner file. Creates the section if absent.
    Returns True if the file was modified."""
    content = owner_path.read_text(encoding="utf-8")
    if OPEN_ITEMS_HEADER not in content:
        new = content.rstrip() + f"\n\n{OPEN_ITEMS_HEADER}\n\n{entry}\n"
        owner_path.write_text(new, encoding="utf-8")
        return True
    head, _, tail = content.partition(OPEN_ITEMS_HEADER)
    after = tail
    # insert immediately after the header (and one blank line)
    insertion = f"\n{entry}"
    # find the end of the header line
    nl = after.find("\n")
    if nl == -1:
        new = head + OPEN_ITEMS_HEADER + insertion + after
    else:
        new = head + OPEN_ITEMS_HEADER + after[:nl] + insertion + after[nl:]
    owner_path.write_text(new, encoding="utf-8")
    return True


def run(
    *,
    worktree: Path,
    repo,
    user_id: str,
    ontology=None,
) -> Dict[str, Any]:
    """Fold every legacy commitment file into its owner's `## Open Items` section.

    Idempotent: if no commitments folder exists or it is empty, returns with
    zero commits. Each folded commitment is deleted (git rm) and produces a
    per-commit `consolidate(reabsorb):` commit.
    """
    commits_dir = worktree / COMMITMENTS_DIR
    if not commits_dir.is_dir():
        return {"status": "ok", "tool": "reabsorb", "commits": [],
                "folded": 0, "orphaned": 0,
                "summary": "no entities/commitments folder — nothing to reabsorb."}

    commit_files = sorted(commits_dir.glob("*.md"))
    if not commit_files:
        return {"status": "ok", "tool": "reabsorb", "commits": [],
                "folded": 0, "orphaned": 0,
                "summary": "commitments folder empty — nothing to reabsorb."}

    entity_dirs = (
        ontology.entity_dirs(worktree) if ontology is not None else [worktree / "memories"]
    )
    user_entity = worktree / f"{user_id}.md"
    if not user_entity.exists():
        # fall back to the first available entity dir's first file if root missing
        user_entity = entity_dirs[0] if entity_dirs else worktree

    folded = 0
    orphaned = 0
    commit_msgs: List[str] = []

    for cf in commit_files:
        try:
            content = cf.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("REABSORB_READ_FAIL: path=%s err=%s", cf, e)
            continue
        parsed = _parse_commitment(content)
        canon = canonicalize_status(parsed["status"], OPEN_ITEM_ENUM) or ACTIVE_DEFAULT
        owner_path, owner_slug = _resolve_owner(
            parsed["related"], entity_dirs, worktree, user_entity
        )
        orphaned_this = owner_path == user_entity
        if orphaned_this:
            orphaned += 1
        title = parsed["title"] or cf.stem
        parts = [f"- **[{canon}]** {title}"]
        if parsed["assignee"]:
            parts.append(f"assignee [[{parsed['assignee']}]]")
        if parsed["due"]:
            parts.append(f"due {parsed['due']}")
        entry = " — ".join(parts)
        try:
            _append_open_item(owner_path, entry)
        except Exception as e:
            logger.warning("REABSORB_APPEND_FAIL: owner=%s err=%s", owner_path, e)
            continue
        rel = str(cf.relative_to(worktree))
        try:
            repo.git.rm(rel)
        except Exception:
            cf.unlink(missing_ok=True)
            repo.git.add("--all")
        msg = f"{COMMIT_PREFIX} {cf.stem} → {owner_path.stem}"
        repo.index.commit(msg)
        commit_msgs.append(msg)
        folded += 1

    return {
        "status": "ok",
        "tool": "reabsorb",
        "commits": commit_msgs,
        "folded": folded,
        "orphaned": orphaned,
        "summary": f"Folded {folded} commitment(s) into owners ({orphaned} orphaned → root).",
    }
