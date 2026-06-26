# CAPABILITY: Conformance check — flags entity files whose frontmatter `type`
# does not match their folder's index_type family. Catches the "wrong bucket"
# class of bug (e.g. analysis work mis-filed into a commitments folder, or a
# project file whose frontmatter type says 'human').
# INPUTS:  repo_root (Path) + OntologyProfile
# OUTPUTS: list of violation dicts {path, folder_type, index_type, frontmatter_type}
# CONSTRAINTS: read-only. No mutations, no LLM. Tolerates files lacking frontmatter.
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from .frontmatter import parse_frontmatter

logger = logging.getLogger(__name__)


def check_conformance(repo_root: Path, ontology) -> List[Dict[str, Any]]:
    """Scan every entity file and flag frontmatter/folder type mismatches.

    A file in entities/<type>/ should carry frontmatter `type` equal to that
    folder's `index_type` (people→human, external→company, projects→project,
    decisions→decision). Files with no parseable frontmatter are flagged as
    MISSING_FRONTMATTER; files whose type mismatches are flagged as TYPE_MISMATCH.

    Returns [] for a conformant corpus.
    """
    violations: List[Dict[str, Any]] = []
    for et in ontology.entity_types:
        folder = repo_root / et["folder"]
        if not folder.is_dir():
            continue
        expected = et["index_type"].lower()
        for md in folder.rglob("*.md"):
            if md.name in {"index.md"}:
                continue
            try:
                content = md.read_text(encoding="utf-8")
            except OSError as e:
                logger.warning("CONFORMANCE_READ_FAIL: path=%s err=%s", md, e)
                continue
            fm, _ = parse_frontmatter(content)
            rel = str(md.relative_to(repo_root))
            if fm is None:
                violations.append({
                    "path": rel,
                    "folder_type": et["name"],
                    "index_type": et["index_type"],
                    "frontmatter_type": None,
                    "violation": "MISSING_FRONTMATTER",
                })
                continue
            actual = str(fm.get("type", "")).strip().lower()
            if not actual:
                violations.append({
                    "path": rel,
                    "folder_type": et["name"],
                    "index_type": et["index_type"],
                    "frontmatter_type": None,
                    "violation": "MISSING_TYPE",
                })
            elif actual != expected:
                violations.append({
                    "path": rel,
                    "folder_type": et["name"],
                    "index_type": et["index_type"],
                    "frontmatter_type": actual,
                    "violation": "TYPE_MISMATCH",
                })
    return violations
