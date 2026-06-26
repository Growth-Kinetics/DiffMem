# CAPABILITY: YAML frontmatter read/merge utilities for entity files.
# Structured metadata lives in a frontmatter block at the top of each file:
#   ---\n<yaml>\n---\n<body>
# This is the primary location for queryable fields (type, status, cues, ...).
# Legacy files may carry a trailing `## SEMANTIC INDEX` JSON block instead;
# helpers here tolerate and migrate that shape gracefully.
# INPUTS:  file text (str)
# OUTPUTS: frontmatter dict, body str, merged text
# CONSTRAINTS: no LLM calls. Pure text/YAML utilities. PyYAML-backed.
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

FRONTMATTER_DELIM = "---"
SEMANTIC_INDEX_HEADER = "## SEMANTIC INDEX"


def parse_frontmatter(content: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Split a markdown file into (frontmatter_dict, body).

    Returns (None, content) when no frontmatter block is present. Tolerates a
    leading BOM / blank lines. Never raises on malformed YAML — returns (None, content).
    """
    s = content.lstrip("\ufeff").lstrip("\n")
    if not s.startswith(FRONTMATTER_DELIM):
        return None, content
    # First line is the opening delimiter.
    lines = s.split("\n")
    if not lines or lines[0].strip() != FRONTMATTER_DELIM:
        return None, content
    fm_lines: list[str] = []
    body_start = None
    for i in range(1, len(lines)):
        if lines[i].strip() == FRONTMATTER_DELIM:
            body_start = i + 1
            break
        fm_lines.append(lines[i])
    if body_start is None:
        # No closing delimiter — treat as no frontmatter.
        return None, content
    raw = "\n".join(fm_lines)
    try:
        fm = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        logger.warning("FRONTMATTER_PARSE_FAIL: err=%s", e)
        return None, content
    if not isinstance(fm, dict):
        # Non-mapping YAML (scalar, list, or empty) → treat as empty frontmatter.
        fm = {}
    body = "\n".join(lines[body_start:])
    return fm, body.lstrip("\n")


def dump_frontmatter(fm: Dict[str, Any]) -> str:
    """Render a frontmatter dict as a `---\\n<yaml>\\n---` block (block style, key order preserved)."""
    cleaned = {k: v for k, v in fm.items() if v is not None}
    body = yaml.safe_dump(cleaned, sort_keys=False, default_flow_style=False, allow_unicode=True)
    return f"{FRONTMATTER_DELIM}\n{body}{FRONTMATTER_DELIM}\n"


def merge_frontmatter(content: str, updates: Dict[str, Any]) -> str:
    """Merge `updates` into the file's frontmatter, preserving existing keys
    not in `updates` and the body verbatim. Creates frontmatter if absent.
    Always strips any legacy trailing `## SEMANTIC INDEX` block first."""
    content = strip_legacy_semantic_index(content)
    fm, body = parse_frontmatter(content)
    if fm is None:
        fm = {}
        body = content  # no frontmatter; whole content is body
    fm.update(updates)
    fm_block = dump_frontmatter(fm)
    body = body.rstrip() + "\n"
    return fm_block + "\n" + body


def strip_legacy_semantic_index(content: str) -> str:
    """Remove a trailing `## SEMANTIC INDEX` JSON block (legacy format).
    Frontmatter is preserved. Returns content unchanged if no block present."""
    if SEMANTIC_INDEX_HEADER not in content:
        return content
    head = content.split(SEMANTIC_INDEX_HEADER, 1)[0]
    return head.rstrip() + "\n"


def has_frontmatter(content: str) -> bool:
    fm, _ = parse_frontmatter(content)
    return fm is not None
