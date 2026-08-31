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
from typing import Any, Dict, List, Optional, Tuple
import json

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


# --- semantic-index list normalization ----------------------------------------
#
# WHY THIS EXISTS: the semantic index (frontmatter on entity files) is produced
# by LLMs. When asked for a JSON/YAML object whose fields are "a list of cue
# strings", models occasionally return NESTED lists, e.g.
#     hard_cues: ["a", "b", ["c", "d"]]
# Every downstream consumer joins or hashes these fields assuming flat lists
# of strings (",".join in the redistribute/link report builders, set() in the
# dedupe prefilter) and crashes with TypeError — which took down ~80% of
# consolidation runs in ChatBarry production before `redistribute`/`link` were
# disabled there. Normalizing here, at the shared read/write choke points,
# repairs already-poisoned stores (read side) and stops new nesting from
# entering the store (write side) without touching each consumer.

#: Fields of the semantic index that are contractually flat string lists.
SI_LIST_FIELDS = (
    "hard_cues",
    "soft_cues",
    "emotional_cues",
    "aliases",
    "related_entities",
)


def flatten_str_list(values: Any) -> List[str]:
    """Deep-flatten an arbitrarily nested list into flat `List[str]`.

    Scalars inside the structure (e.g. a bare int cue) are stringified;
    ``None`` entries and empty strings are dropped; duplicates are removed
    preserving first-seen order. Non-list scalars at the top level are
    treated as a single-element list (``"x" -> ["x"]``) — LLM outputs are
    not trusted to keep the shape stable.
    """
    out: List[str] = []
    seen: set = set()

    def _walk(v: Any) -> None:
        if v is None:
            return
        if isinstance(v, (list, tuple, set)):
            for item in v:
                _walk(item)
            return
        if isinstance(v, dict):
            # A dict where a list was expected (another LLM shape slip):
            # stringify deterministically so the value survives, flattens no further.
            s = json.dumps(v, sort_keys=True, ensure_ascii=False)
        else:
            s = v if isinstance(v, str) else str(v)
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)

    _walk(values)
    return out


def normalize_semantic_index(si: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return `si` with every :data:`SI_LIST_FIELDS` entry flattened to a flat
    string list (see module notes for why). Mutates and returns the same dict
    for in-place callers; non-dict input is replaced with an empty dict.
    Unknown fields are passed through untouched."""
    if not isinstance(si, dict):
        return {}
    for field in SI_LIST_FIELDS:
        if field in si:
            si[field] = flatten_str_list(si[field])
    return si
