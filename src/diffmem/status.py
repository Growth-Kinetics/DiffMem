# CAPABILITY: Status canonicalization — normalizes freeform LLM status prose to a
# closed enum, in code (not the model). This is the fix for the Model Judgment
# Boundary violation where freeform statuses like "done (previously tracked as
# active)" escaped the exact-match drop filter in the followups builder.
# INPUTS:  raw status string + the enum list (from schema['status_enums'][<type>])
# OUTPUTS: canonical enum value (str) or None if no confident mapping
# CONSTRAINTS: pure function, deterministic, no LLM. Code owns the decision.
from __future__ import annotations

import re
from typing import List, Optional

# Synonym map → canonical token. Keys are matched AFTER lowercasing + stripping
# parentheticals + collapsing spaces. Order matters: longer/more-specific
# phrases first (e.g. "in progress" before "pro").
_SYNONYMS = [
    # terminal / done states
    ("completed", "done"),
    ("complete", "done"),
    ("fulfilled", "done"),
    ("delivered", "done"),
    ("shipped", "done"),
    ("finished", "done"),
    ("done", "done"),
    ("closed", "done"),
    ("rejected", "rejected"),       # decisions
    ("superseded", "superseded"),   # decisions
    # cancelled family
    ("cancelled", "cancelled"),
    ("canceled", "cancelled"),
    ("broken", "cancelled"),
    ("abandoned", "cancelled"),
    ("dropped", "cancelled"),
    ("no longer pursued", "cancelled"),
    # active family
    ("in progress", "in_progress"),
    ("in-progress", "in_progress"),
    ("under way", "in_progress"),
    ("underway", "in_progress"),
    ("in review", "in_progress"),
    ("in queue", "in_progress"),
    ("queued", "in_progress"),
    ("planned", "open"),
    ("planning", "open"),
    ("scoping", "open"),
    ("proposed", "proposed"),
    ("upcoming", "open"),
    ("on hold", "blocked"),
    ("blocked", "blocked"),
    ("stuck", "blocked"),
    ("paused", "paused"),
    ("pending", "open"),
    ("open", "open"),
    ("active", "active"),
    # open_item fallback: "active" is not in the open_item enum, so when the
    # primary ("active","active") mapping above is skipped (canonical not in
    # this type's enum) the scan reaches here and resolves to in_progress.
    ("active", "in_progress"),
    ("accepted", "accepted"),       # decisions
    ("decided", "accepted"),
    # person lifecycle
    ("left", "left"),
    # client relationships
    ("active_client", "active_client"),
    ("active client", "active_client"),
    ("prospect", "prospect"),
    ("vendor", "vendor"),
    ("partner", "partner"),
    ("former", "former"),
]


def _normalize(raw: str) -> str:
    """lowercase, strip parentheticals, collapse whitespace."""
    s = raw.lower().strip()
    # remove parenthetical asides: "done (previously tracked as active)" → "done"
    s = re.sub(r"\s*\([^)]*\)", " ", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def canonicalize_status(raw: Optional[str], enum: Optional[List[str]]) -> Optional[str]:
    """Map a freeform status string to a canonical enum value.

    Returns the canonical value if it (or a known synonym) is in `enum`, else None.
    If `enum` is None/empty (ontology declares no status enum for this type),
    returns the normalized string (best-effort, no enforcement).

    Code is the source of truth: the writer prompt constrains the LLM to the enum,
    but this function guarantees done/cancelled items self-evict from filters even
    when the LLM emits freeform phrasing.
    """
    if not raw:
        # No status present — callers decide the default (treat as active/open).
        return None
    norm = _normalize(raw)
    if not norm:
        return None
    if enum is None:
        return norm
    canon_set = {e.lower() for e in enum}
    # 1. direct canonical match
    if norm in canon_set:
        return norm
    # 2. synonym match
    for phrase, canonical in _SYNONYMS:
        if norm == phrase or norm.startswith(phrase + " ") or norm == phrase:
            if canonical in canon_set:
                return canonical
            # synonym maps to a value not in this type's enum; keep scanning
    # 3. substring fallback: does any canonical value appear as a token in norm?
    tokens = set(norm.replace(",", " ").split())
    for c in canon_set:
        if c in tokens:
            return c
    # 4. unmatched freeform — return None so callers can default (never silently
    #    match a terminal state and wrongly drop an active item).
    return None
