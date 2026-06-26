"""M2 tests: frontmatter engine, status canonicalization, conformance check."""
import json
from pathlib import Path

from diffmem.frontmatter import (
    parse_frontmatter, merge_frontmatter, strip_legacy_semantic_index,
)
from diffmem.status import canonicalize_status
from diffmem.conformance import check_conformance
from diffmem.consolidator_agent._shared import (
    extract_semantic_index, write_with_semantic_index, strip_semantic_index,
)
from diffmem.ontology.loader import load_ontology


# --- frontmatter parse / merge ------------------------------------------------

def test_parse_frontmatter_returns_dict_and_body():
    content = "---\ntype: project\ntitle: Atlas\nstatus: active\n---\n\n# Project: Atlas\nbody.\n"
    fm, body = parse_frontmatter(content)
    assert fm == {"type": "project", "title": "Atlas", "status": "active"}
    assert body.startswith("# Project: Atlas")

def test_parse_frontmatter_none_when_absent():
    fm, body = parse_frontmatter("# just a body\nno frontmatter")
    assert fm is None
    assert "just a body" in body

def test_merge_frontmatter_preserves_existing_keys():
    content = "---\ntype: project\ntitle: Atlas\nstatus: active\nowning_engagement: northwind\n---\n\n# Atlas\n"
    merged = merge_frontmatter(content, {"strength": "Medium", "hard_cues": ["atlas", "northwind"]})
    fm, body = parse_frontmatter(merged)
    assert fm["title"] == "Atlas"           # preserved
    assert fm["owning_engagement"] == "northwind"  # preserved
    assert fm["strength"] == "Medium"        # added
    assert "atlas" in fm["hard_cues"]        # added
    assert body.startswith("# Atlas")

def test_merge_frontmatter_creates_when_absent():
    merged = merge_frontmatter("# Atlas\nbody", {"type": "project", "name": "atlas"})
    fm, body = parse_frontmatter(merged)
    assert fm["type"] == "project"
    assert fm["name"] == "atlas"
    assert body.startswith("# Atlas")

def test_strip_legacy_semantic_index_keeps_frontmatter():
    content = "---\ntype: project\n---\n\n# Atlas\n\n## SEMANTIC INDEX\n{\"name\":\"atlas\"}\n"
    out = strip_legacy_semantic_index(content)
    assert "## SEMANTIC INDEX" not in out
    assert out.startswith("---")   # frontmatter preserved
    assert "# Atlas" in out

def test_merge_strips_legacy_trailing_block():
    content = "---\ntype: project\n---\n\n# Atlas\n\n## SEMANTIC INDEX\n{\"name\":\"atlas\"}\n"
    merged = merge_frontmatter(content, {"strength": "High"})
    assert "## SEMANTIC INDEX" not in merged
    fm, _ = parse_frontmatter(merged)
    assert fm["strength"] == "High"
    # name is NOT auto-migrated from the trailing block (build_index regenerates it
    # via LLM); merge only applies explicit updates + strips the legacy block.
    assert "name" not in fm


# --- _shared: extract prefers frontmatter, falls back to trailing block -------

def test_extract_prefers_frontmatter():
    content = "---\ntype: project\nname: atlas\nhard_cues: [a, b]\n---\n\n# Atlas\n"
    si = extract_semantic_index(content)
    assert si is not None
    assert si["name"] == "atlas"
    assert si["hard_cues"] == ["a", "b"]

def test_extract_falls_back_to_trailing_block():
    content = "# Atlas\n\n## SEMANTIC INDEX\n{\"name\":\"atlas\",\"type\":\"project\"}\n"
    si = extract_semantic_index(content)
    assert si == {"name": "atlas", "type": "project"}

def test_extract_returns_none_when_neither():
    assert extract_semantic_index("# just prose\n") is None

def test_write_merges_into_frontmatter_and_drops_file_key():
    content = "---\ntype: project\ntitle: Atlas\n---\n\n# Atlas\n"
    out = write_with_semantic_index(content, {"name": "atlas", "file": "should/not/persist.md", "strength": "Low"})
    assert "## SEMANTIC INDEX" not in out
    fm, _ = parse_frontmatter(out)
    assert fm["name"] == "atlas"
    assert fm["strength"] == "Low"
    assert "file" not in fm          # computed at read time, never persisted
    assert fm["title"] == "Atlas"    # preserved

def test_strip_semantic_index_keeps_frontmatter():
    content = "---\ntype: project\n---\n\n# Atlas\n\n## SEMANTIC INDEX\n{\"name\":\"atlas\"}\n"
    out = strip_semantic_index(content)
    assert "## SEMANTIC INDEX" not in out
    assert out.startswith("---")     # frontmatter kept


# --- status canonicalization ---------------------------------------------------

_OPEN_ITEM = ["open", "in_progress", "blocked", "done", "cancelled"]

def test_canonical_direct_match():
    assert canonicalize_status("done", _OPEN_ITEM) == "done"
    assert canonicalize_status("In Progress", _OPEN_ITEM) == "in_progress"

def test_canonical_strips_parentheticals():
    # The exact Model Judgment Boundary bug: freeform "done (previously tracked as active)"
    assert canonicalize_status("done (previously tracked as active)", _OPEN_ITEM) == "done"
    assert canonicalize_status("Done (marked as active in some earlier drafts)", _OPEN_ITEM) == "done"

def test_canonical_synonyms():
    assert canonicalize_status("fulfilled", _OPEN_ITEM) == "done"
    assert canonicalize_status("broken", _OPEN_ITEM) == "cancelled"
    assert canonicalize_status("on hold", _OPEN_ITEM) == "blocked"

def test_canonical_none_when_unmatched():
    # unmatched freeform must NOT silently map to a terminal state
    assert canonicalize_status("flummoxed", _OPEN_ITEM) is None

def test_canonical_none_for_empty():
    assert canonicalize_status("", _OPEN_ITEM) is None
    assert canonicalize_status(None, _OPEN_ITEM) is None

def test_canonical_no_enum_returns_normalized():
    # ontologies without a status enum for this type (e.g. personal) → best-effort
    assert canonicalize_status("Done (foo)", None) == "done"

def test_canonical_does_not_drop_active():
    # 'active' is not an open_item value; it returns None (unmatched) — but None is
    # NOT a terminal state, so the followups drop logic keeps it (active items stay).
    canon = canonicalize_status("active", _OPEN_ITEM)
    assert canon not in {"done", "cancelled"}, "active must never canonicalize to a terminal state"
    assert canonicalize_status("Active (under investigation)", _OPEN_ITEM) not in {"done", "cancelled"}

def test_canonical_active_for_project_enum():
    # 'active' IS canonical for the project status enum.
    project_enum = ["proposed", "active", "paused", "completed", "cancelled"]
    assert canonicalize_status("Active", project_enum) == "active"


# --- conformance check ---------------------------------------------------------

def test_conformance_clean_corpus_returns_empty(tmp_path):
    p = load_ontology("corporate")
    people = tmp_path / "entities" / "people"
    people.mkdir(parents=True)
    (people / "maya_rivera.md").write_text(
        "---\ntype: human\ntitle: Maya Rivera\n---\n\n# Person: Maya Rivera\n"
    )
    assert check_conformance(tmp_path, p) == []

def test_conformance_flags_type_mismatch(tmp_path):
    p = load_ontology("corporate")
    projects = tmp_path / "entities" / "projects"
    projects.mkdir(parents=True)
    # project folder but frontmatter says human → mismatch
    (projects / "atlas.md").write_text(
        "---\ntype: human\ntitle: Atlas\n---\n\n# Atlas\n"
    )
    v = check_conformance(tmp_path, p)
    assert len(v) == 1
    assert v[0]["violation"] == "TYPE_MISMATCH"
    assert v[0]["index_type"] == "project"
    assert v[0]["frontmatter_type"] == "human"

def test_conformance_flags_missing_frontmatter(tmp_path):
    p = load_ontology("corporate")
    people = tmp_path / "entities" / "people"
    people.mkdir(parents=True)
    (people / "nobody.md").write_text("# just prose, no frontmatter\n")
    v = check_conformance(tmp_path, p)
    assert len(v) == 1
    assert v[0]["violation"] == "MISSING_FRONTMATTER"
