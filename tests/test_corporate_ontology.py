"""M1 structural tests for the redesigned corporate ontology (v2).

Verifies the ontology contract declared in SESSION_SPEC_2026-06-26-001 M1:
4 entity types (no commitments), status_enums, frontmatter convention, the
company-lens vectors, and that prompts reference frontmatter (not the trailing
SEMANTIC INDEX JSON block).
"""
import json
from pathlib import Path

from diffmem.ontology.loader import load_ontology


CORP = load_ontology("corporate")


# ---------------------------------------------------------------------------
# Entity types: 4, no commitments
# ---------------------------------------------------------------------------

def test_corporate_has_exactly_four_entity_types():
    names = [e["name"] for e in CORP.entity_types]
    assert len(names) == 4, f"expected 4 entity types, got {names}"

def test_corporate_drops_commitments_type():
    names = [e["name"] for e in CORP.entity_types]
    assert "commitments" not in names, "commitments must be demoted, not a top-level type"

def test_corporate_keeps_core_types():
    names = set(e["name"] for e in CORP.entity_types)
    # the four load-bearing types
    assert {"people", "external", "projects", "decisions"}.issubset(names)

def test_each_entity_type_has_folder_and_index_type():
    for et in CORP.entity_types:
        assert "folder" in et and et["folder"], et
        assert "index_type" in et and et["index_type"], et


# ---------------------------------------------------------------------------
# status_enums declared in schema
# ---------------------------------------------------------------------------

def test_status_enums_present():
    se = CORP.schema.get("status_enums")
    assert isinstance(se, dict) and se, "schema must declare status_enums"

def test_status_enums_for_project_decision_open_item():
    se = CORP.schema["status_enums"]
    for key in ("project", "decision", "open_item"):
        assert key in se and se[key], f"missing status enum for {key}"

def test_open_item_enum_has_terminal_states():
    se = CORP.schema["status_enums"]["open_item"]
    assert "done" in se and "cancelled" in se, "open_item enum must include done/cancelled for self-eviction"


# ---------------------------------------------------------------------------
# Frontmatter convention
# ---------------------------------------------------------------------------

def test_followups_source_is_open_items_not_commitments_folder():
    # followups must NOT scan a commitments folder
    assert CORP.schema.get("followups_source") != "commitments_folder"
    # either the new open_items_sections contract, or at least no commitments source folder
    src_folder = CORP.schema.get("followups_source_folder")
    assert src_folder != "entities/commitments", "must not scan a commitments folder"

def test_frontmatter_block_declared():
    fm = CORP.schema.get("frontmatter")
    assert isinstance(fm, dict), "schema must declare a frontmatter convention"
    assert fm.get("required") == ["type"], "type is the single required frontmatter field"


# ---------------------------------------------------------------------------
# repo_guide documents the new contract
# ---------------------------------------------------------------------------

def test_repo_guide_documents_frontmatter():
    rg = CORP.repo_guide_path.read_text(encoding="utf-8").lower()
    assert "frontmatter" in rg
    assert "open items" in rg

def test_repo_guide_documents_company_lens():
    rg = CORP.repo_guide_path.read_text(encoding="utf-8").lower()
    # at least one company-lens vector field
    assert any(tok in rg for tok in ("relationship_to_company", "owning_engagement", "affiliation")), \
        "repo_guide must document a company-lens vector"

def test_repo_guide_no_commitments_template():
    rg = CORP.repo_guide_path.read_text(encoding="utf-8").lower()
    # the commitments entity template must be gone from the guide
    assert "### commitments" not in rg and "entities/commitments/" not in rg, \
        "repo_guide must not present commitments as a top-level entity type"


# ---------------------------------------------------------------------------
# Prompts reference frontmatter (not the trailing SEMANTIC INDEX block)
# ---------------------------------------------------------------------------

CORP_PROMPTS = CORP.prompts_dir
assert CORP_PROMPTS is not None, "corporate ontology must ship its own prompts dir"

def test_identify_prompt_forbids_commitment_files():
    txt = (CORP_PROMPTS / "1_identify_entities.txt").read_text(encoding="utf-8")
    low = txt.lower()
    # the granularity gate must explicitly forbid commitment files
    assert "no commitments type" in low or "no commitments" in low or "never create a file for a task" in low
    # and must not list commitments as an entity kind
    assert "- **commitments**" not in txt

def test_create_prompt_uses_frontmatter():
    txt = (CORP_PROMPTS / "2_create_entity_file.txt").read_text(encoding="utf-8")
    assert "frontmatter" in txt.lower()
    # templates must carry frontmatter blocks (--- delimiters), not just body sections
    assert txt.count("---") >= 2

def test_update_prompt_appends_open_items():
    txt = (CORP_PROMPTS / "3_update_entity_file.txt").read_text(encoding="utf-8").lower()
    assert "open items" in txt
    assert "status" in txt  # instructs enum-constrained status updates

def test_build_index_prompt_no_commitment_type():
    txt = (CORP_PROMPTS / "build_index.txt").read_text(encoding="utf-8")
    low = txt.lower()
    assert "commitment" not in low or "no commitments" in low, \
        "build_index must not reference a commitment index_type"

def test_onboard_prompts_have_four_types():
    for name in ("onboard_identify_entities.txt", "onboard_user_entity.txt"):
        txt = (CORP_PROMPTS / name).read_text(encoding="utf-8").lower()
        assert "commitment" not in txt or "no commitments" in txt, f"{name} must drop commitments"
