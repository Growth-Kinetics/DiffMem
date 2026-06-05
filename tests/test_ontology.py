"""Unit tests for the ontology loader (M1)."""
import json
import pytest
from pathlib import Path
import tempfile, os

from diffmem.ontology.loader import load_ontology, OntologyProfile


# ---------------------------------------------------------------------------
# Built-in: personal
# ---------------------------------------------------------------------------

def test_load_personal_profile():
    p = load_ontology("personal")
    assert p.name == "personal"
    assert isinstance(p.schema, dict)
    names = [e["name"] for e in p.entity_types]
    assert "people" in names
    assert "contexts" in names

def test_personal_folder_map():
    p = load_ontology("personal")
    fm = p.folder_map
    assert fm["people"] == "memories/people"
    assert fm["contexts"] == "memories/contexts"

def test_personal_prompts_fallback_to_writer_agent():
    """personal ontology has no prompts/ dir — everything falls back."""
    p = load_ontology("personal")
    assert p.prompts_dir is None
    # Should resolve from writer_agent/prompts/
    resolved = p.resolve_prompt("0_system")
    assert resolved.exists()
    assert "writer_agent" in str(resolved)

def test_personal_repo_guide_exists():
    p = load_ontology("personal")
    assert p.repo_guide_path.exists()


# ---------------------------------------------------------------------------
# Unknown built-in → ValueError
# ---------------------------------------------------------------------------

def test_unknown_builtin_raises():
    with pytest.raises(ValueError, match="Unknown ontology"):
        load_ontology("nonexistent_ontology_xyz")

def test_unknown_builtin_lists_available():
    with pytest.raises(ValueError, match="personal"):
        load_ontology("nonexistent_ontology_xyz")


# ---------------------------------------------------------------------------
# Absolute path
# ---------------------------------------------------------------------------

def test_absolute_path_valid(tmp_path):
    schema = {
        "name": "test",
        "entity_types": [
            {"name": "contacts", "folder": "mem/contacts", "index_type": "human"}
        ]
    }
    (tmp_path / "schema.json").write_text(json.dumps(schema))
    (tmp_path / "repo_guide.md").write_text("# Guide")

    p = load_ontology(str(tmp_path))
    assert p.name == tmp_path.name
    assert p.folder_map["contacts"] == "mem/contacts"

def test_absolute_path_missing_schema_raises(tmp_path):
    (tmp_path / "repo_guide.md").write_text("# Guide")
    with pytest.raises(ValueError, match="schema.json"):
        load_ontology(str(tmp_path))

def test_absolute_path_missing_repo_guide_raises(tmp_path):
    schema = {
        "name": "test",
        "entity_types": [{"name": "x", "folder": "y", "index_type": "z"}]
    }
    (tmp_path / "schema.json").write_text(json.dumps(schema))
    with pytest.raises(ValueError, match="repo_guide.md"):
        load_ontology(str(tmp_path))

def test_absolute_path_nonexistent_raises():
    with pytest.raises(ValueError, match="does not exist"):
        load_ontology("/tmp/diffmem_no_such_ontology_dir_xyz_123")


# ---------------------------------------------------------------------------
# DIFFMEM_ONTOLOGY env var
# ---------------------------------------------------------------------------

def test_env_var_default_is_personal(monkeypatch):
    monkeypatch.delenv("DIFFMEM_ONTOLOGY", raising=False)
    p = load_ontology(None)
    assert p.name == "personal"

def test_env_var_respected(monkeypatch):
    monkeypatch.setenv("DIFFMEM_ONTOLOGY", "personal")
    p = load_ontology(None)
    assert p.name == "personal"

def test_env_var_bad_value_raises(monkeypatch):
    monkeypatch.setenv("DIFFMEM_ONTOLOGY", "does_not_exist_abc")
    with pytest.raises(ValueError, match="Unknown ontology"):
        load_ontology(None)


# ---------------------------------------------------------------------------
# resolve_prompt fallback chain
# ---------------------------------------------------------------------------

def test_resolve_prompt_falls_back_when_no_prompts_dir(tmp_path):
    """Ontology with no prompts/ dir falls back to writer_agent/prompts/."""
    schema = {
        "name": "minimal",
        "entity_types": [{"name": "things", "folder": "mem/things", "index_type": "concept"}]
    }
    (tmp_path / "schema.json").write_text(json.dumps(schema))
    (tmp_path / "repo_guide.md").write_text("# Guide")

    p = load_ontology(str(tmp_path))
    resolved = p.resolve_prompt("0_system")
    assert resolved.exists()
    assert "writer_agent" in str(resolved)

def test_resolve_prompt_prefers_ontology_dir(tmp_path):
    """Ontology-specific prompt is preferred over fallback."""
    schema = {
        "name": "custom",
        "entity_types": [{"name": "things", "folder": "mem/things", "index_type": "concept"}]
    }
    (tmp_path / "schema.json").write_text(json.dumps(schema))
    (tmp_path / "repo_guide.md").write_text("# Guide")
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    custom_prompt = prompts_dir / "0_system.txt"
    custom_prompt.write_text("CUSTOM SYSTEM PROMPT")

    p = load_ontology(str(tmp_path))
    resolved = p.resolve_prompt("0_system")
    assert resolved == custom_prompt
    assert resolved.read_text() == "CUSTOM SYSTEM PROMPT"

def test_resolve_prompt_missing_everywhere_raises(tmp_path):
    schema = {
        "name": "x",
        "entity_types": [{"name": "a", "folder": "b", "index_type": "c"}]
    }
    (tmp_path / "schema.json").write_text(json.dumps(schema))
    (tmp_path / "repo_guide.md").write_text("# Guide")

    p = load_ontology(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        p.resolve_prompt("prompt_that_does_not_exist_xyz")
