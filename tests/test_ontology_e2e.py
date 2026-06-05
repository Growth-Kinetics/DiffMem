"""
End-to-end tests for the ontology system (M7).
Exercises both built-in profiles, custom path loading, error cases, and the
prompt fallback chain. No LLM calls — pure loader + schema validation.
"""
import json
import pytest
from pathlib import Path

from diffmem.ontology.loader import load_ontology, OntologyProfile

# ---------------------------------------------------------------------------
# personal profile — full validation
# ---------------------------------------------------------------------------

def test_personal_has_expected_entity_types():
    p = load_ontology("personal")
    names = [e["name"] for e in p.entity_types]
    assert "people" in names
    assert "contexts" in names
    assert "events" in names

def test_personal_folder_map_matches_legacy_hardcoded():
    """Regression: personal folders must stay identical to pre-ontology hardcoded values."""
    p = load_ontology("personal")
    fm = p.folder_map
    assert fm["people"] == "memories/people"
    assert fm["contexts"] == "memories/contexts"

def test_personal_repo_guide_is_valid_markdown():
    p = load_ontology("personal")
    content = p.repo_guide_path.read_text(encoding="utf-8")
    assert "## Repository Structure" in content
    assert len(content) > 500

def test_personal_all_prompts_resolvable():
    """Every prompt used by writer_agent must resolve (via fallback) for personal."""
    p = load_ontology("personal")
    required_prompts = [
        "0_system", "1_identify_entities", "2_create_entity_file",
        "3_update_entity_file", "4_create_timeline_entry",
        "build_index", "onboard_user_entity", "onboard_identify_entities",
        "onboard_timeline_entry",
    ]
    for name in required_prompts:
        resolved = p.resolve_prompt(name)
        assert resolved.exists(), f"Prompt '{name}' could not be resolved for 'personal'"

# ---------------------------------------------------------------------------
# corporate profile — full validation
# ---------------------------------------------------------------------------

def test_corporate_has_five_entity_types():
    p = load_ontology("corporate")
    names = [e["name"] for e in p.entity_types]
    assert names == ["people", "projects", "decisions", "commitments", "external"]

def test_corporate_folder_map_uses_entities_prefix():
    p = load_ontology("corporate")
    fm = p.folder_map
    assert fm["people"] == "entities/people"
    assert fm["projects"] == "entities/projects"
    assert fm["decisions"] == "entities/decisions"
    assert fm["commitments"] == "entities/commitments"
    assert fm["external"] == "entities/external"

def test_corporate_index_type_vocab():
    p = load_ontology("corporate")
    vocab = p.index_type_vocab
    assert "human" in vocab
    assert "project" in vocab
    assert "decision" in vocab
    assert "commitment" in vocab
    assert "company" in vocab

def test_corporate_repo_guide_mentions_all_entity_types():
    p = load_ontology("corporate")
    content = p.repo_guide_path.read_text(encoding="utf-8")
    for t in ["people", "projects", "decisions", "commitments", "external"]:
        assert t in content.lower(), f"repo_guide.md missing '{t}'"

def test_corporate_overrides_identify_entities_prompt():
    """corporate should use its own 1_identify_entities, not the personal fallback."""
    p = load_ontology("corporate")
    resolved = p.resolve_prompt("1_identify_entities")
    assert "corporate" in str(resolved), "Expected corporate-specific 1_identify_entities.txt"
    content = resolved.read_text()
    assert "projects" in content
    assert "decisions" in content
    assert "commitments" in content
    assert "external" in content

def test_corporate_overrides_create_entity_prompt():
    p = load_ontology("corporate")
    resolved = p.resolve_prompt("2_create_entity_file")
    assert "corporate" in str(resolved)

def test_corporate_overrides_update_entity_prompt():
    p = load_ontology("corporate")
    resolved = p.resolve_prompt("3_update_entity_file")
    assert "corporate" in str(resolved)

def test_corporate_overrides_onboard_user_entity_prompt():
    p = load_ontology("corporate")
    resolved = p.resolve_prompt("onboard_user_entity")
    assert "corporate" in str(resolved)

def test_corporate_overrides_onboard_identify_entities_prompt():
    p = load_ontology("corporate")
    resolved = p.resolve_prompt("onboard_identify_entities")
    assert "corporate" in str(resolved)

def test_corporate_overrides_build_index_prompt():
    p = load_ontology("corporate")
    resolved = p.resolve_prompt("build_index")
    assert "corporate" in str(resolved)

def test_corporate_falls_back_for_system_prompt():
    """0_system.txt is NOT overridden by corporate — falls back to writer_agent."""
    p = load_ontology("corporate")
    resolved = p.resolve_prompt("0_system")
    assert "writer_agent" in str(resolved)

def test_corporate_falls_back_for_timeline_entry_prompt():
    """4_create_timeline_entry.txt not overridden by corporate."""
    p = load_ontology("corporate")
    resolved = p.resolve_prompt("4_create_timeline_entry")
    assert "writer_agent" in str(resolved)

# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_unknown_builtin_raises_with_helpful_message():
    with pytest.raises(ValueError) as exc_info:
        load_ontology("does_not_exist_xyz")
    msg = str(exc_info.value)
    assert "Unknown ontology" in msg
    assert "personal" in msg  # lists available options

def test_absolute_path_missing_schema_raises():
    with pytest.raises(ValueError, match="schema.json"):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "repo_guide.md").write_text("# Guide")
            load_ontology(d)

def test_absolute_path_missing_repo_guide_raises():
    with pytest.raises(ValueError, match="repo_guide.md"):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            schema = {"name": "x", "entity_types": [{"name": "a", "folder": "b", "index_type": "c"}]}
            (Path(d) / "schema.json").write_text(json.dumps(schema))
            load_ontology(d)

def test_absolute_path_nonexistent_dir_raises():
    with pytest.raises(ValueError, match="does not exist"):
        load_ontology("/tmp/diffmem_no_such_dir_xyz_9999")

def test_schema_missing_entity_types_raises(tmp_path):
    (tmp_path / "schema.json").write_text(json.dumps({"name": "bad"}))
    (tmp_path / "repo_guide.md").write_text("# Guide")
    with pytest.raises(ValueError, match="entity_types"):
        load_ontology(str(tmp_path))

def test_schema_empty_entity_types_raises(tmp_path):
    (tmp_path / "schema.json").write_text(json.dumps({"name": "bad", "entity_types": []}))
    (tmp_path / "repo_guide.md").write_text("# Guide")
    with pytest.raises(ValueError, match="entity_types"):
        load_ontology(str(tmp_path))

# ---------------------------------------------------------------------------
# Custom path — happy path
# ---------------------------------------------------------------------------

def test_custom_path_loads_correctly(tmp_path):
    schema = {
        "name": "research",
        "entity_types": [
            {"name": "papers",   "folder": "mem/papers",   "index_type": "concept"},
            {"name": "authors",  "folder": "mem/authors",  "index_type": "human"},
        ]
    }
    (tmp_path / "schema.json").write_text(json.dumps(schema))
    (tmp_path / "repo_guide.md").write_text("# Research Guide\n\nFor research use.")
    p = load_ontology(str(tmp_path))
    assert p.folder_map["papers"] == "mem/papers"
    assert p.folder_map["authors"] == "mem/authors"
    assert "human" in p.index_type_vocab

def test_custom_path_prompt_override(tmp_path):
    schema = {
        "name": "custom",
        "entity_types": [{"name": "things", "folder": "m/things", "index_type": "concept"}]
    }
    (tmp_path / "schema.json").write_text(json.dumps(schema))
    (tmp_path / "repo_guide.md").write_text("# Guide")
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "1_identify_entities.txt").write_text("CUSTOM IDENTIFY PROMPT {semantic_index} {memory_input}")

    p = load_ontology(str(tmp_path))
    resolved = p.resolve_prompt("1_identify_entities")
    assert resolved.read_text().startswith("CUSTOM IDENTIFY PROMPT")
    # Non-overridden prompt falls back
    fallback = p.resolve_prompt("0_system")
    assert "writer_agent" in str(fallback)

# ---------------------------------------------------------------------------
# Folder listing helper (used by retrieval agent)
# ---------------------------------------------------------------------------

def test_personal_folder_listing_matches_legacy():
    """The rendered folder listing for personal must contain expected paths."""
    from diffmem.retrieval_agent.agent import _build_folder_listing
    p = load_ontology("personal")
    listing = _build_folder_listing(p)
    assert "memories/people" in listing
    assert "memories/contexts" in listing

def test_corporate_folder_listing_uses_entities_prefix():
    from diffmem.retrieval_agent.agent import _build_folder_listing
    p = load_ontology("corporate")
    listing = _build_folder_listing(p)
    assert "entities/people" in listing
    assert "entities/projects" in listing
    assert "entities/decisions" in listing

# ---------------------------------------------------------------------------
# All existing tests still pass (regression guard — just import check)
# ---------------------------------------------------------------------------

def test_ontology_module_importable():
    from diffmem.ontology import OntologyProfile, load_ontology  # noqa: F401
    assert True

# ---------------------------------------------------------------------------
# Bug-fix regression tests (review fixes)
# ---------------------------------------------------------------------------

def test_entity_dirs_personal():
    """Bug 2/5: entity_dirs() must return the correct paths for personal ontology."""
    from pathlib import Path
    p = load_ontology("personal")
    root = Path("/tmp/fake_repo")
    dirs = p.entity_dirs(root)
    paths = [str(d) for d in dirs]
    assert any("memories/people" in s for s in paths)
    assert any("memories/contexts" in s for s in paths)


def test_entity_dirs_corporate():
    """Bug 2/5: entity_dirs() must return entities/ prefix for corporate, not memories/."""
    from pathlib import Path
    p = load_ontology("corporate")
    root = Path("/tmp/fake_repo")
    dirs = p.entity_dirs(root)
    paths = [str(d) for d in dirs]
    assert any("entities/people" in s for s in paths)
    assert any("entities/decisions" in s for s in paths)
    assert not any("memories/" in s for s in paths), \
        "corporate entity_dirs must NOT contain 'memories/' prefix"


def test_default_folder_personal():
    """Issue 4: default_folder() must not hardcode 'memories/' for unknown types."""
    from pathlib import Path
    p = load_ontology("personal")
    root = Path("/tmp/fake_repo")
    d = p.default_folder(root)
    assert "memories/people" in str(d)


def test_default_folder_corporate():
    """Issue 4: corporate default_folder() returns first entity type folder (entities/people)."""
    from pathlib import Path
    p = load_ontology("corporate")
    root = Path("/tmp/fake_repo")
    d = p.default_folder(root)
    assert "entities/people" in str(d)
    assert "memories/" not in str(d)


def test_ontologies_dir_inside_package():
    """Bug 3: _ONTOLOGIES_DIR must resolve inside the package tree (src/diffmem/ontologies/),
    not at a repo-root or site-packages-adjacent path that breaks on pip install."""
    from diffmem.ontology.loader import _ONTOLOGIES_DIR
    assert "diffmem" in str(_ONTOLOGIES_DIR), \
        f"Expected _ONTOLOGIES_DIR to be inside the diffmem package, got: {_ONTOLOGIES_DIR}"
    assert _ONTOLOGIES_DIR.exists(), f"_ONTOLOGIES_DIR does not exist: {_ONTOLOGIES_DIR}"


def test_writer_agent_uses_ontology_entity_dirs(tmp_path):
    """Bug 2: WriterAgent._entity_md_files() scans ontology entity dirs, not hardcoded memories/."""
    import json
    from diffmem.ontology.loader import load_ontology
    from diffmem.writer_agent.agent import WriterAgent

    p = load_ontology("corporate")
    # Create minimal worktree structure
    (tmp_path / "agent.md").write_text("# agent\n\n## SEMANTIC INDEX\n{}")
    (tmp_path / "index.md").write_text("")
    entities_dir = tmp_path / "entities" / "people"
    entities_dir.mkdir(parents=True)
    (entities_dir / "AliceSmith.md").write_text("# Person: Alice Smith")

    agent = WriterAgent(
        str(tmp_path), "agent", "fake-key", model="fake-model",
        validate_paths=False, ontology=p,
    )
    found = list(agent._entity_md_files())
    names = [f.name for f in found]
    assert "AliceSmith.md" in names, \
        f"_entity_md_files() should find AliceSmith.md in entities/people, got: {names}"


def test_writer_agent_entity_md_files_does_not_include_memories_for_corporate(tmp_path):
    """Bug 2: corporate _entity_md_files() must NOT scan memories/ — it doesn't exist."""
    from diffmem.ontology.loader import load_ontology
    from diffmem.writer_agent.agent import WriterAgent

    p = load_ontology("corporate")
    (tmp_path / "agent.md").write_text("# agent")
    (tmp_path / "index.md").write_text("")
    # Create a stray file in memories/ to confirm it is ignored
    memories_dir = tmp_path / "memories" / "people"
    memories_dir.mkdir(parents=True)
    (memories_dir / "stray.md").write_text("# stray")
    # Create a proper corporate entity
    entities_dir = tmp_path / "entities" / "projects"
    entities_dir.mkdir(parents=True)
    (entities_dir / "DiffMem.md").write_text("# Project: DiffMem")

    agent = WriterAgent(
        str(tmp_path), "agent", "fake-key", model="fake-model",
        validate_paths=False, ontology=p,
    )
    found = [f.name for f in agent._entity_md_files()]
    assert "DiffMem.md" in found
    assert "stray.md" not in found, \
        "corporate _entity_md_files() must not pick up files from memories/"
