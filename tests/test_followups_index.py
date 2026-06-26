"""
Regression tests for the corporate followups.md index.

followups.md is a team-shareable to-do list rebuilt on every commit session.
Two sections:
  - ## Active Commitments: deterministic scan of entities/commitments/*.md
    (drops Completed/Done/Cancelled by design — git history is the archive)
  - ## Other Items: LLM-curated lightweight follow-ups (the kind that don't
    warrant their own entity file, e.g. "Alex to call Bob next week")

These tests pin the deterministic half + the LLM-degradation paths.
The LLM call itself is mocked — we are not testing model quality, only contract.
"""
import json
from pathlib import Path
from unittest.mock import patch

import git
import pytest

from diffmem.writer_agent.agent import WriterAgent
from diffmem.ontology.loader import load_ontology


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _commitment_file(parent: Path, slug: str, title: str, status: str,
                     owner: str = "", due: str = "") -> Path:
    """Write a realistic commitment .md file (matches what the LLM produces today)."""
    f = parent / f"{slug}.md"
    body = [
        f"# Commitment: {title}",
        "",
        "## Metadata",
        f"- **Status:** {status}",
    ]
    if owner:
        body.append(f"- **Assignee:** {owner}")
    if due:
        body.append(f"- **Due Date:** {due}")
    body += [
        "",
        "## Description",
        f"Body for {title}.",
        "",
        "## SEMANTIC INDEX",
        json.dumps({"name": slug, "type": "commitment", "role": title}),
        "",
    ]
    f.write_text("\n".join(body), encoding="utf-8")
    return f


def _make_corporate_writer(tmp_path: Path) -> WriterAgent:
    """Minimal corporate worktree + WriterAgent."""
    repo = git.Repo.init(tmp_path)
    repo.config_writer().set_value("user", "name", "test").release()
    repo.config_writer().set_value("user", "email", "test@test").release()

    user_id = "gk"
    (tmp_path / f"{user_id}.md").write_text(f"# {user_id}\n")
    (tmp_path / "index.md").write_text("# index\n")

    # Create the 5 corporate folders to match what onboarding does
    for d in ("people", "projects", "decisions", "commitments", "external"):
        (tmp_path / "entities" / d).mkdir(parents=True, exist_ok=True)

    repo.index.add([f"{user_id}.md", "index.md"])
    repo.index.commit("seed")

    ontology = load_ontology("corporate")
    agent = WriterAgent(
        str(tmp_path), user_id, "fake-key", model="test-model",
        validate_paths=False, ontology=ontology,
    )
    return agent


# ---------------------------------------------------------------------------
# Schema + ontology wiring
# ---------------------------------------------------------------------------

def test_corporate_followups_enabled():
    """Corporate ontology must have followups enabled."""
    p = load_ontology("corporate")
    assert p.followups_enabled is True


def test_personal_followups_disabled():
    """Personal ontology must NOT have followups enabled (no behavior change)."""
    p = load_ontology("personal")
    assert p.followups_enabled is False
    assert p.followups_source_dir(Path("/x")) is None


def _entity_with_open_items(parent: Path, filename: str, entity_title: str,
                             open_items: list, fm_type: str = "project") -> Path:
    """Write an entity file with a `## Open Items` section.

    open_items: list of (status, title, owner, due) tuples. owner/due may be ''.
    """
    f = parent / f"{filename}.md"
    body = [
        "---",
        f"type: {fm_type}",
        f"title: {entity_title}",
        "status: active",
        "---",
        "",
        f"# {entity_title}",
        "",
        "## Open Items",
        "",
    ]
    for status, title, owner, due in open_items:
        line = f"- **[{status}]** {title}"
        if owner:
            line += f" — assignee [[{owner}]]"
        if due:
            line += f", due {due}"
        body.append(line)
    body.append("")
    f.write_text("\n".join(body), encoding="utf-8")
    return f


def test_rebuild_aggregates_open_items_across_entities(tmp_path):
    """Active Items aggregates ## Open Items entries from multiple entity files."""
    agent = _make_corporate_writer(tmp_path)
    # NOTE: no entities/commitments folder is created — v2 has no such folder.
    projects = tmp_path / "entities" / "projects"
    projects.mkdir(parents=True, exist_ok=True)
    _entity_with_open_items(projects, "atlas", "Project Atlas",
                            [("open", "Deliver v1 API", "maya_rivera", "2026-07-15"),
                             ("in_progress", "Build capacity signals", "sam_rivera", "")])
    people = tmp_path / "entities" / "people"
    people.mkdir(parents=True, exist_ok=True)
    _entity_with_open_items(people, "maya_rivera", "Maya Rivera",
                            [("open", "Send deck to Northwind", "", "")],
                            fm_type="human")

    with patch.object(agent, "_call_llm", return_value="_(none)_"):
        agent._rebuild_followups_index("transcript")

    out = (tmp_path / "followups.md").read_text(encoding="utf-8")
    assert "## Active Items" in out
    assert "Deliver v1 API" in out
    assert "Build capacity signals" in out
    assert "Send deck to Northwind" in out
    assert "3 active items" in out
    # Each item links back to its owning entity.
    assert "[[entities/projects/atlas|Project Atlas]]" in out
    assert "[[entities/people/maya_rivera|Maya Rivera]]" in out


def test_rebuild_self_evicts_done_and_cancelled(tmp_path):
    """Open Items with terminal status (done/cancelled) drop from the live list."""
    agent = _make_corporate_writer(tmp_path)
    projects = tmp_path / "entities" / "projects"
    projects.mkdir(parents=True, exist_ok=True)
    _entity_with_open_items(projects, "atlas", "Project Atlas",
                            [("open", "Deliver v1 API", "", ""),
                             ("done", "Ship landing page", "", ""),
                             ("cancelled", "Old POC idea", "", "")])
    with patch.object(agent, "_call_llm", return_value="_(none)_"):
        agent._rebuild_followups_index("transcript")

    out = (tmp_path / "followups.md").read_text(encoding="utf-8")
    assert "Deliver v1 API" in out
    assert "Ship landing page" not in out
    assert "Old POC idea" not in out
    assert "1 active items" in out


def test_rebuild_renders_none_when_no_open_items(tmp_path):
    """No Open Items anywhere → Active Items shows the _(none)_ placeholder."""
    agent = _make_corporate_writer(tmp_path)
    projects = tmp_path / "entities" / "projects"
    projects.mkdir(parents=True, exist_ok=True)
    _entity_with_open_items(projects, "atlas", "Project Atlas",
                            [("done", "Finished task", "", "")])  # only terminal
    with patch.object(agent, "_call_llm", return_value="_(none)_"):
        agent._rebuild_followups_index("transcript")

    out = (tmp_path / "followups.md").read_text(encoding="utf-8")
    assert "## Active Items\n\n_(none)_" in out
    assert "0 active items" in out


def test_rebuild_works_without_commitments_folder(tmp_path):
    """v2: followups rebuild works when no entities/commitments folder exists at all."""
    agent = _make_corporate_writer(tmp_path)
    # _make_corporate_writer created entities/commitments per old onboarding; remove it.
    import shutil
    shutil.rmtree(tmp_path / "entities" / "commitments")
    with patch.object(agent, "_call_llm", return_value="_(none)_"):
        agent._rebuild_followups_index("transcript")  # must not raise
    out = (tmp_path / "followups.md").read_text(encoding="utf-8")
    assert "## Active Items" in out
    assert "## Other Items" in out


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def test_parser_extracts_title_status_owner_due(tmp_path):
    agent = _make_corporate_writer(tmp_path)
    commitments_dir = tmp_path / "entities" / "commitments"
    f = _commitment_file(
        commitments_dir, "deliver_qbr", "Deliver Northwind QBR by May",
        status="In Progress", owner="[[sam_rivera]]", due="Early May 2026",
    )
    parsed = agent._parse_commitment_metadata(f)
    assert parsed is not None
    assert parsed["slug"] == "deliver_qbr"
    assert parsed["title"] == "Deliver Northwind QBR by May"
    assert parsed["status"] == "in_progress"  # v2: canonicalized to the open_item enum
    assert parsed["owner"] == "[[sam_rivera]]"
    assert parsed["due"] == "Early May 2026"


def test_parser_drops_completed(tmp_path):
    """Completed/Done/Cancelled commitments must be filtered out of the list."""
    agent = _make_corporate_writer(tmp_path)
    cdir = tmp_path / "entities" / "commitments"
    for status in ("Completed", "Done", "Cancelled", "Canceled", "Closed"):
        f = _commitment_file(cdir, f"slug_{status.lower()}", f"Title {status}",
                             status=status)
        assert agent._parse_commitment_metadata(f) is None, (
            f"Status='{status}' should drop from the live followups list"
        )


def test_parser_tolerates_field_name_variants(tmp_path):
    """Real LLM output varies — 'Assignee' vs 'Assignees' vs 'Owner' all map to owner."""
    agent = _make_corporate_writer(tmp_path)
    cdir = tmp_path / "entities" / "commitments"

    # 'Assignees' variant
    f1 = cdir / "c1.md"
    f1.write_text(
        "# Commitment: One\n\n## Metadata\n- **Status:** Active\n"
        "- **Assignees:** [[a]], [[b]]\n- **Due Date:** 2026-04-23\n", encoding="utf-8"
    )
    p1 = agent._parse_commitment_metadata(f1)
    assert p1["owner"] == "[[a]], [[b]]"
    assert p1["due"] == "2026-04-23"

    # 'Owner' variant (some commitments use this)
    f2 = cdir / "c2.md"
    f2.write_text(
        "# Commitment: Two\n\n## Metadata\n- **Status:** Active\n"
        "- **Owner**: [[c]]\n", encoding="utf-8"
    )
    p2 = agent._parse_commitment_metadata(f2)
    assert p2["owner"] == "[[c]]"


# ---------------------------------------------------------------------------
# Full rebuild
# ---------------------------------------------------------------------------

def test_rebuild_uses_existing_other_items_when_llm_fails(tmp_path):
    """If the LLM call raises, the previous Other Items section is preserved."""
    agent = _make_corporate_writer(tmp_path)

    # Seed a previous followups.md with Other Items.
    (tmp_path / "followups.md").write_text(
        "# Follow-ups\n\n## Active Items\n\n_(none)_\n\n"
        "## Other Items\n\n- previous item one\n- previous item two\n",
        encoding="utf-8",
    )

    def boom(*args, **kwargs):
        raise RuntimeError("openrouter exploded")

    with patch.object(agent, "_call_llm", side_effect=boom):
        agent._rebuild_followups_index("transcript")

    out = (tmp_path / "followups.md").read_text(encoding="utf-8")
    assert "previous item one" in out, "Previous Other Items must be preserved on LLM failure"
    assert "previous item two" in out


# ---------------------------------------------------------------------------
# Ontology gating — process_session skips when followups disabled
# ---------------------------------------------------------------------------

def test_process_session_skips_followups_for_personal_ontology(tmp_path):
    """Personal ontology must NOT generate followups.md (followups_enabled=False)."""
    repo = git.Repo.init(tmp_path)
    repo.config_writer().set_value("user", "name", "test").release()
    repo.config_writer().set_value("user", "email", "test@test").release()
    (tmp_path / "alex.md").write_text("# alex\n")
    (tmp_path / "index.md").write_text("# index\n")
    (tmp_path / "memories" / "people").mkdir(parents=True)
    repo.index.add(["alex.md", "index.md"])
    repo.index.commit("seed")

    ontology = load_ontology("personal")
    agent = WriterAgent(
        str(tmp_path), "alex", "fake-key", model="test-model",
        validate_paths=False, ontology=ontology,
    )

    # Mock everything in process_session that would need real LLM/git work.
    # We only care that _rebuild_followups_index is NOT called.
    with patch.object(agent, "_rebuild_followups_index") as mock_followups, \
         patch.object(agent, "_identify_relevant_entities", return_value={
             "entities_to_create": [], "entities_to_update": []
         }), \
         patch.object(agent, "_create_new_entities"), \
         patch.object(agent, "_update_existing_entities"), \
         patch.object(agent, "_create_timeline_entry"), \
         patch.object(agent, "_get_modified_files", return_value=[]), \
         patch.object(agent, "_build_entity_indexes"), \
         patch.object(agent, "_rebuild_master_index"):
        agent.process_session("transcript", "session-x")

    assert mock_followups.call_count == 0, (
        "Personal ontology must NOT trigger followups rebuild"
    )


def test_process_session_invokes_followups_for_corporate(tmp_path):
    """Corporate ontology must call _rebuild_followups_index in process_session."""
    agent = _make_corporate_writer(tmp_path)
    with patch.object(agent, "_rebuild_followups_index") as mock_followups, \
         patch.object(agent, "_identify_relevant_entities", return_value={
             "entities_to_create": [], "entities_to_update": []
         }), \
         patch.object(agent, "_create_new_entities"), \
         patch.object(agent, "_update_existing_entities"), \
         patch.object(agent, "_create_timeline_entry"), \
         patch.object(agent, "_get_modified_files", return_value=[]), \
         patch.object(agent, "_build_entity_indexes"), \
         patch.object(agent, "_rebuild_master_index"):
        agent.process_session("transcript", "session-x")

    assert mock_followups.call_count == 1
    # Verify the transcript flowed through as first arg
    assert mock_followups.call_args[0][0] == "transcript"


def test_followups_failure_does_not_break_session(tmp_path):
    """If _rebuild_followups_index raises, process_session must still complete."""
    agent = _make_corporate_writer(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("intentional")

    with patch.object(agent, "_rebuild_followups_index", side_effect=boom), \
         patch.object(agent, "_identify_relevant_entities", return_value={
             "entities_to_create": [], "entities_to_update": []
         }), \
         patch.object(agent, "_create_new_entities"), \
         patch.object(agent, "_update_existing_entities"), \
         patch.object(agent, "_create_timeline_entry"), \
         patch.object(agent, "_get_modified_files", return_value=[]), \
         patch.object(agent, "_build_entity_indexes"), \
         patch.object(agent, "_rebuild_master_index"):
        # Should not raise
        agent.process_session("transcript", "session-x")
