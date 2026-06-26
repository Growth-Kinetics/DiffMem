"""M5: end-to-end fixture validation of the corporate v2 ontology.

Mirrors the growth-kinetics pathologies (oversized commitments corpus, mis-bucketed
types, freeform/legacy statuses) on a neutral (sanitized) fixture, then runs the
v2 pipeline — reabsorb + conformance + followups rebuild — and asserts the
deterministic post-conditions. No real client data; uses neutral Acme/Atlas names.
"""
import json
import shutil
from pathlib import Path

import git
import pytest

from diffmem.api import DiffMemory
from diffmem.ontology.loader import load_ontology
from diffmem.consolidator_agent.agent import ConsolidatorAgent
from diffmem.consolidator_agent._reabsorb import _parse_commitment, OPEN_ITEM_ENUM
from diffmem.conformance import check_conformance
from diffmem.writer_agent.agent import WriterAgent
from diffmem.frontmatter import parse_frontmatter


# ---------------------------------------------------------------------------
# Fixture: a corporate worktree with a legacy, pathological commitments corpus
# ---------------------------------------------------------------------------

def _build_pathological_worktree(tmp_path: Path) -> Path:
    wt = tmp_path / "acme"
    wt.mkdir(parents=True, exist_ok=True)
    for d in ("entities/people", "entities/projects", "entities/decisions",
              "entities/external", "entities/commitments", "timeline"):
        (wt / d).mkdir(parents=True, exist_ok=True)
    # Root user entity (company).
    (wt / "acme.md").write_text(
        "---\ntype: company\ntitle: Acme\nstatus: active\n---\n\n# acme\n\n## Overview\nAcme Data Co.\n",
        encoding="utf-8",
    )
    (wt / "repo_guide.md").write_text("# guide\n", encoding="utf-8")
    (wt / "index.md").write_text("# index\n", encoding="utf-8")

    # Two owner projects (proper v2 frontmatter).
    _write_project(wt, "atlas", "Project Atlas")
    _write_project(wt, "hikari", "Project Hikari", with_open_items=True)

    # One person (so owner-resolution can fall back to people too).
    _write_person(wt, "maya_rivera", "Maya Rivera")

    # 10 legacy commitments — the explosion. Mix of:
    #  - real deliverables (open/in-progress)
    #  - freeform terminal statuses that must self-evict
    #    ("done (previously tracked as active)", "Completed", "Cancelled")
    #  - a mis-bucketed # Project: heading (analysis work in commitments/)
    #  - an orphan with no resolvable owner
    cdir = wt / "entities" / "commitments"
    _legacy_commitment(cdir, "deliver_v1_api", "Deliver v1 API", "In Progress",
                       assignee="maya_rivera", due="2026-07-15", related=["atlas"])
    _legacy_commitment(cdir, "ship_landing_page", "Ship landing page", "Open",
                       assignee="sam_rivera", related=["atlas"])
    _legacy_commitment(cdir, "build_capacity_signals", "Build capacity signals", "open",
                       related=["hikari"])
    _legacy_commitment(cdir, "deliver_qbr", "Deliver QBR by May", "Active",
                       assignee="maya_rivera", related=["hikari"])
    _legacy_commitment(cdir, "old_poc_idea", "Old POC idea", "done (previously tracked as active)",
                       related=["atlas"])
    _legacy_commitment(cdir, "finished_mvp", "Finished MVP", "Completed",
                       related=["atlas"])
    _legacy_commitment(cdir, "dropped_initiative", "Dropped initiative", "Cancelled",
                       related=["hikari"])
    # mis-bucketed: a # Project: heading in a commitments file
    _legacy_commitment(cdir, "analyze_coffee_appu", "Analyze Coffee APPU", "Active",
                       related=["atlas"], heading="Project")
    # orphan: no related AND no assignee wikilink → root user
    _legacy_commitment(cdir, "stray_followup", "Stray follow-up", "Open",
                       assignee="", related=[])
    _legacy_commitment(cdir, "send_deck", "Send deck to Northwind", "in progress",
                       related=["atlas"])

    repo = git.Repo.init(wt)
    repo.config_writer().set_value("user", "name", "t").release()
    repo.config_writer().set_value("user", "email", "t@t").release()
    repo.git.add("--all")
    repo.index.commit("seed pathological corpus")
    return wt


def _write_project(wt: Path, slug: str, title: str, with_open_items: bool = False) -> Path:
    p = wt / "entities" / "projects" / f"{slug}.md"
    body = ["---", "type: project", f"title: {title}", "status: active",
            f"owning_engagement: internal", "company_priority: high", "---", "",
            f"# {title}", "", "## Objectives & Scope", "do the thing.", ""]
    if with_open_items:
        body += ["## Open Items", "", "- **[done]** Already shipped task", ""]
    p.write_text("\n".join(body), encoding="utf-8")
    return p


def _write_person(wt: Path, slug: str, title: str) -> Path:
    p = wt / "entities" / "people" / f"{slug}.md"
    p.write_text("\n".join([
        "---", "type: human", f"title: {title}", "status: active",
        "affiliation: internal", "---", "", f"# {title}", "",
        "## Active Context & Relationships", "lead engineer.", "",
    ]), encoding="utf-8")
    return p


def _legacy_commitment(parent: Path, slug: str, title: str, status: str,
                        assignee: str = "", due: str = "", related: list = None,
                        heading: str = "Commitment") -> Path:
    f = parent / f"{slug}.md"
    body = [f"# {heading}: {title}", "", "## Metadata", f"- **Status:** {status}"]
    if assignee:
        body.append(f"- **Assignee:** [[{assignee}]]")
    if due:
        body.append(f"- **Due Date:** {due}")
    body += ["", "## Description", f"Body for {title}.", "", "## Related Links"]
    for r in (related or []):
        body.append(f"- [[{r}]]")
    body.append("")
    f.write_text("\n".join(body), encoding="utf-8")
    return f


def _agent(wt: Path) -> ConsolidatorAgent:
    p = load_ontology("corporate")
    return ConsolidatorAgent(repo_path=str(wt), user_id="acme",
                             openrouter_api_key="dummy", model="test-model", ontology=p)


# ---------------------------------------------------------------------------
# The e2e test
# ---------------------------------------------------------------------------

def test_v2_pipeline_reduces_commitments_and_passes_conformance(tmp_path):
    """Full v2 chain on a pathological corpus: reabsorb -> conformance -> followups.

    Asserts the four deterministic post-conditions from M5:
      1. commitments-equivalent file count reduced by >=80%.
      2. zero conformance violations on remaining entity files.
      3. followups.md Active Items contains no done/cancelled entry.
      4. deterministic (re-running reabsorb produces no new commits).
    """
    wt = _build_pathological_worktree(tmp_path)
    cdir = wt / "entities" / "commitments"
    initial_commitments = len(list(cdir.glob("*.md")))
    assert initial_commitments == 10

    # --- 1. reabsorb folds the legacy commitments into owners ---
    agent = _agent(wt)
    r = agent.run_reabsorb()
    assert r["status"] == "ok"
    assert r["folded"] == 10
    remaining = len(list(cdir.glob("*.md")))
    reduction = (initial_commitments - remaining) / initial_commitments
    assert reduction >= 0.80, f"expected >=80% reduction, got {reduction:.0%} ({remaining} left)"
    assert remaining == 0, "all legacy commitments should be folded+removed"

    # --- 2. conformance: remaining entity files all conform ---
    p = load_ontology("corporate")
    violations = check_conformance(wt, p)
    # the two projects + one person must all carry correct frontmatter types
    assert violations == [], f"conformance violations: {violations}"

    # --- 3. followups.md: Active Items self-evicts done/cancelled ---
    # Build a WriterAgent and rebuild followups (mock the Other-Items LLM call).
    wa = WriterAgent(str(wt), "acme", "fake-key", model="test-model",
                     validate_paths=False, ontology=p)
    from unittest.mock import patch
    with patch.object(wa, "_call_llm", return_value="_(none)_"):
        wa._rebuild_followups_index("transcript")
    out = (wt / "followups.md").read_text(encoding="utf-8")
    assert "## Active Items" in out
    # The folded deliverables (open/in-progress) appear:
    assert "Deliver v1 API" in out
    assert "Build capacity signals" in out
    # Terminal entries self-evict (not in the live list):
    assert "Old POC idea" not in out          # "done (previously tracked as active)"
    assert "Finished MVP" not in out           # "Completed"
    assert "Dropped initiative" not in out     # "Cancelled"
    # And no rendered Active Item carries a terminal canonical status:
    assert "**[done]**" not in out
    assert "**[cancelled]**" not in out
    # active count reflects the non-terminal folded items
    assert "active items" in out

    # --- 4. determinism: re-running reabsorb is a no-op ---
    r2 = agent.run_reabsorb()
    assert r2["folded"] == 0
    assert r2["commits"] == []


def test_v2_pipeline_orphans_fold_to_root_user(tmp_path):
    """The orphan commitment (no resolvable owner) attaches to the root company entity."""
    wt = _build_pathological_worktree(tmp_path)
    agent = _agent(wt)
    r = agent.run_reabsorb()
    assert r["orphaned"] >= 1, "expected the stray_followup to be orphaned → root"
    root = (wt / "acme.md").read_text(encoding="utf-8")
    assert "## Open Items" in root
    assert "Stray follow-up" in root


def test_v2_open_items_carry_canonical_statuses(tmp_path):
    """Folded Open Items use the canonical open_item enum (in_progress/open), not raw prose."""
    wt = _build_pathological_worktree(tmp_path)
    _agent(wt).run_reabsorb()
    atlas = (wt / "entities" / "projects" / "atlas.md").read_text(encoding="utf-8")
    # "In Progress" -> in_progress ; "in progress" (send_deck) -> in_progress ; "Open" -> open
    assert "**[in_progress]**" in atlas
    assert "**[open]**" in atlas
    # The mis-bucketed # Project: commitment is folded as an Open Item, not re-created as a file.
    assert "Analyze Coffee APPU" in atlas
    assert not (wt / "entities" / "commitments" / "analyze_coffee_appu.md").exists()
