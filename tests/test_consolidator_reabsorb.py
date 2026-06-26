"""M4 tests: the reabsorb consolidator tool — folds legacy commitments into
owners' `## Open Items` sections, canonicalizes status, deletes the files, and
produces `consolidate(reabsorb):` commits. Idempotent + chainable."""
import json
from pathlib import Path

import git
import pytest

from tests._fixtures import build_worktree
from diffmem.consolidator_agent.agent import ConsolidatorAgent
from diffmem.ontology.loader import load_ontology


def _commitment_file(parent: Path, slug: str, title: str, status: str,
                     assignee: str = "", due: str = "", related: list = None) -> Path:
    """Write a legacy corporate commitment file (pre-v2 format)."""
    f = parent / f"{slug}.md"
    body = [f"# Commitment: {title}", "", "## Metadata",
            f"- **Status:** {status}"]
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


def _corporate_worktree(tmp_path: Path) -> Path:
    """Build a corporate-ontology worktree with an entities/ layout (incl. a
    legacy entities/commitments/ folder)."""
    wt = tmp_path / "alex"
    wt.mkdir(parents=True, exist_ok=True)
    for d in ("entities/people", "entities/projects", "entities/decisions",
              "entities/external", "entities/commitments", "timeline"):
        (wt / d).mkdir(parents=True, exist_ok=True)
    (wt / "alex.md").write_text("# Alex\n\n## Overview\nAcme.\n", encoding="utf-8")
    (wt / "repo_guide.md").write_text("# guide\n", encoding="utf-8")
    repo = git.Repo.init(wt)
    repo.config_writer().set_value("user", "name", "t").release()
    repo.config_writer().set_value("user", "email", "t@t").release()
    repo.index.add(["alex.md", "repo_guide.md"])
    repo.index.commit("seed")
    return wt


def _agent(wt: Path) -> ConsolidatorAgent:
    p = load_ontology("corporate")
    return ConsolidatorAgent(
        repo_path=str(wt), user_id="alex", openrouter_api_key="dummy",
        model="test-model", ontology=p,
    )


def _project_file(wt: Path, slug: str, title: str, with_open_items: bool = False) -> Path:
    p = wt / "entities" / "projects" / f"{slug}.md"
    body = ["---", "type: project", f"title: {title}", "status: active", "---", "",
            f"# {title}", "", "## Objectives & Scope", "do the thing.", ""]
    if with_open_items:
        body += ["## Open Items", "", "- **[done]** Already shipped task", ""]
    p.write_text("\n".join(body), encoding="utf-8")
    return p


def test_reabsorb_folds_commitments_into_owner_projects(tmp_path):
    wt = _corporate_worktree(tmp_path)
    cdir = wt / "entities" / "commitments"
    _commitment_file(cdir, "deliver_v1_api", "Deliver v1 API", "In Progress",
                     assignee="maya_rivera", due="2026-07-15", related=["atlas"])
    _commitment_file(cdir, "ship_landing_page", "Ship landing page", "Open",
                     assignee="sam_rivera", related=["atlas"])
    _project_file(wt, "atlas", "Project Atlas")  # the owner
    agent = _agent(wt)

    r = agent.run_reabsorb()
    assert r["status"] == "ok"
    assert r["tool"] == "reabsorb"
    assert r["folded"] == 2
    assert len(r["commits"]) == 2
    assert all(c.startswith("consolidate(reabsorb):") for c in r["commits"])

    # commitment files gone
    assert not (cdir / "deliver_v1_api.md").exists()
    assert not (cdir / "ship_landing_page.md").exists()
    # owner file gained the Open Items entries
    atlas = (wt / "entities" / "projects" / "atlas.md").read_text(encoding="utf-8")
    assert "## Open Items" in atlas
    assert "Deliver v1 API" in atlas
    assert "Ship landing page" in atlas
    # status canonicalized to the open_item enum
    assert "**[in_progress]**" in atlas
    assert "**[open]**" in atlas
    # assignee + due preserved
    assert "[[maya_rivera]]" in atlas
    assert "2026-07-15" in atlas


def test_reabsorb_self_evicts_terminal_commitments(tmp_path):
    """Done/cancelled commitments are folded with their terminal status (history
    in git), and the live followups list would self-evict them — but the entry
    IS still written so the owner file records it."""
    wt = _corporate_worktree(tmp_path)
    cdir = wt / "entities" / "commitments"
    _commitment_file(cdir, "old_poc", "Old POC idea", "Completed",
                     assignee="maya_rivera", related=["atlas"])
    _project_file(wt, "atlas", "Project Atlas")
    agent = _agent(wt)

    r = agent.run_reabsorb()
    assert r["folded"] == 1
    atlas = (wt / "entities" / "projects" / "atlas.md").read_text(encoding="utf-8")
    assert "**[done]**" in atlas  # canonicalized Completed → done


def test_reabsorb_orphans_attach_to_root_user(tmp_path):
    """A commitment with no resolvable owner attaches to the root user entity."""
    wt = _corporate_worktree(tmp_path)
    cdir = wt / "entities" / "commitments"
    _commitment_file(cdir, "stray_task", "Stray follow-up", "Open",
                     assignee="alex", related=[])  # no related → no owner
    _project_file(wt, "atlas", "Project Atlas")  # exists but commitment doesn't link it
    agent = _agent(wt)

    r = agent.run_reabsorb()
    assert r["folded"] == 1
    assert r["orphaned"] == 1
    root = (wt / "alex.md").read_text(encoding="utf-8")
    assert "## Open Items" in root
    assert "Stray follow-up" in root


def test_reabsorb_idempotent_no_commitments_folder(tmp_path):
    """No entities/commitments folder → zero commits (idempotent, v2 steady state)."""
    wt = _corporate_worktree(tmp_path)
    # remove the (empty) commitments folder to simulate a clean v2 corpus
    (wt / "entities" / "commitments").rmdir()
    agent = _agent(wt)
    r = agent.run_reabsorb()
    assert r["status"] == "ok"
    assert r["folded"] == 0
    assert r["commits"] == []


def test_reabsorb_idempotent_empty_commitments_folder(tmp_path):
    """Empty commitments folder → zero commits."""
    wt = _corporate_worktree(tmp_path)
    agent = _agent(wt)
    r = agent.run_reabsorb()
    assert r["folded"] == 0
    assert r["commits"] == []


def test_reabsorb_second_run_produces_no_new_commits(tmp_path):
    """Running reabsorb twice: the second run finds an empty folder → 0 commits."""
    wt = _corporate_worktree(tmp_path)
    cdir = wt / "entities" / "commitments"
    _commitment_file(cdir, "deliver_v1_api", "Deliver v1 API", "Open",
                     related=["atlas"])
    _project_file(wt, "atlas", "Project Atlas")
    agent = _agent(wt)

    first = agent.run_reabsorb()
    assert first["folded"] == 1
    second = agent.run_reabsorb()
    assert second["folded"] == 0
    assert second["commits"] == []


def test_reabsorb_is_chainable_in_canonical_order(tmp_path):
    """reabsorb runs first in the canonical consolidate order."""
    wt = _corporate_worktree(tmp_path)
    cdir = wt / "entities" / "commitments"
    _commitment_file(cdir, "deliver_v1_api", "Deliver v1 API", "Open", related=["atlas"])
    _project_file(wt, "atlas", "Project Atlas")
    from diffmem.api import DiffMemory
    p = load_ontology("corporate")
    mem = DiffMemory(repo_path=str(wt), user_id="alex", openrouter_api_key="dummy",
                     model="test-model", ontology=p)
    # run only reabsorb via the consolidate() orchestrator
    r = mem.consolidate(tools=["reabsorb"])
    assert r["status"] == "ok"
    assert r["tools_run"] == ["reabsorb"]
    assert len(r["commits"]) == 1
