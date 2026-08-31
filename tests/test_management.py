# CAPABILITY: Tests for the management surface (consolidator_agent/management.py
#             + /memory/{uid}/manage/* routes).
# INPUTS: tmp_path worktrees + FakeLLM; FastAPI TestClient with stubbed memory.
# OUTPUTS: Proof that merge/move/rename/edit/alias/delete/link/add-note/
#          merge-suggestions produce correct commits, enforce same-type merges,
#          sandbox paths, record User Context, and map ManagementError → 400.
# CONSTRAINTS: No network, no real LLM.

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest

from tests._fixtures import FakeLLM, build_worktree, write_person

from diffmem.consolidator_agent._shared import extract_semantic_index, rebuild_master_index
from diffmem.consolidator_agent.management import (
    ManagementAgent,
    ManagementError,
    _append_user_context,
)


def _agent(wt: Path, llm: FakeLLM | None = None) -> ManagementAgent:
    return ManagementAgent(
        repo_path=str(wt),
        user_id="alex",
        openrouter_api_key="test-key",
        model="test-model",
        llm_call=llm,
    )


def _merged_maya_payload(loser_stem: str) -> str:
    si = {
        "name": "Maya Chen",
        "aliases": [],
        "type": "human",
        "role": "VP of Technology",
        "strength": "High",
        "hard_cues": ["Acme", "Project X"],
        "soft_cues": [],
        "emotional_cues": [],
        "related_entities": ["alex"],
        "file": "memories/people/maya_chen.md",
    }
    return (
        f"# Person: Maya Chen\n\n## Role\nVP of Technology at Acme.\n\n"
        f"## Merged from {loser_stem}\n\n## SEMANTIC INDEX\n"
        + json.dumps(si, separators=(",", ":"))
        + "\n"
    )


# ── User Context ──────────────────────────────────────────────────────────────


def test_user_context_creates_section_and_appends():
    doc = "# Person: Maya\n\n## Role\nVP.\n"
    out = _append_user_context(doc, "Same person, spells it Joneson.", "merge")
    assert "## User Context" in out
    assert "(merge): Same person, spells it Joneson." in out
    # Second note appends a second bullet.
    out2 = _append_user_context(out, "Second note.", "note")
    assert out2.count("- **") == 2


# ── MERGE ─────────────────────────────────────────────────────────────────────


def _seed_two_mayas(wt: Path) -> None:
    write_person(
        wt, filename="maya_chen.md", name="Maya Chen", body="VP of Technology.",
        semantic={"memory_strength": 0.9, "number_of_edits": 9,
                  "related_entities": ["alex", "acme"], "hard_cues": ["Acme", "Project X"]},
    )
    write_person(
        wt, filename="maya_b.md", name="Maya B.", body="Head of Technical Sales.",
        semantic={"memory_strength": 0.2, "number_of_edits": 1, "aliases": ["Mai"],
                  "related_entities": ["alex", "acme"], "hard_cues": ["Acme", "Project X"]},
    )


def test_merge_same_type_with_context_and_aliases(tmp_path: Path):
    wt = build_worktree(tmp_path)
    _seed_two_mayas(wt)
    llm = FakeLLM()
    llm.add_response(matches="Dedupe Merge", payload=_merged_maya_payload("maya_b"))
    a = _agent(wt, llm)

    result = a.manage_merge(
        "memories/people/maya_chen.md", ["memories/people/maya_b.md"],
        context="Same person — B was a typo'd duplicate.",
    )
    assert result["status"] == "ok"
    assert result["losers_merged"] == ["memories/people/maya_b.md"]
    assert "maya_b" in result["aliases_added"] and "Mai" in result["aliases_added"]
    assert not (wt / "memories" / "people" / "maya_b.md").exists()

    content = (wt / "memories" / "people" / "maya_chen.md").read_text(encoding="utf-8")
    si = extract_semantic_index(content)
    assert set(si["aliases"]) >= {"maya_b", "Maya B.", "Mai"}
    assert "(merge): Same person" in content  # User Context recorded

    import git
    log = git.Repo(wt).git.log("--oneline").split("\n")
    # Lines are `<hash> <subject>` — the hash prefix is not fixed-width.
    assert any("manage(merge): maya_chen ← maya_b" in m for m in log)


def test_merge_rejects_cross_type(tmp_path: Path):
    wt = build_worktree(tmp_path)
    _seed_two_mayas(wt)
    (wt / "memories" / "contexts").mkdir(exist_ok=True)
    ctx = wt / "memories" / "contexts" / "maya_theme.md"
    ctx.write_text("# Maya Theme\n\n## SEMANTIC INDEX\n" + json.dumps(
        {"name": "maya_theme", "type": "concept", "aliases": []}), encoding="utf-8")
    a = _agent(wt)
    with pytest.raises(ManagementError, match="cross-type"):
        a.manage_merge("memories/people/maya_chen.md", ["memories/contexts/maya_theme.md"])


def test_merge_dry_run_commits_nothing(tmp_path: Path):
    wt = build_worktree(tmp_path)
    _seed_two_mayas(wt)
    import git
    before = git.Repo(wt).git.rev_parse("HEAD")
    llm = FakeLLM()
    llm.add_response(matches="Dedupe Merge", payload=_merged_maya_payload("maya_b"))
    a = _agent(wt, llm)
    result = a.manage_merge(
        "memories/people/maya_chen.md", ["memories/people/maya_b.md"], dry_run=True
    )
    assert result["dry_run"] is True
    assert result["previews"][0]["preview_markdown"]
    assert git.Repo(wt).git.rev_parse("HEAD") == before  # nothing committed
    assert (wt / "memories" / "people" / "maya_b.md").exists()  # loser intact


# ── MOVE ──────────────────────────────────────────────────────────────────────


def test_move_retypes_entity(tmp_path: Path):
    wt = build_worktree(tmp_path)
    write_person(
        wt, filename="acme_corp.md", name="Acme Corp", body="A company misfiled as a person.",
        semantic={"type": "human", "hard_cues": ["acme"]},
    )
    a = _agent(wt)
    result = a.manage_move(
        ["memories/people/acme_corp.md"], "contexts", context="Not a person — it's a company."
    )
    assert result["status"] == "ok"
    assert not (wt / "memories" / "people" / "acme_corp.md").exists()
    moved = wt / "memories" / "contexts" / "acme_corp.md"
    assert moved.exists()
    si = extract_semantic_index(moved.read_text(encoding="utf-8"))
    assert si["type"] == "concept"  # contexts' index_type in the personal ontology
    assert "(moved people → contexts)" in moved.read_text(encoding="utf-8")


def test_move_rejects_unknown_type(tmp_path: Path):
    wt = build_worktree(tmp_path)
    write_person(wt, filename="x.md", name="X", body="x", semantic={})
    with pytest.raises(ManagementError, match="unknown entity type"):
        _agent(wt).manage_move(["memories/people/x.md"], "wombats")


# ── RENAME ────────────────────────────────────────────────────────────────────


def test_rename_keeps_old_stem_as_alias(tmp_path: Path):
    wt = build_worktree(tmp_path)
    write_person(
        wt, filename="maya_chen.md", name="Maya Chen", body="VP.", semantic={}
    )
    a = _agent(wt)
    result = a.manage_rename("memories/people/maya_chen.md", "Maya Chen-Wu")
    assert result["status"] == "ok"
    new_file = wt / "memories" / "people" / "maya_chen_chen_wu.md"
    # slug of "Maya Chen-Wu": maya_chen_wu — check the actual result path
    new_file = wt / "memories" / "people" / (result["new_path"].split("/")[-1])
    assert new_file.exists()
    content = new_file.read_text(encoding="utf-8")
    si = extract_semantic_index(content)
    assert si["name"] == "maya_chen_wu"
    assert "maya_chen" in si["aliases"] and "Maya Chen" in si["aliases"]
    # H1 updated anywhere in the doc (frontmatter precedes it).
    assert any(line.startswith("# ") and "Maya Chen-Wu" in line
               for line in content.split("\n"))


# ── EDIT / ALIAS / DELETE ────────────────────────────────────────────────────


def test_edit_requires_parseable_si(tmp_path: Path):
    wt = build_worktree(tmp_path)
    write_person(wt, filename="maya.md", name="Maya", body="VP.", semantic={})
    a = _agent(wt)
    with pytest.raises(ManagementError, match="semantic index"):
        a.manage_edit("memories/people/maya.md", "# Just a heading\n\nNo index.\n")


def test_edit_ok_and_commits(tmp_path: Path):
    wt = build_worktree(tmp_path)
    write_person(wt, filename="maya.md", name="Maya", body="VP.", semantic={})
    new_md = (
        "---\nname: maya\ntype: human\naliases: []\n---\n"
        "# Person: Maya\n\n## Role\nVP of Engineering (promoted).\n"
    )
    result = _agent(wt).manage_edit("memories/people/maya.md", new_md)
    assert result["status"] == "ok"
    assert (wt / "memories" / "people" / "maya.md").read_text(encoding="utf-8") == new_md


def test_alias_adds_and_dedupes(tmp_path: Path):
    wt = build_worktree(tmp_path)
    write_person(wt, filename="maya.md", name="Maya", body="VP.",
                 semantic={"aliases": ["M"]})
    result = _agent(wt).manage_alias("memories/people/maya.md", ["M", "Mai", "Maya B"])
    assert result["added"] == ["Mai", "Maya B"]
    si = extract_semantic_index((wt / "memories" / "people" / "maya.md").read_text(encoding="utf-8"))
    assert set(si["aliases"]) == {"M", "Mai", "Maya B"}


def test_delete_removes_and_rebuilds_index(tmp_path: Path):
    wt = build_worktree(tmp_path)
    write_person(wt, filename="temp.md", name="Temp", body="x", semantic={})
    rebuild_master_index(wt, "alex")
    assert "temp" in (wt / "index.md").read_text(encoding="utf-8")
    result = _agent(wt).manage_delete("memories/people/temp.md")
    assert result["status"] == "ok"
    assert not (wt / "memories" / "people" / "temp.md").exists()
    assert "temp.md" not in (wt / "index.md").read_text(encoding="utf-8")


# ── LINK ──────────────────────────────────────────────────────────────────────


def test_link_bidirectional(tmp_path: Path):
    wt = build_worktree(tmp_path)
    write_person(wt, filename="maya.md", name="Maya", body="VP.", semantic={})
    write_person(wt, filename="sam_rivera.md", name="Sam Rivera", body="Partner.", semantic={})
    result = _agent(wt).manage_link(
        "memories/people/maya.md", "memories/people/sam_rivera.md", note="strategy sessions"
    )
    assert result["status"] == "ok"
    maya = extract_semantic_index((wt / "memories" / "people" / "maya.md").read_text(encoding="utf-8"))
    sam = extract_semantic_index((wt / "memories" / "people" / "sam_rivera.md").read_text(encoding="utf-8"))
    assert "sam_rivera" in maya["related_entities"]
    assert "maya" in sam["related_entities"]
    assert "[[sam_rivera]] — strategy sessions" in (wt / "memories" / "people" / "maya.md").read_text(encoding="utf-8")


# ── ADD-NOTE (LLM) ────────────────────────────────────────────────────────────


def test_add_note_weaves_and_normalizes(tmp_path: Path):
    wt = build_worktree(tmp_path)
    write_person(wt, filename="maya.md", name="Maya", body="## Role\nVP of Technology.", semantic={})
    llm = FakeLLM()
    llm.add_response(
        matches="Memory Editor",
        payload=(
            "---\nname: maya\ntype: human\naliases: []\n"
            'hard_cues: ["acme", ["nested", "cue"]]\n'
            "---\n# Person: Maya\n\n## Role\nVP of Technology.\n\n"
            "## Background\n- (2026-07-15, user) Prefers async communication.\n"
        ),
    )
    result = _agent(wt, llm).manage_add_note(
        "memories/people/maya.md", "Maya prefers async communication."
    )
    assert result["status"] == "ok"
    si = extract_semantic_index((wt / "memories" / "people" / "maya.md").read_text(encoding="utf-8"))
    # Nested cue list from the LLM was flattened before persisting (choke point).
    assert si["hard_cues"] == ["acme", "nested", "cue"]
    assert "user) Prefers async communication" in (wt / "memories" / "people" / "maya.md").read_text(encoding="utf-8")


# ── SUGGESTIONS ───────────────────────────────────────────────────────────────


def test_merge_suggestions_lists_candidate_pairs(tmp_path: Path):
    wt = build_worktree(tmp_path)
    _seed_two_mayas(wt)
    result = _agent(wt).merge_suggestions()
    assert result["count"] >= 1
    top = result["suggestions"][0]
    assert {top["a_path"], top["b_path"]} == {
        "memories/people/maya_chen.md", "memories/people/maya_b.md"
    }
    assert top["shared_cues"]  # corroboration surfaced for the UI


# ── path sandboxing ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", [
    "index.md", "timeline/2026-07.md", "../outside.md", "/abs/path.md",
    "alex.md", "nonexistent.md", "memories/people/nope.md",
])
def test_path_sandbox_rejects(tmp_path: Path, bad: str):
    wt = build_worktree(tmp_path)
    a = _agent(wt)
    with pytest.raises(ManagementError):
        a.manage_delete(bad)


# ── HTTP layer ────────────────────────────────────────────────────────────────


def _http_client(monkeypatch, tmp_path, memory: Any):
    from fastapi.testclient import TestClient
    import importlib
    import diffmem.server as server_mod
    importlib.reload(server_mod)
    from concurrent.futures import ThreadPoolExecutor
    from diffmem.executor.inline import InlineExecutor

    monkeypatch.setenv("DEFAULT_MODEL", "test-model")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.setenv("REQUIRE_AUTH", "false")
    server_mod.memory_instances["alex"] = memory
    monkeypatch.setattr(server_mod, "get_memory_instance", lambda uid, allow_unboarded=False: memory)

    async def noop_backup(uid):
        return None
    monkeypatch.setattr(server_mod, "backup_user", noop_backup)
    server_mod.app.state.executor = InlineExecutor(ThreadPoolExecutor(max_workers=2))
    return TestClient(server_mod.app)


def test_http_move_maps_management_error_to_400(monkeypatch, tmp_path):
    class M:
        def manage_move(self, paths, to_type, context=None):
            raise ManagementError("unknown entity type 'wombats'")
    client = _http_client(monkeypatch, tmp_path, M())
    r = client.post("/memory/alex/manage/move", json={"paths": ["memories/people/x.md"], "to_type": "wombats"})
    assert r.status_code == 400
    assert "unknown entity type" in r.json()["detail"]


def test_http_link_returns_result(monkeypatch, tmp_path):
    class M:
        def manage_link(self, path, target_path, note=None):
            return {"status": "ok", "tool": "link", "commits": ["c1"], "summary": "Linked."}
    client = _http_client(monkeypatch, tmp_path, M())
    r = client.post("/memory/alex/manage/link", json={"path": "memories/people/a.md", "target_path": "memories/people/b.md"})
    assert r.status_code == 200
    assert r.json()["tool"] == "link"


def test_http_merge_suggestions_get(monkeypatch, tmp_path):
    class M:
        def merge_suggestions(self, name_threshold=None):
            return {"status": "ok", "count": 0, "suggestions": [], "summary": "none"}
    client = _http_client(monkeypatch, tmp_path, M())
    r = client.get("/memory/alex/manage/merge-suggestions?name_threshold=0.7")
    assert r.status_code == 200
    assert r.json()["count"] == 0


def test_http_merge_job_path(monkeypatch, tmp_path):
    """Merge (LLM op) goes through the executor as a job; result merges into the
    success response shape (inline executor, sync default)."""
    from tests._fixtures import build_worktree as bw

    wt = bw(tmp_path)
    _seed_two_mayas(wt)
    from diffmem.api import DiffMemory
    monkeypatch.setenv("DEFAULT_MODEL", "test-model")
    memory = DiffMemory(str(wt), "alex", "dummy", "test-model")

    # Inject fake LLM at the agent level.
    import diffmem.consolidator_agent.management as mgmt
    llm = FakeLLM()
    llm.add_response(matches="Dedupe Merge", payload=_merged_maya_payload("maya_b"))
    orig_init = mgmt.ManagementAgent.__init__
    def patched(self, *a, **k):
        k.setdefault("llm_call", llm)
        orig_init(self, *a, **k)
    monkeypatch.setattr(mgmt.ManagementAgent, "__init__", patched)

    client = _http_client(monkeypatch, tmp_path, memory)
    r = client.post("/memory/alex/manage/merge", json={
        "survivor_path": "memories/people/maya_chen.md",
        "loser_paths": ["memories/people/maya_b.md"],
        "context": "dup cleanup",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    # Sync job path: the executor's {status: success, metadata} response gets
    # the manage result dict merged over it — `status` becomes the result's
    # "ok" and all result fields are flattened in alongside metadata.job_id.
    assert body["status"] == "ok"
    assert "job_id" in body["metadata"]
    assert body["losers_merged"] == ["memories/people/maya_b.md"]
    assert not (wt / "memories" / "people" / "maya_b.md").exists()


def test_http_cross_type_merge_maps_to_400(monkeypatch, tmp_path):
    """ManagementError raised inside an LLM manage JOB must surface as HTTP 400
    (embedded via _manage_work), not an opaque executor 500."""
    from tests._fixtures import build_worktree as bw

    wt = bw(tmp_path)
    write_person(wt, filename="maya.md", name="Maya", body="VP.", semantic={})
    (wt / "memories" / "contexts").mkdir(exist_ok=True)
    (wt / "memories" / "contexts" / "theme.md").write_text(
        "---\nname: theme\ntype: concept\naliases: []\n---\n# Theme\n",
        encoding="utf-8",
    )
    from diffmem.api import DiffMemory
    monkeypatch.setenv("DEFAULT_MODEL", "test-model")
    memory = DiffMemory(str(wt), "alex", "dummy", "test-model")
    client = _http_client(monkeypatch, tmp_path, memory)
    r = client.post("/memory/alex/manage/merge", json={
        "survivor_path": "memories/people/maya.md",
        "loser_paths": ["memories/contexts/theme.md"],
    })
    assert r.status_code == 400, r.text
    assert "cross-type" in r.json()["detail"]


# ── REVIEWED COMMIT (dry-run preview → user edit → commit, no 2nd LLM) ───────


def test_merge_reviewed_markdown_commits_verbatim_no_llm(tmp_path: Path):
    wt = build_worktree(tmp_path)
    _seed_two_mayas(wt)
    llm = FakeLLM()  # NO scripted responses — any LLM call would yield "" →
    # deterministic fallback, which the verbatim assertions below would catch.
    a = _agent(wt, llm)

    reviewed = "# Person: Maya Chen\n\n## Role\nVP of Technology at Acme.\n\nEDITED BY USER — keeps everything.\n"
    result = a.manage_merge(
        "memories/people/maya_chen.md",
        ["memories/people/maya_b.md"],
        context="Same person, spelling variant.",
        reviewed_markdown=reviewed,
        reviewed_semantic_index={
            "name": "Maya Chen", "type": "human", "hard_cues": ["Acme"],
        },
    )

    assert result["status"] == "ok"
    assert result["reviewed"] is True
    assert len(llm.calls) == 0  # the whole point: zero LLM calls on commit

    # Survivor file contains the user's body VERBATIM + forced loser aliases
    # + the user-context note; loser file is gone.
    final = (wt / "memories" / "people" / "maya_chen.md").read_text(encoding="utf-8")
    assert "EDITED BY USER — keeps everything." in final
    assert "## Merged from maya_b" not in final  # deterministic fallback NOT used
    si = extract_semantic_index(final) or {}
    assert "maya_b" in (si.get("aliases") or [])
    assert "Mai" in (si.get("aliases") or [])
    assert "(merge): Same person, spelling variant." in final
    assert not (wt / "memories" / "people" / "maya_b.md").exists()


# ── NATIVE /entities ENDPOINT (replaces N paged run-command greps) ──────────


def test_http_list_entities_returns_index_md_json(monkeypatch, tmp_path: Path):
    from tests._fixtures import build_worktree, write_person
    from diffmem.consolidator_agent._shared import rebuild_master_index
    from diffmem.api import DiffMemory

    wt = build_worktree(tmp_path)
    write_person(wt, filename="maya.md", name="Maya", body="VP of Technology.",
                 semantic={"memory_strength": 0.9, "number_of_edits": 9,
                           "hard_cues": ["Acme"], "related_entities": ["alex"]})
    # index.md is the catalog the endpoint reads directly.
    rebuild_master_index(wt, "alex", repo=None, entity_dirs=[wt / "memories"])
    memory = DiffMemory(str(wt), "alex", "dummy", "test-model")
    client = _http_client(monkeypatch, tmp_path, memory)

    r = client.get("/memory/alex/entities")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["count"] == 1
    ent = body["entities"][0]
    assert ent["name"] == "Maya"
    assert ent["file"].endswith("maya.md")
    assert "Acme" in (ent.get("hard_cues") or [])


def test_http_list_entities_empty_when_no_index(monkeypatch, tmp_path: Path):
    from tests._fixtures import build_worktree
    from diffmem.api import DiffMemory

    wt = build_worktree(tmp_path)
    memory = DiffMemory(str(wt), "alex", "dummy", "test-model")
    client = _http_client(monkeypatch, tmp_path, memory)
    r = client.get("/memory/alex/entities")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "entities": [], "count": 0}
