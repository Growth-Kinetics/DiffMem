# CAPABILITY: Integration tests for run_dedupe.
# INPUTS: tmp_path -> built fixture worktree with two Mayas.
# OUTPUTS: Verifies merge, survivor selection, aliases, commit message,
#          and that no-match cases are silently skipped.
# CONSTRAINTS: No network. FakeLLM injected via ConsolidatorAgent(llm_call=...).

from __future__ import annotations

import json
from pathlib import Path

import git
import pytest

from tests._fixtures import FakeLLM, build_worktree, write_person

from diffmem.consolidator_agent.agent import ConsolidatorAgent
from diffmem.consolidator_agent._shared import extract_semantic_index


# --- helpers ------------------------------------------------------------------


def _maya_long() -> dict:
    return {
        "name": "Maya",
        "type": "human",
        "role": "VP of Technology, Acme",
        "strength": "High",
        "hard_cues": ["Acme", "Project X", "Phoenix", "Northgate", "Helios"],
        "soft_cues": ["aligning the partners", "high-level operator"],
        "emotional_cues": ["tactical friction"],
        "related_entities": ["alex", "sam_rivera", "greg", "david"],
        "memory_strength": 0.9,
        "number_of_edits": 9,
    }


def _maya_short() -> dict:
    return {
        "name": "Maya",
        "type": "human",
        "role": "Head of Technical Sales at Acme",
        "strength": "Low",
        "hard_cues": ["Acme", "Globex", "Northgate", "Northgate region"],
        "soft_cues": ["frenemy dynamic"],
        "emotional_cues": ["competitive tension"],
        "related_entities": ["alex", "project_x", "acme"],
        "memory_strength": 0.2,
        "number_of_edits": 1,
    }


def _make_merged_content(survivor_path: str, loser_stem: str) -> str:
    """The fake-LLM's merge payload — minimal but well-formed."""
    si = _maya_long()
    si["aliases"] = [loser_stem]
    si["hard_cues"] = sorted(set(_maya_long()["hard_cues"]) | set(_maya_short()["hard_cues"]))
    si["related_entities"] = sorted(
        set(_maya_long()["related_entities"]) | set(_maya_short()["related_entities"])
    )
    si["file"] = survivor_path
    return (
        "# Maya (from Acme)\n\n"
        "## Role\nVP of Technology, Acme. Also covers technical sales across APAC.\n\n"
        "## Merged from " + loser_stem + "\nIncludes prior Head-of-Technical-Sales framing.\n\n"
        "## SEMANTIC INDEX\n" + json.dumps(si, separators=(",", ":")) + "\n"
    )


# --- the canonical test -------------------------------------------------------


def test_two_mayas_merge(tmp_path: Path) -> None:
    wt = build_worktree(tmp_path)

    # The high-strength Maya.
    write_person(
        wt,
        filename="maya_(acme).md",
        name="Maya (from Acme)",
        body="Maya is the VP of Technology for Acme. Based in Phoenix. Project X account.",
        semantic=_maya_long(),
        commit_msg="add maya (long)",
    )
    # The low-strength Maya.
    write_person(
        wt,
        filename="maya.md",
        name="Maya",
        body="Maya is the Head of Technical Sales at Acme. Pushes sales in APAC.",
        semantic=_maya_short(),
        commit_msg="add maya (short)",
    )

    llm = FakeLLM()
    llm.add_response(
        matches="Dedupe Judge",
        payload={
            "same_entity": True,
            "confidence": "high",
            "rationale": "Both describe Maya at Acme.",
        },
    )
    llm.add_response(
        matches="Dedupe Merge",
        payload=_make_merged_content("memories/people/maya_(acme).md", "maya"),
    )

    agent = ConsolidatorAgent(
        repo_path=str(wt),
        user_id="alex",
        openrouter_api_key="dummy",
        model="test-model",
        llm_call=llm,
    )
    result = agent.run_dedupe()

    assert result["status"] == "ok"
    assert result["tool"] == "dedupe"
    assert result["candidates_evaluated"] >= 1
    assert result["merges_performed"] == 1
    assert len(result["commits"]) >= 1

    survivor = wt / "memories" / "people" / "maya_(acme).md"
    loser = wt / "memories" / "people" / "maya.md"
    assert survivor.exists(), "survivor file should remain"
    assert not loser.exists(), "loser file should be git-removed"

    # Alias preserved in survivor's structured metadata (frontmatter in v2;
    # legacy files carry a trailing SEMANTIC INDEX block — extract handles both).
    content = survivor.read_text(encoding="utf-8")
    si = extract_semantic_index(content)
    assert si is not None, "survivor must carry structured metadata"
    assert "maya" in si.get("aliases", []), f"aliases must include 'maya', got {si.get('aliases')}"

    # Commit message matches the protocol.
    repo = git.Repo(wt)
    last_msgs = [c.message.strip() for c in repo.iter_commits(max_count=5)]
    assert any(m.startswith("consolidate(dedupe): maya_(acme) \u2190 maya") for m in last_msgs), last_msgs

    # index.md rebuilt.
    assert (wt / "index.md").exists()


def test_low_confidence_does_not_merge(tmp_path: Path) -> None:
    wt = build_worktree(tmp_path)
    write_person(
        wt,
        filename="maya_(acme).md",
        name="Maya (from Acme)",
        body="VP at Acme.",
        semantic=_maya_long(),
    )
    write_person(
        wt,
        filename="maya.md",
        name="Maya",
        body="Head of technical sales at Acme.",
        semantic=_maya_short(),
    )

    llm = FakeLLM()
    llm.add_response(
        matches="Dedupe Judge",
        payload={"same_entity": True, "confidence": "low", "rationale": "Unsure."},
    )

    agent = ConsolidatorAgent(
        repo_path=str(wt),
        user_id="alex",
        openrouter_api_key="dummy",
        model="test-model",
        llm_call=llm,
    )
    result = agent.run_dedupe()

    assert result["candidates_evaluated"] >= 1
    assert result["merges_performed"] == 0
    assert (wt / "memories" / "people" / "maya.md").exists()
    assert (wt / "memories" / "people" / "maya_(acme).md").exists()


def test_unrelated_entities_not_paired(tmp_path: Path) -> None:
    wt = build_worktree(tmp_path)
    write_person(
        wt,
        filename="maya.md",
        name="Maya",
        body="VP at Acme.",
        semantic=_maya_long(),
    )
    write_person(
        wt,
        filename="beatrice.md",
        name="Priya",
        body="Friend from Lisbon.",
        semantic={
            "type": "human",
            "role": "friend",
            "hard_cues": ["Lisbon", "salsa"],
            "related_entities": ["alex"],
            "memory_strength": 0.4,
        },
    )

    llm = FakeLLM()  # Judge will never be called.
    agent = ConsolidatorAgent(
        repo_path=str(wt),
        user_id="alex",
        openrouter_api_key="dummy",
        model="test-model",
        llm_call=llm,
    )
    result = agent.run_dedupe()
    assert result["candidates_evaluated"] == 0
    assert result["merges_performed"] == 0
    # Judge prompt should never have been invoked.
    assert all("Dedupe Judge" not in c["prompt"] for c in llm.calls)


def test_survivor_is_higher_strength_even_if_listed_second(tmp_path: Path) -> None:
    """Order independence: regardless of which file is created first, the
    higher-memory-strength one survives."""
    wt = build_worktree(tmp_path)
    # Short (low strength) created FIRST.
    write_person(
        wt,
        filename="maya.md",
        name="Maya",
        body="Sales.",
        semantic=_maya_short(),
    )
    write_person(
        wt,
        filename="maya_(acme).md",
        name="Maya (from Acme)",
        body="VP technology.",
        semantic=_maya_long(),
    )

    llm = FakeLLM()
    llm.add_response(
        matches="Dedupe Judge",
        payload={"same_entity": True, "confidence": "high", "rationale": "yes"},
    )
    llm.add_response(
        matches="Dedupe Merge",
        payload=_make_merged_content("memories/people/maya_(acme).md", "maya"),
    )

    agent = ConsolidatorAgent(
        repo_path=str(wt),
        user_id="alex",
        openrouter_api_key="dummy",
        model="test-model",
        llm_call=llm,
    )
    result = agent.run_dedupe()
    assert result["merges_performed"] == 1
    assert (wt / "memories" / "people" / "maya_(acme).md").exists()
    assert not (wt / "memories" / "people" / "maya.md").exists()
