# CAPABILITY: Integration tests for run_dedupe.
# INPUTS: tmp_path -> built fixture worktree with two Andres.
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


# --- helpers ------------------------------------------------------------------


def _andre_long() -> dict:
    return {
        "name": "Andre",
        "type": "human",
        "role": "VP of Technology, Sapient",
        "strength": "High",
        "hard_cues": ["Sapient", "McDonald's", "Chicago", "Japan", "Snowflake"],
        "soft_cues": ["aligning the sharks", "high-level operator"],
        "emotional_cues": ["tactical friction"],
        "related_entities": ["alex", "lars_orloff", "greg", "david"],
        "memory_strength": 0.9,
        "number_of_edits": 9,
    }


def _andre_short() -> dict:
    return {
        "name": "Andre",
        "type": "human",
        "role": "Head of Technical Sales at Sapient",
        "strength": "Low",
        "hard_cues": ["Sapient", "Publicis", "Japan", "APAC markets"],
        "soft_cues": ["frenemy dynamic"],
        "emotional_cues": ["competitive tension"],
        "related_entities": ["alex", "mcdonalds_japan", "sapient"],
        "memory_strength": 0.2,
        "number_of_edits": 1,
    }


def _make_merged_content(survivor_path: str, loser_stem: str) -> str:
    """The fake-LLM's merge payload — minimal but well-formed."""
    si = _andre_long()
    si["aliases"] = [loser_stem]
    si["hard_cues"] = sorted(set(_andre_long()["hard_cues"]) | set(_andre_short()["hard_cues"]))
    si["related_entities"] = sorted(
        set(_andre_long()["related_entities"]) | set(_andre_short()["related_entities"])
    )
    si["file"] = survivor_path
    return (
        "# Andre (from Sapient)\n\n"
        "## Role\nVP of Technology, Sapient. Also covers technical sales across APAC.\n\n"
        "## Merged from " + loser_stem + "\nIncludes prior Head-of-Technical-Sales framing.\n\n"
        "## SEMANTIC INDEX\n" + json.dumps(si, separators=(",", ":")) + "\n"
    )


# --- the canonical test -------------------------------------------------------


def test_two_andres_merge(tmp_path: Path) -> None:
    wt = build_worktree(tmp_path)

    # The high-strength Andre.
    write_person(
        wt,
        filename="andre_(sapient).md",
        name="Andre (from Sapient)",
        body="Andre is the VP of Technology for Sapient. Based in Chicago. McDonald's account.",
        semantic=_andre_long(),
        commit_msg="add andre (long)",
    )
    # The low-strength Andre.
    write_person(
        wt,
        filename="andre.md",
        name="Andre",
        body="Andre is the Head of Technical Sales at Sapient. Pushes sales in APAC.",
        semantic=_andre_short(),
        commit_msg="add andre (short)",
    )

    llm = FakeLLM()
    llm.add_response(
        matches="Dedupe Judge",
        payload={
            "same_entity": True,
            "confidence": "high",
            "rationale": "Both describe Andre at Sapient.",
        },
    )
    llm.add_response(
        matches="Dedupe Merge",
        payload=_make_merged_content("memories/people/andre_(sapient).md", "andre"),
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

    survivor = wt / "memories" / "people" / "andre_(sapient).md"
    loser = wt / "memories" / "people" / "andre.md"
    assert survivor.exists(), "survivor file should remain"
    assert not loser.exists(), "loser file should be git-removed"

    # Alias preserved in survivor's SEMANTIC INDEX.
    content = survivor.read_text(encoding="utf-8")
    assert "## SEMANTIC INDEX" in content
    si_line = content.split("## SEMANTIC INDEX", 1)[1].strip().splitlines()[0]
    si = json.loads(si_line)
    assert "andre" in si.get("aliases", []), f"aliases must include 'andre', got {si.get('aliases')}"

    # Commit message matches the protocol.
    repo = git.Repo(wt)
    last_msgs = [c.message.strip() for c in repo.iter_commits(max_count=5)]
    assert any(m.startswith("consolidate(dedupe): andre_(sapient) \u2190 andre") for m in last_msgs), last_msgs

    # index.md rebuilt.
    assert (wt / "index.md").exists()


def test_low_confidence_does_not_merge(tmp_path: Path) -> None:
    wt = build_worktree(tmp_path)
    write_person(
        wt,
        filename="andre_(sapient).md",
        name="Andre (from Sapient)",
        body="VP at Sapient.",
        semantic=_andre_long(),
    )
    write_person(
        wt,
        filename="andre.md",
        name="Andre",
        body="Head of technical sales at Sapient.",
        semantic=_andre_short(),
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
    assert (wt / "memories" / "people" / "andre.md").exists()
    assert (wt / "memories" / "people" / "andre_(sapient).md").exists()


def test_unrelated_entities_not_paired(tmp_path: Path) -> None:
    wt = build_worktree(tmp_path)
    write_person(
        wt,
        filename="andre.md",
        name="Andre",
        body="VP at Sapient.",
        semantic=_andre_long(),
    )
    write_person(
        wt,
        filename="beatrice.md",
        name="Beatrice",
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
        filename="andre.md",
        name="Andre",
        body="Sales.",
        semantic=_andre_short(),
    )
    write_person(
        wt,
        filename="andre_(sapient).md",
        name="Andre (from Sapient)",
        body="VP technology.",
        semantic=_andre_long(),
    )

    llm = FakeLLM()
    llm.add_response(
        matches="Dedupe Judge",
        payload={"same_entity": True, "confidence": "high", "rationale": "yes"},
    )
    llm.add_response(
        matches="Dedupe Merge",
        payload=_make_merged_content("memories/people/andre_(sapient).md", "andre"),
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
    assert (wt / "memories" / "people" / "andre_(sapient).md").exists()
    assert not (wt / "memories" / "people" / "andre.md").exists()
