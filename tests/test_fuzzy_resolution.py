# CAPABILITY: Tests for writer fuzzy entity resolution + dedupe alias propagation.
# INPUTS: tmp_path worktrees + FakeLLM (no network, no real LLM).
# OUTPUTS: Proof that spelling/punctuation/typo variants resolve to the SAME
#          entity file instead of spawning duplicates (the #1 duplicate source
#          in ChatBarry production), and that merges propagate ALL loser name
#          variants into the survivor's aliases (alias-redirect dedupe).
# CONSTRAINTS: Deterministic tiers only — normalization + similarity, no LLM
#              in the resolution path itself.

from __future__ import annotations

import json
from pathlib import Path

from tests._fixtures import FakeLLM, build_worktree, write_person

from diffmem.consolidator_agent.agent import ConsolidatorAgent
from diffmem.consolidator_agent._shared import rebuild_master_index, extract_semantic_index
from diffmem.writer_agent.agent import WriterAgent, _normalize_name, FUZZY_NAME_THRESHOLD


def _writer(wt: Path) -> WriterAgent:
    """WriterAgent against the fixture worktree — no LLM calls are made."""
    return WriterAgent(
        repo_path=str(wt),
        user_id="alex",
        openrouter_api_key="test-key",
        model="test-model",
        ontology=None,  # personal ontology → memories/ layout, matches fixtures
    )


def _benjamin(wt: Path) -> Path:
    p = write_person(
        wt,
        filename="benjamin_powell.md",
        name="Benjamin Powell",
        body="Owner-operator. Builds things.",
        semantic={"hard_cues": ["owner"], "related_entities": ["alex"]},
        commit_msg="add benjamin",
    )
    return p


# --- _normalize_name -----------------------------------------------------------


def test_normalize_strips_punctuation_diacritics_whitespace():
    assert _normalize_name("Jean-Pierre Ó Sé") == "jeanpierreose"
    assert _normalize_name("Dr. Maya Chen, PhD") == "drmayachenphd"
    assert _normalize_name("MAYA   chen") == "mayachen"
    assert _normalize_name("maya_chen") == "mayachen"
    assert _normalize_name("") == ""
    assert _normalize_name(42) == ""  # type: ignore[arg-type]


# --- resolution tiers -----------------------------------------------------------


def test_exact_lookup_still_works(tmp_path: Path) -> None:
    wt = build_worktree(tmp_path)
    _benjamin(wt)
    rebuild_master_index(wt, "alex")
    w = _writer(wt)
    assert w._resolve_entity_file_path("Benjamin Powell", "people") == wt / "memories" / "people" / "benjamin_powell.md"


def test_punctuation_variant_resolves(tmp_path: Path) -> None:
    """'benjamin-powell' used to MISS exact-lower and spawn a duplicate."""
    wt = build_worktree(tmp_path)
    _benjamin(wt)
    rebuild_master_index(wt, "alex")
    w = _writer(wt)
    resolved = w._resolve_entity_file_path("Benjamin-Powell", "people")
    assert resolved == wt / "memories" / "people" / "benjamin_powell.md"


def test_typo_variant_resolves_fuzzy(tmp_path: Path) -> None:
    """'Benjamen Powell' (ratio ~0.96 vs 'Benjamin Powell') resolves via the
    fuzzy tier instead of creating benjamen_powell.md."""
    wt = build_worktree(tmp_path)
    _benjamin(wt)
    rebuild_master_index(wt, "alex")
    w = _writer(wt)
    resolved = w._resolve_entity_file_path("Benjamen Powell", "people")
    assert resolved == wt / "memories" / "people" / "benjamin_powell.md"


def test_alias_typo_resolves(tmp_path: Path) -> None:
    wt = build_worktree(tmp_path)
    write_person(
        wt,
        filename="maya_chen.md",
        name="Maya Chen",
        body="VP Technology.",
        semantic={"aliases": ["Maya B"]},
        commit_msg="add maya",
    )
    rebuild_master_index(wt, "alex")
    w = _writer(wt)
    # Normalized alias hit.
    assert w._resolve_entity_file_path("Maya B.", "people") == wt / "memories" / "people" / "maya_chen.md"
    # Typo'd alias, fuzzy hit.
    assert w._resolve_entity_file_path("Maya Bee", "people") == wt / "memories" / "people" / "maya_chen.md"


def test_short_form_resolves_via_containment(tmp_path: Path):
    """'maya' (≥4 chars) contained in 'maya chen' → resolves to maya_chen.md."""
    wt = build_worktree(tmp_path)
    write_person(
        wt,
        filename="maya_chen.md",
        name="Maya Chen",
        body="VP Technology.",
        semantic={},
        commit_msg="add maya",
    )
    rebuild_master_index(wt, "alex")
    w = _writer(wt)
    assert w._resolve_entity_file_path("Maya", "people") == wt / "memories" / "people" / "maya_chen.md"


def test_unrelated_name_still_unresolved(tmp_path: Path) -> None:
    """The fuzzy tiers must NOT invent matches: a genuinely different name
    resolves to None (and would correctly become a new entity)."""
    wt = build_worktree(tmp_path)
    _benjamin(wt)
    rebuild_master_index(wt, "alex")
    w = _writer(wt)
    assert w._resolve_entity_file_path("Rajesh Koothrappali", "people") is None


def test_fuzzy_threshold_is_high(tmp_path: Path) -> None:
    """A name close-ish but under the bar must NOT match (dedupe safety)."""
    wt = build_worktree(tmp_path)
    _benjamin(wt)
    rebuild_master_index(wt, "alex")
    w = _writer(wt)
    # ratio("benpowellcalifornia","benjaminpowell") is well under 0.85.
    assert w._resolve_entity_file_path("Ben Powell California", "people") is None


# --- dedupe alias propagation ----------------------------------------------------


def test_merge_propagates_all_loser_name_variants(tmp_path: Path) -> None:
    """After a merge the survivor's aliases must include the loser's stem,
    SEMANTIC INDEX name, AND its aliases — otherwise the writer re-creates
    the loser the next time an old spelling appears."""
    wt = build_worktree(tmp_path)

    write_person(
        wt,
        filename="maya_chen.md",
        name="Maya Chen",
        body="Maya is the VP of Technology.",
        semantic={
            "memory_strength": 0.9,
            "number_of_edits": 9,
            "related_entities": ["alex", "acme"],
            "hard_cues": ["Acme", "Project X", "Phoenix"],
        },
        commit_msg="add maya (long)",
    )
    write_person(
        wt,
        filename="maya_b.md",
        name="Maya B.",
        body="Maya B. is Head of Technical Sales.",
        semantic={
            "memory_strength": 0.2,
            "number_of_edits": 1,
            "aliases": ["Mai", "M.B."],
            "related_entities": ["alex", "acme"],
            "hard_cues": ["Acme", "Project X"],
        },
        commit_msg="add maya (short)",
    )

    # The fake merge payload omits the loser's aliases entirely — the
    # propagation layer must add them regardless of what the LLM returned.
    merged_si = {
        "name": "Maya Chen",
        "type": "human",
        "aliases": [],
        "strength": "High",
        "hard_cues": ["Acme", "Project X"],
        "soft_cues": [],
        "emotional_cues": [],
        "related_entities": ["alex"],
        "file": "memories/people/maya_chen.md",
    }
    merged = (
        "# Maya Chen\n\n## Role\nVP of Technology.\n\n"
        "## Merged from maya_b\n\n## SEMANTIC INDEX\n"
        + json.dumps(merged_si, separators=(",", ":"))
        + "\n"
    )

    llm = FakeLLM()
    llm.add_response(
        matches="Dedupe Judge",
        payload={"same_entity": True, "confidence": "high", "rationale": "Same person."},
    )
    llm.add_response(matches="Dedupe Merge", payload=merged)

    agent = ConsolidatorAgent(
        repo_path=str(wt),
        user_id="alex",
        openrouter_api_key="test-key",
        model="test-model",
        llm_call=llm,
    )
    result = agent.run_dedupe()
    assert result["merges_performed"] == 1, result

    survivor_si = extract_semantic_index(
        (wt / "memories" / "people" / "maya_chen.md").read_text(encoding="utf-8")
    )
    aliases = set(survivor_si.get("aliases") or [])
    assert "maya_b" in aliases          # loser stem
    assert "Maya B." in aliases         # loser SEMANTIC INDEX name
    assert "Mai" in aliases             # loser alias
    assert "M.B." in aliases            # loser alias
    assert not (wt / "memories" / "people" / "maya_b.md").exists()

    # And the writer now resolves the old spellings to the survivor:
    rebuild_master_index(wt, "alex")
    w = _writer(wt)
    for variant in ("Maya B.", "Mai", "maya_b"):
        assert w._resolve_entity_file_path(variant, "people") == wt / "memories" / "people" / "maya_chen.md", variant


def test_threshold_value_sane():
    """Guard the constant itself — lowering it silently would blur distinct people."""
    assert 0.8 <= FUZZY_NAME_THRESHOLD <= 0.95
