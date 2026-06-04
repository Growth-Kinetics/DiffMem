# CAPABILITY: Integration tests for run_redistribute.
# INPUTS: tmp_path -> fixture worktree with one oversized user entity.
# OUTPUTS: Verifies attributed-move + new-context extraction, soft cap respected,
#          commit prefix correct, smaller entities preferred (balancing rule).
# CONSTRAINTS: FakeLLM provides scripted plans; no network.

from __future__ import annotations

import json
from pathlib import Path

import git

from tests._fixtures import FakeLLM, build_worktree, write_person

from diffmem.consolidator_agent.agent import ConsolidatorAgent
from diffmem.consolidator_agent._shared import estimate_tokens


# --- helpers ------------------------------------------------------------------


ANDRE_BLOCK = (
    "## Andre Section\n"
    "Andre is the VP of Technology for Sapient (Publicis). He is based out of Chicago "
    "and has been driving the McDonald's account. Recent collaboration has been productive "
    "but punctuated by tactical friction. Andre has surfaced repeatedly in Alex's strategy "
    "discussions and is a key external partner.\n"
)

DATA_GOV_BLOCK = (
    "## Data Governance Reflections\n"
    "Across the last year, the user has developed a strong perspective on data governance "
    "as a foundational discipline that bridges IT and business strategy. Three principles "
    "have emerged: ownership clarity, semantic layering, and progressive disclosure. These "
    "are increasingly central to how the user frames consulting engagements.\n"
)


def _write_oversized_user_entity(wt: Path, user_id: str = "alex") -> Path:
    """Overwrite the default user file with one ~3x our test soft cap.

    We use a 2_000-token soft cap in tests so we don't have to actually
    write 32k tokens. The test asserts behaviour, not magic numbers.
    """
    user_file = wt / f"{user_id}.md"
    si = {
        "name": user_id.title(),
        "aliases": [],
        "type": "human",
        "role": "user",
        "strength": "High",
        "hard_cues": [user_id],
        "soft_cues": [],
        "emotional_cues": [],
        "related_entities": ["andre"],
        "file": f"{user_id}.md",
        "memory_strength": 1.0,
    }
    # Build a body well over 2_000 tokens (~8_000 chars).
    filler = "Filler paragraph about generic user life events. " * 200  # ~10_000 chars
    body = (
        f"# {user_id.title()} Profile\n\n"
        "## Core Identity [ALWAYS_LOAD]\n- Test user.\n\n"
        + ANDRE_BLOCK
        + "\n"
        + DATA_GOV_BLOCK
        + "\n"
        + "## Misc\n"
        + filler
        + "\n## SEMANTIC INDEX\n"
        + json.dumps(si, separators=(",", ":"))
        + "\n"
    )
    user_file.write_text(body, encoding="utf-8")
    repo = git.Repo(wt)
    repo.index.add([f"{user_id}.md"])
    repo.index.commit("inflate user entity for redistribute test")
    return user_file


# --- the canonical test -------------------------------------------------------


def test_redistribute_moves_attributed_section_and_extracts_orphan(tmp_path: Path) -> None:
    wt = build_worktree(tmp_path)

    # Small target entity — the balancing rule should prefer it.
    write_person(
        wt,
        filename="andre.md",
        name="Andre",
        body="Andre is the VP at Sapient.",
        semantic={
            "type": "human",
            "role": "VP Technology",
            "hard_cues": ["Sapient", "Chicago"],
            "related_entities": ["alex"],
            "memory_strength": 0.6,
        },
    )

    user_file = _write_oversized_user_entity(wt)
    soft_cap = 2_000
    assert estimate_tokens(user_file.read_text(encoding="utf-8")) > soft_cap

    llm = FakeLLM()
    # Plan returned by the analyze prompt — references both blocks LITERALLY
    # so the search-and-replace can find them.
    plan = {
        "moves": [
            {
                "source_section": ANDRE_BLOCK,
                "target_entity": "memories/people/andre.md",
                "reason": "section is about Andre at Sapient",
                "extracted_content": ANDRE_BLOCK,
            }
        ],
        "new_contexts": [
            {
                "name": "Data Governance",
                "source_section": DATA_GOV_BLOCK,
                "extracted_content": "# Data Governance\n\n" + DATA_GOV_BLOCK,
                "reason": "thematic block, no clear subject",
            }
        ],
    }
    llm.add_response(matches="Redistribute Analyst", payload=plan)
    # The build_semantic_index call:
    llm.add_response(
        matches="Semantic Index Builder",
        payload={
            "name": "Data Governance",
            "aliases": [],
            "type": "concept",
            "role": "thematic reference",
            "strength": "Low",
            "hard_cues": ["data governance", "semantic layering"],
            "soft_cues": [],
            "emotional_cues": [],
            "related_entities": ["alex"],
        },
    )

    agent = ConsolidatorAgent(
        repo_path=str(wt),
        user_id="alex",
        openrouter_api_key="dummy",
        model="test-model",
        llm_call=llm,
    )
    result = agent.run_redistribute(soft_cap_tokens=soft_cap)

    assert result["status"] == "ok"
    assert result["tool"] == "redistribute"
    assert result["oversized_entities"] >= 1
    assert result["total_moves"] == 1
    assert result["new_contexts"] == 1
    assert len(result["commits"]) >= 1

    # User entity slimmed: Andre block and data gov block removed.
    new_user = user_file.read_text(encoding="utf-8")
    assert "Andre Section" not in new_user, "attributed move should remove Andre section"
    assert "Data Governance Reflections" not in new_user, "orphan extraction should remove data-gov section"
    # SEMANTIC INDEX preserved.
    assert "## SEMANTIC INDEX" in new_user

    # Andre file received the moved section.
    andre = (wt / "memories" / "people" / "andre.md").read_text(encoding="utf-8")
    assert "Andre Section" in andre
    # SEMANTIC INDEX still last in Andre file.
    assert andre.rstrip().endswith("}") or "SEMANTIC INDEX" in andre.split("Andre Section")[-1]

    # New context file created with SEMANTIC INDEX.
    new_ctx = wt / "memories" / "contexts" / "data_governance.md"
    assert new_ctx.exists()
    new_ctx_content = new_ctx.read_text(encoding="utf-8")
    assert "## SEMANTIC INDEX" in new_ctx_content
    # Content preserved.
    assert "Data Governance Reflections" in new_ctx_content

    # Commit message matches the protocol.
    repo = git.Repo(wt)
    last_msgs = [c.message.strip() for c in repo.iter_commits(max_count=5)]
    assert any(m.startswith("consolidate(redistribute): slim alex") for m in last_msgs), last_msgs

    # Master index rebuilt.
    assert (wt / "index.md").exists()


def test_no_oversized_entities_is_noop(tmp_path: Path) -> None:
    wt = build_worktree(tmp_path)
    write_person(
        wt,
        filename="andre.md",
        name="Andre",
        body="Short.",
        semantic={"type": "human", "memory_strength": 0.6},
    )
    llm = FakeLLM()
    agent = ConsolidatorAgent(
        repo_path=str(wt),
        user_id="alex",
        openrouter_api_key="dummy",
        model="test-model",
        llm_call=llm,
    )
    result = agent.run_redistribute(soft_cap_tokens=100_000)
    assert result["status"] == "ok"
    assert result["oversized_entities"] == 0
    assert result["commits"] == []
    # No LLM calls should have happened.
    assert llm.calls == []


def test_balancing_rule_in_candidates_block(tmp_path: Path) -> None:
    """The candidates block sent to the analyze prompt is sorted ASC by tokens."""
    wt = build_worktree(tmp_path)
    write_person(
        wt,
        filename="big_one.md",
        name="Big One",
        body=("# Big\n\n" + ("Lots of content. " * 500)),
        semantic={"type": "human"},
    )
    write_person(
        wt,
        filename="small_one.md",
        name="Small One",
        body="# Small\n\nTiny.",
        semantic={"type": "human"},
    )
    _write_oversized_user_entity(wt)

    captured: list = []

    def llm_fn(prompt: str, is_json: bool):
        if "Redistribute Analyst" in prompt:
            captured.append(prompt)
            return {"moves": [], "new_contexts": []}
        return {}

    agent = ConsolidatorAgent(
        repo_path=str(wt),
        user_id="alex",
        openrouter_api_key="dummy",
        model="test-model",
        llm_call=llm_fn,
    )
    agent.run_redistribute(soft_cap_tokens=2_000)

    # Find the prompt where alex.md is the source — both big_one and small_one
    # must be in its candidates block.
    marker = "CANDIDATE TARGET ENTITIES"
    target_prompt = next(
        (p for p in captured if "`alex.md`" in p or "SOURCE FILE \u2014 `alex.md`" in p),
        captured[0] if captured else "",
    )
    assert marker in target_prompt
    cand_block = target_prompt[target_prompt.find(marker):]
    small_idx = cand_block.find("small_one.md")
    big_idx = cand_block.find("big_one.md")
    assert small_idx != -1 and big_idx != -1, f"both candidates missing from block: {cand_block[:500]}"
    assert small_idx < big_idx, "candidates block must list smaller entities first"
