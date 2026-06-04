# CAPABILITY: End-to-end smoke test of the full consolidator chain.
# INPUTS: tmp_path -> realistic fixture (oversized user entity, two Andres,
#         lars_orloff, alex co-occurrence pattern).
# OUTPUTS: Verifies dedupe + redistribute + link all fire in canonical order,
#          producing three (or more) consolidate(...) commits, slimming the
#          user entity, merging the two Andres, weaving wikilinks, and that
#          a second run produces zero new commits (chain-level idempotency).
# CONSTRAINTS: Scripted llm_call (no network). Validates orchestration, not
#              prompt quality.

from __future__ import annotations

import json
import re
from pathlib import Path

import git
import pytest

from tests._fixtures import FakeLLM, build_worktree, write_person

from diffmem.api import DiffMemory
from diffmem.consolidator_agent._shared import estimate_tokens


# --- fixture ------------------------------------------------------------------


ANDRE_LONG_SI = {
    "name": "Andre",
    "type": "human",
    "role": "VP of Technology, Sapient",
    "strength": "High",
    "hard_cues": ["Sapient", "McDonald's", "Chicago", "Japan", "Snowflake"],
    "soft_cues": ["aligning the sharks"],
    "emotional_cues": ["tactical friction"],
    "related_entities": ["alex", "lars_orloff", "greg", "david"],
    "memory_strength": 0.9,
}

ANDRE_SHORT_SI = {
    "name": "Andre",
    "type": "human",
    "role": "Head of Technical Sales at Sapient",
    "strength": "Low",
    "hard_cues": ["Sapient", "Publicis", "Japan", "APAC markets"],
    "soft_cues": ["frenemy dynamic"],
    "emotional_cues": ["competitive tension"],
    "related_entities": ["alex", "mcdonalds_japan", "sapient"],
    "memory_strength": 0.2,
}

LARS_SI = {
    "type": "human",
    "role": "Senior Partner at Sapient",
    "hard_cues": ["Sapient", "McDonald's", "Beacon"],
    "related_entities": ["andre", "alex", "greg"],
    "memory_strength": 0.6,
}

ATTRIBUTED_ANDRE_BLOCK = (
    "## Andre Section\n"
    + (
        "Andre is a key external partner. His operational pattern is to surface for "
        "high-level strategy meetings then disappear. McDonald's account work has been "
        "challenging given his style. "
    ) * 25  # ~5k chars
    + "\n"
)

DATA_GOV_BLOCK = (
    "## Reflections on Data Governance\n"
    + (
        "Three principles drive the user's approach: ownership clarity, semantic "
        "layering, and progressive disclosure. These shape every consulting engagement. "
    ) * 25  # ~4.5k chars
    + "\n"
)


def _build_realistic_worktree(tmp_path: Path) -> Path:
    wt = build_worktree(tmp_path)

    # Build an oversized user entity (>2k tokens) with two clearly attributable
    # blocks.
    user_file = wt / "alex.md"
    user_si = {
        "name": "Alex",
        "aliases": [],
        "type": "human",
        "role": "user",
        "strength": "High",
        "hard_cues": ["alex"],
        "soft_cues": [],
        "emotional_cues": [],
        "related_entities": ["andre", "lars_orloff"],
        "file": "alex.md",
        "memory_strength": 1.0,
    }
    # Make filler small enough that removing the two attributed blocks
    # actually drops the user entity below the test's 2k-token soft cap.
    # (In production, soft cap is 32k and oversized entities have a lot of
    # filler; the redistribute tool can only move attributable content.)
    filler = "Generic life-event filler text. " * 30  # ~1k chars; small remainder after redistribute
    body = (
        "# Alex Profile\n\n"
        "## Core Identity [ALWAYS_LOAD]\n- Software engineer in Seattle.\n\n"
        + ATTRIBUTED_ANDRE_BLOCK + "\n"
        + DATA_GOV_BLOCK + "\n"
        + "## Misc\n" + filler + "\n"
        + "## SEMANTIC INDEX\n"
        + json.dumps(user_si, separators=(",", ":")) + "\n"
    )
    user_file.write_text(body, encoding="utf-8")
    repo = git.Repo(wt)
    repo.index.add(["alex.md"])
    repo.index.commit("inflate user entity")

    # Two Andres (different filenames, both about the same person).
    write_person(
        wt,
        filename="andre_(sapient).md",
        name="Andre (from Sapient)",
        body="Andre is the VP of Technology for Sapient. Based in Chicago. Drives McDonald's.",
        semantic=ANDRE_LONG_SI,
    )
    write_person(
        wt,
        filename="andre.md",
        name="Andre",
        body="Andre is the Head of Technical Sales at Sapient. Aggressive APAC push.",
        semantic=ANDRE_SHORT_SI,
    )

    # Lars Orloff (will co-occur with Andre in a commit).
    write_person(
        wt,
        filename="lars_orloff.md",
        name="Lars Orloff",
        body="Lars is a senior partner at Sapient. Frequently aligned with Andre on McDonald's.",
        semantic=LARS_SI,
    )

    # A few other entities to fill out the people directory.
    for n in ("beatrice", "kenichiro_tanaka", "david", "greg"):
        write_person(
            wt,
            filename=f"{n}.md",
            name=n.replace("_", " ").title(),
            body=f"{n} is a colleague.",
            semantic={"type": "human", "memory_strength": 0.3, "related_entities": ["alex"]},
        )

    # Co-occurrence: edit andre_(sapient) and lars_orloff in the same commit.
    andre_long = wt / "memories" / "people" / "andre_(sapient).md"
    lars = wt / "memories" / "people" / "lars_orloff.md"
    andre_long.write_text(
        andre_long.read_text(encoding="utf-8")
        + "\n## Update\nWorked with Lars Orloff on strategy.\n",
        encoding="utf-8",
    )
    lars.write_text(
        lars.read_text(encoding="utf-8") + "\n## Update\nMeeting with Andre on strategy.\n",
        encoding="utf-8",
    )
    repo.index.add(
        [
            str(andre_long.relative_to(wt)),
            str(lars.relative_to(wt)),
        ]
    )
    repo.index.commit("co-edit: andre + lars strategy session")

    return wt


# --- scripted LLM -------------------------------------------------------------


def _make_merged_andre_content(loser_stem: str) -> str:
    si = dict(ANDRE_LONG_SI)
    si["aliases"] = [loser_stem]
    si["hard_cues"] = sorted(set(ANDRE_LONG_SI["hard_cues"]) | set(ANDRE_SHORT_SI["hard_cues"]))
    si["related_entities"] = sorted(
        set(ANDRE_LONG_SI["related_entities"]) | set(ANDRE_SHORT_SI["related_entities"])
    )
    si["file"] = "memories/people/andre_(sapient).md"
    return (
        "# Andre (from Sapient)\n\n"
        "Andre is the VP of Technology for Sapient. He also drives technical sales across APAC "
        "(previously framed as 'Head of Technical Sales'). Based in Chicago; recurring trips to Japan.\n\n"
        "## Merged from " + loser_stem + "\n\n"
        "## Update\nWorked with Lars Orloff on strategy.\n\n"
        "## SEMANTIC INDEX\n" + json.dumps(si, separators=(",", ":")) + "\n"
    )


def _scripted_llm():
    """Return a callable matching ConsolidatorAgent's llm_call contract."""

    state = {"link_runs": 0}

    def call(prompt: str, is_json: bool):
        if "Dedupe Judge" in prompt:
            return {
                "same_entity": True,
                "confidence": "high",
                "rationale": "Both files describe Andre at Sapient.",
            }
        if "Dedupe Merge" in prompt:
            # Extract loser stem from the prompt context.
            m = re.search(r"LOSER .*?`memories/people/(.+?)\.md`", prompt, re.DOTALL)
            loser_stem = m.group(1) if m else "andre"
            return _make_merged_andre_content(loser_stem)

        if "Redistribute Analyst" in prompt:
            # Only fire for alex.md (the user entity); leave merged-andre alone.
            if "SOURCE FILE \u2014 `alex.md`" in prompt:
                return {
                    "moves": [
                        {
                            "source_section": ATTRIBUTED_ANDRE_BLOCK,
                            "target_entity": "memories/people/andre_(sapient).md",
                            "reason": "section is attributed to Andre",
                            "extracted_content": ATTRIBUTED_ANDRE_BLOCK,
                        }
                    ],
                    "new_contexts": [
                        {
                            "name": "Data Governance",
                            "source_section": DATA_GOV_BLOCK,
                            "extracted_content": "# Data Governance\n\n" + DATA_GOV_BLOCK,
                            "reason": "thematic block, no clear subject in candidates",
                        }
                    ],
                }
            return {"moves": [], "new_contexts": []}

        if "Semantic Index Builder" in prompt:
            return {
                "name": "Data Governance",
                "aliases": [],
                "type": "concept",
                "role": "thematic reference",
                "strength": "Low",
                "hard_cues": ["data governance"],
                "soft_cues": [],
                "emotional_cues": [],
                "related_entities": ["alex"],
            }

        if "Link Weaver" in prompt:
            # Already-linked check: if the prompt's FILE body contains the
            # target's wikilink, return empty edits (idempotency).
            file_section_marker = "FILE \u2014 `"
            file_section_start = prompt.find(file_section_marker)
            file_section = prompt[file_section_start:] if file_section_start != -1 else ""
            if "FILE \u2014 `memories/people/andre_(sapient).md`" in prompt:
                if "lars_orloff|" in file_section:
                    return {"edits": []}
                state["link_runs"] += 1
                return {
                    "edits": [
                        {
                            "search_text": "Worked with Lars Orloff",
                            "replacement_text": "Worked with [[memories/people/lars_orloff|Lars Orloff]]",
                        }
                    ]
                }
            if "FILE \u2014 `memories/people/lars_orloff.md`" in prompt:
                if "andre_(sapient)|" in file_section or "/andre_(sapient)]]" in file_section:
                    return {"edits": []}
                return {
                    "edits": [
                        {
                            "search_text": "Meeting with Andre",
                            "replacement_text": "Meeting with [[memories/people/andre_(sapient)|Andre]]",
                        }
                    ]
                }
            return {"edits": []}

        return {} if is_json else ""

    return call


# --- E2E test -----------------------------------------------------------------


def test_full_chain_smoke(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DEFAULT_MODEL", "test-model")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")

    wt = _build_realistic_worktree(tmp_path)

    # Sanity: pre-state.
    assert (wt / "memories" / "people" / "andre.md").exists()
    assert (wt / "memories" / "people" / "andre_(sapient).md").exists()
    user_file = wt / "alex.md"
    pre_tokens = estimate_tokens(user_file.read_text(encoding="utf-8"))
    soft_cap = 2_000
    assert pre_tokens > soft_cap

    # Patch ConsolidatorAgent's llm_call default to the scripted one.
    from diffmem.consolidator_agent.agent import ConsolidatorAgent

    real_init = ConsolidatorAgent.__init__
    scripted = _scripted_llm()

    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("llm_call", scripted)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(ConsolidatorAgent, "__init__", patched_init)

    memory = DiffMemory(
        repo_path=str(wt),
        user_id="alex",
        openrouter_api_key="dummy",
        model="test-model",
    )

    # Run the full chain. Use a wider window so the link tool sees the
    # original co-edit (which is now several commits back after dedupe +
    # redistribute generated their own commits).
    result = memory.consolidate(window=10, soft_cap_tokens=soft_cap)
    assert result["status"] == "ok"
    assert result["tools_run"] == ["dedupe", "redistribute", "link"]
    assert result["results"]["dedupe"]["merges_performed"] == 1
    assert result["results"]["redistribute"]["total_moves"] == 1
    assert result["results"]["redistribute"]["new_contexts"] == 1
    assert result["results"]["link"]["links_added"] >= 2

    # 1) Two Andres → one.
    assert (wt / "memories" / "people" / "andre_(sapient).md").exists()
    assert not (wt / "memories" / "people" / "andre.md").exists()

    # 2) User entity slimmed below cap.
    post_tokens = estimate_tokens(user_file.read_text(encoding="utf-8"))
    assert post_tokens < soft_cap, f"post tokens {post_tokens} should be below cap {soft_cap}"

    # 3) New contexts file extracted.
    new_ctx = wt / "memories" / "contexts" / "data_governance.md"
    assert new_ctx.exists()
    assert "## SEMANTIC INDEX" in new_ctx.read_text(encoding="utf-8")

    # 4) Wikilink in merged andre file (or lars).
    andre_post = (wt / "memories" / "people" / "andre_(sapient).md").read_text(encoding="utf-8")
    lars_post = (wt / "memories" / "people" / "lars_orloff.md").read_text(encoding="utf-8")
    assert "[[memories/people/lars_orloff|Lars Orloff]]" in andre_post \
        or "[[memories/people/andre_(sapient)|Andre]]" in lars_post

    # 5) Three commit prefixes present in canonical order.
    repo = git.Repo(wt)
    msgs = [c.message.strip() for c in repo.iter_commits(max_count=20)]
    msgs_text = "\n".join(msgs)
    assert "consolidate(dedupe):" in msgs_text
    assert "consolidate(redistribute):" in msgs_text
    assert "consolidate(link):" in msgs_text

    # The three prefixes should appear in temporal order: dedupe first, link last.
    # Walk commits newest -> oldest; reverse to oldest -> newest.
    chronological = list(reversed(msgs))
    dedupe_idx = next(i for i, m in enumerate(chronological) if m.startswith("consolidate(dedupe):"))
    redist_idx = next(i for i, m in enumerate(chronological) if m.startswith("consolidate(redistribute):"))
    link_idx = next(i for i, m in enumerate(chronological) if m.startswith("consolidate(link):"))
    assert dedupe_idx < redist_idx < link_idx, (
        f"commits must be dedupe < redistribute < link, got "
        f"{dedupe_idx}, {redist_idx}, {link_idx}"
    )

    # 6) Master index reflects the merged state.
    idx = (wt / "index.md").read_text(encoding="utf-8")
    assert "andre_(sapient)" in idx.lower() or "andre (from sapient)" in idx.lower()
    # The losing andre.md should not appear as an independent entity.
    assert "memories/people/andre.md" not in idx

    # 7) Idempotency: running again produces no new merge / move / link commits.
    head_before = repo.head.commit.hexsha
    result2 = memory.consolidate(window=10, soft_cap_tokens=soft_cap)
    head_after = repo.head.commit.hexsha
    assert head_before == head_after, "second consolidate run should produce no new commits"
    assert result2["results"]["dedupe"]["merges_performed"] == 0
    assert result2["results"]["redistribute"]["total_moves"] == 0
    assert result2["results"]["link"]["links_added"] == 0
