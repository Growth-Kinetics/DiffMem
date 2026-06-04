# CAPABILITY: Integration tests for run_link.
# INPUTS: tmp_path -> fixture with co-occurring commits.
# OUTPUTS: Verifies inline wikilinks created, SEMANTIC INDEX untouched,
#          idempotency (second run = no new commit), commit message correct.
# CONSTRAINTS: FakeLLM scripted; no network.

from __future__ import annotations

import json
from pathlib import Path

import git

from tests._fixtures import FakeLLM, build_worktree, write_person

from diffmem.consolidator_agent.agent import ConsolidatorAgent


def _co_commit(wt: Path, files: list[tuple[Path, str]]) -> str:
    """Write changes to multiple files and commit them together (co-occurrence)."""
    repo = git.Repo(wt)
    for path, content in files:
        path.write_text(content, encoding="utf-8")
        repo.index.add([str(path.relative_to(wt))])
    repo.index.commit("co-edit maya + lars")
    return repo.head.commit.hexsha


def _maya_file(wt: Path) -> Path:
    return write_person(
        wt,
        filename="maya.md",
        name="Maya",
        body=(
            "## About\n"
            "Maya is the VP of Technology for Acme. He drives the Project X account "
            "and frequently collaborates with Sam Rivera on strategic alignment.\n"
        ),
        semantic={
            "type": "human",
            "role": "VP Technology",
            "hard_cues": ["Acme", "Project X"],
            "related_entities": ["sam_rivera", "alex"],
            "memory_strength": 0.7,
        },
    )


def _lars_file(wt: Path) -> Path:
    return write_person(
        wt,
        filename="sam_rivera.md",
        name="Sam Rivera",
        body=(
            "## About\n"
            "Sam Rivera is a senior partner at Acme who works closely with Maya on "
            "the Project X account.\n"
        ),
        semantic={
            "type": "human",
            "role": "Senior Partner",
            "hard_cues": ["Acme", "Project X"],
            "related_entities": ["maya", "alex"],
            "memory_strength": 0.6,
        },
    )


def test_link_weaves_wikilinks_between_cooccurring_entities(tmp_path: Path) -> None:
    wt = build_worktree(tmp_path)
    maya = _maya_file(wt)
    lars = _lars_file(wt)

    # Co-occur them in a single commit.
    _co_commit(
        wt,
        [
            (
                maya,
                maya.read_text(encoding="utf-8").replace(
                    "VP of Technology for Acme", "VP of Technology for Acme (updated)"
                ),
            ),
            (
                lars,
                lars.read_text(encoding="utf-8").replace(
                    "senior partner at Acme", "senior partner at Acme (updated)"
                ),
            ),
        ],
    )

    # FakeLLM script:
    #  - When the prompt mentions maya.md, propose linking to sam_rivera.
    #  - When the prompt mentions sam_rivera.md, propose linking to maya.
    def llm_fn(prompt: str, is_json: bool):
        if "FILE — `memories/people/maya.md`" in prompt:
            return {
                "edits": [
                    {
                        "search_text": "collaborates with Sam Rivera",
                        "replacement_text": "collaborates with [[memories/people/sam_rivera|Sam Rivera]]",
                    }
                ]
            }
        if "FILE — `memories/people/sam_rivera.md`" in prompt:
            return {
                "edits": [
                    {
                        "search_text": "works closely with Maya",
                        "replacement_text": "works closely with [[memories/people/maya|Maya]]",
                    }
                ]
            }
        return {}

    agent = ConsolidatorAgent(
        repo_path=str(wt),
        user_id="alex",
        openrouter_api_key="dummy",
        model="test-model",
        llm_call=llm_fn,
    )
    result = agent.run_link(window=3)

    assert result["status"] == "ok"
    assert result["tool"] == "link"
    assert result["files_touched"] >= 2
    assert result["links_added"] >= 2
    assert len(result["commits"]) == 1
    assert result["window"] == 3

    a_content = maya.read_text(encoding="utf-8")
    l_content = lars.read_text(encoding="utf-8")
    assert "[[memories/people/sam_rivera|Sam Rivera]]" in a_content
    assert "[[memories/people/maya|Maya]]" in l_content

    # SEMANTIC INDEX untouched.
    assert "## SEMANTIC INDEX" in a_content
    assert "## SEMANTIC INDEX" in l_content
    si_line_a = a_content.split("## SEMANTIC INDEX", 1)[1].strip().splitlines()[0]
    assert json.loads(si_line_a)["name"] == "Maya"

    # Commit message correct.
    repo = git.Repo(wt)
    msg = repo.head.commit.message.strip()
    assert msg.startswith("consolidate(link): wikilinks across"), msg
    assert "window=3" in msg


def test_link_is_idempotent(tmp_path: Path) -> None:
    """Running run_link twice should produce only one new commit, not two."""
    wt = build_worktree(tmp_path)
    maya = _maya_file(wt)
    lars = _lars_file(wt)
    _co_commit(
        wt,
        [
            (maya, maya.read_text() + "\n## Note\nshared edit\n"),
            (lars, lars.read_text() + "\n## Note\nshared edit\n"),
        ],
    )

    def llm_fn(prompt: str, is_json: bool):
        if "FILE — `memories/people/maya.md`" in prompt:
            # Only propose link if Sam's wikilink isn't already present.
            if "sam_rivera|" in prompt[prompt.find("FILE"):prompt.find("CO-OCCURRING")]:
                return {"edits": []}
            return {
                "edits": [
                    {
                        "search_text": "collaborates with Sam Rivera",
                        "replacement_text": "collaborates with [[memories/people/sam_rivera|Sam Rivera]]",
                    }
                ]
            }
        if "FILE — `memories/people/sam_rivera.md`" in prompt:
            if "maya|" in prompt[prompt.find("FILE"):prompt.find("CO-OCCURRING")]:
                return {"edits": []}
            return {
                "edits": [
                    {
                        "search_text": "works closely with Maya",
                        "replacement_text": "works closely with [[memories/people/maya|Maya]]",
                    }
                ]
            }
        return {}

    agent = ConsolidatorAgent(
        repo_path=str(wt),
        user_id="alex",
        openrouter_api_key="dummy",
        model="test-model",
        llm_call=llm_fn,
    )
    r1 = agent.run_link(window=3)
    commits_after_1 = list(git.Repo(wt).iter_commits())

    r2 = agent.run_link(window=3)
    commits_after_2 = list(git.Repo(wt).iter_commits())

    assert r1["links_added"] >= 2
    assert r2["commits"] == [], "second run should not produce a new commit"
    assert r2["links_added"] == 0
    assert len(commits_after_1) == len(commits_after_2), "no new commits between runs"


def test_link_window_zero_means_no_cooccurrence(tmp_path: Path) -> None:
    wt = build_worktree(tmp_path)
    _maya_file(wt)
    _lars_file(wt)
    # Don't co-edit; each is in its own commit (write_person commits individually).

    llm = FakeLLM()
    agent = ConsolidatorAgent(
        repo_path=str(wt),
        user_id="alex",
        openrouter_api_key="dummy",
        model="test-model",
        llm_call=llm,
    )
    # With window=1, only the last single-file commit is in the window → no pairs.
    result = agent.run_link(window=1)
    assert result["files_touched"] == 0
    assert result["links_added"] == 0
