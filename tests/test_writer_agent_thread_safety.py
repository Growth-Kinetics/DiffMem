"""
Regression tests for WriterAgent thread-safety around GitPython.

Bug history: GitPython's persistent `cat-file --batch-check` subprocess is not
thread-safe when a single Repo object is shared across worker threads. The
writer agent's `_build_entity_indexes` parallel path used to call
`_get_file_git_stats(...)` from inside a ThreadPoolExecutor, which produced
intermittent deadlocks: one worker would block forever on a pipe.readline()
inside `git/cmd.py:__get_object_header`, the executor would never drain via
as_completed(), and the whole session would hang after the LLM work was done.

Captured live at the Growth Kinetics corporate-ontology deployment 2026-06-06
via py-spy dump on a stuck PID. The fix moves git stats computation OUT of the
parallel section: stats are precomputed serially in the parent thread and
passed into workers, so worker threads never touch `self.repo`.

These tests pin the contract so the bug cannot regress silently.
"""
import json
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

import git
import pytest

from diffmem.writer_agent.agent import WriterAgent
from diffmem.ontology.loader import load_ontology


# ---------------------------------------------------------------------------
# Test fixture: a minimal worktree + initialized git repo + WriterAgent
# ---------------------------------------------------------------------------

def _make_writer(tmp_path: Path) -> WriterAgent:
    """Builds a WriterAgent with a real git repo and 3 entity files."""
    # Init repo
    repo = git.Repo.init(tmp_path)
    repo.config_writer().set_value("user", "name", "test").release()
    repo.config_writer().set_value("user", "email", "test@test").release()

    # Minimal worktree structure (personal ontology default for this test)
    (tmp_path / "alex.md").write_text("# alex\n")
    (tmp_path / "index.md").write_text("# index\n")
    memories = tmp_path / "memories" / "people"
    memories.mkdir(parents=True)

    file_paths = []
    for name in ("alice", "bob", "carol"):
        f = memories / f"{name}.md"
        f.write_text(f"# {name}\n\nSome content about {name}.\n")
        file_paths.append(f)

    repo.index.add([str(f.relative_to(tmp_path)) for f in file_paths]
                   + ["alex.md", "index.md"])
    repo.index.commit("seed")

    ontology = load_ontology("personal")
    agent = WriterAgent(
        str(tmp_path), "alex", "fake-key", model="test-model",
        validate_paths=False, ontology=ontology,
    )
    # Attach file_paths for convenience
    agent._test_file_paths = file_paths
    return agent


# ---------------------------------------------------------------------------
# Contract test: workers must not call _get_file_git_stats in parallel
# ---------------------------------------------------------------------------

def test_build_entity_indexes_precomputes_gitstats_serially(tmp_path):
    """The bug fix: `_build_entity_indexes` MUST compute git stats serially
    in the parent thread before launching the executor. Worker threads MUST
    NOT call `_get_file_git_stats` (which touches self.repo, which is not
    thread-safe).

    We assert this by recording every thread that calls _get_file_git_stats
    and verifying they are ALL the main thread.
    """
    agent = _make_writer(tmp_path)
    caller_threads: list[int] = []
    original = agent._get_file_git_stats

    def tracking_gitstats(file_path):
        caller_threads.append(threading.get_ident())
        return original(file_path)

    # Mock the LLM call so we don't need an API key
    def fake_llm(system_prompt, user_prompt, is_json=False):
        return {
            "name": "test", "type": "human", "role": "test",
            "strength": "Low", "hard_cues": [], "soft_cues": [],
            "emotional_cues": [], "related_entities": [],
        }

    with patch.object(agent, '_get_file_git_stats', side_effect=tracking_gitstats), \
         patch.object(agent, '_call_llm', side_effect=fake_llm):
        agent._build_entity_indexes(agent._test_file_paths)

    main_thread_id = threading.main_thread().ident
    assert len(caller_threads) == 3, (
        f"Expected 3 git stats calls (one per file), got {len(caller_threads)}"
    )
    for tid in caller_threads:
        assert tid == main_thread_id, (
            f"_get_file_git_stats called from thread {tid}, but only the "
            f"main thread ({main_thread_id}) is allowed. This would re-introduce "
            f"the GitPython thread-deadlock bug."
        )


def test_build_single_entity_index_accepts_precomputed_stats(tmp_path):
    """_build_single_entity_index must accept a precomputed git_stats dict and
    NOT call self._get_file_git_stats when one is provided."""
    agent = _make_writer(tmp_path)
    file_path = agent._test_file_paths[0]

    precomputed = {"last_update": "2026-06-06 00:00:00 +0000", "number_of_edits": 7}

    fake_index = {
        "name": "alice", "type": "human", "role": "x",
        "strength": "Low", "hard_cues": [], "soft_cues": [],
        "emotional_cues": [], "related_entities": [],
    }

    captured_prompts = []

    def capturing_llm(system_prompt, user_prompt, is_json=False):
        captured_prompts.append(user_prompt)
        return fake_index

    with patch.object(agent, '_get_file_git_stats') as mock_stats, \
         patch.object(agent, '_call_llm', side_effect=capturing_llm):
        result = agent._build_single_entity_index(file_path, git_stats=precomputed)

    assert result['success'] is True
    assert mock_stats.call_count == 0, (
        "When precomputed git_stats are passed, _build_single_entity_index "
        "MUST NOT call _get_file_git_stats. This is the thread-safety contract."
    )
    # Verify the precomputed values flowed through into the LLM prompt
    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]
    assert "Number of Edits: 7" in prompt, (
        f"Precomputed number_of_edits=7 did not reach the LLM prompt. "
        f"Prompt was: {prompt[:500]}"
    )
    assert "2026-06-06" in prompt, (
        "Precomputed last_update did not reach the LLM prompt."
    )


def test_build_single_entity_index_falls_back_to_live_lookup(tmp_path):
    """When no precomputed stats are provided, _build_single_entity_index falls
    back to live lookup. This preserves the contract for non-parallel callers."""
    agent = _make_writer(tmp_path)
    file_path = agent._test_file_paths[0]

    fake_index = {
        "name": "alice", "type": "human", "role": "x",
        "strength": "Low", "hard_cues": [], "soft_cues": [],
        "emotional_cues": [], "related_entities": [],
    }

    with patch.object(
        agent, '_get_file_git_stats',
        return_value={"last_update": "2026-06-06 00:00:00 +0000", "number_of_edits": 3},
    ) as mock_stats, patch.object(agent, '_call_llm', return_value=fake_index):
        result = agent._build_single_entity_index(file_path)  # no git_stats arg

    assert result['success'] is True
    assert mock_stats.call_count == 1, (
        "Without precomputed git_stats, the live-lookup fallback path must be used."
    )


def test_build_entity_indexes_empty_list_is_noop(tmp_path):
    """Empty file list returns without computing stats or launching workers."""
    agent = _make_writer(tmp_path)
    with patch.object(agent, '_get_file_git_stats') as mock_stats:
        agent._build_entity_indexes([])
    assert mock_stats.call_count == 0
