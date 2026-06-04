# CAPABILITY: Shared test fixtures for the consolidator test suite.
# INPUTS: pytest tmp_path.
# OUTPUTS: Worktree builders, entity factories, fake LLM doubles.
# CONSTRAINTS: No network, no real git remotes, no LLM calls.

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Make src/ importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import git


def _semantic_index_block(d: Dict[str, Any]) -> str:
    return "\n## SEMANTIC INDEX\n" + json.dumps(d, separators=(",", ":")) + "\n"


def build_worktree(tmp_path: Path, user_id: str = "alex") -> Path:
    """Initialise an empty user worktree as a git repo, with the standard layout."""
    wt = tmp_path / user_id
    wt.mkdir(parents=True, exist_ok=True)
    (wt / "memories" / "people").mkdir(parents=True, exist_ok=True)
    (wt / "memories" / "contexts").mkdir(parents=True, exist_ok=True)
    (wt / "timeline").mkdir(parents=True, exist_ok=True)
    repo = git.Repo.init(wt)
    # Configure committer (CI environments often lack this).
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "Test Consolidator")
        cw.set_value("user", "email", "test@diffmem.local")
    # User entity (minimal).
    user_file = wt / f"{user_id}.md"
    user_file.write_text(
        f"# {user_id.title()} Profile\n\n## Core Identity [ALWAYS_LOAD]\n- Test user.\n"
        + _semantic_index_block(
            {
                "name": user_id.title(),
                "aliases": [],
                "type": "human",
                "role": "user",
                "strength": "High",
                "hard_cues": [user_id],
                "soft_cues": [],
                "emotional_cues": [],
                "related_entities": [],
                "file": f"{user_id}.md",
                "memory_strength": 1.0,
            }
        ),
        encoding="utf-8",
    )
    # repo_guide stub (so retrieval skips it cleanly).
    (wt / "repo_guide.md").write_text("# repo_guide\n", encoding="utf-8")
    repo.index.add([str(user_file.relative_to(wt)), "repo_guide.md"])
    repo.index.commit("init")
    return wt


def write_person(
    worktree: Path,
    *,
    filename: str,
    name: str,
    body: str,
    semantic: Dict[str, Any],
    commit_msg: str = "add person",
) -> Path:
    """Write a people entity file with a SEMANTIC INDEX block, then commit it."""
    path = worktree / "memories" / "people" / filename
    semantic.setdefault("name", name)
    semantic.setdefault("type", "human")
    semantic.setdefault("strength", "Medium")
    semantic.setdefault("aliases", [])
    semantic.setdefault("hard_cues", [])
    semantic.setdefault("soft_cues", [])
    semantic.setdefault("emotional_cues", [])
    semantic.setdefault("related_entities", [])
    semantic["file"] = f"memories/people/{filename}"
    content = f"# {name}\n\n{body.strip()}\n" + _semantic_index_block(semantic)
    path.write_text(content, encoding="utf-8")
    repo = git.Repo(worktree)
    repo.index.add([str(path.relative_to(worktree))])
    repo.index.commit(commit_msg)
    return path


def write_context(
    worktree: Path,
    *,
    filename: str,
    name: str,
    body: str,
    semantic: Dict[str, Any],
    commit_msg: str = "add context",
) -> Path:
    path = worktree / "memories" / "contexts" / filename
    semantic.setdefault("name", name)
    semantic.setdefault("type", "concept")
    semantic.setdefault("strength", "Medium")
    semantic.setdefault("aliases", [])
    semantic.setdefault("hard_cues", [])
    semantic.setdefault("soft_cues", [])
    semantic.setdefault("emotional_cues", [])
    semantic.setdefault("related_entities", [])
    semantic["file"] = f"memories/contexts/{filename}"
    content = f"# {name}\n\n{body.strip()}\n" + _semantic_index_block(semantic)
    path.write_text(content, encoding="utf-8")
    repo = git.Repo(worktree)
    repo.index.add([str(path.relative_to(worktree))])
    repo.index.commit(commit_msg)
    return path


# --- Fake LLM -----------------------------------------------------------------


class FakeLLM:
    """Records prompts and returns scripted responses.

    Usage:
        llm = FakeLLM()
        llm.add_response(matches="dedupe_judge", payload={"same_entity": True, "confidence": "high", "rationale": "..."})
        llm.add_response(matches="dedupe_merge", payload="# Andre\\n...full merged content...\\n")
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._responses: List[Dict[str, Any]] = []

    def add_response(self, *, matches: str, payload: Any) -> None:
        self._responses.append({"matches": matches, "payload": payload})

    def __call__(self, prompt: str, is_json: bool = True) -> Any:
        self.calls.append({"prompt": prompt, "is_json": is_json})
        for r in self._responses:
            if r["matches"] in prompt:
                payload = r["payload"]
                if callable(payload):
                    return payload(prompt, is_json)
                return payload
        # Default: empty
        return {} if is_json else ""
