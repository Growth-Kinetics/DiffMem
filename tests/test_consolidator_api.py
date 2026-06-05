# CAPABILITY: Integration tests for the consolidator API + HTTP surface.
# INPUTS: tmp_path -> fixture worktree.
# OUTPUTS: Verifies DiffMemory.consolidate() respects tool subsets + canonical
#          order, and that the HTTP endpoints round-trip via TestClient.
# CONSTRAINTS: No network. Monkey-patches the env vars + ConsolidatorAgent
#              so we can avoid real LLM credentials.

from __future__ import annotations

import json
import os
from pathlib import Path

import git
import pytest

from tests._fixtures import FakeLLM, build_worktree, write_person


# --- DiffMemory.consolidate() direct test -------------------------------------


def test_consolidate_runs_canonical_order_and_respects_tool_subset(monkeypatch, tmp_path: Path) -> None:
    """Calling DiffMemory.consolidate(tools=['link','dedupe']) should:
       - run dedupe before link (canonical order)
       - NOT run redistribute
       - aggregate commits from each tool
    """
    # We import here so the monkeypatch on DEFAULT_MODEL takes effect.
    monkeypatch.setenv("DEFAULT_MODEL", "test-model")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")

    from diffmem.api import DiffMemory

    wt = build_worktree(tmp_path)
    write_person(
        wt,
        filename="maya.md",
        name="Maya",
        body="Maya is VP at Acme.",
        semantic={"type": "human", "memory_strength": 0.5, "related_entities": ["alex"]},
    )

    call_order = []

    # Patch ConsolidatorAgent's tool methods to track call order without
    # involving any real LLM.
    from diffmem.consolidator_agent.agent import ConsolidatorAgent

    real_init = ConsolidatorAgent.__init__

    def fake_init(self, *args, **kwargs):
        kwargs.setdefault("llm_call", lambda p, j: ({} if j else ""))
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(ConsolidatorAgent, "__init__", fake_init)

    def fake_dedupe(self):
        call_order.append("dedupe")
        return {"status": "ok", "tool": "dedupe", "commits": ["c-dedupe"], "summary": "stub"}

    def fake_redistribute(self, soft_cap_tokens: int = 32000):
        call_order.append("redistribute")
        return {"status": "ok", "tool": "redistribute", "commits": ["c-redist"], "summary": "stub"}

    def fake_link(self, window: int = 3):
        call_order.append("link")
        return {"status": "ok", "tool": "link", "commits": ["c-link"], "summary": "stub"}

    monkeypatch.setattr(ConsolidatorAgent, "run_dedupe", fake_dedupe)
    monkeypatch.setattr(ConsolidatorAgent, "run_redistribute", fake_redistribute)
    monkeypatch.setattr(ConsolidatorAgent, "run_link", fake_link)

    memory = DiffMemory(
        repo_path=str(wt),
        user_id="alex",
        openrouter_api_key="dummy",
        model="test-model",
    )

    # Subset, reversed order on input — should be reordered to canonical.
    r = memory.consolidate(tools=["link", "dedupe"], window=5, soft_cap_tokens=12345)
    assert r["status"] == "ok"
    assert r["tools_run"] == ["dedupe", "link"], "canonical order must be enforced"
    assert "dedupe" in r["results"]
    assert "link" in r["results"]
    assert "redistribute" not in r["results"]
    assert r["commits"] == ["c-dedupe", "c-link"]
    assert call_order == ["dedupe", "link"]


def test_consolidate_default_runs_all_three(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DEFAULT_MODEL", "test-model")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")

    from diffmem.api import DiffMemory
    from diffmem.consolidator_agent.agent import ConsolidatorAgent

    wt = build_worktree(tmp_path)

    real_init = ConsolidatorAgent.__init__
    monkeypatch.setattr(ConsolidatorAgent, "__init__",
                        lambda self, *a, **kw: real_init(self, *a, llm_call=(lambda p, j: ({} if j else "")), **{k: v for k, v in kw.items() if k != "llm_call"}))
    monkeypatch.setattr(ConsolidatorAgent, "run_dedupe", lambda self: {"status": "ok", "tool": "dedupe", "commits": [], "summary": "x"})
    monkeypatch.setattr(ConsolidatorAgent, "run_redistribute", lambda self, soft_cap_tokens=32000: {"status": "ok", "tool": "redistribute", "commits": [], "summary": "x"})
    monkeypatch.setattr(ConsolidatorAgent, "run_link", lambda self, window=3: {"status": "ok", "tool": "link", "commits": [], "summary": "x"})

    memory = DiffMemory(
        repo_path=str(wt),
        user_id="alex",
        openrouter_api_key="dummy",
        model="test-model",
    )
    r = memory.consolidate()
    assert r["tools_run"] == ["dedupe", "redistribute", "link"]


def test_consolidate_rejects_unknown_tool(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DEFAULT_MODEL", "test-model")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    from diffmem.api import DiffMemory

    wt = build_worktree(tmp_path)
    memory = DiffMemory(
        repo_path=str(wt),
        user_id="alex",
        openrouter_api_key="dummy",
        model="test-model",
    )
    with pytest.raises(ValueError, match="Unknown consolidator tool"):
        memory.consolidate(tools=["bogus"])


# --- HTTP endpoint test -------------------------------------------------------


def _build_test_client(monkeypatch, tmp_path: Path, llm_call=None):
    """Spin up the FastAPI app with a controlled memory instance.

    We bypass the full server lifespan (no RepoManager, no real backup) by
    pre-populating the global `memory_instances` dict with a fake `DiffMemory`
    pointed at the fixture worktree, and we monkey-patch `get_memory_instance`
    to return it directly.
    """
    monkeypatch.setenv("DEFAULT_MODEL", "test-model")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.setenv("REQUIRE_AUTH", "false")

    from fastapi.testclient import TestClient

    # Import server module fresh so env vars are read.
    import importlib
    import diffmem.server as server_mod
    importlib.reload(server_mod)

    from diffmem.api import DiffMemory
    from diffmem.consolidator_agent.agent import ConsolidatorAgent

    wt = build_worktree(tmp_path)
    write_person(
        wt,
        filename="maya.md",
        name="Maya",
        body="Maya is at Acme.",
        semantic={"type": "human", "memory_strength": 0.5},
    )

    # Stub ConsolidatorAgent so no real LLM call happens.
    real_init = ConsolidatorAgent.__init__
    monkeypatch.setattr(
        ConsolidatorAgent,
        "__init__",
        lambda self, *a, **kw: real_init(
            self, *a,
            llm_call=(llm_call or (lambda p, j: ({} if j else ""))),
            **{k: v for k, v in kw.items() if k != "llm_call"},
        ),
    )
    monkeypatch.setattr(
        ConsolidatorAgent, "run_dedupe",
        lambda self: {"status": "ok", "tool": "dedupe", "commits": ["c1"], "summary": "stub"},
    )
    monkeypatch.setattr(
        ConsolidatorAgent, "run_redistribute",
        lambda self, soft_cap_tokens=32000: {"status": "ok", "tool": "redistribute", "commits": [], "summary": "stub"},
    )
    monkeypatch.setattr(
        ConsolidatorAgent, "run_link",
        lambda self, window=3: {"status": "ok", "tool": "link", "commits": ["c2"], "summary": "stub"},
    )

    memory = DiffMemory(
        repo_path=str(wt),
        user_id="alex",
        openrouter_api_key="dummy",
        model="test-model",
    )

    # Inject into the module-level dict + bypass lookup.
    server_mod.memory_instances["alex"] = memory
    monkeypatch.setattr(server_mod, "get_memory_instance", lambda uid, allow_unboarded=False: memory)

    # No-op backup_user (no RepoManager).
    async def noop_backup(uid):
        return None
    monkeypatch.setattr(server_mod, "backup_user", noop_backup)

    # Inject executor (bypasses lifespan which requires real env + RepoManager).
    from concurrent.futures import ThreadPoolExecutor
    from diffmem.executor.inline import InlineExecutor
    server_mod.app.state.executor = InlineExecutor(ThreadPoolExecutor(max_workers=2))

    return TestClient(server_mod.app), wt


def test_http_consolidate_endpoint_full_chain(monkeypatch, tmp_path: Path) -> None:
    client, wt = _build_test_client(monkeypatch, tmp_path)
    r = client.post("/memory/alex/consolidate", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success"
    assert body["consolidate"]["tools_run"] == ["dedupe", "redistribute", "link"]
    assert "c1" in body["consolidate"]["commits"]
    assert "c2" in body["consolidate"]["commits"]


def test_http_consolidate_endpoint_tool_subset(monkeypatch, tmp_path: Path) -> None:
    client, wt = _build_test_client(monkeypatch, tmp_path)
    r = client.post("/memory/alex/consolidate", json={"tools": ["dedupe"], "window": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["consolidate"]["tools_run"] == ["dedupe"]


def test_http_consolidate_endpoint_rejects_unknown_tool(monkeypatch, tmp_path: Path) -> None:
    client, wt = _build_test_client(monkeypatch, tmp_path)
    r = client.post("/memory/alex/consolidate", json={"tools": ["nope"]})
    assert r.status_code == 400, r.text


def test_http_process_commit_and_consolidate(monkeypatch, tmp_path: Path) -> None:
    client, wt = _build_test_client(monkeypatch, tmp_path)

    # Stub process_and_commit_session to avoid invoking the real WriterAgent.
    import diffmem.server as server_mod

    memory = server_mod.memory_instances["alex"]
    monkeypatch.setattr(memory, "process_and_commit_session", lambda *a, **kw: None)

    r = client.post(
        "/memory/alex/process-commit-and-consolidate",
        json={
            "memory_input": "test transcript",
            "session_id": "s-001",
            "consolidate_tools": ["dedupe", "link"],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success"
    assert body["session_id"] == "s-001"
    assert body["consolidate"]["tools_run"] == ["dedupe", "link"]
