"""
Tests for hatchet_worker.py (M4) — all tests run WITHOUT hatchet-sdk installed.
Uses unittest.mock to simulate the SDK, DiffMemory, and RepoManager.
No real Hatchet connection, no real LLM, no real worktrees.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Helpers: SDK mock (same shape as test_executor_hatchet.py)
# ---------------------------------------------------------------------------

def _make_hatchet_sdk_mock():
    """Build a minimal fake hatchet_sdk module."""
    sdk = types.ModuleType("hatchet_sdk")

    strategy = MagicMock()
    strategy.GROUP_ROUND_ROBIN = "GROUP_ROUND_ROBIN"
    sdk.ConcurrencyLimitStrategy = strategy
    sdk.ConcurrencyExpression = MagicMock(return_value=MagicMock())

    client = MagicMock()
    workflow_obj = MagicMock()
    worker_obj = MagicMock()
    client.workflow.return_value = workflow_obj
    client.worker.return_value = worker_obj
    sdk.Hatchet = MagicMock(return_value=client)

    from pydantic import BaseModel  # real pydantic, always available
    sdk.BaseModel = BaseModel

    return sdk, client, workflow_obj, worker_obj


def _install_sdk_mock(monkeypatch, sdk_mock):
    """Install fake hatchet_sdk and clear cached hatchet* sub-modules."""
    monkeypatch.setitem(sys.modules, "hatchet_sdk", sdk_mock)
    for mod_name in list(sys.modules.keys()):
        if "hatchet" in mod_name and mod_name != "hatchet_sdk":
            monkeypatch.delitem(sys.modules, mod_name, raising=False)


def _reload_worker_modules():
    """Reload hatchet_workflows and hatchet_worker in dependency order."""
    import diffmem.executor.hatchet_workflows as wf_mod
    import diffmem.executor.hatchet_worker as worker_mod
    importlib.reload(wf_mod)
    importlib.reload(worker_mod)
    return wf_mod, worker_mod


# ---------------------------------------------------------------------------
# Shared fixture: SDK + token env var set up
# ---------------------------------------------------------------------------

@pytest.fixture()
def worker_env(monkeypatch):
    """Yield (sdk, client, workflow_obj, worker_obj, worker_mod) with SDK mocked."""
    sdk_mock, client, workflow_obj, worker_obj = _make_hatchet_sdk_mock()
    _install_sdk_mock(monkeypatch, sdk_mock)
    monkeypatch.setenv("HATCHET_CLIENT_TOKEN", "fake-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-api-key")

    wf_mod, worker_mod = _reload_worker_modules()

    # Reset module-level singletons so tests are isolated
    worker_mod._repo_manager_singleton = None
    worker_mod._memory_cache.clear()

    yield sdk_mock, client, workflow_obj, worker_obj, worker_mod


# ---------------------------------------------------------------------------
# Test: main() calls worker.start()
# ---------------------------------------------------------------------------

def test_worker_main_calls_start(worker_env, monkeypatch):
    """main() builds a worker and calls worker.start() exactly once."""
    *_, worker_obj, worker_mod = worker_env

    mock_worker = MagicMock()
    monkeypatch.setattr(worker_mod, "build_worker", lambda: mock_worker)

    worker_mod.main()

    mock_worker.start.assert_called_once()


# ---------------------------------------------------------------------------
# Test: _attach_write_handler registers a task decorator
# ---------------------------------------------------------------------------

def test_attach_write_handler_registers_task(worker_env):
    """_attach_write_handler calls write_workflow.task(execution_timeout='15m', retries=0)."""
    *_, worker_mod = worker_env

    write_workflow = MagicMock()
    # Make the decorator return itself so the inner function is "registered"
    task_decorator = MagicMock(return_value=MagicMock())
    write_workflow.task.return_value = task_decorator

    worker_mod._attach_write_handler(write_workflow)

    write_workflow.task.assert_called_once_with(execution_timeout="15m", retries=0)


# ---------------------------------------------------------------------------
# Test: _attach_consolidate_handler registers a task decorator
# ---------------------------------------------------------------------------

def test_attach_consolidate_handler_registers_task(worker_env):
    """_attach_consolidate_handler calls consolidate_workflow.task(execution_timeout='15m', retries=0)."""
    *_, worker_mod = worker_env

    consolidate_workflow = MagicMock()
    task_decorator = MagicMock(return_value=MagicMock())
    consolidate_workflow.task.return_value = task_decorator

    worker_mod._attach_consolidate_handler(consolidate_workflow)

    consolidate_workflow.task.assert_called_once_with(execution_timeout="15m", retries=0)


# ---------------------------------------------------------------------------
# Helper: capture the inner handler function registered by _attach_*_handler
# ---------------------------------------------------------------------------

def _capture_write_handler(worker_mod):
    """Call _attach_write_handler on a capture-mock; return the registered function."""
    captured = {}

    def fake_task(**kwargs):
        def decorator(fn):
            captured["fn"] = fn
            captured["kwargs"] = kwargs
            return fn
        return decorator

    write_workflow = MagicMock()
    write_workflow.task = fake_task

    worker_mod._attach_write_handler(write_workflow)
    return captured["fn"]


def _capture_consolidate_handler(worker_mod):
    """Call _attach_consolidate_handler on a capture-mock; return the registered function."""
    captured = {}

    def fake_task(**kwargs):
        def decorator(fn):
            captured["fn"] = fn
            captured["kwargs"] = kwargs
            return fn
        return decorator

    consolidate_workflow = MagicMock()
    consolidate_workflow.task = fake_task

    worker_mod._attach_consolidate_handler(consolidate_workflow)
    return captured["fn"]


# ---------------------------------------------------------------------------
# Test: write handler calls process_and_commit_session with right args
# ---------------------------------------------------------------------------

def test_write_handler_calls_process_and_commit_session(worker_env, monkeypatch):
    """The write task handler calls memory.process_and_commit_session with correct args."""
    *_, worker_mod = worker_env

    memory_mock = MagicMock()
    monkeypatch.setattr(worker_mod, "_get_memory", lambda uid: memory_mock)

    handler = _capture_write_handler(worker_mod)

    # Build a real WriteInput via get_input_models
    from diffmem.executor.hatchet_workflows import get_input_models
    WriteInput, _ = get_input_models()
    inp = WriteInput(
        user_id="alice",
        memory_input="some notes",
        session_id="sess-001",
        session_date="2026-06-05",
    )
    ctx = MagicMock()
    ctx.workflow_run_id = "run-abc"

    result = handler(inp, ctx)

    memory_mock.process_and_commit_session.assert_called_once_with(
        "some notes", "sess-001", "2026-06-05"
    )
    assert result["session_id"] == "sess-001"
    assert result["user_id"] == "alice"
    assert result["status"] == "completed"


# ---------------------------------------------------------------------------
# Test: write handler fires callback on success
# ---------------------------------------------------------------------------

def test_write_handler_fires_callback_on_success(worker_env, monkeypatch):
    """Write handler POSTs to callback_url when the job succeeds."""
    *_, worker_mod = worker_env

    memory_mock = MagicMock()
    monkeypatch.setattr(worker_mod, "_get_memory", lambda uid: memory_mock)

    post_mock = MagicMock()
    fake_httpx = types.ModuleType("httpx")
    fake_httpx.post = post_mock  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    handler = _capture_write_handler(worker_mod)

    from diffmem.executor.hatchet_workflows import get_input_models
    WriteInput, _ = get_input_models()
    inp = WriteInput(
        user_id="bob",
        memory_input="test",
        session_id="sess-002",
        callback_url="http://callback.example.com/hook",
    )
    ctx = MagicMock()
    ctx.workflow_run_id = "run-xyz"

    result = handler(inp, ctx)

    assert result["status"] == "completed"
    post_mock.assert_called_once()
    call_kwargs = post_mock.call_args
    assert call_kwargs[0][0] == "http://callback.example.com/hook"
    posted_json = call_kwargs[1]["json"]
    assert posted_json["status"] == "completed"
    assert posted_json["job_id"] == "run-xyz"


# ---------------------------------------------------------------------------
# Test: callback failure does NOT raise
# ---------------------------------------------------------------------------

def test_write_handler_callback_failure_does_not_raise(worker_env, monkeypatch):
    """Write handler swallows httpx.post errors — callback failure must not propagate."""
    *_, worker_mod = worker_env

    memory_mock = MagicMock()
    monkeypatch.setattr(worker_mod, "_get_memory", lambda uid: memory_mock)

    # Install a fake httpx where post raises
    fake_httpx = types.ModuleType("httpx")
    fake_httpx.post = MagicMock(side_effect=Exception("network failure"))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    handler = _capture_write_handler(worker_mod)

    from diffmem.executor.hatchet_workflows import get_input_models
    WriteInput, _ = get_input_models()
    inp = WriteInput(
        user_id="carol",
        memory_input="test",
        session_id="sess-003",
        callback_url="http://broken.example.com/hook",
    )
    ctx = MagicMock()
    ctx.workflow_run_id = "run-fail"

    # Must not raise even though httpx.post raises
    result = handler(inp, ctx)
    assert result["status"] == "completed"


# ---------------------------------------------------------------------------
# Test: write handler propagates exception from memory
# ---------------------------------------------------------------------------

def test_write_handler_propagates_exception(worker_env, monkeypatch):
    """Write handler re-raises exceptions so Hatchet marks the run as failed."""
    *_, worker_mod = worker_env

    memory_mock = MagicMock()
    memory_mock.process_and_commit_session.side_effect = RuntimeError("LLM exploded")
    monkeypatch.setattr(worker_mod, "_get_memory", lambda uid: memory_mock)

    handler = _capture_write_handler(worker_mod)

    from diffmem.executor.hatchet_workflows import get_input_models
    WriteInput, _ = get_input_models()
    inp = WriteInput(
        user_id="dave",
        memory_input="crash",
        session_id="sess-004",
    )
    ctx = MagicMock()
    ctx.workflow_run_id = "run-crash"

    with pytest.raises(RuntimeError, match="LLM exploded"):
        handler(inp, ctx)


# ---------------------------------------------------------------------------
# Test: consolidate handler calls memory.consolidate with right args
# ---------------------------------------------------------------------------

def test_consolidate_handler_calls_consolidate(worker_env, monkeypatch):
    """The consolidate task handler calls memory.consolidate with correct args."""
    *_, worker_mod = worker_env

    memory_mock = MagicMock()
    memory_mock.consolidate.return_value = {
        "status": "ok",
        "tools_run": ["dedupe"],
        "results": {},
        "commits": [],
        "timestamp": "2026-06-05T00:00:00",
        "user_id": "eve",
    }
    monkeypatch.setattr(worker_mod, "_get_memory", lambda uid: memory_mock)

    handler = _capture_consolidate_handler(worker_mod)

    from diffmem.executor.hatchet_workflows import get_input_models
    _, ConsolidateInput = get_input_models()
    inp = ConsolidateInput(
        user_id="eve",
        tools=["dedupe"],
        window=5,
        soft_cap_tokens=16000,
    )
    ctx = MagicMock()
    ctx.workflow_run_id = "run-cons"

    result = handler(inp, ctx)

    memory_mock.consolidate.assert_called_once_with(
        tools=["dedupe"],
        window=5,
        soft_cap_tokens=16000,
    )
    assert result is not None


# ---------------------------------------------------------------------------
# Test: workflow names match between submit-side and worker-side
# ---------------------------------------------------------------------------

def test_workflow_names_match_executor_workflows(worker_env):
    """register_workflows creates workflows named 'diffmem-write' and 'diffmem-consolidate'."""
    *_, worker_mod = worker_env

    from diffmem.executor.hatchet_workflows import register_workflows

    mock_hatchet = MagicMock()
    workflow_obj = MagicMock()
    mock_hatchet.workflow.return_value = workflow_obj

    register_workflows(mock_hatchet)

    calls = mock_hatchet.workflow.call_args_list
    names = [c.kwargs.get("name") or c.args[0] for c in calls]

    assert "diffmem-write" in names, f"diffmem-write not found in {names}"
    assert "diffmem-consolidate" in names, f"diffmem-consolidate not found in {names}"


# ---------------------------------------------------------------------------
# Test: _get_memory caches DiffMemory instances
# ---------------------------------------------------------------------------

def test_get_memory_caches_instances(worker_env, monkeypatch):
    """_get_memory returns the same DiffMemory instance on repeated calls for same user_id."""
    *_, worker_mod = worker_env

    # Reset singletons for clean test
    worker_mod._repo_manager_singleton = None
    worker_mod._memory_cache.clear()

    mock_repo_manager = MagicMock()
    mock_repo_manager.get_user_worktree.return_value = "/data/worktrees/alice"

    mock_memory_1 = MagicMock()
    mock_memory_2 = MagicMock()
    memory_call_count = {"n": 0}

    def fake_diff_memory(path, user_id, api_key, model):
        memory_call_count["n"] += 1
        return mock_memory_1  # always same instance

    mock_repo_manager_cls = MagicMock(return_value=mock_repo_manager)

    # Patch RepoManager and DiffMemory in the modules that _get_memory imports lazily
    import diffmem.repo_manager as rm_mod
    import diffmem.api as api_mod

    monkeypatch.setattr(rm_mod, "RepoManager", mock_repo_manager_cls)
    monkeypatch.setattr(api_mod, "DiffMemory", fake_diff_memory)

    first = worker_mod._get_memory("alice")
    second = worker_mod._get_memory("alice")

    assert first is second, "Second call should return the cached instance"
    assert memory_call_count["n"] == 1, "DiffMemory should be constructed only once"
    mock_repo_manager_cls.assert_called_once(), "RepoManager should be constructed only once"
