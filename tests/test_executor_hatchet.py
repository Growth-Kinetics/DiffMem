"""
Tests for HatchetExecutor (M3) — all tests run WITHOUT hatchet-sdk installed.
Uses unittest.mock to simulate the SDK so no real Hatchet connection is made.
"""

from __future__ import annotations

import sys
import types
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diffmem.executor.base import ConsolidatePayload, WritePayload


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def pool():
    p = ThreadPoolExecutor(max_workers=2)
    yield p
    p.shutdown(wait=True)


def _make_hatchet_sdk_mock():
    """Build a minimal fake hatchet_sdk module that the executor can import."""
    sdk = types.ModuleType("hatchet_sdk")

    # ConcurrencyLimitStrategy enum-like
    strategy = MagicMock()
    strategy.GROUP_ROUND_ROBIN = "GROUP_ROUND_ROBIN"

    # ConcurrencyExpression
    conc_expr = MagicMock()
    sdk.ConcurrencyLimitStrategy = strategy
    sdk.ConcurrencyExpression = MagicMock(return_value=conc_expr)

    # Hatchet client mock
    client = MagicMock()

    # workflow() returns a workflow object whose .run() returns a WorkflowRunRef mock
    workflow_obj = MagicMock()
    run_ref = MagicMock()
    run_ref.workflow_run_id = "test-run-id-123"
    workflow_obj.run.return_value = run_ref
    client.workflow.return_value = workflow_obj

    # runs.get_status returns a mock status
    mock_status = MagicMock()
    mock_status.name = "QUEUED"
    client.runs.get_status.return_value = mock_status

    sdk.Hatchet = MagicMock(return_value=client)

    # pydantic BaseModel — use a real simple class
    from pydantic import BaseModel
    sdk.BaseModel = BaseModel  # not actually on the sdk module, but pydantic is available

    return sdk, client, workflow_obj, run_ref


def _install_sdk_mock(monkeypatch, sdk_mock):
    """Install a fake hatchet_sdk into sys.modules for the duration of a test."""
    monkeypatch.setitem(sys.modules, "hatchet_sdk", sdk_mock)
    # Also clear any cached imports of hatchet_workflows / hatchet so they re-import.
    for mod_name in list(sys.modules.keys()):
        if "hatchet" in mod_name and mod_name != "hatchet_sdk":
            monkeypatch.delitem(sys.modules, mod_name, raising=False)


# ---------------------------------------------------------------------------
# Test: ImportError without SDK
# ---------------------------------------------------------------------------

def test_hatchet_executor_raises_without_sdk(monkeypatch, pool):
    """HatchetExecutor.__init__ raises ImportError with install hint if SDK missing."""
    monkeypatch.setitem(sys.modules, "hatchet_sdk", None)  # type: ignore[arg-type]
    # Remove any cached executor module so fresh import triggers the check.
    for mod_name in list(sys.modules.keys()):
        if "hatchet" in mod_name and mod_name != "hatchet_sdk":
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    # Re-import fresh
    import importlib
    import diffmem.executor.hatchet as hatchet_mod
    importlib.reload(hatchet_mod)

    with pytest.raises(ImportError, match="pip install diffmem\\[hatchet\\]"):
        hatchet_mod.HatchetExecutor(pool)


# ---------------------------------------------------------------------------
# Test: RuntimeError without token
# ---------------------------------------------------------------------------

def test_hatchet_executor_raises_without_token(monkeypatch, pool):
    """HatchetExecutor raises RuntimeError if HATCHET_CLIENT_TOKEN not set."""
    sdk_mock, client, workflow_obj, run_ref = _make_hatchet_sdk_mock()
    _install_sdk_mock(monkeypatch, sdk_mock)
    monkeypatch.delenv("HATCHET_CLIENT_TOKEN", raising=False)

    import importlib
    import diffmem.executor.hatchet as hatchet_mod
    import diffmem.executor.hatchet_workflows as wf_mod
    importlib.reload(wf_mod)
    importlib.reload(hatchet_mod)

    with pytest.raises(RuntimeError, match="HATCHET_CLIENT_TOKEN"):
        hatchet_mod.HatchetExecutor(pool)


# ---------------------------------------------------------------------------
# Shared fixture: a fully mocked HatchetExecutor
# ---------------------------------------------------------------------------

@pytest.fixture()
def hatchet_executor(monkeypatch, pool):
    """Returns a HatchetExecutor with all Hatchet calls mocked."""
    sdk_mock, client, workflow_obj, run_ref = _make_hatchet_sdk_mock()
    _install_sdk_mock(monkeypatch, sdk_mock)
    monkeypatch.setenv("HATCHET_CLIENT_TOKEN", "fake-token")

    import importlib
    import diffmem.executor.hatchet_workflows as wf_mod
    import diffmem.executor.hatchet as hatchet_mod
    importlib.reload(wf_mod)
    importlib.reload(hatchet_mod)

    executor = hatchet_mod.HatchetExecutor(pool)
    # Both workflows share the same mock workflow_obj for simplicity.
    executor._write_workflow = workflow_obj
    executor._consolidate_workflow = workflow_obj
    # Ensure the mock run_ref workflow_run_id is used.
    workflow_obj.run.return_value = run_ref

    return executor, client, workflow_obj, run_ref


# ---------------------------------------------------------------------------
# Test: submit_write
# ---------------------------------------------------------------------------

def test_hatchet_submit_write_returns_job_id(hatchet_executor):
    """submit_write calls workflow.run() and returns a JobHandle with the run_id."""
    executor, client, workflow_obj, run_ref = hatchet_executor

    payload = WritePayload(
        user_id="alice",
        memory_input="hello world",
        session_id="sess-001",
        session_date="2026-06-05",
    )
    handle = executor.submit_write("alice", None, payload=payload, callback_url=None)

    assert handle.job_id == run_ref.workflow_run_id
    assert handle.status == "queued"

    # workflow.run must have been called with wait_for_result=False
    call_kwargs = workflow_obj.run.call_args
    assert call_kwargs is not None
    assert call_kwargs.kwargs.get("wait_for_result") is False
    assert "additional_metadata" in call_kwargs.kwargs
    assert call_kwargs.kwargs["additional_metadata"]["user_id"] == "alice"


# ---------------------------------------------------------------------------
# Test: submit_consolidate
# ---------------------------------------------------------------------------

def test_hatchet_submit_consolidate_returns_job_id(hatchet_executor):
    """submit_consolidate calls workflow.run() and returns a JobHandle with the run_id."""
    executor, client, workflow_obj, run_ref = hatchet_executor

    payload = ConsolidatePayload(
        user_id="bob",
        tools=["dedupe", "link"],
        window=5,
        soft_cap_tokens=16000,
    )
    handle = executor.submit_consolidate("bob", None, payload=payload, callback_url="http://cb")

    assert handle.job_id == run_ref.workflow_run_id
    assert handle.status == "queued"

    call_kwargs = workflow_obj.run.call_args
    assert call_kwargs.kwargs.get("wait_for_result") is False
    assert call_kwargs.kwargs["additional_metadata"]["user_id"] == "bob"


# ---------------------------------------------------------------------------
# Test: get_job status mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hatchet_name,expected", [
    ("PENDING", "queued"),
    ("QUEUED", "queued"),
    ("RUNNING", "running"),
    ("COMPLETED", "completed"),
    ("SUCCEEDED", "completed"),
    ("FAILED", "failed"),
    ("CANCELLED", "failed"),
    ("CANCELLING", "failed"),
])
def test_hatchet_get_job_maps_status_correctly(hatchet_executor, hatchet_name, expected):
    """get_job maps all Hatchet V1TaskStatus variants to our JobStatus."""
    executor, client, workflow_obj, run_ref = hatchet_executor

    # Seed jobstore with a queued job.
    from diffmem.executor.base import JobResult
    from datetime import datetime, timezone
    job_id = "test-job-status-map"
    executor._jobstore.put(JobResult(job_id=job_id, status="queued", submitted_at=datetime.now(timezone.utc)))

    mock_status = MagicMock()
    mock_status.name = hatchet_name
    client.runs.get_status.return_value = mock_status

    result = executor.get_job(job_id)
    assert result is not None
    assert result.status == expected, f"Expected {expected!r} for hatchet status {hatchet_name!r}"


# ---------------------------------------------------------------------------
# Test: wait_for uses ref.result()
# ---------------------------------------------------------------------------

def test_hatchet_wait_for_uses_ref_result(hatchet_executor):
    """wait_for calls ref.result() and returns a completed JobResult."""
    executor, client, workflow_obj, run_ref = hatchet_executor

    result_data = {"session_id": "sess-1", "message": "done"}
    run_ref.result.return_value = result_data

    # Submit a job so the ref is cached.
    payload = WritePayload(
        user_id="carol",
        memory_input="test",
        session_id="sess-1",
    )
    handle = executor.submit_write("carol", None, payload=payload)

    job_result = executor.wait_for(handle.job_id, timeout=5.0)

    assert job_result.status == "completed"
    assert job_result.result == result_data
    run_ref.result.assert_called_once()


# ---------------------------------------------------------------------------
# Test: factory returns HatchetExecutor when SDK available + token set
# ---------------------------------------------------------------------------

def test_factory_hatchet_returns_hatchet_executor(monkeypatch, pool):
    """build_executor(pool) returns HatchetExecutor when EXECUTOR=hatchet and SDK present."""
    sdk_mock, client, workflow_obj, run_ref = _make_hatchet_sdk_mock()
    _install_sdk_mock(monkeypatch, sdk_mock)
    monkeypatch.setenv("EXECUTOR", "hatchet")
    monkeypatch.setenv("HATCHET_CLIENT_TOKEN", "fake-token")

    import importlib
    import diffmem.executor.hatchet_workflows as wf_mod
    import diffmem.executor.hatchet as hatchet_mod
    import diffmem.executor.factory as factory_mod
    importlib.reload(wf_mod)
    importlib.reload(hatchet_mod)
    importlib.reload(factory_mod)

    executor = factory_mod.build_executor(pool)
    assert isinstance(executor, hatchet_mod.HatchetExecutor)


# ---------------------------------------------------------------------------
# Test: factory raises clean ImportError without extras
# ---------------------------------------------------------------------------

def test_factory_hatchet_raises_clean_error_without_extras(monkeypatch, pool):
    """build_executor raises ImportError with install hint when SDK not installed."""
    monkeypatch.setitem(sys.modules, "hatchet_sdk", None)  # type: ignore[arg-type]
    for mod_name in list(sys.modules.keys()):
        if "hatchet" in mod_name and mod_name != "hatchet_sdk":
            monkeypatch.delitem(sys.modules, mod_name, raising=False)
    monkeypatch.setenv("EXECUTOR", "hatchet")

    import importlib
    import diffmem.executor.hatchet as hatchet_mod
    import diffmem.executor.factory as factory_mod
    importlib.reload(hatchet_mod)
    importlib.reload(factory_mod)

    with pytest.raises(ImportError, match="pip install diffmem\\[hatchet\\]"):
        factory_mod.build_executor(pool)
