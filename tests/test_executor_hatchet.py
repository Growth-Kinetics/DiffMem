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


# ---------------------------------------------------------------------------
# Tests for Finding 1 (started_at enrichment) + Finding 2 (output unwrapping)
# ---------------------------------------------------------------------------

def test_unwrap_task_output_unwraps_known_workflow():
    """_unwrap_task_output strips the task-name wrapper for known workflows."""
    from diffmem.executor.hatchet import _unwrap_task_output

    result = _unwrap_task_output(
        "diffmem-write",
        {"process_and_commit": {"ok": True, "session_id": "s1"}},
    )
    assert result == {"ok": True, "session_id": "s1"}

    result = _unwrap_task_output(
        "diffmem-consolidate",
        {"consolidate": {"commits": 3}},
    )
    assert result == {"commits": 3}


def test_unwrap_task_output_passthrough_on_unknown_workflow():
    """Unknown workflow names pass through unchanged (no log)."""
    from diffmem.executor.hatchet import _unwrap_task_output

    wrapped = {"some_step": {"ok": True}}
    assert _unwrap_task_output("unknown-workflow", wrapped) == wrapped


def test_unwrap_task_output_passthrough_on_unexpected_shape(caplog):
    """Known workflow but mismatched shape passes through + logs WARNING."""
    from diffmem.executor.hatchet import _unwrap_task_output
    import logging

    caplog.set_level(logging.WARNING)
    weird = {"wrong_key": {"ok": True}, "second_key": {"ok": False}}
    assert _unwrap_task_output("diffmem-write", weird) == weird
    assert any("HATCHET_OUTPUT_SHAPE_UNEXPECTED" in rec.message for rec in caplog.records)


def test_unwrap_task_output_non_dict_passthrough():
    """Non-dict inputs pass through unchanged (defensive)."""
    from diffmem.executor.hatchet import _unwrap_task_output

    assert _unwrap_task_output("diffmem-write", "not a dict") == "not a dict"
    assert _unwrap_task_output("diffmem-write", None) is None
    assert _unwrap_task_output("diffmem-write", 42) == 42


def test_enrich_from_details_populates_started_and_completed(hatchet_executor):
    """_enrich_from_details copies started_at + finished_at from Hatchet into JobResult."""
    executor, client, workflow_obj, run_ref = hatchet_executor

    # Seed jobstore with a job that has no started_at.
    from diffmem.executor.base import JobResult
    from datetime import datetime, timezone
    job_id = "enrich-job-1"
    submitted = datetime.now(timezone.utc)
    executor._jobstore.put(JobResult(job_id=job_id, status="completed", submitted_at=submitted))
    # Also register the workflow name so unwrap path can find it.
    executor._refs[job_id] = (run_ref, "diffmem-write")

    # Mock hatchet.runs.get to return a details object with started/finished/output.
    started = datetime(2026, 6, 5, 12, 0, 0, tzinfo=timezone.utc)
    finished = datetime(2026, 6, 5, 12, 0, 5, tzinfo=timezone.utc)
    details_mock = MagicMock()
    details_mock.run.started_at = started
    details_mock.run.finished_at = finished
    details_mock.run.output = {"process_and_commit": {"ok": True}}
    client.runs.get.return_value = details_mock

    executor._enrich_from_details(job_id, workflow_name="diffmem-write")

    enriched = executor._jobstore.get(job_id)
    assert enriched is not None
    assert enriched.started_at == started
    assert enriched.completed_at == finished
    assert enriched.result == {"ok": True}, f"Expected unwrapped result; got {enriched.result!r}"


def test_enrich_from_details_swallows_api_errors(hatchet_executor, caplog):
    """Hatchet API errors during enrich are logged at WARNING and do not raise."""
    import logging
    executor, client, workflow_obj, run_ref = hatchet_executor
    caplog.set_level(logging.WARNING)

    # Seed jobstore.
    from diffmem.executor.base import JobResult
    from datetime import datetime, timezone
    job_id = "enrich-err-job"
    executor._jobstore.put(JobResult(job_id=job_id, status="completed", submitted_at=datetime.now(timezone.utc)))

    client.runs.get.side_effect = ConnectionError("network down")

    # Must NOT raise.
    executor._enrich_from_details(job_id)

    # Jobstore unchanged.
    after = executor._jobstore.get(job_id)
    assert after is not None
    assert after.started_at is None
    # WARNING log emitted.
    assert any("HATCHET_DETAILS_ERROR" in rec.message for rec in caplog.records)


def test_get_job_enriches_on_terminal_transition(hatchet_executor):
    """get_job calls _enrich_from_details when status transitions to terminal."""
    executor, client, workflow_obj, run_ref = hatchet_executor

    from diffmem.executor.base import JobResult
    from datetime import datetime, timezone
    job_id = "terminal-job-1"
    executor._jobstore.put(JobResult(job_id=job_id, status="queued", submitted_at=datetime.now(timezone.utc)))
    executor._refs[job_id] = (run_ref, "diffmem-write")

    # Hatchet says: completed.
    mock_status = MagicMock()
    mock_status.name = "COMPLETED"
    client.runs.get_status.return_value = mock_status

    # Hatchet details: full timestamps + wrapped output.
    started = datetime(2026, 6, 5, 12, 0, 0, tzinfo=timezone.utc)
    finished = datetime(2026, 6, 5, 12, 0, 5, tzinfo=timezone.utc)
    details_mock = MagicMock()
    details_mock.run.started_at = started
    details_mock.run.finished_at = finished
    details_mock.run.output = {"process_and_commit": {"ok": True, "session_id": "s9"}}
    client.runs.get.return_value = details_mock

    result = executor.get_job(job_id)
    assert result is not None
    assert result.status == "completed"
    assert result.started_at == started
    assert result.completed_at == finished
    assert result.result == {"ok": True, "session_id": "s9"}
    # The REST get was called exactly once (enrichment on terminal transition).
    assert client.runs.get.call_count == 1


def test_get_job_running_does_not_enrich(hatchet_executor):
    """get_job does NOT call _enrich_from_details while still running."""
    executor, client, workflow_obj, run_ref = hatchet_executor

    from diffmem.executor.base import JobResult
    from datetime import datetime, timezone
    job_id = "running-job-1"
    executor._jobstore.put(JobResult(job_id=job_id, status="queued", submitted_at=datetime.now(timezone.utc)))

    mock_status = MagicMock()
    mock_status.name = "RUNNING"
    client.runs.get_status.return_value = mock_status

    executor.get_job(job_id)
    client.runs.get.assert_not_called()


def test_wait_for_unwraps_and_enriches(hatchet_executor):
    """wait_for unwraps single-step result + populates started_at from Hatchet."""
    executor, client, workflow_obj, run_ref = hatchet_executor

    # ref.result() returns the wrapped shape Hatchet produces.
    run_ref.result.return_value = {"process_and_commit": {"ok": True, "session_id": "s7"}}

    # _enrich_from_details will be called; provide started_at.
    from datetime import datetime, timezone
    started = datetime(2026, 6, 5, 14, 0, 0, tzinfo=timezone.utc)
    details_mock = MagicMock()
    details_mock.run.started_at = started
    details_mock.run.finished_at = None  # finished_at set by wait_for itself
    details_mock.run.output = {"process_and_commit": {"ok": True, "session_id": "s7"}}
    client.runs.get.return_value = details_mock

    from diffmem.executor.base import WritePayload
    payload = WritePayload(user_id="dave", memory_input="x", session_id="s7")
    handle = executor.submit_write("dave", None, payload=payload)

    result = executor.wait_for(handle.job_id, timeout=5.0)
    assert result.status == "completed"
    assert result.result == {"ok": True, "session_id": "s7"}, (
        f"Expected unwrapped result; got {result.result!r}"
    )
    assert result.started_at == started


def test_wait_for_robust_to_enrich_failure(hatchet_executor):
    """wait_for returns completed JobResult even if _enrich_from_details fails."""
    executor, client, workflow_obj, run_ref = hatchet_executor

    run_ref.result.return_value = {"process_and_commit": {"ok": True}}
    client.runs.get.side_effect = ConnectionError("hatchet api unreachable")

    from diffmem.executor.base import WritePayload
    payload = WritePayload(user_id="eve", memory_input="y", session_id="s8")
    handle = executor.submit_write("eve", None, payload=payload)

    # Must not raise.
    result = executor.wait_for(handle.job_id, timeout=5.0)
    assert result.status == "completed"
    assert result.result == {"ok": True}  # unwrap happened before enrich attempt
    assert result.started_at is None  # enrich failed; degraded gracefully


def test_enrich_skips_result_when_jobstore_already_has_one(hatchet_executor, caplog):
    """Don't re-unwrap if wait_for already set the result (avoids noisy WARNINGs)."""
    import logging
    executor, client, workflow_obj, run_ref = hatchet_executor
    caplog.set_level(logging.WARNING)

    from diffmem.executor.base import JobResult
    from datetime import datetime, timezone
    job_id = "already-set-job"
    submitted = datetime.now(timezone.utc)
    # Pre-populate the jobstore with an unwrapped result (as wait_for would have).
    executor._jobstore.put(JobResult(
        job_id=job_id,
        status="completed",
        submitted_at=submitted,
        result={"ok": True, "session_id": "s1"},
    ))
    executor._refs[job_id] = (run_ref, "diffmem-write")

    # Hatchet returns a *raw* (potentially already-unwrapped) output that
    # wouldn't match the wrapper shape — we should ignore it, not warn.
    details_mock = MagicMock()
    details_mock.run.started_at = datetime(2026, 6, 5, 10, 0, 0, tzinfo=timezone.utc)
    details_mock.run.finished_at = datetime(2026, 6, 5, 10, 0, 5, tzinfo=timezone.utc)
    details_mock.run.output = {"ok": True, "session_id": "s1"}  # already unwrapped
    client.runs.get.return_value = details_mock

    executor._enrich_from_details(job_id)

    # No HATCHET_OUTPUT_SHAPE_UNEXPECTED warnings from this call.
    shape_warnings = [r for r in caplog.records if "HATCHET_OUTPUT_SHAPE_UNEXPECTED" in r.message]
    assert not shape_warnings, f"Expected no shape warnings, got: {[r.message for r in shape_warnings]}"

    # Result preserved (not overwritten).
    final = executor._jobstore.get(job_id)
    assert final is not None
    assert final.result == {"ok": True, "session_id": "s1"}
    # Timestamps populated.
    assert final.started_at is not None
    assert final.completed_at is not None


def test_unwrap_silent_when_expect_wrapped_false(caplog):
    """REST-source unwrap (expect_wrapped=False) is silent on already-unwrapped output."""
    from diffmem.executor.hatchet import _unwrap_task_output
    import logging
    caplog.set_level(logging.WARNING)

    # Output is already unwrapped (the shape Hatchet's REST runs.get() returns).
    already_unwrapped = {"session_id": "s1", "status": "completed", "user_id": "alice"}
    result = _unwrap_task_output("diffmem-write", already_unwrapped, expect_wrapped=False)

    # Passthrough, no warning.
    assert result == already_unwrapped
    shape_warns = [r for r in caplog.records if "HATCHET_OUTPUT_SHAPE_UNEXPECTED" in r.message]
    assert not shape_warns, f"Expected silent passthrough, got warnings: {shape_warns}"


def test_unwrap_warns_when_expect_wrapped_true_default(caplog):
    """Default (expect_wrapped=True) still warns on shape mismatch."""
    from diffmem.executor.hatchet import _unwrap_task_output
    import logging
    caplog.set_level(logging.WARNING)

    already_unwrapped = {"session_id": "s1", "status": "completed", "user_id": "alice"}
    result = _unwrap_task_output("diffmem-write", already_unwrapped)
    # Same passthrough behavior...
    assert result == already_unwrapped
    # ...but a warning IS emitted.
    shape_warns = [r for r in caplog.records if "HATCHET_OUTPUT_SHAPE_UNEXPECTED" in r.message]
    assert len(shape_warns) == 1, f"Expected 1 warning, got {len(shape_warns)}"
