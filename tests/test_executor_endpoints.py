# CAPABILITY: Integration tests for M2 executor endpoint wiring.
# INPUTS: tmp_path -> fixture worktree + fake DiffMemory.
# OUTPUTS: Verifies sync/async ?sync query param, GET /jobs/{job_id}, callback_url,
#          failure propagation, and backwards-compatible response shapes.
# CONSTRAINTS: No network (callback_url uses monkeypatched httpx.post).

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tests._fixtures import build_worktree


# ---------------------------------------------------------------------------
# Test client factory
# ---------------------------------------------------------------------------

def _build_test_client(monkeypatch, tmp_path: Path, *, process_fn=None, consolidate_fn=None):
    """Spin up the FastAPI app with a stubbed DiffMemory + real InlineExecutor.

    process_fn / consolidate_fn are callables injected as the relevant
    memory methods so tests can control success / failure.
    """
    monkeypatch.setenv("DEFAULT_MODEL", "test-model")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.setenv("REQUIRE_AUTH", "false")

    from fastapi.testclient import TestClient
    import importlib
    import diffmem.server as server_mod
    importlib.reload(server_mod)

    from diffmem.executor.inline import InlineExecutor

    # Build a stub DiffMemory.
    memory = MagicMock()

    if process_fn is not None:
        memory.process_and_commit_session.side_effect = process_fn
    else:
        memory.process_and_commit_session.return_value = None

    if consolidate_fn is not None:
        memory.consolidate.side_effect = consolidate_fn
    else:
        memory.consolidate.return_value = {
            "status": "ok", "tools_run": ["dedupe"], "commits": ["c1"],
            "results": {}, "summary": "stub",
        }

    memory.process_session.return_value = None
    memory.commit_session.return_value = None
    memory.process_commit_and_consolidate.return_value = {
        "consolidate": {
            "status": "ok", "tools_run": ["dedupe"], "commits": ["c1"],
            "results": {}, "summary": "stub",
        }
    }

    # Inject memory + bypass lookup.
    server_mod.memory_instances["alex"] = memory
    monkeypatch.setattr(server_mod, "get_memory_instance", lambda uid, allow_unboarded=False: memory)

    # No-op backup.
    async def noop_backup(uid):
        return None
    monkeypatch.setattr(server_mod, "backup_user", noop_backup)

    # Inject real InlineExecutor (bypasses lifespan).
    pool = ThreadPoolExecutor(max_workers=4)
    executor = InlineExecutor(pool)
    server_mod.app.state.executor = executor

    client = TestClient(server_mod.app)
    return client, executor


# ---------------------------------------------------------------------------
# Test: process-and-commit sync default (backwards compat)
# ---------------------------------------------------------------------------

def test_process_and_commit_sync_default_returns_success_shape(monkeypatch, tmp_path):
    """No ?sync → InlineExecutor defaults to sync → pre-M2 compatible shape."""
    client, _ = _build_test_client(monkeypatch, tmp_path)
    r = client.post("/memory/alex/process-and-commit", json={
        "memory_input": "transcript",
        "session_id": "s-001",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success"
    assert body["session_id"] == "s-001"
    assert "message" in body
    assert "metadata" in body
    assert "job_id" in body["metadata"]
    assert body["metadata"]["user_id"] == "alex"


# ---------------------------------------------------------------------------
# Test: process-and-commit async
# ---------------------------------------------------------------------------

def test_process_and_commit_async_returns_job_id_immediately(monkeypatch, tmp_path):
    """?sync=false → returns queued shape without waiting."""
    def slow_work(*args, **kwargs):
        time.sleep(1.0)  # would block sync path

    client, _ = _build_test_client(monkeypatch, tmp_path, process_fn=slow_work)
    start = time.monotonic()
    r = client.post("/memory/alex/process-and-commit?sync=false", json={
        "memory_input": "transcript",
        "session_id": "s-async",
    })
    elapsed = time.monotonic() - start
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "queued"
    assert "job_id" in body
    assert "submitted_at" in body
    assert "poll_url" in body["metadata"]
    # Should return fast — well under 1s
    assert elapsed < 0.8, f"Async endpoint was too slow: {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Test: consolidate sync default
# ---------------------------------------------------------------------------

def test_consolidate_sync_default(monkeypatch, tmp_path):
    client, _ = _build_test_client(monkeypatch, tmp_path)
    r = client.post("/memory/alex/consolidate", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success"
    assert "consolidate" in body
    assert "metadata" in body
    assert "job_id" in body["metadata"]


# ---------------------------------------------------------------------------
# Test: consolidate async
# ---------------------------------------------------------------------------

def test_consolidate_async(monkeypatch, tmp_path):
    """?sync=false on consolidate → queued shape."""
    client, _ = _build_test_client(monkeypatch, tmp_path)
    r = client.post("/memory/alex/consolidate?sync=false", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "queued"
    assert "job_id" in body
    assert "/jobs/" in body["metadata"]["poll_url"]


# ---------------------------------------------------------------------------
# Test: GET /jobs/{job_id} — 404 for unknown
# ---------------------------------------------------------------------------

def test_get_job_returns_404_for_unknown(monkeypatch, tmp_path):
    client, _ = _build_test_client(monkeypatch, tmp_path)
    r = client.get("/memory/alex/jobs/nope-not-real")
    assert r.status_code == 404, r.text


# ---------------------------------------------------------------------------
# Test: GET /jobs/{job_id} — completed after async submit
# ---------------------------------------------------------------------------

def test_get_job_returns_result_after_completion(monkeypatch, tmp_path):
    """Submit async, poll until completed, check full job shape."""
    client, _ = _build_test_client(monkeypatch, tmp_path)
    r = client.post("/memory/alex/consolidate?sync=false", json={})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    # Poll until done (InlineExecutor runs in background thread; should be fast)
    deadline = time.monotonic() + 5.0
    status_val = None
    while time.monotonic() < deadline:
        poll = client.get(f"/memory/alex/jobs/{job_id}")
        assert poll.status_code == 200, poll.text
        status_val = poll.json()["job"]["status"]
        if status_val in ("completed", "failed"):
            break
        time.sleep(0.05)

    assert status_val == "completed", f"Job did not complete in time; last status={status_val}"
    poll_body = client.get(f"/memory/alex/jobs/{job_id}").json()
    assert poll_body["status"] == "success"
    assert "job" in poll_body
    assert poll_body["job"]["job_id"] == job_id


# ---------------------------------------------------------------------------
# Test: callback_url invoked on success
# ---------------------------------------------------------------------------

def test_callback_url_invoked(monkeypatch, tmp_path):
    """callback_url causes an httpx.post after job completes."""
    import diffmem.executor.inline as inline_mod

    calls: list[dict] = []

    def fake_post(url, *, json=None, timeout=None):
        calls.append({"url": url, "json": json})

    monkeypatch.setattr(inline_mod._http_lib, "post", fake_post)

    client, _ = _build_test_client(monkeypatch, tmp_path)
    r = client.post("/memory/alex/consolidate?sync=false", json={
        "callback_url": "http://test-cb/hook",
    })
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    # Wait for job + callback to fire
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and len(calls) == 0:
        time.sleep(0.05)

    assert len(calls) == 1, f"Expected 1 callback call, got {len(calls)}"
    assert calls[0]["url"] == "http://test-cb/hook"
    payload = calls[0]["json"]
    assert payload["job_id"] == job_id
    assert payload["status"] == "completed"


# ---------------------------------------------------------------------------
# Test: sync mode failure → HTTP 500
# ---------------------------------------------------------------------------

def test_sync_failure_returns_500(monkeypatch, tmp_path):
    """Thunk raises → sync mode → HTTP 500 with error in detail."""
    def boom(*args, **kwargs):
        raise RuntimeError("deliberate failure")

    client, _ = _build_test_client(monkeypatch, tmp_path, process_fn=boom)
    r = client.post("/memory/alex/process-and-commit", json={
        "memory_input": "x",
        "session_id": "s-fail",
    })
    assert r.status_code == 500, r.text
    detail = r.json()["detail"]
    assert "deliberate failure" in detail


# ---------------------------------------------------------------------------
# Test: async mode failure visible via polling
# ---------------------------------------------------------------------------

def test_async_failure_visible_via_polling(monkeypatch, tmp_path):
    """Thunk raises → async mode → poll shows status=failed, error set."""
    def boom(*args, **kwargs):
        raise RuntimeError("async failure")

    client, _ = _build_test_client(monkeypatch, tmp_path, process_fn=boom)
    r = client.post("/memory/alex/process-and-commit?sync=false", json={
        "memory_input": "x",
        "session_id": "s-async-fail",
    })
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    deadline = time.monotonic() + 5.0
    status_val = None
    while time.monotonic() < deadline:
        poll = client.get(f"/memory/alex/jobs/{job_id}")
        assert poll.status_code == 200, poll.text
        status_val = poll.json()["job"]["status"]
        if status_val in ("completed", "failed"):
            break
        time.sleep(0.05)

    assert status_val == "failed", f"Expected failed, got {status_val}"
    poll_body = client.get(f"/memory/alex/jobs/{job_id}").json()
    assert "async failure" in (poll_body["job"]["error"] or "")
