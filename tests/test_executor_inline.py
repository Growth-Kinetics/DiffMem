"""
Tests for the executor capability (M1).

Covers: InlineExecutor, JobStore, factory dispatch, callback URL handling.
"""

from __future__ import annotations

import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

# Make src/ importable without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diffmem.executor.base import JobStatus
from diffmem.executor.factory import build_executor
from diffmem.executor.inline import InlineExecutor
from diffmem.executor.jobstore import JobStore, _MAX_ENTRIES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def pool():
    """Fresh ThreadPoolExecutor per test; shut down after."""
    p = ThreadPoolExecutor(max_workers=4)
    yield p
    p.shutdown(wait=True)


@pytest.fixture()
def executor(pool):
    return InlineExecutor(pool)


# ---------------------------------------------------------------------------
# Basic submit / wait
# ---------------------------------------------------------------------------

def test_inline_submit_write_completes(executor):
    handle = executor.submit_write("alice", lambda: {"ok": True})
    result = executor.wait_for(handle.job_id, timeout=5)
    assert result.status == "completed"
    assert result.result == {"ok": True}


def test_inline_submit_write_failure_captured(executor):
    def boom():
        raise RuntimeError("boom")

    handle = executor.submit_write("alice", boom)
    result = executor.wait_for(handle.job_id, timeout=5)
    assert result.status == "failed"
    assert "boom" in (result.error or "")


# ---------------------------------------------------------------------------
# Per-user serialization
# ---------------------------------------------------------------------------

def test_inline_per_user_serialization(executor):
    """Two writes for the same user must be sequential (second starts after first ends)."""
    order: list[tuple[str, float]] = []  # (event, monotonic timestamp)

    def slow_work(label: str):
        def _work():
            time.sleep(0.3)
            order.append((label, time.monotonic()))
            return {"label": label}
        return _work

    h1 = executor.submit_write("alice", slow_work("first"))
    # Small delay to ensure first job starts before second is submitted
    time.sleep(0.01)
    h2 = executor.submit_write("alice", slow_work("second"))

    r1 = executor.wait_for(h1.job_id, timeout=5)
    r2 = executor.wait_for(h2.job_id, timeout=5)

    assert r1.status == "completed"
    assert r2.status == "completed"
    # second job must have started at or after first completed
    assert r2.started_at >= r1.completed_at


# ---------------------------------------------------------------------------
# Parallelism across users
# ---------------------------------------------------------------------------

def test_inline_parallel_across_users(executor):
    """Writes for different users run in parallel; all three should finish well under 0.6 s."""
    handles = []
    start = time.monotonic()
    for uid in ("u1", "u2", "u3"):
        h = executor.submit_write(uid, lambda: (time.sleep(0.3), {"done": True})[1])
        handles.append(h)

    results = [executor.wait_for(h.job_id, timeout=5) for h in handles]
    for r in results:
        assert r.status == "completed"

    # All three should complete within 0.6 s of the first submission
    # (they run in parallel on pool with max_workers=4)
    latest_completed = max(r.completed_at.timestamp() for r in results)
    deadline = handles[0].submitted_at.timestamp() + 0.9  # generous margin
    assert latest_completed <= deadline, (
        f"Jobs took too long — parallelism not working. "
        f"Wall time since first submit: {latest_completed - handles[0].submitted_at.timestamp():.2f}s"
    )


# ---------------------------------------------------------------------------
# get_job
# ---------------------------------------------------------------------------

def test_inline_get_job_returns_none_for_unknown(executor):
    assert executor.get_job("nope") is None


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

def test_inline_wait_for_timeout(executor):
    handle = executor.submit_write("alice", lambda: (time.sleep(1.0), {})[1])
    with pytest.raises(TimeoutError):
        executor.wait_for(handle.job_id, timeout=0.1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_factory_default_returns_inline(monkeypatch, pool):
    monkeypatch.delenv("EXECUTOR", raising=False)
    e = build_executor(pool)
    assert isinstance(e, InlineExecutor)


def test_factory_explicit_inline(monkeypatch, pool):
    monkeypatch.setenv("EXECUTOR", "inline")
    e = build_executor(pool)
    assert isinstance(e, InlineExecutor)


def test_factory_hatchet_raises_not_implemented(monkeypatch, pool):
    monkeypatch.setenv("EXECUTOR", "hatchet")
    with pytest.raises(NotImplementedError, match="pip install diffmem\\[hatchet\\]"):
        build_executor(pool)


def test_factory_invalid_raises_valueerror(monkeypatch, pool):
    monkeypatch.setenv("EXECUTOR", "nonsense")
    with pytest.raises(ValueError):
        build_executor(pool)


# ---------------------------------------------------------------------------
# JobStore eviction
# ---------------------------------------------------------------------------

def test_jobstore_fifo_eviction(caplog):
    import logging
    from diffmem.executor.base import JobResult
    from datetime import datetime, timezone

    store = JobStore()
    now = datetime.now(timezone.utc)

    first_id = "job-0000"
    store.put(JobResult(job_id=first_id, status="completed", submitted_at=now))

    with caplog.at_level(logging.INFO, logger="diffmem.executor.jobstore"):
        for i in range(1, _MAX_ENTRIES + 1):
            store.put(JobResult(job_id=f"job-{i:04d}", status="queued", submitted_at=now))

    # The first entry should have been evicted
    assert store.get(first_id) is None
    assert any("JOBSTORE_EVICTED" in r.message and first_id in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Callback URL
# ---------------------------------------------------------------------------

def test_callback_url_invoked_on_success(monkeypatch, executor):
    """httpx.post is called once with the right URL and a payload containing job_id + status."""
    import diffmem.executor.inline as inline_mod

    calls: list[dict] = []

    def fake_post(url, *, json=None, timeout=None):
        calls.append({"url": url, "json": json})

    # Patch at the module level where inline.py imported it
    monkeypatch.setattr(inline_mod._http_lib, "post", fake_post)

    handle = executor.submit_write(
        "alice",
        lambda: {"ok": True},
        callback_url="http://test/cb",
    )
    result = executor.wait_for(handle.job_id, timeout=5)
    # Give the callback a moment to fire (it happens after lock release)
    time.sleep(0.1)

    assert result.status == "completed"
    assert len(calls) == 1
    assert calls[0]["url"] == "http://test/cb"
    payload = calls[0]["json"]
    assert payload["job_id"] == handle.job_id
    assert payload["status"] == "completed"


def test_callback_url_failure_does_not_raise(monkeypatch, executor, caplog):
    """A callback POST failure must not propagate — job still completes normally."""
    import logging
    import httpx
    import diffmem.executor.inline as inline_mod

    def exploding_post(url, *, json=None, timeout=None):
        raise httpx.ConnectError("unreachable")

    monkeypatch.setattr(inline_mod._http_lib, "post", exploding_post)

    handle = executor.submit_write(
        "alice",
        lambda: {"ok": True},
        callback_url="http://test/cb",
    )
    with caplog.at_level(logging.WARNING, logger="diffmem.executor.inline"):
        result = executor.wait_for(handle.job_id, timeout=5)
        time.sleep(0.1)  # let callback attempt complete

    assert result.status == "completed"
    assert any("CALLBACK_FAILED" in r.message for r in caplog.records)
