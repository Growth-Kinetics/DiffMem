# CAPABILITY: End-to-end integration tests for M7 — full stack smoke tests.
# SCOPE: HTTP endpoint → executor.submit_write/submit_consolidate → InlineExecutor
#        → thread pool → thunk → JobStore → polling → callback receipt.
# PURPOSE: Proves the whole stack works together. Component tests live in
#          test_executor_inline.py / test_executor_endpoints.py. This file is
#          the integration layer future engineers read first when debugging a
#          regression in the executor pipeline.
# CONSTRAINTS: No LLM calls, no real git worktrees. All DiffMemory operations
#              are replaced by MagicMock stubs that return controlled values or
#              raise on demand.

from __future__ import annotations

import importlib
import sys
import time
import threading
import json
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Shared test client factory
# ---------------------------------------------------------------------------

def _build_client(monkeypatch, *, process_fn=None, consolidate_fn=None):
    """Return (TestClient, executor) with a fully wired InlineExecutor.

    All DiffMemory methods are MagicMock stubs unless overridden via
    process_fn / consolidate_fn.  No real LLM or git operations happen.
    """
    monkeypatch.setenv("DEFAULT_MODEL", "test-model")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.setenv("REQUIRE_AUTH", "false")

    from fastapi.testclient import TestClient
    import diffmem.server as server_mod
    importlib.reload(server_mod)

    from diffmem.executor.inline import InlineExecutor

    # ---- stub DiffMemory ------------------------------------------------
    memory = MagicMock()

    # process_and_commit_session
    if process_fn is not None:
        memory.process_and_commit_session.side_effect = process_fn
    else:
        memory.process_and_commit_session.return_value = None

    # consolidate
    _default_consolidate = {
        "status": "ok",
        "tools_run": ["dedupe"],
        "commits": ["abc123"],
        "results": {},
        "summary": "e2e-stub",
    }
    if consolidate_fn is not None:
        memory.consolidate.side_effect = consolidate_fn
    else:
        memory.consolidate.return_value = _default_consolidate

    # passthrough stubs
    memory.process_session.return_value = None
    memory.commit_session.return_value = None
    memory.process_commit_and_consolidate.return_value = {
        "consolidate": _default_consolidate
    }

    # ---- wire into server -----------------------------------------------
    server_mod.memory_instances["u1"] = memory
    monkeypatch.setattr(
        server_mod, "get_memory_instance", lambda uid, allow_unboarded=False: memory
    )

    # no-op backup so tests don't need RepoManager
    async def noop_backup(uid: str) -> None:
        return None

    monkeypatch.setattr(server_mod, "backup_user", noop_backup)

    # inject real InlineExecutor (bypasses lifespan startup)
    pool = ThreadPoolExecutor(max_workers=4)
    executor = InlineExecutor(pool)
    server_mod.app.state.executor = executor

    client = TestClient(server_mod.app)
    return client, executor


# ---------------------------------------------------------------------------
# Helper: poll GET /memory/u1/jobs/{job_id} until terminal or timeout
# ---------------------------------------------------------------------------

def _poll_until_done(client, job_id: str, timeout: float = 5.0) -> dict:
    """Return the job dict from the polling endpoint when status is terminal."""
    deadline = time.monotonic() + timeout
    while True:
        r = client.get(f"/memory/u1/jobs/{job_id}")
        assert r.status_code == 200, r.text
        body = r.json()
        if body["job"]["status"] in ("completed", "failed"):
            return body["job"]
        if time.monotonic() >= deadline:
            pytest.fail(
                f"Job {job_id} did not reach terminal state within {timeout}s; "
                f"last status={body['job']['status']}"
            )
        time.sleep(0.05)


# ===========================================================================
# Test 1: sync default full cycle
# ===========================================================================

def test_e2e_sync_default_full_cycle(monkeypatch, tmp_path):
    """POST /memory/u1/process-and-commit (no ?sync param) → legacy success shape.

    Verifies:
    - Response shape matches pre-M2 contract (status=success, session_id, message, metadata).
    - job_id is present in metadata.
    - GET /jobs/{job_id} returns status=completed.
    """
    client, _ = _build_client(monkeypatch)

    r = client.post("/memory/u1/process-and-commit", json={
        "memory_input": "Today Alex talked about the project.",
        "session_id": "e2e-sync-001",
    })
    assert r.status_code == 200, r.text
    body = r.json()

    # --- legacy response shape ---
    assert body["status"] == "success", f"Unexpected body: {body}"
    assert body.get("session_id") == "e2e-sync-001"
    assert "message" in body
    assert "metadata" in body

    # --- job_id in metadata ---
    meta = body["metadata"]
    assert "job_id" in meta, f"job_id missing from metadata: {meta}"
    job_id = meta["job_id"]
    assert meta["user_id"] == "u1"

    # --- poll the job endpoint ---
    job = _poll_until_done(client, job_id, timeout=5.0)
    assert job["status"] == "completed", f"Unexpected job status: {job}"
    assert job["job_id"] == job_id


# ===========================================================================
# Test 2: async full cycle with real callback server
# ===========================================================================

class _CallbackServer:
    """Tiny HTTP server that records incoming POST bodies.

    Runs in a background thread; binds to localhost:0 so the OS assigns a
    free port.  Use as a context manager.
    """

    def __init__(self):
        self.received: list[dict] = []
        self._lock = threading.Lock()
        # httpd is set in __enter__
        self._httpd: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        assert self._httpd is not None
        host, port = self._httpd.server_address
        return f"http://localhost:{port}"

    def __enter__(self):
        received_ref = self.received
        lock_ref = self._lock

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("content-length", 0))
                raw = self.rfile.read(length)
                with lock_ref:
                    try:
                        received_ref.append(json.loads(raw))
                    except Exception:
                        received_ref.append({"_raw": raw.decode(errors="replace")})
                self.send_response(200)
                self.end_headers()

            def log_message(self, fmt, *args):  # silence noisy access log
                pass

        self._httpd = HTTPServer(("localhost", 0), _Handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._httpd:
            self._httpd.shutdown()


def test_e2e_async_full_cycle_with_callback(monkeypatch, tmp_path):
    """POST /process-and-commit?sync=false + callback_url.

    Verifies:
    - Response arrives quickly (queued shape: status=queued, job_id, submitted_at).
    - Polling GET /jobs/{job_id} eventually shows status=completed.
    - The real HTTP callback server receives a POST with matching job_id and
      status=completed.
    """
    import diffmem.executor.inline as inline_mod

    with _CallbackServer() as cb_server:
        client, _ = _build_client(monkeypatch)

        start = time.monotonic()
        r = client.post(
            "/memory/u1/process-and-commit?sync=false",
            json={
                "memory_input": "async transcript",
                "session_id": "e2e-async-001",
                "callback_url": cb_server.url + "/hook",
            },
        )
        elapsed = time.monotonic() - start

        # --- immediate queued response ---
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "queued", f"Expected queued, got: {body}"
        assert "job_id" in body
        assert "submitted_at" in body
        assert "poll_url" in body.get("metadata", {})
        job_id = body["job_id"]

        # Should return well under 1 s (no LLM / git blocking)
        assert elapsed < 0.8, f"Async response was too slow: {elapsed:.2f}s"

        # --- poll until completed ---
        job = _poll_until_done(client, job_id, timeout=5.0)
        assert job["status"] == "completed"

        # --- callback received ---
        cb_deadline = time.monotonic() + 3.0
        while time.monotonic() < cb_deadline and len(cb_server.received) == 0:
            time.sleep(0.05)

        assert len(cb_server.received) >= 1, (
            f"Callback was not received within 3s. "
            f"job_id={job_id}"
        )
        payload = cb_server.received[0]
        assert payload.get("job_id") == job_id, f"job_id mismatch: {payload}"
        assert payload.get("status") == "completed", f"Unexpected callback status: {payload}"


# ===========================================================================
# Test 3: per-user serialization observable via timing
# ===========================================================================

def test_e2e_per_user_serialization_observable(monkeypatch, tmp_path):
    """Two writes for same user with ?sync=false; thunks sleep 0.3s each.

    Per-user serialization guarantee: the second job must start AFTER the first
    completes.  Verified by comparing started_at / completed_at timestamps on
    the two JobResults.
    """
    SLEEP_TIME = 0.3  # seconds per thunk

    def slow_work(*args, **kwargs):
        time.sleep(SLEEP_TIME)

    client, _ = _build_client(monkeypatch, process_fn=slow_work)

    # Submit both jobs quickly before either can complete
    r1 = client.post("/memory/u1/process-and-commit?sync=false", json={
        "memory_input": "job A",
        "session_id": "serial-A",
    })
    r2 = client.post("/memory/u1/process-and-commit?sync=false", json={
        "memory_input": "job B",
        "session_id": "serial-B",
    })
    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text

    job_id_a = r1.json()["job_id"]
    job_id_b = r2.json()["job_id"]

    # Poll both until done — allow enough time for two sequential 0.3s thunks
    job_a = _poll_until_done(client, job_id_a, timeout=6.0)
    job_b = _poll_until_done(client, job_id_b, timeout=6.0)

    assert job_a["status"] == "completed", f"Job A not completed: {job_a}"
    assert job_b["status"] == "completed", f"Job B not completed: {job_b}"

    # Parse timestamps
    from datetime import datetime, timezone

    def _parse(ts: str | None) -> datetime:
        assert ts is not None, "Missing timestamp"
        # Python 3.10 fromisoformat doesn't handle 'Z' suffix; replace it.
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))

    a_started = _parse(job_a["started_at"])
    a_completed = _parse(job_a["completed_at"])
    b_started = _parse(job_b["started_at"])

    # The second job must have started AFTER the first completed (serialized)
    assert b_started >= a_completed, (
        f"Jobs ran concurrently (overlapping) — serialization broken!\n"
        f"  Job A: started={a_started.isoformat()} completed={a_completed.isoformat()}\n"
        f"  Job B: started={b_started.isoformat()}"
    )


# ===========================================================================
# Test 4: failure path — sync 500 and async poll→failed
# ===========================================================================

def test_e2e_failure_path(monkeypatch, tmp_path):
    """process_and_commit_session raises → correct error surfacing in both modes.

    Sync mode: HTTP 500 with error message in detail.
    Async mode: poll shows status=failed, error field populated.
    """
    def explode(*args, **kwargs):
        raise RuntimeError("e2e deliberate failure")

    client, _ = _build_client(monkeypatch, process_fn=explode)

    # --- sync mode → HTTP 500 ---
    r_sync = client.post("/memory/u1/process-and-commit", json={
        "memory_input": "boom",
        "session_id": "fail-sync",
    })
    assert r_sync.status_code == 500, r_sync.text
    detail = r_sync.json().get("detail", "")
    assert "e2e deliberate failure" in detail, f"Error text missing from detail: {detail}"

    # --- async mode → poll → failed ---
    r_async = client.post("/memory/u1/process-and-commit?sync=false", json={
        "memory_input": "boom",
        "session_id": "fail-async",
    })
    assert r_async.status_code == 200, r_async.text
    job_id = r_async.json()["job_id"]

    job = _poll_until_done(client, job_id, timeout=5.0)
    assert job["status"] == "failed", f"Expected failed, got: {job['status']}"
    assert job.get("error") is not None, "error field must be populated on failure"
    assert "e2e deliberate failure" in job["error"], (
        f"Error string not propagated: {job['error']}"
    )


# ===========================================================================
# Test 5: consolidate endpoint — same wiring shape as write
# ===========================================================================

def test_e2e_consolidate_endpoint(monkeypatch, tmp_path):
    """POST /memory/u1/consolidate exercises the submit_consolidate path.

    Verifies:
    - Sync default returns status=success, consolidate payload in body, job_id in metadata.
    - Async (?sync=false) returns queued shape.
    - Polling the queued job reaches status=completed.
    """
    expected_consolidate = {
        "status": "ok",
        "tools_run": ["dedupe", "link"],
        "commits": ["deadbeef"],
        "results": {"dedupe": {"merged": 0}},
        "summary": "e2e-consolidate",
    }

    def consolidate_fn(**kwargs):
        return expected_consolidate

    client, _ = _build_client(monkeypatch, consolidate_fn=consolidate_fn)

    # --- sync default ---
    r = client.post("/memory/u1/consolidate", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", f"Unexpected body: {body}"
    assert "consolidate" in body, "consolidate key missing from sync response"
    assert "metadata" in body
    assert "job_id" in body["metadata"], f"job_id missing: {body['metadata']}"

    # The consolidate payload should be embedded in the response
    assert body["consolidate"]["status"] == "ok"

    # --- async path ---
    r2 = client.post("/memory/u1/consolidate?sync=false", json={})
    assert r2.status_code == 200, r2.text
    body2 = r2.json()
    assert body2["status"] == "queued"
    job_id = body2["job_id"]
    assert "/jobs/" in body2["metadata"]["poll_url"]

    job = _poll_until_done(client, job_id, timeout=5.0)
    assert job["status"] == "completed"
    # The result dict stored in the job should contain the consolidate payload
    assert job.get("result") is not None
    assert "consolidate" in job["result"]


# ===========================================================================
# Test 6: session_id idempotency passthrough — return value flows through
# ===========================================================================

def test_e2e_session_id_idempotency_passthrough(monkeypatch, tmp_path):
    """Verify the thunk's return value reaches JobResult.result.

    DiffMemory.process_and_commit_session is DiffMemory's concern; we only
    verify that whatever the thunk returns is stored faithfully in
    JobResult.result and is accessible via the polling endpoint.

    This is a "we didn't break it" test — if the executor pipeline drops or
    mutates the return value the sync response would lose data and callers
    relying on the result dict would silently misbehave.
    """
    # Simulate a dedup scenario: DiffMemory returns None (as it does in
    # production — the thunk closes over the session_id and puts it in the
    # return dict itself).  Verify that the session_id echoed in the server
    # thunk makes it into the job result.

    client, _ = _build_client(monkeypatch)

    # Sync mode — the result ends up embedded in the HTTP response AND in the job
    r = client.post("/memory/u1/process-and-commit?sync=false", json={
        "memory_input": "idempotency test",
        "session_id": "idem-001",
    })
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    job = _poll_until_done(client, job_id, timeout=5.0)
    assert job["status"] == "completed"

    # The thunk in server.py returns {"session_id": ..., "message": ...}
    result = job.get("result")
    assert result is not None, "JobResult.result must be populated"
    assert result.get("session_id") == "idem-001", (
        f"session_id not propagated through executor into JobResult: {result}"
    )
    assert "message" in result, f"message missing from thunk result: {result}"
