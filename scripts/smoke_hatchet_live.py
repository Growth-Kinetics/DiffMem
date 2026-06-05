"""
Phase 1 live smoke test for HatchetExecutor against Hatchet Cloud.

PURPOSE: Validates that the executor abstraction + Hatchet SDK integration work
end-to-end against a live Hatchet engine, WITHOUT pulling in DiffMemory / LLM /
git. If this passes, the executor wiring is sound and the only remaining unknown
is the downstream pipeline (Phase 2).

WHAT IT TESTS:
  1. HatchetExecutor can connect to Hatchet Cloud with a real token.
  2. Workflows register successfully on the engine.
  3. submit_write enqueues runs that get picked up by a worker.
  4. Per-user concurrency: two jobs for the same user_id serialize via
     ConcurrencyExpression(expression="input.user_id", max_runs=1).
  5. Different users run in parallel.
  6. get_job / wait_for reflect terminal state correctly.

HOW IT RUNS:
  This script forks itself: the parent spawns a worker subprocess (worker mode),
  waits for it to register, then submits the test jobs (submitter mode). Both
  modes import the SAME hatchet_workflows.register_workflows() so the workflow
  names line up across processes (this is the cross-process contract we're
  validating).

REQUIREMENTS:
  - HATCHET_CLIENT_TOKEN env var set (load from .env.hatchet-test).
  - hatchet-sdk installed (pip install -e ".[hatchet]" or already in image).

USAGE:
  source <(grep -v '^#' /data/projects/DiffMem/.env.hatchet-test)
  PYTHONPATH=src python3 scripts/smoke_hatchet_live.py
"""
# NOTE: intentionally NO `from __future__ import annotations` here.
# Hatchet's @workflow.task() uses typing.get_type_hints() to introspect handler
# signatures; with PEP-563 lazy annotations the type names (e.g. WriteInput)
# would not be resolvable from this module's globals at decoration time.

import logging
from datetime import timedelta
import os
import sys
import time
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Make src/ importable for development runs.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("smoke")


# ─── WORKER MODE ──────────────────────────────────────────────────────────────
# When invoked with `--worker`, this process registers the two real DiffMem
# workflows with DUMMY task handlers (sleep + return), then blocks in
# worker.start(). It exits when killed by the parent.

def run_worker() -> None:
    log.info("WORKER_BOOT: starting dummy worker for smoke test")
    from hatchet_sdk import Context  # noqa: F401  (kept for future handler params)
    from diffmem.executor.hatchet_workflows import (
        get_input_models,
        build_hatchet_client,
        register_workflows,
    )

    # MUST live at module scope of run_worker for get_type_hints() to resolve
    # them — Hatchet introspects the handler signature at decoration time.
    global WriteInput, ConsolidateInput
    WriteInput, ConsolidateInput = get_input_models()
    hatchet = build_hatchet_client()
    write_wf, consolidate_wf = register_workflows(hatchet)

    # Use production handler names so HatchetExecutor's unwrap (which is keyed
    # on these names) sees the same shape as in production.
    @write_wf.task(execution_timeout=timedelta(minutes=2), retries=0)
    def process_and_commit(input: WriteInput, ctx: Context) -> dict:
        picked = time.monotonic()
        log.info(
            "WORKER_PICKED_UP_WRITE: user_id=%s session_id=%s run_id=%s",
            input.user_id, input.session_id, ctx.workflow_run_id,
        )
        time.sleep(0.5)  # simulated work; enough to observe queueing for same user
        done = time.monotonic()
        log.info(
            "WORKER_DONE_WRITE: user_id=%s session_id=%s run_id=%s",
            input.user_id, input.session_id, ctx.workflow_run_id,
        )
        return {
            "ok": True,
            "kind": "write",
            "user_id": input.user_id,
            "session_id": input.session_id,
            "run_id": ctx.workflow_run_id,
            "picked_up_at": picked,
            "done_at": done,
        }

    @consolidate_wf.task(execution_timeout=timedelta(minutes=2), retries=0)
    def consolidate(input: ConsolidateInput, ctx: Context) -> dict:
        log.info("WORKER_PICKED_UP_CONSOLIDATE: user_id=%s run_id=%s", input.user_id, ctx.workflow_run_id)
        time.sleep(0.3)
        return {"ok": True, "kind": "consolidate", "user_id": input.user_id, "run_id": ctx.workflow_run_id}

    slots = int(os.getenv("HATCHET_WORKER_SLOTS", "10"))
    worker = hatchet.worker("diffmem-smoke-worker", slots=slots, workflows=[write_wf, consolidate_wf])
    log.info("WORKER_REGISTERED: slots=%d workflows=[diffmem-write, diffmem-consolidate]", slots)
    log.info("WORKER_STARTING")
    worker.start()  # blocking


# ─── SUBMITTER MODE ───────────────────────────────────────────────────────────

def run_submitter() -> int:
    """Returns 0 on full pass, non-zero on any failure."""
    log.info("SUBMITTER_BOOT")

    from diffmem.executor import build_executor, WritePayload, ConsolidatePayload  # type: ignore

    pool = ThreadPoolExecutor(max_workers=4)
    executor = build_executor(pool)
    log.info("EXECUTOR_BUILT: type=%s", type(executor).__name__)

    # ─── Test 1: per-user serialization for "alice" ────────────────────────
    log.info("TEST_1_START: per-user serialization (2 alice jobs simultaneously)")
    alice_payload_1 = WritePayload(
        user_id="alice",
        memory_input="smoke test alice 1",
        session_id="smoke-alice-1",
    )
    alice_payload_2 = WritePayload(
        user_id="alice",
        memory_input="smoke test alice 2",
        session_id="smoke-alice-2",
    )
    bob_payload = WritePayload(
        user_id="bob",
        memory_input="smoke test bob 1",
        session_id="smoke-bob-1",
    )

    # Submit all three nearly simultaneously
    submit_t = time.monotonic()
    h_alice_1 = executor.submit_write(user_id="alice", work=None, payload=alice_payload_1)
    h_alice_2 = executor.submit_write(user_id="alice", work=None, payload=alice_payload_2)
    h_bob = executor.submit_write(user_id="bob", work=None, payload=bob_payload)
    log.info(
        "SUBMITTED: alice_1=%s alice_2=%s bob=%s (elapsed=%.3fs)",
        h_alice_1.job_id[:8], h_alice_2.job_id[:8], h_bob.job_id[:8],
        time.monotonic() - submit_t,
    )

    # Wait for all to complete
    log.info("WAITING_FOR_TERMINAL: timeout=60s each")
    try:
        r_alice_1 = executor.wait_for(h_alice_1.job_id, timeout=60.0)
        r_alice_2 = executor.wait_for(h_alice_2.job_id, timeout=60.0)
        r_bob = executor.wait_for(h_bob.job_id, timeout=60.0)
    except TimeoutError as e:
        log.error("FAIL: timeout waiting for jobs: %s", e)
        return 1

    log.info(
        "COMPLETED: alice_1.status=%s alice_2.status=%s bob.status=%s",
        r_alice_1.status, r_alice_2.status, r_bob.status,
    )

    # Assertions.  HatchetExecutor unwraps Hatchet's task-name wrapper for
    # single-step workflows, so r.result is the handler's direct return value
    # (matches InlineExecutor contract).  See ED-013 in .pi/DECISIONS.md.
    failures = []
    for label, r in [("alice_1", r_alice_1), ("alice_2", r_alice_2), ("bob", r_bob)]:
        if r.status != "completed":
            failures.append(f"{label} status={r.status} error={r.error!r}")
            continue
        inner = r.result or {}
        if not inner.get("ok"):
            failures.append(f"{label} result missing ok=True: {r.result!r}")

    if failures:
        for f in failures:
            log.error("FAIL: %s", f)
        return 1

    # The load-bearing assertions: alice_1 and alice_2 must NOT overlap; bob
    # must overlap with alice_1.
    #
    # FINDING: HatchetExecutor.JobResult.started_at is None (no signal from
    # submit-side process about worker pickup time).  We assert using worker-
    # side monotonic timestamps captured by the handler instead.
    a1 = r_alice_1.result or {}
    a2 = r_alice_2.result or {}
    b  = r_bob.result or {}
    a1p = a1.get("picked_up_at"); a1d = a1.get("done_at")
    a2p = a2.get("picked_up_at"); a2d = a2.get("done_at")
    bp  = b.get("picked_up_at");  bd  = b.get("done_at")

    # Validate that HatchetExecutor populated started_at via _enrich_from_details.
    # This is the post-fix-for-Finding-1 assertion.
    if r_alice_1.started_at is None or r_bob.started_at is None:
        log.error("FAIL: HatchetExecutor did not populate started_at (Finding 1 regression)")
        log.error("  alice_1.started_at=%s bob.started_at=%s", r_alice_1.started_at, r_bob.started_at)
        return 1
    log.info("PASS_STARTED_AT_POPULATED: alice_1.started_at=%s bob.started_at=%s",
             r_alice_1.started_at, r_bob.started_at)

    log.info("TIMING_WORKER: alice_1 picked=%.3f done=%.3f", a1p or 0, a1d or 0)
    log.info("TIMING_WORKER: alice_2 picked=%.3f done=%.3f", a2p or 0, a2d or 0)
    log.info("TIMING_WORKER: bob     picked=%.3f done=%.3f", bp or 0, bd or 0)

    if None in (a1p, a1d, a2p, a2d, bp, bd):
        log.error("FAIL: worker-side timestamps not populated")
        return 1

    if a2p < a1d:
        log.error("FAIL: alice_2 picked at %.3f BEFORE alice_1 done at %.3f - concurrency key not enforcing", a2p, a1d)
        return 1
    log.info("PASS_TEST_1_SERIALIZATION: alice_2 picked %.3f >= alice_1 done %.3f (gap=%.3fs)", a2p, a1d, a2p - a1d)

    if bp >= a1d:
        log.error("FAIL: bob picked at %.3f AFTER alice_1 done at %.3f - cross-user serialization bug", bp, a1d)
        return 1
    log.info("PASS_TEST_2_PARALLEL_USERS: bob picked %.3f < alice_1 done %.3f (overlap=%.3fs)", bp, a1d, a1d - bp)

    # ─── Test 3: get_job round-trips ───────────────────────────────────────
    fetched = executor.get_job(h_alice_1.job_id)
    if fetched is None or fetched.job_id != h_alice_1.job_id:
        log.error("FAIL: get_job returned unexpected: %s", fetched)
        return 1
    log.info("PASS_TEST_3_GET_JOB: round-trip ok")

    # ─── Test 4: consolidate workflow ──────────────────────────────────────
    log.info("TEST_4_START: consolidate workflow")
    c_payload = ConsolidatePayload(
        user_id="alice",
        tools=["dedupe"],
        window=3,
        soft_cap_tokens=32000,
    )
    h_c = executor.submit_consolidate(user_id="alice", work=None, payload=c_payload)
    try:
        r_c = executor.wait_for(h_c.job_id, timeout=30.0)
    except TimeoutError:
        log.error("FAIL: consolidate timeout")
        return 1
    c_inner = r_c.result or {}
    if r_c.status != "completed" or not c_inner.get("ok"):
        log.error("FAIL: consolidate status=%s result=%s", r_c.status, r_c.result)
        return 1
    log.info("PASS_TEST_4_CONSOLIDATE: status=%s", r_c.status)

    log.info("ALL_TESTS_PASS")
    return 0


# ─── ENTRYPOINT ──────────────────────────────────────────────────────────────

def main() -> int:
    if "--worker" in sys.argv:
        run_worker()
        return 0

    # Parent: spawn worker subprocess, give it a few seconds to register, then submit.
    log.info("PARENT: spawning worker subprocess")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_SRC) + ":" + env.get("PYTHONPATH", "")
    worker_proc = subprocess.Popen(
        [sys.executable, __file__, "--worker"],
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    try:
        # Give the worker time to register with the engine
        log.info("PARENT: waiting 8s for worker to register with Hatchet Cloud")
        time.sleep(8)

        if worker_proc.poll() is not None:
            log.error("FAIL: worker subprocess exited before submitter started (rc=%d)", worker_proc.returncode)
            return 1

        rc = run_submitter()
    finally:
        log.info("PARENT: terminating worker subprocess")
        worker_proc.terminate()
        try:
            worker_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log.warning("PARENT: worker did not exit cleanly, killing")
            worker_proc.kill()

    return rc


if __name__ == "__main__":
    sys.exit(main())
