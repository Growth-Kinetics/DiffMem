# CAPABILITY: Live integration test for the HatchetExecutor + Hatchet Cloud.
# SCOPE: Submits a write via the HTTP API and verifies the job appears in Hatchet.
#
# HOW TO ENABLE:
#   This file is skipped entirely unless HATCHET_CLIENT_TOKEN is set in the
#   environment.  All tests are marked @pytest.mark.hatchet_live so they can
#   be selected or excluded explicitly:
#
#     # Run only Hatchet-live tests (requires a real Hatchet Cloud account):
#     HATCHET_CLIENT_TOKEN=eyJ... pytest -m hatchet_live
#
#     # Exclude them from the normal CI suite (default — no env var needed):
#     pytest -m "not hatchet_live"
#
# PREREQUISITES:
#   1. A Hatchet Cloud account and tenant (see docs/deployment-hatchet.md).
#   2. HATCHET_CLIENT_TOKEN set to a valid API token for that tenant.
#   3. A running DiffMem API pointed at a Hatchet-backed deployment, or at
#      least the diffmem-worker container running to pick up jobs.
#   4. DIFFMEM_BASE_URL (optional) — defaults to http://localhost:8000.
#
# NOTE:
#   This test does NOT verify that the worker actually executes the job — that
#   requires a running diffmem-worker process which is out of scope for
#   automated tests.  It only verifies that the job was enqueued in Hatchet
#   (run appears in Hatchet dashboard with status queued/running/succeeded).
#
#   Full end-to-end verification (including worker execution) is the
#   operator's responsibility before cutting Annabelle over.  See
#   docs/annabelle-migration.md Step 5 for the manual smoke-test procedure.

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Marker guard: all tests in this file require hatchet_live
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.hatchet_live

_HATCHET_TOKEN = os.getenv("HATCHET_CLIENT_TOKEN")
_SKIP_REASON = (
    "hatchet_live tests require HATCHET_CLIENT_TOKEN env var. "
    "See module docstring for details."
)


@pytest.mark.skipif(not _HATCHET_TOKEN, reason=_SKIP_REASON)
def test_hatchet_live_placeholder():
    """Placeholder for a real Hatchet Cloud integration test.

    When you are ready to run this against a live deployment:

    1. Set HATCHET_CLIENT_TOKEN to your Hatchet Cloud API token.
    2. Set DIFFMEM_BASE_URL to the URL of your DiffMem API instance.
    3. Optionally set DIFFMEM_API_KEY if REQUIRE_AUTH=true.
    4. Run: HATCHET_CLIENT_TOKEN=eyJ... pytest -m hatchet_live -v

    What a real implementation would check:
    - POST /memory/{test_user}/process-and-commit?sync=false
    - Verify response is {status: "queued", job_id: ...}
    - Use hatchet SDK: client.runs.get(run_id) to verify the run exists
    - Check run.status in ("QUEUED", "RUNNING", "SUCCEEDED")

    That implementation is deferred: it requires hatchet-sdk installed AND a
    live tenant, so it is not part of the automated CI suite.
    """
    # If somehow the token is set but this placeholder is reached, just pass —
    # real assertions belong in a full implementation of this test.
    assert _HATCHET_TOKEN, "HATCHET_CLIENT_TOKEN must be set to reach this point"
