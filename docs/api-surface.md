# External API Surface

DiffMem talks to a small, fixed set of external services. This document enumerates them,
the endpoints used, auth conventions, and which capability owns the integration.

---

## OpenRouter (LLM)

**Purpose:** all LLM calls — writer agent, retrieval agent, consolidator agent, onboarding.
**Required:** yes (no LLM-free mode).
**Capability owners:** `writer_agent/`, `retrieval_agent/`, `consolidator_agent/`.

| Property | Value |
|---|---|
| Base URL | `https://openrouter.ai/api/v1` |
| Auth | `Authorization: Bearer ${OPENROUTER_API_KEY}` |
| Endpoint | `POST /chat/completions` (OpenAI-compatible) |
| Client | `openai` Python SDK pointed at OpenRouter base URL |

**Env vars:**
- `OPENROUTER_API_KEY` (required)
- `DEFAULT_MODEL` (required; e.g. `openai/gpt-4o-mini`, `anthropic/claude-3.5-sonnet`)
- `RETRIEVAL_MODEL` (optional; overrides `DEFAULT_MODEL` for retrieval-agent only)

**Failure mode:** OpenRouter errors propagate as exceptions inside the writer/retrieval/
consolidator agents. The retrieval agent has a baseline fallback (returns user-entity +
timeline) if the agentic loop fails.

**Cost shape:** writes are 60–600s of LLM calls (multiple turns, parallel entity updates).
Consolidates are 30–180s. Reads are 5–30s. Per-job spend at GPT-4o-mini scale:
$0.005–$0.05/write, $0.01–$0.10/consolidate.

---

## GitHub (optional backup backend)

**Purpose:** mirror per-user branches to a private GitHub repo for offsite durability.
**Required:** no (default `BACKUP_BACKEND=none` skips it entirely).
**Capability owner:** `storage/github_backup.py`.

| Property | Value |
|---|---|
| Auth | HTTP Basic via PAT, passed to `git` via `GIT_ASKPASS` at call time |
| Operations | `git push` (per-commit + periodic), `git fetch` (at worktree mount time) |
| Library | `gitpython` (shells out to `git` binary) |

**Env vars (only when `BACKUP_BACKEND=github`):**
- `GITHUB_REPO_URL` (e.g. `https://github.com/yourname/diffmem-backup`)
- `GITHUB_TOKEN` (PAT with `repo` scope)

**Failure mode:**
- Push failures are logged at WARNING and never block the request hot path (background task).
  The periodic backup scheduler catches up on the next tick (default 30 min).
- Pull failures at worktree mount time are non-fatal — service proceeds with local state.

**Security:** token is passed to git via `GIT_ASKPASS` at call time, never written into
`.git/config` on the persistent volume.

---

## Hatchet (optional executor backend)

**Purpose:** durable task orchestration + per-user concurrency enforcement when running
with `EXECUTOR=hatchet`.
**Required:** no (default `EXECUTOR=inline` skips it entirely; SDK is an optional Poetry
extras group `[hatchet]`).
**Capability owner:** `executor/hatchet.py` (submit-side) + `executor/hatchet_worker.py`
(execute-side).

| Property | Value |
|---|---|
| Default endpoint | `cloud.onhatchet.run` (Hatchet Cloud) |
| Auth | JWT in `HATCHET_CLIENT_TOKEN` env var |
| Protocols | gRPC (long-poll for worker → engine) + REST (status queries from submit-side) |
| Library | `hatchet-sdk` Python SDK (lazy-imported; only loaded when `EXECUTOR=hatchet`) |

**Endpoints used:**
- gRPC: `workflow.run(input, wait_for_result=False)` — enqueue a new workflow run.
- gRPC: workers long-poll for assigned runs and stream results back.
- REST: `hatchet.runs.get_status(run_id)` — cheap status poll (used by `get_job`).
- REST: `hatchet.runs.get(run_id)` — full run details with `started_at`, `finished_at`,
  `output` (used by `_enrich_from_details` on terminal transition).

**Env vars (only when `EXECUTOR=hatchet`):**
- `HATCHET_CLIENT_TOKEN` (required; JWT from Hatchet Cloud dashboard or self-hosted admin UI)
- `HATCHET_NAMESPACE` (optional, default `diffmem`)
- `HATCHET_CLIENT_HOST_PORT` (optional; for self-hosted Hatchet Lite, e.g.
  `engine.example.com:7077`)
- `HATCHET_CLIENT_TLS_STRATEGY` (optional; `tls` | `mtls` | `none`; default `tls` for Cloud)
- `HATCHET_WORKER_SLOTS` (optional; worker-side; default 10 concurrent jobs per worker)

**Failure mode:**
- Connection errors during status queries (`get_job`): logged at WARNING, return cached
  state.
- Errors during `_enrich_from_details`: logged at WARNING, swallowed — job still returns
  with whatever state was already known (just no `started_at` populated).
- Worker process crash: Hatchet redelivers the run to another available worker (no
  duplicate execution because the writer agent is idempotent on `session_id`).

**Workflow names (cross-process contract):**
- `diffmem-write` — handler `process_and_commit` in `hatchet_worker.py`
- `diffmem-consolidate` — handler `consolidate` in `hatchet_worker.py`
- Per-user concurrency on both: `ConcurrencyExpression(expression="input.user_id", max_runs=1)`
- See `_TASK_NAMES` in `executor/hatchet.py` for the submit-side map (must stay in sync).

**Cost:** Hatchet Cloud free tier covers 1000 runs/month; Annabelle's ~6000 runs/month
require the Hobby plan (~$25/mo at time of writing).

---

## Inbound APIs (this service exposes)

For completeness, DiffMem itself exposes a FastAPI surface. See README.md and the
Swagger UI at `/docs` for the full schema. The endpoints that route through the executor
abstraction (`POST /memory/{id}/process-session`, `commit-session`, `process-and-commit`,
`consolidate`, `process-commit-and-consolidate`) accept `?sync=true|false` and an
optional `callback_url`. `GET /memory/{id}/jobs/{job_id}` polls async-mode jobs.
