# executor CONTEXT

## BUSINESS PURPOSE
Pluggable task-execution abstraction that decouples *what* gets run (writer-agent
or consolidator work) from *how* it is scheduled. Two backends ship today:

- **`inline`** (default): reuses the existing `_writer_pool` with per-user
  `threading.Lock` serialization. Zero external infrastructure. Identical UX
  to the pre-executor DiffMem for single-user / local-host use.
- **`hatchet`**: enqueues runs into a Hatchet engine (Cloud or self-hosted
  Hatchet Lite). Workers consume runs in a separate process. Per-user
  serialization via `ConcurrencyExpression(expression="input.user_id")`.
  Durable runs, retries, dashboard, cron, observability. Opt-in via
  `EXECUTOR=hatchet` env var.

## USER STORIES
- As a **self-hoster**, I run DiffMem with no environment changes and get
  inline execution backed by `ThreadPoolExecutor(max_workers=4)` — identical
  to pre-executor behaviour.
- As a **production operator** running many users, I set `EXECUTOR=hatchet`
  to route all jobs through Hatchet for durability, retry, and
  observability.
- As a **developer** adding a new backend, I implement `TaskExecutor`, add an
  elif branch to `factory.py`, and the rest of the system picks it up.

## INFORMATION FLOW

### Inline mode (default)
```
HTTP endpoint (API process)
  → executor.submit_write(user_id, thunk, payload, callback_url?)
      → InlineExecutor pushes a wrapper into _writer_pool
      → returns JobHandle(job_id, status="queued", submitted_at)
  → wrapper inside the ThreadPoolExecutor:
      → acquire per-user threading.Lock   # serialises same-user writes
      → update_status("running")
      → thunk()                            # blocking LLM + git work
      → set_result / set_error
      → update_status("completed" | "failed")
      → release per-user lock
      → _fire_callback(callback_url)        # best-effort POST
  → endpoint either waits (sync=True default) or returns job_id (sync=False).
```

### Hatchet mode (`EXECUTOR=hatchet`)
```
API process                              Hatchet engine          Worker process (diffmem-worker)
────────────────────────────────────   ───────────────   ───────────────────────────────
HatchetExecutor.submit_write(...)         (Postgres,
  → write_workflow.run(WriteInput,         queue, runs)
        wait_for_result=False)     ──gRPC─→
  ←── WorkflowRunRef(run_id) ──────────────
  → JobHandle(run_id, queued)

endpoint either polls GET /jobs/{id}     ←──gRPC── worker pulls run
  → hatchet.runs.get_status(run_id)                       → _get_memory(user_id)
                                                            → DiffMemory.process_and_commit_session()
                                                            → returns result dict
                                            ──gRPC─→    ← (engine records terminal status)
```

The `WritePayload` / `ConsolidatePayload` dataclass is what crosses the
API/worker boundary (JSON-serialised by Hatchet over gRPC). Closures cannot
cross processes, so the worker reconstitutes the work from the payload.

## TERMINOLOGY
- **JobHandle** — lightweight receipt (job_id, status, submitted_at) returned
  immediately by `submit_*()`. Lets endpoints return without blocking.
- **JobResult** — full job record: JobHandle fields + result dict, error string,
  started_at, completed_at. Stored in JobStore, serialised via `to_dict()`.
- **JobStatus** — one of `queued | running | completed | failed`.
- **Per-user lock** — a `threading.Lock` keyed by `user_id`. Guarantees that
  two writes for the same user are never executed concurrently (same guarantee
  as the implicit FIFO ordering in the old single-pool approach).
- **JobStore** — in-process `OrderedDict`-backed store with FIFO eviction at
  1 000 entries. Entries are never persisted to disk (InlineExecutor is
  in-process only).
- **Thunk** — a `Callable[[], dict]` that closes over the DiffMemory instance
  and call parameters. The endpoint constructs the thunk; the executor calls it.

## KEY FILES
| File | Description |
|---|---|
| `base.py` | `JobStatus`, `JobHandle`, `JobResult`, `WritePayload`, `ConsolidatePayload`, `TaskExecutor` ABC — the full public contract. |
| `jobstore.py` | `JobStore`: thread-safe OrderedDict store with FIFO eviction and INFO-level eviction log. |
| `inline.py` | `InlineExecutor`: default backend; wraps `_writer_pool`; per-user `threading.Lock`; `supports_async_api=False`. |
| `hatchet.py` | `HatchetExecutor`: submit-side; calls `register_workflows()` on init; submits runs; polls status. |
| `hatchet_workflows.py` | `WriteInput`/`ConsolidateInput` Pydantic models; `build_hatchet_client()`; `register_workflows()`. Shared by API and worker. |
| `hatchet_worker.py` | Worker process: attaches `@workflow.task()` handlers; calls `worker.start()`. Entry point for `diffmem-worker` console script. |
| `factory.py` | `build_executor(pool)`: reads `EXECUTOR` env var, constructs and returns the correct backend. |
| `__init__.py` | Re-exports public surface: `TaskExecutor`, `JobHandle`, `JobResult`, `JobStatus`, `WritePayload`, `ConsolidatePayload`, `build_executor`. |
| `CONTEXT.md` | This file — capability-level documentation. |

## CONSTRAINTS
- **Executor is constructed in `server.py` lifespan** (`app.state.executor =
  build_executor(_writer_pool)`). The 5 write/consolidate endpoints call
  `app.state.executor.submit_*()`; `?sync=true|false` query param overrides the
  default response mode; `GET /memory/{id}/jobs/{job_id}` polls status.
- **Hatchet backend** (`EXECUTOR=hatchet`): `HatchetExecutor` lives behind the
  optional `[hatchet]` Poetry extra; factory imports it lazily. Deploy with
  `deploy/docker-compose.hatchet.yml` — see `docs/deployment-hatchet.md`.
- **Reads stay direct**: read endpoints (context retrieval, user-entity, timeline)
  bypass the executor and use `_writer_pool.run_in_executor` directly — the
  per-user lock is a write/consolidate concern only.
- **InlineExecutor's JobStore is in-process only**: entries are lost on restart.
  Acceptable for single-user / local-host use; the Hatchet backend's job state
  lives in the Hatchet engine's Postgres and survives worker/API restarts.
- **`supports_async_api`** drives default endpoint behavior:
  - `InlineExecutor.supports_async_api = False` → endpoints default to sync
    (block-until-done), preserving the pre-executor API contract.
  - `HatchetExecutor.supports_async_api = True` → endpoints default to async
    (return job_id immediately). Callers pass `?sync=true` for the legacy
    block-until-done shape (used during Annabelle migration window).

## ATTENTION GUIDANCE
| When you want to… | Look at… |
|---|---|
| Change how endpoints invoke the executor | `server.py` + M2 spec |
| Change per-user serialisation semantics (lock granularity, queue ordering) | `inline.py` |
| Add a new executor backend | `factory.py` (add elif) + new `<backend>.py` module |
| Change job lifecycle types / to_dict() serialisation | `base.py` |
| Change eviction policy or job store implementation | `jobstore.py` |
| Understand the full capability contract | `base.py` (TaskExecutor ABC) |

## Hatchet Backend

### Required env vars
| Variable | Required | Default | Notes |
|---|---|---|---|
| `HATCHET_CLIENT_TOKEN` | Yes | — | Copy from Hatchet Cloud dashboard or self-hosted admin UI |
| `HATCHET_NAMESPACE` | No | `diffmem` | Namespace scoping |
| `HATCHET_CLIENT_HOST_PORT` | No | — | For self-hosted Hatchet (e.g. `engine.example.com:7077`) |
| `HATCHET_CLIENT_TLS_STRATEGY` | No | `tls` | `tls` \| `mtls` \| `none` |

### WritePayload / ConsolidatePayload contract
The `work` thunk model used by InlineExecutor does not translate to Hatchet —
worker processes have no access to Python closures. Instead, endpoints build
a **structured payload** (`WritePayload` or `ConsolidatePayload`) alongside the
thunk and pass both to `submit_*()`. The executor's choice which to use:
- **InlineExecutor**: uses `work`, ignores `payload`.
- **HatchetExecutor**: uses `payload`, ignores `work`.

The payload dataclasses live in `base.py` and are exported from `__init__.py`:
```python
from diffmem.executor import WritePayload, ConsolidatePayload
```

### Workflow registration split
| File | Responsibility |
|---|---|
| `hatchet_workflows.py` | Defines `WriteInput` / `ConsolidateInput` Pydantic models, `build_hatchet_client()`, `register_workflows()`. Importable by both API and worker. NO task handlers. |
| `hatchet.py` | `HatchetExecutor`: submit-side. Calls `register_workflows()` on init, submits runs, queries status, waits for results. |
| `hatchet_worker.py` | Worker process: attaches `@workflow.task()` handlers and calls `worker.start()`. |

Per-user serialisation is handled by Hatchet's `ConcurrencyExpression(expression="input.user_id", max_runs=1)` — the second run for the same user queues until the first completes, even across worker restarts.

### JobResult shape symmetry (ED-013)

Hatchet returns workflow outputs as `{<task_name>: <handler_return>}` because workflows can be multi-step. DiffMem's workflows are single-step, so `HatchetExecutor` strips the wrapper via `_unwrap_task_output()` to match `InlineExecutor`'s contract (the caller sees the handler's direct return value, no wrapper layer).

`HatchetExecutor.JobResult.started_at` is populated lazily via `_enrich_from_details()`, which calls `hatchet.runs.get(run_id)` on transition to a terminal state. Only called once per job (cheap REST call); errors are logged at WARNING and swallowed (best-effort).

**Cross-process contract:** the `_TASK_NAMES` dict in `hatchet.py` must stay in sync with the `@workflow.task()`-decorated function names in `hatchet_worker.py`. Currently:
- `diffmem-write` → `process_and_commit`
- `diffmem-consolidate` → `consolidate`

Drift here silently disables the unwrap (the WARNING surfaces it, the executor still works — callers just see the wrapper layer).

**REST vs ref.result() asymmetry:** Hatchet's `ref.result()` returns wrapped output; `hatchet.runs.get().run.output` returns *already unwrapped* output. `_unwrap_task_output()` accepts an `expect_wrapped` kwarg — `True` (default, used by `wait_for`) warns on shape mismatch; `False` (used by `_enrich_from_details`) is silent. See ED-013.

---

## Worker process

### API / worker split

```
API process (diffmem-server)          Worker process (diffmem-worker)
──────────────────────────────        ────────────────────────────────────
HatchetExecutor.submit_*()            hatchet_worker.py:
  → workflow.run(input, ...)            register_workflows(hatchet)
  → returns job_id immediately          _attach_write_handler(write_wf)
                                        _attach_consolidate_handler(cons_wf)
HatchetExecutor.get_job()               worker.start()  ← blocks forever
  → hatchet.runs.get_status(job_id)       → pulls runs from Hatchet engine
                                          → reconstitutes DiffMemory
                                          → runs git + LLM work
                                          → returns result dict
```

Both processes share the same `/data` volume.  They each hold their own
`RepoManager` + `DiffMemory` cache in process memory — no cross-process
coordination needed because DiffMemory holds no mutable shared state.

Per-user serialisation is enforced by Hatchet's
`ConcurrencyExpression(expression="input.user_id", max_runs=1)` — a second job
for the same user queues until the first completes, even across worker restarts.

### Entrypoint

```bash
# pip-installed package
diffmem-worker

# development (no install needed)
python -m diffmem.executor.hatchet_worker
```

### Worker env vars

The worker requires the same env vars as the API process:

| Variable | Required | Notes |
|---|---|---|
| `HATCHET_CLIENT_TOKEN` | Yes | Same token as the API process |
| `OPENROUTER_API_KEY` | Yes | LLM API key |
| `DEFAULT_MODEL` | Yes | Model name (or set in DiffMemory init) |
| `STORAGE_PATH` | Yes | Base path for user worktrees (read by RepoManager) |
| `WORKTREE_ROOT` | No | Worktree root override (read by RepoManager) |
| `HATCHET_WORKER_SLOTS` | No | Max concurrent jobs on this worker (default `10`) |
| `HATCHET_NAMESPACE` | No | Namespace scoping |
| `HATCHET_CLIENT_HOST_PORT` | No | For self-hosted Hatchet |
| `HATCHET_CLIENT_TLS_STRATEGY` | No | `tls` \| `mtls` \| `none` |

### Per-worker memoisation

`hatchet_worker.py` keeps a module-level `_repo_manager_singleton` and a
`_memory_cache: dict[str, DiffMemory]` (protected by a `threading.Lock`).
This mirrors the `memory_instances` dict in `server.py`: workers run for
hours and we avoid reconstructing `RepoManager` / `DiffMemory` on every job.

### Live validation notes

The live smoke test at `scripts/smoke_hatchet_live.py` validates the per-user concurrency key against a real Hatchet engine. Run it before any production deployment:

```bash
source .env.hatchet-test  # or whatever env file has HATCHET_CLIENT_TOKEN
EXECUTOR=hatchet PYTHONPATH=src python3 scripts/smoke_hatchet_live.py
```

The script spawns its own worker subprocess + submits 3 jobs (2 same-user, 1 different-user), asserts:
- `alice_2` picked up AFTER `alice_1` completed (serialization)
- `bob` picked up DURING `alice_1` (cross-user parallelism)
- `JobResult.started_at` populated (ED-013)
- `JobResult.result` is unwrapped (ED-013)

Expected runtime: ~10 seconds. No LLM cost (dummy handlers).

A full end-to-end test against real DiffMemory + LLM + git was run during PR #14 development against the tommy_demo transcripts; see TIMELINE 2026-06-05.

### Task handler notes

- Handlers are **sync** (not async) — matches the blocking git+LLM workload.
- `retries=0` on both tasks — LLM output is non-deterministic; Hatchet's
  default retry could produce duplicate commits. Failures surface immediately
  to the caller via the polling API. See ADR ED-011 in `.pi/DECISIONS.md`.
- Callbacks (`input.callback_url`) are best-effort: POST on success only,
  never on failure, swallowed if the POST itself fails.

### Deployment

See `deploy/docker-compose.hatchet.yml` for the production compose template
(API + worker as separate services from the same image) and
`docs/deployment-hatchet.md` for the full deployment guide.
