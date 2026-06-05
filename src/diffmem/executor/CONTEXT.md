# executor CONTEXT

## BUSINESS PURPOSE
Pluggable task-execution abstraction that decouples *what* gets run (writer-agent
thunk, consolidator thunk) from *how* it is scheduled. The default backend
(`inline`) reuses the existing `_writer_pool` with zero extra infrastructure,
giving self-hosters the same behaviour as before M1. A future Hatchet backend
(M3) will add durable queues, retries, and observability for multi-user
production deployments.

## USER STORIES
- As a **self-hoster**, I run DiffMem with no environment changes and get
  inline execution backed by `ThreadPoolExecutor(max_workers=4)` — identical
  to pre-M1 behaviour.
- As a **production operator** running many users, I set `EXECUTOR=hatchet`
  to route all jobs through Hatchet for durability, retry, and
  dead-letter-queue observability (available M3+).
- As a **developer** adding a new backend, I implement `TaskExecutor`, add an
  elif branch to `factory.py`, and the rest of the system picks it up.

## INFORMATION FLOW
```
HTTP endpoint
  → executor.submit_write(user_id, thunk, callback_url?)
      → returns JobHandle(job_id, status="queued", submitted_at)
  → endpoint either:
      a) calls executor.wait_for(job_id) → blocks until terminal → returns result (sync mode, M2)
      b) returns job_id immediately for async polling via GET /jobs/{job_id} (M2)

InlineExecutor internals:
  ThreadPoolExecutor
    → _run_job(job_id, user_id, thunk, callback_url)
        → acquire per-user lock         # serialises same-user writes
        → update_status("running")
        → thunk()                       # blocking LLM + git work
        → set_result / set_error
        → update_status("completed" | "failed")
        → release per-user lock
        → _fire_callback(callback_url)  # best-effort POST
```

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
| `base.py` | `JobStatus`, `JobHandle`, `JobResult`, `TaskExecutor` ABC — the full public contract. |
| `jobstore.py` | `JobStore`: thread-safe OrderedDict store with FIFO eviction and INFO-level eviction log. |
| `inline.py` | `InlineExecutor`: default backend; wraps existing `_writer_pool`; per-user locks. |
| `factory.py` | `build_executor(pool)`: reads `EXECUTOR` env var, constructs and returns the right backend. |
| `__init__.py` | Re-exports public surface: `TaskExecutor`, `JobHandle`, `JobResult`, `JobStatus`, `build_executor`. |
| `CONTEXT.md` | This file — capability-level documentation. |

## CONSTRAINTS
- **M1**: executor is constructed in `server.py` lifespan (`app.state.executor =
  build_executor(_writer_pool)`) but **NOT wired into any endpoint**. Endpoints
  continue to call `_writer_pool.run_in_executor` directly. No external
  behaviour change.
- **M2**: endpoints are wired through `app.state.executor.submit_write()` /
  `submit_consolidate()`; `?sync=true` param + `GET /jobs/{job_id}` added.
- **M3**: `HatchetExecutor` added behind optional `[hatchet]` extra; factory
  updated.
- **Reads stay direct**: read endpoints (context retrieval, user-entity, timeline)
  bypass the executor and continue using `_writer_pool.run_in_executor` directly
  even after M2 — the per-user lock is a write concern only.
- **No new dependencies in M1**: `httpx` is already a transitive dep; `requests`
  is a direct dep fallback.
- **JobStore is in-process only**: entries are lost on restart. This is acceptable
  for the inline backend; the Hatchet backend will use durable storage.

## ATTENTION GUIDANCE
| When you want to… | Look at… |
|---|---|
| Change how endpoints invoke the executor | `server.py` + M2 spec |
| Change per-user serialisation semantics (lock granularity, queue ordering) | `inline.py` |
| Add a new executor backend | `factory.py` (add elif) + new `<backend>.py` module |
| Change job lifecycle types / to_dict() serialisation | `base.py` |
| Change eviction policy or job store implementation | `jobstore.py` |
| Understand the full capability contract | `base.py` (TaskExecutor ABC) |
