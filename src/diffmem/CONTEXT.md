# DiffMem Service

## Purpose
FastAPI service providing git-native persistent memory for AI agents. Each user
gets an isolated orphan branch (`user/{id}`) in a single bare git repo, checked
out into a per-user worktree on demand. Memory files are Markdown; history lives
in the git commit graph. No vector DB, no embeddings.

## User Stories
- As an AI agent, I POST a session transcript and get memory written to git atomically.
- As an AI agent, I POST a conversation and get targeted context back from git history.
- As a self-hoster, I deploy with Docker Compose and point it at a volume — zero external
  dependencies unless I opt into the GitHub backup backend.

## Information Flow
- **Inputs:** Session transcripts (`process-and-commit`), conversation history (`context`),
  raw user info (`onboard`).
- **Outputs:** Context blobs (user entity + timeline + agent-retrieved blocks), commit hashes.
- **Internal:** `server.py` → `api.py` (DiffMemory) → `writer_agent/` or `retrieval_agent/`
  → git worktree on `/data/worktrees/{user_id}`.
- **Write turn (detailed):** `API endpoint → app.state.executor.submit_*()` →
  `InlineExecutor._run_job()` in `_writer_pool` (default) OR
  `HatchetExecutor.submit_write()` → Hatchet engine queue → `diffmem-worker` process →
  `DiffMemory.process_and_commit_session()` → git worktree.

## Terminology
- **Worktree:** Per-user git working directory. Mounted lazily on first request.
- **Storage repo:** The bare central repo at `/data/storage`. All user branches live here.
- **Backup backend:** Optional out-of-band mirror (GitHub). Push is out-of-band; pull happens
  at mount time (first request after restart) to pick up remote edits.
- **Writer pool:** `ThreadPoolExecutor(max_workers=4)` in `server.py` that runs all blocking
  operations off the uvicorn event loop — writes, reads, and remote pulls.
- **Executor:** Pluggable task-scheduling abstraction (`executor/` package). Chosen at startup
  via the `EXECUTOR` env var. Decouples *what* gets run from *how* it is scheduled.
- **InlineExecutor:** Default executor backend. Runs jobs directly in `_writer_pool` with a
  `threading.Lock` per user for serialization. Zero extra infrastructure.
- **HatchetExecutor:** Optional executor backend (`EXECUTOR=hatchet`). Submits jobs to the
  Hatchet workflow engine for durable, observable, per-user-serialized execution.
- **JobHandle:** Lightweight receipt returned immediately by `executor.submit_*()` — contains
  `job_id`, `status`, and `submitted_at`. Lets endpoints return without blocking.
- **JobResult:** Full job record: `JobHandle` fields plus `result` dict, `error` string,
  `started_at`, `completed_at`. Stored in `JobStore` (inline) or Hatchet (hatchet mode).
- **`EXECUTOR` env var:** Selects the executor backend at startup. `inline` (default) or
  `hatchet`. Read by `executor/factory.py`.
- **OntologyProfile:** Resolved once at server startup from `DIFFMEM_ONTOLOGY` env var.
  Defines entity types, folder map, contexts folder, and prompt overrides. Propagated
  to every `DiffMemory`, `WriterAgent`, `OnboardingAgent`, `ConsolidatorAgent`, and
  `run_retrieval_agent()` call. Built-ins: `personal` (default), `corporate`.
- **`DIFFMEM_ONTOLOGY` env var:** Selects the active ontology at startup. Built-in name
  (e.g. `personal`, `corporate`) or absolute path to a custom ontology directory.
  Read by `ontology/factory.py` → `load_ontology()`. Unknown name raises at startup.

## Key Files
- `server.py` — FastAPI app, all HTTP endpoints, `_writer_pool` thread pool, lifespan.
  Loads ontology at startup (`app.state.ontology`); passes it to every `DiffMemory` instance.
- `api.py` — `DiffMemory` class: public Python API, delegates to writer/retrieval/consolidator.
  Loads `OntologyProfile` once in `__init__`; propagates to all sub-agents.
- `repo_manager.py` — Worktree mount/unmount, `list_active_users()`, post-commit hook install.
- `writer_agent/agent.py` — Multi-step LLM pipeline: identifies entities → stages git changes
  → commits. Ontology-driven scanning via `_entity_md_files()`.
- `retrieval_agent/agent.py` — Multi-turn agent with sandboxed shell tool. System prompt
  folder listing rendered from ontology at call time.
- `storage/factory.py` — Pluggable storage/backup backend factory.
- `consolidator_agent/agent.py` — `ConsolidatorAgent`: out-of-band repair pass
  with four tools (`run_reabsorb`, `run_dedupe`, `run_redistribute`, `run_link`).
  `reabsorb` is a migration-only tool excluded from the default run set.
  Ontology-aware entity scanning and index rebuilding.
- `ontology/loader.py` — `load_ontology()` + `OntologyProfile` dataclass. Central
  resolution point for all ontology concerns. Built-ins in `ontologies/`.
- `frontmatter.py` — YAML frontmatter parse/merge utilities; the v2 location for
  structured entity metadata (`type`, `status`, `cues`, …). Tolerates + migrates
  the legacy trailing `## SEMANTIC INDEX` JSON block.
- `status.py` — `canonicalize_status()`: deterministic, code-owned mapping of
  freeform LLM status prose to the closed enum (fixes the Model Judgment
  Boundary violation where freeform statuses escaped the followups drop filter).
- `conformance.py` — `check_conformance()`: read-only scan flagging entity files
  whose frontmatter `type` ≠ folder `index_type` (mis-bucketed analysis work)
  or that lack frontmatter.
- `executor/factory.py` — `build_executor(pool)`: reads `EXECUTOR` env var.
- `executor/inline.py` — `InlineExecutor`: default backend; per-user `threading.Lock`.
- `executor/hatchet.py` — `HatchetExecutor`: opt-in backend; Hatchet workflow engine.
- `executor/hatchet_workflows.py` — Workflow registrations shared by API + worker.
- `executor/hatchet_worker.py` — Long-running worker process (`diffmem-worker` script).

## External Dependencies
- **OpenRouter** — all LLM calls (writer, onboarding, retrieval agents). Model configured
  via `DEFAULT_MODEL` / `RETRIEVAL_MODEL` env vars.
- **GitHub** (optional) — backup backend when `BACKUP_BACKEND=github`.

## Constraints
- **All blocking operations run in `_writer_pool`:** writes (process/commit), reads
  (`/context`), remote pulls at mount time, AND consolidation runs. Keeps the
  uvicorn event loop free for health probes at all times.
- **Consolidator commits use the `consolidate:` prefix** — specifically
  `consolidate(reabsorb):`, `consolidate(dedupe):`, `consolidate(redistribute):`,
  and `consolidate(link):`. This lets retrieval agents and human auditors
  distinguish them from session-formation commits. `reabsorb` is a one-time
  v2 migration tool (folds a legacy `entities/commitments/` corpus into owner
  `## Open Items`); it is excluded from the default `consolidate()` run set.
  See ADR-D006.
- **Pull happens at mount time only** (first request per user per process lifetime, i.e. after
  a service restart). If the service stays up for days and you push memory edits from another
  machine, DiffMem won't see them until the next restart. Pull is fast-forward only — diverged
  branches fail cleanly with a warning log.
- **Per-user write serialization is enforced server-side by the task executor.**
  In inline mode: `threading.Lock` per user in `executor/inline.py` (`_get_user_lock()`).
  In hatchet mode: `ConcurrencyExpression(expression='input.user_id', max_runs=1)` in
  `executor/hatchet_workflows.py`. Callers no longer need to serialize.
- **In hatchet mode, the writer/consolidator workload runs in a separate `diffmem-worker`
  process**, not in the API uvicorn process. Both processes share the `/data` volume and
  each maintain their own in-process `RepoManager` / `DiffMemory` cache.
- **Volume at `/data` is required.** Storage and worktrees are on disk; the service has no
  in-memory-only mode.
- **Health endpoint is always unauthenticated.** Required for Railway/Coolify probes.

## Attention Guidance
- For write latency issues (inline mode): `writer_agent/agent.py` → `process_session` / `commit_session`.
- For write latency issues (hatchet mode): check Hatchet dashboard for queue depth, worker
  availability, and per-job duration first; then inspect `executor/hatchet_worker.py` handlers.
- For per-user serialization issues: check `executor/inline.py` `_get_user_lock()` (inline mode)
  or the workflow's `ConcurrencyExpression` in `executor/hatchet_workflows.py` (hatchet mode).
- For retrieval quality issues: `retrieval_agent/agent.py` → `run_retrieval_agent`.
- For remote sync issues (edits from other machines not visible): `storage/github_backup.py`
  → `pull_user()`, `storage/local_storage.py` → `get_user_worktree()` pull call site.
- For auth / CORS / startup: `server.py` lifespan + `verify_api_key`.
- For backup push failures: `storage/github_backup.py` → `sync_user()`.
- For consolidation (dedupe / redistribute / link):
  `consolidator_agent/agent.py` → `ConsolidatorAgent` + see
  `consolidator_agent/CONTEXT.md`.
