# Code Index

## Repository Shape

```
src/diffmem/             — Core package (importable as a library or run as a server)
  server.py              — FastAPI app: all HTTP endpoints, _writer_pool (ThreadPoolExecutor),
                           lifespan, backup scheduler, auth middleware
  api.py                 — DiffMemory class: public Python API surface, delegates to agents
  repo_manager.py        — Worktree lifecycle (mount/unmount), post-commit hook install,
                           list_active_users()
  writer_agent/          — Synchronous LLM pipeline: identifies entities, stages + commits
    agent.py             —   WriterAgent: process_session(), commit_session()
    onboarding_agent.py  —   OnboardingAgent: first-time user setup
    prompts/             —   Prompt files for writer and onboarding agents
  retrieval_agent/       — Multi-turn LLM agent with sandboxed shell tool
    agent.py             —   run_retrieval_agent(): explores repo, returns RetrievalPlan
    command_router.py    —   Sandboxed shell command executor (allowlisted git + unix cmds)
    baseline.py          —   load_baseline(), load_user_entity(), load_recent_timeline()
    resolver.py          —   resolve_pointers(): converts RetrievalPlan → context blocks
    prompts/             —   Prompt files for retrieval agent
  consolidator_agent/    — Out-of-band repair pass: dedupe, redistribute, link
    agent.py             —   ConsolidatorAgent: run_dedupe(), run_redistribute(), run_link()
    lock.py              —   ConsolidatorLock context manager + LockBusyError
    prompts/             —   Prompt files for consolidator tools (populated in M2–M4)
  storage/               — Pluggable storage + backup backends
    factory.py           —   Backend factory; reads STORAGE_BACKEND / BACKUP_BACKEND env vars
    local_storage.py     —   LocalStorageBackend: bare repo + worktrees on disk; calls
                             backup.pull_user() at worktree mount time for existing branches
    github_backup.py     —   GitHubBackupBackend: push (sync_user) + pull (pull_user) user
                             branches against a private GitHub repo
    base.py              —   Abstract base classes; BackupBackend defines pull_user() contract

  ontology/              — Ontology profile loader (DIFFMEM_ONTOLOGY env var)
    loader.py            —   OntologyProfile dataclass + load_ontology(); entity_dirs(), contexts_folder(),
                             resolve_prompt() fallback chain; raises ValueError on unknown name
    CONTEXT.md           —   Capability context
    __init__.py
  ontologies/            — Built-in ontology profiles (co-located with package for pip install)
    personal/            —   Default: memories/people, memories/contexts, memories/events
    corporate/           —   CRM: entities/people, projects, decisions, commitments, external
    README.md            —   schema.json contract + OSS contribution guide

  executor/              — Pluggable task executor; backend chosen via EXECUTOR env var
    base.py              —   TaskExecutor ABC, JobStatus/JobHandle/JobResult, WritePayload/ConsolidatePayload
    jobstore.py          —   JobStore: thread-safe OrderedDict with FIFO eviction (inline mode)
    inline.py            —   InlineExecutor: _writer_pool + threading.Lock per user (default)
    hatchet.py           —   HatchetExecutor: submits workflow runs to Hatchet engine
    hatchet_workflows.py —   WriteInput/ConsolidateInput models + register_workflows(); shared by API + worker
    hatchet_worker.py    —   Worker process: @workflow.task() handlers + worker.start() loop
    factory.py           —   build_executor(pool): reads EXECUTOR env var, returns correct impl

ontologies/              — Human-browsable mirrors of src/diffmem/ontologies/ (kept in sync; see README)
  personal/              —   Mirror of personal profile
  corporate/             —   Mirror of corporate profile
  README.md              —   Sync note + contribution guide (canonical copy)

docs/                    — Structural documentation
  CODE_INDEX.md          —   This file
  deployment.md          —   Docker / Coolify inline-mode deployment guide
  deployment-hatchet.md  —   Hetzner + Coolify + Hatchet Cloud production deployment guide

scripts/                 — Utility scripts
  smoke_hatchet_live.py  —   Live Phase-1 smoke test against Hatchet Cloud (PR #14 ED-013 validation)
tests/                   — Test suite
Dockerfile               — Production container image (includes hatchet-sdk)
docker-compose.yml       — Self-hosting entry point, inline executor (mounts /data volume)
deploy/
  docker-compose.hatchet.yml — Production template: diffmem-api + diffmem-worker services
repo_guide.md            — Memory schema reference (copied into each user worktree)
pyproject.toml           — Package metadata and dependencies
```

## Entry Points

- **HTTP server (`diffmem-server`):** `uvicorn diffmem.server:app` → `server.py`
- **Hatchet worker (`diffmem-worker`):** `hatchet_worker.main()` → `executor/hatchet_worker.py`
- **Python library:** `from diffmem import DiffMemory` → `api.py`
- **Write pipeline:** `DiffMemory.process_and_commit_session()` →
  `writer_agent/agent.WriterAgent.process_session()` + `commit_session()`
- **Read pipeline:** `DiffMemory.get_context()` →
  `retrieval_agent/agent.run_retrieval_agent()` → `resolver.resolve_pointers()`
- **Storage factory:** `storage/factory.py` → `LocalStorageBackend` (default) +
  optional `GitHubBackupBackend`
- **Ontology:** `DiffMemory.__init__()` calls `load_ontology()` once; propagated to
  `WriterAgent` (prompt resolution, folder map), `OnboardingAgent` (dir creation,
  repo_guide copy), and `run_retrieval_agent()` (folder listing in system prompt)
- **Consolidation pipeline:** `consolidator_agent/agent.py` →
  `ConsolidatorAgent.run_dedupe()` / `run_redistribute()` / `run_link()`.
  Commits use the `consolidate(...)` prefix.
- **Consolidate API:** `DiffMemory.consolidate(tools, window, soft_cap_tokens)`
  and `DiffMemory.process_commit_and_consolidate(...)` in `api.py`.
- **Consolidate HTTP:** `POST /memory/{user_id}/consolidate` and
  `POST /memory/{user_id}/process-commit-and-consolidate` in `server.py`;
  both routed through `_writer_pool`.

## Cross-Capability Flows

1. **Write turn:** `POST /memory/{id}/process-and-commit` →
   `app.state.executor.submit_write()` →
   **inline:** per-user lock → `_writer_pool` → `DiffMemory.process_and_commit_session()` → git commit on `/data/worktrees/{id}` → fire-and-forget backup, OR
   **hatchet:** Hatchet engine → `diffmem-worker` process → `DiffMemory.process_and_commit_session()` → git commit → fire-and-forget backup.

2. **Read turn:** `POST /memory/{id}/context` → `_writer_pool.run_in_executor` →
   `DiffMemory.get_context()` → `run_retrieval_agent()` (blocking, in thread) →
   sandboxed shell commands on worktree → `resolve_pointers()` → context blocks returned.

3. **Remote pull (mount time):** First request for a user after restart →
   `local_storage.get_user_worktree()` (cache miss) → `backup.pull_user(user_id)` →
   fetch + fast-forward local branch + worktree HEAD from GitHub origin. Non-fatal.

4. **Backup (push):** post-commit git hook fires `POST /memory/{id}/webhook/post-commit` →
   `backup_user(id)` (background task) → `RepoManager.sync_user()` →
   `GitHubBackupBackend.sync_user()`. Also runs on periodic scheduler
   (`BACKUP_INTERVAL_MINUTES`, default 30).

5. **Consolidate (out-of-band):** `POST /memory/{id}/consolidate` →
   `app.state.executor.submit_consolidate()` → (inline pool or hatchet worker) →
   `DiffMemory.consolidate()` → acquire `.diffmem/consolidator.lock` per tool →
   dedupe → redistribute → link → each tool produces `consolidate(...)`-prefixed
   commits → fire-and-forget backup of the new commits.
