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
  storage/               — Pluggable storage + backup backends
    factory.py           —   Backend factory; reads STORAGE_BACKEND / BACKUP_BACKEND env vars
    local_storage.py     —   LocalStorageBackend: bare repo + worktrees on disk; calls
                             backup.pull_user() at worktree mount time for existing branches
    github_backup.py     —   GitHubBackupBackend: push (sync_user) + pull (pull_user) user
                             branches against a private GitHub repo
    base.py              —   Abstract base classes; BackupBackend defines pull_user() contract

docs/                    — Structural documentation
  CODE_INDEX.md          —   This file
  deployment.md          —   Docker / Coolify / Railway deployment guide

scripts/                 — Utility scripts (Docker healthcheck helpers, etc.)
tests/                   — Test suite
Dockerfile               — Production container image
docker-compose.yml       — Self-hosting entry point (mounts /data volume)
repo_guide.md            — Memory schema reference (copied into each user worktree)
pyproject.toml           — Package metadata and dependencies
```

## Entry Points

- **HTTP server:** `src/diffmem/server.py` → `uvicorn diffmem.server:app`
- **Python library:** `from diffmem import DiffMemory` → `api.py`
- **Write pipeline:** `DiffMemory.process_and_commit_session()` →
  `writer_agent/agent.WriterAgent.process_session()` + `commit_session()`
- **Read pipeline:** `DiffMemory.get_context()` →
  `retrieval_agent/agent.run_retrieval_agent()` → `resolver.resolve_pointers()`
- **Storage factory:** `storage/factory.py` → `LocalStorageBackend` (default) +
  optional `GitHubBackupBackend`

## Cross-Capability Flows

1. **Write turn:** `POST /memory/{id}/process-and-commit` → `_writer_pool.run_in_executor`
   → `WriterAgent.process_and_commit_session()` (blocks in thread) → git commit on
   `/data/worktrees/{id}` → fire-and-forget backup → HTTP 200 returned immediately after
   thread completes.

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
