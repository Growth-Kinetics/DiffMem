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

## Terminology
- **Worktree:** Per-user git working directory. Mounted lazily on first request.
- **Storage repo:** The bare central repo at `/data/storage`. All user branches live here.
- **Backup backend:** Optional out-of-band mirror (GitHub). Push is out-of-band; pull happens
  at mount time (first request after restart) to pick up remote edits.
- **Writer pool:** `ThreadPoolExecutor(max_workers=4)` in `server.py` that runs all blocking
  operations off the uvicorn event loop — writes, reads, and remote pulls.

## Key Files
- `server.py` — FastAPI app, all HTTP endpoints, `_writer_pool` thread pool, lifespan.
- `api.py` — `DiffMemory` class: public Python API, delegates to writer/retrieval agents.
- `repo_manager.py` — Worktree mount/unmount, `list_active_users()`, post-commit hook install.
- `writer_agent/agent.py` — Multi-step LLM pipeline: identifies entities → stages git changes
  → commits. Uses `ThreadPoolExecutor` internally for parallel LLM calls.
- `retrieval_agent/agent.py` — Multi-turn agent with sandboxed shell tool. Explores repo and
  returns a structured retrieval plan.
- `storage/factory.py` — Pluggable storage/backup backend factory.
- `consolidator_agent/agent.py` — `ConsolidatorAgent`: out-of-band repair pass
  with three tools (`run_dedupe`, `run_redistribute`, `run_link`). Produces
  `consolidate(...)`-prefixed commits.

## External Dependencies
- **OpenRouter** — all LLM calls (writer, onboarding, retrieval agents). Model configured
  via `DEFAULT_MODEL` / `RETRIEVAL_MODEL` env vars.
- **GitHub** (optional) — backup backend when `BACKUP_BACKEND=github`.

## Constraints
- **All blocking operations run in `_writer_pool`:** writes (process/commit), reads
  (`/context`), remote pulls at mount time, AND consolidation runs. Keeps the
  uvicorn event loop free for health probes at all times.
- **Consolidator commits use the `consolidate:` prefix** — specifically
  `consolidate(dedupe):`, `consolidate(redistribute):`, and
  `consolidate(link):`. This lets retrieval agents and human auditors
  distinguish them from session-formation commits. See ADR-D006.
- **Pull happens at mount time only** (first request per user per process lifetime, i.e. after
  a service restart). If the service stays up for days and you push memory edits from another
  machine, DiffMem won't see them until the next restart. Pull is fast-forward only — diverged
  branches fail cleanly with a warning log.
- **One writer per user at a time.** No concurrency locks on worktrees; callers must serialize.
- **Volume at `/data` is required.** Storage and worktrees are on disk; the service has no
  in-memory-only mode.
- **Health endpoint is always unauthenticated.** Required for Railway/Coolify probes.

## Attention Guidance
- For write latency issues: `writer_agent/agent.py` → `process_session` / `commit_session`.
- For retrieval quality issues: `retrieval_agent/agent.py` → `run_retrieval_agent`.
- For remote sync issues (edits from other machines not visible): `storage/github_backup.py`
  → `pull_user()`, `storage/local_storage.py` → `get_user_worktree()` pull call site.
- For auth / CORS / startup: `server.py` lifespan + `verify_api_key`.
- For backup push failures: `storage/github_backup.py` → `sync_user()`.
- For consolidation (dedupe / redistribute / link):
  `consolidator_agent/agent.py` → `ConsolidatorAgent` + see
  `consolidator_agent/CONTEXT.md`.
