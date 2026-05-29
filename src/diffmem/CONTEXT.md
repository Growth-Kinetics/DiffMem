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
- **Backup backend:** Optional out-of-band mirror (GitHub). Never in the request hot path.
- **Writer pool:** `ThreadPoolExecutor(max_workers=4)` in `server.py` that runs blocking
  writer-agent operations off the uvicorn event loop.

## Key Files
- `server.py` — FastAPI app, all HTTP endpoints, `_writer_pool` thread pool, lifespan.
- `api.py` — `DiffMemory` class: public Python API, delegates to writer/retrieval agents.
- `repo_manager.py` — Worktree mount/unmount, `list_active_users()`, post-commit hook install.
- `writer_agent/agent.py` — Multi-step LLM pipeline: identifies entities → stages git changes
  → commits. Uses `ThreadPoolExecutor` internally for parallel LLM calls.
- `retrieval_agent/agent.py` — Multi-turn agent with sandboxed shell tool. Explores repo and
  returns a structured retrieval plan.
- `storage/factory.py` — Pluggable storage/backup backend factory.

## External Dependencies
- **OpenRouter** — all LLM calls (writer, onboarding, retrieval agents). Model configured
  via `DEFAULT_MODEL` / `RETRIEVAL_MODEL` env vars.
- **GitHub** (optional) — backup backend when `BACKUP_BACKEND=github`.

## Constraints
- **Write endpoints are blocking by design (writer agent).** They run in `_writer_pool`
  (4 threads) to keep the uvicorn event loop free for health probes and reads. A large
  session (80+ entities) can take 60–600s — this is expected.
- **One writer per user at a time.** No concurrency locks on worktrees; callers must serialize.
- **Volume at `/data` is required.** Storage and worktrees are on disk; the service has no
  in-memory-only mode.
- **Health endpoint is always unauthenticated.** Required for Railway/Coolify probes.

## Attention Guidance
- For write latency issues: start at `writer_agent/agent.py` → `process_session` / `commit_session`.
- For retrieval quality issues: start at `retrieval_agent/agent.py` → `run_retrieval_agent`.
- For auth / CORS / startup: `server.py` lifespan + `verify_api_key`.
- For backup failures: `storage/github_backup.py`.
