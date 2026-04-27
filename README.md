# DiffMem: Git-Based Differential Memory for AI Agents

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Prototype](https://img.shields.io/badge/status-prototype-orange.svg)](https://github.com/Growth-Kinetics/DiffMem)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Growth-Kinetics/DiffMem)

DiffMem is a lightweight, git-based memory backend for AI agents and conversational systems. It uses Markdown files for human-readable storage, Git for tracking temporal evolution through differentials, and a git-native retrieval agent that explores the repository via shell commands (`grep`, `git log`, `git diff`, `git blame`) to build targeted context. No vector databases, no embeddings, no BM25 — just git and an LLM.

At its core, DiffMem treats memory as a versioned repository: the "current state" of knowledge is stored in editable files, while historical changes are preserved in Git's commit graph. This separation allows agents to query and search against a compact, up-to-date surface without the overhead of historical data, while enabling deep dives into evolution when needed.

## Live in Production

DiffMem powers [Annabelle](https://withanna.io/), a simulated intelligence that maintains persistent memory across thousands of conversations on WhatsApp and Messenger.

In production, DiffMem enables Annabelle to:

- Reference details from conversations weeks ago
- Track the evolution of relationships over time
- Build structured understanding of each person she talks to
- Consolidate memories automatically as conversations grow

→ [See how DiffMem processes a novel chapter by chapter](https://github.com/Growth-Kinetics/diffmem_sample_memory)

## Roadmap

- [ ] Indexing strategy from PoC needs to be made more robust, too memory intensive without need.
- [ ] Parametrized method for context caps on retrieval.
- [ ] Sometimes an entity will become a catch-all and the thing will insist in overloading it.
- [ ] Retrieval History so that we can spin up a "linked entities" model to support wikification
- [ ] PDF export.
- [ ] Research PoC: Visual retrieval for context compression.

## Why Git for AI Memory?

Traditional memory systems for AI agents often rely on databases, vector stores, or graph structures. These work well for certain scales but can become bloated or inefficient when dealing with long-term, evolving personal knowledge. DiffMem takes a different path by leveraging Git's strengths:

- **Current-State Focus**: Memory files store only the "now" view of information (e.g., current relationships, facts, or timelines). This reduces the surface area for queries and searches, making operations faster and more token-efficient in LLM contexts. Historical states are not loaded by default — they live in Git's history, accessible on-demand.

- **Differential Intelligence**: Git diffs and logs provide a natural way to track how memories evolve. Agents can ask "How has this fact changed over time?" without scanning entire histories, pulling only relevant commits.

- **Durability and Portability**: Plaintext Markdown ensures memories are human-readable and tool-agnostic. Git's distributed nature means your data is backup-friendly and not locked into proprietary formats.

- **Efficiency for Agents**: By separating "surface" (current files) from "depth" (git history), agents can be selective — load the now for quick responses, dive into diffs for analytical tasks. This keeps context windows lean while enabling rich temporal reasoning.

This approach shines for long-horizon AI systems where memories accumulate over years: it scales without sprawl, maintains auditability, and allows "smart forgetting" through pruning while preserving reconstructability.

## How It Works

DiffMem ships as a small FastAPI service. Key components:

- **Writer Agent** (`writer_agent`): Analyzes conversation transcripts, identifies/creates entities, and stages updates in Git's working tree. Commits are explicit and atomic.

- **Retrieval Agent** (`retrieval_agent`): A multi-turn LLM agent with a single `run(command="...")` tool that explores the memory repository via sandboxed shell commands. It reads `index.md`, probes git history for temporal patterns, and outputs a structured retrieval plan (file sections, git diffs, commit logs) that gets resolved into context.

- **API Layer** (`api.py` + `server.py`): HTTP endpoints for onboarding users, processing sessions, and retrieving context. Also importable as a Python library.

Each user gets an isolated **orphan branch** (`user/{user_id}`) inside a single local storage repo, checked out into a per-user worktree when active. Branches share no history with each other — it's strict isolation without per-user repos.

### Storage architecture

The service has two pluggable concerns:

- **Storage backend** — where the repo and worktrees live. Default is `local` (a mounted disk). This is a hard requirement of the retrieval agent, which shells out to `grep`/`git log` on a real directory.
- **Backup backend** — an *optional* out-of-band mirror. Options: `none` (default; rely on volume snapshots) or `github` (mirror user branches to a private GitHub repo you own). Backups run on a scheduler — they're never in the request hot path, so LLM latency is never blocked on a push.

This separation means self-hosters can run DiffMem with zero external dependencies, and users who want an offsite mirror can opt in with two env vars.

## Self-Hosting

DiffMem is designed to be deployed on a single small Linux box with a mounted volume. It's I/O-bound, not compute-bound — an e2-small / 1 vCPU VPS is plenty for thousands of conversations.

### One-click deploy with Coolify

[Coolify](https://coolify.io/) is an open-source, self-hostable Heroku/Vercel alternative. It's the easiest way to run DiffMem.

1. In Coolify, create a new **Docker Compose** resource.
2. Point it at this repository: `https://github.com/Growth-Kinetics/DiffMem`.
3. Set the compose file path to `docker-compose.yml` (default).
4. In the **Environment Variables** tab, set `OPENROUTER_API_KEY` to your key from [openrouter.ai/keys](https://openrouter.ai/keys).
5. (Optional) Attach a domain — Coolify handles TLS via Let's Encrypt automatically.
6. Click **Deploy**.

Coolify will build the image, provision a named volume at `/data` (persists across deployments), run the healthcheck, and route traffic through its built-in Traefik reverse proxy. No TLS certs, no nginx configs, no open ports on the host.

DiffMem listens on `PORT`, defaulting to `8000`. If Coolify asks for the
service port or proxy target, use the same value you set for `PORT`.

Leave `REQUIRE_AUTH=false` (the default) if you're only calling DiffMem from another service on the same Coolify instance. Set `REQUIRE_AUTH=true` + `API_KEY=<long-random-string>` if you expose the domain publicly.

### Plain Docker Compose

On any Linux box with Docker:

```bash
git clone https://github.com/Growth-Kinetics/DiffMem.git
cd DiffMem
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY
docker compose up -d
```

The service listens on `http://localhost:8000` by default. Set `PORT` to change
both the app listener and the Docker host binding, for example
`PORT=8062 docker compose up -d`. All state lives in the `diffmem_data` named
volume — back it up with:

```bash
docker run --rm -v diffmem_data:/data -v "$(pwd):/backup" alpine tar czf "/backup/diffmem-$(date +%F).tar.gz" /data
```

### As a Python library

```python
from diffmem import DiffMemory

memory = DiffMemory("/path/to/worktree", "alex", "your-openrouter-key")
memory.process_and_commit_session("Had coffee with mom today...", "session-123")
context = memory.get_context([{"role": "user", "content": "Tell me about mom"}])
```

### Configuration

Everything is configured via environment variables. Only `OPENROUTER_API_KEY` is required; see `.env.example` for the full list with defaults.

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | *(required)* | Your OpenRouter key |
| `DEFAULT_MODEL` | `xiaomi/mimo-v2-omni` | LLM used by agents (any OpenRouter model slug) |
| `REQUIRE_AUTH` | `false` | Enable bearer-token auth (set true for public deployments) |
| `API_KEY` | *(unset)* | Shared bearer token when `REQUIRE_AUTH=true` |
| `ALLOWED_ORIGINS` | `*` | CORS origins, comma-separated |
| `PORT` | `8000` | App listener and Docker host port |
| `BACKUP_BACKEND` | `none` | `none` or `github` |
| `BACKUP_INTERVAL_MINUTES` | `30` | Backup cadence (0 disables periodic backups) |
| `GITHUB_REPO_URL` | *(unset)* | Private repo for the `github` backup backend |
| `GITHUB_TOKEN` | *(unset)* | PAT with `repo` scope, for `github` backup |
| `STORAGE_PATH` | `/data/storage` | Where the central git repo lives |
| `WORKTREE_ROOT` | `/data/worktrees` | Where per-user worktrees are mounted |

### Enabling GitHub backup (optional)

Want an offsite mirror without paying for external storage? Create a private GitHub repo (e.g. `yourname/my-diffmem-backup`), generate a [Personal Access Token](https://github.com/settings/tokens) with `repo` scope, then set:

```
BACKUP_BACKEND=github
GITHUB_REPO_URL=https://github.com/yourname/my-diffmem-backup
GITHUB_TOKEN=ghp_...
```

**Cold-start restore**: on a brand-new deployment with an empty `/data` volume, DiffMem fetches any existing `user/*` branches from the remote so you don't start from scratch (useful for migrations and disaster recovery). Once the volume has user branches, startup-time restores are skipped — the mounted volume is the source of truth.

**Per-commit backup**: a post-commit hook fires a webhook that pushes the user's branch to GitHub in the background. Push failures never block the request — the periodic backup (`BACKUP_INTERVAL_MINUTES`) catches up on the next tick.

**Credentials**: the token is passed to git via `GIT_ASKPASS` at call time, never written into `.git/config` on the volume.

## Migrating from earlier DiffMem versions

If you're upgrading from a pre-`0.4` deployment (where GitHub was the primary datastore), three behaviors change:

- **GitHub is now a backup, not a database.** The mounted `/data` volume is the source of truth. Snapshot it like any other stateful service.
- **Writes no longer block on GitHub.** `POST /process-and-commit` returns as soon as the local commit lands; the push runs in the background. Expect noticeably faster API responses.
- **Default storage paths moved** from `/app/storage` and `/app/worktrees` to `/data/storage` and `/data/worktrees`. Existing deployments that set these env vars explicitly are unaffected; deployments relying on the old defaults should either rebind the volume or set `STORAGE_PATH` / `WORKTREE_ROOT` to the old paths.

To preserve backwards compatibility, setting `GITHUB_REPO_URL` + `GITHUB_TOKEN` without an explicit `BACKUP_BACKEND` automatically enables the GitHub backup backend.

## API

Full interactive docs live at `http://<your-host>/docs` (Swagger UI) once the server is running.

The endpoints you'll actually use:

- `POST /memory/{user_id}/onboard` — create a new user
- `POST /memory/{user_id}/process-and-commit` — ingest a session transcript and commit
- `POST /memory/{user_id}/context` — retrieve context for a conversation

Example:

```bash
curl -X POST "http://localhost:8000/memory/alex/onboard" \
  -H "Content-Type: application/json" \
  -d '{"user_info": "Alex is a software engineer from Seattle.", "session_id": "onboard-001"}'

curl -X POST "http://localhost:8000/memory/alex/process-and-commit" \
  -H "Content-Type: application/json" \
  -d '{"memory_input": "Had coffee with mom today. She mentioned her new job.", "session_id": "s-001"}'

curl -X POST "http://localhost:8000/memory/alex/context" \
  -H "Content-Type: application/json" \
  -d '{"conversation": [{"role": "user", "content": "Tell me about mom"}], "max_tokens": 15000}'
```

If `REQUIRE_AUTH=true`, add `-H "Authorization: Bearer $API_KEY"` to every request.

## Repository layout

Each user's memory is organized as:

```
<worktree_root>/{user_id}/
├── {user_id}.md              # User's own profile
├── index.md                  # Auto-generated keyword index
├── memories/
│   ├── people/               # Per-person profiles
│   └── contexts/             # Thematic contexts (health, work, ...)
└── timeline/
    └── YYYY-MM.md            # Monthly timeline entries
```

See `repo_guide.md` in the repo root for the full memory schema (this file is copied into each user's worktree as `repo_guide.md` so the writer agent can reference it).

## Prototype Status and Limitations

What's working:
- Entity creation/update from transcripts.
- Git-native agent retrieval with temporal reasoning.
- Targeted context assembly (file sections, diffs, commit logs).
- Fallback to baseline (user entity) when agent fails.

Known limitations:
- Agent retrieval quality depends on the LLM model used.
- No multi-user concurrency locks (one worktree = one writer at a time).
- Writer agent prompt tuning is ongoing.

We're sharing this as open-source R&D to spark discussion. Feedback welcome!

## Future Vision

DiffMem points to a future where AI memory is as versioned and collaborative as code:

- **Agent-Driven Pruning**: LLMs that "forget" low-strength memories by archiving to git branches, mimicking neural plasticity.
- **Collaborative Memories**: Multi-agent systems sharing repos, with merge requests for "memory reconciliation."
- **Temporal Agents**: Specialized models that query git logs to answer "how did I change?"
- **Multi-Provider Retrieval**: Swap between OpenRouter, Cerebras, or any OpenAI-compatible provider.
- **Open-Source Ecosystem**: Plugins for voice input, mobile sync, or integration with tools like Obsidian.

This is an R&D project from Growth Kinetics, a boutique data solutions agency specializing in AI enablement. We'd love collaborations, PRs, or honest feedback to improve it.

## Contributing

Fork, experiment, PR. We're especially interested in:
- Alternative storage / backup backends (S3, GCS, plain rsync).
- Retrieval strategy improvements.
- Real-world integrations.

License: MIT
Growth Kinetics © 2025
