# Production Deployment: Hetzner Cloud + Coolify + Hatchet

This guide provisions a production DiffMem instance with durable, observable task
execution via [Hatchet](https://cloud.onhatchet.run). Two containers share one image:

| Container | Role |
|---|---|
| `diffmem-api` | FastAPI HTTP server. Enqueues jobs into Hatchet. |
| `diffmem-worker` | Long-running Hatchet worker. Executes jobs. Outbound-only. |

Both mount the same `/data` volume (worktrees + storage). The worker is never
reachable from the internet; it opens an outbound gRPC connection to Hatchet Cloud.

---

## Prerequisites

- **Hetzner Cloud account** — any VPS size (CX22 / 2 vCPU / 4 GB RAM is plenty).
- **Coolify** installed on the VPS. One-liner install:
  ```bash
  curl -fsSL https://cdn.coollabs.io/coolify/install.sh | bash
  ```
  Full docs: [https://coolify.io/docs](https://coolify.io/docs/installation)
- **Hatchet Cloud account** — the free tier covers 1 000 runs/month;
  Annabelle-scale (~6 000 runs/month) requires the Hobby plan (~$25/mo).
  Sign up at [https://cloud.onhatchet.run](https://cloud.onhatchet.run).

---

## Step 1 — Hatchet Cloud setup

1. Go to [https://cloud.onhatchet.run](https://cloud.onhatchet.run) and sign up.
2. Create a new **Tenant** (one tenant per environment is fine).
3. Note your **Tenant ID** (visible in the dashboard URL and Settings page).
4. Open **Settings → API Tokens** → click **Create API Token**.
5. Name it `diffmem-prod`. Copy the token and save it somewhere safe — it's only
   shown once. This becomes `HATCHET_CLIENT_TOKEN`.

---

## Step 2 — Coolify project setup

1. Log into your Coolify instance at `https://<your-vps-ip>:8000`.
2. Create a new **Project** (e.g. `diffmem-prod`).
3. Inside the project, add a new **Resource → Docker Compose**.
4. Set **Source** to this repo: `https://github.com/Growth-Kinetics/DiffMem`
5. Set **Compose File Path** to: `deploy/docker-compose.hatchet.yml`
6. Leave **Branch** as `main`.

---

## Step 3 — Environment variables in Coolify

In the **Environment Variables** tab of the new resource, set these values.

### Required

| Variable | Example value | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | `sk-or-v1-…` | From [openrouter.ai/keys](https://openrouter.ai/keys) |
| `DEFAULT_MODEL` | `anthropic/claude-3.5-sonnet` | Any OpenRouter model slug |
| `HATCHET_CLIENT_TOKEN` | `eyJ…` | From Step 1 |

`EXECUTOR` is already hardcoded to `hatchet` in the compose file — don't override it.

### Recommended

| Variable | Value | Notes |
|---|---|---|
| `HATCHET_NAMESPACE` | `diffmem` | Prefix for workflow/worker names. Default: `diffmem` |
| `REQUIRE_AUTH` | `true` | Enables bearer-token auth on all endpoints |
| `API_KEY` | `<long random string>` | Required when `REQUIRE_AUTH=true`. Generate with `openssl rand -hex 32` |

### Optional

| Variable | Default | Notes |
|---|---|---|
| `RETRIEVAL_MODEL` | *(unset — uses DEFAULT_MODEL)* | Override the retrieval-only model |
| `HATCHET_WORKER_SLOTS` | `10` | In-process concurrency per worker replica. Raise if needed |
| `HATCHET_CLIENT_HOST_PORT` | *(unset — Hatchet Cloud)* | Set only for self-hosted Hatchet (e.g. `engine.example.com:7077`) |
| `HATCHET_CLIENT_TLS_STRATEGY` | `tls` | `tls` \| `mtls` \| `none` |
| `BACKUP_BACKEND` | `none` | Set to `github` + credentials for offsite mirror |
| `BACKUP_INTERVAL_MINUTES` | `30` | Backup cadence in minutes |
| `GITHUB_REPO_URL` | *(unset)* | Private GitHub repo for the `github` backup backend |
| `GITHUB_TOKEN` | *(unset)* | PAT with `repo` scope |
| `PORT` | `8000` | API listen port — Coolify auto-detects this |

---

## Step 4 — Volume

Coolify auto-provisions the `diffmem_data` named volume when the first deploy runs.
Both containers mount it at `/data`. The volume persists across deploys.

**Durability options:**

- **Hetzner Volume Snapshot** (recommended): attach a Hetzner Block Storage volume,
  mount it at `/data`, and take snapshots via the Hetzner Cloud console. Costs ~€0.05/GB/month.
- **GitHub backup**: set `BACKUP_BACKEND=github` — pushes user branches offsite after
  each write. Survives VPS death. Does not backup binary or non-git data.

For Annabelle scale, the local volume + periodic Hetzner snapshot is sufficient.

---

## Step 5 — Deploy

1. In Coolify, click **Deploy** (or push a commit to the configured branch).
2. Coolify builds the image (first time: ~3 min; subsequent: ~30s with layer cache).
3. `diffmem-api` starts first; healthcheck polls `/health` every 30s.
4. `diffmem-worker` starts and opens an outbound gRPC connection to Hatchet Cloud.
5. In the **Hatchet Cloud dashboard → Workers**, within ~30s you should see
   `diffmem-worker` (or `diffmem.worker` if using the `diffmem` namespace) listed
   as **Online**.

---

## Step 6 — Verify

```bash
# API responds
curl https://<your-coolify-domain>/health
# → {"status": "healthy", "storage_backend": "local", "backup_backend": "none", "executor_type": "HatchetExecutor", ...}

# Worker visible in Hatchet
# → Open https://cloud.onhatchet.run → your tenant → Workers tab
# → Should show: diffmem-worker  |  Online  |  slots: 10
```

If `REQUIRE_AUTH=true`, add `-H "Authorization: Bearer $API_KEY"` to every request.

---

## Step 7 — First write

Onboard a test user and run a write to confirm the end-to-end Hatchet flow:

```bash
# Onboard
curl -X POST "https://<your-domain>/memory/test-user/onboard" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{"user_info": "Test user.", "session_id": "onboard-001"}'

# Write (enqueues a Hatchet workflow run)
curl -X POST "https://<your-domain>/memory/test-user/process-and-commit" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{"memory_input": "First memory entry.", "session_id": "s-001"}'
```

In the **Hatchet dashboard → Workflow Runs**, you should see:
1. A new run appear within ~1s of the POST.
2. The worker picks it up and transitions to `Running`.
3. Within 30–60s, it completes with status `Succeeded`.

---

## Scaling workers

To handle more concurrent sessions, add worker replicas. In `deploy/docker-compose.hatchet.yml`,
add a `deploy` block to the `diffmem-worker` service:

```yaml
  diffmem-worker:
    ...
    deploy:
      replicas: 2
```

Hatchet distributes workflow runs across all connected workers automatically using
its built-in `ConcurrencyExpression` — per-user write serialization is guaranteed
even across replicas.

You can also increase `HATCHET_WORKER_SLOTS` (default 10) to raise in-process
concurrency per replica. Keep in mind each slot holds an open writer agent run
(LLM calls + git I/O), so don't raise it above the VPS memory ceiling.

---

## Backups

The `/data` volume is the **source of truth**. Two protection layers:

| Layer | What it covers | How to enable |
|---|---|---|
| Hetzner Volume Snapshot | Full `/data`, binary-safe | Hetzner Cloud console → Volumes → Snapshot |
| GitHub backup backend | User git branches only | `BACKUP_BACKEND=github` + credentials |

Neither is automatic by default. For Annabelle, the GitHub backend is sufficient as a
cold-start restore path; Hetzner snapshots add belt-and-suspenders coverage.

**Cold-start restore** (GitHub backend): on a fresh VPS with an empty volume, DiffMem
fetches existing `user/*` branches from the GitHub remote at first access. No manual
intervention required.

---

## Troubleshooting

### Worker doesn't appear in Hatchet dashboard

1. Check worker container logs in Coolify: `diffmem-worker` → Logs.
2. Common causes:
   - `HATCHET_CLIENT_TOKEN` is wrong or expired — regenerate in Hatchet dashboard.
   - `HATCHET_NAMESPACE` mismatch — must be the same value in both services.
   - Network issue: the VPS must have outbound internet access on port 443 (gRPC/TLS).

### API starts but worker never picks up jobs

- Worker is not connected — see above.
- `EXECUTOR` env var is not `hatchet` on the worker container — verify in Coolify env tab.

### `docker compose config` fails locally

Run with the required vars set:
```bash
OPENROUTER_API_KEY=x DEFAULT_MODEL=x HATCHET_CLIENT_TOKEN=x \
  docker compose -f deploy/docker-compose.hatchet.yml config
```
Or use `:-` defaults already present in the file — the file passes config with no vars set.

### Healthcheck fails after deploy

- The API container may still be starting. Coolify retries for `start_period: 10s`.
- Check that `PORT` matches between the `ports:` mapping and the service config.
- Inspect logs: Coolify → resource → `diffmem-api` → Logs.

---

## Dashboard URLs

| Service | URL |
|---|---|
| Hatchet Cloud | [https://cloud.onhatchet.run](https://cloud.onhatchet.run) |
| API Swagger UI | `https://<your-coolify-domain>/docs` |
| API Health | `https://<your-coolify-domain>/health` |
| Coolify | `https://<your-vps-ip>:8000` |
