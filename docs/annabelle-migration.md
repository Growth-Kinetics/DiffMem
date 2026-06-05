# Annabelle Migration Runbook: Railway → Hetzner + Coolify + Hatchet

**Audience:** alex (primary operator) and any future operator deploying DiffMem-for-Annabelle.  
**Purpose:** Step-by-step instructions for cutting Annabelle's DiffMem instance over from the current Railway deployment (inline executor, single thread pool, 4 workers) to a new Hetzner Cloud VPS running under Coolify with Hatchet Cloud as the task executor.  
**After migration:** durable jobs, per-user concurrency visible in a dashboard, no more "all 4 workers are busy with one session" pile-ups during Annabelle peak traffic.

---

## 1. Overview

Today, Annabelle's DiffMem instance lives on Railway. It uses the **InlineExecutor** — a thread pool with 4 workers. When Annabelle submits multiple sessions in quick succession (e.g. a busy day with many conversations), all 4 threads fill up and subsequent writes queue silently inside the process. If Railway restarts the container mid-write, that job is lost with no record of failure. There is no visibility into queue depth or per-session durations.

The new deployment runs DiffMem on a **Hetzner Cloud CX22 VPS** managed by **Coolify**, with **Hatchet Cloud** as the task executor. Every write and consolidate call becomes a durable Hatchet workflow run, visible in the Hatchet dashboard. Per-user serialization is handled by Hatchet's built-in `ConcurrencyExpression` rather than an in-process lock, so it survives worker restarts. Data durability is unchanged: the GitHub backup backend is the source of truth before, during, and after migration.

---

## 2. Pre-migration checklist

Complete every item before starting the migration steps.

- [ ] **Hetzner Cloud account set up.** You can sign up at [https://console.hetzner.cloud](https://console.hetzner.cloud). Have billing configured before provisioning a VPS.

- [ ] **Hatchet Cloud account + Hobby plan provisioned.** Sign up at [https://cloud.onhatchet.run](https://cloud.onhatchet.run). The free tier is too small for ~6 000 runs/month (Annabelle's approximate throughput). Upgrade to the Hobby plan before going live.

- [ ] **DiffMem deployment docs read.** Review `docs/deployment-hatchet.md` for detailed Hatchet Cloud setup, environment variables, and Coolify configuration steps.

- [ ] **Railway DiffMem backup verified up-to-date.**
  - In the Railway DiffMem service, confirm `BACKUP_BACKEND=github`, `GITHUB_REPO_URL`, and `GITHUB_TOKEN` are set.
  - Verify the most recent push to the GitHub mirror happened within the last hour:
    ```bash
    git clone <GITHUB_REPO_URL> /tmp/diffmem-mirror-check
    git -C /tmp/diffmem-mirror-check log --oneline -5
    # Should show commits timestamped within the last hour.
    ```
  - If the mirror is stale, trigger a manual backup by hitting `/server/sync` on Railway.

- [ ] **Railway `/data` volume snapshot taken.**
  - Railway dashboard → your project → the DiffMem service → Volumes tab → Snapshot.
  - This is belt-and-suspenders in case the GitHub mirror is missing any data.

- [ ] **Railway env vars exported.** Note the values (you'll paste them into Coolify):
  - `OPENROUTER_API_KEY`
  - `DEFAULT_MODEL`
  - `BACKUP_BACKEND` (should be `github`)
  - `GITHUB_REPO_URL`
  - `GITHUB_TOKEN`
  - `REQUIRE_AUTH` and `API_KEY` (if auth is enabled)
  - Any other custom vars you set.

- [ ] **Annabelle harness deployment access.** You'll need to update Annabelle's `DIFFMEM_BASE_URL` env var in its deployment. Confirm you can deploy Annabelle (e.g. have Coolify / Railway / deployment tool access for Annabelle's side).

---

## 3. Migration steps

### Step 1 — Provision new VPS on Hetzner Cloud

1. Log into [https://console.hetzner.cloud](https://console.hetzner.cloud).
2. Create a new server:
   - **Type:** CX22 (2 vCPU, 4 GB RAM, 40 GB SSD) — minimum recommended. ~€4/mo.
   - **Image:** Ubuntu 24.04 LTS.
   - **Location:** Pick one close to Annabelle's primary users (e.g. Nuremberg or Helsinki for Europe).
   - **SSH key:** Add your public SSH key.
   - **Firewall:** Allow port 22 (SSH) and 443 (HTTPS). Coolify also needs 8000 for its dashboard, but you can lock that to your Tailscale IP only.

3. Once the server is up, SSH in and install Coolify:
   ```bash
   ssh root@<vps-ip>
   curl -fsSL https://cdn.coollabs.io/coolify/install.sh | bash
   ```
   Follow the on-screen prompts. When done, Coolify's dashboard is at `http://<vps-ip>:8000`.

4. Log into Coolify, complete initial setup (admin password), and create a new **Project** named `diffmem-prod`.

---

### Step 2 — Deploy DiffMem on the new VPS

1. Inside the `diffmem-prod` project, add a new **Resource → Docker Compose**.
2. Set the source to the DiffMem GitHub repo (`https://github.com/Growth-Kinetics/DiffMem`), branch `main`.
3. Set the **Compose File Path** to: `deploy/docker-compose.hatchet.yml`.
4. In the **Environment Variables** tab, add:

   | Variable | Value |
   |---|---|
   | `OPENROUTER_API_KEY` | *(from Railway export)* |
   | `DEFAULT_MODEL` | *(from Railway export)* |
   | `BACKUP_BACKEND` | `github` |
   | `GITHUB_REPO_URL` | *(same repo URL as Railway)* |
   | `GITHUB_TOKEN` | *(from Railway export)* |
   | `HATCHET_CLIENT_TOKEN` | *(from Hatchet Cloud → Settings → API Tokens)* |
   | `REQUIRE_AUTH` | *(from Railway export — `true` or `false`)* |
   | `API_KEY` | *(from Railway export, if REQUIRE_AUTH=true)* |

   **Important:** `EXECUTOR=hatchet` is already set in `deploy/docker-compose.hatchet.yml`. Do not override it.

5. Click **Deploy**. First build takes ~3 minutes (Docker layer cache warms up). Subsequent deploys take ~30 seconds.

---

### Step 3 — Cold-start restore from GitHub mirror

On first startup, DiffMem's `RepoManager` discovers existing `user/*` branches from the GitHub mirror and creates local worktrees for them. This is automatic — no manual intervention needed.

To verify the restore completed before proceeding:

```bash
# On the Hetzner VPS (or via Coolify → diffmem-api → Logs):
docker logs diffmem-api 2>&1 | grep -E "(RESTORE_COMPLETE|MEMORY_MOUNT|MEMORY_INSTANCE_CREATED)"
```

Expected output: one `MEMORY_MOUNT` or `MEMORY_INSTANCE_CREATED` log line per user when first accessed. For Annabelle's user specifically, the restore happens on the first write request — it's lazy. You don't need to pre-warm it; Step 5's smoke test triggers it.

---

### Step 4 — Worker connection check

1. Open the [Hatchet Cloud dashboard](https://cloud.onhatchet.run) → your tenant → **Workers** tab.
2. Within ~30 seconds of the `diffmem-worker` container starting, you should see a worker registered as **Online**.
   - Worker name: `diffmem-worker` (or `diffmem.worker` if `HATCHET_NAMESPACE=diffmem`).
   - Status: **Online**.
   - Slots: 10 (default; configurable via `HATCHET_WORKER_SLOTS`).

If the worker does not appear within 60 seconds, see **Troubleshooting** below.

---

### Step 5 — Smoke test (TEST user only — not Annabelle's real user_id)

Run this against the new VPS before touching Annabelle at all.

```bash
BASE_URL="https://<your-coolify-domain>"   # or http://<vps-ip>:8000
AUTH_HEADER=""                             # or: -H "Authorization: Bearer $API_KEY"

# 1. Onboard a test user
curl -s -X POST "$BASE_URL/memory/smoketest-001/onboard" \
  -H "Content-Type: application/json" $AUTH_HEADER \
  -d '{"user_info": "Test user for migration smoke test.", "session_id": "onboard-smoke"}' \
  | python3 -m json.tool

# 2. Submit a write (async — so we can observe Hatchet)
RESPONSE=$(curl -s -X POST "$BASE_URL/memory/smoketest-001/process-and-commit?sync=false" \
  -H "Content-Type: application/json" $AUTH_HEADER \
  -d '{"memory_input": "Annabelle migration smoke test.", "session_id": "smoke-001"}')
echo "$RESPONSE" | python3 -m json.tool
JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

# 3. Poll until completed
for i in $(seq 1 30); do
  STATUS=$(curl -s "$BASE_URL/memory/smoketest-001/jobs/$JOB_ID" $AUTH_HEADER \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['job']['status'])")
  echo "Poll $i: status=$STATUS"
  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then break; fi
  sleep 2
done
echo "Final job status: $STATUS"

# 4. Verify in repo status
curl -s "$BASE_URL/memory/smoketest-001/status" $AUTH_HEADER | python3 -m json.tool
```

Expected outcome:
- Step 2 returns `{"status": "queued", "job_id": "..."}` within 1 second.
- **Hatchet dashboard → Workflow Runs** shows a new run created, picked up by the worker, and completed (green ✓).
- Step 4 repo status shows at least one commit in the user branch.

If `$STATUS` is `failed`, check the Hatchet run's logs in the dashboard and the `diffmem-worker` container logs in Coolify.

---

### Step 6 — Annabelle client update (Phase 1)

> **Phase 1 preserves today's blocking behavior.** Annabelle keeps using `?sync=true` so its request handlers block until the write completes — exactly as Railway behaves today. This is the safest cutover path; no changes to Annabelle's retry logic or error handling are needed.

1. In Annabelle's deployment (wherever its env vars are managed — Coolify, Railway, Kubernetes, etc.), update:
   ```
   DIFFMEM_BASE_URL=https://<your-new-coolify-domain>
   ```
   If Annabelle is currently using `?sync=false` or no sync param on write endpoints, add `?sync=true` to all `process-and-commit` and `consolidate` calls to preserve blocking behavior.  
   If Annabelle is already using blocking calls (the default for InlineExecutor was always sync), no URL parameter changes are needed.

2. Deploy Annabelle.

3. **Monitor for ~24 hours:**
   - **Hatchet dashboard → Workflow Runs:** queue depth should be near-zero between bursts; run durations should match Railway's historical latency.
   - **Hatchet dashboard → Workers:** worker should stay Online continuously.
   - **Annabelle logs:** no new error patterns (HTTP 500s, timeouts, auth failures).
   - **DiffMem API logs** (Coolify → diffmem-api → Logs): watch for `JOB_FAILED` lines.

4. If anything looks wrong during the 24h window, execute the **Rollback procedure** below before data diverges significantly.

---

### Step 7 — Decommission Railway

Once Annabelle has been stable on the new deployment for ~24 hours:

1. **Scale Railway DiffMem to 0 replicas.** Railway dashboard → your project → DiffMem service → Settings → Scale → 0 replicas. (Do not delete the project yet.)

2. **Keep the Railway volume snapshot for 30 days** as an additional fallback. After 30 days, if everything is stable, delete the Railway project entirely.

---

## 4. Phase 2 (optional, post-migration)

Once you are comfortable with the new deployment, you can drop the `?sync=true` parameter from Annabelle's write calls and switch to **async mode with polling or callbacks**.

Benefits of Phase 2:
- Annabelle's HTTP request handlers return in ~100ms instead of blocking for 60–600s per write.
- Hatchet handles retries automatically if a worker restarts mid-job.
- You can fan out multiple sessions concurrently without holding open HTTP connections.

**This phase is entirely optional.** `?sync=true` works indefinitely. The only operational pressure to switch is if Annabelle's HTTP gateway has a short timeout (e.g. < 30s), causing 504s on large sessions.

Phase 2 implementation sketch:
1. Remove `?sync=true` from Annabelle's `process-and-commit` calls.
2. Parse the `job_id` from the `{"status": "queued", ...}` response.
3. Either poll `GET /memory/{user_id}/jobs/{job_id}` or register a `callback_url` pointing at Annabelle's webhook handler.
4. The callback receives `{"job_id": "...", "status": "completed", "result": {...}}` via HTTP POST.

---

## 5. Rollback procedure

> **This section is operationally critical. Read it before starting the migration.**

### When to roll back

Roll back if any of the following occur during or after the cutover:
- Annabelle's error rate rises significantly (e.g. > 1% HTTP 5xx over 15 minutes).
- Jobs are stuck in `queued` for > 5 minutes (worker offline, unresolvable quickly).
- Data inconsistency observed (e.g. writes accepted but not appearing in user status).

### How to roll back (fast path — < 5 minutes)

1. **Point Annabelle back at Railway:**
   ```
   DIFFMEM_BASE_URL=https://<railway-diffmem-domain>
   ```
   Deploy Annabelle. Annabelle is now routing to Railway again.

2. **Scale Railway DiffMem back up** (if you already scaled it to 0):
   - Railway dashboard → DiffMem service → Settings → Scale → 1 replica.
   - Railway pulls the GitHub mirror on startup. Any commits made on the new Hetzner VPS between cutover and rollback decision will already be in the GitHub mirror (pushed by the backup backend). Railway's RepoManager picks them up on first access. **No data loss.**

3. **Verify Railway is live:**
   ```bash
   curl https://<railway-diffmem-domain>/health
   # → {"status": "healthy", ...}
   ```

4. (Optional) Scale the Hetzner/Coolify deployment to 0 to stop it from accepting further writes during investigation.

### Data safety caveat

Any commits made on the new Hetzner VPS between the cutover decision and the Railway scale-up are pushed to the GitHub mirror by the backup backend (because `BACKUP_BACKEND=github` is set on the new deployment). When Railway's DiffMem starts and mounts a user's worktree, it clones from (or fast-forwards from) the GitHub mirror, so those commits will be present. There is **no data loss** in the rollback path.

The one edge case: if a write is in-flight on Hatchet at the moment Railway restarts, that write might complete on Hatchet after Railway has already mounted the worktree. The backup push will still happen, but Railway won't see it until the next worktree mount for that user. This window is very short and self-correcting.

---

## 6. Troubleshooting

### "Worker not appearing in Hatchet dashboard"

1. Open Coolify → `diffmem-worker` → Logs. Look for connection errors.
2. Common causes:
   - `HATCHET_CLIENT_TOKEN` is wrong, expired, or not set on the **worker** container. Coolify env vars must be applied to **both** `diffmem-api` and `diffmem-worker` services — they are separate containers.
   - Regenerate the token in Hatchet dashboard → Settings → API Tokens.
   - The VPS must have outbound internet access on port 443 (gRPC/TLS to Hatchet Cloud). Test: `curl -v https://cloud.onhatchet.run` from the VPS.

### "Job stuck in queued"

- No worker is online. Check the Workers tab in Hatchet dashboard.
- Restart the `diffmem-worker` container in Coolify.
- If the worker keeps crashing on startup, check its logs for the exception.

### "Job failed with OPENROUTER_API_KEY not set"

The worker container is missing env vars. Coolify env vars must be set on **both** the `api` and `worker` services. In the Coolify resource view, ensure both containers show the same env vars. Re-deploy after updating.

### "Commits not appearing in GitHub mirror"

- Verify `BACKUP_BACKEND=github`, `GITHUB_REPO_URL`, and `GITHUB_TOKEN` are set on both services.
- Check `diffmem-api` logs for `BACKUP_ERROR` lines.
- Verify the GitHub token has `repo` (write) scope on the mirror repository.
- Test the token manually: `git ls-remote <GITHUB_REPO_URL>` — if this fails, the token is wrong.

### Sync mode "504 Gateway Timeout"

A job took longer than the 900s `wait_for` timeout (or your reverse proxy's timeout is shorter). Options:
- Switch Annabelle to async mode (`?sync=false`) and poll `GET /jobs/{job_id}`.
- Increase your reverse proxy's upstream timeout (Coolify/Traefik config).
- If the job itself is taking too long, investigate the DiffMem writer logs for that session.

### "API returns 401 Unauthorized"

- `REQUIRE_AUTH=true` is set. Add `-H "Authorization: Bearer $API_KEY"` to all requests.
- Verify `API_KEY` is the same on the API container and in Annabelle's deployment.

### Coolify build fails with "no such file deploy/docker-compose.hatchet.yml"

Ensure the repo is on branch `main` (or the branch that has the `deploy/` directory — it was added in M6 on `consolidator_agent`). After the PR to `main` is merged, this is `main`.

---

## 7. Cost notes

Rough monthly cost for Annabelle-scale DiffMem (~6 000 write runs/month):

| Item | Cost |
|---|---|
| Hetzner CX22 VPS (2 vCPU, 4 GB RAM, 40 GB SSD) | ~€4/mo |
| Hatchet Cloud Hobby plan | ~$25/mo |
| Hetzner volume snapshot (optional, ~20 GB) | ~€1/mo |
| GitHub (backup backend, private repo) | $0 (free tier covers it) |
| OpenRouter API spend | Unchanged — same as Railway |
| **Total new infra** | **~€30/mo (~$33/mo)** |

For comparison, a Railway Hobby plan with a persistent volume and similar compute runs ~$10–25/mo depending on usage, with no observability into job execution. The Hatchet addition roughly doubles the infra cost but eliminates the "blind pool" problem entirely and provides a full job history for debugging.

---

## References

- `docs/deployment-hatchet.md` — full Hatchet + Coolify deployment guide
- `deploy/docker-compose.hatchet.yml` — production compose file
- [Hatchet Cloud dashboard](https://cloud.onhatchet.run)
- [Coolify docs](https://coolify.io/docs)
- [Hetzner Cloud console](https://console.hetzner.cloud)
