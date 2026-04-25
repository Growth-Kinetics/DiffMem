# PORT Variable Support — Implementation Plan

## Summary

Replace the hardcoded `"8000:8000"` host port in docker-compose.yml with a configurable variable so that deployers can set the external port via an environment variable without editing the compose file. The container port (8000) stays fixed — only the host-side mapping becomes variable. The plan also covers all downstream files that reference port 8000 to keep the project consistent.

## Scope

- Included: Changing the `ports` mapping in docker-compose.yml to use a `${PORT}` variable.
- Included: Updating the docker-compose healthcheck to use the same variable (it currently hardcodes `localhost:8000`).
- Included: Adding `PORT` to `.env.example` with a comment.
- Included: Updating README documentation where port 8000 is referenced in curl examples and descriptive text.
- Included: Updating the `API_URL` default in `server.py` to derive from `PORT` instead of hardcoding `8000`.
- Included: Updating the Dockerfile `EXPOSE` directive comment to reference the variable pattern.
- Excluded: Changing the internal container port (stays 8000). This would require changes to the uvicorn command, which already uses `${PORT:-8000}` correctly.
- Excluded: Any changes to the Python application logic (the server already reads `PORT` from the environment).

## Locked Code

No `@lock` tags were found in any file in this project. No locked code is affected.

## Affected Files

| File | Change Type | Description |
|------|-------------|-------------|
| docker-compose.yml | Modify | Change host port from `8000` to `${PORT:-8000}` in `ports` and healthcheck |
| .env.example | Modify | Add `PORT` variable with comment and default |
| Dockerfile | Modify | Update `EXPOSE 8000` to document that the host port is configurable via the compose variable |
| src/diffmem/server.py | Modify | Update `API_URL` default to use `PORT` env var instead of hardcoded `8000` |
| README.md | Modify | Update curl examples and descriptive text referencing `localhost:8000` to note the port is configurable |

## Dependencies and Connections

### docker-compose.yml
- **Imports/called by**: Coolify deployments, `docker compose up`
- **Calls/imports**: Dockerfile (build context), diffmem_data volume
- **Breaks if changed incorrectly**: Port mapping mismatch, healthcheck fails, users cannot reach the service

### .env.example
- **Imported by**: Users who `cp .env.example .env`
- **References in README**: README links to this file for the full config list
- **Breaks if omitted**: Users won't know PORT is a configurable variable

### Dockerfile
- **Imported into**: docker-compose.yml build context
- **Calls**: Python uvicorn server via CMD
- **Already uses**: `${PORT:-8000}` in CMD and HEALTHCHECK (already variable-aware)
- **Breaks if changed**: The internal container port would mismatch the compose port mapping

### src/diffmem/server.py
- **Called by**: Dockerfile CMD via uvicorn
- **Calls**: FastAPI, api.py, repo_manager.py, storage modules
- **Already reads**: `PORT` env var on line 561
- **Affected line**: `API_URL` default on line 25 hardcodes `http://localhost:8000`
- **Breaks if not updated**: Webhook callback URL would have wrong port if user changes PORT

### README.md
- **References**: curl examples with hardcoded `localhost:8000` (lines 100, 169, 173, 177)
- **Breaks if not updated**: Documentation would be inconsistent with the configurable port

## Prerequisites

- None. All changes are in existing files. No new dependencies, tools, or environment setup required.

## Implementation Steps

### Phase 1: docker-compose.yml — Port Mapping and Healthcheck

#### Step 1.1: Update the ports mapping
- File: `/Users/benjaminpowell/Desktop/Coding_projects/DiffMem/docker-compose.yml`
- Location: line 18
- Action: Change `"8000:8000"` to `"${PORT:-8000}:8000"`
- Reason: Makes the host port configurable via environment variable while keeping the container port fixed at 8000
- Affects: Any deployer who sets PORT in their environment or Coolify UI
- Verify: `docker compose config` should show the resolved port or the default 8000

#### Step 1.2: Update the healthcheck URL
- File: `/Users/benjaminpowell/Desktop/Coding_projects/DiffMem/docker-compose.yml`
- Location: line 44
- Action: The healthcheck currently uses the exec form `["CMD", "curl", "-fsS", "http://localhost:8000/health"]`. Since docker-compose variable substitution (`${PORT:-8000}`) operates at compose-parse time (not shell runtime), and the healthcheck runs inside the container where the compose variable is not available, this line can stay as-is. The container always listens on port 8000 internally. No change needed.
- Reason: The container's internal uvicorn server always binds to port 8000. The healthcheck curls `localhost:8000` inside the container, which never changes. The compose PORT variable only affects the host-side mapping.
- Affects: Nothing
- Verify: Healthcheck still passes on `docker compose up`

### Phase 2: .env.example — Document the PORT variable

#### Step 2.1: Add PORT to .env.example
- File: `/Users/benjaminpowell/Desktop/Coding_projects/DiffMem/.env.example`
- Location: Insert a new section before `REQUIRED` or after `ALLOWED_ORIGINS` (under OPTIONAL, as PORT has a default)
- Action: Add `# Host port to bind (default 8000). Set to change the external port.` followed by `PORT=`
- Reason: Users who copy .env.example will see PORT as a configurable option
- Affects: README config table reference
- Verify: The file renders cleanly with the new variable

### Phase 3: Dockerfile — Update EXPOSE comment

#### Step 3.1: Update EXPOSE directive
- File: `/Users/benjaminpowell/Desktop/Coding_projects/DiffMem/Dockerfile`
- Location: line 27
- Action: Change `EXPOSE 8000` to `EXPOSE 8000`. The `EXPOSE` instruction does not support variable expansion and is purely documentary. The line already correctly documents the container's internal port. No change needed.
- Reason: `EXPOSE` only documents which port the container listens on internally. It does not affect host port mapping. The internal port is always 8000.
- Affects: Nothing
- Verify: No change required

### Phase 4: server.py — Update API_URL default

#### Step 4.1: Derive API_URL default from PORT env var
- File: `/Users/benjaminpowell/Desktop/Coding_projects/DiffMem/src/diffmem/server.py`
- Location: line 25
- Action: Change `API_URL = os.getenv("API_URL", "http://localhost:8000")` to use the PORT env var in its default. Since `API_URL` is a separate config that users may explicitly set, the change should only affect the fallback default: read `PORT` from env, default to 8000, and construct the URL string dynamically.
- Reason: If someone changes PORT but not API_URL, the webhook callback URL would still point to port 8000, breaking post-commit GitHub backup webhooks.
- Affects: The `API_URL` value used for the post-commit webhook in the GitHub backup flow
- Verify: With no env vars set, `API_URL` should default to `http://localhost:8000`. With `PORT=9000`, it should default to `http://localhost:9000` (unless `API_URL` is explicitly overridden).

### Phase 5: README.md — Document the configurable port

#### Step 5.1: Update the config table
- File: `/Users/benjaminpowell/Desktop/Coding_projects/DiffMem/README.md`
- Location: The environment variables table (around line 125)
- Action: Add a new row for `PORT` with default `8000` and purpose "Host port to bind (maps to container port 8000)"
- Reason: The config table lists all available env vars; PORT should be included for discoverability
- Affects: Users reading the README to understand configuration options
- Verify: Table renders correctly with the new row

#### Step 5.2: Update curl examples
- File: `/Users/benjaminpowell/Desktop/Coding_projects/DiffMem/README.md`
- Location: lines 169, 173, 177 (the curl example block)
- Action: Optionally add a comment above the curl block noting that the port shown (8000) changes if PORT is configured differently. Or leave the examples as-is with 8000 as the default, since they are documentation examples showing the default. The latter approach is cleaner and less noisy.
- Decision: No change needed. The curl examples show the default port, which is correct. Users who change PORT will know their own port.

#### Step 5.3: Update plain Docker Compose section
- File: `/Users/benjaminpowell/Desktop/Coding_projects/DiffMem/README.md`
- Location: line 100
- Action: Change `The service listens on http://localhost:8000.` to `The service listens on http://localhost:8000 (configurable via the PORT environment variable).`
- Reason: Informs self-hosters that the port is adjustable
- Affects: Users reading the Docker Compose deployment section
- Verify: Text reads naturally

## Documentation Updates

No files in `/docs/` exist (the `/docs/` directory does not exist in this project). The README.md at the project root is the primary documentation and is covered in Phase 5 above.

## Testing

- Run `docker compose config` after the change and confirm the default port mapping resolves to `"8000:8000"`.
- Run `PORT=9000 docker compose config` and confirm the port mapping resolves to `"9000:8000"`.
- Run `docker compose up` with no PORT set and confirm the service is reachable at `http://localhost:8000/health` (200 OK).
- Run `PORT=9000 docker compose up` and confirm the service is reachable at `http://localhost:9000/health`.
- Verify that the server.py API_URL default correctly picks up the PORT env var without breaking the explicit override case (`API_URL` set directly).

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| User sets PORT to a non-numeric value | `docker compose up` fails with invalid port error | The `${PORT:-8000}` form will produce whatever string the user provides; docker-compose validates port format at parse time |
| Healthcheck fails because it still curls port 8000 inside the container | Service marked unhealthy | The healthcheck curls `localhost:8000` inside the container where uvicorn always binds to port 8000. This is correct and unaffected by the host-side PORT variable. |
| API_URL default logic change breaks explicit API_URL overrides | Webhook URLs wrong | The change in server.py must only affect the fallback default, not override an explicitly set API_URL. `os.getenv("API_URL")` is checked first. |
| Existing Coolify deployments break | Service unreachable | Coolify deployments that don't set PORT will use the default 8000 — no behavioral change. Existing deployments are unaffected. |

## Rollback

If the changes cause issues:

1. **Revert docker-compose.yml**: Change `"${PORT:-8000}:8000"` back to `"8000:8000"`.
2. **Revert .env.example**: Remove the PORT entry.
3. **Revert server.py**: Restore `API_URL = os.getenv("API_URL", "http://localhost:8000")`.
4. **Revert README.md**: Restore the original text on line 100.
5. Run `docker compose down && docker compose up -d` to restart with the old configuration.
