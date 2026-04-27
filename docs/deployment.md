# Deployment

DiffMem runs as a single FastAPI service inside Docker. Persistent state is
stored under `/data`, which should be mounted as a Docker volume.

## Ports

The application listens inside the container on port `8000`.

For plain Docker Compose, `HOST_PORT` controls the port exposed on the Docker
host:

```bash
HOST_PORT=8062 docker compose up -d
```

That example exposes the service at `http://localhost:8062` on the host while
the container still listens on `8000`.

Do not add `PORT` to the Compose environment for host binding. If you need to
change the internal app port, the compose port target and health check must be
changed to match.

## Coolify

Use the Docker Compose deployment type and point Coolify at
`docker-compose.yml`.

Set `OPENROUTER_API_KEY` in Coolify's environment variables. If you attach a
domain, configure the service to proxy to container port `8000`. Public traffic
still enters through Coolify's normal HTTP or HTTPS entrypoints.

## Health Check

The health endpoint is an unauthenticated liveness check implemented by
`health_check()` in `src/diffmem/server.py`.

It accepts no request body or query parameters and returns JSON with service
status, timestamp, version, active context count, storage backend, and backup
backend.

It is called by Docker Compose, Coolify, and operators checking whether the
service is reachable.

The endpoint is:

```text
GET /health
```

The Docker Compose health check calls:

```text
http://localhost:8000/health
```

This check runs inside the container network namespace, so it must target the
container port, not the host port.
