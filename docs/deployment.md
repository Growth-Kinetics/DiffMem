# Deployment

DiffMem runs as a single FastAPI service inside Docker. Persistent state is
stored under `/data`, which should be mounted as a Docker volume.

## Port

`PORT` controls the port the app listens on inside the container and the port
published on the Docker host.

For example:

```bash
PORT=8062 docker compose up -d
```

That example makes Uvicorn listen on container port `8062`, publishes host port
`8062`, and checks health on `http://localhost:8062/health` inside the
container.

## Coolify

Use the Docker Compose deployment type and point Coolify at
`docker-compose.yml`.

Set `OPENROUTER_API_KEY` in Coolify's environment variables. If you attach a
domain, set `PORT` to the port you want the container to listen on and configure
the service to proxy to that same container port. Public traffic still enters
through Coolify's normal HTTP or HTTPS entrypoints.

## Models

`DEFAULT_MODEL` is required. It is the shared OpenRouter model for writer,
onboarding, and retrieval agents. Any OpenRouter model slug can be used.

Leave `RETRIEVAL_MODEL` empty to use `DEFAULT_MODEL` for retrieval. Set
`RETRIEVAL_MODEL` only when retrieval should use a different model from writer
and onboarding.

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

The Docker Compose health check calls the selected `PORT`:

```text
http://localhost:${PORT:-8000}/health
```

This check runs inside the container network namespace, so it must target the
container port, not the host port.
