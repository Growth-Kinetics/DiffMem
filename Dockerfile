# DiffMem server -- single-image build for self-hosting.
# All state lives under /data, which should be a mounted volume.

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-server.txt ./
RUN pip install --no-cache-dir -r requirements-server.txt

COPY src/ ./src/
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Persistent state lives under /data. Mount a volume here.
RUN mkdir -p /data/storage /data/worktrees && \
    git config --global user.name "DiffMem" && \
    git config --global user.email "diffmem@localhost" && \
    git config --global credential.helper "" && \
    git config --global http.postBuffer 524288000 && \
    git config --global --add safe.directory '*'

EXPOSE 8000

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    STORAGE_PATH=/data/storage \
    WORKTREE_ROOT=/data/worktrees \
    PORT=8000

VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:${PORT:-8000}/health || exit 1

# tini as PID 1 so SIGTERM from Coolify/Docker triggers FastAPI's lifespan
# shutdown (which flushes final backups).
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sh", "-c", "uvicorn diffmem.server:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
