# DiffMem Server - Cloud-ready FastAPI deployment
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements-server.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-server.txt

# Copy source code
COPY src/ ./src/
COPY pyproject.toml ./

# Install the package in development mode
RUN pip install -e .

# Create directory for repository
RUN mkdir -p /app/memory_repo

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables
ENV PYTHONPATH=/app/src
ENV REPO_PATH=/app/memory_repo
ENV SYNC_INTERVAL_MINUTES=5

# Run the server
CMD ["uvicorn", "diffmem.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"] 