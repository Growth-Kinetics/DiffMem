# DiffMem FastAPI Server

A cloud-ready FastAPI server for DiffMem differential memory operations with GitHub integration.

## Features

- **GitHub Integration**: Direct clone/sync with GitHub repositories
- **JWT Authentication**: Secure token-based authentication
- **Async Operations**: High-performance async endpoints
- **Auto-sync**: Background repository synchronization
- **Docker Ready**: Container-based deployment
- **Cloud Run Compatible**: Optimized for Google Cloud Run

## Quick Start

### Local Development

1. **Install Dependencies**
   ```bash
   pip install -r requirements-server.txt
   ```

2. **Set Environment Variables**
   ```bash
   export OPENROUTER_API_KEY="your-openrouter-api-key"
   export JWT_SECRET="your-jwt-secret"
   export GITHUB_TOKEN="your-github-token"  # For client usage
   ```

3. **Run Server**
   ```bash
   uvicorn diffmem.server:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Access API Documentation**
   - OpenAPI docs: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

### Docker Deployment

1. **Build Image**
   ```bash
   docker build -t diffmem-server .
   ```

2. **Run Container**
   ```bash
   docker run -p 8000:8000 \
     -e OPENROUTER_API_KEY="your-key" \
     -e JWT_SECRET="your-secret" \
     diffmem-server
   ```

### Docker Compose

1. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Start Services**
   ```bash
   docker-compose up -d
   ```

## API Endpoints

### Authentication

#### `POST /auth/github`
Authenticate using GitHub token and receive JWT access token.

**Parameters:**
- `github_token` (query): GitHub personal access token

**Response:**
```json
{
  "access_token": "jwt-token-here",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Repository Management

#### `POST /repos/setup`
Setup and clone GitHub repository for memory operations.

**Headers:**
- `Authorization: Bearer {jwt_token}`

**Body:**
```json
{
  "repo_url": "https://github.com/username/memory-repo",
  "branch": "main",
  "user_id": "alex"
}
```

#### `GET /repos/status/{user_id}`
Get repository status and statistics.

**Headers:**
- `Authorization: Bearer {jwt_token}`

**Response:**
```json
{
  "repo_url": "https://github.com/username/memory-repo",
  "user_id": "alex",
  "last_sync": "2024-01-15T10:30:00",
  "status": "active",
  "memory_files_count": 15,
  "index_stats": {
    "total_blocks": 234,
    "avg_tokens": 45.2
  }
}
```

### Memory Operations

#### `POST /memory/context`
Get assembled context for a conversation.

**Body:**
```json
{
  "conversation": [
    {"role": "user", "content": "How am I doing with my goals?"}
  ],
  "depth": "basic",
  "user_id": "alex"
}
```

**Depth Options:**
- `basic`: Core entities with ALWAYS_LOAD blocks
- `wide`: Semantic search with ALWAYS_LOAD blocks
- `deep`: Complete entity files
- `temporal`: Complete files with Git history

#### `POST /memory/search`
Search memory using BM25 indexing.

**Body:**
```json
{
  "query": "health fitness goals",
  "k": 5,
  "user_id": "alex"
}
```

#### `POST /memory/orchestrated-search`
LLM-guided search from conversation context.

**Body:**
```json
{
  "conversation": [
    {"role": "user", "content": "Tell me about my relationship progress"}
  ],
  "user_id": "alex"
}
```

#### `POST /memory/process-session`
Process session transcript and stage changes (doesn't commit).

**Body:**
```json
{
  "memory_input": "Had coffee with mom today. Great conversation about family.",
  "session_id": "session-2024-01-15-001",
  "session_date": "2024-01-15",
  "user_id": "alex"
}
```

#### `POST /memory/commit-session`
Commit staged changes for a session.

**Body:**
```json
{
  "session_id": "session-2024-01-15-001",
  "user_id": "alex"
}
```

#### `POST /memory/process-and-commit`
Process and immediately commit a session (convenience method).

**Body:** Same as `process-session`

### Utility

#### `GET /health`
Server health check.

#### `POST /memory/rebuild-index`
Force rebuild of BM25 search index.

## Client Usage Example

```python
import asyncio
from examples.server_client import DiffMemClient

async def main():
    client = DiffMemClient("http://localhost:8000")
    
    # Authenticate
    await client.authenticate_github("your-github-token")
    
    # Setup repository
    await client.setup_repository(
        "https://github.com/username/memory-repo",
        "alex"
    )
    
    # Get context for conversation
    conversation = [
        {"role": "user", "content": "How are my health goals?"}
    ]
    context = await client.get_context(conversation, "alex", depth="basic")
    
    # Process new memory
    await client.process_and_commit_session(
        "Went for a run today, feeling great!",
        "session-001",
        "alex"
    )

asyncio.run(main())
```

## Cloud Deployment

### Google Cloud Run

1. **Build and Push Image**
   ```bash
   # Configure gcloud
   gcloud auth configure-docker
   
   # Build and tag
   docker build -t gcr.io/YOUR_PROJECT/diffmem-server .
   docker push gcr.io/YOUR_PROJECT/diffmem-server
   ```

2. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy diffmem-server \
     --image gcr.io/YOUR_PROJECT/diffmem-server \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 1Gi \
     --cpu 1 \
     --timeout 300 \
     --set-env-vars OPENROUTER_API_KEY="your-key",JWT_SECRET="your-secret"
   ```

3. **Configure Custom Domain** (Optional)
   ```bash
   gcloud run domain-mappings create \
     --service diffmem-server \
     --domain api.yourdomain.com \
     --region us-central1
   ```

### AWS ECS/Fargate

1. **Create Task Definition**
   ```json
   {
     "family": "diffmem-server",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "512",
     "memory": "1024",
     "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
     "containerDefinitions": [
       {
         "name": "diffmem-server",
         "image": "your-account.dkr.ecr.region.amazonaws.com/diffmem-server:latest",
         "portMappings": [{"containerPort": 8000}],
         "environment": [
           {"name": "OPENROUTER_API_KEY", "value": "your-key"},
           {"name": "JWT_SECRET", "value": "your-secret"}
         ]
       }
     ]
   }
   ```

2. **Create Service with Load Balancer**

### Azure Container Instances

```bash
az container create \
  --resource-group myResourceGroup \
  --name diffmem-server \
  --image your-registry/diffmem-server:latest \
  --cpu 1 \
  --memory 1 \
  --restart-policy Always \
  --ports 8000 \
  --environment-variables \
    OPENROUTER_API_KEY="your-key" \
    JWT_SECRET="your-secret"
```

## Environment Variables

### Required
- `OPENROUTER_API_KEY`: Your OpenRouter API key for LLM operations
- `JWT_SECRET`: Secret key for JWT token signing (change in production)

### Optional
- `REPO_CACHE_DIR`: Directory for repository cache (default: `/tmp/diffmem_repos`)
- `MAX_REPO_SIZE_MB`: Maximum repository size in MB (default: `100`)
- `SYNC_INTERVAL_MINUTES`: Background sync interval (default: `5`)
- `LOG_LEVEL`: Logging level (default: `info`)

## Security Considerations

### Production Deployment

1. **Change JWT Secret**
   ```bash
   export JWT_SECRET=$(openssl rand -base64 32)
   ```

2. **Use HTTPS Only**
   - Configure reverse proxy (nginx/traefik)
   - Enable SSL/TLS certificates
   - Set secure headers

3. **GitHub Token Permissions**
   - Use fine-grained personal access tokens
   - Limit to specific repositories
   - Rotate tokens regularly

4. **Network Security**
   - Use private networks where possible
   - Configure firewall rules
   - Enable VPC/subnet isolation

5. **Resource Limits**
   - Set memory/CPU limits
   - Configure request timeouts
   - Implement rate limiting

### Repository Access

- Each user's GitHub token is used for their repositories
- Server never stores GitHub tokens persistently
- Repository cache is isolated per user
- Automatic cleanup on server shutdown

## Monitoring and Observability

### Health Checks
- `/health` endpoint for basic health monitoring
- Container health checks in Docker
- Kubernetes liveness/readiness probes

### Logging
- Structured JSON logging
- Request/response correlation IDs
- Performance metrics logging
- Error tracking and alerting

### Metrics
- Repository sync status
- Memory operation latency
- GitHub API rate limits
- Server resource usage

## Troubleshooting

### Common Issues

1. **Authentication Failures**
   - Verify GitHub token has correct permissions
   - Check JWT secret is set correctly
   - Ensure token hasn't expired

2. **Repository Access Issues**
   - Verify repository exists and is accessible
   - Check GitHub token permissions
   - Ensure repository follows DiffMem structure

3. **Memory Operation Timeouts**
   - Increase request timeouts
   - Check OpenRouter API connectivity
   - Verify repository size limits

4. **Sync Issues**
   - Check GitHub API rate limits
   - Verify network connectivity
   - Review repository write permissions

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=debug
uvicorn diffmem.server:app --reload --log-level debug
```

### Performance Tuning

1. **Memory Usage**
   - Monitor BM25 index size
   - Implement index caching strategies
   - Use memory-mapped files for large repos

2. **Concurrency**
   - Adjust worker count for production
   - Use async database connections
   - Implement connection pooling

3. **Caching**
   - Cache frequently accessed contexts
   - Implement Redis for session storage
   - Use CDN for static content

## Contributing

1. Follow the agentic development principles in `agentic_dev_guide.md`
2. Maintain cognitive compartmentalization
3. Use structured logging for LLM feedback
4. Test all async endpoints thoroughly
5. Update documentation for API changes

## License

MIT License - see LICENSE file for details. 