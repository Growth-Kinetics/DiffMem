# API Examples for DiffMem

## Base URLs

- **Local (Unraid)**: `http://192.168.60.108:8000`
- **Domain**: `https://difmem.kingbarry.cc`

## Authentication

If `REQUIRE_AUTH=true`, include header:
```bash
-H "Authorization: Bearer YOUR_API_KEY"
```

## 📝 Complete API Examples

### 1. Create a New User

```bash
curl -X POST "https://difmem.kingbarry.cc/memory/jon-morgenstern/onboard" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "user_info": "Jon Morgenstern is a paid media expert specializing in social media and search advertising.",
    "session_id": "onboard-001"
  }'
```

### 2. Check User Status

```bash
curl -X GET "https://difmem.kingbarry.cc/memory/jon-morgenstern/onboard-status" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### 3. Import Content (Process and Commit)

```bash
curl -X POST "https://difmem.kingbarry.cc/memory/jon-morgenstern/process-and-commit" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "memory_input": "Jon optimized a Facebook ad campaign today. Increased CTR by 15% by adjusting targeting parameters. The campaign is now performing above industry benchmarks.",
    "session_id": "session-001",
    "session_date": "2025-01-31"
  }'
```

### 4. Get Context for Conversation

```bash
curl -X POST "https://difmem.kingbarry.cc/memory/jon-morgenstern/context" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "conversation": [
      {"role": "user", "content": "What campaigns has Jon been working on?"}
    ],
    "depth": "deep"
  }'
```

**Depth Options:**
- `basic` - Top entities with ALWAYS_LOAD blocks (fastest)
- `wide` - Semantic search with ALWAYS_LOAD blocks
- `deep` - Complete entity files (comprehensive)
- `temporal` - Complete files with Git history (most detailed)

### 5. Search Memories

```bash
curl -X POST "https://difmem.kingbarry.cc/memory/jon-morgenstern/search" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "query": "Facebook ad campaign",
    "k": 5
  }'
```

### 6. LLM-Orchestrated Search

```bash
curl -X POST "https://difmem.kingbarry.cc/memory/jon-morgenstern/orchestrated-search" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "conversation": [
      {"role": "user", "content": "What did Jon work on last month?"}
    ]
  }'
```

### 7. Get User Entity

```bash
curl -X GET "https://difmem.kingbarry.cc/memory/jon-morgenstern/user-entity" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### 8. Get Recent Timeline

```bash
curl -X GET "https://difmem.kingbarry.cc/memory/jon-morgenstern/recent-timeline?days_back=30" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### 9. Get Repository Status

```bash
curl -X GET "https://difmem.kingbarry.cc/memory/jon-morgenstern/status" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### 10. List All Users

```bash
curl -X GET "https://difmem.kingbarry.cc/server/users" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### 11. Manual Sync

```bash
curl -X POST "https://difmem.kingbarry.cc/server/sync" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### 12. Health Check

```bash
curl -X GET "https://difmem.kingbarry.cc/health"
```

## 🔄 Two-Step Process (Process then Commit)

If you want to process and commit separately:

```bash
# Step 1: Process (stages changes)
curl -X POST "https://difmem.kingbarry.cc/memory/jon-morgenstern/process-session" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "memory_input": "Jon worked on Google Ads today...",
    "session_id": "session-002",
    "session_date": "2025-01-31"
  }'

# Step 2: Commit (commits staged changes)
curl -X POST "https://difmem.kingbarry.cc/memory/jon-morgenstern/commit-session" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "session_id": "session-002"
  }'
```

## 📊 Response Examples

### Successful Onboard
```json
{
  "status": "success",
  "message": "User jon-morgenstern successfully onboarded",
  "result": {
    "success": true,
    "user_id": "jon-morgenstern",
    "session_id": "onboard-001"
  }
}
```

### Successful Import
```json
{
  "status": "success",
  "session_id": "session-001",
  "message": "Session processed, committed, and synced to GitHub",
  "metadata": {
    "user_id": "jon-morgenstern",
    "timestamp": "2025-01-31T10:30:00"
  }
}
```

### Search Results
```json
{
  "status": "success",
  "results": [
    {
      "score": 0.85,
      "snippet": {
        "id": "facebook_campaign",
        "content": "Facebook ad campaign optimization...",
        "file_path": "memories/contexts/campaigns.md"
      }
    }
  ]
}
```

## 🐍 Python Example

```python
import requests

BASE_URL = "https://difmem.kingbarry.cc"
API_KEY = "your-api-key-here"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# Create user
response = requests.post(
    f"{BASE_URL}/memory/jon-morgenstern/onboard",
    headers=headers,
    json={
        "user_info": "Jon Morgenstern is a paid media expert...",
        "session_id": "onboard-001"
    }
)
print(response.json())

# Import content
response = requests.post(
    f"{BASE_URL}/memory/jon-morgenstern/process-and-commit",
    headers=headers,
    json={
        "memory_input": "Jon optimized Facebook ads today...",
        "session_id": "session-001",
        "session_date": "2025-01-31"
    }
)
print(response.json())
```

## 🔗 Interactive API Docs

Once deployed, visit:
- **Swagger UI**: `https://difmem.kingbarry.cc/docs`
- **ReDoc**: `https://difmem.kingbarry.cc/redoc`
