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

The retrieval agent explores the git repository to find relevant context. It reads the entity index, probes git history for temporal patterns, and returns targeted file sections and diffs.

```bash
curl -X POST "https://difmem.kingbarry.cc/memory/jon-morgenstern/context" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "conversation": [
      {"role": "user", "content": "What campaigns has Jon been working on?"}
    ],
    "max_tokens": 15000,
    "max_turns": 4,
    "timeout_seconds": 30
  }'
```

**Parameters:**
- `conversation` (required) - Conversation history as message dicts
- `max_tokens` (default: 20000) - Agent's additional context token budget
- `max_turns` (default: 4) - Max agent exploration turns
- `timeout_seconds` (default: 30) - Hard timeout for the agent loop

### 5. Get User Entity

```bash
curl -X GET "https://difmem.kingbarry.cc/memory/jon-morgenstern/user-entity" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### 6. Get Recent Timeline

```bash
curl -X GET "https://difmem.kingbarry.cc/memory/jon-morgenstern/recent-timeline?days_back=30" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### 7. Get Repository Status

```bash
curl -X GET "https://difmem.kingbarry.cc/memory/jon-morgenstern/status" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### 8. List All Users

```bash
curl -X GET "https://difmem.kingbarry.cc/server/users" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### 9. Manual Sync

```bash
curl -X POST "https://difmem.kingbarry.cc/server/sync" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### 10. Health Check

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

### Context Retrieval
```json
{
  "status": "success",
  "context": {
    "user_entity": {"source": "jon-morgenstern.md", "type": "user_entity", "content": "...", "tokens": 5000},
    "recent_timeline": [],
    "agent_context": [
      {"source": "git diff HEAD~3.. -- memories/contexts/campaigns.md", "type": "git_diff", "content": "...", "reason": "Recent campaign changes", "tokens": 400},
      {"source": "memories/contexts/campaigns.md:10-45", "type": "file_section", "content": "...", "reason": "Facebook campaign details", "tokens": 300}
    ],
    "always_load_blocks": [
      {"source": "memories/contexts/campaigns.md", "type": "always_load", "header": "Core Identity", "content": "...", "tokens": 150}
    ],
    "retrieval_plan": {
      "synthesis": "Agent found campaign entities with recent activity...",
      "entities_identified": ["campaigns", "facebook_ads"],
      "pointers": ["..."],
      "agent_turns": 4,
      "agent_elapsed_ms": 8500
    },
    "session_metadata": {
      "user_id": "jon-morgenstern",
      "retrieval_version": "agent",
      "max_tokens": 15000,
      "baseline_tokens": 5000,
      "agent_tokens": 700,
      "always_load_tokens": 150,
      "total_tokens": 5850,
      "agent_ms": 8500,
      "timestamp": "2026-03-17T10:30:00"
    }
  }
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

# Get context for a conversation
response = requests.post(
    f"{BASE_URL}/memory/jon-morgenstern/context",
    headers=headers,
    json={
        "conversation": [
            {"role": "user", "content": "What campaigns has Jon been working on?"}
        ],
        "max_tokens": 15000
    }
)
context = response.json()["context"]
print(f"Agent found {len(context['agent_context'])} blocks in {context['session_metadata']['agent_ms']}ms")
```

## 🔗 Interactive API Docs

Once deployed, visit:
- **Swagger UI**: `https://difmem.kingbarry.cc/docs`
- **ReDoc**: `https://difmem.kingbarry.cc/redoc`
