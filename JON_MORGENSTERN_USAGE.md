# Using Jon Morgenstern's Memory System

## ✅ User Created

**User ID**: `jon-morgenstern`  
**Profile**: Paid media expert for social and search advertising

## 📝 How to Use

### Import Content for Jon

```bash
curl -X POST "http://localhost:8000/memory/jon-morgenstern/process-and-commit" \
  -H "Content-Type: application/json" \
  -d '{
    "memory_input": "Jon worked on optimizing a Facebook ad campaign for a client today. He increased the CTR by 15% by adjusting the targeting parameters. The campaign is now performing above industry benchmarks.",
    "session_id": "jon-session-001",
    "session_date": "2025-01-31"
  }'
```

### Query Jon's Memories

```bash
curl -X POST "http://localhost:8000/memory/jon-morgenstern/context" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation": [{"role": "user", "content": "What campaigns has Jon been working on?"}],
    "depth": "deep"
  }'
```

### Search Jon's Memories

```bash
curl -X POST "http://localhost:8000/memory/jon-morgenstern/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Facebook ad campaign",
    "k": 5
  }'
```

## 🎯 Example Use Cases

### Track Campaign Performance

```bash
curl -X POST "http://localhost:8000/memory/jon-morgenstern/process-and-commit" \
  -H "Content-Type: application/json" \
  -d '{
    "memory_input": "Google Ads campaign for Q1 2025 is performing well. CPC decreased by 20%, conversion rate increased by 12%. Budget is on track.",
    "session_id": "jon-campaign-001"
  }'
```

### Store Client Information

```bash
curl -X POST "http://localhost:8000/memory/jon-morgenstern/process-and-commit" \
  -H "Content-Type: application/json" \
  -d '{
    "memory_input": "Client ABC Corp wants to expand their Instagram presence. They have a budget of $50k/month and target B2B professionals aged 30-45.",
    "session_id": "jon-client-abc"
  }'
```

### Track Learning & Insights

```bash
curl -X POST "http://localhost:8000/memory/jon-morgenstern/process-and-commit" \
  -H "Content-Type: application/json" \
  -d '{
    "memory_input": "Discovered that video ads perform 3x better than static images for the tech industry. Will apply this insight to future campaigns.",
    "session_id": "jon-insight-001"
  }'
```

## 📂 Where to Find Jon's Data

**Local**: `worktrees/jon-morgenstern/`  
**GitHub**: https://github.com/kingfisherfox/diffmem/tree/user/jon-morgenstern

## 🔒 Isolation

Jon's memories are **completely isolated** from:
- Your personal memories (`benjamin`)
- Any other users
- Work memories (if you create a `work` user)

Each user has their own branch and data structure.

## 📊 Check Status

```bash
curl "http://localhost:8000/memory/jon-morgenstern/status"
```

## 🎯 Remember

- **User ID**: Always use `jon-morgenstern` in the URL
- **Isolation**: Jon's data is completely separate from yours
- **Auto-Sync**: All changes automatically sync to GitHub
