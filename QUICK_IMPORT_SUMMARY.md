# Quick Import & Git Summary

## 📥 Importing Content - Super Simple!

**Just send plain text** - that's it! No special format needed.

```bash
curl -X POST "http://localhost:8000/memory/benjamin/process-and-commit" \
  -H "Content-Type: application/json" \
  -d '{
    "memory_input": "I had coffee with Sarah today. She mentioned starting a new job at Google.",
    "session_id": "session-001",
    "session_date": "2025-01-31"
  }'
```

**What happens:**
1. ✅ LLM extracts: People (Sarah), Topics (work), Events (coffee meeting)
2. ✅ Creates/updates files: `memories/people/sarah.md`, `timeline/2025-01.md`
3. ✅ Commits to Git: "Session session-001: Updated entities and timeline"
4. ✅ **Auto-pushes to GitHub** - you don't need to do anything!

## 🔀 Git Syncing - Automatic!

**You don't push manually!** The server does it automatically:

```
Your Import
    ↓
Process & Commit (local)
    ↓
Auto-Push to GitHub ← Happens automatically!
    ↓
Your GitHub Repo
    └── user/benjamin branch
        ├── benjamin.md
        ├── memories/
        └── timeline/
```

**When syncing happens:**
- ✅ Immediately after `/process-and-commit`
- ✅ Every 5 minutes (periodic sync)
- ✅ On server startup (fetches latest)

## 📝 Content Examples

### Conversation
```json
{
  "memory_input": "User: I talked to Sarah today. Assistant: What about? User: Her new job at Google.",
  "session_id": "chat-001"
}
```

### Notes
```json
{
  "memory_input": "Worked on DiffMem project. Learned about Git-based memory systems.",
  "session_id": "notes-001"
}
```

### Journal Entry
```json
{
  "memory_input": "Today was productive. Had coffee with Sarah, worked on DiffMem, feeling good.",
  "session_id": "journal-001"
}
```

## ✅ Verify It Worked

### Check Local Files
```bash
ls worktrees/benjamin/memories/people/
# Should see: sarah.md (if you mentioned Sarah)
```

### Check GitHub
1. Go to: `https://github.com/kingfisherfox/diffmem`
2. Click "Branches" → Look for `user/benjamin`
3. Browse files - your memories are there!

### Check via API
```bash
curl "http://localhost:8000/memory/benjamin/status"
```

## 🎯 That's It!

- **Import**: Just send text via API
- **Git**: Automatic - no manual pushing needed
- **Files**: Created automatically in organized structure
- **GitHub**: Synced automatically to your repo

See `IMPORT_AND_GIT_GUIDE.md` for detailed examples!
