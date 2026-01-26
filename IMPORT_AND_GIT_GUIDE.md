# Importing Content & Git Syncing Guide

## 📥 How to Import Content

### The Good News: It's Just Plain Text!

The `memory_input` field accepts **any plain text** - conversation transcripts, notes, journal entries, etc. The system automatically:
- Extracts entities (people, places, topics)
- Creates/updates memory files
- Adds timeline entries
- Links everything together

### Input Format

**Simple**: Just a string of text. No special formatting required!

```json
{
  "memory_input": "Your text here...",
  "session_id": "unique-session-id",
  "session_date": "2025-01-31"  // Optional, defaults to today
}
```

## 📝 Examples of Content You Can Import

### Example 1: Conversation Transcript
```json
{
  "memory_input": "User: I had coffee with Sarah today. She mentioned she's starting a new job at Google next month. We talked about her upcoming trip to Japan. She's really excited about it.\n\nAssistant: That sounds great! When is she going?\n\nUser: In March. She's been planning it for months.",
  "session_id": "chat-2025-01-31-001",
  "session_date": "2025-01-31"
}
```

### Example 2: Journal Entry / Notes
```json
{
  "memory_input": "Today I worked on the DiffMem project. Learned about Git-based memory systems. Had a meeting with the team about next quarter's goals. Feeling productive but a bit tired.",
  "session_id": "journal-2025-01-31",
  "session_date": "2025-01-31"
}
```

### Example 3: Meeting Notes
```json
{
  "memory_input": "Team meeting with Alex, Maria, and John. Discussed Q2 roadmap. Alex is leading the new feature development. Maria will handle testing. John mentioned he's taking vacation next week.",
  "session_id": "meeting-2025-01-31",
  "session_date": "2025-01-31"
}
```

### Example 4: Simple Facts
```json
{
  "memory_input": "My favorite programming language is Python. I've been using it for 5 years. I prefer FastAPI for web development.",
  "session_id": "facts-001",
  "session_date": "2025-01-31"
}
```

## 🔄 The Import Process

When you call `/process-and-commit`, here's what happens:

1. **Text Analysis**: LLM analyzes your text to find:
   - People mentioned (creates/updates `memories/people/{name}.md`)
   - Topics/themes (creates/updates `memories/contexts/{topic}.md`)
   - Events (adds to `timeline/YYYY-MM.md`)

2. **File Updates**: System creates or updates Markdown files:
   ```
   benjamin.md                    # Your profile (if about you)
   memories/people/sarah.md       # Sarah's profile
   memories/contexts/work.md     # Work-related context
   timeline/2025-01.md           # Timeline entry
   ```

3. **Git Commit**: Changes are committed with message like:
   ```
   Session chat-2025-01-31-001: Updated entities and timeline
   ```

4. **Auto-Sync to GitHub**: Server automatically pushes to your GitHub repo!

## 🚀 Complete Import Workflow

### Step 1: Onboard (First Time Only)
```bash
curl -X POST "http://localhost:8000/memory/benjamin/onboard" \
  -H "Content-Type: application/json" \
  -d '{
    "user_info": "Benjamin is a developer who likes building AI systems and working with Python.",
    "session_id": "onboard-001"
  }'
```

### Step 2: Import Content
```bash
curl -X POST "http://localhost:8000/memory/benjamin/process-and-commit" \
  -H "Content-Type: application/json" \
  -d '{
    "memory_input": "Today I had coffee with Sarah. She mentioned starting a new job at Google next month.",
    "session_id": "session-001",
    "session_date": "2025-01-31"
  }'
```

**Response:**
```json
{
  "status": "success",
  "session_id": "session-001",
  "message": "Session processed, committed, and synced to GitHub",
  "metadata": {
    "user_id": "benjamin",
    "timestamp": "2025-01-31T10:30:00"
  }
}
```

### Step 3: Verify It Worked

Check your local files:
```bash
# See what was created
ls -la worktrees/benjamin/memories/people/
ls -la worktrees/benjamin/timeline/
```

Check GitHub:
```bash
# Visit your GitHub repo
# You should see a new branch: user/benjamin
# With all the memory files!
```

## 🔀 Git Syncing - How It Works

### Automatic Syncing

**You don't need to manually push!** The server handles everything:

1. **On Commit**: When you call `/process-and-commit` or `/commit-session`:
   - Changes are committed locally
   - Server automatically pushes to GitHub
   - Your branch `user/benjamin` is updated

2. **Periodic Sync**: Every 5 minutes (configurable), the server:
   - Checks for uncommitted changes
   - Commits and pushes them

3. **On Startup**: Server fetches latest from GitHub to sync any remote changes

### Git Architecture

```
GitHub Repository (your repo)
├── user/benjamin (orphan branch) ← Your memories here!
├── user/alice (orphan branch)    ← Other users isolated
└── user/bob (orphan branch)      ← Each user has own branch

Local Storage
├── storage/                      ← Central Git repo (all branches)
└── worktrees/benjamin/          ← Your active worktree
    ├── benjamin.md
    ├── memories/
    └── timeline/
```

### Manual Sync (If Needed)

You can manually trigger a sync:
```bash
curl -X POST "http://localhost:8000/server/sync" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## 📂 What Gets Created

After importing content, you'll see:

### Directory Structure
```
worktrees/benjamin/
├── benjamin.md                    # Your profile
├── index.md                       # Auto-generated keyword index
├── memories/
│   ├── people/
│   │   ├── sarah.md              # Created from "Sarah" mention
│   │   └── alex.md               # Created from "Alex" mention
│   └── contexts/
│       ├── work.md               # Work-related topics
│       └── travel.md             # Travel-related topics
├── timeline/
│   └── 2025-01.md                # January timeline entries
└── sessions/
    └── 2025-01-31_session-001.txt # Raw session archive
```

### Example: What Sarah's File Looks Like
```markdown
# Sarah Profile (Biographical Core) [Strength: Medium]

## Core Identity [ALWAYS_LOAD]
- Essential markers: Friend, mentioned in conversation
- Key traits: Starting new job at Google

### ❤️ Relationship Dynamics [Strength: High]
• With Benjamin: Friends → mentioned in conversation
  ↳ Connection Quality: Friendly relationship
  ↳ Interaction Pattern: Coffee meetings, conversations
/END ❤️ Relationship Dynamics

### Work & Career
• New Job: Starting at Google (mentioned 2025-01-31)
  ↳ Evolution: Recently mentioned new position
/END Work & Career
```

## 🔍 Verifying Git Sync

### Check Local Git Status
```bash
cd worktrees/benjamin
git log --oneline -5
# Should show recent commits like:
# abc123 Session session-001: Updated entities and timeline
```

### Check GitHub
1. Go to your GitHub repo: `https://github.com/kingfisherfox/diffmem`
2. Click "Branches" dropdown
3. Look for `user/benjamin` branch
4. Browse the files - you should see all your memories!

### Check via API
```bash
curl "http://localhost:8000/memory/benjamin/status"
```

## 📊 Batch Importing

### Import Multiple Sessions

You can import multiple pieces of content:

```bash
# Session 1
curl -X POST "http://localhost:8000/memory/benjamin/process-and-commit" \
  -H "Content-Type: application/json" \
  -d '{
    "memory_input": "Had coffee with Sarah today...",
    "session_id": "session-001",
    "session_date": "2025-01-31"
  }'

# Session 2
curl -X POST "http://localhost:8000/memory/benjamin/process-and-commit" \
  -H "Content-Type: application/json" \
  -d '{
    "memory_input": "Worked on the DiffMem project. Made good progress.",
    "session_id": "session-002",
    "session_date": "2025-01-31"
  }'

# Session 3
curl -X POST "http://localhost:8000/memory/benjamin/process-and-commit" \
  -H "Content-Type: application/json" \
  -d '{
    "memory_input": "Team meeting with Alex and Maria. Discussed roadmap.",
    "session_id": "session-003",
    "session_date": "2025-01-31"
  }'
```

Each session will:
- Be processed independently
- Update/create relevant entities
- Create a timeline entry
- Commit and sync to GitHub

## 🎯 Best Practices

### 1. Use Descriptive Session IDs
```json
{
  "session_id": "chat-2025-01-31-morning-001"  // Good: descriptive
}
```
Not:
```json
{
  "session_id": "001"  // Bad: not unique/descriptive
}
```

### 2. Include Dates
Always include `session_date` for accurate timeline:
```json
{
  "session_date": "2025-01-31"  // Format: YYYY-MM-DD
}
```

### 3. Natural Language
Write naturally - the LLM will extract entities:
```json
{
  "memory_input": "I talked to my friend Sarah about her new job. She's excited about working at Google."
}
```

### 4. Regular Imports
Import content regularly to build up your memory:
- Daily journal entries
- Meeting notes
- Conversation summaries
- Important events

## 🐛 Troubleshooting

### Content Not Appearing in GitHub
1. Check server logs for sync errors
2. Verify `GITHUB_TOKEN` is valid
3. Check `GITHUB_REPO_URL` is correct
4. Manually trigger sync: `POST /server/sync`

### Entities Not Being Created
- The LLM might not detect entities in very short text
- Try including more context
- Check server logs for LLM errors

### Git Errors
- Ensure Git is installed: `git --version`
- Check storage directory permissions
- Verify GitHub token has `repo` scope

## 📚 Next Steps

1. **Start Importing**: Use `/process-and-commit` to add your first memories
2. **Check Files**: Look in `worktrees/benjamin/` to see what was created
3. **Check GitHub**: Visit your repo to see the `user/benjamin` branch
4. **Query Memories**: Use `/context` or `/search` to retrieve information
5. **Build Up Memory**: Keep importing content regularly!

## 💡 Pro Tips

- **Session IDs**: Use UUIDs or timestamps for uniqueness
- **Archive Raw Sessions**: The system saves raw input in `sessions/` folder
- **Incremental Updates**: Same entities get updated, not duplicated
- **Timeline**: All events go into monthly timeline files
- **Search**: Use `/search` to find specific memories later
