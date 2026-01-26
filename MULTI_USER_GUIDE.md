# Multi-User Guide: Creating and Managing Multiple Users

## 🎯 How User Isolation Works

DiffMem uses **orphan branches** in Git to ensure **complete isolation** between users:

- **Each user gets their own branch**: `user/{user_id}`
- **No shared history**: Orphan branches have no common ancestor
- **No shared files**: Each user's data is completely separate
- **Your personal memories stay separate**: `user/benjamin` is isolated from all other users

### Architecture

```
GitHub Repository
├── user/benjamin    ← Your personal memories (isolated)
├── user/alice      ← Alice's memories (isolated)
├── user/bob        ← Bob's memories (isolated)
└── user/client-1   ← Client memories (isolated)
```

Each branch is **completely independent** - they share nothing!

## 🚀 Creating a New User

### Step 1: Onboard the New User

Simply call the onboard endpoint with a **different `user_id`**:

```bash
curl -X POST "http://localhost:8000/memory/alice/onboard" \
  -H "Content-Type: application/json" \
  -d '{
    "user_info": "Alice is a software engineer who works on frontend development. She loves React and TypeScript.",
    "session_id": "onboard-alice-001"
  }'
```

**That's it!** The system will:
- ✅ Create a new orphan branch: `user/alice`
- ✅ Create isolated worktree directory
- ✅ Set up Alice's memory structure
- ✅ Push to GitHub automatically

### Step 2: Use That User ID for All Operations

All API calls use the `user_id` in the URL path:

```bash
# Import content for Alice
curl -X POST "http://localhost:8000/memory/alice/process-and-commit" \
  -H "Content-Type: application/json" \
  -d '{
    "memory_input": "Alice worked on a React component today. She fixed a bug in the user authentication flow.",
    "session_id": "alice-session-001"
  }'

# Query Alice's memories
curl -X POST "http://localhost:8000/memory/alice/context" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation": [{"role": "user", "content": "What did Alice work on?"}],
    "depth": "deep"
  }'
```

## 📝 Examples: Different Use Cases

### Example 1: Personal vs Work Memories

**Your Personal Memories:**
```bash
# Use your personal user ID
curl -X POST "http://localhost:8000/memory/benjamin/process-and-commit" \
  -H "Content-Type: application/json" \
  -d '{
    "memory_input": "Had coffee with Sarah today...",
    "session_id": "personal-001"
  }'
```

**Work/Client Memories:**
```bash
# Create a work user ID
curl -X POST "http://localhost:8000/memory/work/onboard" \
  -H "Content-Type: application/json" \
  -d '{
    "user_info": "Work memories and professional context",
    "session_id": "onboard-work"
  }'

# Then use it for work-related content
curl -X POST "http://localhost:8000/memory/work/process-and-commit" \
  -H "Content-Type: application/json" \
  -d '{
    "memory_input": "Team meeting about Q2 roadmap. Discussed new features.",
    "session_id": "work-001"
  }'
```

### Example 2: Multiple Clients/Projects

```bash
# Client 1
curl -X POST "http://localhost:8000/memory/client-acme/onboard" \
  -H "Content-Type: application/json" \
  -d '{"user_info": "ACME Corp client project", "session_id": "onboard-acme"}'

# Client 2
curl -X POST "http://localhost:8000/memory/client-beta/onboard" \
  -H "Content-Type: application/json" \
  -d '{"user_info": "Beta Inc client project", "session_id": "onboard-beta"}'

# Each client's data is completely isolated!
```

### Example 3: Different AI Assistants

```bash
# Assistant for coding
curl -X POST "http://localhost:8000/memory/coding-assistant/onboard" \
  -H "Content-Type: application/json" \
  -d '{"user_info": "Coding assistant context", "session_id": "onboard-coding"}'

# Assistant for writing
curl -X POST "http://localhost:8000/memory/writing-assistant/onboard" \
  -H "Content-Type: application/json" \
  -d '{"user_info": "Writing assistant context", "session_id": "onboard-writing"}'
```

## 🔍 Verifying User Isolation

### Check Local Directories

Each user has their own worktree:
```bash
ls worktrees/
# Should show:
# benjamin/
# alice/
# bob/
```

### Check GitHub Branches

Visit your GitHub repo and check branches:
```bash
# List all user branches
curl -s https://api.github.com/repos/kingfisherfox/diffmem/branches | grep "user/"
```

Or on GitHub:
1. Go to your repo
2. Click "Branches"
3. You'll see: `user/benjamin`, `user/alice`, etc.

### Check via API

```bash
# List all active users
curl "http://localhost:8000/server/users"
```

## 📂 Directory Structure Per User

Each user has their own isolated structure:

```
worktrees/benjamin/          ← Your personal memories
├── benjamin.md
├── memories/
│   ├── people/
│   └── contexts/
└── timeline/

worktrees/alice/             ← Alice's memories (completely separate)
├── alice.md
├── memories/
│   ├── people/
│   └── contexts/
└── timeline/
```

**No cross-contamination!** Each user's data is completely isolated.

## 🎯 Best Practices

### 1. Use Descriptive User IDs

```bash
# Good: Descriptive
user/benjamin-personal
user/work-client-acme
user/assistant-coding

# Bad: Unclear
user/user1
user/test
user/abc
```

### 2. Keep User IDs Consistent

Once you create a user, always use the same `user_id`:
- ✅ `benjamin` for your personal memories
- ✅ `work` for work-related content
- ✅ `client-acme` for ACME client

### 3. Organize by Purpose

```bash
# Personal
user/benjamin

# Work/Professional
user/work
user/professional

# Clients
user/client-acme
user/client-beta

# Projects
user/project-diffmem
user/project-other
```

### 4. Check Before Creating

```bash
# Check if user exists
curl "http://localhost:8000/memory/alice/onboard-status"
```

## 🔄 Switching Between Users

Just change the `user_id` in the URL:

```bash
# Work with benjamin
curl -X POST "http://localhost:8000/memory/benjamin/process-and-commit" ...

# Switch to alice
curl -X POST "http://localhost:8000/memory/alice/process-and-commit" ...

# Switch to work
curl -X POST "http://localhost:8000/memory/work/process-and-commit" ...
```

The server handles everything - no manual switching needed!

## 🗑️ Removing a User

If you need to remove a user's data:

```bash
# This would need to be implemented, but for now:
# 1. Delete the worktree
rm -rf worktrees/alice

# 2. Delete the branch (via Git)
cd storage
git branch -D user/alice
git push origin --delete user/alice
```

**⚠️ Warning**: This permanently deletes the user's data!

## 📊 Managing Multiple Users

### List All Users

```bash
curl "http://localhost:8000/server/users"
```

Response:
```json
{
  "status": "success",
  "users": ["benjamin", "alice", "work"],
  "count": 3
}
```

### Check User Status

```bash
curl "http://localhost:8000/memory/alice/status"
```

### Sync All Users

```bash
curl -X POST "http://localhost:8000/server/sync"
```

## 💡 Common Patterns

### Pattern 1: Personal + Work Separation

```bash
# Personal user
user/benjamin

# Work user  
user/work
```

### Pattern 2: Multiple Clients

```bash
user/client-1
user/client-2
user/client-3
```

### Pattern 3: Different Contexts

```bash
user/personal
user/professional
user/learning
user/projects
```

## ✅ Summary

1. **Create a user**: Call `/memory/{user_id}/onboard` with any `user_id`
2. **Complete isolation**: Each user gets their own orphan branch
3. **No mixing**: Your personal memories (`benjamin`) stay separate
4. **Use user_id**: All operations use the `user_id` in the URL
5. **Automatic sync**: Each user's data syncs to their own branch on GitHub

**Your personal memories are safe and isolated!** 🎉
