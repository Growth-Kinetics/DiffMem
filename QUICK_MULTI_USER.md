# Quick Multi-User Reference

## 🎯 Key Point: Complete Isolation

**Each user is completely isolated:**
- Your personal memories (`benjamin`) stay separate
- Each user gets their own Git branch: `user/{user_id}`
- No data mixing between users
- Each user has their own worktree directory

## 🚀 Create a New User (2 Steps)

### Step 1: Onboard
```bash
curl -X POST "http://localhost:8000/memory/{user_id}/onboard" \
  -H "Content-Type: application/json" \
  -d '{
    "user_info": "Description of the user/context",
    "session_id": "onboard-001"
  }'
```

### Step 2: Use That User ID
```bash
# All operations use the user_id in the URL
curl -X POST "http://localhost:8000/memory/{user_id}/process-and-commit" ...
curl -X POST "http://localhost:8000/memory/{user_id}/context" ...
```

## 📝 Examples

### Personal vs Work
```bash
# Your personal memories
/memory/benjamin/...

# Work memories (separate!)
/memory/work/onboard
/memory/work/process-and-commit
```

### Multiple Clients
```bash
/memory/client-acme/onboard
/memory/client-beta/onboard
# Each client is completely isolated!
```

## ✅ What Happens

1. **Onboard**: Creates `user/{user_id}` branch (orphan - no shared history)
2. **Isolation**: Each user's data is completely separate
3. **GitHub**: Each user gets their own branch on GitHub
4. **No Mixing**: Your personal memories never mix with others

## 🔍 Verify Isolation

```bash
# List all users
curl "http://localhost:8000/server/users"

# Check GitHub branches
# Visit: https://github.com/kingfisherfox/diffmem/branches
# You'll see: user/benjamin, user/alice, etc.
```

## 💡 User ID Ideas

- `benjamin` - Your personal
- `work` - Work/professional
- `client-acme` - Client-specific
- `project-x` - Project-specific
- `assistant-coding` - Different AI contexts

**That's it!** Just use different `user_id` values and everything stays isolated.
