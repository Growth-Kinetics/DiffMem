# Quick Start Guide

## ✅ What You've Already Done

1. ✅ Installed dependencies
2. ✅ Installed the package (`pip install -e .`)
3. ✅ Configured `.env` file with your keys
4. ✅ Fixed paths and user ID

## 🔑 About the Keys (You Already Have Them!)

### OpenRouter API Key
- **Where you got it**: You already have it in your `.env` file!
- **What it's for**: Allows DiffMem to use LLMs (like Gemini) to process and understand memories
- **Where to get a new one**: [OpenRouter.ai](https://openrouter.ai/) → Keys section

### GitHub Token
- **Where you got it**: You already have it in your `.env` file!
- **What it's for**: Allows DiffMem to push/pull memories to your GitHub repository
- **Where to get a new one**: GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate new token (classic) with `repo` scope

## 👤 About "alex" vs "benjamin"

- **"alex"** was just an **example** in the documentation
- **"benjamin"** is now your user ID (I changed it in `.env`)
- You can use **any user ID you want** - it's just an identifier
- Each user ID gets their own isolated memory branch in Git

## 🚀 Starting the Server

```bash
cd /Users/benjaminpowell/Desktop/Coding_projects/DiffMem
python -m diffmem.server
```

Or using uvicorn directly:
```bash
uvicorn diffmem.server:app --host 0.0.0.0 --port 8000 --reload
```

The server will:
- Start on `http://localhost:8000`
- Create storage directories automatically
- Connect to your GitHub repo
- Be ready to use!

## 📝 First Steps After Server Starts

### 1. Check Health
```bash
curl http://localhost:8000/health
```

### 2. Onboard Yourself
```bash
curl -X POST "http://localhost:8000/memory/benjamin/onboard" \
  -H "Content-Type: application/json" \
  -d '{
    "user_info": "Benjamin is a developer. He likes coding and building AI systems.",
    "session_id": "onboard-001"
  }'
```

### 3. Store Your First Memory
```bash
curl -X POST "http://localhost:8000/memory/benjamin/process-and-commit" \
  -H "Content-Type: application/json" \
  -d '{
    "memory_input": "Today I worked on setting up DiffMem. It was interesting learning about Git-based memory systems.",
    "session_id": "session-001",
    "session_date": "2025-01-31"
  }'
```

### 4. Query Your Memory
```bash
curl -X POST "http://localhost:8000/memory/benjamin/context" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation": [{"role": "user", "content": "What did I work on today?"}],
    "depth": "deep"
  }'
```

## 📚 API Documentation

Once the server is running, visit:
- **Interactive Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'diffmem'"
**Fixed!** You installed it with `pip install -e .`

### Server won't start
- Check that `.env` file exists and has all required values
- Make sure you're in the project directory
- Check that all dependencies are installed: `pip install -r requirements-server.txt`

### GitHub sync errors
- Verify your `GITHUB_TOKEN` is valid
- Check that `GITHUB_REPO_URL` is correct
- Make sure the repository exists and you have access

### Path errors
- The paths in `.env` are now set to local directories (not `/app/storage`)
- They'll be created automatically when the server starts

## 📂 Directory Structure

After running, you'll see:
```
DiffMem/
├── storage/          # Central Git repository (created automatically)
├── worktrees/        # Active user contexts (created automatically)
│   └── benjamin/     # Your memory worktree (created when onboarded)
└── .env              # Your configuration
```

## 🎯 Next Steps

1. **Start the server**: `python -m diffmem.server`
2. **Onboard yourself**: Use the `/onboard` endpoint
3. **Start storing memories**: Process conversation transcripts
4. **Query memories**: Use `/context` or `/search` endpoints
5. **Check GitHub**: Your memories will sync to your repo automatically!
