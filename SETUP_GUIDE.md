# DiffMem Setup and Interaction Guide

## What is DiffMem?

DiffMem is a **Git-based differential memory system** for AI agents. It stores memories as Markdown files in a Git repository, using Git's version control to track how memories evolve over time. Think of it as a "hippocampus" for AI systems that need to remember and learn from past conversations.

### Key Concepts:

1. **Current State Focus**: Only current memories are stored in files (not full history), keeping things lightweight
2. **Git History**: All changes are tracked via Git commits, allowing you to see how memories evolved
3. **Agent Retrieval**: An LLM agent explores the git repo (grep, git log, git diff) to build targeted context
4. **Multi-User Architecture**: Each user gets an isolated "orphan branch" in Git, completely separate from others
5. **Worktree System**: Active users have their memory "mounted" in a worktree directory for operations

### Architecture:

```
Storage Repo (Central Git Repository)
├── user/alex (orphan branch)
├── user/anna (orphan branch)
└── user/bob (orphan branch)

Active Contexts (Worktrees - mounted when user is active)
├── /app/active_contexts/alex/  (worktree for user alex)
├── /app/active_contexts/anna/    (worktree for user anna)
└── /app/active_contexts/bob/     (worktree for user bob)
```

## Setup Instructions

### 1. Prerequisites

- Python 3.10+ installed
- Git installed
- A GitHub account (for remote storage)
- An OpenRouter API key (for LLM operations)

### 2. Install Dependencies

```bash
# Navigate to project directory
cd /Users/benjaminpowell/Desktop/Coding_projects/DiffMem

# Install Python dependencies
pip install -r requirements.txt

# If running the server, also install server dependencies
pip install -r requirements-server.txt
```

### 3. Create GitHub Repository

You need a GitHub repository to store memories. This can be:
- **Public or Private** (your choice)
- **Empty or with initial content** (doesn't matter, DiffMem will manage it)

**Steps:**
1. Go to GitHub and create a new repository (e.g., `my-diffmem-storage`)
2. Note the repository URL (e.g., `https://github.com/yourusername/my-diffmem-storage`)
3. Create a **Personal Access Token** (PAT) with `repo` permissions:
   - Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
   - Generate new token with `repo` scope
   - Save the token securely

### 4. Configure Environment Variables

Create a `.env` file from the example:

```bash
cp env.example .env
```

Edit `.env` with your values:

```bash
# Required: OpenRouter API key for LLM operations
OPENROUTER_API_KEY=your-openrouter-api-key-here

# Required: GitHub repository configuration
GITHUB_REPO_URL=https://github.com/your-username/your-memory-repo
GITHUB_TOKEN=your-github-personal-access-token
GITHUB_BRANCH=main
GITHUB_USERNAME=your-github-username

# Optional: Server configuration
DEFAULT_MODEL=google/gemini-2.5-pro
STORAGE_PATH=/app/storage
WORKTREE_ROOT=/app/worktrees
SYNC_INTERVAL_MINUTES=5
LOG_LEVEL=info

# For client usage
DIFFMEM_SERVER_URL=http://localhost:8000
USER_ID=alex
```

**Important Configuration Notes:**
- `STORAGE_PATH`: Where the central Git repository is stored (default: `/app/storage`)
- `WORKTREE_ROOT`: Where active user contexts are mounted (default: `/app/worktrees`)
- `SYNC_INTERVAL_MINUTES`: How often to auto-sync to GitHub (default: 5 minutes)
- `DEFAULT_MODEL`: LLM model to use (default: `google/gemini-2.5-pro`)

### 5. Get OpenRouter API Key

1. Go to [OpenRouter.ai](https://openrouter.ai/)
2. Sign up/login
3. Go to Keys section
4. Create a new API key
5. Copy it to your `.env` file

## Running DiffMem

### Option 1: As a Server (Recommended for Production)

The server provides a REST API for interacting with DiffMem:

```bash
# Start the server
python -m diffmem.server

# Or using uvicorn directly
uvicorn diffmem.server:app --host 0.0.0.0 --port 8000 --reload
```

The server will:
- Initialize the storage repository
- Connect to GitHub
- Start listening on port 8000
- Auto-sync to GitHub every 5 minutes (configurable)

**API Documentation**: Once running, visit `http://localhost:8000/docs` for interactive API docs.

### Option 2: As a Python Library (For Development)

You can import and use DiffMem directly in Python:

```python
from diffmem import DiffMemory

# Initialize
memory = DiffMemory(
    repo_path="/path/to/worktree",  # Worktree path for user
    user_id="alex",
    openrouter_api_key="your-key",
    model="google/gemini-2.5-pro"
)

# Use it
context = memory.get_context(conversation, max_tokens=15000)
```

## How to Interact with DiffMem

### 1. Onboard a New User

Before using DiffMem, you need to "onboard" a user. This creates their memory structure.

**Via API:**
```bash
curl -X POST "http://localhost:8000/memory/alex/onboard" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "user_info": "Alex is a software engineer, born in 1990, lives in Seattle. Loves hiking and coffee.",
    "session_id": "onboard-001"
  }'
```

**Via Python:**
```python
from diffmem import onboard_new_user

result = onboard_new_user(
    repo_path="/app/worktrees/alex",
    user_id="alex",
    user_info="Alex is a software engineer...",
    openrouter_api_key="your-key",
    session_id="onboard-001"
)
```

### 2. Process and Store Memories

When you have a conversation transcript or memory content to store:

**Via API:**
```bash
curl -X POST "http://localhost:8000/memory/alex/process-and-commit" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "memory_input": "Had coffee with mom today. She mentioned her new job at the hospital. We discussed my upcoming vacation to Japan.",
    "session_id": "session-2024-01-15-001",
    "session_date": "2024-01-15"
  }'
```

**Via Python:**
```python
memory = DiffMemory(...)

# Process and commit in one step
memory.process_and_commit_session(
    memory_input="Had coffee with mom today...",
    session_id="session-2024-01-15-001",
    session_date="2024-01-15"
)

# Or process and commit separately
memory.process_session("Had coffee...", "session-001")
# ... do other things ...
memory.commit_session("session-001")
```

### 3. Retrieve Context for Conversations

When you need context for an AI conversation, the retrieval agent explores the git repository to find relevant entity sections, diffs, and temporal patterns:

**Via API:**
```bash
curl -X POST "http://localhost:8000/memory/alex/context" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation": [
      {"role": "user", "content": "Tell me about my relationship with my mom"}
    ],
    "max_tokens": 15000
  }'
```

**Parameters:**
- `conversation` (required) - Conversation history as message dicts
- `max_tokens` (default: 20000) - Agent's additional context token budget
- `max_turns` (default: 4) - Max agent exploration turns
- `timeout_seconds` (default: 30) - Hard timeout for the agent loop

**Via Python:**
```python
conversation = [
    {"role": "user", "content": "Tell me about my relationship with my mom"}
]

context = memory.get_context(conversation, max_tokens=15000)
print(context)
```

## Memory Structure

After onboarding, each user's memory is organized as:

```
<worktree_root>/alex/
├── alex.md                    # User's own profile (biographical)
├── index.md                   # Auto-generated keyword index
├── memories/
│   ├── people/               # People profiles
│   │   ├── mom.md
│   │   └── friend_john.md
│   └── contexts/             # Thematic contexts
│       ├── health.md
│       └── work.md
└── timeline/                 # Chronological events
    ├── 2024-01.md
    └── 2024-02.md
```

## Key API Endpoints

### Onboarding
- `POST /memory/{user_id}/onboard` - Onboard a new user
- `GET /memory/{user_id}/onboard-status` - Check if user is onboarded

### Reading
- `POST /memory/{user_id}/context` - Get context for conversation (agent-based retrieval)
- `GET /memory/{user_id}/user-entity` - Get user's profile
- `GET /memory/{user_id}/recent-timeline` - Get recent timeline entries

### Writing
- `POST /memory/{user_id}/process-session` - Process session (stage changes)
- `POST /memory/{user_id}/commit-session` - Commit staged changes
- `POST /memory/{user_id}/process-and-commit` - Process and commit in one step

### Utilities
- `GET /memory/{user_id}/status` - Get repository status
- `GET /memory/{user_id}/validate` - Validate setup
- `POST /memory/{user_id}/webhook/post-commit` - Post-commit sync hook
- `GET /health` - Health check
- `GET /server/users` - List active users
- `POST /server/sync` - Manual sync all users

## How GitHub Integration Works

1. **Initial Setup**: When you start the server, it:
   - Creates/loads a local Git repository at `STORAGE_PATH`
   - Configures GitHub remote using your `GITHUB_REPO_URL` and `GITHUB_TOKEN`
   - Fetches existing user branches from GitHub

2. **User Onboarding**: When a user is onboarded:
   - Creates an orphan branch `user/{user_id}` in the storage repo
   - Mounts it as a worktree at `WORKTREE_ROOT/{user_id}`
   - Pushes the branch to GitHub

3. **Memory Updates**: When memories are processed:
   - Changes are staged in the worktree
   - Committed with a descriptive message (includes modified entity names)
   - Post-commit hook triggers sync to GitHub
   - Changes are pushed to GitHub automatically

4. **Periodic Sync**: Every `SYNC_INTERVAL_MINUTES`, the server:
   - Checks all active users for uncommitted changes
   - Commits and pushes to GitHub

## Troubleshooting

### Server won't start
- Check that all environment variables are set in `.env`
- Verify GitHub token has `repo` permissions
- Ensure OpenRouter API key is valid

### User not found errors
- Make sure user is onboarded first (`/onboard` endpoint)
- Check that worktree exists at `WORKTREE_ROOT/{user_id}`

### GitHub sync issues
- Verify `GITHUB_REPO_URL` is correct
- Check that `GITHUB_TOKEN` is valid and has `repo` scope
- Ensure repository exists and is accessible

### Context retrieval returning only baseline
- Check that the retrieval agent LLM provider is configured (OPENROUTER_API_KEY or RETRIEVAL_AGENT_API_KEY)
- Review the `retrieval_plan.synthesis` field in the response for agent errors
- If `retrieval_version` is "fallback", the agent failed and returned only the user entity

## Next Steps

1. **Start the server**: `python -m diffmem.server`
2. **Onboard yourself**: Use the `/onboard` endpoint with your information
3. **Process some memories**: Send conversation transcripts via `/process-and-commit`
4. **Query memories**: Use `/context` to retrieve information
5. **Check GitHub**: Visit your repository to see how memories are stored

## Example Workflow

```bash
# 1. Start server
python -m diffmem.server

# 2. Onboard user
curl -X POST "http://localhost:8000/memory/alex/onboard" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"user_info": "Alex is a developer...", "session_id": "onboard-001"}'

# 3. Store a memory
curl -X POST "http://localhost:8000/memory/alex/process-and-commit" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"memory_input": "Had coffee with mom...", "session_id": "session-001"}'

# 4. Query memory
curl -X POST "http://localhost:8000/memory/alex/context" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"conversation": [{"role": "user", "content": "Tell me about mom"}], "max_tokens": 15000}'
```

## Additional Resources

- **Repository Guide**: See `repo_guide.md` for detailed memory structure
- **API Docs**: Visit `http://localhost:8000/docs` when server is running
- **README**: See `README.md` for project overview and philosophy
