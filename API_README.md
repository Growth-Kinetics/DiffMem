# DiffMem API Documentation

This document describes the API interface for DiffMem, a git-based differential memory system for AI agents.

## Quick Start

### For New Users (Onboarding)

If you're setting up DiffMem for a completely new user, you'll need to onboard them first:

```python
from diffmem.api import onboard_new_user

# Onboard a new user with their information
user_info = """
John is a 32-year-old software engineer living in San Francisco.
He works at TechCorp and is married to Sarah. They have a daughter Emma.
John enjoys rock climbing and reading science fiction novels.
"""

result = onboard_new_user(
    repo_path="/path/to/memory/repo",
    user_id="john_doe", 
    user_info=user_info,
    openrouter_api_key="your-key-here"
)

if result['success']:
    print(f"User onboarded! Created {result['entities_created']} entities")
else:
    print(f"Onboarding failed: {result['error']}")
```

### For Existing Users

```python
from diffmem.api import create_memory_interface

# Create memory interface for existing user
memory = create_memory_interface(
    repo_path="/path/to/memory/repo",
    user_id="john_doe",
    openrouter_api_key="your-key-here"
)

# Check if user is properly onboarded
if not memory.is_onboarded():
    print("User needs to be onboarded first!")
    # Use onboard_new_user() or memory.onboard_user()
```

## Core API Classes

### DiffMemory

The main interface for memory operations.

#### Constructor

```python
DiffMemory(repo_path, user_id, openrouter_api_key, model="google/gemini-2.5-pro", auto_onboard=False)
```

- `repo_path`: Path to git repository containing memory files
- `user_id`: User identifier (must exist in `users/` directory unless auto_onboard=True)
- `openrouter_api_key`: API key for OpenRouter LLM access
- `model`: Default model for LLM operations
- `auto_onboard`: If True, allows initialization even if user doesn't exist yet

#### Onboarding Methods

```python
# Check if user is onboarded
is_onboarded = memory.is_onboarded()

# Onboard a user (if not already onboarded)
result = memory.onboard_user(user_info, session_id="optional-id")
```

#### Read Operations

```python
# Get context for a conversation
context = memory.get_context(conversation, depth="basic")
# depth options: "basic", "wide", "deep", "temporal"

# Direct BM25 search
results = memory.search("relationship dynamics", k=5)

# LLM-orchestrated search
results = memory.orchestrated_search(conversation, k=5)

# Get user entity file
user_entity = memory.get_user_entity()

# Get recent timeline
timeline = memory.get_recent_timeline(days_back=30)
```

#### Write Operations

```python
# Process a session (stages changes, doesn't commit)
memory.process_session(
    memory_input="Had coffee with mom today...", 
    session_id="session-123",
    session_date="2024-01-15"  # optional, defaults to today
)

# Commit staged changes
memory.commit_session("session-123")

# Process and commit in one step
memory.process_and_commit_session(
    memory_input="Had coffee with mom today...",
    session_id="session-123"
)
```

#### Utility Operations

```python
# Get repository status
status = memory.get_repo_status()

# Validate setup
validation = memory.validate_setup()

# Rebuild search index
memory.rebuild_index()
```

## Server API Endpoints

If you're running the DiffMem server, you can use these HTTP endpoints:

### Onboarding Endpoints

#### POST `/memory/{user_id}/onboard`

Onboard a new user to the memory system.

**Request Body:**
```json
{
    "user_info": "Raw information dump about the user...",
    "session_id": "optional-session-id"
}
```

**Response:**
```json
{
    "status": "success",
    "message": "User john_doe successfully onboarded",
    "result": {
        "success": true,
        "user_id": "john_doe",
        "session_id": "onboard-20241201-143022",
        "entities_created": 5,
        "files_created": {
            "user_file": "/path/to/users/john_doe/john_doe.md",
            "entities": ["Sarah", "TechCorp", "Emma", "Dr. Martinez", "Mission District"],
            "timeline": "2024-12.md",
            "master_index": "index.md"
        }
    }
}
```

#### GET `/memory/{user_id}/onboard-status`

Check if a user is onboarded.

**Response:**
```json
{
    "status": "success",
    "user_id": "john_doe",
    "onboarded": true,
    "message": "User john_doe is onboarded"
}
```

### Memory Operations

#### POST `/memory/{user_id}/context`

Get assembled context for a conversation.

#### POST `/memory/{user_id}/search`

Search memory using BM25.

#### POST `/memory/{user_id}/process-and-commit`

Process and commit a session in one step.

#### GET `/memory/{user_id}/status`

Get repository status (works for both onboarded and non-onboarded users).

#### GET `/memory/{user_id}/validate`

Validate memory setup (works for both onboarded and non-onboarded users).

## Convenience Functions

```python
from diffmem.api import create_memory_interface, onboard_new_user, quick_search

# Create memory interface with auto-onboard capability
memory = create_memory_interface(
    repo_path="/path/to/repo",
    user_id="john_doe",
    auto_onboard=True  # Allows initialization even if user doesn't exist
)

# Onboard a completely new user
result = onboard_new_user(
    repo_path="/path/to/repo",
    user_id="jane_doe", 
    user_info="Jane is a data scientist...",
    session_id="onboard-001"
)

# Quick search without full initialization
results = quick_search("/path/to/repo", "machine learning")
```

## Error Handling

The system provides clear error messages for common issues:

### User Not Onboarded

If you try to use memory operations on a user who hasn't been onboarded:

```python
# This will raise ValueError
try:
    context = memory.get_context(conversation)
except ValueError as e:
    print(f"Error: {e}")
    # Error: User john_doe has not been onboarded. Call onboard_user() first.
```

### User Already Onboarded

If you try to onboard a user who already exists:

```python
result = memory.onboard_user(user_info)
if not result['success']:
    print(result['error'])
    # Error: User john_doe is already onboarded
```

## Repository Structure

After onboarding, the repository will have this structure:

```
repo/
├── users/
│   └── john_doe/
│       ├── john_doe.md          # Main user entity file
│       ├── index.md             # Master index of all entities
│       ├── memories/
│       │   ├── people/          # Person entities
│       │   │   ├── sarah.md
│       │   │   └── emma.md
│       │   └── contexts/        # Organization, place, concept entities
│       │       ├── techcorp.md
│       │       └── mission_district.md
│       └── timeline/
│           └── 2024-12.md       # Monthly timeline entries
└── .git/                        # Git repository
```

## Examples

See `examples/onboarding_example.py` for a complete demonstration of the onboarding functionality using both the API interface and server endpoints.

## Migration from Pre-Onboarding Versions

If you have an existing DiffMem setup without onboarding, your existing users should continue to work normally. The new onboarding system is designed to be backward compatible.

To check if a user needs onboarding:

```python
memory = create_memory_interface(repo_path, user_id, api_key, auto_onboard=True)
if not memory.is_onboarded():
    # User needs onboarding
    result = memory.onboard_user(user_info)
``` 