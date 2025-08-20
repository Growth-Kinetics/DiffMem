# DiffMem API Documentation

A module-driven interface for differential memory operations. No servers or endpoints required - import directly into your chat agent.

## Quick Start

```python
from diffmem import DiffMemory

# Initialize for a user
memory = DiffMemory("/path/to/repo", "alex", "your-openrouter-key")

# Read operations
conversation = [{"role": "user", "content": "How has my relationship with mom evolved?"}]
context = memory.get_context(conversation, depth="basic")

# Write operations
memory.process_session("Had coffee with mom today...", "session-123")
memory.commit_session("session-123")
```

## Installation

Ensure you have the DiffMem package installed and your memory repository set up according to the `repo_guide.md`.

Required environment variable:
```bash
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

## Core Classes

### DiffMemory

Main interface for all memory operations.

#### Initialization

```python
DiffMemory(repo_path: str, user_id: str, openrouter_api_key: str, model: str = "google/gemini-2.5-pro")
```

- `repo_path`: Path to your git repository containing memory files
- `user_id`: User identifier (must exist in `users/` directory)
- `openrouter_api_key`: Your OpenRouter API key
- `model`: Default LLM model for operations

## Read Operations

### get_context(conversation, depth="basic")

Assembles context for a conversation based on memory retrieval.

**Depth modes:**
- `"basic"`: Top entities with ALWAYS_LOAD blocks only
- `"wide"`: Semantic search with ALWAYS_LOAD blocks  
- `"deep"`: Complete entity files
- `"temporal"`: Complete files with Git blame history

```python
conversation = [
    {"role": "user", "content": "Tell me about my goals"},
    {"role": "assistant", "content": "Let me check your memories..."}
]

# Basic context (fastest)
basic_context = memory.get_context(conversation, depth="basic")

# Deep context (complete entity files)
deep_context = memory.get_context(conversation, depth="deep")

# Temporal context (with git history)
temporal_context = memory.get_context(conversation, depth="temporal")
```

**Returns:**
```python
{
    'always_load_blocks': [...],     # Core memory blocks
    'recent_timeline': [...],        # Recent timeline entries  
    'session_metadata': {...},       # Assembly metadata
    'complete_entities': [...],      # Full entities (deep/temporal)
    'temporal_blame': [...]          # Git history (temporal only)
}
```

### search(query, k=5)

Direct BM25 search over memory blocks.

```python
results = memory.search("family dynamics", k=3)
for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Content: {result['snippet']['content'][:100]}...")
```

### orchestrated_search(conversation, model=None, k=5)

LLM-guided search that derives queries from conversation context.

```python
orchestrated = memory.orchestrated_search(conversation)
print(f"Derived query: {orchestrated['derived_query']}")
print(f"Response: {orchestrated['response']}")
```

### get_user_entity()

Returns the complete user entity file.

```python
user_data = memory.get_user_entity()
print(f"User: {user_data['entity_name']}")
```

### get_recent_timeline(days_back=30)

Gets recent timeline entries.

```python
recent = memory.get_recent_timeline(days_back=7)
print(f"Found {len(recent)} entries from last week")
```

## Write Operations

### process_session(memory_input, session_id, session_date=None)

Processes session content and stages changes (doesn't commit).

```python
session_transcript = """
Had a great conversation with mom today. She told me about her 
childhood and seemed more relaxed than usual. Our relationship 
is definitely improving.
"""

memory.process_session(session_transcript, "session-2024-01-15-001")
```

**What it does:**
1. Identifies entities mentioned in the input
2. Creates new entity files if needed
3. Updates existing entity files
4. Creates timeline entries
5. Rebuilds semantic indexes
6. Stages all changes in git

### commit_session(session_id)

Commits all staged changes for a session.

```python
memory.commit_session("session-2024-01-15-001")
```

### process_and_commit_session(memory_input, session_id, session_date=None)

Convenience method that processes and immediately commits.

```python
memory.process_and_commit_session(
    "Today I learned that mom is taking painting classes...",
    "session-123"
)
```

## Utility Operations

### get_repo_status()

Returns repository statistics and status.

```python
status = memory.get_repo_status()
print(f"Memory files: {status['memory_files_count']}")
print(f"Index blocks: {status['index_stats']['total_blocks']}")
```

### validate_setup()

Validates that the memory repository is correctly configured.

```python
validation = memory.validate_setup()
if not validation['valid']:
    print("Issues found:", validation['issues'])
if validation['warnings']:
    print("Warnings:", validation['warnings'])
```

### rebuild_index()

Forces a rebuild of the BM25 search index.

```python
memory.rebuild_index()  # Called automatically after write operations
```

## Convenience Functions

### create_memory_interface(repo_path, user_id, openrouter_api_key=None, model="anthropic/claude-3-haiku")

Creates a DiffMemory instance with optional environment variable fallback for API key.

```python
from diffmem import create_memory_interface

# Uses OPENROUTER_API_KEY environment variable
memory = create_memory_interface("/path/to/repo", "alex")

# Or provide key explicitly  
memory = create_memory_interface("/path/to/repo", "alex", "your-key")
```

### quick_search(repo_path, query, k=5)

Performs a search without full initialization (index-only).

```python
from diffmem import quick_search

results = quick_search("/path/to/repo", "work stress", k=3)
```

## Integration Examples

### Simple Chat Agent

```python
from diffmem import create_memory_interface

class ChatAgent:
    def __init__(self, repo_path: str, user_id: str):
        self.memory = create_memory_interface(repo_path, user_id)
        self.conversation = []
    
    def process_message(self, user_message: str) -> str:
        self.conversation.append({"role": "user", "content": user_message})
        
        # Get relevant context
        context = self.memory.get_context(self.conversation[-5:], depth="basic")
        
        # Generate response using context (your LLM logic here)
        response = self.generate_response(context, user_message)
        
        self.conversation.append({"role": "assistant", "content": response})
        return response
    
    def end_session(self, session_id: str):
        # Save conversation to memory
        full_transcript = "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in self.conversation
        ])
        self.memory.process_and_commit_session(full_transcript, session_id)
```

### Memory-Aware Assistant

```python
class MemoryAssistant:
    def __init__(self, repo_path: str, user_id: str):
        self.memory = create_memory_interface(repo_path, user_id)
    
    def recall_memories(self, query: str) -> str:
        """Search memories and return formatted results"""
        results = self.memory.search(query, k=5)
        
        if not results:
            return "I don't have any memories matching that query."
        
        formatted = "Here's what I remember:\n\n"
        for i, result in enumerate(results, 1):
            content = result['snippet']['content'][:200] + "..."
            formatted += f"{i}. {content}\n\n"
        
        return formatted
    
    def update_memory(self, experience: str) -> str:
        """Add new experience to memory"""
        session_id = f"update-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        try:
            self.memory.process_and_commit_session(experience, session_id)
            return f"Memory updated successfully (session: {session_id})"
        except Exception as e:
            return f"Failed to update memory: {str(e)}"
```

## Error Handling

The API raises standard Python exceptions:

- `FileNotFoundError`: Repository or user paths don't exist
- `ValueError`: Invalid parameters (e.g., missing API key)
- `Exception`: LLM or git operation failures

```python
try:
    memory = DiffMemory("/path/to/repo", "alex", "api-key")
    context = memory.get_context(conversation)
except FileNotFoundError as e:
    print(f"Repository setup issue: {e}")
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Operation failed: {e}")
```

## Performance Notes

- **Index Rebuilding**: Automatically triggered after write operations
- **Lazy Loading**: Components are initialized only when first used
- **Memory Usage**: BM25 index is kept in memory for fast searches
- **Git Operations**: All writes are staged before commit for atomicity

## Supported Models

Config is set up for calling LLMs through openrouter. 


## Repository Structure

Your memory repository should follow this structure:

```
repo/
├── users/
│   └── alex/
│       ├── alex.md              # User entity file
│       ├── index.md             # Master index (auto-generated)
│       ├── memories/
│       │   ├── people/          # Person entities
│       │   ├── contexts/        # Context entities  
│       │   └── events/          # Event entities
│       └── timeline/            # Timeline entries
│           ├── 2024-01.md
│           └── 2024-02.md
└── repo_guide.md               # Repository documentation
```

See `repo_guide.md` for detailed setup instructions.

## Complete Example

```python
#!/usr/bin/env python3
import os
from diffmem import DiffMemory
import json 

# Setup
REPO_PATH = "/path/to/memory/repo" 
USER_ID = "alex"
API_KEY = os.getenv("OPENROUTER_API_KEY")

def main():
    # Initialize
    memory = DiffMemory(REPO_PATH, USER_ID, API_KEY)
    
    # Validate setup
    validation = memory.validate_setup()
    if not validation['valid']:
        print(f"Setup issues: {validation['issues']}")
        return
    
    # Example conversation
    conversation = [
        {"role": "user", "content": "How am I doing with my health goals?"}
    ]
    
    # Get context (loads user entity plus 5 highest ranked indexes)
    context = memory.get_context(conversation, depth="basic")
 
    # Get wide context (loads user entity plus summary from relevant entities)
    context = memory.get_context(conversation, depth="wide")

    # Get deep context  (loads user entity plus full relevant entities)
    context = memory.get_context(conversation, depth="deep")

    # Get temporal context  (loads user entity plus full relevant entities along with git blame)
    context = memory.get_context(conversation, depth="temporal")

    print(f"Loaded {len(context['always_load_blocks'])} memory blocks")
    
    # Search memories
    health_memories = memory.search("health fitness exercise", k=5)
    print(f"Found {len(health_memories)} health-related memories")
    
    # Update with new information
    new_experience = """
    Went for a 5-mile run today. Felt great and my pace is improving. 
    Also started meal prepping on Sundays which is helping with my 
    nutrition goals. Overall feeling more energetic.
    """

    session_id = "health-update-001"
    session_date="2025-08-10"
    memory.process_and_commit_session(new_experience, session_id,session_date)
    
    #or pass entire conversation as text, you can also process without committing if you want to review first.
    memory.process_session(json.dumps(conversation), session_id,session_date)
   

    
    
    # Get updated status
    status = memory.get_repo_status()
    print(f"Repository now has {status['memory_files_count']} memory files")

if __name__ == "__main__":
    main()
``` 