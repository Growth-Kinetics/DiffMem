# writer_agent CONTEXT

## Business Purpose
Process conversational session transcripts, identify necessary memory updates
(creations, modifications), and stage them in the git working directory. Acts
as the primary "write head" for the memory system, translating unstructured
dialogue into structured, differential memory. Folder structure and entity
vocabulary are driven by the active `OntologyProfile` — the agent is
ontology-agnostic at runtime.

## User Stories
- As a chat agent, I POST a session transcript and get memory files updated atomically.
- As a developer, changes are staged first (`process_session`) then committed explicitly
  (`commit_session`) so writes are atomic.
- As a self-hoster, I set `DIFFMEM_ONTOLOGY=corporate` and the writer uses the correct
  entity types and folder layout without code changes.

## Information Flow
- **Inputs:** Session transcript string, `user_id`, `repo_path` (worktree root), `OntologyProfile`.
- **Outputs:** Staged file changes in the git working directory (uncommitted).
- **Pipeline:** Identify entities → create new entity files (parallel LLM) → update existing
  entity files (parallel LLM) → create timeline entry (from git diff) → rebuild entity
  semantic indexes (parallel LLM) → rebuild master index.md.

## Terminology
- **session_transcript:** Raw text of a user-agent conversation.
- **staged_changes:** Modifications in the working directory, not yet committed.
- **semantic_index:** JSON descriptor appended to each entity file for fast retrieval triage.
- **memory_strength:** Score based on git edit frequency and recency; used to sort index.md.
- **master_index:** `index.md` at worktree root — all entities sorted by memory_strength.
- **OntologyProfile:** Resolved at `WriterAgent.__init__` time. Drives `_load_prompt()`,
  `_entity_md_files()`, `_create_single_entity()` folder routing, and `_resolve_entity_file_path()`.

## Key Files
- `agent.py` — `WriterAgent`: full write pipeline. `_entity_md_files()` replaces
  hardcoded `memories/` with ontology-aware scanning.
- `onboarding_agent.py` — `OnboardingAgent(WriterAgent)`: creates initial directory
  structure from `ontology.entity_types`, copies `ontology.repo_guide_path` into worktree.
- `prompts/` — Default prompt files (personal ontology). Ontology-specific overrides live in
  `src/diffmem/ontologies/{name}/prompts/` and are resolved via `ontology.resolve_prompt()`.

## External Dependencies
- **OpenRouter** (`docs/api-surface.md §OpenRouter`) — all LLM calls.
- **OntologyProfile** (`src/diffmem/ontology/loader.py`) — injected at init; never call
  `load_ontology()` inside agent methods.

## Constraints
- No commits until `commit_session()` is called. `process_session()` is purely preparatory.
- Use `_entity_md_files()` (not `self.memories_path`) for all entity file scanning.
  `self.memories_path` is kept for backwards compatibility but must not be used for scanning.
- Prompt loading always goes through `self.ontology.resolve_prompt(name)` — never open
  `self.prompts_path / ...` directly.
- All LLM calls are synchronous; the caller (`server.py`, executor) is responsible for
  running the agent in a thread pool off the uvicorn event loop.

## Attention Guidance
- For ontology-related issues: read `src/diffmem/ontology/loader.py` and the active
  ontology's `schema.json` first.
- For write pipeline latency: `process_session()` → per-step LLM calls, most time is
  in `_create_new_entities` / `_update_existing_entities` (parallel but LLM-bound).
- For master index staleness: `_rebuild_master_index()` scans `_entity_md_files()` and
  re-extracts SEMANTIC INDEX blocks — check that entity files have a `## SEMANTIC INDEX`.