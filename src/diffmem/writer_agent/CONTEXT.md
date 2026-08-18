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
- **LLM semantic-index responses are normalized before persisting (v0.4.1).**
  `_build_single_entity_index()` passes the LLM's build_index JSON through
  `frontmatter.normalize_semantic_index()` before `merge_frontmatter()`. WHY:
  models occasionally return nested lists for contractually-flat fields
  (`hard_cues: ["a", ["b", "c"]]`), and those poisoned files crashed the
  consolidator's joins/prefilters downstream with `TypeError` (see
  `consolidator_agent/CONTEXT.md` for the full incident note). This is the
  ingress guard; the consolidator's `extract_semantic_index` is the repair
  guard for stores written before v0.4.1.

## Attention Guidance
- For ontology-related issues: read `src/diffmem/ontology/loader.py` and the active
  ontology's `schema.json` first.
- For write pipeline latency: `process_session()` → per-step LLM calls, most time is
  in `_create_new_entities` / `_update_existing_entities` (parallel but LLM-bound).
- **Writer _rebuild_master_index now reads v2 frontmatter (v0.5.2).** The prior
  implementation only parsed the legacy `## SEMANTIC INDEX` JSON block — every
  v2 entity (YAML frontmatter) was silently skipped, leaving index.md empty.
  The identify step then saw no existing entities → proposed creating
  everything → the duplicate-spawning behavior the user originally reported.
  Fixed by delegating to the shared `rebuild_master_index` (consolidator_agent/
  _shared) which uses `extract_semantic_index` (handles BOTH formats). The
  self-healing behavior (rebuilding SI for files that lack one) is preserved.
- For master index staleness: `_rebuild_master_index()` scans `_entity_md_files()` and
  re-extracts SEMANTIC INDEX blocks — check that entity files have a `## SEMANTIC INDEX`.
- For duplicate entity creation: `_resolve_entity_file_path()` resolves in tiers —
  exact index lookup → normalized index key (`_normalize_name`: lowercase,
  diacritics stripped, punctuation removed) → computed filename → fuzzy
  (`SequenceMatcher` ≥ `FUZZY_NAME_THRESHOLD`=0.85 over index names+aliases,
  plus stem containment ≥4 chars) → filesystem stem scan. Structured log lines:
  `ENTITY_RESOLVED_INDEX` / `_COMPUTED` / `_NORMALIZED` / `_FUZZY` / `_NOT_FOUND`.
  A resolve miss is what turns a transcript mention into a NEW file — before
  v0.4.1 only the exact-lower + computed tiers existed and spelling variants
  duplicated freely. The identify prompt (ontology side) instructs the LLM to
  check aliases first; these tiers are the deterministic backstop.