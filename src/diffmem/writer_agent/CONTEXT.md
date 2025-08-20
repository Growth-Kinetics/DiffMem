# writer_agent CONTEXT  
## BUSINESS PURPOSE  
To process conversational session transcripts, identify necessary memory updates (creations, modifications), and stage them in the Git working directory. It acts as the primary "write head" for the memory system, translating unstructured dialogue into structured, differential memory.  

## USER STORIES  
- As Anna, I want to feed a session transcript to an agent that intelligently updates Alex's memory files without my manual intervention.  
- As a developer, I want changes to be buffered (staged) so I can review them before a final, atomic commit seals the session's memory update.  

## INFO FLOW  
Session Transcript -> Load Core Context -> LLM: Identify New Entities -> Create Files -> LLM: Update Existing Files -> Git Diff -> LLM: Create Timeline Entry -> Build Entity Indexes -> Rebuild Master Index -> Staged Changes Ready.  
[ASCII DIAGRAM]  
Transcript --> [LLM: Entity Triage] --> Create Files --> [LLM: Update Files] --> [Git Diff] --> [LLM: Timeline Gen] --> [LLM: Build Indexes] --> [Rebuild Master Index] --> Staged Files  

## TERMINOLOGY  
- "session_transcript": Raw text of a user-agent conversation.  
- "core_context": Concatenated `[ALWAYS_LOAD]` blocks from all entities.  
- "staged_changes": Modifications made to files in the working directory, not yet committed.  
- "search_text": Unique content identifier used for locating update positions in files.
- "replacement_text": New content to replace or insert at identified locations.
- "context-based update": Update mechanism using semantic search-and-replace rather than line numbers.
- "semantic_index": JSON descriptor appended to entity files for fast retrieval triage.
- "memory_strength": Quantitative score based on edit frequency and recency for ranking entity importance.
- "master_index": Consolidated index.md file containing all entities sorted by memory strength.

## ARCHITECTURAL CONSTRAINTS  
- **Transactional Integrity**: No commits are made until `commit_session()` is explicitly called. `process_session()` is purely preparatory.  
- **LLM Dependency**: All reasoning (entity creation, updates, summarization, indexing) is delegated to an external LLM. This agent orchestrates the prompts and file I/O.  
- **Context-Based Editing**: Uses search-and-replace with surrounding context to avoid line-shift problems. Updates target semantic content, not positional indices.
- **Fuzzy Matching**: Implements whitespace-tolerant text matching to handle formatting variations while maintaining precision.
- **Incremental Indexing**: Only modified files get their semantic indexes rebuilt; master index consolidates all entities.
- **Git-Based Metrics**: Memory strength calculation uses git log data for edit frequency and recency scoring.
- **PoC Limit**: No complex templating for new entities; uses a comprehensive existing file as a one-shot example.