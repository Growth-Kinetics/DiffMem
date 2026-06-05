# consolidator_agent CONTEXT

## BUSINESS PURPOSE
Out-of-band repair capability over a user's memory worktree. The writer agent's
session-formation hot path accumulates three failure modes at scale: duplicate
entities, an overstuffed user entity (catch-all), and no interlinking. The
consolidator runs a separate pass to fix each, producing distinct commits with
a `consolidate:` prefix so they can be told apart from session writes in git
history.

## USER STORIES
- As a memory operator, I want a separate consolidation step I can trigger
  after a session commit (or on a schedule) so the writer agent stays fast.
- As a retrieval agent, I want consolidation commits visibly tagged in
  `git log` so I can weight them differently from session-formation commits.
- As a human reading the memory in Obsidian, I want inline `[[wikilinks]]`
  so I can navigate the memory folder as a knowledge vault.

## INFO FLOW
Trigger (API or chained from process-and-commit)
  → acquire `.diffmem/consolidator.lock`
  → `run_dedupe` (prefilter candidates → LLM judge → LLM merge → commit per merge)
  → `run_redistribute` (token-scan → LLM analyze → move/extract → commit per source)
  → `run_link` (git log co-occurrence → LLM weave wikilinks → single commit)
  → release lock
  → return per-tool result dicts

Tools are independently invokable; canonical order when chained is
dedupe → redistribute → link (dedupe changes filenames so links would break
if generated first; redistribution alters co-occurrence signal).

## TERMINOLOGY
- **Consolidate commit:** git commit produced by this agent, message starts
  with `consolidate(dedupe):`, `consolidate(redistribute):`, or
  `consolidate(link):`.
- **Canonical file:** the survivor of a merge — the file with higher
  `memory_strength` in its SEMANTIC INDEX (ties broken by longer filename).
- **Soft cap:** the token threshold above which an entity is considered
  oversized for redistribution (default 32 000 tokens, `len // 4` heuristic).
- **Window:** runtime parameter for `run_link` — number of most-recent commits
  to mine for co-occurrence (default 3).
- **Wikilink:** Obsidian-style `[[memories/people/maya|Maya]]` — full path
  inside the user vault + display name. Rendered as a link in Obsidian and
  greppable as `\[\[` for downstream agents.

## ARCHITECTURAL CONSTRAINTS
- **Out-of-band only.** Never called from the writer's session pipeline.
  Operators invoke explicitly via `DiffMemory.consolidate(...)` or the HTTP
  endpoint. Default trigger model is operator-driven, not automatic.
- **High-confidence only, no human review.** Dedupe merges only when LLM judge
  returns `same_entity=true AND confidence=high`. Low-confidence pairs are
  dropped silently and may re-surface in future runs.
- **Lock required.** All three tools acquire `<worktree>/.diffmem/consolidator.lock`
  before any mutation. Stale locks (dead PID + >30 min old) are reclaimed.
  Concurrent writer or consolidator runs raise `LockBusyError`.
- **Runs in the writer pool.** Honours ADR-D001: blocking work goes through
  `_writer_pool.run_in_executor` so the uvicorn event loop stays free.
- **Distinct commit prefix.** All commits start with `consolidate(...)`. The
  retrieval agent can use this prefix to weight or filter history.
- **Survivor = higher memory_strength.** Loser's filename is preserved as an
  `alias` in the survivor's SEMANTIC INDEX so writer-agent recognition catches
  it on future sessions.
- **No coupling to writer-agent internals.** Where helpers are needed
  (e.g. fuzzy text matching, index rebuilding), prefer extracting to a shared
  module rather than reaching into `writer_agent.agent.WriterAgent` directly.
- **Consolidator prompts are not ontology-scoped (by design).** The three repair
  tools (dedupe, redistribute, link) operate on the semantic structure of files
  regardless of ontology. Their prompts live in `consolidator_agent/prompts/`
  and are not resolved via `OntologyProfile.resolve_prompt()`. If a future
  ontology requires custom consolidation behavior, add a `consolidator_prompts/`
  key to `schema.json` and extend the loader. Do not silently inherit from the
  personal ontology without documenting the decision.
