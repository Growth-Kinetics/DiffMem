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
  → `run_reabsorb` (migration-only; deterministic, no LLM — folds legacy
     `entities/commitments/*` into owner `## Open Items` + `git rm` + commit)
  → `run_dedupe` (prefilter candidates → LLM judge → LLM merge → commit per merge)
  → `run_redistribute` (token-scan → LLM analyze → move/extract → commit per source)
  → `run_link` (git log co-occurrence → LLM weave wikilinks → single commit)
  → release lock
  → return per-tool result dicts

Tools are independently invokable; canonical order when chained is
reabsorb → dedupe → redistribute → link (reabsorb must run first to eliminate
the legacy commitments folder before dedupe reasons about the corpus; dedupe
changes filenames so links would break if generated first; redistribution
alters co-occurrence signal). **`reabsorb` is excluded from the default run set**
(`_DEFAULT_TOOLS = dedupe, redistribute, link`) — it is a one-time v2 migration
invoked explicitly via `consolidate(tools=["reabsorb"])`. Routine
`consolidate()` behaviour is unchanged.

## TERMINOLOGY
- **Consolidate commit:** git commit produced by this agent, message starts
  with `consolidate(reabsorb):`, `consolidate(dedupe):`,
  `consolidate(redistribute):`, or `consolidate(link):`.
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
- **Per-file commits require explicit staging.** Each tool commits with
  `repo.index.commit()`, which snapshots the git INDEX, not the working tree.
  Any file mutated on disk (e.g. an owner file gaining a `## Open Items` entry
  in reabsorb) MUST be staged with `repo.git.add(<path>)` before commit, or the
  edit is left as a dirty working-tree change and silently dropped. The sibling
  `run_dedupe` pattern (`_dedupe.py`) is the reference. Never use
  `repo.git.add("--all")` as an error fallback — it sweeps the runtime
  `.diffmem/consolidator.lock` into user history.
- **reabsorb is deterministic + idempotent.** No LLM. Owner resolution prefers
  projects globally, then people, then the root user entity — folder priority
  dominates wikilink parse order (an assignee person link parsed before a
  related-project link must NOT win over the project). Empty/absent
  `entities/commitments/` → zero commits (v2 steady state).
- **Survivor = higher memory_strength.** Loser's filename is preserved as an
  `alias` in the survivor's SEMANTIC INDEX so writer-agent recognition catches
  it on future sessions.
- **Merge propagation = ALL loser name variants (v0.4.1+).** Not just the
  loser's file stem — its SEMANTIC INDEX `name` AND every alias land in the
  survivor's aliases (`_dedupe.loser_name_variants` + `_ensure_aliases`).
  This is the alias-redirect trick: any old spelling resolves to the survivor
  via the writer's normalized/fuzzy index lookup, so the loser is never
  re-created by later sessions. The LLM merge payload may omit them; the
  propagation layer adds them regardless of what the LLM returned.
- **Prefilter surfaces, the judge decides (v0.4.1+).** `find_candidate_pairs`
  uses ANY-ONE-signal corroboration (name similarity ≥0.8 OR stem containment
  OR ≥2 shared related_entities OR ≥3 shared hard_cues) — the name gate is no
  longer a hard precondition. WHY: nickname-level variants of the same person
  ("Maya Chen" vs "Maya B.", ratio 0.63) previously could never surface even
  with full corroboration, so duplicates accumulated. The LLM judge
  (same_entity=true AND confidence=high) remains the sole merge arbiter;
  corroborated-but-different pairs surface and get rejected (encoded in the
  e2e scripted-judge fixture).
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
- **Semantic-index list fields are normalized at the read/write choke points
  (v0.4.1).** LLMs occasionally emit NESTED lists for contractually-flat
  string fields (`hard_cues: ["a", ["b", "c"]]`). Unnormalized, those shapes
  crashed the consolidate chain with `TypeError` at three sites — the
  `",".join` report builders in `_redistribute._candidates_block` and
  `_link._cooccurrence_block`, and the `set(map(str.lower, ...))` prefilter in
  `_dedupe._overlap` — which is why downstream consumers (ChatBarry) ran
  dedupe-only for months. Fix is structural, not per-site:
  `frontmatter.normalize_semantic_index()` (deep-flatten of
  `hard_cues/soft_cues/emotional_cues/aliases/related_entities`) runs inside
  `extract_semantic_index()` (READ choke point — repairs already-poisoned
  stores on next pass) and inside `write_with_semantic_index()` (WRITE choke
  point — no consolidator path can persist nesting). Regression suite:
  `tests/test_semantic_index_normalization.py`, including reproduction of the
  exact production crash shapes. Do NOT add defensive flattening at
  individual consumers — the choke points are the single source of truth.
