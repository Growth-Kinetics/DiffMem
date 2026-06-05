# ontology

## Purpose
Resolve the `DIFFMEM_ONTOLOGY` env var to a validated `OntologyProfile` that every
agent reads at runtime. The ontology defines the entity taxonomy (types, folders,
index vocabulary) and prompt overrides for a deployment. This lets self-hosters
run DiffMem with a different lens (e.g. corporate CRM) without forking the engine.

## User Stories
- As a self-hoster, I set `DIFFMEM_ONTOLOGY=corporate` and get CRM-style entity
  types without touching Python.
- As a contributor, I add `ontologies/my-ontology/` with `schema.json` and
  `repo_guide.md`, open a PR, and the community gains a new built-in profile.
- As an operator, I set `DIFFMEM_ONTOLOGY=/etc/diffmem/acme` to use a private
  ontology on disk without forking the repo.

## Information Flow
- **Inputs:** `DIFFMEM_ONTOLOGY` env var (or explicit `name_or_path` arg).
- **Outputs:** `OntologyProfile` dataclass — consumed by `WriterAgent`,
  `OnboardingAgent`, `ConsolidatorAgent`, `run_retrieval_agent()`, and
  `load_always_load_for_entities()`.
- **Resolution order:** env var → built-in name lookup in
  `src/diffmem/ontologies/{name}/` → absolute path on disk.
- **Loaded once** in `DiffMemory.__init__()` and `server.py` lifespan; propagated
  downward. Never re-loaded per request.

## Terminology
- **OntologyProfile:** Dataclass holding parsed `schema.json`, resolved `prompts_dir`
  (may be None), `fallback_prompts_dir` (always `writer_agent/prompts/`), and
  `repo_guide_path`.
- **entity_types:** List of `{name, folder, index_type}` dicts from `schema.json`.
- **folder_map:** `{entity_type_name → relative_folder_path}` derived from entity_types.
- **entity_dirs(repo_root):** Absolute paths to every entity folder under a worktree.
  Use this for scanning, not hardcoded `memories/`.
- **contexts_folder(repo_root):** Folder for orphan-theme extraction by the redistribute
  tool. Set via `schema.json` `"contexts_folder"` field.
- **resolve_prompt(name):** Check ontology `prompts_dir` first; fall back to
  `writer_agent/prompts/`. Raises `FileNotFoundError` if missing from both.

## Key Files
- `loader.py` — `load_ontology()`, `OntologyProfile`, `_validate_schema()`,
  `_load_from_dir()`. Single entry point for all ontology resolution.
- `__init__.py` — re-exports `OntologyProfile` and `load_ontology`.
- `../ontologies/personal/` — default built-in (people/contexts/events, `memories/`).
- `../ontologies/corporate/` — CRM built-in (people/projects/decisions/commitments/external,
  `entities/`). Includes 6 overridden prompt files.
- `../ontologies/README.md` — schema.json contract, contribution guide.

## External Dependencies
None. Pure filesystem + JSON parsing.

## Constraints
- `schema.json` and `repo_guide.md` are always required in every ontology directory.
  No fallback — missing either raises `ValueError` at startup.
- Prompt files fall back to `writer_agent/prompts/`; schema/repo_guide do not.
- Unknown built-in name raises `ValueError` immediately (never silently falls back to
  `personal`). This is intentional — misconfiguration must be loud.
- `_ONTOLOGIES_DIR` is resolved relative to this file (`../ontologies/`) so it works
  in dev, pip install, and editable installs without env vars or path manipulation.

## Attention Guidance
- When adding a new built-in ontology: create `src/diffmem/ontologies/{name}/`
  with `schema.json`, `repo_guide.md`, and optional `prompts/`. Mirror to
  `ontologies/{name}/` for GitHub browsing.
- When entity scanning breaks for a non-personal ontology: check that all callers
  use `ontology.entity_dirs(repo_path)`, not `worktree / "memories"`.
- When prompts produce wrong vocabulary: check that the ontology's `build_index.txt`
  override lists the correct `index_type` values to match `schema.json`.
