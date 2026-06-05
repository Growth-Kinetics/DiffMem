# DiffMem Ontologies

> **Note for contributors:** The canonical built-in ontology files that ship with
> the package live in [`src/diffmem/ontologies/`](../src/diffmem/ontologies/).
> The files in this directory (`ontologies/`) are mirrors kept in sync for easy
> GitHub browsing. If you are adding or editing a built-in ontology, edit the
> files under `src/diffmem/ontologies/` — those are what the engine loads.
> Then copy the changes here so the docs stay in sync.


An **ontology** in DiffMem is a named profile that defines the entity taxonomy for a
memory repository: what kinds of entities exist, where their files live, and what
vocabulary the agents use when creating and retrieving memories.

Selecting an ontology is a **deployment-time decision** made via the `DIFFMEM_ONTOLOGY`
environment variable. It is set once and applies to all users on that deployment.

---

## Picking a Built-in Ontology

```
DIFFMEM_ONTOLOGY=personal    # default — people, contexts, events
DIFFMEM_ONTOLOGY=corporate   # CRM-style — people, projects, decisions, commitments, external
```

| Profile | Best for | Entity types |
|---|---|---|
| `personal` | Personal AI companions, neurodivergent support, long-horizon conversational agents | people, contexts, events |
| `corporate` | Agency/team agents, CRM-style memory, project and decision tracking | people, projects, decisions, commitments, external |

---

## Using a Custom Ontology

Point `DIFFMEM_ONTOLOGY` at an absolute directory path on disk:

```
DIFFMEM_ONTOLOGY=/etc/diffmem/my-ontology
```

DiffMem will error at startup with a clear message if the directory is missing or invalid.

### Required files

Every ontology directory **must** contain:

| File | Purpose |
|---|---|
| `schema.json` | Machine-readable entity type definitions (see contract below) |
| `repo_guide.md` | Schema reference document — copied into every user's worktree at onboard time so agents can read it |

### Optional: prompt overrides

Create a `prompts/` subdirectory with any `.txt` files you want to override.
Any prompt **not present** in your ontology's `prompts/` directory falls back to
the default `personal` prompts in `src/diffmem/writer_agent/prompts/`.

Prompts you can override:

| File | What it controls |
|---|---|
| `1_identify_entities.txt` | Entity kind vocabulary and creation rules |
| `2_create_entity_file.txt` | Per-type file templates |
| `3_update_entity_file.txt` | Update instructions and conventions |
| `onboard_user_entity.txt` | Root entity file structure at onboarding |
| `onboard_identify_entities.txt` | Entity selection rules during onboarding |
| `build_index.txt` | SEMANTIC INDEX schema and type vocabulary |
| `0_system.txt` | Global system persona (rarely needs changing) |
| `4_create_timeline_entry.txt` | Timeline entry format |
| `onboard_timeline_entry.txt` | First timeline entry format |

---

## `schema.json` Contract

```json
{
  "name": "my-ontology",
  "description": "One-line description of this ontology's purpose.",
  "entity_types": [
    { "name": "contacts",  "folder": "memories/contacts",  "index_type": "human"   },
    { "name": "accounts",  "folder": "memories/accounts",  "index_type": "company" },
    { "name": "topics",    "folder": "memories/topics",    "index_type": "concept" }
  ],
  "timeline_enabled": true,
  "user_entity_enabled": true
}
```

| Field | Required | Description |
|---|---|---|
| `name` | ✅ | Short identifier (matches directory name for built-ins) |
| `entity_types` | ✅ | Non-empty list of entity type objects |
| `entity_types[].name` | ✅ | Type name used in agent prompts and API calls |
| `entity_types[].folder` | ✅ | Relative path to the entity directory inside the worktree |
| `entity_types[].index_type` | ✅ | Vocabulary term used in SEMANTIC INDEX `type` field |
| `description` | — | Human-readable description |
| `timeline_enabled` | — | Defaults to `true` |
| `user_entity_enabled` | — | Always `true` (user entity is always required) |

### `index_type` vocabulary

Use these values in `index_type` (they feed into the `build_index` prompt's type enum):

`human` · `project` · `company` · `decision` · `commitment` · `location` · `concept`

You can introduce new values — just make sure your `build_index.txt` prompt override
lists them in its schema documentation.

---

## Contributing a New Ontology

Community-contributed ontologies are welcome as pull requests. To add one:

1. Create `ontologies/<your-name>/` with `schema.json` and `repo_guide.md`.
2. Add a `prompts/` subdirectory with any prompts that differ from `personal`.
3. Test it: `DIFFMEM_ONTOLOGY=<your-name> python -c "from diffmem.ontology import load_ontology; load_ontology()"`
4. Open a PR. The description should explain the use case this ontology targets.

**What makes a good ontology contribution:**
- A clearly distinct use case that isn't well-served by existing profiles
- `repo_guide.md` that thoroughly documents the entity types and templates
- Prompt overrides that are tightly scoped to what actually needs to change
- A brief entry in this README's table

Ideas for future community profiles:
- `research` — papers, authors, concepts, citations
- `support` — tickets, customers, issues, resolutions
- `creative` — characters, scenes, plotlines, worldbuilding
- `health` — conditions, treatments, providers, timelines
