# CAPABILITY: Ontology profile loader — resolves DIFFMEM_ONTOLOGY env var to a
#             fully-validated OntologyProfile used by writer, onboarding, and retrieval agents.
# INPUTS:  name_or_path (str) — built-in name (e.g. "personal", "corporate") OR absolute path
# OUTPUTS: OntologyProfile dataclass
# CONSTRAINTS:
#   - schema.json and repo_guide.md are REQUIRED in every ontology (no fallback)
#   - Prompt files fall back to writer_agent/prompts/ if absent in the ontology dir
#   - Unknown built-in name raises ValueError at startup (never silently falls back)

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Locate the built-in ontologies/ directory.
# ontologies/ lives at src/diffmem/ontologies/ (co-located with the package).
# This ensures it is always reachable via a relative path from this file,
# whether running from the repo, installed via pip, or installed with --prefix.
# Path: src/diffmem/ontology/loader.py  → parent = src/diffmem/ontology/
#        parent.parent = src/diffmem/  →  / "ontologies" = src/diffmem/ontologies/
_ONTOLOGIES_DIR = Path(__file__).parent.parent / "ontologies"

# Fallback prompts directory (writer_agent default prompts)
_DEFAULT_PROMPTS_DIR = Path(__file__).parent.parent / "writer_agent" / "prompts"


@dataclass
class OntologyProfile:
    """Resolved, validated ontology profile ready for use by agents."""

    name: str
    schema: Dict[str, Any]            # parsed schema.json
    prompts_dir: Optional[Path]        # ontology-specific prompts dir (may be None)
    fallback_prompts_dir: Path         # writer_agent/prompts/ — always valid
    repo_guide_path: Path              # path to this ontology's repo_guide.md

    # Derived convenience helpers
    @property
    def entity_types(self) -> List[Dict[str, str]]:
        """List of entity type dicts: [{name, folder, index_type}, ...]"""
        return self.schema["entity_types"]

    @property
    def folder_map(self) -> Dict[str, str]:
        """Maps entity type name → folder path string (relative to worktree root)."""
        return {et["name"]: et["folder"] for et in self.entity_types}

    def entity_dirs(self, repo_root: Path) -> List[Path]:
        """Absolute paths to every entity folder for this ontology under repo_root.
        Use this instead of a hardcoded 'memories/' prefix for scanning."""
        return [repo_root / et["folder"] for et in self.entity_types]

    def default_folder(self, repo_root: Path) -> Path:
        """Fallback entity folder when an entity type is unknown.
        Returns the first entity type's folder (safer than a hardcoded 'memories/')."""
        return repo_root / self.entity_types[0]["folder"]

    def contexts_folder(self, repo_root: Path) -> Path:
        """Folder for orphan-theme extraction by the redistribute consolidator tool.
        Configured via schema.json 'contexts_folder'; defaults to the first entity
        type's folder if not specified (never falls back to a hardcoded path)."""
        rel = self.schema.get("contexts_folder") or self.entity_types[0]["folder"]
        return repo_root / rel

    @property
    def followups_enabled(self) -> bool:
        """Whether to build followups.md on every session. Set per-ontology in schema.json."""
        return bool(self.schema.get("followups_enabled", False))

    def followups_source_dir(self, repo_root: Path) -> Optional[Path]:
        """Folder scanned to build the Active Commitments section of followups.md.
        Returns None if followups are disabled or no source folder is configured."""
        if not self.followups_enabled:
            return None
        rel = self.schema.get("followups_source_folder")
        if not rel:
            return None
        return repo_root / rel

    @property
    def index_type_vocab(self) -> List[str]:
        """Unique index_type values for this ontology (fed into build_index prompt)."""
        seen = []
        for et in self.entity_types:
            if et["index_type"] not in seen:
                seen.append(et["index_type"])
        return seen

    def resolve_prompt(self, prompt_name: str) -> Path:
        """
        Returns the path to a prompt file, applying the fallback rule:
        1. Check ontology-specific prompts_dir (if set)
        2. Fall back to writer_agent/prompts/
        Raises FileNotFoundError only if the prompt is missing from both locations.
        """
        if self.prompts_dir is not None:
            candidate = self.prompts_dir / f"{prompt_name}.txt"
            if candidate.exists():
                return candidate
        fallback = self.fallback_prompts_dir / f"{prompt_name}.txt"
        if fallback.exists():
            return fallback
        raise FileNotFoundError(
            f"Prompt '{prompt_name}.txt' not found in ontology '{self.name}' "
            f"or fallback dir '{self.fallback_prompts_dir}'"
        )


def _validate_schema(schema: Dict[str, Any], source: Path) -> None:
    """Raises ValueError if schema is missing required fields."""
    required_keys = ["name", "entity_types"]
    for key in required_keys:
        if key not in schema:
            raise ValueError(
                f"schema.json at '{source}' is missing required field '{key}'"
            )
    if not isinstance(schema["entity_types"], list) or not schema["entity_types"]:
        raise ValueError(
            f"schema.json at '{source}': 'entity_types' must be a non-empty list"
        )
    for i, et in enumerate(schema["entity_types"]):
        for field_name in ("name", "folder", "index_type"):
            if field_name not in et:
                raise ValueError(
                    f"schema.json at '{source}': entity_types[{i}] missing '{field_name}'"
                )


def _load_from_dir(profile_dir: Path, name: str) -> OntologyProfile:
    """Load and validate an ontology profile from a directory."""
    schema_path = profile_dir / "schema.json"
    if not schema_path.exists():
        raise ValueError(
            f"Ontology '{name}': required file 'schema.json' not found at '{schema_path}'"
        )

    repo_guide_path = profile_dir / "repo_guide.md"
    if not repo_guide_path.exists():
        raise ValueError(
            f"Ontology '{name}': required file 'repo_guide.md' not found at '{repo_guide_path}'"
        )

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    _validate_schema(schema, schema_path)

    prompts_dir = profile_dir / "prompts"
    resolved_prompts_dir = prompts_dir if prompts_dir.exists() else None

    return OntologyProfile(
        name=name,
        schema=schema,
        prompts_dir=resolved_prompts_dir,
        fallback_prompts_dir=_DEFAULT_PROMPTS_DIR,
        repo_guide_path=repo_guide_path,
    )


def load_ontology(name_or_path: Optional[str] = None) -> OntologyProfile:
    """
    Resolve a DIFFMEM_ONTOLOGY value to a validated OntologyProfile.

    Resolution order:
      1. If name_or_path is None or empty, read DIFFMEM_ONTOLOGY env var (default "personal")
      2. If the value is an absolute path → load from that directory
      3. Otherwise → look up in ontologies/{name}/ inside the package

    Raises ValueError on unknown built-in name or invalid schema.
    Raises ValueError if schema.json or repo_guide.md are missing.
    """
    if not name_or_path:
        name_or_path = os.getenv("DIFFMEM_ONTOLOGY", "personal")

    path = Path(name_or_path)
    if path.is_absolute():
        if not path.is_dir():
            raise ValueError(
                f"DIFFMEM_ONTOLOGY absolute path '{name_or_path}' does not exist or is not a directory"
            )
        return _load_from_dir(path, name=path.name)

    # Built-in name lookup
    builtin_dir = _ONTOLOGIES_DIR / name_or_path
    if not builtin_dir.is_dir():
        available = sorted(p.name for p in _ONTOLOGIES_DIR.iterdir() if p.is_dir())
        raise ValueError(
            f"Unknown ontology '{name_or_path}'. "
            f"Built-in ontologies available: {available}. "
            f"Or set DIFFMEM_ONTOLOGY to an absolute path for a custom ontology."
        )

    return _load_from_dir(builtin_dir, name=name_or_path)
