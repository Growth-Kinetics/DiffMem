
### 3. Refactored `writer_agent/agent.py`
# CAPABILITY: Process session transcripts to create and update memory files, staging all changes.
# INPUTS: Memory input (str), user_id (str), repo_path (str)
# OUTPUTS: Staged file changes in the Git working directory.
# CONSTRAINTS: Uses OpenRouter via OpenAI lib. Prompts are abstracted to files.

import git
import json
import logging
import math
import re
import unicodedata
from difflib import SequenceMatcher
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

from diffmem.ontology.loader import OntologyProfile, load_ontology

# --- name normalization + fuzzy matching --------------------------------------
#
# WHY: the writer's entity resolution used to be exact-lowercase on index
# names/aliases plus a computed filename. Spelling/nickname variants
# ("Benjamin-Powell", "Benjamen Powell", "benjimin") missed and spawned
# duplicate entity files — the #1 duplicate-source in ChatBarry production.
# Two deterministic tiers fix it without LLM calls: key normalization
# (punctuation/diacritics/whitespace fold) and a similarity tier for typos.


def _normalize_name(name: str) -> str:
    """Fold a name to a canonical lookup key: lowercase, diacritics stripped
    (NFKD), all non-alphanumerics removed, whitespace collapsed.
    "Jean-Pierre Ó Sé" → "jeanpierreose"."""
    if not isinstance(name, str):
        return ""
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_folded = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    collapsed = re.sub(r"\s+", " ", ascii_folded.strip().lower())
    return re.sub(r"[^a-z0-9 ]", "", collapsed).replace(" ", "")


FUZZY_NAME_THRESHOLD = 0.85  # mirrors the dedupe prefilter's same-name notion


def _as_str(value: Any) -> str:
    """Coerce an LLM-produced name/type scalar to str. Lists are space-joined
    (the LLM occasionally returns e.g. name: ["Maya", "Chen"]; calling .lower()
    on it crashed ingest jobs on the VPS — 2026-08-18 rebuild incident)."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(str(v).strip() for v in value if str(v).strip())
    if value is None:
        return ""
    return str(value)

class WriterAgent:
    """Orchestrates the process of updating memory files based on a session."""

    def __init__(self, repo_path: str, user_id: str, openrouter_api_key: str, model: Optional[str] = None, max_concurrent_llm_calls: int = 8, validate_paths: bool = True, ontology: Optional[OntologyProfile] = None):
        self.repo_path = Path(repo_path)
        self.user_id = user_id
        self.user_path = self.repo_path
        self.user_file = self.user_path / f"{user_id}.md"  # Core user file at root of user folder
        # Ontology drives folder structure, entity vocabulary, and prompt resolution
        self.ontology: OntologyProfile = ontology if ontology is not None else load_ontology()
        # NOTE: do NOT use self.memories_path for scanning — it only covers the personal
        # ontology. Use self._entity_md_files() to iterate all entity files correctly.
        self.memories_path = self.user_path / "memories"  # kept for personal-ontology backwards compat
        self.prompts_path = Path(__file__).parent / "prompts"  # fallback; use self.ontology.resolve_prompt()
        self.max_concurrent_llm_calls = max_concurrent_llm_calls  # Configurable concurrency limit
        if not model:
            raise ValueError("model must be set via argument or DEFAULT_MODEL env var")
        self.model = model

        if validate_paths:
            if not self.user_path.exists():
                raise FileNotFoundError(f"User path (worktree) not found: {self.user_path}")
            if not self.user_file.exists():
                raise FileNotFoundError(f"User file not found: {self.user_file}")
            self.repo = git.Repo(self.repo_path)
        else:
            self.repo = None

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def _load_prompt(self, prompt_name: str) -> str:
        """Loads a prompt template, checking ontology-specific dir first, then writer_agent/prompts/ fallback."""
        prompt_file = self.ontology.resolve_prompt(prompt_name)
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read()

    def _entity_md_files(self):
        """Yield all .md files across every ontology entity dir, skipping sessions/.
        Replaces hardcoded self.memories_path.rglob('*.md') so corporate and future
        ontologies with different folder roots are covered correctly."""
        for entity_dir in self.ontology.entity_dirs(self.repo_path):
            if not entity_dir.exists():
                continue
            for md_file in entity_dir.rglob('*.md'):
                if '/sessions/' not in md_file.as_posix():
                    yield md_file

    def _get_relative_entity_path(self, file_path: Path) -> str:
        """
        Calculates the canonical relative path from user_path root.
        This ensures deterministic, consistent paths across the system.

        Args:
            file_path: Absolute or relative Path object to an entity file

        Returns:
            Standardized relative path string (e.g., 'memories/people/alex.md')
        """
        # Ensure we have an absolute path
        if not file_path.is_absolute():
            file_path = file_path.resolve()

        # Get relative path from user_path root
        try:
            rel_path = file_path.relative_to(self.user_path)
            # Convert to forward slashes for consistency (cross-platform)
            return str(rel_path).replace('\\', '/')
        except ValueError:
            # File is outside user_path, log warning and return name only
            self.logger.warning(f"File {file_path} is outside user_path {self.user_path}")
            return f"memories/unknown/{file_path.name}"

    def _call_llm(self, system_prompt: str, prompt: str, is_json: bool = True, model = None) -> Any:
        """Calls the LLM via OpenRouter, enforcing JSON mode if requested."""
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        if model == None:
            model = self.model
        try:
            self.logger.info(f"Calling LLM (model: {model}, JSON mode: {is_json})...")
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.15,
                response_format={"type": "json_object"} if is_json else None,
            )
            content = response.choices[0].message.content
            return json.loads(content) if is_json else content
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return {} if is_json else ""

    def _identify_relevant_entities(self, memory_input: str) -> Dict:
        """STEP 1: Identifies all entities mentioned in the session content and categorizes them."""
        self.logger.info("STEP 1: Identifying relevant entities...")
        semantic_index_path = self.user_path / 'index.md'

        with open(semantic_index_path, 'r', encoding='utf-8') as f:
            semantic_index = f.read()

        # Truncate the index to stay under the model's context window. index.md
        # is sorted by memory_strength (highest first), so we keep the top
        # entries and drop the long tail. ~60K chars ≈ ~15K tokens, leaving
        # ~110K for the memory_input + prompt overhead on a 128K-token model.
        MAX_INDEX_CHARS = 60_000
        if len(semantic_index) > MAX_INDEX_CHARS:
            # Cut at a line boundary to avoid mid-JSON truncation.
            cut = semantic_index.rfind('\n', 0, MAX_INDEX_CHARS)
            if cut < MAX_INDEX_CHARS * 0.8:
                cut = MAX_INDEX_CHARS  # fallback: hard cut
            total = semantic_index.count('\n### ')
            shown = semantic_index[:cut].count('\n### ')
            semantic_index = (
                semantic_index[:cut]
                + f"\n--- (index truncated: showing top {shown} of {total} entities by memory strength) ---\n"
            )
            self.logger.info(
                "INDEX_TRUNCATED: %d/%d entities (%d → %d chars)",
                shown, total, len(semantic_index), cut,
            )

        system_prompt = self._load_prompt("0_system")
        prompt_template = self._load_prompt("1_identify_entities")
        prompt = prompt_template.format(
            semantic_index=semantic_index,
            memory_input=memory_input
        )

        response = self._call_llm(system_prompt, prompt, is_json=True)
        if not response:
            self.logger.info("No entities identified.")
            return {"entities_to_create": [], "entities_to_update": []}

        entities_to_create = response.get('entities_to_create', [])
        entities_to_update = response.get('entities_to_update', [])
        # Defensive coercion: LLMs occasionally return int counts instead of
        # arrays (e.g. {"entities_to_create": 3} instead of [...]). Without
        # this, len(int) raises TypeError and kills the ingest job.
        if not isinstance(entities_to_create, list):
            entities_to_create = []
        if not isinstance(entities_to_update, list):
            entities_to_update = []
        response['entities_to_create'] = entities_to_create
        response['entities_to_update'] = entities_to_update

        self.logger.info(f"Identified {len(entities_to_create)} new entities and {len(entities_to_update)} entities to update")
        return response

    def _create_single_entity(self, entity: Dict, memory_input: str, example_content: str, example_file_name: str, system_prompt: str) -> Dict:
        """Helper method to create a single entity file (for parallel execution)."""
        try:
            creation_prompt_template = self._load_prompt("2_create_entity_file")
            creation_prompt = creation_prompt_template.format(
                example_file_name=example_file_name,
                example_content=example_content,
                entity_name=entity['name'],
                entity_summary=entity['summary'],
                memory_input=memory_input
            )

            new_file_content = self._call_llm(system_prompt, creation_prompt, is_json=False)

            # Resolve folder from ontology schema - falls back to memories/{type} if type unknown
            folder_map = self.ontology.folder_map
            entity_type = entity['type']
            rel_folder = folder_map.get(entity_type) or self.ontology.default_folder(self.repo_path).relative_to(self.repo_path).as_posix()
            target_dir = self.repo_path / rel_folder
            target_dir.mkdir(parents=True, exist_ok=True)

            file_name = _as_str(entity['name']).lower().replace(' ', '_').replace('.', '') + '.md'
            new_file_path = target_dir / file_name

            return {
                'success': True,
                'entity_name': entity['name'],
                'file_path': new_file_path,
                'content': new_file_content,
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'entity_name': entity.get('name', 'unknown'),
                'file_path': None,
                'content': None,
                'error': str(e)
            }

    def _create_new_entities(self, memory_input: str, entities_to_create: List[Dict]):
        """STEP 2: Creates files for new entities in parallel."""
        if not entities_to_create:
            self.logger.info("No new entities to create.")
            return

        self.logger.info(f"Creating {len(entities_to_create)} new entity files in parallel...")

        # Use the main user file as the example
        example_file_path = self.user_file
        with open(example_file_path, 'r', encoding='utf-8') as f:
            example_content = f.read()

        system_prompt = self._load_prompt("0_system")

        # Process entities in parallel
        max_workers = min(len(entities_to_create), self.max_concurrent_llm_calls)  # Limit concurrent LLM calls
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all entity creation tasks
            future_to_entity = {
                executor.submit(
                    self._create_single_entity,
                    entity,
                    memory_input,
                    example_content,
                    example_file_path.name,
                    system_prompt
                ): entity for entity in entities_to_create
            }

            # Collect results as they complete
            for future in as_completed(future_to_entity):
                entity = future_to_entity[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to create entity {entity.get('name', 'unknown')}: {e}")
                    results.append({
                        'success': False,
                        'entity_name': entity.get('name', 'unknown'),
                        'error': str(e)
                    })

        # Write all successful results to files
        successful_creates = 0
        for result in results:
            if result['success']:
                with open(result['file_path'], 'w', encoding='utf-8') as f:
                    f.write(result['content'])
                self.logger.info(f"ENTITY_CREATED: Staged new file at {result['file_path']}")
                successful_creates += 1
            else:
                self.logger.error(f"Failed to create entity {result['entity_name']}: {result['error']}")

        self.logger.info(f"Successfully created {successful_creates}/{len(entities_to_create)} entities in parallel")

    def _find_text_position(self, content: str, search_text: str, fuzzy: bool = True) -> int:
        """Finds the position of search_text in content, with optional fuzzy matching.

        Returns the starting position of the match, or -1 if not found.
        """
        # First try exact match
        position = content.find(search_text)
        if position != -1:
            return position

        if not fuzzy:
            return -1

        # Try with normalized whitespace (collapse multiple spaces/newlines)
        import re
        normalized_content = re.sub(r'\s+', ' ', content)
        normalized_search = re.sub(r'\s+', ' ', search_text)

        position = normalized_content.find(normalized_search)
        if position != -1:
            # Map back to original position (approximate)
            # This is a simplified mapping - could be made more precise
            return self._map_normalized_position(content, position)

        return -1

    def _map_normalized_position(self, original: str, normalized_pos: int) -> int:
        """Maps a position in normalized text back to the original text."""
        import re
        # Simple approach: count characters up to the normalized position
        char_count = 0
        normalized_count = 0

        for i, char in enumerate(original):
            if re.match(r'\s', char):
                if normalized_count < normalized_pos and not (i > 0 and re.match(r'\s', original[i-1])):
                    normalized_count += 1
            else:
                if normalized_count < normalized_pos:
                    normalized_count += 1

            if normalized_count >= normalized_pos:
                return i

        return len(original)

    def _update_single_file(self, file_path: Path, memory_input: str, system_prompt: str, prompt_template: str) -> Dict:
        """Helper method to update a single file (for parallel execution)."""
        try:
            if not file_path.is_file() or 'repo_guide' in str(file_path) or 'index' in str(file_path):
                return {
                    'success': False,
                    'file_path': file_path,
                    'error': 'File not eligible for update',
                    'updates_applied': 0,
                    'total_updates': 0
                }

            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            prompt = prompt_template.format(
                file_path_name=file_path.name,
                file_content=original_content,
                memory_input=memory_input
            )

            response = self._call_llm(system_prompt, prompt, is_json=True)
            updates = response.get('updates', [])
            if not updates:
                return {
                    'success': True,
                    'file_path': file_path,
                    'error': None,
                    'updates_applied': 0,
                    'total_updates': 0,
                    'modified_content': None
                }

            # Apply updates using search and replace
            modified_content = original_content
            successful_updates = 0

            for update in updates:
                operation = update.get('operation', 'replace')
                search_text = update.get('search_text', '')
                replacement_text = update.get('replacement_text', '')

                if operation == 'replace':
                    if search_text in modified_content:
                        # Replace only the first occurrence to maintain precision
                        modified_content = modified_content.replace(search_text, replacement_text, 1)
                        successful_updates += 1
                    else:
                        self.logger.warning(f"Could not find text to replace in {file_path.name}: {search_text[:50]}...")

                elif operation == 'insert_after':
                    if search_text in modified_content:
                        # Insert the new text after the search text
                        insert_position = modified_content.find(search_text) + len(search_text)
                        # Ensure we add a newline if not present
                        separator = '\n' if not search_text.endswith('\n') else ''
                        modified_content = (modified_content[:insert_position] +
                                          separator + replacement_text +
                                          modified_content[insert_position:])
                        successful_updates += 1
                    else:
                        self.logger.warning(f"Could not find insertion point in {file_path.name}: {search_text[:50]}...")

                elif operation == 'append':
                    # Append to the end of the file
                    separator = '\n' if not modified_content.endswith('\n') else ''
                    modified_content = modified_content + separator + replacement_text
                    successful_updates += 1

            return {
                'success': True,
                'file_path': file_path,
                'error': None,
                'updates_applied': successful_updates,
                'total_updates': len(updates),
                'modified_content': modified_content if successful_updates > 0 else None
            }

        except Exception as e:
            return {
                'success': False,
                'file_path': file_path,
                'error': str(e),
                'updates_applied': 0,
                'total_updates': 0
            }

    def _fuzzy_index_match(
        self, entity_name: str, index_lookup: Dict[str, str]
    ) -> Optional[Tuple[float, str, str]]:
        """Best fuzzy match of `entity_name` against the index lookup keys.

        Returns (score, matched_key, file_path) for the best candidate with
        similarity ≥ FUZZY_NAME_THRESHOLD (or a stem-containment match — the
        same disambiguator heuristic the dedupe prefilter uses), else None.
        Normalized comparison; deterministic; no LLM.
        """
        query = _normalize_name(entity_name)
        if not query or not index_lookup:
            return None
        best: Optional[Tuple[float, str, str]] = None
        for key, rel in index_lookup.items():
            candidate = _normalize_name(key)
            if not candidate:
                continue
            score = SequenceMatcher(None, query, candidate).ratio()
            # Stem containment: one slug containing the other (e.g. the dedupe
            # disambiguator case "maya" inside "maya_chen"). Both directions.
            # Min-length 4 guard stops tiny fragments ("ai") matching everything.
            if (
                query != candidate
                and min(len(query), len(candidate)) >= 4
                and (query in candidate or candidate in query)
            ):
                score = max(score, FUZZY_NAME_THRESHOLD + 0.01)  # just over the bar
            if score >= FUZZY_NAME_THRESHOLD and (best is None or score > best[0]):
                best = (score, key, rel)
        return best

    def _load_master_index_lookup(self) -> Dict[str, str]:
        """
        Loads the master index and creates a name->path lookup dict.
        Handles aliases and name variations.

        Each entity's name and aliases are indexed under BOTH exact-lowercase
        and normalized keys (see _normalize_name) so punctuation, casing,
        diacritics, and whitespace variants resolve deterministically — the
        exact-lower map alone let "Benjamin-Powell" / "benjamin powell" miss
        and spawn duplicate files.

        Returns:
            Dict mapping entity names (and aliases) to file paths
        """
        index_file = self.user_path / 'index.md'
        lookup = {}

        if not index_file.exists():
            return lookup

        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse entity entries - look for JSON blocks after entity headers
            import re
            import ast

            # Pattern to match entity sections with JSON
            pattern = r'### (.+?)\n.*?```\{(.+?)\}```'
            matches = re.findall(pattern, content, re.DOTALL)

            for entity_title, json_str in matches:
                try:
                    # Clean up the JSON string and parse
                    json_str = json_str.strip()
                    if not json_str.startswith('{'):
                        json_str = '{' + json_str
                    if not json_str.endswith('}'):
                        json_str = json_str + '}'

                    entity_data = ast.literal_eval(json_str)

                    if 'name' in entity_data and 'file' in entity_data:
                        # Map primary name (exact-lower + normalized). Guarded:
                        # a stale index.md can carry a list-valued name from a
                        # pre-normalization build (VPS incident 2026-08-18).
                        name_str = _as_str(entity_data['name'])
                        if name_str:
                            lookup[name_str.lower()] = entity_data['file']
                            lookup[_normalize_name(name_str)] = entity_data['file']

                        # Map all aliases (exact-lower + normalized)
                        for alias in entity_data.get('aliases', []):
                            if not isinstance(alias, str):
                                continue  # tolerate malformed LLM output
                            lookup[alias.lower()] = entity_data['file']
                            lookup[_normalize_name(alias)] = entity_data['file']

                except Exception as e:
                    self.logger.debug(f"Could not parse entity in index: {e}")
                    continue

        except Exception as e:
            self.logger.warning(f"Could not load master index for lookup: {e}")

        return lookup

    def _resolve_entity_file_path(self, entity_name: str, entity_type: str) -> Optional[Path]:
        """
        Resolves an entity's file path using master index first, then filesystem fallback.

        Args:
            entity_name: Name of the entity (e.g., "Benjamin Powell")
            entity_type: Type of entity (e.g., "people", "contexts", "events")

        Returns:
            Path object if found, None otherwise
        """
        # Strategy 1: Look up in master index (handles aliases and exact names)
        index_lookup = self._load_master_index_lookup()
        # Identify-LLM responses can carry non-str names (lists, numbers) —
        # coerce once; every use below is a string op.
        entity_name = _as_str(entity_name)
        if not entity_name:
            self.logger.warning("ENTITY_RESOLVE_SKIPPED: empty/coercion-failed name")
            return None
        entity_name_lower = entity_name.lower()

        if entity_name_lower in index_lookup:
            index_path = self.user_path / index_lookup[entity_name_lower]
            if index_path.exists():
                self.logger.debug(f"ENTITY_RESOLVED_INDEX: {entity_name} → {index_path}")
                return index_path
            else:
                self.logger.warning(f"ENTITY_INDEX_STALE: {entity_name} index points to {index_path} but file not found")

        # Strategy 2: Try computed filename using ontology folder map
        expected_filename = entity_name.lower().replace(' ', '_').replace('.', '') + '.md'
        folder_map = self.ontology.folder_map
        # Look up by exact type name, then try singular/plural variants
        _et = entity_type.lower()
        _singular = _et[:-1] if _et.endswith('s') else _et  # safe suffix strip (not rstrip)
        _plural = _et if _et.endswith('s') else _et + 's'
        rel_folder = (
            folder_map.get(_et)
            or folder_map.get(_singular)
            or folder_map.get(_plural)
            or self.ontology.default_folder(self.repo_path).relative_to(self.repo_path).as_posix()
        )
        search_dir = self.repo_path / rel_folder

        entity_file = search_dir / expected_filename
        if entity_file.exists():
            self.logger.debug(f"ENTITY_RESOLVED_COMPUTED: {entity_name} → {entity_file}")
            return entity_file

        # Strategy 2b: Normalized index lookup (punctuation / diacritics /
        # whitespace variants map to the same key — deterministic, no LLM).
        normalized = _normalize_name(entity_name)
        if normalized in index_lookup:
            index_path = self.user_path / index_lookup[normalized]
            if index_path.exists():
                self.logger.info(
                    "ENTITY_RESOLVED_NORMALIZED: %s (norm=%s) → %s",
                    entity_name, normalized, index_path,
                )
                return index_path

        # Strategy 2c: Fuzzy match over index names + aliases. Catches typos
        # and nicknames the exact/normalized tiers miss ("Benjamen" →
        # "Benjamin", ratio 0.96). Threshold mirrors the dedupe prefilter's
        # notion of "same name"; stem containment reuses the dedupe
        # disambiguator heuristic (a slug that contains the other).
        fuzzy_hit = self._fuzzy_index_match(entity_name, index_lookup)
        if fuzzy_hit is not None:
            score, matched_key, rel = fuzzy_hit
            index_path = self.user_path / rel
            if index_path.exists():
                self.logger.info(
                    "ENTITY_RESOLVED_FUZZY: %s ≈ %s (score=%.2f) → %s",
                    entity_name, matched_key, score, index_path,
                )
                return index_path

        # Strategy 3: Fuzzy search across all ontology entity dirs
        for md_file in self._entity_md_files():
            if md_file.stem.lower() == entity_name.lower().replace(' ', '_').replace('.', ''):
                self.logger.debug(f"ENTITY_RESOLVED_FUZZY: {entity_name} → {md_file}")
                return md_file

        self.logger.warning(f"ENTITY_NOT_FOUND: Could not locate file for entity '{entity_name}' (type: {entity_type})")
        return None

    def _update_existing_entities(self, memory_input: str, entities_to_update: List[Dict]):
        """STEP 2: Updates only the identified existing entity files in parallel."""
        if not entities_to_update:
            self.logger.info("No entities to update.")
            return

        self.logger.info(f"Updating entities for {len(entities_to_update)} existing entities in parallel...")

        prompt_template = self._load_prompt("3_update_entity_file")
        system_prompt = self._load_prompt("0_system")

        # Build list of files to update - resolve paths from entity names
        files_to_update = [self.user_file]

        for entity in entities_to_update:
            entity_name = entity.get('name', '')
            entity_type = entity.get('type', 'contexts')

            if not entity_name:
                self.logger.warning(f"Entity missing name field: {entity}")
                continue

            # Resolve file path deterministically from filesystem
            file_path = self._resolve_entity_file_path(entity_name, entity_type)

            if file_path:
                files_to_update.append(file_path)
                self.logger.debug(f"Resolved entity '{entity_name}' to {file_path}")
            else:
                self.logger.warning(f"Could not resolve file path for entity: {entity_name}")

        unique_files = list(set(files_to_update))  # Remove duplicates
        if not unique_files:
            self.logger.info("No files to update.")
            return

        # Process files in parallel
        max_workers = min(len(unique_files), self.max_concurrent_llm_calls)  # Limit concurrent LLM calls for updates
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all file update tasks
            future_to_file = {
                executor.submit(
                    self._update_single_file,
                    file_path,
                    memory_input,
                    system_prompt,
                    prompt_template
                ): file_path for file_path in unique_files
            }

            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to update file {file_path}: {e}")
                    results.append({
                        'success': False,
                        'file_path': file_path,
                        'error': str(e)
                    })

        # Write all successful results to files
        total_files_updated = 0
        total_updates_applied = 0

        for result in results:
            if result['success'] and result.get('modified_content'):
                with open(result['file_path'], 'w', encoding='utf-8') as f:
                    f.write(result['modified_content'])
                self.logger.info(f"FILE_UPDATED: Applied {result['updates_applied']}/{result['total_updates']} updates to {result['file_path'].name}")
                total_files_updated += 1
                total_updates_applied += result['updates_applied']
            elif result['success'] and result['updates_applied'] == 0:
                self.logger.debug(f"No updates needed for {result['file_path'].name}")
            elif not result['success']:
                self.logger.error(f"Failed to update {result['file_path']}: {result['error']}")

        self.logger.info(f"Successfully updated {total_files_updated} files with {total_updates_applied} total updates in parallel")

    def _create_timeline_entry(self, session_id: str, session_date: str, memory_input: str):
        """STEP 3: Diffs staged changes and creates a timeline entry."""
        self.logger.info("STEP 3: Creating timeline entry from diff and conversation...")
        diff_text = self.repo.git.diff(self.repo.head.commit)
        if not diff_text:
            self.logger.info("No changes detected, skipping timeline entry.")
            return
        system_prompt = self._load_prompt("0_system")
        prompt_template = self._load_prompt("4_create_timeline_entry")
        prompt = prompt_template.format(
            diff_text=diff_text,
            session_id=session_id,
            session_date=session_date,
            memory_input=memory_input
        )

        timeline_entry = self._call_llm(system_prompt, prompt, is_json=False)

        # Create timeline directory if it doesn't exist
        timeline_dir = self.user_path / 'timeline'
        timeline_dir.mkdir(parents=True, exist_ok=True)

        # Parse session_date to get the year-month for filename
        try:
            session_datetime = datetime.strptime(session_date, '%Y-%m-%d')
            timeline_filename = timeline_dir / f"{session_datetime.strftime('%Y-%m')}.md"
        except ValueError:
            # Fallback to current date if session_date is invalid
            self.logger.warning(f"Invalid session_date format: {session_date}, using current date")
            timeline_filename = timeline_dir / f"{datetime.now().strftime('%Y-%m')}.md"

        with open(timeline_filename, 'a', encoding='utf-8') as f:
            f.write("\n" + timeline_entry)
        self.logger.info(f"TIMELINE_UPDATED: Appended entry to {timeline_filename}")

    def _get_modified_files(self) -> List[Path]:
        """Gets list of markdown files that have been modified/created in this session."""
        # Get staged changes
        diff_text = self.repo.git.diff(self.repo.head.commit)
        if not diff_text:
            return []

        # Parse diff to find modified .md files in memories directory
        modified_files = []
        for line in diff_text.split('\n'):
            if line.startswith('diff --git'):
                # Extract file path from diff header
                parts = line.split(' ')
                if len(parts) >= 4:
                    file_path = parts[3][2:]  # Remove 'b/' prefix
                    full_path = self.repo_path / file_path
                    # Only include .md files in entity dirs (ontology-aware)
                    if (full_path.suffix == '.md' and
                        any(full_path.is_relative_to(d) for d in self.ontology.entity_dirs(self.repo_path)) and
                        full_path.exists()):
                        modified_files.append(full_path)

        return list(set(modified_files))  # Remove duplicates

    def _get_file_git_stats(self, file_path: Path) -> Dict[str, Any]:
        """Gets git statistics for a file: last_update and number_of_edits."""
        try:
            # Get relative path from repo root
            rel_path = file_path.relative_to(self.repo_path)

            # Check if the repo has any commits yet (handling fresh repos)
            try:
                self.repo.head.commit
            except ValueError:
                # Repo has no commits (empty/fresh)
                return {
                    'last_update': "New File",
                    'number_of_edits': 1  # Count creation as first edit
                }

            # Get last commit date for this file
            try:
                last_commit = self.repo.git.log('-1', '--format=%ci', str(rel_path))
                last_update = last_commit.strip() if last_commit else "Unknown"

                # Get number of commits that touched this file
                commit_count = self.repo.git.rev_list('--count', 'HEAD', '--', str(rel_path))
                number_of_edits = int(commit_count.strip()) if commit_count.strip() else 0
            except git.exc.GitCommandError as e:
                # If file is new and not committed yet, git log fails
                if "does not have any commits yet" in str(e) or "ambiguous argument" in str(e):
                    last_update = "New File"
                    number_of_edits = 1
                else:
                    raise e

            return {
                'last_update': last_update,
                'number_of_edits': number_of_edits
            }
        except Exception as e:
            self.logger.warning(f"Could not get git stats for {file_path}: {e}")
            return {
                'last_update': "Unknown",
                'number_of_edits': 0
            }

    def _calculate_memory_strength(self, number_of_edits: int, last_update: str) -> float:
        """Calculates memory strength score based on edit frequency and recency."""
        # Base score from edit frequency (logarithmic scaling)
        edit_score = math.log(max(1, number_of_edits)) / math.log(10)  # log10

        # Recency bonus (decay over time)
        try:
            if last_update != "Unknown":
                # Parse git date format: "2024-01-15 10:30:45 -0800"
                last_date = datetime.strptime(last_update[:19], "%Y-%m-%d %H:%M:%S")
                days_ago = (datetime.now() - last_date).days
                # Exponential decay with half-life of 30 days
                recency_score = math.exp(-days_ago / 30.0)
            else:
                recency_score = 0.1
        except Exception:
            recency_score = 0.1

        # Combined score (weighted average)
        memory_strength = (edit_score * 0.7) + (recency_score * 0.3)
        return round(memory_strength, 3)

    def _strip_existing_semantic_index(self, content: str) -> str:
        """Removes existing SEMANTIC INDEX section from file content."""
        lines = content.split('\n')
        result_lines = []
        in_semantic_index = False

        for line in lines:
            if line.strip().startswith('## SEMANTIC INDEX'):
                in_semantic_index = True
                continue
            elif in_semantic_index and line.strip().startswith('##') and not line.strip().startswith('## SEMANTIC INDEX'):
                # Hit next section, stop skipping
                in_semantic_index = False
                result_lines.append(line)
            elif not in_semantic_index:
                result_lines.append(line)

        return '\n'.join(result_lines)

    def _build_single_entity_index(self, file_path: Path,
                                   git_stats: Optional[Dict[str, Any]] = None) -> Dict:
        """Helper method to build semantic index for a single file (for parallel execution).

        IMPORTANT (thread-safety): GitPython's persistent `cat-file --batch-check`
        process is NOT thread-safe when a single Repo is shared across worker
        threads. Calling `self._get_file_git_stats(...)` from inside a
        ThreadPoolExecutor caused intermittent deadlocks where one worker would
        block forever on a pipe.readline() in git/cmd.py:__get_object_header.

        Callers in parallel paths MUST precompute git_stats in the parent
        thread and pass them in. The inline fallback below remains for direct
        callers; do not invoke this method concurrently without precomputed stats.
        """
        try:
            # Read current file content
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            # Strip existing semantic index if present
            content_without_index = self._strip_existing_semantic_index(original_content)

            # Use precomputed git stats when provided (thread-safe path);
            # fall back to live lookup for non-parallel callers.
            if git_stats is None:
                git_stats = self._get_file_git_stats(file_path)
            memory_strength = self._calculate_memory_strength(
                git_stats['number_of_edits'],
                git_stats['last_update']
            )

            # Calculate deterministic file path (system-computed, not LLM-generated)
            canonical_path = self._get_relative_entity_path(file_path)

            # Build semantic index using LLM
            prompt_template = self._load_prompt("build_index")
            prompt = prompt_template.format(
                file_content=content_without_index,
                file_path=canonical_path,
                last_update=git_stats['last_update'],
                number_of_edits=git_stats['number_of_edits'],
                memory_strength=memory_strength
            )

            # Get semantic index JSON from LLM
            semantic_index_data = self._call_llm("", prompt, is_json=True)

            # LLMs occasionally return nested lists for contractually-flat
            # cue/alias/related fields (e.g. hard_cues: ["a", ["b", "c"]]).
            # Normalize BEFORE persisting so poisoned shapes never enter the
            # store — downstream consumers (consolidator joins, set() filters,
            # master-index JSON) assume flat string lists and crash otherwise.
            # See frontmatter.normalize_semantic_index for the full rationale.
            from ..frontmatter import normalize_semantic_index
            semantic_index_data = normalize_semantic_index(semantic_index_data)

            # `file` is a path computed at read time (scan_entities sets it);
            # never persist it into frontmatter.
            semantic_index_data.pop("file", None)

            # v2: structured metadata lives in YAML frontmatter (merged), not a
            # trailing `## SEMANTIC INDEX` JSON block. merge_frontmatter also
            # strips any legacy trailing block (migration).
            from ..frontmatter import merge_frontmatter
            updated_content = merge_frontmatter(content_without_index, semantic_index_data)

            return {
                'success': True,
                'file_path': file_path,
                'content': updated_content,
                'error': None
            }

        except Exception as e:
            return {
                'success': False,
                'file_path': file_path,
                'content': None,
                'error': str(e)
            }

    def _build_entity_indexes(self, file_paths: List[Path]):
        """STEP 5: Builds semantic indexes for modified entity files in parallel.

        Thread-safety: git stats are computed SERIALLY in this (parent) thread
        before launching the executor. Worker threads receive precomputed stats
        and never touch `self.repo`. This eliminates the GitPython
        persistent-process deadlock that caused the writer agent to hang
        indefinitely on workloads with multiple modified entity files. See
        `_build_single_entity_index` docstring for the full rationale.
        """
        if not file_paths:
            self.logger.info("No files to index.")
            return

        # Pre-compute git stats serially in the parent thread. This is the
        # cheap part of the work; gitstats per file is microseconds, while the
        # LLM call (which IS what we parallelize) takes seconds.
        self.logger.info(
            f"Pre-computing git stats for {len(file_paths)} files (serial)..."
        )
        precomputed_stats: Dict[Path, Dict[str, Any]] = {
            fp: self._get_file_git_stats(fp) for fp in file_paths
        }

        self.logger.info(
            f"Building semantic indexes for {len(file_paths)} modified files in parallel..."
        )

        # Process files in parallel (LLM calls only; no shared git access)
        max_workers = min(len(file_paths), self.max_concurrent_llm_calls)
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all indexing tasks with precomputed stats
            future_to_file = {
                executor.submit(
                    self._build_single_entity_index, file_path, precomputed_stats[file_path]
                ): file_path
                for file_path in file_paths
            }

            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to build index for {file_path}: {e}")
                    results.append({
                        'success': False,
                        'file_path': file_path,
                        'error': str(e)
                    })

        # Write all successful results to files
        successful_indexes = 0
        for result in results:
            if result['success']:
                with open(result['file_path'], 'w', encoding='utf-8') as f:
                    f.write(result['content'])
                self.logger.info(f"INDEX_UPDATED: Added semantic index to {result['file_path'].name}")
                successful_indexes += 1
            else:
                self.logger.error(f"Failed to build index for {result['file_path']}: {result['error']}")

        self.logger.info(f"Successfully built {successful_indexes}/{len(file_paths)} indexes in parallel")

    def _rebuild_master_index(self):
        """STEP 5: Rebuilds the master index.md file with all memory entities.

        Delegates to the shared `rebuild_master_index` (consolidator_agent._shared)
        which uses `extract_semantic_index` — handling BOTH v2 YAML frontmatter
        AND the legacy `## SEMANTIC INDEX` JSON block. The prior manual parser
        only read the legacy format, silently skipping every v2 entity and
        leaving index.md empty — the root cause of the identify step's
        duplicate-spawning behavior on entrepreneur-ontology stores.
        """
        from ..consolidator_agent._shared import extract_semantic_index, rebuild_master_index

        self.logger.info("STEP 5: Rebuilding master index.md...")

        # Self-heal: find files without a parseable semantic index and rebuild
        # them before generating the master index (preserves prior behavior).
        files_needing_index = []
        for md_file in self._entity_md_files():
            if md_file.name == 'index.md':
                continue
            try:
                content = md_file.read_text(encoding='utf-8')
                if extract_semantic_index(content) is None:
                    files_needing_index.append(md_file)
            except OSError:
                continue
        if files_needing_index:
            self.logger.info(
                "MASTER_INDEX_HEAL: rebuilding semantic index for %d file(s)",
                len(files_needing_index),
            )
            self._build_entity_indexes(files_needing_index)

        # Delegate to the shared rebuild — ontology-aware entity_dirs, both SI
        # formats, git stats + memory strength augmentation.
        try:
            import git as gitlib
            repo = gitlib.Repo(self.repo_path)
        except Exception:
            repo = None

        rebuild_master_index(
            self.user_path,
            self.user_id,
            repo=repo,
            entity_dirs=self.ontology.entity_dirs(self.repo_path),
        )

    def _parse_commitment_metadata(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Best-effort parse of a commitment file's ## Metadata block + display name.
        Returns dict with keys: title, status, owner (str), due (str), slug, path.
        Returns None for files whose Status reads completed/done/cancelled
        (those drop off the live followups list — history lives in git log).
        """
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError:
            return None

        slug = file_path.stem
        # Title from first heading line
        title = slug.replace("_", " ").title()
        for line in content.splitlines():
            if line.startswith("# "):
                # e.g. "# Commitment: Deliver Hikari Prediction by April 23"
                t = line[2:].strip()
                if ":" in t:
                    title = t.split(":", 1)[1].strip()
                else:
                    title = t
                break

        # Extract Metadata block — lines like "- **Status:** Active" or "- **Status**: Active"
        # (LLM output varies; tolerate both ** placements and missing colons.)
        meta: Dict[str, str] = {}
        in_metadata = False
        for line in content.splitlines():
            s = line.strip()
            if s.startswith("## Metadata"):
                in_metadata = True
                continue
            if in_metadata and s.startswith("## "):
                break
            if in_metadata and s.startswith("- "):
                # Strip the leading "- " then split on first colon outside the ** **.
                body = s[2:].lstrip()
                # Remove markdown bolding to simplify key matching.
                body_plain = body.replace("**", "")
                if ":" in body_plain:
                    k, v = body_plain.split(":", 1)
                    meta[k.strip().lower()] = v.strip()

        status = meta.get("status", "Active")
        # Drop completed/done/cancelled — they self-evict from the live list.
        # v2: canonicalize against the ontology's open_item enum (code owns the
        # decision; freeform prose like "done (previously active)" still maps to
        # a terminal state and is dropped).
        from ..status import canonicalize_status
        open_item_enum = (
            self.ontology.schema.get("status_enums", {}).get("open_item")
            if self.ontology is not None else None
        )
        canon = canonicalize_status(status, open_item_enum)
        if canon in {"done", "cancelled"}:
            return None
        # Keep the canonical form for downstream rendering.
        if canon is not None:
            status = canon

        # Owner — accept several common header variants the LLM produces
        owner = (
            meta.get("owner")
            or meta.get("assignee")
            or meta.get("assignees")
            or meta.get("deciders")
            or ""
        )
        due = meta.get("due date") or meta.get("due") or ""

        return {
            "slug": slug,
            "title": title,
            "status": status,
            "owner": owner,
            "due": due,
            "path": file_path,
        }

    def _read_existing_other_items(self, followups_path: Path) -> str:
        """Extract the existing ## Other Items section from followups.md (for prompt context).
        Returns the section body as a string, or '_(none)_' if missing/empty.
        """
        if not followups_path.exists():
            return "_(none)_"
        try:
            content = followups_path.read_text(encoding="utf-8")
        except OSError:
            return "_(none)_"
        marker = "## Other Items"
        if marker not in content:
            return "_(none)_"
        body = content.split(marker, 1)[1]
        # Stop at next heading if present
        for nh in ("\n## ", "\n# "):
            if nh in body:
                body = body.split(nh, 1)[0]
        body = body.strip()
        return body or "_(none)_"

    def _collect_open_items(self) -> List[Dict[str, Any]]:
        """Walk entity files and collect active `## Open Items` entries.

        Each entry: {title, status, owner, due, entity_path, entity_title}. Only entries
        whose canonical status is in {open, in_progress, blocked} are returned —
        done/cancelled self-evict (history lives in git).
        """
        from ..status import canonicalize_status
        from ..frontmatter import parse_frontmatter
        import re
        enum = (
            self.ontology.schema.get("status_enums", {}).get("open_item")
            if self.ontology is not None else None
        )
        active = {"open", "in_progress", "blocked"}
        items: List[Dict[str, Any]] = []
        entity_dirs = (
            self.ontology.entity_dirs(self.user_path)
            if self.ontology is not None
            else [self.user_path / "memories"]
        )
        for d in entity_dirs:
            if not d.exists():
                continue
            for md in sorted(d.rglob("*.md")):
                if md.name in {"index.md", "episodes_index.md"}:
                    continue
                try:
                    content = md.read_text(encoding="utf-8")
                except OSError:
                    continue
                fm, _ = parse_frontmatter(content)
                entity_title = (fm or {}).get("title") or md.stem
                # find the ## Open Items section
                if "## Open Items" not in content:
                    continue
                after = content.split("## Open Items", 1)[1]
                for line in after.splitlines():
                    s = line.strip()
                    if s.startswith("## "):  # next section
                        break
                    if not s.startswith("- "):
                        continue
                    # parse `- **[status]** title — details`
                    m = re.match(r"^- \*\*\[(?P<st>[^\]]+)\]\*\*\s*(?P<title>.*)$", s)
                    if not m:
                        continue
                    raw_status = m.group("st")
                    rest = m.group("title").strip()
                    canon = canonicalize_status(raw_status, enum)
                    if canon not in active:
                        continue  # terminal or unknown → self-evict
                    # crude owner/due extraction from the rest (best-effort)
                    owner = ""
                    om = re.search(r"assignee[:\s]*\[\[([^\]]+)\]\]", rest, re.I)
                    if om:
                        owner = f"[[{om.group(1)}]]"
                    due = ""
                    dm = re.search(r"due[:\s]*([0-9]{4}-[0-9]{2}-[0-9]{2}[A-Za-z0-9:-]*)", rest, re.I)
                    if dm:
                        due = dm.group(1)
                    title = re.split(r"\s+[—-]\s+|\s+assignee\s*[:=]", rest, maxsplit=1, flags=re.I)[0].strip()
                    if not title:
                        title = rest
                    items.append({
                        "title": title,
                        "status": canon,
                        "owner": owner,
                        "due": due,
                        "entity_path": md,
                        "entity_title": entity_title,
                    })
        return items

    def _rebuild_followups_index(self, memory_input: str) -> None:
        """STEP 7: Rebuild followups.md from `## Open Items` sections + LLM-curated Other Items.

        v2: the Active Items section is a deterministic aggregation of every
        `## Open Items` entry across all entity files whose canonical status is
        open/in_progress/blocked (done/cancelled self-evict — history lives in git).
        There is no longer a commitments folder to scan. The Other Items section
        is LLM-refreshed and must not duplicate an existing Open Items entry.
        """
        if not self.ontology or not self.ontology.followups_enabled:
            self.logger.debug("FOLLOWUPS_SKIP: followups disabled for this ontology.")
            return

        self.logger.info("STEP 7: Rebuilding followups.md (Open Items aggregation)...")

        # --- Section 1: deterministic Active Items (Open Items sections) ---
        items = self._collect_open_items()

        # --- Section 2: LLM-curated Other Items ---
        followups_path = self.user_path / "followups.md"
        previous_other = self._read_existing_other_items(followups_path)
        item_titles = ", ".join(i["title"] for i in items) or "(none yet)"

        other_items_md = "_(none)_"
        try:
            prompt_template = self._load_prompt("build_followups")
            prompt = prompt_template.format(
                commitment_slugs=item_titles,
                previous_other_items=previous_other,
                memory_input=memory_input,
            )
            llm_out = self._call_llm("", prompt, is_json=False)
            if isinstance(llm_out, str):
                cleaned = llm_out.strip()
                if cleaned:
                    other_items_md = cleaned
        except FileNotFoundError:
            self.logger.debug("FOLLOWUPS: build_followups prompt absent; skipping Other Items.")
            other_items_md = "_(none — `build_followups` prompt not configured)_"
        except Exception as e:
            self.logger.warning(f"FOLLOWUPS_OTHER_ITEMS_LLM_FAILED: {e}. Preserving previous.")
            other_items_md = previous_other

        # --- Render ---
        ts = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        lines: List[str] = [
            "# Follow-ups",
            "_Live to-do list. When a follow-up completes, it disappears from this file — "
            "history lives in `git log`._",
            f"_Updated: {ts} | {len(items)} active items_",
            "",
            "## Active Items",
            "",
        ]
        if items:
            for it in items:
                rel = it["entity_path"].relative_to(self.user_path).as_posix().removesuffix(".md")
                lines.append(f"### {it['title']}")
                lines.append(f"- **Status:** {it['status']}")
                if it["owner"]:
                    lines.append(f"- **Owner:** {it['owner']}")
                if it["due"]:
                    lines.append(f"- **Due:** {it['due']}")
                lines.append(f"- **Context:** [[{rel}|{it['entity_title']}]]")
                lines.append("")
        else:
            lines.append("_(none)_")
            lines.append("")

        lines.append("## Other Items")
        lines.append("")
        lines.append(other_items_md)
        lines.append("")

        followups_path.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info(
            f"FOLLOWUPS_REBUILT: {followups_path} ({len(items)} active items)"
        )

    def process_session(self, memory_input: str, session_id: str, session_date: str = None):
        """Runs the full pipeline to stage changes for a session."""
        self.logger.info(f"--- Processing session {session_id} for user {self.user_path.name} ---")

        # Default to today's date if not provided
        if session_date is None:
            session_date = datetime.now().strftime('%Y-%m-%d')

        # Archive raw session input for future replay/rebuild
        sessions_dir = self.user_path / 'sessions'
        sessions_dir.mkdir(exist_ok=True)
        session_file = sessions_dir / f"{session_date}_{session_id}.txt"
        with open(session_file, 'w', encoding='utf-8') as f:
            f.write(memory_input)
        self.logger.info(f"SESSION_ARCHIVED: {session_file.name}")

        memory_input = memory_input.replace("{", "{{").replace("}", "}}")

        # Step 1: Identify all relevant entities
        entity_analysis = self._identify_relevant_entities(memory_input)
        entities_to_create = entity_analysis.get('entities_to_create', [])
        entities_to_update = entity_analysis.get('entities_to_update', [])

        # Step 2: Create new entities
        self._create_new_entities(memory_input, entities_to_create)

        # Step 3: Update existing entities (only the ones identified)
        self._update_existing_entities(memory_input, entities_to_update)

        # Step 4: Create timeline entry
        self._create_timeline_entry(session_id, session_date, memory_input)

        # Step 5: Build entity indexes for modified files in parallel
        modified_files = self._get_modified_files()
        self._build_entity_indexes(modified_files)

        # Step 6: Rebuild master index
        self._rebuild_master_index()

        # Step 7: Rebuild followups.md (ontology-gated; corporate-only by default).
        # Non-fatal: if this step fails, the previous followups.md is preserved
        # and the session still commits cleanly.
        if self.ontology.followups_enabled:
            try:
                self._rebuild_followups_index(memory_input)
            except Exception as e:
                self.logger.warning(
                    f"FOLLOWUPS_REBUILD_FAILED: {e}. Previous followups.md preserved."
                )

        self.logger.info(f"--- Session {session_id} processing complete. Changes are staged. ---")

    def commit_session(self, session_id: str):
        """STEP 4: Commits all staged changes with a given summary."""
        if not self.repo.is_dirty(untracked_files=True):
            self.logger.warning("No changes to commit.")
            return
        self.repo.git.add(A=True)

        try:
            cached_diff = self.repo.git.diff('--cached', '--name-only').strip()
            modified = [f for f in cached_diff.split('\n') if f.strip()]
            entity_names = [
                Path(f).stem for f in modified
                if f.strip() and '/sessions/' not in f and f != 'index.md'
            ]
            entities_str = ', '.join(entity_names[:8])
            commit_msg = f"Session {session_id} | {entities_str}" if entities_str else f"Session {session_id}"
        except Exception:
            commit_msg = f"Session {session_id}"

        self.repo.index.commit(commit_msg)
        self.logger.info(f"COMMIT_SUCCESS: {commit_msg}")