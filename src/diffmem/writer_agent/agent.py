
### 3. Refactored `writer_agent/agent.py`  
# CAPABILITY: Process session transcripts to create and update memory files, staging all changes.  
# INPUTS: Memory input (str), user_id (str), repo_path (str)  
# OUTPUTS: Staged file changes in the Git working directory.  
# CONSTRAINTS: Uses OpenRouter via OpenAI lib. Prompts are abstracted to files.  

import git  
import json  
import logging  
import math
from datetime import datetime, timedelta
from pathlib import Path  
from typing import List, Dict, Any, Optional
from openai import OpenAI  
from concurrent.futures import ThreadPoolExecutor, as_completed

class WriterAgent:  
    """Orchestrates the process of updating memory files based on a session."""  

    def __init__(self, repo_path: str, user_id: str, openrouter_api_key: str, model: str = "anthropic/claude-3-haiku", max_concurrent_llm_calls: int = 8, validate_paths: bool = True):  
        self.repo_path = Path(repo_path)  
        self.user_id = user_id  
        self.user_path = self.repo_path 
        self.user_file = self.user_path / f"{user_id}.md"  # Core user file at root of user folder  
        self.memories_path = self.user_path / "memories"  
        self.prompts_path = Path(__file__).parent / "prompts"  
        self.max_concurrent_llm_calls = max_concurrent_llm_calls  # Configurable concurrency limit
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
        """Loads a prompt template from the prompts directory."""  
        prompt_file = self.prompts_path / f"{prompt_name}.txt"  
        with open(prompt_file, 'r', encoding='utf-8') as f:  
            return f.read()
    
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
            
            # Create memories subdirectories if they don't exist
            folder = entity['type']
            target_dir = self.memories_path / folder
            target_dir.mkdir(parents=True, exist_ok=True)
            
            file_name = entity['name'].lower().replace(' ', '_').replace('.', '') + '.md'
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

    def _load_master_index_lookup(self) -> Dict[str, str]:
        """
        Loads the master index and creates a name->path lookup dict.
        Handles aliases and name variations.
        
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
                        # Map primary name
                        lookup[entity_data['name'].lower()] = entity_data['file']
                        
                        # Map all aliases
                        for alias in entity_data.get('aliases', []):
                            lookup[alias.lower()] = entity_data['file']
                        
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
        entity_name_lower = entity_name.lower()
        
        if entity_name_lower in index_lookup:
            index_path = self.user_path / index_lookup[entity_name_lower]
            if index_path.exists():
                self.logger.debug(f"ENTITY_RESOLVED_INDEX: {entity_name} → {index_path}")
                return index_path
            else:
                self.logger.warning(f"ENTITY_INDEX_STALE: {entity_name} index points to {index_path} but file not found")
        
        # Strategy 2: Try computed filename in expected folder
        expected_filename = entity_name.lower().replace(' ', '_').replace('.', '') + '.md'
        type_folders = {
            'people': 'people', 'human': 'people',
            'contexts': 'contexts', 'context': 'contexts',
            'events': 'events', 'event': 'events',
            'project': 'contexts', 'company': 'contexts',
            'location': 'contexts', 'concept': 'contexts'
        }
        folder = type_folders.get(entity_type.lower(), 'contexts')
        search_dir = self.memories_path / folder
        
        entity_file = search_dir / expected_filename
        if entity_file.exists():
            self.logger.debug(f"ENTITY_RESOLVED_COMPUTED: {entity_name} → {entity_file}")
            return entity_file
        
        # Strategy 3: Fuzzy search across all memories
        for md_file in self.memories_path.rglob('*.md'):
            if '/sessions/' in str(md_file).replace('\\', '/'):
                continue
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
                    # Only include .md files in memories directory
                    if (full_path.suffix == '.md' and 
                        full_path.is_relative_to(self.memories_path) and
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

    def _build_single_entity_index(self, file_path: Path) -> Dict:
        """Helper method to build semantic index for a single file (for parallel execution)."""
        try:
            # Read current file content
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Strip existing semantic index if present
            content_without_index = self._strip_existing_semantic_index(original_content)
            
            # Get git stats
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
            
            # Override any LLM-generated 'file' path with the computed canonical path
            semantic_index_data['file'] = canonical_path
            
            # Convert to properly formatted JSON string
            semantic_index_json = json.dumps(semantic_index_data, separators=(',', ':'))
            
            # Prepare updated content
            updated_content = content_without_index.rstrip() + '\n\n## SEMANTIC INDEX\n' + semantic_index_json + '\n'
            
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
        """STEP 5: Builds semantic indexes for modified entity files in parallel."""
        if not file_paths:
            self.logger.info("No files to index.")
            return
        
        self.logger.info(f"Building semantic indexes for {len(file_paths)} modified files in parallel...")
        
        # Process files in parallel
        max_workers = min(len(file_paths), self.max_concurrent_llm_calls)
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all indexing tasks
            future_to_file = {
                executor.submit(self._build_single_entity_index, file_path): file_path 
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
        """STEP 5: Rebuilds the master index.md file with all memory entities."""
        self.logger.info("STEP 5: Rebuilding master index.md...")
        
        index_entries = []
        
        # Scan all markdown files in memories directory
        for md_file in self.memories_path.rglob('*.md'):
            if md_file.name == 'index.md' or '/sessions/' in str(md_file).replace('\\', '/'):  # Skip index and sessions
                continue
                
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for SEMANTIC INDEX section
                if '## SEMANTIC INDEX' in content:
                    # Extract the JSON from the semantic index
                    lines = content.split('\n')
                    in_semantic_index = False
                    json_lines = []
                    
                    for line in lines:
                        if line.strip().startswith('## SEMANTIC INDEX'):
                            in_semantic_index = True
                            continue
                        elif in_semantic_index and line.strip().startswith('##'):
                            break
                        elif in_semantic_index and line.strip():
                            json_lines.append(line.strip())
                    
                    if json_lines:
                        # Parse the JSON and add git stats
                        try:
                            semantic_data = json.loads(''.join(json_lines))
                            if len(semantic_data) < 2:
                                self.logger.warning(f"Not enough semantic data found in {md_file}")
                                self._build_entity_indexes([md_file])
                            
                            # Override file path with computed canonical path
                            # This ensures consistency even if the embedded index has wrong paths
                            canonical_path = self._get_relative_entity_path(md_file)
                            llm_path = semantic_data.get('file', '')
                            
                            if llm_path != canonical_path:
                                self.logger.debug(f"PATH_MISMATCH: {md_file.name} index has '{llm_path}', correcting to '{canonical_path}'")
                            
                            semantic_data['file'] = canonical_path
                            
                            git_stats = self._get_file_git_stats(md_file)
                            
                            # Add git metadata
                            semantic_data['last_update'] = git_stats['last_update']
                            semantic_data['number_of_edits'] = git_stats['number_of_edits']
                            semantic_data['memory_strength'] = self._calculate_memory_strength(
                                git_stats['number_of_edits'], 
                                git_stats['last_update']
                            )
                            
                            index_entries.append(semantic_data)
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Could not parse semantic index in {md_file}: {e}")
                else:
                    self.logger.warning(f"No semantic index found in {md_file}")
                    self._build_entity_indexes([md_file])
            except Exception as e:
                self.logger.warning(f"Could not process {md_file} for master index: {e}")
        
        # Sort by memory strength (descending)
        index_entries.sort(key=lambda x: x.get('memory_strength', 0), reverse=True)
        
        # Generate master index content
        master_index_content = f"""# Memory Index for {self.user_id}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total entities: {len(index_entries)}

## Entity Index (by memory strength)

"""
        
        for entry in index_entries:
            name = entry.get('name', 'Unknown')
            entity_type = entry.get('type', 'unknown')
            strength = entry.get('strength', 'Low')
            memory_strength = entry.get('memory_strength', 0)
            file_path = entry.get('file', 'unknown')
            hard_cues = ', '.join(entry.get('hard_cues', [])[:3])  # First 3 cues
            
            master_index_content += f"""### {name} ({entity_type})
- **File**: `{file_path}`
- **Strength**: {strength} (Score: {memory_strength})
```{entry}```

"""
        
        # Write master index
        master_index_path = self.user_path / 'index.md'
        with open(master_index_path, 'w', encoding='utf-8') as f:
            f.write(master_index_content)
        
        self.logger.info(f"MASTER_INDEX_REBUILT: Created {master_index_path} with {len(index_entries)} entities")

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
        
        self.logger.info(f"--- Session {session_id} processing complete. Changes are staged. ---")

    def commit_session(self, session_id: str):  
        """STEP 4: Commits all staged changes with a given summary."""  
        if not self.repo.is_dirty(untracked_files=True):  
            self.logger.warning("No changes to commit.")  
            return  
        self.repo.git.add(A=True)  
        self.repo.index.commit(f"Session {session_id}")
        self.logger.info(f"COMMIT_SUCCESS: Committed session with id: '{session_id}'")