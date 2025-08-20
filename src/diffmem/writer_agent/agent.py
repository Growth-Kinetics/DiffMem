
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
from typing import List, Dict, Any
from openai import OpenAI  

class WriterAgent:  
    """Orchestrates the process of updating memory files based on a session."""  

    def __init__(self, repo_path: str, user_id: str, openrouter_api_key: str, model: str = "anthropic/claude-3-haiku"):  
        self.repo_path = Path(repo_path)  
        self.user_id = user_id  
        self.user_path = self.repo_path / "users" / user_id  
        self.user_file = self.user_path / f"{user_id}.md"  # Core user file at root of user folder  
        self.memories_path = self.user_path / "memories"  
        self.prompts_path = Path(__file__).parent / "prompts"  
        
        if not self.user_path.exists():  
            raise FileNotFoundError(f"User path not found: {self.user_path}")  
        if not self.user_file.exists():  
            raise FileNotFoundError(f"User file not found: {self.user_file}")
        
        self.repo = git.Repo(self.repo_path)  
        self.model = model  
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

    def _create_new_entities(self, memory_input: str, entities_to_create: List[Dict]):
        """STEP 2: Creates files for new entities."""
        if not entities_to_create:
            self.logger.info("No new entities to create.")
            return
            
        self.logger.info(f"Creating {len(entities_to_create)} new entity files...")
      
        # Use the main user file as the example
        example_file_path = self.user_file
        with open(example_file_path, 'r', encoding='utf-8') as f:
            example_content = f.read()

        system_prompt = self._load_prompt("0_system")
        
        for entity in entities_to_create:
            creation_prompt_template = self._load_prompt("2_create_entity_file")
            creation_prompt = creation_prompt_template.format(
                example_file_name=example_file_path.name,
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
            with open(new_file_path, 'w', encoding='utf-8') as f:
                f.write(new_file_content)
            self.logger.info(f"ENTITY_CREATED: Staged new file at {new_file_path}")

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

    def _update_existing_entities(self, memory_input: str, entities_to_update: List[Dict]):
        """STEP 2: Updates only the identified existing entity files."""
        if not entities_to_update:
            self.logger.info("No entities to update.")
            return
            
        self.logger.info(f"Updating {len(entities_to_update)} existing entities...")

        prompt_template = self._load_prompt("3_update_entity_file")
        system_prompt = self._load_prompt("0_system")
        # Build list of files to update based on provided file paths
        files_to_update = [self.user_file]
        
        for entity in entities_to_update:
            if 'file_path' in entity and entity['file_path']:
                # Use the exact path provided by the LLM from core context
                file_path = self.user_path / entity['file_path']
                if file_path.exists():
                    files_to_update.append(file_path)
                    self.logger.debug(f"Found entity file: {entity['file_path']}")
                else:
                    self.logger.warning(f"Entity file not found: {entity['file_path']}")
            else:
                self.logger.warning(f"No file_path provided for entity: {entity.get('name', 'unknown')}")

        for file_path in set(files_to_update):  # Remove duplicates
            if not file_path.is_file() or 'repo_guide' in str(file_path) or 'index' in str(file_path):  
                continue  
                
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
                continue  
                
            self.logger.info(f"Applying {len(updates)} update(s) to {file_path.name}")  
            
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
                        self.logger.debug(f"Replaced text in {file_path.name}")
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
                        self.logger.debug(f"Inserted text after marker in {file_path.name}")
                    else:
                        self.logger.warning(f"Could not find insertion point in {file_path.name}: {search_text[:50]}...")
                
                elif operation == 'append':
                    # Append to the end of the file
                    separator = '\n' if not modified_content.endswith('\n') else ''
                    modified_content = modified_content + separator + replacement_text
                    successful_updates += 1
                    self.logger.debug(f"Appended text to {file_path.name}")
            
            if successful_updates > 0:
                with open(file_path, 'w', encoding='utf-8') as f:  
                    f.write(modified_content)  
                self.logger.info(f"FILE_UPDATED: Applied {successful_updates}/{len(updates)} updates to {file_path.name}")

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
            
            # Get last commit date for this file
            last_commit = self.repo.git.log('-1', '--format=%ci', str(rel_path))
            last_update = last_commit.strip() if last_commit else "Unknown"
            
            # Get number of commits that touched this file
            commit_count = self.repo.git.rev_list('--count', 'HEAD', '--', str(rel_path))
            number_of_edits = int(commit_count.strip()) if commit_count.strip() else 0
            
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

    def _build_entity_indexes(self, file_path: Path):
        """STEP 4: Builds semantic indexes for all modified entity files."""
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
            
            # Build semantic index using LLM
            prompt_template = self._load_prompt("build_index")
            prompt = prompt_template.format(
                file_content=content_without_index,
                file_path=str(file_path.relative_to(self.user_path)),
                last_update=git_stats['last_update'],
                number_of_edits=git_stats['number_of_edits'],
                memory_strength=memory_strength
            )
            
            # Get semantic index JSON from LLM
            semantic_index_data = self._call_llm("", prompt, is_json=True)
            
            # Convert to properly formatted JSON string
            semantic_index_json = json.dumps(semantic_index_data, separators=(',', ':'))
            
            # Append semantic index to file
            updated_content = content_without_index.rstrip() + '\n\n## SEMANTIC INDEX\n' + semantic_index_json + '\n'
            
            # Write updated content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            self.logger.info(f"INDEX_UPDATED: Added semantic index to {file_path.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to build index for {file_path}: {e}")

    def _rebuild_master_index(self):
        """STEP 5: Rebuilds the master index.md file with all memory entities."""
        self.logger.info("STEP 5: Rebuilding master index.md...")
        
        index_entries = []
        
        # Scan all markdown files in memories directory
        for md_file in self.memories_path.rglob('*.md'):
            if md_file.name == 'index.md':  # Skip existing index files
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
                                self._build_entity_indexes(md_file)
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
                    self._build_entity_indexes(md_file)
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
        
        # Step 5: Build entity indexes for modified files
        modified_files = self._get_modified_files()
        if not modified_files:
            self.logger.info("No modified files to index.")
            return
        
        self.logger.info(f"STEP 5: Building semantic indexes for {len(modified_files)} modified files...")
        for file_path in modified_files:
            self._build_entity_indexes(file_path)
        
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