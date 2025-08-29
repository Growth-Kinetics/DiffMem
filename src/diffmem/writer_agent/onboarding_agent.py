# CAPABILITY: Onboard new users by creating initial directory structure and entity files
# INPUTS: Raw user information dump, user_id, repo_path
# OUTPUTS: Complete user directory structure with initial files
# CONSTRAINTS: Uses OpenRouter via OpenAI lib. Creates minimal viable memory structure.

import git
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from openai import OpenAI

class OnboardingAgent:
    """Handles initial user setup and directory structure creation."""
    
    def __init__(self, repo_path: str, user_id: str, openrouter_api_key: str, model: str = "google/gemini-2.5-pro"):
        self.repo_path = Path(repo_path)
        self.user_id = user_id
        self.user_path = self.repo_path / "users" / user_id
        self.user_file = self.user_path / f"{user_id}.md"
        self.memories_path = self.user_path / "memories"
        self.timeline_path = self.user_path / "timeline"
        self.prompts_path = Path(__file__).parent / "prompts"
        self.openrouter_api_key = openrouter_api_key
        self.model = model
        self.repo = git.Repo(self.repo_path)
        
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
    
    def _call_llm(self, system_prompt: str, prompt: str, is_json: bool = True, model=None) -> Any:
        """Calls the LLM via OpenRouter, enforcing JSON mode if requested."""
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        if model is None:
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
    
    def _create_directory_structure(self):
        """Creates the basic directory structure for a new user."""
        self.logger.info(f"Creating directory structure for user: {self.user_id}")
        
        # Create main directories
        self.user_path.mkdir(parents=True, exist_ok=True)
        self.memories_path.mkdir(parents=True, exist_ok=True)
        self.timeline_path.mkdir(parents=True, exist_ok=True)
        
        # Create memory subdirectories
        (self.memories_path / "people").mkdir(exist_ok=True)
        (self.memories_path / "contexts").mkdir(exist_ok=True)
        
        self.logger.info(f"Directory structure created at: {self.user_path}")
    
    def _create_initial_user_file(self, user_info: str) -> str:
        """Creates the main user entity file from the information dump."""
        self.logger.info("Creating initial user entity file...")
        
        # Load system prompt and onboarding prompt
        system_prompt = self._load_prompt("0_system")
        onboarding_prompt_template = self._load_prompt("onboard_user_entity")
        
        onboarding_prompt = onboarding_prompt_template.format(
            user_id=self.user_id,
            user_info=user_info
        )
        
        user_content = self._call_llm(system_prompt, onboarding_prompt, is_json=False)
        
        # Write the user file
        with open(self.user_file, 'w', encoding='utf-8') as f:
            f.write(user_content)
        
        self.logger.info(f"USER_FILE_CREATED: {self.user_file}")
        return user_content
    
    def _identify_initial_entities(self, user_info: str) -> Dict:
        """Identifies entities to create from the initial information dump."""
        self.logger.info("Identifying initial entities from user information...")
        
        system_prompt = self._load_prompt("0_system")
        entity_prompt_template = self._load_prompt("onboard_identify_entities")
        
        entity_prompt = entity_prompt_template.format(
            user_id=self.user_id,
            user_info=user_info
        )
        
        response = self._call_llm(system_prompt, entity_prompt, is_json=True)
        
        entities_to_create = response.get('entities_to_create', [])
        self.logger.info(f"Identified {len(entities_to_create)} initial entities to create")
        
        return response
    
    def _create_initial_entities(self, user_info: str, entities_to_create: list, user_content: str):
        """Creates initial entity files from the information dump."""
        if not entities_to_create:
            self.logger.info("No initial entities to create.")
            return
        
        self.logger.info(f"Creating {len(entities_to_create)} initial entity files...")
        
        system_prompt = self._load_prompt("0_system")
        
        for entity in entities_to_create:
            creation_prompt_template = self._load_prompt("2_create_entity_file")
            creation_prompt = creation_prompt_template.format(
                example_file_name=f"{self.user_id}.md",
                example_content=user_content,
                entity_name=entity['name'],
                entity_summary=entity['summary'],
                memory_input=user_info
            )
            
            new_file_content = self._call_llm(system_prompt, creation_prompt, is_json=False)
            
            # Create target directory and file
            folder = entity['type']
            target_dir = self.memories_path / folder
            target_dir.mkdir(parents=True, exist_ok=True)
            
            file_name = entity['name'].lower().replace(' ', '_').replace('.', '').replace('/', '_') + '.md'
            new_file_path = target_dir / file_name
            
            with open(new_file_path, 'w', encoding='utf-8') as f:
                f.write(new_file_content)
            
            self.logger.info(f"ENTITY_CREATED: {new_file_path}")
    
    def _create_initial_timeline_entry(self, user_info: str, session_id: str):
        """Creates the first timeline entry documenting the onboarding."""
        self.logger.info("Creating initial timeline entry...")
        
        current_date = datetime.now()
        timeline_filename = self.timeline_path / f"{current_date.strftime('%Y-%m')}.md"
        
        system_prompt = self._load_prompt("0_system")
        timeline_prompt_template = self._load_prompt("onboard_timeline_entry")
        
        timeline_prompt = timeline_prompt_template.format(
            user_id=self.user_id,
            session_id=session_id,
            session_date=current_date.strftime('%Y-%m-%d'),
            user_info=user_info
        )
        
        timeline_entry = self._call_llm(system_prompt, timeline_prompt, is_json=False)
        
        with open(timeline_filename, 'w', encoding='utf-8') as f:
            f.write(f"# Timeline: {current_date.strftime('%Y-%m')} (Onboarding) [Initial Setup]\n\n")
            f.write(timeline_entry)
        
        self.logger.info(f"TIMELINE_CREATED: {timeline_filename}")
    
    def _build_initial_semantic_indexes(self):
        """Builds semantic indexes for all created files."""
        self.logger.info("Building initial semantic indexes...")
        
        # Get all markdown files in memories
        md_files = []
        md_files.append(self.user_file)  # Include main user file
        md_files.extend(list(self.memories_path.rglob('*.md')))
        
        system_prompt = self._load_prompt("0_system")
        index_prompt_template = self._load_prompt("build_index")
        
        for file_path in md_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Skip if already has semantic index
                if '## SEMANTIC INDEX' in content:
                    continue
                
                prompt = index_prompt_template.format(
                    file_content=content,
                    file_path=str(file_path.relative_to(self.user_path)),
                    last_update=datetime.now().strftime('%Y-%m-%d %H:%M:%S +0000'),
                    number_of_edits=1,
                    memory_strength=0.1  # Low strength for new files
                )
                
                semantic_index_data = self._call_llm("", prompt, is_json=True)
                semantic_index_json = json.dumps(semantic_index_data, separators=(',', ':'))
                
                # Append semantic index to file
                updated_content = content.rstrip() + '\n\n## SEMANTIC INDEX\n' + semantic_index_json + '\n'
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
                
                self.logger.info(f"INDEX_CREATED: {file_path.name}")
                
            except Exception as e:
                self.logger.error(f"Failed to build index for {file_path}: {e}")
    
    def _create_master_index(self):
        """Creates the initial master index file."""
        self.logger.info("Creating initial master index...")
        
        index_entries = []
        
        # Scan all markdown files in memories directory
        for md_file in self.memories_path.rglob('*.md'):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract semantic index
                if '## SEMANTIC INDEX' in content:
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
                        try:
                            semantic_data = json.loads(''.join(json_lines))
                            semantic_data['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S +0000')
                            semantic_data['number_of_edits'] = 1
                            semantic_data['memory_strength'] = 0.1
                            index_entries.append(semantic_data)
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Could not parse semantic index in {md_file}: {e}")
                            
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
            
            master_index_content += f"""### {name} ({entity_type})
- **File**: `{file_path}`
- **Strength**: {strength} (Score: {memory_strength})
```{entry}```

"""
        
        # Write master index
        master_index_path = self.user_path / 'index.md'
        with open(master_index_path, 'w', encoding='utf-8') as f:
            f.write(master_index_content)
        
        self.logger.info(f"MASTER_INDEX_CREATED: {master_index_path} with {len(index_entries)} entities")
    
    def onboard_user(self, user_info: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete onboarding process for a new user.
        
        Args:
            user_info: Raw information dump about the user
            session_id: Optional session ID for tracking
            
        Returns:
            Dict with onboarding results and metadata
        """
        if session_id is None:
            session_id = f"onboard-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        self.logger.info(f"--- Starting onboarding for user {self.user_id} (session: {session_id}) ---")
        
        try:
            # Step 1: Create directory structure
            self._create_directory_structure()
            
            # Step 2: Create main user file
            user_content = self._create_initial_user_file(user_info)
            
            # Step 3: Identify and create initial entities
            entity_analysis = self._identify_initial_entities(user_info)
            entities_to_create = entity_analysis.get('entities_to_create', [])
            
            self._create_initial_entities(user_info, entities_to_create, user_content)
            
            # Step 4: Create initial timeline entry
            self._create_initial_timeline_entry(user_info, session_id)
            
            # Step 5: Build semantic indexes
            self._build_initial_semantic_indexes()
            
            # Step 6: Create master index
            self._create_master_index()
            
            # Step 7: Commit all changes
            if self.repo.is_dirty(untracked_files=True):
                self.repo.git.add(A=True)
                self.repo.index.commit(f"Initial onboarding for user {self.user_id} (session: {session_id})")
                self.logger.info(f"ONBOARDING_COMMITTED: User {self.user_id} successfully onboarded")
            
            self.logger.info(f"--- Onboarding complete for user {self.user_id} ---")
            
            return {
                'success': True,
                'user_id': self.user_id,
                'session_id': session_id,
                'entities_created': len(entities_to_create),
                'files_created': {
                    'user_file': str(self.user_file),
                    'entities': [entity['name'] for entity in entities_to_create],
                    'timeline': f"{datetime.now().strftime('%Y-%m')}.md",
                    'master_index': 'index.md'
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Onboarding failed for user {self.user_id}: {e}")
            return {
                'success': False,
                'user_id': self.user_id,
                'session_id': session_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            } 