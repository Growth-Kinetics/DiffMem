# CAPABILITY: Onboard new users by creating initial directory structure and entity files
# INPUTS: Raw user information dump, user_id, repo_path
# OUTPUTS: Complete user directory structure with initial files
# CONSTRAINTS: Uses OpenRouter via OpenAI lib. Creates minimal viable memory structure.

import git
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from .agent import WriterAgent

class OnboardingAgent(WriterAgent):
    """Handles initial user setup and directory structure creation."""
    
    def __init__(self, repo_path: str, user_id: str, openrouter_api_key: str, model: str = "google/gemini-2.5-pro"):
        super().__init__(
            repo_path=repo_path, 
            user_id=user_id, 
            openrouter_api_key=openrouter_api_key, 
            model=model,
            validate_paths=False
        )
        self.timeline_path = self.user_path / "timeline"
    
    def _create_directory_structure(self):
        """Creates the basic directory structure for a new user."""
        self.logger.info(f"Creating directory structure for user: {self.user_id}")
        
        self.memories_path.mkdir(parents=True, exist_ok=True)
        self.timeline_path.mkdir(parents=True, exist_ok=True)
        
        # Create memory subdirectories
        (self.memories_path / "people").mkdir(exist_ok=True)
        (self.memories_path / "contexts").mkdir(exist_ok=True)
        
        self.logger.info(f"Directory structure created at: {self.user_path}")
        
        # Initialize git repo object if not already done
        # Assuming the path is already a git worktree or repo
        self.repo = git.Repo(self.user_path)

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
        """Builds semantic indexes for all created files in parallel."""
        self.logger.info("Building initial semantic indexes...")
        
        # Get all markdown files in memories
        md_files = []
        if self.user_file.exists():
            md_files.append(self.user_file)
        md_files.extend(list(self.memories_path.rglob('*.md')))
        
        # Delegate to parent's parallelized method (DRY)
        self._build_entity_indexes(md_files)
    
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
            
            # Reuse WriterAgent's parallel entity creation
            # Note: _create_new_entities expects memory_input to be the context (user_info here)
            self._create_new_entities(user_info, entities_to_create)
            
            # Step 4: Create initial timeline entry
            self._create_initial_timeline_entry(user_info, session_id)
            
            # Step 5: Build semantic indexes
            self._build_initial_semantic_indexes()
            
            # Step 6: Create master index (Reuse WriterAgent's method)
            self._rebuild_master_index()
            
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
