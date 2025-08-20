# CAPABILITY: Context assembly for chat agents via memory retrieval and Git temporal analysis
# INPUTS: Conversation transcript (List[Dict]), user_id (str), repo_path (str), depth ("basic"|"wide"|"deep")
# OUTPUTS: Structured context dict with memories, session metadata, and Git evolution data
# CONSTRAINTS: Uses BM25 indexer, searcher agent, and Git for temporal insights; optimized for token efficiency

import git
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import Counter  
import re 
import numpy as np

from sentence_transformers import SentenceTransformer  
from sklearn.metrics.pairwise import cosine_similarity  
from diffmem.bm25_indexer.api import build_index, search
from diffmem.searcher_agent.agent import orchestrate_query

# Structured logging for agent perception
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ContextRequest:
    """Request structure for context assembly"""
    conversation: List[Dict[str, str]]  # [{'role': 'user', 'content': '...'}, ...]
    user_id: str
    depth: str = "basic"  # "basic", "wide", "deep"
    temporal_focus: Optional[str] = None  # "recent", "evolution", "historical"

@dataclass
class ContextResponse:
    """Response structure with assembled context"""
    always_load_blocks: List[Dict[str, Any]]
    recent_timeline: List[Dict[str, Any]]
    session_metadata: Dict[str, Any]
    complete_entities: List[Dict[str, Any]] = None  # For deep mode
    temporal_blame: List[Dict[str, Any]] = None      # For temporal mode

class SemanticEntityScorer:  
    """Semantic relevance scoring with intelligent thresholding."""  
    
    def __init__(self, entities: List[Dict[str, Any]], model_name: str = 'all-MiniLM-L6-v2'):  
        """Initialize with a lightweight sentence transformer model."""  
        self.entities = entities  
        self.model = SentenceTransformer(model_name)  
        
        # Pre-compute entity embeddings  
        print("Pre-computing entity embeddings...")  
        for entity in self.entities:  
            entity_text = self._entity_to_text(entity)  
            entity['_embedding'] = self.model.encode(entity_text)  
            entity['_text_repr'] = entity_text  
    
    def _entity_to_text(self, entity: Dict) -> str:  
        """Convert entity metadata to semantic text representation."""  
        parts = [  
            entity['name'],  
            entity.get('role', ''),  
            ' '.join(entity.get('aliases', [])),  
            ' '.join(entity.get('hard_cues', [])),  
            ' '.join(entity.get('soft_cues', [])),  
            ' '.join(entity.get('emotional_cues', []))  
        ]  
        return ' '.join(filter(None, parts))  
    
    def _find_relevance_threshold(self, scores: List[float]) -> float:  
        """Find natural cutoff point in score distribution."""  
        if len(scores) < 2:  
            return 0.0  
        
        # Sort scores descending  
        sorted_scores = sorted(scores, reverse=True)  
        
        # Calculate differences between consecutive scores  
        diffs = [sorted_scores[i] - sorted_scores[i+1] for i in range(len(sorted_scores)-1)]  
        
        # Find the largest gap (elbow point)  
        if diffs:  
            max_gap_idx = np.argmax(diffs)  
            # Set threshold just below the score before the gap  
            threshold = sorted_scores[max_gap_idx + 1] + (diffs[max_gap_idx] * 0.1)  
            return threshold  
        
        return 0.0  
    
    def score_conversation(self,   
                         conversation: List[Dict],   
                         min_similarity: float = 0.3,  # Absolute minimum  
                         relevance_margin: float = 0.15,  # How much below top score is still relevant  
                         max_results: Optional[int] = None) -> List[Dict[str, Any]]:  
        """  
        Score entities using semantic similarity with intelligent filtering.  
        
        Args:  
            conversation: List of message dicts  
            min_similarity: Absolute minimum similarity threshold (default 0.3)  
            relevance_margin: Scores within this margin of top score are relevant (default 0.15)  
            max_results: Optional hard limit on results (default None = no limit)  
        """  
        
        # 1. Embed the conversation  
        conv_text = ' '.join([msg['content'] for msg in conversation])  
        conv_embedding = self.model.encode(conv_text)  
        
        # 2. Compute similarities  
        scored_entities = []  
        raw_scores = []  
        
        for entity in self.entities:  
            # Base semantic similarity  
            similarity = cosine_similarity(  
                conv_embedding.reshape(1, -1),  
                entity['_embedding'].reshape(1, -1)  
            )[0][0]  
            
            raw_scores.append(similarity)  
            
            # Apply memory strength multiplier  
            adjusted_score = similarity * (1 + entity.get('memory_strength', 0.5))  
            
            # Recency boost  
            if 'last_update' in entity:  
                days_ago = (datetime.now() - datetime.fromisoformat(entity['last_update'].replace(' +0200', ''))).days  
                recency_factor = max(0.7, 1 - (days_ago / 365))  
                adjusted_score *= recency_factor  
            
            scored_entities.append({  
                'entity': entity,  
                'score': adjusted_score,  
                'raw_similarity': similarity,  
                'type': entity['type']  
            })  
        
        # 3. Determine relevance threshold  
        # Use raw similarities for threshold detection (before boosts)  
        auto_threshold = self._find_relevance_threshold(raw_scores)  
        threshold = max(min_similarity, auto_threshold)  
        
        # Also use relative threshold: within X of the top score  
        if scored_entities:  
            top_score = max(s['raw_similarity'] for s in scored_entities)  
            relative_threshold = top_score - relevance_margin  
            threshold = max(threshold, relative_threshold)  
        
        # 4. Filter by relevance  
        relevant_entities = [  
            e for e in scored_entities   
            if e['raw_similarity'] >= threshold  
        ]  
        
        # 5. Sort by adjusted score  
        relevant_entities.sort(key=lambda x: x['score'], reverse=True)  
        
        # 6. Apply spreading activation only to relevant entities  
        if relevant_entities:  
            # Get top 3 relevant entities for spreading  
            top_relevant = relevant_entities[:3]  
            related_boost = {}  
            
            for item in top_relevant:  
                for related in item['entity'].get('related_entities', []):  
                    related_boost[related] = max(  
                        related_boost.get(related, 0),  
                        item['score'] * 0.3  
                    )  
            
            # Apply boosts and check if it brings any new entities above threshold  
            for item in scored_entities:  
                entity_name = item['entity']['name'].lower().replace(' ', '_')  
                if entity_name in related_boost:  
                    boosted_score = item['score'] + related_boost[entity_name]  
                    # If boost brings it above threshold and it wasn't already included  
                    if item not in relevant_entities and item['raw_similarity'] >= (threshold * 0.8):  
                        item['score'] = boosted_score  
                        item['spreading_boost'] = True  
                        relevant_entities.append(item)  
        
        # 7. Final sort and optional limit  
        relevant_entities.sort(key=lambda x: x['score'], reverse=True)  
        
        if max_results:  
            relevant_entities = relevant_entities[:max_results]  
        
        # 8. Add relevance metadata  
        for item in relevant_entities:  
            item['relevance_level'] = self._classify_relevance(item['raw_similarity'], top_score if scored_entities else 0)  
        
        return relevant_entities  
    
    def _classify_relevance(self, similarity: float, top_score: float) -> str:  
        """Classify relevance level based on similarity score."""  
        if similarity >= 0.7:  
            return 'high'  
        elif similarity >= 0.5:  
            return 'medium'  
        elif similarity >= top_score - 0.15:  
            return 'contextual'  
        else:  
            return 'peripheral'  


class ContextManager:
    """Orchestrates memory retrieval and context assembly for chat agents"""
    
    def __init__(self, repo_path: str, user_id: str, openrouter_api_key: str, model: str = "openai/gpt-4o"):

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
        self.openrouter_api_key = openrouter_api_key
        self.model = model
        
        # Initialize BM25 index (rebuild on start)
        logger.info("CONTEXT_MANAGER_INIT: Building BM25 index...")
        self.index = build_index(str(self.repo_path))
        logger.info(f"CONTEXT_MANAGER_READY: Index built with {len(self.index['corpus'])} blocks")

    def assemble_context(self, request: ContextRequest) -> ContextResponse:
        """
        Main orchestration method - analyzes conversation and assembles appropriate context
        
        Flow:
        1. Analyze conversation depth and extract entities
        2. Load always_load blocks and recent timeline (basic/wide)
        3. Load complete entity files (deep mode)
        4. Extract Git blame data (temporal mode)
        5. Assemble and return structured context
        """
        logger.info(f"CONTEXT_ASSEMBLY_START: user={request.user_id} depth={request.depth} conv_pairs={len(request.conversation)}")
        
        user_path = self.repo_path / "users" / request.user_id
        if not user_path.exists():
            raise FileNotFoundError(f"User path not found: {user_path}")
        
        # Step 1: Load master index for entity ranking
        master_index = self._load_master_index(user_path)
        entities = [Path(x["file"]).stem for x in master_index["entities"]][0:5]
        
        # Extract entities for non-basic modes
        if request.depth != "basic":
            entities = self._extract_entities_from_conversation(request.conversation, master_index)
            
        unique_entities = [x for x in list(set(entities)) if x != self.user_id.lower()]
        
        
        # Step 2: Load complete user entity file (always included)
        user_entity = self._load_complete_user_entity()
        
        # Initialize response fields
        always_load_blocks = []
        complete_entities = None
        temporal_blame = None
        
        # Step 3: Load content based on depth mode
        if request.depth in ["basic", "wide"]:
            # For basic/wide modes: load only ALWAYS_LOAD blocks
            logger.info(f"BASIC_MODE: Loading {len(unique_entities)} entities")
            always_load_blocks = self._load_always_load_blocks_from_index(unique_entities, user_path)
            
        elif request.depth == "deep":
            # For deep mode: load complete entity files
            complete_entities = self._load_complete_entity_files(unique_entities, user_path)
            logger.info(f"DEEP_MODE: Loaded {len(complete_entities)} complete entity files")
            
        elif request.depth == "temporal":
            # For temporal mode: load complete files + git blame
            complete_entities = self._load_complete_entity_files(unique_entities, user_path)
            
            # Get blame data for each loaded entity
            temporal_blame = []
            memories_path = user_path / "memories"
            
            for entity_name in unique_entities[:5]:  # Limit blame to top 5 to prevent context explosion
                # Find the entity file
                for subdir in ['people', 'contexts', 'events']:
                    entity_file = memories_path / subdir / f"{entity_name}.md"
                    if entity_file.exists():
                        blame_data = self._get_entity_blame(entity_file)
                        temporal_blame.append(blame_data)
                        break
            
            logger.info(f"TEMPORAL_MODE: Loaded {len(complete_entities)} entities with {len(temporal_blame)} blame records")
        
        # Step 4: Recent timeline (included for all modes)
        recent_timeline = self._load_recent_timeline(user_path)
        
        # Step 5: Assemble response
        response = ContextResponse(
            always_load_blocks=[user_entity] + always_load_blocks if always_load_blocks else [user_entity],
            recent_timeline=recent_timeline,
            session_metadata={
                "user_id": request.user_id,
                "depth": request.depth,
                "timestamp": datetime.now().isoformat(),
                "entities_found": unique_entities,
                "mode_info": {
                    "basic": "Top entities with ALWAYS_LOAD blocks",
                    "wide": "Semantic search with ALWAYS_LOAD blocks",
                    "deep": "Complete entity files",
                    "temporal": "Complete files with Git blame"
                }.get(request.depth, "Unknown mode")
            },
            complete_entities=complete_entities,
            temporal_blame=temporal_blame
        )
        
        logger.info(f"CONTEXT_ASSEMBLY_COMPLETE: depth={request.depth} entities={len(unique_entities)}")
        return response

    def _load_master_index(self, user_path: Path) -> Dict[str, Any]:
        """Load and parse the master index.md file with ranked entities"""
        index_file = user_path / "index.md"
        master_index = {
            'entities': [],
            'by_strength': {},
            'by_name': {},
            'total_entities': 0
        }
        
        if not index_file.exists():
            logger.warning(f"MASTER_INDEX_MISSING: {index_file} - falling back to file discovery")
            return master_index
        
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse entity entries - look for JSON blocks after entity headers
            import re
            import json
            
            # Pattern to match entity sections with JSON
            pattern = r'### (.+?)\n.*?```\{(.+?)\}```'
            matches = re.findall(pattern, content, re.DOTALL)
            
            entities = []
            for entity_title, json_str in matches:
                try:
                    # Clean up the JSON string and parse
                    json_str = json_str.strip()
                    if not json_str.startswith('{'):
                        json_str = '{' + json_str
                    if not json_str.endswith('}'):
                        json_str = json_str + '}'
                    
                    # Use ast.literal_eval for Python dict-style parsing instead of JSON
                    import ast
                    entity_data = ast.literal_eval(json_str)
                    entity_data['title'] = entity_title.strip()
                    
                    # Extract memory strength score if available
                    strength_score = entity_data.get('memory_strength', 0.0)
                    if isinstance(strength_score, str):
                        try:
                            strength_score = float(strength_score)
                        except ValueError:
                            strength_score = 0.5  # Default
                    
                    entity_data['strength_score'] = strength_score
                    entities.append(entity_data)
                    
                except (ValueError, SyntaxError, KeyError) as e:
                    logger.warning(f"INDEX_PARSE_ERROR: entity={entity_title[:30]} error={str(e)}")
                    continue
            
            # Sort by memory strength (highest first)
            entities.sort(key=lambda x: x.get('strength_score', 0), reverse=True)
            
            # Build lookup dictionaries
            by_strength = {}
            by_name = {}
            
            for entity in entities:
                name = entity.get('name', '').lower()
                strength = entity.get('strength', 'Medium').lower()
                
                if strength not in by_strength:
                    by_strength[strength] = []
                by_strength[strength].append(entity)
                
                if name:
                    by_name[name] = entity
                    # Also index aliases
                    for alias in entity.get('aliases', []):
                        by_name[alias.lower()] = entity
            
            master_index = {
                'entities': entities,
                'by_strength': by_strength,
                'by_name': by_name,
                'total_entities': len(entities)
            }
            
            logger.debug(f"MASTER_INDEX_LOADED: total={len(entities)} high={len(by_strength.get('high', []))} medium={len(by_strength.get('medium', []))} low={len(by_strength.get('low', []))}")
            
        except Exception as e:
            logger.error(f"MASTER_INDEX_ERROR: {str(e)}")
        
        return master_index

    def _extract_entities_from_conversation(self, conversation: List[Dict[str, str]], index: Dict[str, Any]) -> List[str]:
        """Extract mentioned entities and keywords from conversation using multiple strategies"""
        extracted_entities = set()
        
        # Get conversation text
        conversation_text = " ".join([
            msg.get('content', '') for msg in conversation 
            if msg.get('content')
        ]).lower()
        
        # Strategy 1: Using Searcher Agent
        result = orchestrate_query(conversation, self.index, model="google/gemini-2.5-flash")
        
        # Strategy 2: Using Semantic Index
        scorer = SemanticEntityScorer(index['entities'])
        relevant = scorer.score_conversation(  
        conversation,  
        min_similarity=0.2,  # Nothing below 60% similarity  
        relevance_margin=0.2,   # Include anything within 10% of top match 
        max_results=5
        ) 
        
        entities = [Path(f["file_path"]).stem for f in result["snippets"]] +  [Path(f["entity"]["file"]).stem for f in relevant]
        unique_entities = list(set(entities))
        logger.debug(f"ENTITIES_EXTRACTED: total={len(unique_entities)} entities={unique_entities}")
        return unique_entities

    def _load_complete_user_entity(self) -> Dict[str, Any]:
        """Load the complete user entity file (alex.md) in its entirety"""
        if not self.user_file.exists():
            logger.warning(f"USER_ENTITY_NOT_FOUND: {self.user_file}")
            return {
                'entity_name': self.user_id,
                'file_path': f"{self.user_id}.md",
                'content': f"# {self.user_id.title()} (User Entity Not Found)",
                'type': 'user_entity_missing'
            }
        
        try:
            with open(self.user_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            user_entity = {
                'entity_name': self.user_id.title(),
                'file_path': f"{self.user_id}.md",
                'content': content,
                'strength_score': 1.0,  # Maximum strength for user entity
                'type': 'complete_user_entity'
            }
            
            logger.debug(f"USER_ENTITY_LOADED: {self.user_file} chars={len(content)}")
            return user_entity
            
        except Exception as e:
            logger.error(f"USER_ENTITY_ERROR: {str(e)}")
            return {
                'entity_name': self.user_id,
                'file_path': f"{self.user_id}.md",
                'content': f"# {self.user_id.title()} (Error Loading User Entity: {str(e)})",
                'type': 'user_entity_error'
            }

    def _load_always_load_blocks_from_index(self, entities_list: List[str], user_path: Path, max_entities: int = 10) -> List[Dict[str, Any]]:
        """Load ALWAYS_LOAD blocks only from top entities in master index"""
        always_load_blocks = []
        entity_set = {name.lower() for name in entities_list}
        entity_files = [] 
        
        # Recursively find all .md files  
        for md_file in self.repo_path.rglob('*.md'):  
            # Check if this file matches any entity  
            if md_file.stem.lower() in entity_set:  
                entity_files.append(md_file)        
        
        for file_path in entity_files:
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find blocks with ALWAYS_LOAD marker
                import re
                pattern = r'/START\s*(.*?)/END'
                matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
                
                for block_content in matches:
                    if '[ALWAYS_LOAD]' in block_content:
                        # Extract header from first line
                        lines = block_content.strip().split('\n')
                        header = lines[0] if lines else "Unknown Block"
                        
                        always_load_blocks.append({
                            'file_path': file_path,
                            'header': header.strip('#').strip(),
                            'content': block_content.strip(),
                            'type': 'always_load'
                        })
                        
            except Exception as e:
                logger.warning(f"ALWAYS_LOAD_PARSE_ERROR: file={file_path} error={str(e)}")
        
        return always_load_blocks

    def _load_recent_timeline(self, user_path: Path, days_back: int = 30) -> List[Dict[str, Any]]:
        """Load recent timeline entries from the last N days"""
        timeline_blocks = []
        timeline_path = user_path / "timeline"
        
        if not timeline_path.exists():
            return timeline_blocks
        
        # Get recent timeline files (YYYY-MM.md format)
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for timeline_file in timeline_path.glob("*.md"):
            try:
                # Parse date from filename (YYYY-MM.md or YYYY-MM-DD_event.md)
                filename = timeline_file.stem
                if filename.count('-') >= 1:
                    date_part = filename.split('_')[0] if '_' in filename else filename
                    
                    # Handle both YYYY-MM and YYYY-MM-DD formats
                    if date_part.count('-') == 1:  # YYYY-MM
                        file_date = datetime.strptime(date_part, '%Y-%m')
                    else:  # YYYY-MM-DD
                        file_date = datetime.strptime(date_part, '%Y-%m-%d')
                    
                    if file_date >= cutoff_date:
                        with open(timeline_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        timeline_blocks.append({
                            'file_path': str(timeline_file.relative_to(self.repo_path)),
                            'date': file_date.isoformat(),
                            'content': content[:1000] + "..." if len(content) > 1000 else content,  # Truncate for context
                            'type': 'recent_timeline'
                        })
                        
            except Exception as e:
                logger.warning(f"TIMELINE_PARSE_ERROR: file={timeline_file} error={str(e)}")
        
        # Sort by date, most recent first
        timeline_blocks.sort(key=lambda x: x['date'], reverse=True)
        logger.debug(f"RECENT_TIMELINE: found={len(timeline_blocks)} files")
        return timeline_blocks[:5]  # Limit to 5 most recent

    def _load_complete_entity_files(self, entities_list: List[str], user_path: Path, max_entities: int = 10) -> List[Dict[str, Any]]:
        """Load complete entity files for deep mode - entire file content, not just ALWAYS_LOAD blocks"""
        complete_entities = []
        entity_set = {name.lower() for name in entities_list[:max_entities]}  # Limit to prevent context explosion
        
        # Search for entity files across all memory subdirectories
        memories_path = user_path / "memories"
        if not memories_path.exists():
            return complete_entities
        
        # Check people, contexts, and timeline directories
        for subdir in ['people', 'contexts']:
            subdir_path = memories_path / subdir
            if subdir_path.exists():
                for md_file in subdir_path.glob('*.md'):
                    if md_file.stem.lower() in entity_set:
                        try:
                            with open(md_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            complete_entities.append({
                                'entity_name': md_file.stem,
                                'file_path': str(md_file.relative_to(self.repo_path)),
                                'content': content,
                                'type': 'complete_entity',
                                'subdir': subdir
                            })
                            logger.debug(f"DEEP_MODE_LOADED: {md_file.stem} from {subdir} ({len(content)} chars)")
                            
                        except Exception as e:
                            logger.warning(f"DEEP_MODE_LOAD_ERROR: file={md_file} error={str(e)}")
        
        return complete_entities

    def _get_entity_blame(self, file_path: Path, max_lines: int = 100) -> Dict[str, Any]:
        """Get git blame information for an entity file - for temporal mode"""
        blame_info = {
            'file_path': str(file_path.relative_to(self.repo_path)),
            'blame_data': [],
            'unique_commits': set(),
            'first_commit': None,
            'last_commit': None,
            'total_lines': 0
        }
        
        try:
            # Use git blame to get line-by-line commit info
            blame_output = self.repo.blame('HEAD', str(file_path.relative_to(self.repo_path)))
            
            for commit, lines in blame_output:
                commit_info = {
                    'hash': commit.hexsha[:8],
                    'author': commit.author.name,
                    'date': datetime.fromtimestamp(commit.committed_date).isoformat(),
                    'message': commit.message.strip()[:100],  # Truncate message
                    'lines': len(lines)
                }
                
                blame_info['blame_data'].append(commit_info)
                blame_info['unique_commits'].add(commit.hexsha)
                blame_info['total_lines'] += len(lines)
                
                # Track first and last commits (compare timestamps directly)
                if blame_info['first_commit'] is None or commit.committed_date < blame_info['first_commit']['timestamp']:
                    blame_info['first_commit'] = {
                        'hash': commit.hexsha[:8],
                        'date': datetime.fromtimestamp(commit.committed_date).isoformat(),
                        'message': commit.message.strip()[:100],
                        'timestamp': commit.committed_date  # Keep timestamp for comparison
                    }
                    
                if blame_info['last_commit'] is None or commit.committed_date > blame_info['last_commit']['timestamp']:
                    blame_info['last_commit'] = {
                        'hash': commit.hexsha[:8],
                        'date': datetime.fromtimestamp(commit.committed_date).isoformat(),
                        'message': commit.message.strip()[:100],
                        'timestamp': commit.committed_date  # Keep timestamp for comparison
                    }
            
            # Remove timestamp from final output (it was only for comparison)
            if blame_info['first_commit']:
                blame_info['first_commit'].pop('timestamp', None)
            if blame_info['last_commit']:
                blame_info['last_commit'].pop('timestamp', None)
            
            # Convert set to count for serialization
            blame_info['unique_commits'] = len(blame_info['unique_commits'])
            
            # Limit blame data to prevent context explosion
            if len(blame_info['blame_data']) > max_lines:
                blame_info['blame_data'] = blame_info['blame_data'][:max_lines]
                blame_info['truncated'] = True
            
            logger.debug(f"TEMPORAL_BLAME: {file_path.stem} has {blame_info['unique_commits']} unique commits")
            
        except Exception as e:
            logger.warning(f"BLAME_ERROR: file={file_path} error={str(e)}")
            blame_info['error'] = str(e)
        
        return blame_info

    def get_session_context(self, conversation: List[Dict[str, str]], 
                          depth: str = "basic") -> Dict[str, Any]:
        """
        Convenience method for external callers - returns context as a simple dict
        Note: Uses the user_id from the context manager initialization
        Entities are automatically extracted from the conversation
        
        Supported depth modes:
        - "basic": Top entities with ALWAYS_LOAD blocks only
        - "wide": Semantic search with ALWAYS_LOAD blocks
        - "deep": Complete entity files
        - "temporal": Complete entity files with Git blame history
        """
        request = ContextRequest(
            conversation=conversation,
            user_id=self.user_id,  # Use the initialized user_id
            depth=depth
        )
        
        response = self.assemble_context(request)
        
        # Convert to simple dict for external use
        result = {
            'always_load_blocks': response.always_load_blocks,
            'recent_timeline': response.recent_timeline,
            'session_metadata': response.session_metadata
        }
        
        # Include mode-specific fields if present
        if response.complete_entities is not None:
            result['complete_entities'] = response.complete_entities
            
        if response.temporal_blame is not None:
            result['temporal_blame'] = response.temporal_blame
        
        return result 