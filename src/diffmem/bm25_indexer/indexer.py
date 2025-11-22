# CAPABILITY: In-memory BM25 indexing and search over Markdown memory files  
# INPUTS: Repo path (str), query (str), k (int)  
# OUTPUTS: List of ranked snippets (dicts)  
# CONSTRAINTS: PoC minimal—basic parsing, strength boosts; rebuild on each run (<1s target)  
# DEPENDENCIES: from rank_bm25 import BM25Okapi; import re, os, pathlib  

from rank_bm25 import BM25Okapi  
import re  
import os  
from pathlib import Path  
from typing import List, Dict  

# Structured logging for agent perception (e.g., feedback loops)  
import logging  
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  

def parse_blocks(file_path: str) -> List[Dict[str, str]]:  
    """  
    Parses Markdown file into blocks using /START to /END delimiters.  
    - Captures full block content.  
    - Extracts header from first line, derives clean ID and strength.  
    - Handles case-insensitivity, missing tags.  
    - PoC: Log parse details for perception.  
    @returns: List[{'id': str, 'content': str, 'strength': float, 'always_load': bool, 'file_path': str}]  
    """  
    with open(file_path, 'r', encoding='utf-8') as f:  
        content = f.read()  
    blocks = []  
    # Regex: Capture from /START to /END (non-greedy, case-insensitive)  
    pattern = r'/START\s*(.*?)/END'  
    matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)  
    for block_content in matches:  
        if not block_content.strip(): continue  # Skip empty  
        # Extract header: First line starting with ###  
        header_match = re.search(r'###\s*(.+)', block_content, re.MULTILINE)  
        header = header_match.group(1).strip() if header_match else 'unnamed_block'  
        # Extract strength from header (e.g., [Strength: High])  
        strength_match = re.search(r'\[Strength:\s*(High|Medium|Low)\]', header, re.IGNORECASE)  
        strength_str = strength_match.group(1) if strength_match else 'Low'  # Default Low if missing  
        strength_map = {'High': 1.5, 'Medium': 1.2, 'Low': 1.0}  
        strength = strength_map.get(strength_str.capitalize(), 1.0)  
        # Clean ID: Remove emojis/bold/tags, normalize  
        clean_header = re.sub(r'\[.*?\]|\*+|[^\w\s]', '', header).strip().replace(' ', '_').lower()  
        always_load = '[ALWAYS_LOAD]' in block_content  
        blocks.append({  
            'id': clean_header,  
            'content': block_content.strip(),  
            'strength': strength,  
            'always_load': always_load,  
            'file_path': file_path  
        })  
        logger.info(f"BLOCK_PARSED: file={file_path} id={clean_header} strength={strength} length={len(block_content)}")  
    if not blocks:  
        logger.warning(f"NO_BLOCKS: file={file_path} - Ensure /START and /END delimiters")  
    return blocks

def build_index(repo_path: str) -> Dict:  
    """  
    Builds in-memory BM25 index from all .md files in repo.  
    - Traverses users/*/memories/**.md recursively.  
    - Parses into blocks, tokenizes (simple split).  
    - PoC: Flat corpus; later add hierarchy.  
    @returns: {'bm25': BM25Okapi, 'corpus': List[Dict], 'tokens': List[List[str]]}  
    """  
    corpus = []  
    all_blocks = []  
    path = Path(repo_path)  
    for md_file in path.rglob('*.md'):  
        if 'repo_guide.md' in str(md_file) or 'index.md' in str(md_file) or '/sessions/' in str(md_file).replace('\\', '/'): continue  # Skip guides and sessions  
        blocks = parse_blocks(str(md_file))  
        all_blocks.extend(blocks)  
    # Tokenize: Simple word split + lowercase (PoC; improve with NLTK later)  
    tokens = [[word.lower() for word in block['content'].split()] for block in all_blocks]  
    bm25 = BM25Okapi(tokens)  
    logger.info(f"INDEX_BUILT: blocks={len(all_blocks)} tokens_avg={sum(len(t) for t in tokens)/len(tokens) if tokens else 0}")  
    return {'bm25': bm25, 'corpus': all_blocks, 'tokens': tokens}  

def search(index: Dict, query: str, k: int = 5) -> List[Dict]:  
    """  
    Searches BM25 index with strength/recency boosts (PoC: strength only; recency via metadata later).  
    - Tokenize query similarly.  
    - Get scores, apply strength multiplier.  
    - Sort and return top-K snippets with metadata.  
    @returns: List[{'score': float, 'snippet': Dict (from corpus)}] sorted descending  
    """  
    query_tokens = [word.lower() for word in query.split()]  
    scores = index['bm25'].get_scores(query_tokens)  
    # Apply strength boost  
    boosted_scores = [score * corpus_item['strength'] for score, corpus_item in zip(scores, index['corpus'])]  
    # Get top-K indices  
    top_indices = sorted(range(len(boosted_scores)), key=lambda i: boosted_scores[i], reverse=True)[:k]  
    results = []  
    for i in top_indices:  
        results.append({  
            'score': boosted_scores[i],  
            'snippet': index['corpus'][i]  
        })  
        logger.debug(f"SEARCH_HIT: query={query} id={index['corpus'][i]['id']} score={boosted_scores[i]} strength={index['corpus'][i]['strength']}")  
    if not results:  
        logger.warning(f"SEARCH_MISS: query={query} - Suggest refinements")  
    return results