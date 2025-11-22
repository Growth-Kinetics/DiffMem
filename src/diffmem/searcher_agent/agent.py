# CAPABILITY: LLM-orchestrated search over BM25 index via OpenRouter  
# INPUTS: Session pairs (List[Dict[str, str]] e.g., [{'role': 'user', 'content': msg}, {'role': 'assistant', 'content': resp}]), index (Dict), model (str)  
# OUTPUTS: Dict with synthesized response, confidence, snippets  
# CONSTRAINTS: PoC minimal—distill query from conversation; up to 2 iterations; env-based API key  
# DEPENDENCIES: import requests; from diffmem.bm25_indexer.api import search  

import requests  
import os  
import json  
import re  
from typing import Dict, List, Union
from diffmem.bm25_indexer.api import search

# Structured logging for agent perception  
import logging  
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  

#for local dev
from dotenv import load_dotenv
load_dotenv()

# OpenRouter config
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"  
API_KEY = os.getenv("OPENROUTER_API_KEY")  
if not API_KEY:  
    raise ValueError("OPENROUTER_API_KEY not set in environment")  


def extract_json_from_llm_response(text: str) -> dict:
    """
    Robust JSON extraction from LLM responses.
    Handles markdown fences, trailing text, and malformed wrapping.
    """
    if not text or not text.strip():
        raise ValueError("Empty response from LLM")
    
    cleaned = text.strip()
    
    # Strip markdown code fences
    if cleaned.startswith('```'):
        # Remove opening fence and language identifier
        lines = cleaned.split('\n')
        if lines[0].startswith('```'):
            lines = lines[1:]  # Drop first line
        cleaned = '\n'.join(lines)
    
    if cleaned.endswith('```'):
        cleaned = cleaned[:-3]
    
    cleaned = cleaned.strip()
    
    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Extract first complete JSON object using regex
    # Handles nested braces
    brace_count = 0
    start_idx = cleaned.find('{')
    
    if start_idx == -1:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    
    for i in range(start_idx, len(cleaned)):
        if cleaned[i] == '{':
            brace_count += 1
        elif cleaned[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                # Found complete JSON object
                json_str = cleaned[start_idx:i+1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON extracted: {e}")
    
    raise ValueError(f"Incomplete JSON object in response: {text[:200]}")


def llm_call(messages: List[Dict[str, str]], model: str, max_tokens: int = 500, json: bool = True) -> str:  
    """  
    Calls OpenRouter API with messages.
    """  
    headers = {  
        "Authorization": f"Bearer {API_KEY}",  
        "Content-Type": "application/json"  
    }  
    payload = {  
        "model": model,  
        "messages": messages,  
        "max_tokens": max_tokens, 
        "temperature": 0.2 
    }  
    
    if json == True:
        payload["response_format"] = { "type": "json_object"}
    
    try:  
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)  
        response.raise_for_status()  
        content = response.json()["choices"][0]["message"]["content"]  
        logger.info(f"LLM_CALL: model={model} tokens_in={sum(len(m['content'].split()) for m in messages)} response_length={len(content)}")  
        return content  
    except Exception as e:  
        logger.error(f"LLM_ERROR: {str(e)}")  
        return ""  

def distill_query_from_conversation(session_pairs: List[Dict[str, str]], model: str) -> List[str]:  
    """  
    Distills implicit queries from conversation pairs.  
    Returns list of query strings optimized for BM25.
    """  
    if not session_pairs:
        logger.warning("DISTILL_EMPTY: No conversation pairs provided")
        return []
    
    transcript = "\n".join([
        f"{pair['role'].capitalize()}: {pair['content']}" 
        for pair in session_pairs 
        if 'role' in pair and 'content' in pair
    ])
    
    if not transcript.strip():
        logger.warning("DISTILL_NO_CONTENT: Conversation pairs contain no content")
        return []
    
    distill_prompt = [  
        {
            "role": "system", 
            "content": """[CONVERSATION_DISTILL]: Extract implicit memory queries from transcript.
Focus on: entities (people, places), contexts, emotional patterns, temporal references.
Output ONLY valid JSON (no markdown fences): {"derived_queries": ["query1", "query2", ...]}
Queries should be 1-3 terms optimized for BM25 (e.g., "family dynamics", "mother relationship", "grandfather death")."""
        },  
        {"role": "user", "content": f"Transcript:\n{transcript}\n\nExtract queries:"}  
    ]  
    
    for attempt in range(2):
        analysis_raw = llm_call(distill_prompt, model, max_tokens=150, json=True)
        
        try:
            # ROBUST JSON EXTRACTION
            analysis = extract_json_from_llm_response(analysis_raw)
            
            # Validate structure
            if "derived_queries" not in analysis:
                raise ValueError("Missing 'derived_queries' field in response")
            
            derived_queries = analysis["derived_queries"]
            
            # Type validation
            if not isinstance(derived_queries, list):
                raise ValueError(f"'derived_queries' must be list, got {type(derived_queries)}")
            
            # Filter empty/invalid queries
            valid_queries = [
                q.strip() for q in derived_queries 
                if isinstance(q, str) and q.strip()
            ]
            
            if not valid_queries:
                logger.warning(f"DISTILL_EMPTY_QUERIES: attempt={attempt} raw={analysis_raw[:200]}")
                if attempt == 0:
                    # Retry with more explicit instruction
                    distill_prompt.append({"role": "assistant", "content": analysis_raw})
                    distill_prompt.append({
                        "role": "user", 
                        "content": "No valid queries extracted. Provide at least one query string."
                    })
                    continue
                return []  # Give up after retries
            
            logger.info(f"DISTILLED_QUERIES: count={len(valid_queries)} queries={valid_queries} attempt={attempt}")
            return valid_queries
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"DISTILL_PARSE_FAIL: attempt={attempt} error={e} raw={analysis_raw[:200]}")
            
            if attempt == 0:
                # Add validation message for retry
                distill_prompt.append({"role": "assistant", "content": analysis_raw})
                distill_prompt.append({
                    "role": "user", 
                    "content": f"Parse error: {str(e)}. Return ONLY valid JSON with no markdown fences."
                })
            else:
                logger.error(f"DISTILL_MAX_RETRIES: Failed after {attempt + 1} attempts")
                return []  # Fallback to empty
    
    return []


def orchestrate_query(session_pairs: List[Dict[str, str]], index: Dict, model: str = "openai/gpt-4o", k: int = 5) -> Dict:  
    """  
    Orchestrates search from conversation pairs.
    Returns synthesized context with snippets and metadata.
    """  
    # Guard: Handle empty conversations
    if not session_pairs:
        logger.warning("ORCHESTRATE_EMPTY: No conversation provided")
        return {
            'response': '',
            'snippets': [],
            'derived_queries': [],
            'metadata': {'reason': 'empty_conversation'}
        }
    
    # Step 1: Distill queries
    search_terms = distill_query_from_conversation(session_pairs, model)
    
    if not search_terms:
        logger.warning("ORCHESTRATE_NO_QUERIES: Distillation produced no valid queries")
        return {
            'response': '',
            'snippets': [],
            'derived_queries': [],
            'metadata': {'reason': 'no_queries_distilled'}
        }
    
    # Step 2: BM25 Retrieve
    all_snippets = []
    for term in search_terms:  
        results = search(index, term, k)
        all_snippets.extend([res['snippet'] for res in results])
    
    # Deduplicate snippets
    unique_snippets = list({
        frozenset(d.items()): d 
        for d in all_snippets
    }.values())
    
    if not unique_snippets:
        logger.warning(f"ORCHESTRATE_NO_RESULTS: No snippets found for queries={search_terms}")
        return {
            'response': 'No relevant context found in memory.',
            'snippets': [],
            'derived_queries': search_terms,
            'metadata': {'reason': 'no_search_results'}
        }
    
    snippet_text = '\n'.join([
        f"[Snippet {i}] {s['content'][:200]}..." 
        for i, s in enumerate(unique_snippets, 1)
    ])

    # Step 3: LLM Synthesize
    synth_prompt = [  
        {
            "role": "system", 
            "content": """[SYNTHESIS]: Create concise context summary from memory snippets.
Focus on: relevance to current conversation, temporal patterns, relational dynamics.
End with: CONFIDENCE: <0.0-1.0> (relevance score)"""
        },  
        {
            "role": "user", 
            "content": f"Queries: {', '.join(search_terms)}\n\nMemory Snippets:\n{snippet_text}\n\nSynthesize:"
        }  
    ]  
    
    synth_raw = llm_call(synth_prompt, model, max_tokens=400, json=False)

    return {  
        'response': synth_raw,  
        'snippets': unique_snippets,
        'derived_queries': search_terms,
        'metadata': {
            'snippet_count': len(unique_snippets),
            'query_count': len(search_terms)
        }
    }