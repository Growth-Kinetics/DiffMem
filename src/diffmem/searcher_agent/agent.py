# CAPABILITY: LLM-orchestrated search over BM25 index via OpenRouter  
# INPUTS: Session pairs (List[Dict[str, str]] e.g., [{'role': 'user', 'content': msg}, {'role': 'assistant', 'content': resp}]), index (Dict), model (str)  
# OUTPUTS: Dict with synthesized response, confidence, snippets  
# CONSTRAINTS: PoC minimal—distill query from conversation; up to 2 iterations; env-based API key  
# DEPENDENCIES: import requests; from diffmem.bm25_indexer.api import search  

import requests  
import os  
import json  
import re  
from typing import Dict, List  
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

def distill_query_from_conversation(session_pairs: List[Dict[str, str]], model: str) -> str:  
    """  
    Distills implicit query from conversation pairs.  
    - Formats transcript with roles.  
    - LLM extracts as JSON for parse safety.   
    @returns: Derived query (str) or empty on fail  
    """  
    transcript = "\n".join([f"{pair['role'].capitalize()}: {pair['content']}" for pair in session_pairs if 'role' in pair and 'content' in pair])  
    distill_prompt = [  
        {"role": "system", "content": "[CONVERSATION_DISTILL]: From TRANSCRIPT ONLY, extract implicit memory query. Focus on key entities, patterns, evolving needs (e.g., family dynamics, past events). Output JSON: {'derived_queries': ['concise query string', 'concise query string', ...]}. The query strings should be optimized for BM25 search and no more that three terms signaling people, context, places, etc. An example of a good query is 'family dynamics' or 'mother' or 'grandfather death'."},  
        {"role": "user", "content": f"Transcript:\n{transcript}\nDistill:"}  
    ]  
    for attempt in range(2):  # PoC retry  
        analysis_raw = llm_call(distill_prompt, model, max_tokens=100, json=True)  
        try:  
            analysis = json.loads(analysis_raw)  
            derived_query = analysis.get('derived_queries', '')  
            if derived_query:  
                logger.debug(f"DISTILLED_QUERY: from_pairs={len(session_pairs)} query={derived_query} attempt={attempt}")  
                return derived_query  
        except json.JSONDecodeError:  
            logger.warning(f"DISTILL_PARSE_FAIL: attempt={attempt} raw={analysis_raw[:100]}... Retrying with validation.")  
            # Add validation message for retry  
            distill_prompt.append({"role": "assistant", "content": analysis_raw})  
            distill_prompt.append({"role": "user", "content": "Invalid JSON. Retry with valid JSON output."})  
    logger.error("DISTILL_MAX_RETRIES: Failed to parse valid query")  
    return ""  # Fallback  

def orchestrate_query(session_pairs: List[Dict[str, str]], index: Dict, model: str = "openai/gpt-4o", k: int = 5) -> Dict:  
    """  
    Orchestrates search from conversation pairs. (Core logic unchanged, but now benefits from robust distillation)  
    """  
    iterations = 0  
    search_terms = distill_query_from_conversation(session_pairs, model)
    all_snippets = []
    # Step 1: BM25 Retrieve
    
    for term in search_terms:  
        results = search(index, term, k)
        snippets = [res['snippet'] for res in results]
        all_snippets.extend(snippets)
    unique = list({frozenset(d.items()): d for d in all_snippets}.values())
    snippet_text = '\n'.join([f"Snippet {i}: {s['content']}" for i, s in enumerate(unique, 1)])

    # Step 2: LLM Synthesize + Confidence
    synth_prompt = [  
        {"role": "system", "content": "[SYNTHESIS]: Build context summary from snippets for search results. Output: Summary text, then CONFIDENCE: 0-1 (relevance to implicit needs)."},  
        {"role": "user", "content": f"Derived Query: {search_terms}\nSnippets:\n{snippet_text}"}  
    ]  
    synth_raw = llm_call(synth_prompt, model, json=False)

    return {  
        'response': synth_raw,  
        'snippets': snippets,  
        'derived_query': search_terms  
    }