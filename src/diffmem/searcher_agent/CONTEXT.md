# searcher_agent CONTEXT  
## BUSINESS PURPOSE  
LLM-orchestrated search: Analyze query → BM25 retrieve → Synthesize results → Iterate if low confidence. Uses OpenRouter for model abstraction/swapping. PoC: Basic iteration (up to 2 refinements); supports reconstructive retrieval.  

## USER STORIES  
- As Anna, I want query breakdown and synthesis so responses feel natural and complete.  
- As Alex, I want iterative refinements for vague memories (e.g., "that time in India" → date-filtered hits).  

## INFO FLOW  
Query → LLM_Analyze (concepts/keywords) → BM25_Search → LLM_Synthesize (summary + confidence) → If low conf: Refine → Repeat.  
[ASCII DIAGRAM]  
Query --> LLM_Analyze --> BM25 --> LLM_Synth --> CheckConf --> [Refine Loop] --> Results  

## TERMINOLOGY  
- "query_breakdown": Dict with concepts, keywords, filters (e.g., {'keywords': ['relationship', 'quelis'], 'temporal': 'evolution'}).  
- "synthesis": LLM-generated summary from snippets, with confidence score (0-1).  
- "refinement": LLM-suggested query tweaks (e.g., add synonyms).  

## ARCHITECTURAL CONSTRAINTS  
- OpenRouter only: API key via env (OPENROUTER_API_KEY).  
- Model swapping: Param-driven (e.g., 'openai/gpt-4o').  
- Iteration cap: Max 2 for PoC to prevent loops.  
- Hallucination Breaker: ALWAYS ground in BM25 snippets—prefix prompts with exact content.  