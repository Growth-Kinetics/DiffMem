# bm25_indexer CONTEXT  
## BUSINESS PURPOSE  
In-memory BM25 indexing for fast, explainable search over Markdown blocks. Supports reconstructive retrieval: Pull current-state snippets with strength/recency boosts. PoC: Basic keyword matching for context assembly.  

## USER STORIES  
- As Anna, I want top-K relevant blocks for a query so I can build context without full repo loads.  
- As Alex (user), I want searches to prioritize high-strength memories for accurate "now" recall.  

## INFO FLOW  
Repo path → Read/parse blocks → Tokenize → Build BM25 index → Query → Ranked snippets.  
[ASCII DIAGRAM]  
Repo --> ParseBlocks --> Tokenize --> BM25Index --> Search --> Snippets (w/ metadata)  

## TERMINOLOGY  
- "block": Markdown section from ### Header to /END (e.g., Relationship Dynamics).  
- "strength_boost": Multiply BM25 score by factor (High=1.5, Medium=1.2, Low=1.0).  
- "snippet": Dict with content, file_path, block_id, score, metadata (e.g., strength, last_diff).  

## ARCHITECTURAL CONSTRAINTS  
- In-memory only: Rebuild on init; no persistence.  
- Token limit: Snippets < 10k chars each.  
- Hallucination Breaker: NEVER synthesize content—return exact parsed blocks.  
- PoC Limit: Keyword BM25; no LLM orchestration yet (add in context_builder later).