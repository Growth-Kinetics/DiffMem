# DiffMem: Git-Based Differential Memory for AI Agents

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Prototype](https://img.shields.io/badge/status-prototype-orange.svg)](https://github.com/alexmrval/DiffMem)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Growth-Kinetics/DiffMem)

DiffMem is a lightweight, git-based memory backend designed for AI agents and conversational systems. It uses Markdown files for human-readable storage, Git for tracking temporal evolution through differentials, and an in-memory BM25 index for fast, explainable retrieval. This project is a proof-of-concept (PoC) exploring how version control systems can serve as a foundation for efficient, scalable memory in AI applications.

At its core, DiffMem treats memory as a versioned repository: the "current state" of knowledge is stored in editable files, while historical changes are preserved in Git's commit graph. This separation allows agents to query and search against a compact, up-to-date surface without the overhead of historical data, while enabling deep dives into evolution when needed.

## Why Git for AI Memory?

Traditional memory systems for AI agents often rely on databases, vector stores, or graph structures. These work well for certain scales but can become bloated or inefficient when dealing with long-term, evolving personal knowledge. DiffMem takes a different path by leveraging Git's strengths:

- **Current-State Focus**: Memory files store only the "now" view of information (e.g., current relationships, facts, or timelines). This reduces the surface area for queries and searches, making operations faster and more token-efficient in LLM contexts. Historical states are not loaded by default—they live in Git's history, accessible on-demand.

- **Differential Intelligence**: Git diffs and logs provide a natural way to track how memories evolve. Agents can ask "How has this fact changed over time?" without scanning entire histories, pulling only relevant commits. This mirrors how human memory reconstructs events from cues, not full replays.

- **Durability and Portability**: Plaintext Markdown ensures memories are human-readable and tool-agnostic. Git's distributed nature means your data is backup-friendly and not locked into proprietary formats.

- **Efficiency for Agents**: By separating "surface" (current files) from "depth" (git history), agents can be selective—load the now for quick responses, dive into diffs for analytical tasks. This keeps context windows lean while enabling rich temporal reasoning.

This approach shines for long-horizon AI systems where memories accumulate over years: it scales without sprawl, maintains auditability, and allows "smart forgetting" through pruning while preserving reconstructability.

## How It Works

DiffMem is structured as importable modules—no servers required. Key components:

- **Writer Agent** (`writer_agent`): Analyzes conversation transcripts, identifies/creates entities, stages updates in Git's working tree. Commits are explicit, ensuring atomic changes.

- **Context Manager** (`context_manager`): Assembles query-relevant context at varying depths (basic: core blocks; wide: semantic search; deep: full files; temporal: with git history).

- **Searcher Agent** (`searcher_agent`): LLM-orchestrated BM25 search—distills queries from conversations, retrieves snippets, synthesizes responses.

- **API Layer** (`api.py`): Clean interface for read/write operations. Example:
  ```python
  from diffmem import DiffMemory

  memory = DiffMemory("/path/to/repo", "alex", "your-api-key")
  
  # Get context for a conversation
  context = memory.get_context(conversation, depth="deep")
  
  # Process and commit new memory
  memory.process_and_commit_session("Had coffee with mom today...", "session-123")
  ```

The repo follows a structured layout (see `repo_guide.md` for details), with current states in Markdown files and evolution in Git commits. Indexing is in-memory for speed, rebuilt on demand.

## Why This Works

DiffMem's git-centric design solves key challenges in AI memory systems:

- **Reduced Query Surface**: Only current-state files are indexed/searched by default. This minimizes noise in BM25 results and keeps LLM contexts concise—crucial for token limits. When history is needed, agents pull targeted diffs (e.g., `git diff HEAD~1 file.md`), not full archives.

- **Scalable Evolution Tracking**: Git handles 50+ years of changes efficiently. Agents can reconstruct past states (`git show <commit>:file.md`) without bloating active memory.

- **Developer-Friendly**: No DB schemas or migrations—edit Markdown directly. Git provides free versioning, branching (e.g., monthly timelines), and collaboration.

- **Lightweight PoC**: Runs in-process, minimal deps (gitpython, rank-bm25, sentence-transformers). Easy to hack on.


## Prototype Status and Limitations

This is an early PoC—functional but not production-hardened. What's working:
- Entity creation/update from transcripts.
- Multi-depth context assembly.
- Semantic/BM25 hybrid search.
- Git-based temporal queries.

Known limitations:
- No automatic git sync (manual pulls/pushes).
- Basic error handling.
- Index rebuilds on every init (add caching for production).
- No multi-user concurrency locks.

We're sharing this as open-source R&D to spark discussion. Feedback welcome!

## Future Vision: Where This Could Go

DiffMem points to a future where AI memory is as versioned and collaborative as code. Imagine:

- **Agent-Driven Pruning**: LLMs that "forget" low-strength memories by archiving to git branches, mimicking neural plasticity.

- **Collaborative Memories**: Multi-agent systems sharing repos, with merge requests for "memory reconciliation."

- **Temporal Agents**: Specialized models that query git logs to answer "how did I change?"—enabling self-reflective AI.

- **Hybrid Stores**: Combine with vector embeddings for semantic depth, using git as the "diff layer" over embeddings.

- **Open-Source Ecosystem**: Plugins for voice input, mobile sync, or integration with tools like Obsidian.

As AI agents become long-lived companions, git-like systems could make them evolvable without data silos. We're excited to see where the community takes this—perhaps toward distributed, privacy-first personal AIs.

This is an R&D project from Growth Kinetics, a boutique data solutions agency specializing in AI enablement. We're exploring how differential memory can power next-gen agents. We'd love collaborations, PRs, or honest feedback to improve it.

## Getting Started

1. Clone the repo: `git clone https://github.com/alexmrval/DiffMem.git`
2. Install deps: `pip install -r requirements.txt`
3. Set env: `export OPENROUTER_API_KEY=your_key`
4. Run examples: `python examples/usage.py`

See `examples/` for full demos.

## Contributing

Fork, experiment, PR! We're looking for:
- Git sync optimizations.
- Advanced search plugins.
- Real-world integrations.

Issues/PRs welcome. Let's build the future of AI memory together.

License: MIT  
Growth Kinetics © 2025
