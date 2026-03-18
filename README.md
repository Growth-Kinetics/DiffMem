# DiffMem: Git-Based Differential Memory for AI Agents

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Prototype](https://img.shields.io/badge/status-prototype-orange.svg)](https://github.com/alexmrval/DiffMem)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Growth-Kinetics/DiffMem)

DiffMem is a lightweight, git-based memory backend designed for AI agents and conversational systems. It uses Markdown files for human-readable storage, Git for tracking temporal evolution through differentials, and a git-native retrieval agent that explores the repository via shell commands (grep, git log, git diff, git blame) to build targeted context. No vector databases, no embeddings, no BM25 -- just git and an LLM.

At its core, DiffMem treats memory as a versioned repository: the "current state" of knowledge is stored in editable files, while historical changes are preserved in Git's commit graph. This separation allows agents to query and search against a compact, up-to-date surface without the overhead of historical data, while enabling deep dives into evolution when needed.

## Live in Production

DiffMem powers [Annabelle](https://withanna.io/), a simulated intelligence that maintains persistent memory across thousands of conversations on WhatsApp and Messenger.

In production, DiffMem enables Annabelle to:

- Reference details from conversations weeks ago
- Track the evolution of relationships over time  
- Build structured understanding of each person she talks to
- Consolidate memories automatically as conversations grow

→ [Try a conversation with Annabelle](https://wa.me/34641376527?text=Hi%20Annabelle,%20saw%20you%20on%20Github%20%E2%80%94%20I%E2%80%99d%20like%20to%20start.%20What%20should%20I%20tell%20you%20first%3F)
→ [See how DiffMem processes a novel chapter by chapter](https://github.com/Growth-Kinetics/diffmem_sample_memory)

## Roadmap

- [ ] Indexing strategy from PoC needs to be made more robust, too memory intensive without need.
- [ ] Parametrized method for context caps on retrieval.
- [ ] Sometimes an entity will become a catch-all and the thing will insist in overloading it.
- [ ] Retrieval History so that we can spin up a "linked entities" model to support wikification
- [ ] PDF export.
- [ ] Research PoC: Visual retrieval for context compression.

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

- **Retrieval Agent** (`retrieval_agent`): A multi-turn LLM agent with a single `run(command="...")` tool that explores the memory repository via sandboxed shell commands. It reads `index.md`, probes git history for temporal patterns, and outputs a structured retrieval plan (file sections, git diffs, commit logs) that gets resolved into context.

- **API Layer** (`api.py`): Clean interface for read/write operations. Example:
  ```python
  from diffmem import DiffMemory

  memory = DiffMemory("/path/to/repo", "alex", "your-api-key")
  
  # Get context for a conversation (agent explores git to find what's relevant)
  context = memory.get_context(conversation, max_tokens=15000)
  
  # Process and commit new memory
  memory.process_and_commit_session("Had coffee with mom today...", "session-123")
  ```

The repo follows a structured layout (see `repo_guide.md` for details), with current states in Markdown files and evolution in Git commits. The retrieval agent navigates this structure using the same git commands a human developer would use.

## Why This Works

DiffMem's git-centric design solves key challenges in AI memory systems:

- **Reduced Query Surface**: Only current-state files are explored by default. The retrieval agent reads `index.md` to understand the entity landscape, then surgically loads only what's relevant. When history is needed, it pulls targeted diffs (e.g., `git diff HEAD~3 file.md`), not full archives.

- **Scalable Evolution Tracking**: Git handles 50+ years of changes efficiently. Agents can reconstruct past states (`git show <commit>:file.md`) without bloating active memory.

- **Developer-Friendly**: No DB schemas or migrations—edit Markdown directly. Git provides free versioning, branching (e.g., monthly timelines), and collaboration.

- **Lightweight**: Runs in-process, minimal deps (gitpython, openai). No ML models, no embeddings, no vector databases.


## Prototype Status and Limitations

What's working:
- Entity creation/update from transcripts.
- Git-native agent retrieval with temporal reasoning.
- Targeted context assembly (file sections, diffs, commit logs).
- Fallback to baseline (user entity) when agent fails.

Known limitations:
- Agent retrieval quality depends on the LLM model used.
- No multi-user concurrency locks.
- Writer agent prompt tuning ongoing.

We're sharing this as open-source R&D to spark discussion. Feedback welcome!

## Future Vision: Where This Could Go

DiffMem points to a future where AI memory is as versioned and collaborative as code. Imagine:

- **Agent-Driven Pruning**: LLMs that "forget" low-strength memories by archiving to git branches, mimicking neural plasticity.

- **Collaborative Memories**: Multi-agent systems sharing repos, with merge requests for "memory reconciliation."

- **Temporal Agents**: Specialized models that query git logs to answer "how did I change?"—enabling self-reflective AI.

- **Multi-Provider Retrieval**: Swap between OpenRouter, Cerebras, or any OpenAI-compatible provider for the retrieval agent.

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
