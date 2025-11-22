Module:
- Clean up call_llm so that it shunted to a util and called more sensibly.
- Prompt caching for retrieval during ongoing conversation turns.

Maintaining:
- Randomly sample a couple of entities for merge/link review for ongoing prunig maintenante, use last_edited and last_accessed for adjusting strenght, have the job maintain the "semantic index".
- Have a "review" agent that tries to catch obvious mistakes in entity identification. 

Storing:
- Use git tags during a commit for ease of timeline traversal.
- Be able to pass instructions on a per-session basis to filter/modify how the input is created. 

Loading:
- Create traversal system so that it can navigate links between memories.
- Have a mechanism to pull git-diff on a specific block by text searching a specific memory or memory block
- Update temporal pull so that it uses git log, but also so that it ignores the commit where the file was first created

Init: 
- Need a demo repo to show how this works in practice, and an init script.

Eval: 
https://github.com/snap-research/locomo
https://github.com/plastic-labs/honcho
https://x.com/vintrotweets
https://x.com/plastic_labs/status/1958628537969037438
https://x.com/honchodotdev
https://blog.plasticlabs.ai/blog/Beyond-the-User-Assistant-Paradigm;-Introducing-Peers
