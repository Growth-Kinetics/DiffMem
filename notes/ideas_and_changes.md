Module:
- Verify that all agents have only access to the user's folder and put in checks to prevent accessing other folders.
- Clean up call_llm so that it shunted to a util and called more sensibly.
- Fix that when an index is not found it makes it but doesn´t immediately load it in same process.
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
- Using a vector db that’s created as a post commithook is probably a reasonable improvement
-  I'd reach for a single lucene index here, making the commit hash one of the document's indexed fields. This would make FTS across history nearly instant. I'd store it in an ignored folder right next to .git.

Init: 
- Need a demo repo to show how this works in practice, and an init script.