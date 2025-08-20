The context manager is responsible for creating the ongoing system prompt the chat agent will use to continue the conversation with the user. 

It will use the current conversation to determine the memories that are required and the depth of the construction needed. It will then craft facts and narratives, determine session_id's and return them all to the requesting agent. 

The Input is the ongoing conversation transcript, the output will be built from the memory repository. 

For ongoing conversations that require nothing but basic context, the manager will return the users's entity + the "allways load" and "semantic index" blocks for the top 5 entities and those mentioned in the conversation thread, as well as the latest timeline file. The agent will use the index.md file that contains the library of all semantic indexes for the user. 

When a conversation requires a wider memory search, the context builder will use the searcher_agent to search additional information on the given topic. It will use git functionality to figure out what commits created the information that was retrieved, and return the commit message (which is an ID of the session where the memory was created)

When a conversation requires a deeper dive on an specific entity (inlcuding the user's own), it will return the entire entity but will also use git to understand how the entity has changed over time, and return that additional information along with the rest of the context. 

For any given entity, if the conversation requires understanding how that entity or group of entities have changed, the context manager can use the timeline files to understand when they were created or referenced, and use git funcitonality to understand how the whole memory repository has changed over time to build large-scale arcs of knowledge. 

This git functionality is not limite to whole entities, it can be used when callenging specific blocks from an entity. 
