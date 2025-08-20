#!/usr/bin/env python3
"""
DiffMem API Usage Examples

This file demonstrates how to use the DiffMemory API in various scenarios.
All operations are module-driven - no servers or endpoints required.
"""

import os
from pathlib import Path
from diffmem import DiffMemory, create_memory_interface, quick_search

# Example repository and user setup
REPO_PATH = "/path/to/your/memory/repo"  # Update this path
USER_ID = "alex"  # Update this to your user ID
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # Set in environment

def basic_usage_example():
    """Basic read/write operations"""
    print("=== Basic Usage Example ===")
    
    # Initialize memory interface
    memory = DiffMemory(REPO_PATH, USER_ID, OPENROUTER_API_KEY)
    
    # Validate setup
    validation = memory.validate_setup()
    if not validation['valid']:
        print(f"Setup issues: {validation['issues']}")
        return
    
    # Get repository status
    status = memory.get_repo_status()
    print(f"Repository has {status['memory_files_count']} memory files")
    print(f"BM25 index contains {status['index_stats']['total_blocks']} blocks")
    
    # Example conversation for context
    conversation = [
        {"role": "user", "content": "How has my relationship with my mother evolved?"},
        {"role": "assistant", "content": "I'd be happy to help you understand that. Let me look at your memories..."}
    ]
    
    # Get basic context (ALWAYS_LOAD blocks only)
    basic_context = memory.get_context(conversation, depth="basic")
    print(f"Basic context loaded {len(basic_context['always_load_blocks'])} blocks")
    
    # Get deep context (complete entity files)
    deep_context = memory.get_context(conversation, depth="deep")
    if 'complete_entities' in deep_context:
        print(f"Deep context loaded {len(deep_context['complete_entities'])} complete entities")
    
    # Direct search
    search_results = memory.search("family dynamics", k=3)
    print(f"Direct search found {len(search_results)} results")
    for i, result in enumerate(search_results):
        print(f"  Result {i+1}: Score {result['score']:.3f} - {result['snippet']['id']}")
    
    # Orchestrated search (LLM-guided)
    orchestrated = memory.orchestrated_search(conversation)
    print(f"Orchestrated search derived query: {orchestrated['derived_query']}")
    print(f"Found {len(orchestrated['snippets'])} relevant snippets")

def write_operations_example():
    """Memory writing and session management"""
    print("\n=== Write Operations Example ===")
    
    memory = DiffMemory(REPO_PATH, USER_ID, OPENROUTER_API_KEY)
    
    # Example session transcript
    session_transcript = """
    Had a great conversation with mom today. She told me about her childhood in Chicago
    and how she met dad at the university library. She seemed more relaxed than usual
    and mentioned she's been taking painting classes. I think our relationship is 
    getting stronger since we started having these weekly calls.
    
    Also met with my friend Sarah for coffee. She's dealing with some work stress
    and asked for advice about switching careers. I shared my experience from when
    I made my career change last year.
    """
    
    session_id = "session-2024-01-15-001"
    
    # Process session (stages changes, doesn't commit)
    print("Processing session...")
    memory.process_session(session_transcript, session_id)
    print("Session processed and changes staged")
    
    # At this point, changes are staged but not committed
    # You can review them with git diff if needed
    
    # Commit the session
    print("Committing session...")
    memory.commit_session(session_id)
    print("Session committed to repository")
    
    # Alternative: process and commit in one step
    # memory.process_and_commit_session(session_transcript, session_id)

def context_depth_examples():
    """Demonstrate different context depths"""
    print("\n=== Context Depth Examples ===")
    
    memory = DiffMemory(REPO_PATH, USER_ID, OPENROUTER_API_KEY)
    
    conversation = [
        {"role": "user", "content": "Tell me about my relationship patterns over time"},
        {"role": "assistant", "content": "I'll analyze your relationship memories..."}
    ]
    
    # Basic: Top entities with ALWAYS_LOAD blocks only
    basic = memory.get_context(conversation, depth="basic")
    print(f"Basic depth: {len(basic['always_load_blocks'])} always-load blocks")
    
    # Wide: Semantic search with ALWAYS_LOAD blocks
    wide = memory.get_context(conversation, depth="wide")
    print(f"Wide depth: {len(wide['always_load_blocks'])} blocks with semantic search")
    
    # Deep: Complete entity files
    deep = memory.get_context(conversation, depth="deep")
    if 'complete_entities' in deep:
        print(f"Deep depth: {len(deep['complete_entities'])} complete entity files")
    
    # Temporal: Complete files with Git blame history
    temporal = memory.get_context(conversation, depth="temporal")
    if 'temporal_blame' in temporal:
        print(f"Temporal depth: {len(temporal['temporal_blame'])} files with git history")

def convenience_functions_example():
    """Using convenience functions for quick operations"""
    print("\n=== Convenience Functions Example ===")
    
    # Quick search without full initialization
    results = quick_search(REPO_PATH, "work stress", k=3)
    print(f"Quick search found {len(results)} results")
    
    # Create interface with environment variable for API key
    memory = create_memory_interface(REPO_PATH, USER_ID)
    
    # Get specific data
    user_entity = memory.get_user_entity()
    print(f"User entity loaded: {user_entity['entity_name']}")
    
    recent_timeline = memory.get_recent_timeline(days_back=7)
    print(f"Recent timeline: {len(recent_timeline)} entries from last 7 days")

def error_handling_example():
    """Demonstrate proper error handling"""
    print("\n=== Error Handling Example ===")
    
    try:
        # This will fail if paths don't exist
        memory = DiffMemory("/nonexistent/path", "nonuser", "fake-key")
    except FileNotFoundError as e:
        print(f"Expected error: {e}")
    
    try:
        # This will fail without API key
        memory = create_memory_interface(REPO_PATH, USER_ID, openrouter_api_key="")
    except ValueError as e:
        print(f"Expected error: {e}")

def integration_example():
    """Example of integrating with a chat agent"""
    print("\n=== Chat Agent Integration Example ===")
    
    class SimpleChatAgent:
        def __init__(self, memory_repo_path: str, user_id: str):
            self.memory = create_memory_interface(memory_repo_path, user_id)
            self.conversation_history = []
        
        def process_user_message(self, user_message: str) -> str:
            # Add user message to conversation
            self.conversation_history.append({
                "role": "user", 
                "content": user_message
            })
            
            # Get context for the conversation
            context = self.memory.get_context(
                self.conversation_history[-5:],  # Last 5 messages
                depth="basic"
            )
            
            # Here you would use the context to generate a response
            # For this example, we'll just return a summary
            response = f"I found {len(context['always_load_blocks'])} relevant memory blocks"
            
            # Add assistant response to conversation
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })
            
            return response
        
        def end_session(self, session_id: str):
            # Process the entire conversation as a memory session
            full_conversation = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in self.conversation_history
            ])
            
            self.memory.process_and_commit_session(full_conversation, session_id)
            print(f"Session {session_id} saved to memory")
    
    # Example usage of the chat agent
    if OPENROUTER_API_KEY and Path(REPO_PATH).exists():
        agent = SimpleChatAgent(REPO_PATH, USER_ID)
        
        response1 = agent.process_user_message("How am I doing with my goals?")
        print(f"Agent response: {response1}")
        
        response2 = agent.process_user_message("What about my relationships?")
        print(f"Agent response: {response2}")
        
        # End session and save to memory
        agent.end_session("chat-session-001")

if __name__ == "__main__":
    # Update these paths before running
    if not Path(REPO_PATH).exists() or not OPENROUTER_API_KEY:
        print("Please update REPO_PATH and set OPENROUTER_API_KEY environment variable")
        print("Then run the examples:")
        print(f"  REPO_PATH = '{REPO_PATH}'")
        print(f"  USER_ID = '{USER_ID}'")
        print(f"  OPENROUTER_API_KEY = {'set' if OPENROUTER_API_KEY else 'NOT SET'}")
    else:
        # Run all examples
        basic_usage_example()
        write_operations_example()
        context_depth_examples()
        convenience_functions_example()
        error_handling_example()
        integration_example() 