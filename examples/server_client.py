#!/usr/bin/env python3
"""
DiffMem Server Client Example

Demonstrates how to interact with the simplified DiffMem FastAPI server.
The server manages its own GitHub repository and provides direct access to memory operations.
"""

import asyncio
import os
import json
from typing import Dict, List, Any
import httpx
from datetime import datetime

class DiffMemClient:
    """
    Async client for DiffMem FastAPI server
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
    
    async def get_context(self, user_id: str, conversation: List[Dict[str, str]], 
                         depth: str = "basic") -> Dict:
        """Get assembled context for a conversation"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/memory/{user_id}/context",
                json={
                    "conversation": conversation,
                    "depth": depth
                }
            )
            response.raise_for_status()
            return response.json()
    
    async def search_memory(self, user_id: str, query: str, k: int = 5) -> Dict:
        """Search memory using BM25"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/memory/{user_id}/search",
                json={
                    "query": query,
                    "k": k
                }
            )
            response.raise_for_status()
            return response.json()
    
    async def orchestrated_search(self, user_id: str, conversation: List[Dict[str, str]]) -> Dict:
        """LLM-orchestrated search from conversation"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/memory/{user_id}/orchestrated-search",
                json={
                    "conversation": conversation
                }
            )
            response.raise_for_status()
            return response.json()
    
    async def get_user_entity(self, user_id: str) -> Dict:
        """Get the complete user entity file"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/memory/{user_id}/user-entity")
            response.raise_for_status()
            return response.json()
    
    async def get_recent_timeline(self, user_id: str, days_back: int = 30) -> Dict:
        """Get recent timeline entries"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/memory/{user_id}/recent-timeline",
                params={"days_back": days_back}
            )
            response.raise_for_status()
            return response.json()
    
    async def process_session(self, user_id: str, memory_input: str, session_id: str, 
                            session_date: str = None) -> Dict:
        """Process session transcript and stage changes"""
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/memory/{user_id}/process-session",
                json={
                    "memory_input": memory_input,
                    "session_id": session_id,
                    "session_date": session_date
                }
            )
            response.raise_for_status()
            return response.json()
    
    async def commit_session(self, user_id: str, session_id: str) -> Dict:
        """Commit staged changes for a session"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/memory/{user_id}/commit-session",
                json={
                    "session_id": session_id
                }
            )
            response.raise_for_status()
            return response.json()
    
    async def process_and_commit_session(self, user_id: str, memory_input: str, session_id: str,
                                       session_date: str = None) -> Dict:
        """Process and immediately commit a session"""
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/memory/{user_id}/process-and-commit",
                json={
                    "memory_input": memory_input,
                    "session_id": session_id,
                    "session_date": session_date
                }
            )
            response.raise_for_status()
            return response.json()
    
    async def rebuild_index(self, user_id: str) -> Dict:
        """Force rebuild of BM25 index"""
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{self.base_url}/memory/{user_id}/rebuild-index")
            response.raise_for_status()
            return response.json()
    
    async def get_status(self, user_id: str) -> Dict:
        """Get repository status and statistics"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/memory/{user_id}/status")
            response.raise_for_status()
            return response.json()
    
    async def validate_setup(self, user_id: str) -> Dict:
        """Validate memory setup for user"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/memory/{user_id}/validate")
            response.raise_for_status()
            return response.json()
    
    async def manual_sync(self) -> Dict:
        """Manually trigger GitHub sync"""
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{self.base_url}/server/sync")
            response.raise_for_status()
            return response.json()
    
    async def list_users(self) -> Dict:
        """List available users in the repository"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/server/users")
            response.raise_for_status()
            return response.json()
    
    async def health_check(self) -> Dict:
        """Check server health"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()


async def demo_workflow():
    """Demonstrate complete DiffMem server workflow"""
    
    # Configuration
    SERVER_URL = os.getenv("DIFFMEM_SERVER_URL", "http://localhost:8000")
    USER_ID = "alex"  # Your user ID (must exist in the server's repository)
    
    # Initialize client
    client = DiffMemClient(SERVER_URL)
    
    try:
        print("🏥 Checking server health...")
        health = await client.health_check()
        print(f"✅ Server healthy: {health['status']}")
        print(f"📁 Repository: {health.get('github_repo', 'Not configured')}")
        
        print("👥 Listing available users...")
        users = await client.list_users()
        print(f"✅ Found {users['count']} users: {users['users']}")
        
        if USER_ID not in users['users']:
            print(f"❌ User {USER_ID} not found in repository")
            return
        
        print(f"🔍 Validating setup for user {USER_ID}...")
        validation = await client.validate_setup(USER_ID)
        if not validation['validation']['valid']:
            print(f"❌ Setup issues: {validation['validation']['issues']}")
            return
        
        print("📊 Getting repository status...")
        status = await client.get_status(USER_ID)
        print(f"✅ Repository status: {status['repo_status']['memory_files_count']} memory files")
        
        # Example conversation
        conversation = [
            {"role": "user", "content": "How am I doing with my health goals?"},
            {"role": "assistant", "content": "Let me check your memories..."}
        ]
        
        print("🧠 Getting context...")
        context = await client.get_context(USER_ID, conversation, depth="basic")
        print(f"✅ Context loaded: {len(context['context']['always_load_blocks'])} blocks")
        
        print("🔍 Searching memories...")
        search_results = await client.search_memory(USER_ID, "health fitness", k=3)
        print(f"✅ Search complete: {len(search_results['results'])} results")
        
        # Process new memory
        new_memory = """
        Had a great workout today. Went for a 5-mile run and felt energetic.
        Also meal prepped for the week which should help with nutrition goals.
        Feeling optimistic about my health journey.
        """
        
        session_id = f"demo-session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        print("💾 Processing new memory session...")
        process_result = await client.process_and_commit_session(
            USER_ID, new_memory, session_id
        )
        print(f"✅ Memory processed: {process_result['message']}")
        
        print("🔄 Rebuilding index...")
        rebuild_result = await client.rebuild_index(USER_ID)
        print(f"✅ Index rebuilt: {rebuild_result['message']}")
        
        print("🔄 Manual sync...")
        sync_result = await client.manual_sync()
        print(f"✅ Sync complete: {sync_result['message']}")
        
        print("\n🎉 Demo workflow completed successfully!")
        
    except httpx.HTTPStatusError as e:
        print(f"❌ HTTP error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


async def simple_chat_agent_example():
    """Example of using DiffMem server in a chat agent"""
    
    SERVER_URL = os.getenv("DIFFMEM_SERVER_URL", "http://localhost:8000")
    USER_ID = os.getenv("USER_ID", "alex")
    
    client = DiffMemClient(SERVER_URL)
    
    # Verify user exists
    try:
        users = await client.list_users()
        if USER_ID not in users['users']:
            print(f"❌ User {USER_ID} not found. Available users: {users['users']}")
            return
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")
        return
    
    print(f"🤖 DiffMem Chat Agent (User: {USER_ID})")
    print("Type 'quit', 'exit', or 'bye' to end the session")
    
    # Simulate chat conversation
    conversation_history = []
    
    while True:
        user_input = input("\nUser: ").strip()
        if user_input.lower() in ['quit', 'exit', 'bye']:
            break
        
        conversation_history.append({"role": "user", "content": user_input})
        
        # Get relevant context
        try:
            context_response = await client.get_context(
                USER_ID,
                conversation_history[-5:],  # Last 5 messages
                depth="basic"
            )
            
            context = context_response["context"]
            print(f"\n🧠 Retrieved {len(context['always_load_blocks'])} memory blocks")
            
            # Simple response (in real implementation, use this context with your LLM)
            assistant_response = f"I found {len(context['always_load_blocks'])} relevant memories. [This would be processed by your LLM with the context]"
            
            print(f"Assistant: {assistant_response}")
            conversation_history.append({"role": "assistant", "content": assistant_response})
            
        except Exception as e:
            print(f"❌ Error retrieving context: {e}")
    
    # Save conversation to memory
    if conversation_history:
        session_id = f"chat-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        full_conversation = "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in conversation_history
        ])
        
        try:
            await client.process_and_commit_session(
                USER_ID, full_conversation, session_id
            )
            print(f"\n💾 Conversation saved to memory (session: {session_id})")
        except Exception as e:
            print(f"❌ Error saving conversation: {e}")


async def memory_explorer():
    """Interactive memory exploration tool"""
    
    SERVER_URL = os.getenv("DIFFMEM_SERVER_URL", "http://localhost:8000")
    USER_ID = os.getenv("USER_ID", "alex")
    
    client = DiffMemClient(SERVER_URL)
    
    print(f"🔍 DiffMem Memory Explorer (User: {USER_ID})")
    print("Commands: search <query>, timeline [days], entity, status, sync, quit")
    
    while True:
        command = input("\n> ").strip().lower()
        
        if command in ['quit', 'exit']:
            break
        
        try:
            if command.startswith('search '):
                query = command[7:]
                results = await client.search_memory(USER_ID, query, k=5)
                print(f"\n📋 Search Results for '{query}':")
                for i, result in enumerate(results['results'], 1):
                    print(f"{i}. Score: {result['score']:.3f}")
                    print(f"   {result['snippet']['content'][:100]}...")
                    print()
            
            elif command.startswith('timeline'):
                parts = command.split()
                days = int(parts[1]) if len(parts) > 1 else 30
                timeline = await client.get_recent_timeline(USER_ID, days)
                print(f"\n📅 Timeline (last {days} days):")
                for entry in timeline['timeline']:
                    print(f"- {entry.get('date', 'Unknown')}: {entry.get('content', 'No content')[:100]}...")
            
            elif command == 'entity':
                entity = await client.get_user_entity(USER_ID)
                print(f"\n👤 User Entity:")
                print(f"Name: {entity['entity'].get('entity_name', 'Unknown')}")
                print(f"Type: {entity['entity'].get('entity_type', 'Unknown')}")
            
            elif command == 'status':
                status = await client.get_status(USER_ID)
                repo_status = status['repo_status']
                print(f"\n📊 Repository Status:")
                print(f"Memory files: {repo_status['memory_files_count']}")
                print(f"Index blocks: {repo_status['index_stats']['total_blocks']}")
                print(f"Has timeline: {repo_status['has_timeline']}")
            
            elif command == 'sync':
                result = await client.manual_sync()
                print(f"✅ {result['message']}")
            
            else:
                print("Unknown command. Available: search <query>, timeline [days], entity, status, sync, quit")
        
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "chat":
            asyncio.run(simple_chat_agent_example())
        elif sys.argv[1] == "explore":
            asyncio.run(memory_explorer())
        else:
            print("Usage: python server_client.py [chat|explore]")
    else:
        asyncio.run(demo_workflow()) 