#!/usr/bin/env python3
"""
Minimal debug script for onboarding API key issue
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from diffmem.writer_agent.onboarding_agent import OnboardingAgent
from diffmem.api import onboard_new_user, DiffMemory
import tempfile
import git

def test_api_key_passing():
    """Test that API key makes it through the chain"""
    
    # Get API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    print(f"Environment API key: {api_key[:10]}..." if api_key else "None")
    
    if not api_key:
        print("❌ No API key in environment")
        return
    
    # Test 1: Direct OnboardingAgent creation
    print("\n=== Test 1: Direct OnboardingAgent ===")
    with tempfile.TemporaryDirectory() as tmp_dir:
        git.Repo.init(tmp_dir)
        
        agent = OnboardingAgent(tmp_dir, "test_user", api_key)
        print(f"Agent openrouter_api_key: {agent.openrouter_api_key[:10]}..." if hasattr(agent, 'openrouter_api_key') else "Missing!")
        print(f"Agent client.api_key: {agent.client.api_key[:10]}..." if agent.client.api_key else "Missing!")
    
    # Test 2: Through DiffMemory
    print("\n=== Test 2: Through DiffMemory ===")
    with tempfile.TemporaryDirectory() as tmp_dir:
        git.Repo.init(tmp_dir)
        
        memory = DiffMemory(tmp_dir, "test_user", api_key, auto_onboard=True)
        print(f"Memory openrouter_api_key: {memory.openrouter_api_key[:10]}..." if memory.openrouter_api_key else "Missing!")
        
        # Try to create OnboardingAgent through memory
        try:
            from diffmem.writer_agent.onboarding_agent import OnboardingAgent
            onboarding_agent = OnboardingAgent(
                str(memory.repo_path),
                memory.user_id,
                memory.openrouter_api_key,  # This should be the issue
                memory.model
            )
            print(f"OnboardingAgent from memory - api_key: {onboarding_agent.client.api_key[:10]}..." if onboarding_agent.client.api_key else "Missing!")
        except Exception as e:
            print(f"❌ Failed to create OnboardingAgent: {e}")
    
    # Test 3: Through onboard_new_user
    print("\n=== Test 3: Through onboard_new_user ===")
    with tempfile.TemporaryDirectory() as tmp_dir:
        git.Repo.init(tmp_dir)
        
        print(f"Calling onboard_new_user with api_key: {api_key[:10]}...")
        
        # This should fail at the LLM call, but we want to see if the key makes it there
        try:
            result = onboard_new_user(
                tmp_dir,
                "test_user", 
                "Test user info",
                api_key,
                session_id="debug-test"
            )
            print(f"Result: {result}")
        except Exception as e:
            print(f"❌ onboard_new_user failed: {e}")

if __name__ == "__main__":
    test_api_key_passing() 