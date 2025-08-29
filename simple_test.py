#!/usr/bin/env python3
"""
Super simple test for OnboardingAgent API key
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_simple():
    """Simple test of API key passing"""
    
    # Get API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    print(f"Environment API key: {api_key[:10]}..." if api_key else "None")
    
    if not api_key:
        print("❌ No API key in environment")
        return
    
    # Test direct import and creation
    try:
        from diffmem.writer_agent.onboarding_agent import OnboardingAgent
        import tempfile
        import git
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            git.Repo.init(tmp_dir)
            
            print("Creating OnboardingAgent...")
            agent = OnboardingAgent(tmp_dir, "test_user", api_key)
            
            print(f"✅ Agent created successfully")
            print(f"   Agent.openrouter_api_key: {agent.openrouter_api_key[:10]}..." if hasattr(agent, 'openrouter_api_key') and agent.openrouter_api_key else "❌ Missing!")
            print(f"   Agent.client.api_key: {agent.client.api_key[:10]}..." if agent.client.api_key else "❌ Missing!")
            
            # Test a simple method call (without LLM)
            agent._create_directory_structure()
            print(f"✅ Directory structure created")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple() 