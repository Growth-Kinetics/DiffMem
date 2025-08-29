#!/usr/bin/env python3
"""
DiffMem Onboarding Example

This example demonstrates how to onboard a new user to the DiffMem system
using both the API interface and the server endpoints.
"""

import os
import sys
import requests
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diffmem.api import onboard_new_user, create_memory_interface

# Configuration
REPO_PATH = "./example_memory_repo"
USER_ID = "john_doe"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SERVER_URL = "http://localhost:8000"  # Adjust if running on different host/port
API_KEY = os.getenv("API_KEY")  # Server API key if authentication is enabled

# Example user information dump
USER_INFO = """
John Doe is a 32-year-old software engineer living in San Francisco, CA. He works at TechCorp as a Senior Backend Developer, specializing in Python and distributed systems. He's been with the company for 3 years and enjoys the collaborative environment.

Personal life: John is married to Sarah, a graphic designer. They have a 2-year-old daughter named Emma. The family lives in a two-bedroom apartment in the Mission District. John is originally from Portland, Oregon, and his parents still live there.

Interests and hobbies: John is passionate about rock climbing and goes to the local climbing gym twice a week. He also enjoys reading science fiction novels, particularly works by Kim Stanley Robinson and Liu Cixin. On weekends, he likes to explore San Francisco's food scene with his family.

Current challenges: John has been dealing with work-life balance issues, especially since Emma was born. He's been considering a transition to a more senior role or possibly starting his own consulting business. He's also been learning Spanish to better connect with his neighbors in the Mission District.

Health: John has mild anxiety, which he manages through regular exercise and occasional therapy sessions with Dr. Martinez. He's generally healthy but has been working on improving his sleep schedule.

Goals: John wants to become a tech lead within the next year, improve his Spanish fluency, and take his family on a trip to visit Sarah's relatives in Italy next summer.
"""

def example_api_onboarding():
    """Example of onboarding using the direct API interface"""
    print("=== API Onboarding Example ===")
    
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY environment variable not set")
        return
    
    try:
        # Onboard the user
        print(f"Onboarding user: {USER_ID}")
        result = onboard_new_user(
            REPO_PATH,
            USER_ID,
            USER_INFO,
            OPENROUTER_API_KEY,
            session_id="onboard-example-001"
        )
        
        if result.get('success'):
            print("✅ Onboarding successful!")
            print(f"   Session ID: {result['session_id']}")
            print(f"   Entities created: {result['entities_created']}")
            print(f"   Files created: {len(result['files_created']['entities'])} entity files")
            print(f"   User file: {result['files_created']['user_file']}")
        else:
            print("❌ Onboarding failed!")
            print(f"   Error: {result.get('error')}")
            return
        
        # Now test the memory interface
        print("\n--- Testing memory interface ---")
        memory = create_memory_interface(REPO_PATH, USER_ID, OPENROUTER_API_KEY)
        
        # Check if onboarded
        print(f"Is onboarded: {memory.is_onboarded()}")
        
        # Get repo status
        status = memory.get_repo_status()
        print(f"Memory files: {status['memory_files_count']}")
        print(f"Has timeline: {status['has_timeline']}")
        print(f"Has master index: {status['has_master_index']}")
        
        # Test search
        print("\n--- Testing search ---")
        results = memory.search("software engineer", k=3)
        for i, result in enumerate(results):
            print(f"Result {i+1}: {result['snippet'][:100]}...")
        
    except Exception as e:
        print(f"❌ Error during API onboarding: {e}")

def example_server_onboarding():
    """Example of onboarding using the server API"""
    print("\n=== Server API Onboarding Example ===")
    
    headers = {}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    
    try:
        # Check onboard status first
        print(f"Checking onboard status for user: {USER_ID}")
        response = requests.get(
            f"{SERVER_URL}/memory/{USER_ID}/onboard-status",
            headers=headers
        )
        
        if response.status_code == 200:
            status_data = response.json()
            print(f"   Onboarded: {status_data['onboarded']}")
            
            if status_data['onboarded']:
                print("   User is already onboarded, skipping onboarding step")
                return
        else:
            print(f"   Status check failed: {response.status_code}")
        
        # Onboard the user
        print(f"Onboarding user via server API: {USER_ID}")
        onboard_data = {
            "user_info": USER_INFO,
            "session_id": "server-onboard-001"
        }
        
        response = requests.post(
            f"{SERVER_URL}/memory/{USER_ID}/onboard",
            headers=headers,
            json=onboard_data
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Server onboarding successful!")
            print(f"   Message: {result['message']}")
            if 'result' in result:
                onboard_result = result['result']
                print(f"   Entities created: {onboard_result.get('entities_created', 0)}")
        else:
            print("❌ Server onboarding failed!")
            print(f"   Status code: {response.status_code}")
            print(f"   Response: {response.text}")
            return
        
        # Test memory operations via server
        print("\n--- Testing server memory operations ---")
        
        # Get user entity
        response = requests.get(
            f"{SERVER_URL}/memory/{USER_ID}/user-entity",
            headers=headers
        )
        
        if response.status_code == 200:
            entity_data = response.json()
            print("✅ Retrieved user entity")
        else:
            print(f"❌ Failed to retrieve user entity: {response.status_code}")
        
        # Test search
        search_data = {
            "query": "software engineer",
            "k": 3
        }
        
        response = requests.post(
            f"{SERVER_URL}/memory/{USER_ID}/search",
            headers=headers,
            json=search_data
        )
        
        if response.status_code == 200:
            search_results = response.json()
            print(f"✅ Search returned {len(search_results['results'])} results")
            for i, result in enumerate(search_results['results']):
                print(f"   Result {i+1}: {result['snippet'][:100]}...")
        else:
            print(f"❌ Search failed: {response.status_code}")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure the DiffMem server is running.")
        print("   Start the server with: python -m diffmem.server")
    except Exception as e:
        print(f"❌ Error during server onboarding: {e}")

def main():
    """Run the onboarding examples"""
    print("DiffMem Onboarding Examples")
    print("=" * 50)
    
    # Create repo directory if it doesn't exist
    os.makedirs(REPO_PATH, exist_ok=True)
    
    # Initialize git repo if needed
    if not (Path(REPO_PATH) / ".git").exists():
        import git
        git.Repo.init(REPO_PATH)
        print(f"Initialized git repository at {REPO_PATH}")
    
    # Run API example
    example_api_onboarding()
    
    # Run server example
    example_server_onboarding()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print(f"Check the repository at {REPO_PATH} to see the created files.")

if __name__ == "__main__":
    main() 