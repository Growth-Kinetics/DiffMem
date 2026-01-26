#!/bin/bash
# Example script to import content into DiffMem

BASE_URL="http://localhost:8000"
USER_ID="benjamin"

echo "=== DiffMem Import Example ==="
echo ""

# Example 1: Import a conversation
echo "1. Importing conversation about Sarah..."
curl -X POST "$BASE_URL/memory/$USER_ID/process-and-commit" \
  -H "Content-Type: application/json" \
  -d '{
    "memory_input": "I had coffee with Sarah today. She mentioned she is starting a new job at Google next month. We talked about her upcoming trip to Japan in March. She is really excited about it.",
    "session_id": "chat-2025-01-31-001",
    "session_date": "2025-01-31"
  }' | jq '.'

echo ""
echo "2. Importing work notes..."
curl -X POST "$BASE_URL/memory/$USER_ID/process-and-commit" \
  -H "Content-Type: application/json" \
  -d '{
    "memory_input": "Today I worked on the DiffMem project. Learned about Git-based memory systems. It is interesting how Git can be used for versioning memories. Made good progress on understanding the architecture.",
    "session_id": "work-2025-01-31",
    "session_date": "2025-01-31"
  }' | jq '.'

echo ""
echo "3. Checking status..."
curl -X GET "$BASE_URL/memory/$USER_ID/status" | jq '.'

echo ""
echo "=== Done! Check GitHub repo for user/benjamin branch ==="
