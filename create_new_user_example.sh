#!/bin/bash
# Example: Creating a new user

BASE_URL="http://localhost:8000"
NEW_USER_ID="alice"  # Change this to any user ID you want

echo "=== Creating New User: $NEW_USER_ID ==="
echo ""

# Step 1: Onboard the new user
echo "1. Onboarding $NEW_USER_ID..."
curl -X POST "$BASE_URL/memory/$NEW_USER_ID/onboard" \
  -H "Content-Type: application/json" \
  -d "{
    \"user_info\": \"Alice is a software engineer who works on frontend development. She loves React and TypeScript.\",
    \"session_id\": \"onboard-$NEW_USER_ID-001\"
  }" | jq '.'

echo ""
echo "2. Importing first memory for $NEW_USER_ID..."
curl -X POST "$BASE_URL/memory/$NEW_USER_ID/process-and-commit" \
  -H "Content-Type: application/json" \
  -d "{
    \"memory_input\": \"Alice worked on a React component today. She fixed a bug in the user authentication flow.\",
    \"session_id\": \"$NEW_USER_ID-session-001\",
    \"session_date\": \"2025-01-31\"
  }" | jq '.'

echo ""
echo "3. Checking $NEW_USER_ID status..."
curl -X GET "$BASE_URL/memory/$NEW_USER_ID/status" | jq '.'

echo ""
echo "=== Done! User $NEW_USER_ID created and isolated ==="
echo "Check GitHub: https://github.com/kingfisherfox/diffmem/tree/user/$NEW_USER_ID"
