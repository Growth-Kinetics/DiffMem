# Unraid Docker Deployment Guide

## 🚀 Quick Deployment

### Step 1: Prepare Your Unraid Server

1. **SSH into your Unraid server** (or use terminal)
2. **Create a directory** for DiffMem:
   ```bash
   mkdir -p /mnt/user/appdata/diffmem
   cd /mnt/user/appdata/diffmem
   ```

### Step 2: Clone or Copy the Repository

**Option A: Clone from GitHub** (recommended)
```bash
git clone https://github.com/kingfisherfox/diffmem.git .
```

**Option B: Copy files manually**
- Copy all files from your local DiffMem directory to `/mnt/user/appdata/diffmem`

### Step 3: Configure Environment Variables

Create `.env` file in the diffmem directory:

```bash
cd /mnt/user/appdata/diffmem
nano .env
```

Paste this configuration (update with your actual values):

```bash
# Required: OpenRouter API key
OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here

# Required: GitHub repository
GITHUB_REPO_URL=https://github.com/kingfisherfox/diffmem.git
GITHUB_TOKEN=your-github-token-here
GITHUB_USERNAME=kingfisherfox

# API Security (recommended)
API_KEY=your-secure-random-key-here
REQUIRE_AUTH=true

# CORS Configuration
ALLOWED_ORIGINS=https://difmem.kingbarry.cc,http://192.168.60.108:8000

# Server URL
DIFFMEM_SERVER_URL=https://difmem.kingbarry.cc
```

Save and exit (Ctrl+X, Y, Enter)

### Step 4: Deploy with Docker Compose

```bash
cd /mnt/user/appdata/diffmem
docker-compose up -d
```

### Step 5: Verify It's Running

```bash
# Check logs
docker-compose logs -f diffmem

# Check health
curl http://192.168.60.108:8000/health
```

## 🌐 Setting Up Reverse Proxy (Nginx Proxy Manager)

If you're using Nginx Proxy Manager on Unraid:

1. **Add Proxy Host**:
   - Domain: `difmem.kingbarry.cc`
   - Forward IP: `192.168.60.108`
   - Forward Port: `8000`
   - Scheme: `http`

2. **SSL Certificate**: Enable SSL (Let's Encrypt recommended)

3. **WebSocket Support**: Enable if needed

## 📝 Using the API

### Base URL
- **Local**: `http://192.168.60.108:8000`
- **Domain**: `https://difmem.kingbarry.cc`

### Create a New User

```bash
curl -X POST "https://difmem.kingbarry.cc/memory/jon-morgenstern/onboard" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "user_info": "Jon Morgenstern is a paid media expert...",
    "session_id": "onboard-001"
  }'
```

### Import Content

```bash
curl -X POST "https://difmem.kingbarry.cc/memory/jon-morgenstern/process-and-commit" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "memory_input": "Jon worked on optimizing Facebook ads today...",
    "session_id": "session-001",
    "session_date": "2025-01-31"
  }'
```

### Search Memories

```bash
curl -X POST "https://difmem.kingbarry.cc/memory/jon-morgenstern/search" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "query": "Facebook campaign",
    "k": 5
  }'
```

### Get Context

```bash
curl -X POST "https://difmem.kingbarry.cc/memory/jon-morgenstern/context" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "conversation": [{"role": "user", "content": "What did Jon work on?"}],
    "depth": "deep"
  }'
```

## 🔒 Security Notes

1. **API Key**: Set `REQUIRE_AUTH=true` and use a strong `API_KEY`
2. **HTTPS**: Use your domain with SSL (`https://difmem.kingbarry.cc`)
3. **Firewall**: Only expose port 8000 if needed (use reverse proxy instead)

## 📂 Volume Persistence

Data is stored in:
- `/mnt/user/appdata/diffmem/storage` - Git repository
- `/mnt/user/appdata/diffmem/worktrees` - User worktrees

These persist across container restarts.

## 🔄 Updating

```bash
cd /mnt/user/appdata/diffmem
git pull  # If using git
docker-compose down
docker-compose build
docker-compose up -d
```

## 🐛 Troubleshooting

### Check Logs
```bash
docker-compose logs -f diffmem
```

### Restart Container
```bash
docker-compose restart diffmem
```

### Check Health
```bash
curl http://192.168.60.108:8000/health
```

### Verify Environment
```bash
docker-compose exec diffmem env | grep -E "GITHUB|OPENROUTER|API_KEY"
```

## 📊 Monitoring

### Health Endpoint
```bash
curl https://difmem.kingbarry.cc/health
```

### List Users
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  https://difmem.kingbarry.cc/server/users
```

## 🎯 Next Steps

1. ✅ Deploy to Unraid
2. ✅ Set up reverse proxy
3. ✅ Test API endpoints
4. ✅ Start creating users and importing content!
