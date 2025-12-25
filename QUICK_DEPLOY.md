# 🚀 Quick Deployment Guide

## Fastest Options (5 minutes)

### Option 1: Streamlit Cloud (Easiest) ⭐

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/ai-me-v1.git
   git push -u origin main
   ```

2. **Deploy:**
   - Go to https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app"
   - Repository: `YOUR_USERNAME/ai-me-v1`
   - Main file: `src/dashboard.py`
   - Click "Deploy"

**Done!** Your app is live at `https://YOUR_APP.streamlit.app`

---

### Option 2: Docker (Local/Server)

```bash
# Build and run
docker-compose up -d

# Or use the deploy script
./deploy.sh docker-compose
```

**Access at:** http://localhost:8501

---

### Option 3: Railway (Free Tier)

1. Go to https://railway.app
2. Click "New Project" → "Deploy from GitHub"
3. Select your repository
4. Railway auto-detects and deploys

**Done!** Railway provides a URL automatically

---

### Option 4: Render (Free Tier)

1. Go to https://render.com
2. Click "New" → "Web Service"
3. Connect GitHub repository
4. Settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run src/dashboard.py --server.port=$PORT --server.address=0.0.0.0`
5. Click "Create Web Service"

**Done!** Render provides a URL automatically

---

## For Sach Engg Internal Deployment

### Quick Docker Deployment

```bash
# On your server
git clone https://github.com/YOUR_REPO/ai-me-v1.git
cd ai-me-v1
./deploy.sh docker-compose
```

### With Custom Domain

1. Deploy with Docker
2. Set up nginx (see DEPLOYMENT.md)
3. Configure SSL with Let's Encrypt
4. Point domain to your server

---

## Need More Details?

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for:
- Detailed instructions for all platforms
- Production setup
- Security configuration
- Troubleshooting

