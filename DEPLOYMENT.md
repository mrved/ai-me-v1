# Deployment Guide - Hosting Options

## 🚀 Quick Deployment Options

### Option 1: Streamlit Cloud (Easiest - Recommended) ⭐

**Best for**: Quick deployment, free tier available, zero configuration

#### Steps:
1. **Push code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/ai-me-v1.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `src/dashboard.py`
   - Click "Deploy"

3. **Configure secrets** (if needed)
   - Database credentials
   - API keys
   - Add in Streamlit Cloud settings

**Pros:**
- ✅ Free tier available
- ✅ Automatic HTTPS
- ✅ Auto-deploy on git push
- ✅ Zero server management

**Cons:**
- ⚠️ Limited to Streamlit apps
- ⚠️ Free tier has usage limits

---

### Option 2: Docker + Cloud Platform

**Best for**: Full control, scalable, production-ready

#### 2a. AWS EC2 / Google Cloud / Azure VM

**Steps:**
1. **Create Dockerfile** (see below)
2. **Deploy to cloud VM**
   ```bash
   # On your VM
   git clone https://github.com/yourusername/ai-me-v1.git
   cd ai-me-v1
   docker build -t ai-engineering-dashboard .
   docker run -d -p 8501:8501 ai-engineering-dashboard
   ```

3. **Set up reverse proxy** (nginx)
4. **Configure domain** and SSL

#### 2b. AWS ECS / Google Cloud Run / Azure Container Instances

**Best for**: Serverless containers, auto-scaling

**Steps:**
1. Build and push Docker image
2. Deploy to container service
3. Configure auto-scaling
4. Set up load balancer

---

### Option 3: Heroku

**Best for**: Simple PaaS deployment

#### Steps:
1. **Install Heroku CLI**
   ```bash
   # macOS
   brew install heroku/brew/heroku
   
   # Or download from heroku.com
   ```

2. **Create Procfile**
   ```
   web: streamlit run src/dashboard.py --server.port=$PORT --server.address=0.0.0.0
   ```

3. **Deploy**
   ```bash
   heroku login
   heroku create ai-engineering-dashboard
   git push heroku main
   ```

4. **Set up database** (if using external DB)
   ```bash
   heroku addons:create heroku-postgresql:hobby-dev
   ```

**Pros:**
- ✅ Easy deployment
- ✅ Free tier available
- ✅ Automatic SSL

**Cons:**
- ⚠️ Free tier sleeps after inactivity
- ⚠️ Limited resources on free tier

---

### Option 4: Railway / Render

**Best for**: Modern PaaS, good free tier

#### Railway:
1. Go to https://railway.app
2. Connect GitHub repo
3. Set start command: `streamlit run src/dashboard.py --server.port=$PORT`
4. Deploy automatically

#### Render:
1. Go to https://render.com
2. Create new Web Service
3. Connect GitHub repo
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `streamlit run src/dashboard.py --server.port=$PORT --server.address=0.0.0.0`

---

### Option 5: Self-Hosted (On-Premise)

**Best for**: Sach Engg internal deployment, data security

#### Steps:
1. **Set up server** (Linux/Windows)
   ```bash
   # Install Python 3.9+
   sudo apt update
   sudo apt install python3.9 python3-pip
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Run as service**
   ```bash
   # Create systemd service (see below)
   sudo systemctl enable ai-dashboard
   sudo systemctl start ai-dashboard
   ```

3. **Set up nginx reverse proxy**
4. **Configure firewall**
5. **Set up SSL** (Let's Encrypt)

---

## 📦 Docker Deployment (Recommended for Production)

### Create Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run application
CMD ["streamlit", "run", "src/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
```

### Docker Compose (with database)

```yaml
version: '3.8'

services:
  dashboard:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./model.pkl:/app/model.pkl
    environment:
      - DB_PATH=sqlite:///data/metadata.db
    restart: unless-stopped

  # Optional: PostgreSQL for production
  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: engineering_db
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  postgres_data:
```

### Build and Run

```bash
# Build image
docker build -t ai-engineering-dashboard .

# Run container
docker run -d \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/model.pkl:/app/model.pkl \
  --name ai-dashboard \
  ai-engineering-dashboard

# Or use docker-compose
docker-compose up -d
```

---

## 🔧 Systemd Service (Linux)

Create `/etc/systemd/system/ai-dashboard.service`:

```ini
[Unit]
Description=AI Engineering Dashboard
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/ai-me-v1
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
ExecStart=/usr/local/bin/streamlit run src/dashboard.py --server.port=8501 --server.address=0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable ai-dashboard
sudo systemctl start ai-dashboard
sudo systemctl status ai-dashboard
```

---

## 🌐 Nginx Reverse Proxy

Create `/etc/nginx/sites-available/ai-dashboard`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
```

Enable:
```bash
sudo ln -s /etc/nginx/sites-available/ai-dashboard /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 🔒 SSL Setup (Let's Encrypt)

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal (already set up by certbot)
```

---

## 📊 Environment Variables

Create `.env` file for configuration:

```bash
# Database
DB_PATH=sqlite:///data/metadata.db
# Or for PostgreSQL:
# DB_PATH=postgresql://user:pass@host:5432/dbname

# Model
MODEL_PATH=model.pkl

# Streamlit config
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
```

Load in your app:
```python
import os
from dotenv import load_dotenv
load_dotenv()
```

---

## 🚀 Quick Deploy Scripts

### Deploy to Streamlit Cloud
```bash
# Just push to GitHub, then deploy via web UI
git push origin main
```

### Deploy to Docker
```bash
./deploy.sh  # See deploy.sh below
```

### Deploy to Heroku
```bash
heroku create ai-dashboard
git push heroku main
```

---

## 📝 Deployment Checklist

- [ ] Code pushed to Git repository
- [ ] Dependencies listed in requirements.txt
- [ ] Environment variables configured
- [ ] Database/data files accessible
- [ ] Model file included
- [ ] Port configured correctly
- [ ] Health checks working
- [ ] SSL certificate (for production)
- [ ] Monitoring/logging set up
- [ ] Backup strategy in place

---

## 🆘 Troubleshooting

### App won't start
- Check port is available: `lsof -i :8501`
- Check logs: `docker logs ai-dashboard` or `journalctl -u ai-dashboard`
- Verify dependencies: `pip install -r requirements.txt`

### Database connection issues
- Check DB_PATH environment variable
- Verify database file exists
- Check file permissions

### Performance issues
- Increase server resources
- Enable caching
- Use production database (PostgreSQL)

---

## 💡 Recommended for Sach Engg

**Option 1: Internal Server (On-Premise)**
- Best for data security
- Full control
- No external dependencies

**Option 2: Private Cloud (AWS/Azure)**
- Scalable
- Professional infrastructure
- Good for growth

**Option 3: Streamlit Cloud (Quick Start)**
- Fastest deployment
- Good for pilot/testing
- Can migrate later

---

**Need help? Check the deployment scripts in the repository!**

