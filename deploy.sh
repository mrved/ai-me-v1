#!/bin/bash

# Deployment script for AI Engineering Dashboard
# Usage: ./deploy.sh [docker|streamlit|heroku]

set -e

DEPLOY_TYPE=${1:-docker}

echo "🚀 Deploying AI Engineering Dashboard..."
echo "Deployment type: $DEPLOY_TYPE"

case $DEPLOY_TYPE in
  docker)
    echo "📦 Building Docker image..."
    docker build -t ai-engineering-dashboard:latest .
    
    echo "🛑 Stopping existing container..."
    docker stop ai-engineering-dashboard 2>/dev/null || true
    docker rm ai-engineering-dashboard 2>/dev/null || true
    
    echo "▶️  Starting new container..."
    docker run -d \
      --name ai-engineering-dashboard \
      -p 8501:8501 \
      -v $(pwd)/data:/app/data \
      -v $(pwd)/model.pkl:/app/model.pkl \
      --restart unless-stopped \
      ai-engineering-dashboard:latest
    
    echo "✅ Deployment complete!"
    echo "🌐 Dashboard available at: http://localhost:8501"
    echo ""
    echo "View logs: docker logs -f ai-engineering-dashboard"
    ;;
    
  docker-compose)
    echo "📦 Deploying with Docker Compose..."
    docker-compose down
    docker-compose up -d --build
    
    echo "✅ Deployment complete!"
    echo "🌐 Dashboard available at: http://localhost:8501"
    echo ""
    echo "View logs: docker-compose logs -f"
    ;;
    
  streamlit)
    echo "☁️  Preparing for Streamlit Cloud deployment..."
    echo ""
    echo "Steps:"
    echo "1. Push code to GitHub:"
    echo "   git add ."
    echo "   git commit -m 'Deploy to Streamlit Cloud'"
    echo "   git push origin main"
    echo ""
    echo "2. Go to https://share.streamlit.io"
    echo "3. Click 'New app'"
    echo "4. Select repository and set main file: src/dashboard.py"
    echo "5. Click 'Deploy'"
    ;;
    
  heroku)
    echo "☁️  Deploying to Heroku..."
    
    if ! command -v heroku &> /dev/null; then
      echo "❌ Heroku CLI not found. Install from https://devcenter.heroku.com/articles/heroku-cli"
      exit 1
    fi
    
    # Check if Heroku app exists
    if ! heroku apps:info &> /dev/null; then
      echo "📱 Creating Heroku app..."
      heroku create ai-engineering-dashboard
    fi
    
    echo "📤 Pushing to Heroku..."
    git push heroku main
    
    echo "✅ Deployment complete!"
    echo "🌐 Dashboard available at: https://ai-engineering-dashboard.herokuapp.com"
    ;;
    
  *)
    echo "❌ Unknown deployment type: $DEPLOY_TYPE"
    echo ""
    echo "Usage: ./deploy.sh [docker|docker-compose|streamlit|heroku]"
    exit 1
    ;;
esac

