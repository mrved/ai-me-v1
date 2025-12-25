#!/bin/bash
# Setup script for Streamlit Cloud - runs before app starts

echo "🚀 Setting up AI Engineering Dashboard..."

# Create necessary directories
mkdir -p data/raw

# Run setup script
python setup_streamlit.py

echo "✅ Setup complete!"

