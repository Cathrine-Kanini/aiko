#!/bin/bash

echo "🚀 Starting AikoLearn API..."

# Download NLTK data if needed (uncomment if your app uses NLTK)
# echo "📦 Downloading NLTK data..."
# python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

# Create necessary directories
mkdir -p /tmp/chroma_db

echo "✅ Setup complete. Starting server..."

# Start the FastAPI application
exec uvicorn main:app --host 0.0.0.0 --port $PORT