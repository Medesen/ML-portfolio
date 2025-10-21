#!/bin/bash
# setup.sh - Complete RAG pipeline setup
# One-command setup for portfolio demonstration

set -e  # Exit on any error

echo "================================================================================"
echo "RAG Pipeline - Automated Setup"
echo "================================================================================"
echo ""
echo "This script will:"
echo "  1. Build Docker containers"
echo "  2. Start Ollama LLM service"
echo "  3. Download Llama 3.2 model (~2GB)"
echo "  4. Preprocess 416 documents"
echo "  5. Build vector index (3 strategies, 4006 chunks)"
echo ""
echo "Estimated time: 5-10 minutes (depending on internet speed)"
echo ""
echo "Starting in 3 seconds... (Ctrl+C to cancel)"
sleep 3
echo ""

# Step 1: Build containers
echo "================================================================================"
echo "📦 Step 1/5: Building Docker containers..."
echo "================================================================================"
docker compose build
echo "✅ Containers built successfully"
echo ""

# Step 2: Start Ollama service
echo "================================================================================"
echo "🚀 Step 2/5: Starting Ollama LLM service..."
echo "================================================================================"
docker compose up -d ollama
echo "⏳ Waiting for Ollama to be ready (10 seconds)..."
sleep 10
echo "✅ Ollama service running"
echo ""

# Step 3: Pull LLM model
echo "================================================================================"
echo "📥 Step 3/5: Downloading LLM model (llama3.2:3b, ~2GB)..."
echo "================================================================================"
echo "⏳ This may take a few minutes on first run..."
docker compose exec ollama ollama pull llama3.2:3b
echo "✅ Model downloaded and cached"
echo ""

# Step 4: Run preprocessing
echo "================================================================================"
echo "⚙️  Step 4/5: Preprocessing corpus (416 documents)..."
echo "================================================================================"
docker compose run --rm rag-pipeline preprocess
echo "✅ Preprocessing complete"
echo ""

# Step 5: Build vector index
echo "================================================================================"
echo "🔍 Step 5/5: Building vector index (3 chunking strategies)..."
echo "================================================================================"
docker compose run --rm rag-pipeline index
echo "✅ Vector index built"
echo ""

# Success message
echo "================================================================================"
echo "✅ SETUP COMPLETE!"
echo "================================================================================"
echo ""
echo "The RAG pipeline is ready to use!"
echo ""
echo "Try these commands:"
echo ""
echo "  # Query with retrieval only"
echo "  docker compose run --rm rag-pipeline query \"How do I use StandardScaler?\""
echo ""
echo "  # Query with answer generation (LLM)"
echo "  docker compose run --rm rag-pipeline query \"How do I use StandardScaler?\" --generate"
echo ""
echo "  # More examples"
echo "  docker compose run --rm rag-pipeline query \"What is PCA?\" --generate"
echo "  docker compose run --rm rag-pipeline query \"How to handle missing values?\" --generate"
echo ""
echo "To stop services:"
echo "  docker compose down"
echo ""
echo "To clean everything:"
echo "  docker compose down -v"
echo "  ./scripts/clean.sh"
echo ""
echo "================================================================================"

