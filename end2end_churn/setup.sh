#!/bin/bash
# setup.sh - Complete end2end churn setup
# One-command setup for portfolio demonstration

set -e  # Exit on any error

echo "================================================================================"
echo "Customer Churn Prediction - Automated Setup"
echo "================================================================================"
echo ""
echo "This script will:"
echo "  1. Build Docker containers"
echo "  2. Train initial model (Random Forest)"
echo ""
echo "Estimated time: 3-5 minutes"
echo ""
echo "Starting in 3 seconds... (Ctrl+C to cancel)"
sleep 3
echo ""

# Check if Docker is running
echo "Checking Docker..."
if ! docker ps > /dev/null 2>&1; then
    echo "ERROR: Docker is not running!"
    echo "Please start Docker Desktop and try again."
    exit 1
fi
echo "✅ Docker is running"
echo ""

# Step 1: Build containers
echo "================================================================================"
echo "📦 Step 1/2: Building Docker containers..."
echo "================================================================================"
docker compose build
echo "✅ Containers built successfully"
echo ""

# Step 2: Train initial model
echo "================================================================================"
echo "🤖 Step 2/2: Training initial model (Random Forest)..."
echo "================================================================================"
echo "⏳ This may take a few minutes..."
docker compose run --rm api python train.py
echo "✅ Model trained and saved"
echo ""

# Success message
echo "================================================================================"
echo "✅ SETUP COMPLETE!"
echo "================================================================================"
echo ""
echo "The churn prediction service is ready to use!"
echo ""
echo "Try these commands:"
echo ""
echo "  # Start API service"
echo "  make up"
echo ""
echo "  # Test prediction endpoint"
echo "  make test-api"
echo ""
echo "  # View API documentation"
echo "  make docs"
echo ""
echo "  # Train additional models"
echo "  make compare-models"
echo ""
echo "  # View experiments in MLflow"
echo "  make mlflow-ui"
echo ""
echo "To stop services:"
echo "  make down"
echo ""
echo "For help:"
echo "  make help"
echo ""
echo "================================================================================"

