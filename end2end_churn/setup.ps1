# setup.ps1 - Windows PowerShell Setup Script for Churn Prediction
# This script provides the same functionality as setup.sh for Windows users

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Customer Churn Prediction - Automated Setup (Windows)" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This script will:" -ForegroundColor White
Write-Host "  1. Build Docker containers" -ForegroundColor White
Write-Host "  2. Train initial model (Random Forest)" -ForegroundColor White
Write-Host ""
Write-Host "Estimated time: 3-5 minutes" -ForegroundColor Yellow
Write-Host ""
Write-Host "Starting in 3 seconds... (Ctrl+C to cancel)" -ForegroundColor Yellow
Start-Sleep -Seconds 3
Write-Host ""

# Function to check if Docker is running
function Test-DockerRunning {
    try {
        docker ps | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# Check if Docker is running
Write-Host "Checking Docker..." -ForegroundColor Yellow
if (-not (Test-DockerRunning)) {
    Write-Host "ERROR: Docker is not running!" -ForegroundColor Red
    Write-Host "Please start Docker Desktop and try again." -ForegroundColor Red
    exit 1
}
Write-Host "Docker is running ✓" -ForegroundColor Green
Write-Host ""

# Create secrets directory if it doesn't exist (for optional authentication)
if (-not (Test-Path "secrets")) {
    Write-Host "Creating secrets directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path "secrets" | Out-Null
    # Create empty token file (authentication is optional)
    New-Item -ItemType File -Path "secrets\service_token.txt" | Out-Null
    Write-Host "✓ Secrets directory created" -ForegroundColor Green
    Write-Host ""
}

# Step 1: Build containers
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "📦 Step 1/2: Building Docker containers..." -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
docker compose build
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker build failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Containers built successfully" -ForegroundColor Green
Write-Host ""

# Step 2: Train initial model
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "🤖 Step 2/2: Training initial model (Random Forest)..." -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "⏳ This may take a few minutes..." -ForegroundColor Yellow
docker compose run --rm api python train.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Training failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Model trained and saved" -ForegroundColor Green
Write-Host ""

# Success message
Write-Host "================================================================================" -ForegroundColor Green
Write-Host "✓ SETUP COMPLETE!" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "The churn prediction service is ready to use!" -ForegroundColor White
Write-Host ""
Write-Host "Try these commands:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  # Start API service" -ForegroundColor Cyan
Write-Host "  docker compose up -d api" -ForegroundColor White
Write-Host ""
Write-Host "  # Test prediction endpoint" -ForegroundColor Cyan
Write-Host "  docker compose run --rm api pytest tests/integration/test_api.py::test_predict_endpoint -v" -ForegroundColor White
Write-Host ""
Write-Host "  # Or test with curl (if available)" -ForegroundColor Cyan
Write-Host "  curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '@test_request.json'" -ForegroundColor White
Write-Host ""
Write-Host "  # Train additional models" -ForegroundColor Cyan
Write-Host "  docker compose run --rm api python train.py --compare-all" -ForegroundColor White
Write-Host ""
Write-Host "  # View experiments in MLflow (in separate terminal)" -ForegroundColor Cyan
Write-Host "  docker compose run --rm -p 5000:5000 --entrypoint mlflow api ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000" -ForegroundColor White
Write-Host "  # Then open http://localhost:5000 in browser" -ForegroundColor White
Write-Host ""
Write-Host "To stop services:" -ForegroundColor Yellow
Write-Host "  docker compose down" -ForegroundColor White
Write-Host ""
Write-Host "Note:" -ForegroundColor Yellow
Write-Host "  For easier command usage, consider installing Make for Windows:" -ForegroundColor White
Write-Host "    choco install make" -ForegroundColor White
Write-Host "  Then you can use 'make up', 'make test-api', etc." -ForegroundColor White
Write-Host ""
Write-Host "================================================================================" -ForegroundColor Green

