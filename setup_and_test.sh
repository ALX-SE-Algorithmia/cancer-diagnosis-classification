#!/bin/bash
# Setup script for Cancer Diagnosis DVC-MLflow integration

# Create conda environment
echo "Creating conda environment..."
conda create -p venv python=3.12 -y

# Activate conda environment
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./venv

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Initialize DVC if not already initialized
if [ ! -d ".dvc" ]; then
    echo "Initializing DVC..."
    dvc init
    git add .dvc
    git commit -m "Initialize DVC" || echo "Git commit failed, but continuing..."
fi

# Create necessary directories
mkdir -p data/raw data/processed models/checkpoints

# Run tests
echo "Running tests..."
pytest

echo "Setup and tests completed!"
echo "Next steps:"
echo "1. Create a repository on DAGsHub"
echo "2. Create a .env file from .env.example with your DAGsHub credentials"
echo "3. Run 'python run_demo.py --create-data' to create dummy data and run the pipeline"
