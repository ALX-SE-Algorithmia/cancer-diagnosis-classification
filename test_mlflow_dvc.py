#!/usr/bin/env python
"""
Test script for MLflow and DVC integration with DAGsHub.
This script:
1. Creates dummy data
2. Tracks it with DVC
3. Runs a simple ML experiment with MLflow tracking
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import random
import json

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def run_command(cmd):
    """Run a shell command and print output."""
    print(f"Running: {cmd}")
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    for line in process.stdout:
        print(line.strip())
    process.wait()
    return process.returncode


def create_dummy_data(data_dir="data/raw", num_samples=100, num_classes=2):
    """Create dummy medical imaging data for cancer diagnosis."""
    print(f"Creating dummy data in {data_dir}...")
    data_dir = Path(data_dir)
    
    # Create directories for each class
    for class_idx in range(num_classes):
        class_dir = data_dir / f"cancer_type_{class_idx}"
        class_dir.mkdir(exist_ok=True, parents=True)
        
        # Create dummy patient directories
        for patient_idx in range(10):
            patient_dir = class_dir / f"patient_{patient_idx}"
            patient_dir.mkdir(exist_ok=True)
            
            # Create dummy images
            num_patient_samples = num_samples // (num_classes * 10)
            for i in range(num_patient_samples):
                # Create a random 3D image (1 channel, 64x64x64)
                image = np.random.rand(1, 64, 64, 64).astype(np.float32)
                image_path = patient_dir / f"scan_{i}.npy"
                np.save(image_path, image)
    
    print(f"Created {num_samples} dummy samples across {num_classes} classes")
    return data_dir


def track_data_with_dvc(data_dir="data/raw"):
    """Track data with DVC."""
    print("Tracking data with DVC...")
    run_command(f"dvc add {data_dir}")
    run_command("git add data/raw.dvc .gitignore")
    run_command("git commit -m 'Add dummy data tracked by DVC'")
    
    # Push to DAGsHub (optional - uncomment if you want to push)
    # run_command("dvc push")


def setup_mlflow():
    """Set up MLflow tracking."""
    load_dotenv()
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(mlflow_uri)
    
    # Set authentication if provided
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    
    if username and password:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password
    
    print(f"MLflow tracking URI: {mlflow_uri}")


class SimpleModel(nn.Module):
    """Simple CNN model for testing."""
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def run_test_experiment():
    """Run a test ML experiment with MLflow tracking."""
    print("Running test experiment with MLflow tracking...")
    
    # Set up MLflow
    setup_mlflow()
    
    # Create a simple model
    model = SimpleModel(in_channels=1, num_classes=2)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create dummy data for training
    X = torch.randn(10, 1, 64, 64, 64)  # 10 samples, 1 channel, 64x64x64 voxels
    y = torch.randint(0, 2, (10,))  # Binary labels
    
    # Start MLflow run
    with mlflow.start_run(run_name="test_experiment") as run:
        print(f"MLflow run ID: {run.info.run_id}")
        
        # Log parameters
        mlflow.log_param("model_type", "SimpleModel")
        mlflow.log_param("in_channels", 1)
        mlflow.log_param("num_classes", 2)
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("optimizer", "Adam")
        
        # Train for a few epochs
        for epoch in range(5):
            # Forward pass
            outputs = model(X)
            loss = criterion(outputs, y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y).sum().item() / y.size(0)
            
            # Log metrics
            mlflow.log_metrics({
                "loss": loss.item(),
                "accuracy": accuracy
            }, step=epoch)
            
            print(f"Epoch {epoch+1}/5, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
        
        # Log model
        mlflow.pytorch.log_model(model, "model")
        
        # Save metrics for DVC
        metrics = {
            "final_loss": loss.item(),
            "final_accuracy": accuracy
        }
        
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Experiment completed. Metrics saved to metrics.json")
        print(f"View this run on DAGsHub: {os.getenv('MLFLOW_TRACKING_URI').replace('.mlflow', '')}/experiments")


def main():
    """Main function."""
    # Create dummy data
    create_dummy_data()
    
    # Track data with DVC
    track_data_with_dvc()
    
    # Run test experiment
    run_test_experiment()
    
    print("Integration test completed successfully!")


if __name__ == "__main__":
    main()
