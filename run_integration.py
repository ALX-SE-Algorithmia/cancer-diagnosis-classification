#!/usr/bin/env python
"""
Script to run the DVC, MLflow, and DAGsHub integration for the Cancer Diagnosis Classification project.
"""
import os
import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
import mlflow
import dagshub
from dotenv import load_dotenv


def run_command(command, cwd=None):
    """Run a shell command and print output."""
    print(f"Running: {command}")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        cwd=cwd
    )
    
    for line in process.stdout:
        print(line.strip())
    
    process.wait()
    return process.returncode


def create_dummy_data(data_dir, num_patients=10, num_classes=2):
    """Create dummy data for demonstration purposes."""
    print("Creating dummy data...")
    
    # Create directory structure
    raw_dir = Path(data_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy data for each class
    for class_idx in range(num_classes):
        class_dir = raw_dir / f"cancer_type_{class_idx}"
        class_dir.mkdir(exist_ok=True)
        
        # Create patient directories
        for patient_idx in range(num_patients):
            patient_dir = class_dir / f"patient_{patient_idx}"
            patient_dir.mkdir(exist_ok=True)
            
            # Create dummy image files (just random numpy arrays saved as .npy)
            for img_idx in range(2):  # 2 images per patient
                img_data = np.random.rand(1, 224, 224).astype(np.float32)
                img_path = patient_dir / f"image_{img_idx}.npy"
                np.save(img_path, img_data)
    
    print(f"Created dummy data in {raw_dir}")


def setup_dvc():
    """Initialize DVC and set up remote storage."""
    # Initialize DVC if not already initialized
    if not os.path.exists(".dvc"):
        run_command("dvc init")
        run_command("git add .dvc")
        run_command("git commit -m 'Initialize DVC'")
    
    # Set up DVC remote storage using DAGsHub
    # This requires that the repository already exists on DAGsHub
    load_dotenv()
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    repo_name = os.path.basename(os.getcwd())
    
    if username:
        remote_url = f"https://dagshub.com/{username}/{repo_name}.dvc"
        run_command(f"dvc remote add -d dagshub {remote_url}")
        run_command("git add .dvc/config")
        run_command("git commit -m 'Configure DVC remote storage'")
    else:
        print("Warning: MLFLOW_TRACKING_USERNAME not found in .env, skipping DVC remote setup")


def setup_dagshub(repo_name, username=None):
    """Set up DAGsHub integration."""
    if username is None:
        # Try to get username from environment variable
        username = os.getenv("MLFLOW_TRACKING_USERNAME")
        if username is None:
            raise ValueError("Username not provided and not found in environment variables")
    
    # Create .env file for MLflow tracking
    env_content = f"""# MLflow configuration for DAGsHub
MLFLOW_TRACKING_URI=https://dagshub.com/{username}/{repo_name}.mlflow
MLFLOW_TRACKING_USERNAME={username}
MLFLOW_TRACKING_PASSWORD={os.getenv('MLFLOW_TRACKING_PASSWORD', 'your-token')}
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print(f"Created .env file with DAGsHub configuration for {username}/{repo_name}")
    
    # Initialize DAGsHub login if token is available
    token = os.getenv('MLFLOW_TRACKING_PASSWORD')
    if token and token != 'your-token':
        try:
            dagshub.auth.add_app_token(token)
            dagshub.init(repo_owner=username, repo_name=repo_name, mlflow=True)
            print(f"DAGsHub integration set up for {username}/{repo_name}")
        except Exception as e:
            print(f"Error setting up DAGsHub integration: {e}")


def track_data_with_dvc():
    """Track data with DVC."""
    # Track raw data directory
    if os.path.exists("data/raw"):
        run_command("dvc add data/raw")
        run_command("git add data/raw.dvc .gitignore")
        run_command("git commit -m 'Add raw data to DVC tracking'")
    else:
        print("Warning: data/raw directory not found, skipping DVC tracking")


def run_tests():
    """Run pytest tests."""
    print("Running tests...")
    return run_command("pytest")


def run_dvc_pipeline():
    """Run the DVC pipeline."""
    print("Running DVC pipeline...")
    return run_command("dvc repro")


def main():
    """Main function to run the integration."""
    parser = argparse.ArgumentParser(description="Run DVC, MLflow, and DAGsHub integration")
    parser.add_argument("--repo-name", type=str, default="cancer-diagnosis-classification",
                        help="DAGsHub repository name")
    parser.add_argument("--username", type=str, default=None,
                        help="DAGsHub username (if not provided, will use MLFLOW_TRACKING_USERNAME from .env)")
    parser.add_argument("--create-data", action="store_true",
                        help="Create dummy data for demonstration")
    parser.add_argument("--setup-only", action="store_true",
                        help="Only set up the environment, don't run the pipeline")
    parser.add_argument("--skip-tests", action="store_true",
                        help="Skip running tests")
    args = parser.parse_args()
    
    # Create dummy data if requested
    if args.create_data:
        create_dummy_data("data")
    
    # Set up DAGsHub integration
    setup_dagshub(args.repo_name, args.username)
    
    # Set up DVC
    setup_dvc()
    
    # Track data with DVC
    track_data_with_dvc()
    
    if args.setup_only:
        print("Setup completed. Use 'python run_integration.py' to run the pipeline.")
        return
    
    # Run tests
    if not args.skip_tests:
        test_result = run_tests()
        if test_result != 0:
            print("Tests failed. Aborting pipeline run.")
            return
    
    # Run DVC pipeline
    pipeline_result = run_dvc_pipeline()
    if pipeline_result == 0:
        print("DVC pipeline completed successfully.")
    else:
        print("DVC pipeline failed.")


if __name__ == "__main__":
    main()
