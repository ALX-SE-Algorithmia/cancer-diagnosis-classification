#!/usr/bin/env python
"""
Script to initialize DAGsHub integration with DVC and MLflow.
"""
import os
import argparse
import dagshub
from dotenv import load_dotenv


def init_dagshub():
    """Initialize DAGsHub integration."""
    # Load environment variables
    load_dotenv()
    
    # Get credentials from environment variables
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    token = os.getenv("MLFLOW_TRACKING_PASSWORD")
    repo_name = "cancer-diagnosis-classification"
    
    if not username or not token:
        raise ValueError("DAGsHub credentials not found in .env file")
    
    print(f"Initializing DAGsHub integration for {username}/{repo_name}")
    
    # Initialize DAGsHub
    try:
        dagshub.auth.add_app_token(token)
        dagshub.init(repo_owner=username, repo_name=repo_name, mlflow=True)
        print(f"DAGsHub integration initialized successfully")
        print(f"MLflow tracking URI: {os.getenv('MLFLOW_TRACKING_URI')}")
    except Exception as e:
        print(f"Error initializing DAGsHub integration: {e}")


if __name__ == "__main__":
    init_dagshub()
