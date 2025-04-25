#!/usr/bin/env python
"""
Script to demonstrate DAGsHub integration with DVC and MLflow.
"""
import os
import argparse
import dagshub
from dotenv import load_dotenv


def setup_dagshub_integration(repo_name, username=None, token=None):
    """
    Set up DAGsHub integration for DVC and MLflow.
    
    Args:
        repo_name: Name of the DAGsHub repository
        username: DAGsHub username (if not provided, will use environment variable)
        token: DAGsHub token (if not provided, will use environment variable)
    """
    # Load environment variables
    load_dotenv()
    
    # Get credentials from environment variables if not provided
    if username is None:
        username = os.getenv("MLFLOW_TRACKING_USERNAME")
        if username is None:
            raise ValueError("Username not provided and not found in environment variables")
    
    if token is None:
        token = os.getenv("MLFLOW_TRACKING_PASSWORD")
        if token is None:
            raise ValueError("Token not provided and not found in environment variables")
    
    # Initialize DAGsHub login
    dagshub.auth.add_app_token(token)
    
    # Set up MLflow tracking
    dagshub.init(repo_owner=username, repo_name=repo_name, mlflow=True)
    
    print(f"DAGsHub integration set up for {username}/{repo_name}")
    print(f"MLflow tracking URI: {os.getenv('MLFLOW_TRACKING_URI')}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Set up DAGsHub integration")
    parser.add_argument("--repo-name", type=str, required=True,
                        help="DAGsHub repository name")
    parser.add_argument("--username", type=str, default=None,
                        help="DAGsHub username (if not provided, will use environment variable)")
    parser.add_argument("--token", type=str, default=None,
                        help="DAGsHub token (if not provided, will use environment variable)")
    args = parser.parse_args()
    
    setup_dagshub_integration(args.repo_name, args.username, args.token)


if __name__ == "__main__":
    main()
