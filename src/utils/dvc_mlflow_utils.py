"""
Utility functions for DVC and MLflow integration.
"""
import os
import json
import random
import torch
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import mlflow


def setup_mlflow():
    """Configure MLflow from environment variables."""
    load_dotenv()
    
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    
    # Set authentication if provided
    if os.getenv("MLFLOW_TRACKING_USERNAME") and os.getenv("MLFLOW_TRACKING_PASSWORD"):
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """Get the device to use for training."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_dict_to_json(d, json_path):
    """Save a dictionary to a JSON file."""
    with open(json_path, "w") as f:
        json.dump(d, f, indent=4)


def load_params(params_file="config/params.yaml"):
    """Load parameters from params.yaml."""
    with open(params_file, "r") as f:
        import yaml
        params = yaml.safe_load(f)
    return params


def get_class_weights(labels):
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        labels: List of class labels
        
    Returns:
        Tensor of class weights
    """
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    
    # Calculate weights as inverse of frequency
    weights = total_samples / (len(class_counts) * class_counts)
    
    return torch.tensor(weights, dtype=torch.float32)
