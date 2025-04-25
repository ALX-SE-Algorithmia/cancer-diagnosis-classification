"""
Tests for utility functions.
"""
import os
import json
import torch
import numpy as np
import pytest
from pathlib import Path
import sys

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.dvc_mlflow_utils import set_seed, get_device, save_dict_to_json, get_class_weights


def test_set_seed():
    """Test that set_seed function sets random seeds correctly."""
    # Set seed to a specific value
    set_seed(42)
    
    # Generate random numbers
    random_np_1 = np.random.rand()
    random_torch_1 = torch.rand(1).item()
    
    # Set seed to the same value again
    set_seed(42)
    
    # Generate random numbers again
    random_np_2 = np.random.rand()
    random_torch_2 = torch.rand(1).item()
    
    # Check that the random numbers are the same
    assert random_np_1 == random_np_2
    assert random_torch_1 == random_torch_2


def test_get_device():
    """Test that get_device returns a valid torch device."""
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ["cpu", "cuda"]


def test_save_dict_to_json(tmp_path):
    """Test that save_dict_to_json saves a dictionary to a JSON file."""
    # Create a test dictionary
    test_dict = {
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.79,
        "f1": 0.80
    }
    
    # Create a temporary file path
    json_path = tmp_path / "test.json"
    
    # Save the dictionary to the JSON file
    save_dict_to_json(test_dict, json_path)
    
    # Check that the file exists
    assert json_path.exists()
    
    # Load the JSON file and check that it contains the correct data
    with open(json_path, "r") as f:
        loaded_dict = json.load(f)
    
    assert loaded_dict == test_dict


def test_get_class_weights():
    """Test that get_class_weights calculates class weights correctly."""
    # Create a test set of labels with imbalanced classes
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 2])
    
    # Calculate class weights
    weights = get_class_weights(labels)
    
    # Check that weights is a torch tensor with the correct shape
    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (3,)  # 3 classes
    
    # Check that the weights are inversely proportional to class frequencies
    # Class 0: 7/10 samples, weight should be proportional to 1/(7/10) = 10/7
    # Class 1: 2/10 samples, weight should be proportional to 1/(2/10) = 10/2
    # Class 2: 1/10 samples, weight should be proportional to 1/(1/10) = 10/1
    assert weights[0] < weights[1] < weights[2]
    
    # Calculate expected weights
    class_counts = np.array([7, 2, 1])
    expected_weights = 10 / (3 * class_counts)
    expected_weights = torch.tensor(expected_weights, dtype=torch.float32)
    
    # Check that the weights are correct
    assert torch.allclose(weights, expected_weights)
