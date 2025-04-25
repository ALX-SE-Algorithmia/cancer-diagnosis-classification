"""
Tests for preprocessing transforms.
"""
import pytest
import torch
import numpy as np
from monai.data import MetaTensor

from src.preprocessing.transforms import (
    get_training_transforms,
    get_validation_transforms,
    get_inference_transforms,
)


def test_training_transforms():
    """Test training transforms."""
    # Create a dictionary with an image
    keys = ["image"]
    spatial_size = (64, 64, 64)
    
    # Get transforms
    transforms = get_training_transforms(
        keys=keys,
        spatial_size=spatial_size,
    )
    
    # Create a dummy 3D image (simulating a loaded medical image)
    image_data = np.random.rand(100, 100, 100).astype(np.float32)
    data_dict = {"image": image_data}
    
    # Apply transforms
    result = transforms(data_dict)
    
    # Check that the result is a dictionary with the expected keys
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "image" in result, "Result should contain 'image' key"
    
    # Check that the image is a tensor with the expected shape
    assert isinstance(result["image"], (torch.Tensor, MetaTensor)), "Image should be a tensor"
    assert result["image"].shape[1:] == spatial_size, f"Expected shape {(1, *spatial_size)}, got {result['image'].shape}"
    assert result["image"].shape[0] == 1, "Expected 1 channel"


def test_validation_transforms():
    """Test validation transforms."""
    # Create a dictionary with an image
    keys = ["image"]
    spatial_size = (64, 64, 64)
    
    # Get transforms
    transforms = get_validation_transforms(
        keys=keys,
        spatial_size=spatial_size,
    )
    
    # Create a dummy 3D image (simulating a loaded medical image)
    image_data = np.random.rand(100, 100, 100).astype(np.float32)
    data_dict = {"image": image_data}
    
    # Apply transforms
    result = transforms(data_dict)
    
    # Check that the result is a dictionary with the expected keys
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "image" in result, "Result should contain 'image' key"
    
    # Check that the image is a tensor with the expected shape
    assert isinstance(result["image"], (torch.Tensor, MetaTensor)), "Image should be a tensor"
    assert result["image"].shape[1:] == spatial_size, f"Expected shape {(1, *spatial_size)}, got {result['image'].shape}"
    assert result["image"].shape[0] == 1, "Expected 1 channel"


def test_inference_transforms():
    """Test inference transforms."""
    # Create a dictionary with an image
    keys = ["image"]
    spatial_size = (64, 64, 64)
    
    # Get transforms
    transforms = get_inference_transforms(
        keys=keys,
        spatial_size=spatial_size,
    )
    
    # Create a dummy 3D image (simulating a loaded medical image)
    image_data = np.random.rand(100, 100, 100).astype(np.float32)
    data_dict = {"image": image_data}
    
    # Apply transforms
    result = transforms(data_dict)
    
    # Check that the result is a dictionary with the expected keys
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "image" in result, "Result should contain 'image' key"
    
    # Check that the image is a tensor with the expected shape
    assert isinstance(result["image"], (torch.Tensor, MetaTensor)), "Image should be a tensor"
    assert result["image"].shape[1:] == spatial_size, f"Expected shape {(1, *spatial_size)}, got {result['image'].shape}"
    assert result["image"].shape[0] == 1, "Expected 1 channel"
