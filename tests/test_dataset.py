"""
Tests for the dataset module.
"""
import os
import numpy as np
import pandas as pd
import torch
import pytest
from pathlib import Path
import sys
import tempfile
import shutil

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import CancerDataset


@pytest.fixture
def dummy_data_dir():
    """Create a temporary directory with dummy data."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create class directories
    for class_idx in range(2):
        class_dir = os.path.join(temp_dir, f"cancer_type_{class_idx}")
        os.makedirs(class_dir, exist_ok=True)
        
        # Create patient directories
        for patient_idx in range(3):
            patient_dir = os.path.join(class_dir, f"patient_{patient_idx}")
            os.makedirs(patient_dir, exist_ok=True)
            
            # Create dummy image files
            for img_idx in range(2):
                img_data = np.random.rand(1, 224, 224).astype(np.float32)
                img_path = os.path.join(patient_dir, f"image_{img_idx}.npy")
                np.save(img_path, img_data)
    
    yield temp_dir
    
    # Clean up
    shutil.rmtree(temp_dir)


@pytest.fixture
def dummy_metadata_file(dummy_data_dir):
    """Create a dummy metadata file."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
        metadata_path = temp_file.name
    
    # Create metadata
    metadata = []
    
    # Scan class directories
    for class_dir in Path(dummy_data_dir).glob("*"):
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        
        # Scan patient directories
        for patient_dir in class_dir.glob("*"):
            if not patient_dir.is_dir():
                continue
            
            patient_id = patient_dir.name
            
            # Scan image files
            for img_path in patient_dir.glob("*.npy"):
                # Assign to train/val/test based on patient_id
                patient_num = int(patient_id.split("_")[1])
                if patient_num == 0:
                    split = "train"
                elif patient_num == 1:
                    split = "val"
                else:
                    split = "test"
                
                metadata.append({
                    "file_path": str(img_path.relative_to(dummy_data_dir)),
                    "label": class_name,
                    "patient_id": patient_id,
                    "split": split
                })
    
    # Save metadata
    pd.DataFrame(metadata).to_csv(metadata_path, index=False)
    
    yield metadata_path
    
    # Clean up
    os.unlink(metadata_path)


def test_cancer_dataset_with_directory(dummy_data_dir):
    """Test CancerDataset with directory scanning."""
    # Create dataset
    dataset = CancerDataset(dummy_data_dir, split="train")
    
    # Check that the dataset has the correct length
    # We have 2 classes, 1 patient per class for train, 2 images per patient
    assert len(dataset) > 0
    
    # Check that the dataset returns tensors of the correct shape
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (1, 224, 224)
    assert isinstance(label, torch.Tensor)
    assert label.shape == ()  # Scalar


def test_cancer_dataset_with_metadata(dummy_data_dir, dummy_metadata_file):
    """Test CancerDataset with metadata file."""
    # Create dataset
    dataset = CancerDataset(
        dummy_data_dir,
        split="train",
        metadata_file=dummy_metadata_file
    )
    
    # Check that the dataset has the correct length
    # We have 2 classes, 1 patient per class for train, 2 images per patient
    assert len(dataset) > 0
    
    # Check that the dataset returns tensors of the correct shape
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (1, 224, 224)
    assert isinstance(label, torch.Tensor)
    assert label.shape == ()  # Scalar


def test_dataloader(dummy_data_dir):
    """Test the dataloader creation."""
    # Create dataset
    dataset = CancerDataset(dummy_data_dir, split="train")
    
    # Create dataloader
    dataloader = CancerDataset.get_dataloader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0  # Use 0 for testing
    )
    
    # Check that the dataloader returns batches of the correct shape
    for images, labels in dataloader:
        assert isinstance(images, torch.Tensor)
        assert images.shape[0] <= 2  # Batch size
        assert images.shape[1:] == (1, 224, 224)
        assert isinstance(labels, torch.Tensor)
        assert labels.shape[0] <= 2  # Batch size
        break  # Only check the first batch
