"""
Utility functions for rare genetic disorders classification.
"""
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """
    Get device for PyTorch.

    Returns:
        PyTorch device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_metadata_csv(
    data_dir: str,
    output_file: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create metadata CSV file from data directory.

    Args:
        data_dir: Directory containing the medical images
        output_file: Path to save the metadata CSV file
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        seed: Random seed

    Returns:
        Pandas DataFrame with metadata
    """
    # Validate ratios
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

    # Get all image files
    data_dir = Path(data_dir)
    image_files = []
    patient_ids = []
    disorder_types = []
    labels = []

    # Walk through the directory structure
    for disorder_dir in [d for d in data_dir.iterdir() if d.is_dir()]:
        disorder_type = disorder_dir.name
        
        for patient_dir in [d for d in disorder_dir.iterdir() if d.is_dir()]:
            patient_id = patient_dir.name
            
            for img_file in patient_dir.glob("**/*.*"):
                if img_file.suffix.lower() in [".nii", ".nii.gz", ".dcm", ".mha", ".nrrd"]:
                    image_files.append(str(img_file.relative_to(data_dir)))
                    patient_ids.append(patient_id)
                    disorder_types.append(disorder_type)
                    # Use disorder directory index as label
                    labels.append(list(data_dir.iterdir()).index(disorder_dir))

    # Create DataFrame
    df = pd.DataFrame({
        "image_path": image_files,
        "patient_id": patient_ids,
        "disorder_type": disorder_types,
        "label": labels,
    })

    # Split into train, validation, and test sets
    # Stratify by label and ensure patients are not split across sets
    unique_patients = df[["patient_id", "label"]].drop_duplicates()
    
    # First split: train vs (val+test)
    train_patients, valtest_patients = train_test_split(
        unique_patients,
        train_size=train_ratio,
        stratify=unique_patients["label"],
        random_state=seed,
    )
    
    # Second split: val vs test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_patients, test_patients = train_test_split(
        valtest_patients,
        train_size=val_ratio_adjusted,
        stratify=valtest_patients["label"],
        random_state=seed,
    )
    
    # Assign splits
    df["split"] = "unknown"
    df.loc[df["patient_id"].isin(train_patients["patient_id"]), "split"] = "train"
    df.loc[df["patient_id"].isin(val_patients["patient_id"]), "split"] = "val"
    df.loc[df["patient_id"].isin(test_patients["patient_id"]), "split"] = "test"
    
    # Save to CSV
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    return df


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count number of trainable parameters in a PyTorch model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model_summary(
    model: torch.nn.Module,
    input_size: Tuple[int, ...],
    output_file: str,
) -> None:
    """
    Save model summary to file.

    Args:
        model: PyTorch model
        input_size: Input size for model summary
        output_file: Path to save the model summary
    """
    try:
        from torchsummary import summary
        
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Redirect stdout to file
        import sys
        original_stdout = sys.stdout
        with open(output_file, "w") as f:
            sys.stdout = f
            summary(model, input_size=input_size)
            sys.stdout = original_stdout
    except ImportError:
        print("torchsummary not installed. Install with: pip install torchsummary")


def get_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.

    Args:
        labels: Array of labels

    Returns:
        Tensor of class weights
    """
    # Count class frequencies
    class_counts = np.bincount(labels)
    
    # Calculate weights as inverse of frequency
    weights = 1.0 / class_counts
    
    # Normalize weights
    weights = weights / weights.sum() * len(weights)
    
    return torch.tensor(weights, dtype=torch.float32)


def create_experiment_directory(
    base_dir: str,
    experiment_name: str,
) -> Dict[str, str]:
    """
    Create directory structure for an experiment.

    Args:
        base_dir: Base directory
        experiment_name: Name of the experiment

    Returns:
        Dictionary of paths
    """
    base_dir = Path(base_dir)
    experiment_dir = base_dir / experiment_name
    
    # Create subdirectories
    checkpoints_dir = experiment_dir / "checkpoints"
    logs_dir = experiment_dir / "logs"
    results_dir = experiment_dir / "results"
    
    # Create directories
    for directory in [checkpoints_dir, logs_dir, results_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Return paths
    return {
        "experiment_dir": str(experiment_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "logs_dir": str(logs_dir),
        "results_dir": str(results_dir),
    }
