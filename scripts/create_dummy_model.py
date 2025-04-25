#!/usr/bin/env python
"""
Create a dummy model checkpoint for testing.
"""
import os
import torch
from pathlib import Path

from src.models.cnn_classifier import CNNClassifier


def create_dummy_model(output_path: str, num_classes: int = 2):
    """
    Create a dummy model checkpoint for testing.
    
    Args:
        output_path: Path to save the checkpoint
        num_classes: Number of classes
    """
    # Create model
    model = CNNClassifier(
        in_channels=1,
        num_classes=num_classes,
        backbone="densenet",
        pretrained=False,
        spatial_dims=3,
    )
    
    # Create checkpoint
    checkpoint = {
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {"param_groups": [], "state": {}},
        "metrics": {
            "loss": 0.5,
            "accuracy": 85.0,
        },
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save checkpoint
    torch.save(checkpoint, output_path)
    print(f"Dummy model checkpoint saved to {output_path}")


if __name__ == "__main__":
    # Create dummy model
    output_path = "models/checkpoints/dummy_model.pth"
    create_dummy_model(output_path)
