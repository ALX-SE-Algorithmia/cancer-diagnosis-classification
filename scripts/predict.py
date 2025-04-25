#!/usr/bin/env python
"""
Inference script for rare genetic disorders classification.
"""
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from monai.transforms import Compose, LoadImage, ScaleIntensity, ToTensor

from rare_genetic_disorders_classifier.models.cnn_classifier import CNNClassifier, CustomCNNClassifier
from rare_genetic_disorders_classifier.preprocessing.transforms import get_inference_transforms
from rare_genetic_disorders_classifier.utils.config import Config
from rare_genetic_disorders_classifier.utils.logger import setup_logger
from rare_genetic_disorders_classifier.utils.utils import get_device, set_seed
from rare_genetic_disorders_classifier.visualization.visualize import visualize_3d_volume


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Predict rare genetic disorders from medical images")
    parser.add_argument(
        "--config", type=str, default="configs/default_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input image or directory of images"
    )
    parser.add_argument(
        "--output", type=str, default="results/predictions",
        help="Path to output directory"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Visualize predictions"
    )
    parser.add_argument(
        "--disorder_names", type=str, default=None,
        help="Path to CSV file with disorder names (must have 'label' and 'disorder_name' columns)"
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, config: Config, device: torch.device) -> torch.nn.Module:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration object
        device: PyTorch device

    Returns:
        Loaded PyTorch model
    """
    # Create model
    backbone = config.get("model.backbone")
    num_classes = config.get("model.num_classes")
    
    if backbone == "custom":
        model = CustomCNNClassifier(
            in_channels=config.get("model.in_channels"),
            num_classes=num_classes,
            initial_filters=config.get("model.initial_filters"),
            dropout_prob=config.get("model.dropout_prob"),
            spatial_dims=config.get("model.spatial_dims"),
        )
    else:
        model = CNNClassifier(
            in_channels=config.get("model.in_channels"),
            num_classes=num_classes,
            backbone=backbone,
            pretrained=False,  # No need for pretrained weights when loading checkpoint
            spatial_dims=config.get("model.spatial_dims"),
            dropout_prob=config.get("model.dropout_prob"),
        )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model


def predict_single_image(
    image_path: str,
    model: torch.nn.Module,
    transforms: Compose,
    device: torch.device,
    disorder_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """
    Predict disorder for a single image.

    Args:
        image_path: Path to input image
        model: PyTorch model
        transforms: Preprocessing transforms
        device: PyTorch device
        disorder_names: Dictionary mapping label indices to disorder names

    Returns:
        Dictionary with prediction results
    """
    # Load and preprocess image
    image_data = transforms({"image": image_path})["image"]
    
    # Add batch dimension
    image_data = image_data.unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_data)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    # Get disorder name if available
    disorder_name = disorder_names.get(predicted_class, f"Class {predicted_class}") if disorder_names else f"Class {predicted_class}"
    
    return {
        "image_path": image_path,
        "predicted_class": predicted_class,
        "disorder_name": disorder_name,
        "confidence": confidence,
        "probabilities": probabilities[0].cpu().numpy(),
        "image_data": image_data[0].cpu(),
    }


def main():
    """Main inference function."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = Config(args.config)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logger
    logger = setup_logger(
        name="rare_genetic_disorders_predict",
        log_dir=str(output_dir),
        level="INFO",
    )

    # Set random seed
    seed = config.get("data.seed")
    set_seed(seed)

    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, config, device)
    logger.info("Model loaded successfully")

    # Get transforms
    transforms = get_inference_transforms(
        spatial_size=config.get("preprocessing.spatial_size"),
    )

    # Load disorder names if provided
    disorder_names = None
    if args.disorder_names:
        try:
            disorder_df = pd.read_csv(args.disorder_names)
            disorder_names = {row["label"]: row["disorder_name"] for _, row in disorder_df.iterrows()}
            logger.info(f"Loaded disorder names: {disorder_names}")
        except Exception as e:
            logger.warning(f"Failed to load disorder names: {e}")

    # Process input
    input_path = Path(args.input)
    results = []

    if input_path.is_file():
        # Process single file
        logger.info(f"Processing single file: {input_path}")
        result = predict_single_image(str(input_path), model, transforms, device, disorder_names)
        results.append(result)
    elif input_path.is_dir():
        # Process directory
        logger.info(f"Processing directory: {input_path}")
        image_files = list(input_path.glob("**/*.*"))
        image_files = [f for f in image_files if f.suffix.lower() in [".nii", ".nii.gz", ".dcm", ".mha", ".nrrd"]]
        
        if not image_files:
            logger.error(f"No supported image files found in {input_path}")
            return
        
        logger.info(f"Found {len(image_files)} image files")
        
        for image_file in image_files:
            logger.info(f"Processing: {image_file}")
            result = predict_single_image(str(image_file), model, transforms, device, disorder_names)
            results.append(result)
    else:
        logger.error(f"Input path does not exist: {input_path}")
        return

    # Save results
    results_df = pd.DataFrame([
        {
            "image_path": r["image_path"],
            "predicted_class": r["predicted_class"],
            "disorder_name": r["disorder_name"],
            "confidence": r["confidence"],
        }
        for r in results
    ])
    
    results_csv = output_dir / "predictions.csv"
    results_df.to_csv(results_csv, index=False)
    logger.info(f"Saved predictions to {results_csv}")

    # Visualize predictions if requested
    if args.visualize:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        for i, result in enumerate(results):
            # Visualize image
            fig = visualize_3d_volume(
                result["image_data"],
                title=f"Prediction: {result['disorder_name']} (Confidence: {result['confidence']:.2f})",
            )
            
            # Save visualization
            image_name = Path(result["image_path"]).stem
            fig_path = vis_dir / f"{image_name}_prediction.png"
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            
            # Create bar chart of probabilities
            if disorder_names:
                class_names = [disorder_names.get(i, f"Class {i}") for i in range(len(result["probabilities"]))]
            else:
                class_names = [f"Class {i}" for i in range(len(result["probabilities"]))]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(class_names, result["probabilities"])
            ax.set_xlabel("Disorder Class")
            ax.set_ylabel("Probability")
            ax.set_title(f"Prediction Probabilities for {image_name}")
            ax.set_ylim(0, 1)
            
            # Highlight predicted class
            bars[result["predicted_class"]].set_color("red")
            
            # Rotate x-axis labels if there are many classes
            if len(class_names) > 5:
                plt.xticks(rotation=45, ha="right")
            
            plt.tight_layout()
            prob_path = vis_dir / f"{image_name}_probabilities.png"
            fig.savefig(prob_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        
        logger.info(f"Saved visualizations to {vis_dir}")

    logger.info("Prediction completed successfully!")


if __name__ == "__main__":
    main()
