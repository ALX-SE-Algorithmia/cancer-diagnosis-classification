#!/usr/bin/env python
"""
Training script for cancer diagnosis classification with MLflow integration.
"""
import argparse
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from torch.utils.data import DataLoader

# Import from project modules
from src.data.dataset import RareGeneticDisordersDataset
from src.models.cnn_classifier import CNNClassifier, CustomCNNClassifier
from src.preprocessing.transforms import get_training_transforms, get_validation_transforms
from src.training.trainer import Trainer
from src.utils.config import Config, get_default_config
from src.utils.logger import setup_logger
from src.utils.mlflow_utils import setup_mlflow, log_params_from_config, log_metrics
from src.utils.utils import (
    create_experiment_directory,
    get_class_weights,
    get_device,
    save_model_summary,
    set_seed,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train cancer diagnosis classifier with MLflow")
    parser.add_argument(
        "--config", type=str, default="config/default_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None,
        help="Name of the experiment (overrides config)"
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Path to data directory (overrides config)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint to resume training"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (overrides config)"
    )
    parser.add_argument(
        "--dvc", action="store_true",
        help="Save metrics for DVC"
    )
    return parser.parse_args()


def get_optimizer(
    optimizer_name: str,
    model_parameters: Any,
    learning_rate: float,
    weight_decay: float,
) -> optim.Optimizer:
    """
    Get optimizer based on name.

    Args:
        optimizer_name: Name of the optimizer
        model_parameters: Model parameters to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay

    Returns:
        PyTorch optimizer
    """
    if optimizer_name.lower() == "adam":
        return optim.Adam(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(model_parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adamw":
        return optim.AdamW(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_scheduler(
    scheduler_name: str,
    optimizer: optim.Optimizer,
    **kwargs,
) -> Optional[Any]:
    """
    Get learning rate scheduler based on name.

    Args:
        scheduler_name: Name of the scheduler
        optimizer: PyTorch optimizer
        **kwargs: Additional scheduler parameters

    Returns:
        PyTorch scheduler
    """
    if scheduler_name.lower() == "reduce_lr_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            verbose=True,
        )
    elif scheduler_name.lower() == "cosine_annealing":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get("num_epochs", 100),
            eta_min=1e-6,
        )
    elif scheduler_name.lower() == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 30),
            gamma=kwargs.get("gamma", 0.1),
        )
    elif scheduler_name.lower() == "none" or not scheduler_name:
        return None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def main():
    """Main training function with MLflow integration."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = Config(args.config)

    # Override configuration with command line arguments
    if args.experiment_name:
        config.set("training.experiment_name", args.experiment_name)
    if args.data_dir:
        config.set("data.data_dir", args.data_dir)
    if args.seed:
        config.set("data.seed", args.seed)

    # Create experiment directory
    experiment_name = config.get("training.experiment_name")
    experiment_paths = create_experiment_directory("experiments", experiment_name)

    # Set up logger
    logger = setup_logger(
        name=experiment_name,
        log_dir=experiment_paths["logs_dir"],
        level="INFO",
    )

    # Log configuration
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Configuration: {config.config}")

    # Set random seed
    seed = config.get("data.seed")
    set_seed(seed)
    logger.info(f"Random seed: {seed}")

    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Set up MLflow
    setup_mlflow()
    
    # Start MLflow run
    with mlflow.start_run(run_name=experiment_name) as run:
        # Log parameters to MLflow
        log_params_from_config(config)
        
        # Create datasets and dataloaders
        logger.info("Creating datasets and dataloaders...")
        
        # Get transforms
        train_transforms = get_training_transforms(
            spatial_size=config.get("preprocessing.spatial_size"),
            rotation_range=config.get("preprocessing.rotation_range"),
            scale_range=config.get("preprocessing.scale_range"),
        )
        
        val_transforms = get_validation_transforms(
            spatial_size=config.get("preprocessing.spatial_size"),
        )
        
        # Create datasets
        train_dataset = RareGeneticDisordersDataset(
            data_dir=config.get("data.data_dir"),
            metadata_file=config.get("data.metadata_file"),
            transforms=train_transforms,
            cache_dir=config.get("data.cache_dir"),
            mode="train",
            use_cache=True,
        )
        
        val_dataset = RareGeneticDisordersDataset(
            data_dir=config.get("data.data_dir"),
            metadata_file=config.get("data.metadata_file"),
            transforms=val_transforms,
            cache_dir=config.get("data.cache_dir"),
            mode="val",
            use_cache=True,
        )
        
        # Create dataloaders
        train_loader = RareGeneticDisordersDataset.get_dataloader(
            dataset=train_dataset,
            batch_size=config.get("training.batch_size"),
            shuffle=True,
            num_workers=config.get("training.num_workers", 4),
        )
        
        val_loader = RareGeneticDisordersDataset.get_dataloader(
            dataset=val_dataset,
            batch_size=config.get("training.batch_size"),
            shuffle=False,
            num_workers=config.get("training.num_workers", 4),
        )
        
        # Create model
        logger.info("Creating model...")
        
        model_name = config.get("model.name")
        if model_name == "custom":
            model = CustomCNNClassifier(
                num_classes=config.get("model.num_classes"),
                in_channels=config.get("model.in_channels", 1),
                dropout_rate=config.get("model.dropout_rate", 0.2),
            )
        else:
            model = CNNClassifier(
                model_name=model_name,
                num_classes=config.get("model.num_classes"),
                pretrained=config.get("model.pretrained", True),
                dropout_rate=config.get("model.dropout_rate", 0.2),
                in_channels=config.get("model.in_channels", 1),
            )
        
        model = model.to(device)
        
        # Log model architecture to MLflow
        mlflow.log_text(str(model), "model_architecture.txt")
        
        # Save model summary
        save_model_summary(
            model=model,
            input_size=(1, *config.get("preprocessing.spatial_size")),
            output_file=os.path.join(experiment_paths["experiment_dir"], "model_summary.txt"),
        )
        
        # Define loss function
        if config.get("training.use_class_weights", False) and hasattr(train_dataset, "get_class_weights"):
            class_weights = train_dataset.get_class_weights()
            class_weights = class_weights.to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            logger.info(f"Using class weights: {class_weights}")
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Define optimizer
        optimizer = get_optimizer(
            optimizer_name=config.get("training.optimizer"),
            model_parameters=model.parameters(),
            learning_rate=config.get("training.learning_rate"),
            weight_decay=config.get("training.weight_decay"),
        )
        
        # Define scheduler
        scheduler = get_scheduler(
            scheduler_name=config.get("training.scheduler"),
            optimizer=optimizer,
            num_epochs=config.get("training.num_epochs"),
        )
        
        # Resume from checkpoint if provided
        start_epoch = 0
        if args.checkpoint:
            logger.info(f"Loading checkpoint: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            logger.info(f"Resuming from epoch {start_epoch}")
        
        # Create trainer
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger,
            checkpoint_dir=experiment_paths["checkpoints_dir"],
            early_stopping_patience=config.get("training.early_stopping_patience", 10),
            log_interval=config.get("training.log_interval", 10),
        )
        
        # Train model
        logger.info("Starting training...")
        best_val_loss, best_val_metrics = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.get("training.num_epochs"),
            start_epoch=start_epoch,
            log_mlflow=True,  # Enable MLflow logging
        )
        
        # Log best validation metrics to MLflow
        log_metrics({f"best_{k}": v for k, v in best_val_metrics.items()})
        
        # Save metrics for DVC if requested
        if args.dvc:
            metrics = {
                "best_val_loss": best_val_loss,
                **{f"best_{k}": v for k, v in best_val_metrics.items()}
            }
            
            with open("metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)
            
            logger.info(f"Saved metrics for DVC: {metrics}")
        
        logger.info("Training completed!")


if __name__ == "__main__":
    main()
