#!/usr/bin/env python
"""
Training script for rare genetic disorders classification.
"""
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from rare_genetic_disorders_classifier.data.dataset import RareGeneticDisordersDataset
from rare_genetic_disorders_classifier.models.cnn_classifier import CNNClassifier, CustomCNNClassifier
from rare_genetic_disorders_classifier.preprocessing.transforms import get_training_transforms, get_validation_transforms
from rare_genetic_disorders_classifier.training.trainer import Trainer
from rare_genetic_disorders_classifier.utils.config import Config, get_default_config
from rare_genetic_disorders_classifier.utils.logger import setup_logger
from rare_genetic_disorders_classifier.utils.utils import (
    create_experiment_directory,
    get_class_weights,
    get_device,
    save_model_summary,
    set_seed,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train rare genetic disorders classifier")
    parser.add_argument(
        "--config", type=str, default="configs/default_config.yaml",
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
    """Main training function."""
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
        num_workers=config.get("training.num_workers"),
        pin_memory=True,
    )
    
    val_loader = RareGeneticDisordersDataset.get_dataloader(
        dataset=val_dataset,
        batch_size=config.get("training.batch_size"),
        shuffle=False,
        num_workers=config.get("training.num_workers"),
        pin_memory=True,
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    # Create model
    logger.info("Creating model...")
    model_name = config.get("model.name")
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
            pretrained=config.get("model.pretrained"),
            spatial_dims=config.get("model.spatial_dims"),
            dropout_prob=config.get("model.dropout_prob"),
        )
    
    logger.info(f"Model: {model_name} with backbone {backbone}")
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Save model summary
    save_model_summary(
        model=model,
        input_size=(config.get("model.in_channels"), *config.get("preprocessing.spatial_size")),
        output_file=os.path.join(experiment_paths["experiment_dir"], "model_summary.txt"),
    )

    # Set up loss function
    if config.get("training.class_weights", False):
        # Get class weights from training data
        all_labels = [data["label"] for data in train_dataset]
        class_weights = get_class_weights(torch.tensor(all_labels).numpy())
        logger.info(f"Using class weights: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    # Set up optimizer
    optimizer = get_optimizer(
        optimizer_name=config.get("training.optimizer"),
        model_parameters=model.parameters(),
        learning_rate=config.get("training.learning_rate"),
        weight_decay=config.get("training.weight_decay"),
    )

    # Set up scheduler
    scheduler = get_scheduler(
        scheduler_name=config.get("training.scheduler"),
        optimizer=optimizer,
        num_epochs=config.get("training.num_epochs"),
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_classes=num_classes,
        checkpoint_dir=experiment_paths["checkpoints_dir"],
        tensorboard_dir=experiment_paths["logs_dir"],
        experiment_name=experiment_name,
    )

    # Load checkpoint if provided
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = trainer.load_checkpoint(args.checkpoint)
        logger.info(f"Resuming from epoch {checkpoint['epoch']}")

    # Train model
    logger.info("Starting training...")
    trainer.train(
        num_epochs=config.get("training.num_epochs"),
        early_stopping_patience=config.get("training.early_stopping_patience"),
    )
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
