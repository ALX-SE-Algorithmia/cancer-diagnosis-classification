"""
Trainer module with MLflow integration for training CNN models for cancer diagnosis classification.
"""
import os
import mlflow
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.evaluation.metrics import (
    calculate_accuracy,
    calculate_auc,
    calculate_confusion_matrix,
    calculate_f1_score,
    calculate_precision_recall,
)


class Trainer:
    """Trainer class for training CNN models with MLflow integration."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module = None,
        optimizer: optim.Optimizer = None,
        scheduler: Any = None,
        device: torch.device = None,
        logger: Any = None,
        checkpoint_dir: str = "./models/checkpoints",
        tensorboard_dir: str = "./logs/tensorboard",
        early_stopping_patience: int = 10,
        log_interval: int = 10,
    ):
        """
        Initialize the trainer.

        Args:
            model: PyTorch model to train
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            logger: Logger instance
            checkpoint_dir: Directory to save checkpoints
            tensorboard_dir: Directory to save tensorboard logs
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            log_interval: Interval for logging during training
        """
        self.model = model
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.optimizer = (
            optimizer
            if optimizer is not None
            else optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        )
        self.scheduler = scheduler
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.checkpoint_dir = Path(checkpoint_dir)
        self.tensorboard_dir = Path(tensorboard_dir) if tensorboard_dir else None
        self.early_stopping_patience = early_stopping_patience
        self.log_interval = log_interval

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.tensorboard_dir:
            self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
        else:
            self.writer = None

        # Move model to device
        self.model.to(self.device)

        # Initialize best metrics
        self.best_val_loss = float("inf")
        self.best_val_metrics = {}
        self.best_epoch = 0
        self.patience_counter = 0

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        start_epoch: int = 0,
        log_mlflow: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Train the model.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of epochs to train
            start_epoch: Starting epoch number (for resuming training)
            log_mlflow: Whether to log metrics to MLflow

        Returns:
            Tuple of (best validation loss, best validation metrics)
        """
        self.log(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(start_epoch, num_epochs):
            # Train for one epoch
            train_metrics = self.train_epoch(epoch, train_loader)
            
            # Validate
            val_metrics = self.validate_epoch(epoch, val_loader)
            
            # Log metrics
            self.log(f"Epoch {epoch+1}/{num_epochs}")
            self.log(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
            self.log(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
            
            # Log to MLflow if enabled
            if log_mlflow:
                # Log training metrics
                mlflow.log_metrics(
                    {f"train_{k}": v for k, v in train_metrics.items()},
                    step=epoch
                )
                
                # Log validation metrics
                mlflow.log_metrics(
                    {f"val_{k}": v for k, v in val_metrics.items()},
                    step=epoch
                )
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()
            
            # Check if this is the best model
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_val_metrics = val_metrics
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                self.log(f"New best model saved at epoch {epoch+1}")
                
                if log_mlflow:
                    # Log best model to MLflow
                    mlflow.pytorch.log_model(self.model, "best_model")
            else:
                self.patience_counter += 1
                self.log(f"No improvement for {self.patience_counter} epochs")
                
                # Save regular checkpoint
                self.save_checkpoint(epoch, val_metrics, is_best=False)
                
                # Early stopping
                if self.patience_counter >= self.early_stopping_patience:
                    self.log(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        self.log(f"Training completed. Best model at epoch {self.best_epoch+1}")
        self.log(f"Best validation loss: {self.best_val_loss:.4f}")
        self.log(f"Best validation accuracy: {self.best_val_metrics['accuracy']:.2f}%")
        
        return self.best_val_loss, self.best_val_metrics

    def train_epoch(self, epoch: int, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train the model for one epoch.

        Args:
            epoch: Current epoch number
            train_loader: DataLoader for training data

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_targets = []
        all_predictions = []
        all_probabilities = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for batch_idx, data in enumerate(progress_bar):
            # Get data (handle different data formats)
            if isinstance(data, dict):
                inputs = data["image"].to(self.device)
                targets = data["label"].to(self.device)
            elif isinstance(data, tuple) and len(data) == 2:
                inputs, targets = data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
            else:
                raise ValueError("Unsupported data format")

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Calculate metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Store targets and predictions for metrics calculation
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            # Store probabilities for AUC calculation
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            all_probabilities.extend(probabilities.detach().cpu().numpy())

            # Update progress bar
            if (batch_idx + 1) % self.log_interval == 0 or batch_idx == len(train_loader) - 1:
                progress_bar.set_postfix(
                    {
                        "loss": running_loss / (batch_idx + 1),
                        "acc": 100.0 * correct / total,
                    }
                )

        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100.0 * correct / total
        
        # Convert to numpy arrays
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate additional metrics
        try:
            epoch_precision, epoch_recall = calculate_precision_recall(
                all_targets, all_predictions
            )
            epoch_f1 = calculate_f1_score(
                all_targets, all_predictions
            )
            
            # Only calculate AUC if we have probabilities and multiple classes
            num_classes = all_probabilities.shape[1] if len(all_probabilities) > 0 else 0
            if num_classes > 1:
                epoch_auc = calculate_auc(
                    all_targets, all_probabilities
                )
            else:
                epoch_auc = 0.0
                
        except Exception as e:
            self.log(f"Error calculating metrics: {e}")
            epoch_precision = epoch_recall = epoch_f1 = epoch_auc = 0.0

        # Log metrics to tensorboard
        if self.writer:
            self.writer.add_scalar("Loss/train", epoch_loss, epoch)
            self.writer.add_scalar("Accuracy/train", epoch_accuracy, epoch)
            self.writer.add_scalar("Precision/train", epoch_precision, epoch)
            self.writer.add_scalar("Recall/train", epoch_recall, epoch)
            self.writer.add_scalar("F1/train", epoch_f1, epoch)
            self.writer.add_scalar("AUC/train", epoch_auc, epoch)

        return {
            "loss": epoch_loss,
            "accuracy": epoch_accuracy,
            "precision": epoch_precision,
            "recall": epoch_recall,
            "f1": epoch_f1,
            "auc": epoch_auc,
        }

    def validate_epoch(self, epoch: int, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model for one epoch.

        Args:
            epoch: Current epoch number
            val_loader: DataLoader for validation data

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_targets = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
            for batch_idx, data in enumerate(progress_bar):
                # Get data (handle different data formats)
                if isinstance(data, dict):
                    inputs = data["image"].to(self.device)
                    targets = data["label"].to(self.device)
                elif isinstance(data, tuple) and len(data) == 2:
                    inputs, targets = data
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                else:
                    raise ValueError("Unsupported data format")

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Calculate metrics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Store targets and predictions for metrics calculation
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                
                # Store probabilities for AUC calculation
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                all_probabilities.extend(probabilities.cpu().numpy())

                # Update progress bar
                progress_bar.set_postfix(
                    {
                        "loss": running_loss / (batch_idx + 1),
                        "acc": 100.0 * correct / total,
                    }
                )

        # Calculate epoch metrics
        epoch_loss = running_loss / len(val_loader)
        epoch_accuracy = 100.0 * correct / total
        
        # Convert to numpy arrays
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate additional metrics
        try:
            epoch_precision, epoch_recall = calculate_precision_recall(
                all_targets, all_predictions
            )
            epoch_f1 = calculate_f1_score(
                all_targets, all_predictions
            )
            
            # Calculate confusion matrix
            conf_matrix = calculate_confusion_matrix(
                all_targets, all_predictions
            )
            
            # Only calculate AUC if we have probabilities and multiple classes
            num_classes = all_probabilities.shape[1] if len(all_probabilities) > 0 else 0
            if num_classes > 1:
                epoch_auc = calculate_auc(
                    all_targets, all_probabilities
                )
            else:
                epoch_auc = 0.0
                
        except Exception as e:
            self.log(f"Error calculating metrics: {e}")
            epoch_precision = epoch_recall = epoch_f1 = epoch_auc = 0.0
            conf_matrix = None

        # Log metrics to tensorboard
        if self.writer:
            self.writer.add_scalar("Loss/val", epoch_loss, epoch)
            self.writer.add_scalar("Accuracy/val", epoch_accuracy, epoch)
            self.writer.add_scalar("Precision/val", epoch_precision, epoch)
            self.writer.add_scalar("Recall/val", epoch_recall, epoch)
            self.writer.add_scalar("F1/val", epoch_f1, epoch)
            self.writer.add_scalar("AUC/val", epoch_auc, epoch)

        return {
            "loss": epoch_loss,
            "accuracy": epoch_accuracy,
            "precision": epoch_precision,
            "recall": epoch_recall,
            "f1": epoch_f1,
            "auc": epoch_auc,
            "confusion_matrix": conf_matrix,
        }

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False) -> None:
        """
        Save a checkpoint of the model.

        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if this is the best so far
        if is_best:
            best_model_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_model_path)

    def log(self, message: str) -> None:
        """
        Log a message.

        Args:
            message: Message to log
        """
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
