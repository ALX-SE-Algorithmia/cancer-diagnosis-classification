"""
Trainer module for training CNN models for rare genetic disorders classification.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rare_genetic_disorders_classifier.evaluation.metrics import (
    calculate_accuracy,
    calculate_auc,
    calculate_confusion_matrix,
    calculate_f1_score,
    calculate_precision_recall,
)


class Trainer:
    """Trainer class for training CNN models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module = None,
        optimizer: optim.Optimizer = None,
        scheduler: Any = None,
        device: torch.device = None,
        num_classes: int = 2,
        checkpoint_dir: str = "./models/checkpoints",
        tensorboard_dir: str = "./logs/tensorboard",
        experiment_name: str = "experiment",
    ):
        """
        Initialize the trainer.

        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            num_classes: Number of classes
            checkpoint_dir: Directory to save checkpoints
            tensorboard_dir: Directory to save tensorboard logs
            experiment_name: Name of the experiment
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.optimizer = (
            optimizer
            if optimizer is not None
            else optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        )
        self.scheduler = scheduler
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.checkpoint_dir = Path(checkpoint_dir)
        self.tensorboard_dir = Path(tensorboard_dir)
        self.experiment_name = experiment_name

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir / experiment_name))

        # Move model to device
        self.model.to(self.device)

        # Initialize best metrics
        self.best_val_loss = float("inf")
        self.best_val_accuracy = 0.0
        self.best_epoch = 0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train the model for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_targets = []
        all_predictions = []

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, batch_data in enumerate(progress_bar):
            # Get data
            inputs = batch_data["image"].to(self.device)
            targets = batch_data["label"].to(self.device)

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

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": running_loss / (batch_idx + 1),
                    "acc": 100.0 * correct / total,
                }
            )

        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = 100.0 * correct / total
        epoch_precision, epoch_recall = calculate_precision_recall(
            np.array(all_targets), np.array(all_predictions), self.num_classes
        )
        epoch_f1 = calculate_f1_score(
            np.array(all_targets), np.array(all_predictions), self.num_classes
        )
        epoch_auc = calculate_auc(
            np.array(all_targets), np.array(all_predictions), self.num_classes
        )

        # Log metrics to tensorboard
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

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for batch_idx, batch_data in enumerate(progress_bar):
                # Get data
                inputs = batch_data["image"].to(self.device)
                targets = batch_data["label"].to(self.device)

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

                # Update progress bar
                progress_bar.set_postfix(
                    {
                        "loss": running_loss / (batch_idx + 1),
                        "acc": 100.0 * correct / total,
                    }
                )

        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.val_loader)
        epoch_accuracy = 100.0 * correct / total
        epoch_precision, epoch_recall = calculate_precision_recall(
            np.array(all_targets), np.array(all_predictions), self.num_classes
        )
        epoch_f1 = calculate_f1_score(
            np.array(all_targets), np.array(all_predictions), self.num_classes
        )
        epoch_auc = calculate_auc(
            np.array(all_targets), np.array(all_predictions), self.num_classes
        )

        # Log metrics to tensorboard
        self.writer.add_scalar("Loss/val", epoch_loss, epoch)
        self.writer.add_scalar("Accuracy/val", epoch_accuracy, epoch)
        self.writer.add_scalar("Precision/val", epoch_precision, epoch)
        self.writer.add_scalar("Recall/val", epoch_recall, epoch)
        self.writer.add_scalar("F1/val", epoch_f1, epoch)
        self.writer.add_scalar("AUC/val", epoch_auc, epoch)

        # Save confusion matrix
        cm = calculate_confusion_matrix(
            np.array(all_targets), np.array(all_predictions), self.num_classes
        )
        self.writer.add_figure(
            "Confusion Matrix/val",
            cm,
            epoch,
        )

        return {
            "loss": epoch_loss,
            "accuracy": epoch_accuracy,
            "precision": epoch_precision,
            "recall": epoch_recall,
            "f1": epoch_f1,
            "auc": epoch_auc,
        }

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False) -> None:
        """
        Save model checkpoint.

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

        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / f"{self.experiment_name}_latest.pth"
        torch.save(checkpoint, checkpoint_path)

        # Save epoch checkpoint
        checkpoint_path = self.checkpoint_dir / f"{self.experiment_name}_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            checkpoint_path = self.checkpoint_dir / f"{self.experiment_name}_best.pth"
            torch.save(checkpoint, checkpoint_path)

    def train(self, num_epochs: int, early_stopping_patience: int = 10) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train for
            early_stopping_patience: Number of epochs to wait for improvement before stopping

        Returns:
            Dictionary of training and validation metrics
        """
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            # Train and validate
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)

            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # Store metrics
            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])

            # Check if this is the best model
            is_best = False
            if val_metrics["accuracy"] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics["accuracy"]
                self.best_epoch = epoch
                is_best = True
                patience_counter = 0
            elif val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                is_best = True
                patience_counter = 0
            else:
                patience_counter += 1

            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)

            # Print epoch summary
            print(f"\nEpoch {epoch} summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.2f}%")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.2f}%")
            print(f"  Best Val Accuracy: {self.best_val_accuracy:.2f}% (Epoch {self.best_epoch})")

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping after {epoch} epochs without improvement.")
                break

        # Close tensorboard writer
        self.writer.close()

        return history

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Dictionary of checkpoint data
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint
