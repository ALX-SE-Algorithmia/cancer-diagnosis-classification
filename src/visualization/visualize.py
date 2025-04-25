"""
Visualization utilities for rare genetic disorders classification.
"""
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.visualize import blend_images, matshow3d, plot_2d_or_3d_image
from torch.utils.tensorboard import SummaryWriter


def visualize_batch(
    batch: Dict[str, torch.Tensor],
    slice_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = "gray",
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize a batch of medical images.

    Args:
        batch: Dictionary containing 'image' key with batch of images
        slice_idx: Index of the slice to visualize (for 3D images)
        figsize: Figure size
        cmap: Colormap
        title: Figure title

    Returns:
        Matplotlib figure with visualized images
    """
    images = batch["image"]
    batch_size = images.shape[0]
    
    # Create figure
    fig, axes = plt.subplots(1, batch_size, figsize=figsize)
    if batch_size == 1:
        axes = [axes]
    
    # Visualize each image in the batch
    for i, (ax, img) in enumerate(zip(axes, images)):
        # Get image data
        img_data = img.detach().cpu().numpy()
        
        # For 3D images, select a slice
        if len(img_data.shape) == 4:  # [C, D, H, W]
            if slice_idx is None:
                # Use middle slice by default
                slice_idx = img_data.shape[1] // 2
            img_data = img_data[:, slice_idx, :, :]
        
        # Remove channel dimension if only one channel
        if img_data.shape[0] == 1:
            img_data = img_data[0]
        
        # Display image
        im = ax.imshow(img_data, cmap=cmap)
        ax.set_title(f"Image {i}" if "label" not in batch else f"Label: {batch['label'][i].item()}")
        ax.axis("off")
    
    # Add colorbar
    plt.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    
    # Set title
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    return fig


def visualize_3d_volume(
    volume: torch.Tensor,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = "gray",
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize a 3D volume as a grid of slices.

    Args:
        volume: 3D volume tensor [C, D, H, W]
        figsize: Figure size
        cmap: Colormap
        title: Figure title

    Returns:
        Matplotlib figure with visualized volume
    """
    # Convert to numpy and remove batch dimension if present
    if len(volume.shape) == 5:  # [B, C, D, H, W]
        volume = volume[0]  # Take first batch
    
    vol_data = volume.detach().cpu().numpy()
    
    # Remove channel dimension if only one channel
    if vol_data.shape[0] == 1:
        vol_data = vol_data[0]  # [D, H, W]
    
    # Determine grid size
    depth = vol_data.shape[0]
    grid_size = int(np.ceil(np.sqrt(depth)))
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()
    
    # Visualize each slice
    for i in range(min(depth, grid_size * grid_size)):
        im = axes[i].imshow(vol_data[i], cmap=cmap)
        axes[i].set_title(f"Slice {i}")
        axes[i].axis("off")
    
    # Hide empty subplots
    for i in range(depth, len(axes)):
        axes[i].axis("off")
    
    # Add colorbar
    plt.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    
    # Set title
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    return fig


def visualize_model_predictions(
    images: torch.Tensor,
    true_labels: torch.Tensor,
    predicted_labels: torch.Tensor,
    slice_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (15, 10),
    cmap: str = "gray",
) -> plt.Figure:
    """
    Visualize model predictions alongside ground truth.

    Args:
        images: Batch of images [B, C, ...]
        true_labels: Ground truth labels
        predicted_labels: Predicted labels
        slice_idx: Index of the slice to visualize (for 3D images)
        figsize: Figure size
        cmap: Colormap

    Returns:
        Matplotlib figure with visualized predictions
    """
    batch_size = images.shape[0]
    
    # Create figure
    fig, axes = plt.subplots(1, batch_size, figsize=figsize)
    if batch_size == 1:
        axes = [axes]
    
    # Visualize each image in the batch
    for i, (ax, img, true, pred) in enumerate(zip(axes, images, true_labels, predicted_labels)):
        # Get image data
        img_data = img.detach().cpu().numpy()
        
        # For 3D images, select a slice
        if len(img_data.shape) == 4:  # [C, D, H, W]
            if slice_idx is None:
                # Use middle slice by default
                slice_idx = img_data.shape[1] // 2
            img_data = img_data[:, slice_idx, :, :]
        
        # Remove channel dimension if only one channel
        if img_data.shape[0] == 1:
            img_data = img_data[0]
        
        # Display image
        im = ax.imshow(img_data, cmap=cmap)
        
        # Add prediction info
        true_val = true.item()
        pred_val = pred.item()
        color = "green" if true_val == pred_val else "red"
        ax.set_title(f"True: {true_val}, Pred: {pred_val}", color=color)
        ax.axis("off")
    
    # Add colorbar
    plt.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig


def visualize_attention_maps(
    images: torch.Tensor,
    attention_maps: torch.Tensor,
    slice_idx: Optional[int] = None,
    alpha: float = 0.5,
    figsize: Tuple[int, int] = (15, 10),
    cmap_img: str = "gray",
    cmap_attention: str = "hot",
) -> plt.Figure:
    """
    Visualize attention maps overlaid on images.

    Args:
        images: Batch of images [B, C, ...]
        attention_maps: Batch of attention maps [B, 1, ...]
        slice_idx: Index of the slice to visualize (for 3D images)
        alpha: Transparency of the attention map overlay
        figsize: Figure size
        cmap_img: Colormap for the image
        cmap_attention: Colormap for the attention map

    Returns:
        Matplotlib figure with visualized attention maps
    """
    batch_size = images.shape[0]
    
    # Create figure
    fig, axes = plt.subplots(batch_size, 3, figsize=figsize)
    if batch_size == 1:
        axes = [axes]
    
    # Visualize each image and attention map in the batch
    for i, (ax_row, img, attn) in enumerate(zip(axes, images, attention_maps)):
        # Get image data
        img_data = img.detach().cpu().numpy()
        attn_data = attn.detach().cpu().numpy()
        
        # For 3D images, select a slice
        if len(img_data.shape) == 4:  # [C, D, H, W]
            if slice_idx is None:
                # Use middle slice by default
                slice_idx = img_data.shape[1] // 2
            img_data = img_data[:, slice_idx, :, :]
            attn_data = attn_data[:, slice_idx, :, :]
        
        # Remove channel dimension if only one channel
        if img_data.shape[0] == 1:
            img_data = img_data[0]
        if attn_data.shape[0] == 1:
            attn_data = attn_data[0]
        
        # Normalize attention map
        attn_data = (attn_data - attn_data.min()) / (attn_data.max() - attn_data.min() + 1e-8)
        
        # Display original image
        ax_row[0].imshow(img_data, cmap=cmap_img)
        ax_row[0].set_title("Original Image")
        ax_row[0].axis("off")
        
        # Display attention map
        im_attn = ax_row[1].imshow(attn_data, cmap=cmap_attention)
        ax_row[1].set_title("Attention Map")
        ax_row[1].axis("off")
        
        # Display overlay
        ax_row[2].imshow(img_data, cmap=cmap_img)
        im_overlay = ax_row[2].imshow(attn_data, cmap=cmap_attention, alpha=alpha)
        ax_row[2].set_title("Overlay")
        ax_row[2].axis("off")
    
    # Add colorbars
    plt.colorbar(im_attn, ax=[ax_row[1] for ax_row in axes], fraction=0.046, pad=0.04)
    plt.colorbar(im_overlay, ax=[ax_row[2] for ax_row in axes], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig


def log_images_to_tensorboard(
    writer: SummaryWriter,
    images: torch.Tensor,
    true_labels: torch.Tensor,
    predicted_labels: torch.Tensor,
    step: int,
    tag: str = "predictions",
    max_images: int = 8,
    slice_idx: Optional[int] = None,
) -> None:
    """
    Log images with predictions to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        images: Batch of images
        true_labels: Ground truth labels
        predicted_labels: Predicted labels
        step: Current step
        tag: Tag for the images
        max_images: Maximum number of images to log
        slice_idx: Index of the slice to visualize (for 3D images)
    """
    # Limit the number of images
    batch_size = min(images.shape[0], max_images)
    images = images[:batch_size]
    true_labels = true_labels[:batch_size]
    predicted_labels = predicted_labels[:batch_size]
    
    # Create visualization
    fig = visualize_model_predictions(
        images, true_labels, predicted_labels, slice_idx=slice_idx
    )
    
    # Log to TensorBoard
    writer.add_figure(tag, fig, step)
    plt.close(fig)
