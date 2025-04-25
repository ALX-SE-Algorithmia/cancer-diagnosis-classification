"""
Transforms for preprocessing medical images for rare genetic disorders.
"""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandRotated,
    RandZoomd,
    Resized,
    ScaleIntensityd,
    Spacingd,
    ToTensord,
)


def get_training_transforms(
    keys: List[str] = ["image"],
    spatial_size: Tuple[int, int, int] = (128, 128, 128),
    rotation_range: Tuple[float, float] = (-15, 15),
    scale_range: Tuple[float, float] = (-0.1, 0.1),
) -> Compose:
    """
    Get transforms for training data.

    Args:
        keys: Keys for the transforms
        spatial_size: Size to resize the images to
        rotation_range: Range for random rotation in degrees
        scale_range: Range for random scaling

    Returns:
        MONAI Compose object with training transforms
    """
    return Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(
                keys=keys,
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear"),
            ),
            Resized(keys=keys, spatial_size=spatial_size),
            ScaleIntensityd(keys=keys),
            NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
            RandAffined(
                keys=keys,
                prob=0.5,
                rotate_range=rotation_range,
                scale_range=scale_range,
                mode=("bilinear"),
                padding_mode="zeros",
            ),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            RandRotated(keys=keys, prob=0.5, range_x=rotation_range),
            RandRotated(keys=keys, prob=0.5, range_y=rotation_range),
            RandRotated(keys=keys, prob=0.5, range_z=rotation_range),
            RandZoomd(keys=keys, prob=0.5, min_zoom=0.9, max_zoom=1.1),
            RandGaussianNoised(keys=keys, prob=0.2, mean=0.0, std=0.1),
            RandAdjustContrastd(keys=keys, prob=0.2, gamma=(0.8, 1.2)),
            ToTensord(keys=keys),
        ]
    )


def get_validation_transforms(
    keys: List[str] = ["image"],
    spatial_size: Tuple[int, int, int] = (128, 128, 128),
) -> Compose:
    """
    Get transforms for validation data.

    Args:
        keys: Keys for the transforms
        spatial_size: Size to resize the images to

    Returns:
        MONAI Compose object with validation transforms
    """
    return Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(
                keys=keys,
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear"),
            ),
            Resized(keys=keys, spatial_size=spatial_size),
            ScaleIntensityd(keys=keys),
            NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
            ToTensord(keys=keys),
        ]
    )


def get_inference_transforms(
    keys: List[str] = ["image"],
    spatial_size: Tuple[int, int, int] = (128, 128, 128),
) -> Compose:
    """
    Get transforms for inference.

    Args:
        keys: Keys for the transforms
        spatial_size: Size to resize the images to

    Returns:
        MONAI Compose object with inference transforms
    """
    return get_validation_transforms(keys, spatial_size)
