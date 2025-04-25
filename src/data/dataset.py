"""
Dataset module for loading and processing medical images for rare genetic disorders.
"""
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from monai.data import CacheDataset, Dataset, PersistentDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityd,
    Spacingd,
    ToTensord,
)
from torch.utils.data import DataLoader


class RareGeneticDisordersDataset:
    """Dataset class for rare genetic disorders medical imaging data."""

    def __init__(
        self,
        data_dir: str,
        metadata_file: str,
        transforms: Optional[Compose] = None,
        cache_dir: Optional[str] = None,
        mode: str = "train",
        use_cache: bool = True,
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing the medical images
            metadata_file: CSV file with image paths and labels
            transforms: MONAI transforms to apply to the images
            cache_dir: Directory to cache the processed images
            mode: 'train', 'val', or 'test'
            use_cache: Whether to cache the processed images
        """
        self.data_dir = data_dir
        self.mode = mode
        self.use_cache = use_cache
        self.cache_dir = cache_dir

        # Load metadata
        self.metadata = pd.read_csv(metadata_file)
        if mode != "all":
            self.metadata = self.metadata[self.metadata["split"] == mode]

        # Create data dictionaries for MONAI dataset
        self.data_dicts = self._create_data_dicts()

        # Set up transforms
        self.transforms = transforms if transforms is not None else self._get_default_transforms()

        # Create MONAI dataset
        if use_cache and cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            self.dataset = PersistentDataset(
                data=self.data_dicts,
                transform=self.transforms,
                cache_dir=cache_dir,
            )
        elif use_cache:
            self.dataset = CacheDataset(
                data=self.data_dicts,
                transform=self.transforms,
            )
        else:
            self.dataset = Dataset(
                data=self.data_dicts,
                transform=self.transforms,
            )

    def _create_data_dicts(self) -> List[Dict]:
        """
        Create a list of dictionaries for MONAI dataset.

        Returns:
            List of dictionaries with image paths and labels
        """
        data_dicts = []
        for _, row in self.metadata.iterrows():
            data_dict = {
                "image": os.path.join(self.data_dir, row["image_path"]),
                "label": row["label"],
                "patient_id": row["patient_id"],
                "disorder_type": row["disorder_type"],
            }
            data_dicts.append(data_dict)
        return data_dicts

    def _get_default_transforms(self) -> Compose:
        """
        Get default transforms for the dataset.

        Returns:
            MONAI Compose object with default transforms
        """
        return Compose(
            [
                LoadImaged(keys=["image"]),
                ScaleIntensityd(keys=["image"]),
                Spacingd(
                    keys=["image"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear"),
                ),
                ToTensord(keys=["image"]),
            ]
        )

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict:
        """Get an item from the dataset."""
        return self.dataset[index]

    @staticmethod
    def get_dataloader(
        dataset: Union["RareGeneticDisordersDataset", Dataset],
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> DataLoader:
        """
        Create a DataLoader for the dataset.

        Args:
            dataset: Dataset to create DataLoader for
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for faster data transfer to GPU

        Returns:
            DataLoader for the dataset
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
