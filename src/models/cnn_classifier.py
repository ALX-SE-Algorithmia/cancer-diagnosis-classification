"""
CNN models for rare genetic disorders classification from medical images.
"""
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DenseNet121, ResNet


class CNNClassifier(nn.Module):
    """CNN classifier for rare genetic disorders."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        backbone: str = "densenet",
        pretrained: bool = True,
        spatial_dims: int = 3,
        dropout_prob: float = 0.2,
    ):
        """
        Initialize the CNN classifier.

        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            backbone: Backbone architecture ('densenet' or 'resnet')
            pretrained: Whether to use pretrained weights
            spatial_dims: Number of spatial dimensions (2 or 3)
            dropout_prob: Dropout probability
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.backbone = backbone
        self.pretrained = pretrained
        self.spatial_dims = spatial_dims
        self.dropout_prob = dropout_prob

        # Initialize backbone
        if backbone == "densenet":
            self.feature_extractor = DenseNet121(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=num_classes,
                pretrained=pretrained,
                dropout_prob=dropout_prob,
            )
        elif backbone == "resnet":
            self.feature_extractor = ResNet(
                block="basic",
                layers=[2, 2, 2, 2],  # ResNet-18
                block_inplanes=[64, 128, 256, 512],
                spatial_dims=spatial_dims,
                n_input_channels=in_channels,
                num_classes=num_classes,
                conv1_t_stride=2,
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN classifier.

        Args:
            x: Input tensor of shape (batch_size, in_channels, *spatial_dims)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.feature_extractor(x)


class CustomCNNClassifier(nn.Module):
    """Custom CNN classifier for rare genetic disorders."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        initial_filters: int = 32,
        dropout_prob: float = 0.2,
        spatial_dims: int = 3,
    ):
        """
        Initialize the custom CNN classifier.

        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            initial_filters: Number of filters in the first layer
            dropout_prob: Dropout probability
            spatial_dims: Number of spatial dimensions (2 or 3)
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.initial_filters = initial_filters
        self.dropout_prob = dropout_prob
        self.spatial_dims = spatial_dims

        # Choose convolution and pooling based on spatial dimensions
        if spatial_dims == 3:
            conv_layer = nn.Conv3d
            pool_layer = nn.MaxPool3d
            adaptive_pool = nn.AdaptiveAvgPool3d
        elif spatial_dims == 2:
            conv_layer = nn.Conv2d
            pool_layer = nn.MaxPool2d
            adaptive_pool = nn.AdaptiveAvgPool2d
        else:
            raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}")

        # Feature extraction layers
        self.conv1 = conv_layer(in_channels, initial_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(initial_filters) if spatial_dims == 3 else nn.BatchNorm2d(initial_filters)
        self.pool1 = pool_layer(kernel_size=2, stride=2)

        self.conv2 = conv_layer(initial_filters, initial_filters * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(initial_filters * 2) if spatial_dims == 3 else nn.BatchNorm2d(initial_filters * 2)
        self.pool2 = pool_layer(kernel_size=2, stride=2)

        self.conv3 = conv_layer(initial_filters * 2, initial_filters * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(initial_filters * 4) if spatial_dims == 3 else nn.BatchNorm2d(initial_filters * 4)
        self.pool3 = pool_layer(kernel_size=2, stride=2)

        self.conv4 = conv_layer(initial_filters * 4, initial_filters * 8, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(initial_filters * 8) if spatial_dims == 3 else nn.BatchNorm2d(initial_filters * 8)
        self.pool4 = pool_layer(kernel_size=2, stride=2)

        # Global average pooling
        self.global_pool = adaptive_pool((1, 1, 1) if spatial_dims == 3 else (1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(initial_filters * 8, initial_filters * 4)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(initial_filters * 4, initial_filters * 2)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(initial_filters * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the custom CNN classifier.

        Args:
            x: Input tensor of shape (batch_size, in_channels, *spatial_dims)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x
