"""
Configuration utilities for cancer diagnosis classification.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


class Config:
    """Configuration class for managing experiment settings."""

    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        Initialize configuration.

        Args:
            config_path: Path to configuration file (YAML or JSON)
            **kwargs: Additional configuration parameters
        """
        self.config = {}
        
        # Load configuration from file if provided
        if config_path is not None:
            self.config = self.load_config(config_path)
        
        # Update with additional parameters
        self.config.update(kwargs)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.

        Args:
            config_path: Path to configuration file (YAML or JSON)

        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load based on file extension
        if config_path.suffix.lower() in [".yml", ".yaml"]:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        elif config_path.suffix.lower() == ".json":
            with open(config_path, "r") as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

    def save_config(self, config_path: str) -> None:
        """
        Save configuration to file.

        Args:
            config_path: Path to save configuration file (YAML or JSON)
        """
        config_path = Path(config_path)
        
        # Create directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on file extension
        if config_path.suffix.lower() in [".yml", ".yaml"]:
            with open(config_path, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)
        elif config_path.suffix.lower() == ".json":
            with open(config_path, "w") as f:
                json.dump(self.config, f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.

        Args:
            key: Configuration key
            value: Configuration value
        """
        keys = key.split(".")
        config = self.config
        
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value

    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with dictionary.

        Args:
            config_dict: Dictionary of configuration values
        """
        self.config.update(config_dict)

    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dictionary-like access."""
        return self.config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set configuration value using dictionary-like access."""
        self.config[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if configuration contains key."""
        return key in self.config

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config({self.config})"
