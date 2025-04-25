"""
Tests for DVC and MLflow integration.
"""
import os
import yaml
import json
import pytest
import sys
from unittest.mock import patch, MagicMock

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.dvc_mlflow_utils import load_params, setup_mlflow


def test_load_params():
    """Test that load_params loads parameters correctly."""
    # Load parameters from the default config file
    params = load_params()
    
    # Check that the parameters have the expected structure
    assert "data" in params
    assert "model" in params
    assert "training" in params
    assert "preprocessing" in params
    assert "evaluation" in params
    
    # Check some specific parameters
    assert "raw_dir" in params["data"]
    assert "processed_dir" in params["data"]
    assert "name" in params["model"]
    assert "num_classes" in params["model"]
    assert "experiment_name" in params["training"]
    assert "batch_size" in params["training"]
    assert "spatial_size" in params["preprocessing"]


@patch("mlflow.set_tracking_uri")
@patch("os.environ")
@patch("os.getenv")
def test_setup_mlflow(mock_getenv, mock_environ, mock_set_tracking_uri):
    """Test that setup_mlflow configures MLflow correctly."""
    # Mock environment variables
    mock_getenv.side_effect = lambda x: {
        "MLFLOW_TRACKING_URI": "https://dagshub.com/user/repo.mlflow",
        "MLFLOW_TRACKING_USERNAME": "user",
        "MLFLOW_TRACKING_PASSWORD": "token"
    }.get(x)
    
    # Call setup_mlflow
    setup_mlflow()
    
    # Check that mlflow.set_tracking_uri was called with the correct URI
    mock_set_tracking_uri.assert_called_once_with("https://dagshub.com/user/repo.mlflow")
    
    # Check that environment variables were set
    assert mock_environ.__setitem__.call_count == 2
    mock_environ.__setitem__.assert_any_call("MLFLOW_TRACKING_USERNAME", "user")
    mock_environ.__setitem__.assert_any_call("MLFLOW_TRACKING_PASSWORD", "token")


def test_dvc_yaml_structure():
    """Test that dvc.yaml has the correct structure."""
    # Check that dvc.yaml exists
    dvc_yaml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dvc.yaml")
    assert os.path.exists(dvc_yaml_path)
    
    # Load dvc.yaml
    with open(dvc_yaml_path, "r") as f:
        dvc_yaml = yaml.safe_load(f)
    
    # Check that dvc.yaml has the expected structure
    assert "stages" in dvc_yaml
    assert "prepare" in dvc_yaml["stages"]
    assert "train" in dvc_yaml["stages"]
    assert "evaluate" in dvc_yaml["stages"]
    
    # Check that each stage has the expected components
    for stage in ["prepare", "train", "evaluate"]:
        assert "cmd" in dvc_yaml["stages"][stage]
        assert "deps" in dvc_yaml["stages"][stage]
    
    # Check that metrics are defined
    assert "metrics" in dvc_yaml["stages"]["train"]
    assert "metrics" in dvc_yaml["stages"]["evaluate"]
