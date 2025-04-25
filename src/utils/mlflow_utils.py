"""
Utility functions for MLflow integration.
"""
import os
from dotenv import load_dotenv
import mlflow


def setup_mlflow():
    """Configure MLflow from environment variables."""
    load_dotenv()
    
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    
    # Set authentication if provided
    if os.getenv("MLFLOW_TRACKING_USERNAME") and os.getenv("MLFLOW_TRACKING_PASSWORD"):
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")


def log_params_from_config(config):
    """
    Log parameters from a configuration object to MLflow.
    
    Args:
        config: Configuration object with get method
    """
    # Log data parameters
    mlflow.log_param("seed", config.get("data.seed"))
    
    # Log model parameters
    mlflow.log_param("model_name", config.get("model.name"))
    mlflow.log_param("num_classes", config.get("model.num_classes"))
    mlflow.log_param("pretrained", config.get("model.pretrained"))
    
    # Log training parameters
    mlflow.log_param("learning_rate", config.get("training.learning_rate"))
    mlflow.log_param("batch_size", config.get("training.batch_size"))
    mlflow.log_param("num_epochs", config.get("training.num_epochs"))
    mlflow.log_param("optimizer", config.get("training.optimizer"))
    
    # Log preprocessing parameters
    mlflow.log_param("spatial_size", config.get("preprocessing.spatial_size"))


def log_metrics(metrics_dict, step=None):
    """
    Log metrics to MLflow.
    
    Args:
        metrics_dict: Dictionary of metrics to log
        step: Optional step for the metrics
    """
    mlflow.log_metrics(metrics_dict, step=step)
