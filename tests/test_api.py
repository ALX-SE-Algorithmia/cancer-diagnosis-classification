"""
Tests for the FastAPI application.
"""
import os
import pytest
import tempfile
from fastapi.testclient import TestClient
import numpy as np
import torch
from pathlib import Path

from api.main import app


# Create a test client
client = TestClient(app)


def test_health_endpoint():
    """Test the health endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": False}


def test_load_model_endpoint():
    """Test the load model endpoint."""
    # Skip this test if the dummy model doesn't exist
    model_path = "models/checkpoints/dummy_model.pth"
    if not os.path.exists(model_path):
        pytest.skip(f"Dummy model not found at {model_path}")
    
    # Test loading the model
    response = client.post(
        "/api/v1/load-model",
        json={
            "model_path": model_path,
            "config_path": "config/test_config.yaml",
            "class_names": {0: "Normal", 1: "Cancer"}
        }
    )
    assert response.status_code == 200
    assert response.json() == {"status": "Model loaded successfully"}
