"""
Simple tests for the FastAPI application.
"""
import pytest
from fastapi.testclient import TestClient

from api.main import app


# Create a test client
client = TestClient(app)


def test_health_endpoint():
    """Test the health endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": False}
