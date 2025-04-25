"""
Client for the Cancer Diagnosis Classification API.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder


class CancerDiagnosisClient:
    """Client for the Cancer Diagnosis Classification API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url
        self.headers = {"Accept": "application/json"}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if the API is healthy.
        
        Returns:
            Dictionary with health status
        """
        response = requests.get(f"{self.base_url}/api/v1/health", headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def load_model(
        self,
        model_path: str,
        config_path: str,
        class_names: Optional[Dict[int, str]] = None,
    ) -> Dict[str, str]:
        """
        Load a model.
        
        Args:
            model_path: Path to model checkpoint
            config_path: Path to configuration file
            class_names: Dictionary mapping class indices to names
            
        Returns:
            Dictionary with status message
        """
        payload = {
            "model_path": model_path,
            "config_path": config_path,
            "class_names": class_names,
        }
        
        response = requests.post(
            f"{self.base_url}/api/v1/load-model",
            headers=self.headers,
            json=payload,
        )
        response.raise_for_status()
        return response.json()
    
    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Predict cancer diagnosis from a medical image.
        
        Args:
            image_path: Path to medical image
            
        Returns:
            Dictionary with prediction results
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Create multipart form data
        with open(image_path, "rb") as f:
            form = MultipartEncoder(
                fields={"file": (image_path.name, f, "application/octet-stream")}
            )
            
            headers = {
                "Content-Type": form.content_type,
                "Accept": "application/json",
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/predict",
                headers=headers,
                data=form,
            )
            
        response.raise_for_status()
        return response.json()


def main():
    """Command-line interface for the client."""
    parser = argparse.ArgumentParser(description="Cancer Diagnosis Classification API Client")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Health check command
    health_parser = subparsers.add_parser("health", help="Check API health")
    health_parser.add_argument(
        "--url", type=str, default="http://localhost:8000",
        help="Base URL of the API"
    )
    
    # Load model command
    load_parser = subparsers.add_parser("load", help="Load a model")
    load_parser.add_argument(
        "--url", type=str, default="http://localhost:8000",
        help="Base URL of the API"
    )
    load_parser.add_argument(
        "--model", type=str, required=True,
        help="Path to model checkpoint"
    )
    load_parser.add_argument(
        "--config", type=str, required=True,
        help="Path to configuration file"
    )
    load_parser.add_argument(
        "--class-names", type=str, default=None,
        help="Path to JSON file with class names"
    )
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict cancer diagnosis")
    predict_parser.add_argument(
        "--url", type=str, default="http://localhost:8000",
        help="Base URL of the API"
    )
    predict_parser.add_argument(
        "--image", type=str, required=True,
        help="Path to medical image"
    )
    
    args = parser.parse_args()
    
    # Create client
    client = CancerDiagnosisClient(base_url=args.url)
    
    if args.command == "health":
        # Health check
        result = client.health_check()
        print(json.dumps(result, indent=2))
    
    elif args.command == "load":
        # Load model
        class_names = None
        if args.class_names:
            with open(args.class_names, "r") as f:
                class_names = json.load(f)
        
        result = client.load_model(
            model_path=args.model,
            config_path=args.config,
            class_names=class_names,
        )
        print(json.dumps(result, indent=2))
    
    elif args.command == "predict":
        # Predict
        result = client.predict(image_path=args.image)
        print(json.dumps(result, indent=2))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
