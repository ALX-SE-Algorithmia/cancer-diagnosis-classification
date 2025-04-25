"""
FastAPI application for cancer diagnosis classification.
"""
import io
import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import nibabel as nib
import SimpleITK as sitk
from PIL import Image

from src.models.cnn_classifier import CNNClassifier, CustomCNNClassifier
from src.preprocessing.transforms import get_inference_transforms
from src.utils.config import Config
from src.utils.utils import get_device

# Initialize FastAPI app
app = FastAPI(
    title="Cancer Diagnosis Classification API",
    description="API for cancer diagnosis classification from medical images",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL = None
DEVICE = None
TRANSFORMS = None
CLASS_NAMES = None
CONFIG = None


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    
    predicted_class: int
    class_name: str
    confidence: float
    probabilities: Dict[str, float]


class ModelConfig(BaseModel):
    """Model configuration."""
    
    model_path: str
    config_path: str
    class_names: Optional[Dict[int, str]] = None


@app.on_event("startup")
async def startup_event():
    """Initialize model and device on startup."""
    global DEVICE
    DEVICE = get_device()
    print(f"Using device: {DEVICE}")


@app.post("/api/v1/load-model", response_model=Dict[str, str])
async def load_model(model_config: ModelConfig):
    """
    Load a model from a checkpoint file.
    
    Args:
        model_config: Model configuration
        
    Returns:
        Dictionary with status message
    """
    global MODEL, CONFIG, TRANSFORMS, CLASS_NAMES
    
    try:
        # Load configuration
        CONFIG = Config(model_config.config_path)
        
        # Get transforms
        TRANSFORMS = get_inference_transforms(
            spatial_size=CONFIG.get("preprocessing.spatial_size"),
        )
        
        # Create model
        backbone = CONFIG.get("model.backbone")
        num_classes = CONFIG.get("model.num_classes")
        
        if backbone == "custom":
            MODEL = CustomCNNClassifier(
                in_channels=CONFIG.get("model.in_channels"),
                num_classes=num_classes,
                initial_filters=CONFIG.get("model.initial_filters"),
                dropout_prob=CONFIG.get("model.dropout_prob"),
                spatial_dims=CONFIG.get("model.spatial_dims"),
            )
        else:
            MODEL = CNNClassifier(
                in_channels=CONFIG.get("model.in_channels"),
                num_classes=num_classes,
                backbone=backbone,
                pretrained=False,
                spatial_dims=CONFIG.get("model.spatial_dims"),
                dropout_prob=CONFIG.get("model.dropout_prob"),
            )
        
        # Load checkpoint
        checkpoint = torch.load(model_config.model_path, map_location=DEVICE)
        MODEL.load_state_dict(checkpoint["model_state_dict"])
        MODEL.to(DEVICE)
        MODEL.eval()
        
        # Set class names
        CLASS_NAMES = model_config.class_names or {i: f"Class {i}" for i in range(num_classes)}
        
        return {"status": "Model loaded successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict cancer diagnosis from a medical image.
    
    Args:
        file: Uploaded medical image file
        
    Returns:
        Prediction response
    """
    if MODEL is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Call /api/v1/load-model first.")
    
    try:
        # Read file content
        content = await file.read()
        
        # Determine file type and load image
        filename = file.filename.lower()
        
        if filename.endswith((".nii", ".nii.gz")):
            # NIfTI file
            nib_file = nib.load(io.BytesIO(content))
            image_data = nib_file.get_fdata()
        elif filename.endswith((".dcm")):
            # DICOM file
            reader = sitk.ImageFileReader()
            reader.SetFileName(io.BytesIO(content))
            image = reader.Execute()
            image_data = sitk.GetArrayFromImage(image)
        elif filename.endswith((".jpg", ".jpeg", ".png")):
            # Regular image file
            image = Image.open(io.BytesIO(content))
            image_data = np.array(image)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Preprocess image
        image_dict = {"image": image_data}
        processed_image = TRANSFORMS(image_dict)["image"]
        
        # Add batch dimension
        processed_image = processed_image.unsqueeze(0).to(DEVICE)
        
        # Make prediction
        with torch.no_grad():
            outputs = MODEL(processed_image)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Get class name
        class_name = CLASS_NAMES.get(predicted_class, f"Class {predicted_class}")
        
        # Create probabilities dictionary
        probs_dict = {
            CLASS_NAMES.get(i, f"Class {i}"): prob.item()
            for i, prob in enumerate(probabilities[0])
        }
        
        return {
            "predicted_class": predicted_class,
            "class_name": class_name,
            "confidence": confidence,
            "probabilities": probs_dict,
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/api/v1/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Dictionary with status
    """
    return {"status": "healthy", "model_loaded": MODEL is not None}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
