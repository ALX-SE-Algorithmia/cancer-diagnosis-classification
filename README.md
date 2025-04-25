# Cancer Diagnosis Classification Deep Learning Project

![Medical Imaging](https://img.shields.io/badge/Medical%20Imaging-CNN-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![MONAI](https://img.shields.io/badge/MONAI-0.8+-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-red)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![DVC](https://img.shields.io/badge/DVC-Enabled-9668E0)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2)
![DAGsHub](https://img.shields.io/badge/DAGsHub-Integrated-FF69B4)
![License](https://img.shields.io/badge/License-MIT-yellow)

A deep learning-based CNN classifier for cancer diagnosis from medical imaging data. This project leverages state-of-the-art deep learning techniques to assist medical professionals in identifying and classifying cancerous tissues from various imaging modalities.

## 🔬 Project Overview

Early and accurate cancer diagnosis is critical for effective treatment planning and improved patient outcomes. This project aims to develop a robust, high-performance CNN classifier that can analyze medical images (MRI, CT, histopathology, etc.) and identify patterns associated with different types of cancer, potentially enabling earlier diagnosis and more precise classification.

The classifier is built on top of the [MONAI](https://monai.io/) framework, which is specifically designed for deep learning in healthcare imaging, and provides a RESTful API interface using FastAPI for easy integration into clinical workflows.

## 🔍 Key Features

- **Modular Architecture**: Highly modular codebase allowing for easy experimentation with different models, datasets, and training strategies
- **Multiple Backbone Options**: Support for DenseNet, ResNet, and custom CNN architectures
- **3D Medical Image Support**: Full support for 3D volumetric medical images
- **Advanced Preprocessing**: Comprehensive medical image preprocessing pipeline with MONAI transforms
- **Robust Training**: Training with early stopping, learning rate scheduling, and class weighting for imbalanced datasets
- **Detailed Evaluation**: Comprehensive evaluation metrics including accuracy, precision, recall, F1 score, AUC, and confusion matrices
- **Visualization Tools**: Rich visualization capabilities for model predictions and attention maps
- **Flexible Configuration**: YAML-based configuration system for easy experiment management
- **MLOps Integration**: Full MLflow experiment tracking and DVC data/model versioning
- **Reproducible Pipelines**: DVC pipelines for reproducible machine learning workflows
- **Experiment Tracking**: MLflow integration for tracking experiments, metrics, and artifacts
- **Collaboration Platform**: DAGsHub integration for team collaboration and experiment visualization
- **RESTful API**: FastAPI-based REST API for model serving and integration
- **API Client**: Python client for easy interaction with the API
- **Docker Support**: Containerized environment for consistent development and deployment
- **Comprehensive Testing**: Pytest-based test suite for all components

## 📂 Project Structure

```
cancer-diagnosis-classification/
├── .dvc/                        # DVC configuration
├── api/                         # FastAPI application
│   ├── main.py                  # Main API application
│   ├── client.py                # API client
│   ├── Dockerfile               # API Docker configuration
│   └── __init__.py              # Package initialization
├── config/                      # Configuration files
│   ├── default_config.yaml      # Default configuration
│   ├── test_config.yaml         # Test configuration
│   └── params.yaml              # DVC pipeline parameters
├── data/                        # Data directory
│   ├── raw/                     # Raw data (tracked by DVC)
│   └── processed/               # Processed data (tracked by DVC)
├── src/                         # Main package
│   ├── data/                    # Data loading and processing
│   │   └── dataset.py           # Dataset handling
│   ├── models/                  # Model architectures
│   │   └── cnn_classifier.py    # CNN classifier models
│   ├── preprocessing/           # Image preprocessing
│   │   └── transforms.py        # Image transforms
│   ├── training/                # Training utilities
│   │   ├── trainer.py           # Model trainer
│   │   └── trainer_mlflow.py    # Model trainer with MLflow integration
│   ├── evaluation/              # Evaluation metrics
│   │   └── metrics.py           # Evaluation metrics
│   ├── visualization/           # Visualization tools
│   │   └── visualize.py         # Visualization utilities
│   └── utils/                   # Utility functions
│       ├── config.py            # Configuration utilities
│       ├── logger.py            # Logging utilities
│       ├── utils.py             # General utilities
│       ├── mlflow_utils.py      # MLflow utilities
│       └── dvc_mlflow_utils.py  # DVC and MLflow integration utilities
├── scripts/                     # Training and inference scripts
│   ├── train.py                 # Training script
│   ├── train_with_mlflow.py     # Training script with MLflow integration
│   ├── predict.py               # Inference script
│   └── create_dummy_model.py    # Utility to create test models
├── tests/                       # Unit tests
│   ├── test_models.py           # Tests for models
│   ├── test_preprocessing.py    # Tests for preprocessing
│   ├── test_utils.py            # Tests for utilities
│   ├── test_integration.py      # Tests for DVC and MLflow integration
│   └── test_api_simple.py       # Tests for API
├── models/                      # Model checkpoints
│   └── checkpoints/             # Saved model checkpoints
├── dvc.yaml                     # DVC pipeline definition
├── .env.example                 # Example environment variables
├── Dockerfile                   # Docker configuration
├── pyproject.toml               # Project configuration
├── requirements.txt             # Python dependencies
├── pytest.ini                   # Pytest configuration
└── README.md                    # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git
- DAGsHub account (for MLflow and DVC remote storage)
- Docker (optional)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cancer-diagnosis-classification.git
   cd cancer-diagnosis-classification
   ```

2. Create a conda environment and install dependencies:
   ```bash
   conda create -p venv python=3.12 -y
   conda activate ./venv
   pip install -r requirements.txt
   ```

3. Set up DVC and MLflow:
   ```bash
   # Initialize DVC (if not already initialized)
   dvc init
   
   # Configure DVC remote storage on DAGsHub
   dvc remote add -d dagshub https://dagshub.com/yourusername/cancer-diagnosis-classification.dvc
   
   # Create .env file with DAGsHub credentials
   cp .env.example .env
   # Edit .env with your DAGsHub username and token
   ```

### Using Docker

You can also use Docker to run the project:

```bash
# Build the Docker image for training
docker build -t cancer-diagnosis-classifier .

# Run training
docker run --gpus all -v $(pwd)/data:/app/data cancer-diagnosis-classifier

# Build the API Docker image
docker build -t cancer-diagnosis-api -f api/Dockerfile .

# Run the API server
docker run -p 8000:8000 --gpus all -v $(pwd)/models:/app/models cancer-diagnosis-api
```

### Data Preparation

1. Organize your medical imaging data in the following structure:
   ```
   data/raw/
   ├── cancer_type_1/
   │   ├── patient_id_1/
   │   │   ├── image1.nii.gz
   │   │   └── ...
   │   ├── patient_id_2/
   │   └── ...
   ├── cancer_type_2/
   └── ...
   ```

2. Generate metadata CSV file:
   ```bash
   python -c "from src.utils.utils import create_metadata_csv; create_metadata_csv('data/raw', 'data/metadata.csv')"
   ```

### Training

#### Standard Training

To train the model with default configuration:

```bash
python scripts/train.py --config config/default_config.yaml --experiment_name my_experiment
```

You can customize the training by modifying the configuration file or by passing command-line arguments:

```bash
python scripts/train.py --config config/default_config.yaml --experiment_name custom_experiment --data_dir path/to/data --seed 42
```

#### Training with MLflow Tracking

To train the model with MLflow experiment tracking:

```bash
python scripts/train_with_mlflow.py --config config/default_config.yaml --experiment_name mlflow_experiment --dvc
```

This will track your experiment with MLflow and save metrics for DVC.

### Inference

To run inference on new medical images:

```bash
python scripts/predict.py --checkpoint experiments/my_experiment/checkpoints/cancer_diagnosis_classifier_best.pth --input path/to/image_or_directory --visualize
```

### Using the API

The project includes a FastAPI-based REST API for serving predictions:

1. Start the API server:

```bash
cd /path/to/project
PYTHONPATH=/path/to/project uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

2. Use the API client:

```bash
# Check API health
python -m api.client health

# Load a model
python -m api.client load --model models/checkpoints/my_model.pth --config config/default_config.yaml

# Make a prediction
python -m api.client predict --image path/to/image.jpg
```

3. Or use curl directly:

```bash
# Health check
curl -X GET http://localhost:8000/api/v1/health

# Load model
curl -X POST http://localhost:8000/api/v1/load-model \
  -H "Content-Type: application/json" \
  -d '{"model_path": "models/checkpoints/my_model.pth", "config_path": "config/default_config.yaml", "class_names": {"0": "Normal", "1": "Cancer"}}'

# Predict (requires an image file)
curl -X POST http://localhost:8000/api/v1/predict \
  -F "file=@path/to/image.jpg"
```

4. Access the API documentation at `http://localhost:8000/docs`

## 🔄 MLOps Integration

### DVC Data and Model Versioning

This project uses DVC for data and model versioning:

```bash
# Track raw data with DVC
dvc add data/raw

# Track models with DVC
dvc add models/checkpoints

# Push to remote storage
dvc push
```

### MLflow Experiment Tracking

MLflow is integrated for experiment tracking:

```bash
# Run an experiment with MLflow tracking
python scripts/train_with_mlflow.py --config config/params.yaml --experiment_name my_experiment

# View experiments locally (if MLflow server is running)
mlflow ui

# View experiments on DAGsHub
# Visit: https://dagshub.com/yourusername/cancer-diagnosis-classification/experiments
```

### DVC Pipeline

The project includes a DVC pipeline for reproducible workflows:

```bash
# Run the entire pipeline
dvc repro

# Run a specific stage
dvc repro -s train

# View pipeline structure
dvc dag
```

### DAGsHub Integration

The project is integrated with DAGsHub for experiment tracking and collaboration:

1. Create a repository on [DAGsHub](https://dagshub.com)
2. Configure your credentials in the `.env` file
3. Push your code, data, and models to DAGsHub:
   ```bash
   git push origin main
   dvc push
   ```
4. View your experiments on DAGsHub at `https://dagshub.com/yourusername/cancer-diagnosis-classification/experiments`

## 📊 Model Performance

The model's performance will vary based on the specific cancer types in your dataset. With proper training, the system can achieve:

- High sensitivity in detecting cancerous tissues
- Robust performance across different imaging modalities (MRI, CT, histopathology, etc.)
- Ability to distinguish between different cancer types and stages

## 🔬 Technical Details

### CNN Architecture

The default model uses a DenseNet121 backbone, which has been shown to perform well on medical imaging tasks. The architecture includes:

- Pretrained weights (optional)
- Spatial dimensions support (2D or 3D)
- Customizable dropout for regularization
- Fully connected classification layers

Alternative architectures include ResNet and a custom CNN implementation that can be selected via configuration.

### Data Augmentation

The training pipeline includes a comprehensive set of data augmentation techniques specifically designed for medical imaging:

- Random rotations, flips, and zooms
- Intensity adjustments
- Gaussian noise addition
- Contrast adjustments
- Spatial transformations

### Training Strategy

The training process incorporates several best practices:

- Early stopping to prevent overfitting
- Learning rate scheduling (ReduceLROnPlateau, CosineAnnealing, Step)
- Class weighting for imbalanced datasets
- Comprehensive metrics tracking with TensorBoard
- Checkpoint saving for best models

### API Architecture

The FastAPI application provides a RESTful interface for model serving:

- Health check endpoint
- Model loading endpoint with configuration
- Prediction endpoint supporting various medical image formats
- Swagger documentation
- Python client for easy integration
- Docker containerization

## 🧪 Testing

The project includes a comprehensive test suite using pytest:

```bash
# Run all tests
python -m pytest

# Run specific test modules
python -m pytest tests/test_models.py
python -m pytest tests/test_utils.py

# Run with verbose output
python -m pytest -v
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact the project maintainers.

---

*This project aims to assist medical professionals in the diagnosis of cancer. It is not intended to replace clinical judgment or provide medical advice.*
