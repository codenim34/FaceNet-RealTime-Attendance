# FaceNet Real-Time Attendance System

## Project Overview

A machine learning-based attendance system using face recognition technology, implemented with PyTorch and OpenCV. This project leverages FaceNet (Inception-ResNet-V1) to create an intelligent attendance tracking solution.

## Project Structure

```
FaceNet-RealTime-Attendance/
│
├── dataset/                   # Student face image directories
│   ├── 210042106_Adib_Sakhawat/
│   ├── 210042107_Md_Hasibur_Rahman_Alif/
│   └── ... (individual student directories)
│
├── attendance.ipynb           # Real-time attendance recognition notebook
├── train.ipynb                # Model training notebook
├── face_classifier.pkl        # Trained face classification model
└── other supporting files
```

## Key Components

### 1. Face Recognition
- Uses `facenet_pytorch` library
- MTCNN for face detection
- InceptionResnetV1 pre-trained on VGGFace2

### 2. Notebooks
- `attendance.ipynb`: Real-time attendance system
- `train.ipynb`: Model training pipeline

### 3. Dataset
- Contains individual student directories
- Each directory named with student ID and full name
- Stores student-specific face images for training

## Prerequisites

- Python 3.8+
- PyTorch
- facenet_pytorch
- OpenCV
- Jupyter Notebook

## Installation

1. Clone the repository
2. Install required dependencies:
```bash
pip install torch torchvision facenet_pytorch opencv-python jupyter
```

## Usage

1. Prepare your dataset in the `dataset/` directory
2. Run `train.ipynb` to train the face recognition model
3. Use `attendance.ipynb` to perform real-time attendance tracking

## Technical Details

- **Model**: FaceNet (Inception-ResNet-V1)
- **Face Detection**: MTCNN
- **Pre-trained Weights**: VGGFace2
- **Model Serialization**: Pickle (.pkl)


