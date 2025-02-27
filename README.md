# FaceNet Real-Time Attendance System

A real-time face recognition attendance system built with FaceNet (Inception-ResNet-V1), PyTorch, and OpenCV, leveraging fine-tuned VGGFace2 pretrained model with custom feature processing and data augmentation pipeline.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

### Core Functionality

- Real-time face detection using MTCNN
- Face recognition with fine-tuned FaceNet model
- Automated attendance logging with CSV export
- Student information management system
- Real-time visual feedback and confidence scores

### Technical Features

- Custom feature processing pipeline
- Multi-loss training (Softmax + Center Loss + Triplet Loss)
- Extensive data augmentation for robustness
- Early stopping and learning rate scheduling
- Model checkpointing and best model saving

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- Webcam for attendance capture

### Setup

1. Clone the repository:

```bash
git clone [repository-url]
cd FaceNet-RealTime-Attendance
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Automated Pipeline

```bash
python run.py  # Runs complete pipeline
```

### Manual Setup

1. **Preprocess Images**

```bash
python src/preprocess.py
```

2. **Train Model**

```bash
python src/train.py
```

3. **Run Attendance System**

```bash
python src/attendance.py
```

## Technical Details

### Model Architecture

- **Base Model**: FaceNet (Inception-ResNet-V1)
- **Pretrained**: VGGFace2
- **Custom Layers**:
  - Feature Processing Pipeline
  - Batch Normalization
  - Dropout (0.7, 0.5)
  - Custom Classifier

### Training Parameters

- Batch Size: 16
- Learning Rate: 1e-3 (feature processor), 1e-4 (backbone)
- Weight Decay: 5e-4
- Validation Split: 20%
- Early Stopping Patience: 8 epochs

### Loss Functions

- Cross Entropy Loss (main classification)
- Center Loss (feature clustering)
- Triplet Loss (margin: 0.3)

### Data Augmentation

- Geometric: rotation, scale, shift
- Lighting: brightness, contrast, gamma
- Environmental: shadows, fog
- Noise: Gaussian noise, blur
- Color: hue, saturation, value

### Recognition Parameters

- Confidence Threshold: 0.3
- Margin Threshold: 0.1
- Detection Cooldown: 30 seconds
- Face Detection Parameters:
  - Margin: 20
  - Min Face Size: 50
  - MTCNN Thresholds: [0.5, 0.6, 0.6]

## Project Structure

```
.
├── dataset/                  # Raw student images
├── processed_dataset/        # Preprocessed face images
├── models/                   # Trained models
│   └── best_model.pth        # Best model checkpoint
├── attendance/               # Attendance records
├── logs/                     # Preprocessing and Training logs
├── src/
│   ├── preprocess.py       # Face detection & augmentation
│   ├── train.py           # Model training pipeline
│   ├── attendance.py      # Real-time recognition system
│   └── student_info.py    # Student data management
├── run.py                  # Automated pipeline
└── requirements.txt        # Dependencies
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
