# FaceNet Real-Time Attendance System

A real-time face recognition attendance system using FaceNet and PyTorch. The system uses a pre-trained FaceNet model fine-tuned on your student dataset for accurate face recognition.

## Quick Start (Automated)

The easiest way to set up and train the system is using the automated pipeline:

```bash
python run.py
```

This will:

1. Clean previous processed data and models
2. Preprocess all student images
3. Train the face recognition model
4. Save the best model automatically

After training completes, run the attendance system:

```bash
python src/attendance.py
```

## Manual Setup (Step by Step)

If you prefer more control, you can run each step manually:

1. **Prepare Dataset**

   ```
   dataset/
   ├── 210042106_Student_Name/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ├── 210042107_Another_Student/
   │   ├── image1.jpg
   │   └── ...
   └── ...
   ```

   - Add 5-10 clear face images per student
   - Name folders as: `StudentID_Full_Name`
   - Supported formats: `.jpg`, `.jpeg`, `.png`

2. **Preprocess Images**

   ```bash
   python src/preprocess.py
   ```

   - Detects and aligns faces
   - Creates standardized face images in `processed_dataset/`

3. **Train Model**

   ```bash
   python src/train.py
   ```

   - Fine-tunes FaceNet on your dataset
   - Saves best model to `models/best_model.pth`
   - Training progress shown in real-time

4. **Run Attendance System**
   ```bash
   python src/attendance.py
   ```
   - Opens webcam for real-time recognition
   - Shows student ID and name
   - Saves attendance to CSV files

## System Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- Webcam for attendance

## Dependencies

```bash
pip install -r requirements.txt
```

## Model Architecture

- Base: FaceNet (Inception-ResNet-V1)
- Pretrained on VGGFace2
- Fine-tuned classifier for student recognition
- Optimized for real-time performance

## Training Details

- Data augmentation for better generalization
- Early stopping to prevent overfitting
- Learning rate scheduling
- Validation split: 20%
- Batch size: 32

## Attendance Features

- Real-time face detection and recognition
- Confidence threshold: 0.92
- 30-second cooldown between recognitions
- Daily attendance logs in CSV format
- Student information display
- Easy to use interface

## Troubleshooting

1. **Poor Recognition**

   - Ensure good lighting
   - Keep face clearly visible
   - Add more training images
   - Retrain the model

2. **Training Issues**

   - Check GPU availability
   - Verify dataset structure
   - Ensure clean dataset
   - Monitor validation accuracy

3. **System Performance**
   - Lower resolution if needed
   - Update GPU drivers
   - Close other GPU applications

## File Structure

```
.
├── dataset/                  # Raw student images
├── processed_dataset/        # Preprocessed face images
├── models/                   # Trained models
├── attendance/              # Attendance records
├── logs/                    # Training logs
├── src/
│   ├── preprocess.py       # Face detection & alignment
│   ├── train.py           # Model training
│   └── attendance.py      # Real-time recognition
├── run.py                  # Automated pipeline
└── requirements.txt        # Dependencies
```

## Best Practices

1. **Image Collection**

   - Use clear, well-lit photos
   - Include different angles
   - Avoid extreme expressions
   - Minimum 5 images per student

2. **Training**

   - Let training complete naturally
   - Monitor validation accuracy
   - Save best performing model
   - Use GPU if available

3. **Attendance**
   - Good lighting conditions
   - Face camera directly
   - Keep steady position
   - Check confidence scores

## License

This project is licensed under the MIT License - see the LICENSE file for details.
