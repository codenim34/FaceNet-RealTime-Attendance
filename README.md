# Face Recognition Attendance System

This project implements a real-time face recognition attendance system using FaceNet and MTCNN. It's designed to work with a small dataset of 30 students and uses transfer learning for optimal performance.

## Features

- Face detection using MTCNN
- Face recognition using fine-tuned FaceNet model
- Real-time attendance tracking with student IDs
- CSV export of attendance records
- High accuracy with small dataset
- Confidence threshold to prevent false positives

## Dataset Structure

The dataset must follow this specific structure:

```
dataset/
├── 001_John_Smith/
│   ├── 001_01.jpg
│   ├── 001_02.jpg
│   └── 001_03.jpg
├── 002_Emma_Watson/
│   ├── 002_01.jpg
│   ├── 002_02.jpg
│   └── 002_03.jpg
└── 003_Michael_Brown/
    ├── 003_01.jpg
    ├── 003_02.jpg
    └── 003_03.jpg
```

### Naming Conventions:

1. **Student Folders**:

   - Format: `{student_id}_{first_name}_{last_name}`
   - Example: `001_John_Smith`
   - Student ID should be unique
   - Use underscore (\_) to separate words

2. **Image Files**:
   - Format: `{student_id}_{image_number}.jpg`
   - Example: `001_01.jpg`, `001_02.jpg`
   - Must match the student ID of the folder
   - Supported formats: .jpg, .jpeg, .png

You can validate your dataset structure by running:

```bash
python src/student_info.py
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare your dataset following the structure above:

   - Create the `dataset` folder
   - Create student folders with proper ID_Name format
   - Add 5-10 face images per student with proper naming
   - Run the validation script to check your dataset

3. Preprocess the dataset:

```bash
python src/preprocess.py
```

4. Train the model:

```bash
python src/train.py
```

5. Run the attendance system:

```bash
python src/attendance.py
```

## Project Structure

```
.
├── dataset/                  # Raw face images with ID_Name structure
├── processed_dataset/        # Preprocessed face images
├── models/                   # Trained model files
├── attendance/              # Attendance records (CSV)
├── src/
│   ├── student_info.py      # Dataset structure validation
│   ├── preprocess.py        # Face detection and cropping
│   ├── train.py            # Model training
│   └── attendance.py       # Real-time attendance system
└── requirements.txt         # Python dependencies
```

## Attendance Records

The system generates daily attendance records in CSV format with:

- Student ID
- Full Name
- Time of attendance

Example:

```csv
Student ID,Name,Time
001,John Smith,09:15:23
002,Emma Watson,09:16:45
```

## Usage

1. **Data Collection**:

   - Collect 5-10 clear face images of each student
   - Name folders and images according to the specified format
   - Validate the dataset structure using student_info.py
   - Ensure good lighting and different angles

2. **Preprocessing**:

   - Run the preprocessing script to detect and crop faces
   - This creates standardized face images for training

3. **Training**:

   - Run the training script to fine-tune the FaceNet model
   - Training progress will be displayed
   - The model will be saved in the `models` directory

4. **Attendance**:
   - Run the attendance system
   - It will open your webcam and start detecting faces
   - Shows student ID and name with confidence score
   - Press 'q' to quit
   - Attendance records are saved in CSV format

## Notes

- The system requires a CUDA-capable GPU for optimal performance
- Minimum of 5 images per student recommended
- Keep the confidence threshold at 0.8 to prevent false positives
- The system waits 3 seconds before marking the same person again

## Troubleshooting

1. If face detection is poor:

   - Ensure good lighting
   - Keep face clearly visible
   - Try different angles

2. If recognition is inaccurate:

   - Add more training images
   - Ensure image quality
   - Retrain the model

3. If the system is slow:
   - Check GPU availability
   - Reduce frame resolution
   - Update GPU drivers
