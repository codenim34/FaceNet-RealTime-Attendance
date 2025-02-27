import cv2
import torch
import torch.nn as nn
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
from torchvision import transforms
import datetime
import csv
import os
from student_info import get_all_students, StudentInfo

class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionModel, self).__init__()
        # Load the pretrained FaceNet model
        self.backbone = InceptionResnetV1(pretrained='vggface2', classify=False)
        
        # Add our own classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # Get embeddings from the backbone
        embeddings = self.backbone(x)
        # Apply our classifier
        x = self.classifier(embeddings)
        return x

class FaceRecognitionSystem:
    def __init__(self, model_path='models/best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load MTCNN for face detection
        self.mtcnn = MTCNN(
            image_size=160,
            margin=40,       # Increased margin for better face capture
            keep_all=False,
            min_face_size=80,# Increased minimum face size
            post_process=True,
            device=self.device
        )
        
        # Load the trained model
        print("Loading model...")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.class_names = checkpoint['class_names']
        
        # Load student information
        print("Loading student information...")
        self.students = get_all_students()
        
        # Initialize the model
        self.model = FaceRecognitionModel(len(self.class_names))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully! Number of classes: {len(self.class_names)}")
        
        # Initialize attendance dictionary
        self.attendance = {}
        self.last_detection = {}
        
    def get_student_info(self, folder_name):
        return self.students.get(folder_name)
        
    def recognize_face(self, frame):
        # Convert frame to RGB (MTCNN expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Detect and align face
            face = self.mtcnn(Image.fromarray(frame_rgb))
            
            if face is None:
                return None, None, None
                
            # The face is already a tensor and normalized by MTCNN
            face = face.unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                # Get embeddings from backbone
                embeddings = self.model.backbone(face)
                
                # Get classifier output
                output = self.model.classifier(embeddings)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # Check if the highest probability is significantly higher than the second highest
                sorted_probs, _ = torch.sort(probabilities, dim=1, descending=True)
                prob_diff = sorted_probs[0][0] - sorted_probs[0][1]
                
                # Only accept if:
                # 1. Confidence is very high (>0.92)
                # 2. The difference between top two probabilities is significant (>0.5)
                if confidence.item() > 0.92 and prob_diff > 0.5:
                    folder_name = self.class_names[predicted.item()]
                    student = self.get_student_info(folder_name)
                    if student:
                        return student, confidence.item(), folder_name
        except Exception as e:
            print(f"Error in face recognition: {str(e)}")
        
        return None, None, None
    
    def mark_attendance(self, student: StudentInfo, folder_name: str):
        current_time = datetime.datetime.now()
        
        # Check if enough time has passed since last detection (30 seconds)
        if folder_name in self.last_detection:
            time_diff = (current_time - self.last_detection[folder_name]).total_seconds()
            if time_diff < 30:  # Increased from 3 to 30 seconds
                return
        
        self.attendance[folder_name] = {
            'student_id': student.id,
            'name': student.full_name,
            'time': current_time
        }
        self.last_detection[folder_name] = current_time
        
        # Save attendance to CSV
        self.save_attendance()
        
        # Print real-time notification
        print(f"\nMarked attendance for {student.id}: {student.full_name} at {current_time.strftime('%H:%M:%S')}")
    
    def save_attendance(self):
        os.makedirs('attendance', exist_ok=True)
        date_str = datetime.datetime.now().strftime('%Y-%m-%d')
        filename = f'attendance/attendance_{date_str}.csv'
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Student ID', 'Name', 'Time'])
            for record in self.attendance.values():
                writer.writerow([
                    record['student_id'],
                    record['name'],
                    record['time'].strftime('%H:%M:%S')
                ])
    
    def run(self):
        print("\nStarting Face Recognition Attendance System...")
        print("Press 'q' to quit\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam!")
            return
            
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame!")
                break
            
            # Make a copy for drawing
            display_frame = frame.copy()
            
            # Recognize face
            student, confidence, folder_name = self.recognize_face(frame)
            
            if student is not None:
                # Mark attendance
                self.mark_attendance(student, folder_name)
                
                # Draw student info
                text = f"{student.id}: {student.full_name}"
                conf_text = f"Confidence: {confidence:.2f}"
                
                # Draw background rectangle for better text visibility
                cv2.rectangle(display_frame, (30, 20), (600, 80), (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(display_frame, text, (50, 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display_frame, conf_text, (50, 70),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                # Draw "No face detected" message
                cv2.putText(display_frame, "No face detected", (50, 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Show frame
            cv2.imshow('Face Recognition Attendance System', display_frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\nFinal Attendance Summary:")
        print("-" * 50)
        for record in self.attendance.values():
            print(f"ID: {record['student_id']}")
            print(f"Name: {record['name']}")
            print(f"Time: {record['time'].strftime('%H:%M:%S')}")
            print("-" * 50)
        
        print(f"\nAttendance saved to: attendance/attendance_{datetime.datetime.now().strftime('%Y-%m-%d')}.csv")

if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.run() 