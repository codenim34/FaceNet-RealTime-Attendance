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
        
        # Feature processing
        self.feature_processor = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Dropout(0.7),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5)
        )
        
        # Classifier
        self.classifier = nn.Linear(512, num_classes)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.feature_processor[2].weight)
        nn.init.zeros_(self.feature_processor[2].bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, x, return_embeddings=False):
        # Get backbone features
        embeddings = self.backbone(x)
        
        # L2 normalize embeddings
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Process features
        processed_embeddings = self.feature_processor(embeddings)
        
        if return_embeddings:
            return processed_embeddings
            
        # Get classification output
        return self.classifier(processed_embeddings)

class FaceRecognitionSystem:
    def __init__(self, model_path='models/best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load MTCNN for face detection with more sensitive parameters
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,        # Reduced margin
            min_face_size=50, # Reduced minimum face size
            thresholds=[0.5, 0.6, 0.6],  # Lower detection thresholds
            factor=0.709,
            keep_all=False,
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
            # Detect and align face with bounding boxes
            face, prob = self.mtcnn.detect(Image.fromarray(frame_rgb))
            
            if face is not None:
                # Draw rectangle around the face
                box = face[0]
                box = [int(b) for b in box]
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                
                # Add detection probability
                cv2.putText(frame, f"Detection: {prob[0]:.2f}", 
                          (box[0], box[1] - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Get aligned face
                face = self.mtcnn(Image.fromarray(frame_rgb))
                
                if face is None:
                    cv2.putText(frame, "Face alignment failed", (50, 100),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    return None, None, None
                    
                # The face is already a tensor and normalized by MTCNN
                face = face.unsqueeze(0).to(self.device)
                
                # Normalize using ImageNet stats to match training
                normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
                face = normalize(face)
                
                # Get prediction
                with torch.no_grad():
                    # Get embeddings and process features
                    embeddings = self.model.backbone(face)
                    embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
                    processed_embeddings = self.model.feature_processor(embeddings)
                    
                    # Get classifier output
                    output = self.model.classifier(processed_embeddings)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    
                    # Get top-3 predictions and their probabilities
                    top_probs, top_indices = torch.topk(probabilities, min(3, len(self.class_names)))
                    top_probs = top_probs[0].cpu().numpy()
                    top_indices = top_indices[0].cpu().numpy()
                    
                    # Display top-3 predictions
                    y_offset = 120
                    cv2.putText(frame, "Top matches:", (50, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    for i in range(len(top_indices)):
                        class_name = self.class_names[top_indices[i]]
                        prob = top_probs[i]
                        text = f"{class_name}: {prob:.2f}"
                        y_offset += 30
                        cv2.putText(frame, text, (50, y_offset),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    confidence = top_probs[0]
                    margin = top_probs[0] - top_probs[1] if len(top_probs) > 1 else 1.0
                    
                    # Display thresholds
                    y_offset += 40
                    cv2.putText(frame, f"Confidence threshold (0.3): {confidence:.2f}", 
                              (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                              (0, 255, 0) if confidence > 0.3 else (0, 0, 255), 2)
                    
                    y_offset += 30
                    cv2.putText(frame, f"Margin threshold (0.1): {margin:.2f}", 
                              (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                              (0, 255, 0) if margin > 0.1 else (0, 0, 255), 2)
                    
                    # More lenient recognition criteria for testing
                    if confidence > 0.3 and margin > 0.1:  # Much lower thresholds for testing
                        folder_name = self.class_names[top_indices[0]]
                        student = self.get_student_info(folder_name)
                        if student:
                            return student, confidence, folder_name
                    
            else:
                cv2.putText(frame, "No face detected by MTCNN", (50, 100),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
        except Exception as e:
            print(f"Error in face recognition: {str(e)}")
            cv2.putText(frame, f"Error: {str(e)}", (50, 100),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
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
            
        # Set camera resolution to higher values
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Increased resolution
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame!")
                break
            
            # Make a copy for drawing
            display_frame = frame.copy()
            
            # Recognize face (will also draw face rectangle)
            student, confidence, folder_name = self.recognize_face(display_frame)
            
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