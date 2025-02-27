import os
import re
from dataclasses import dataclass

@dataclass
class StudentInfo:
    id: str
    full_name: str
    folder_name: str

def parse_student_folder(folder_name):
    """Parse student information from folder name format: {id}_{first_name}_{last_name}"""
    try:
        # Split the folder name into components
        parts = folder_name.split('_')
        if len(parts) < 2:
            raise ValueError(f"Invalid folder name format: {folder_name}")
            
        student_id = parts[0]
        # Join the rest as the full name
        full_name = ' '.join(parts[1:])
        
        return StudentInfo(
            id=student_id,
            full_name=full_name,
            folder_name=folder_name
        )
    except Exception as e:
        raise ValueError(f"Error parsing folder name {folder_name}: {str(e)}")

def get_all_students(dataset_path="dataset"):
    """Get all students information from the dataset directory"""
    students = {}
    
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            try:
                student = parse_student_folder(folder_name)
                students[student.folder_name] = student
            except ValueError as e:
                print(f"Warning: {str(e)}")
    
    return students

def validate_image_name(image_name, student_id):
    """Validate image name format: {student_id}_{number}.jpg"""
    pattern = f"^{student_id}_\\d+\\.(jpg|jpeg|png)$"
    return bool(re.match(pattern, image_name.lower()))

def validate_dataset_structure(dataset_path="dataset"):
    """Validate the entire dataset structure and naming conventions"""
    errors = []
    
    try:
        students = get_all_students(dataset_path)
        
        for folder_name, student in students.items():
            folder_path = os.path.join(dataset_path, folder_name)
            
            # Check images in the folder
            images = [f for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not images:
                errors.append(f"No images found for student {student.full_name}")
                continue
            
            # Validate each image name
            for img_name in images:
                if not validate_image_name(img_name, student.id):
                    errors.append(
                        f"Invalid image name format: {img_name} in {folder_name}. "
                        f"Should be: {student.id}_number.jpg"
                    )
    
    except Exception as e:
        errors.append(f"Error validating dataset: {str(e)}")
    
    return errors

if __name__ == "__main__":
    print("Validating dataset structure...")
    errors = validate_dataset_structure()
    
    if errors:
        print("\nFound following issues in the dataset:")
        for error in errors:
            print(f"- {error}")
        print("\nPlease fix these issues before proceeding with training.")
    else:
        print("\nDataset structure is valid!")
        students = get_all_students()
        print("\nFound following students:")
        for student in students.values():
            print(f"ID: {student.id}, Name: {student.full_name}") 