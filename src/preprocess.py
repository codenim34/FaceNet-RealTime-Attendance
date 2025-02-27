from facenet_pytorch import MTCNN
from PIL import Image
import os
import torch
from tqdm import tqdm
import numpy as np

def create_mtcnn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return MTCNN(
        image_size=160,
        margin=20,        # Reduced margin for tighter crops
        min_face_size=40, # Smaller min face size to detect more faces
        thresholds=[0.6, 0.7, 0.7],  # More stringent detection thresholds
        factor=0.709,     # For better accuracy
        post_process=True,
        device=device
    )

def crop_face(mtcnn, image_path, save_path):
    try:
        # Read image
        image = Image.open(image_path).convert("RGB")
        
        # Get both face and probability
        face, prob = mtcnn(image, return_prob=True)
        
        if face is not None and prob > 0.95:  # Only keep high confidence detections
            # Convert to PIL Image
            face = Image.fromarray(face.mul(255).permute(1, 2, 0).byte().numpy())
            
            # Check face image quality
            face_np = np.array(face)
            
            # Check brightness
            brightness = np.mean(face_np)
            if brightness < 30 or brightness > 225:  # Too dark or too bright
                return False
                
            # Check contrast
            contrast = np.std(face_np)
            if contrast < 20:  # Too low contrast
                return False
            
            # Save the face
            face.save(save_path)
            return True
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
    return False

def process_dataset(input_dir="dataset", output_dir="processed_dataset"):
    mtcnn = create_mtcnn()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Track statistics
    total_images = 0
    processed_images = 0
    
    # Process each student's directory
    for person in os.listdir(input_dir):
        person_dir = os.path.join(input_dir, person)
        if not os.path.isdir(person_dir):
            continue
            
        # Create output directory for this person
        output_person_dir = os.path.join(output_dir, person)
        os.makedirs(output_person_dir, exist_ok=True)
        
        # Process each image
        images = [f for f in os.listdir(person_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg')) 
                 and not f.startswith('.')]
        
        if not images:
            print(f"\nNo images found in {person_dir}")
            continue
        
        total_images += len(images)
        successful = 0
        
        print(f"\nProcessing {len(images)} images for {person}")
        
        for idx, img_name in enumerate(tqdm(images)):
            input_path = os.path.join(person_dir, img_name)
            output_path = os.path.join(output_person_dir, f"face_{idx+1}.jpg")
            
            if crop_face(mtcnn, input_path, output_path):
                successful += 1
                processed_images += 1
        
        # Check if we have enough good quality images
        if successful < 5:  # Minimum required faces
            print(f"Warning: Only {successful} good quality faces found for {person}")
    
    print(f"\nProcessing completed!")
    print(f"Total images processed: {total_images}")
    print(f"Successfully detected faces: {processed_images}")
    print(f"Success rate: {processed_images/total_images*100:.1f}%")

if __name__ == "__main__":
    print("Starting face detection and cropping...")
    process_dataset()
    print("\nPreprocessing completed!") 