from facenet_pytorch import MTCNN
from PIL import Image
import os
import torch
from tqdm import tqdm

def create_mtcnn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return MTCNN(
        keep_all=False,
        device=device
    )

def crop_face(mtcnn, image_path, save_path):
    try:
        image = Image.open(image_path).convert("RGB")
        face = mtcnn(image)
        if face is not None:
            face = Image.fromarray(face.mul(255).permute(1, 2, 0).byte().numpy())
            face.save(save_path)
            return True
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
    return False

def process_dataset(input_dir="dataset", output_dir="processed_dataset"):
    mtcnn = create_mtcnn()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
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
                 and not f.startswith('.')]  # Skip hidden files
        
        if not images:
            print(f"\nNo images found in {person_dir}")
            continue
            
        print(f"\nProcessing {len(images)} images for {person}")
        
        for idx, img_name in enumerate(tqdm(images)):
            input_path = os.path.join(person_dir, img_name)
            # Use a simple numbered format for processed images
            output_path = os.path.join(output_person_dir, f"face_{idx+1}.jpg")
            
            if not os.path.exists(output_path):
                crop_face(mtcnn, input_path, output_path)

if __name__ == "__main__":
    print("Starting face detection and cropping...")
    process_dataset()
    print("\nPreprocessing completed!") 