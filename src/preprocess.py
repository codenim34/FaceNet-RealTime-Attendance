from facenet_pytorch import MTCNN
from PIL import Image, ImageEnhance
import os
import torch
from tqdm import tqdm
import numpy as np
import random
import cv2
from torchvision import transforms
import albumentations as A

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

def apply_augmentations(image):
    # Convert PIL to numpy for albumentations
    image_np = np.array(image)
    
    # Create augmentation pipeline
    transform = A.Compose([
        # Lighting and color augmentations
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.7),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        
        # Geometric augmentations
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.7),
        A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.3),
        
        # Weather and environmental augmentations
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.3),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),
        
        # Blur and sharpness
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=(3, 5), p=1.0),
            A.ImageCompression(quality_lower=80, quality_upper=100, p=1.0),
        ], p=0.3),
    ])
    
    # Apply augmentations
    augmented = transform(image=image_np)
    return Image.fromarray(augmented['image'])

def create_variations(image, num_variations=10):
    variations = []
    for _ in range(num_variations):
        # Apply random augmentations
        augmented_image = apply_augmentations(image)
        variations.append(augmented_image)
    return variations

def crop_face(mtcnn, image_path, save_path_template, num_variations=10):
    try:
        # Read image
        image = Image.open(image_path).convert("RGB")
        
        # Get face and probability
        face, prob = mtcnn(image, return_prob=True)
        
        if face is not None and prob > 0.95:
            # Convert to PIL Image
            face_pil = Image.fromarray(face.mul(255).permute(1, 2, 0).byte().numpy())
            
            # Create variations
            variations = create_variations(face_pil, num_variations)
            
            # Save original and variations
            success_count = 0
            
            # Save original
            original_path = save_path_template.format(0)
            face_pil.save(original_path)
            success_count += 1
            
            # Save variations
            for i, var_image in enumerate(variations, 1):
                try:
                    # Check image quality
                    img_np = np.array(var_image)
                    brightness = np.mean(img_np)
                    contrast = np.std(img_np)
                    
                    if brightness >= 30 and brightness <= 225 and contrast >= 20:
                        var_path = save_path_template.format(i)
                        var_image.save(var_path)
                        success_count += 1
                except Exception as e:
                    print(f"Error saving variation {i}: {str(e)}")
                    
            return success_count
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
    return 0

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
            save_path_template = os.path.join(output_person_dir, f"face_{idx+1}_var_{{}}.jpg")
            
            # Process original and create variations
            success_count = crop_face(mtcnn, input_path, save_path_template)
            successful += success_count
            processed_images += success_count
        
        # Check if we have enough good quality images
        if successful < 50:  # Minimum required faces (5 originals * 10 variations)
            print(f"Warning: Only {successful} good quality faces found for {person}")
    
    print(f"\nProcessing completed!")
    print(f"Total original images: {total_images}")
    print(f"Total processed images (with variations): {processed_images}")
    print(f"Average variations per image: {processed_images/total_images:.1f}")

if __name__ == "__main__":
    print("Starting face detection, cropping, and augmentation...")
    process_dataset()
    print("\nPreprocessing completed!") 