import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from facenet_pytorch import InceptionResnetV1
import os
from tqdm import tqdm
import numpy as np

class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionModel, self).__init__()
        # Load the pretrained FaceNet model
        self.backbone = InceptionResnetV1(pretrained='vggface2', classify=False)
        
        # Freeze most of the backbone layers
        for param in list(self.backbone.parameters())[:-10]:  # Only fine-tune last few layers
            param.requires_grad = False
        
        # Simplified classifier with stronger regularization
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Reduced dropout
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # Get embeddings from the backbone
        embeddings = self.backbone(x)
        # Apply our classifier
        x = self.classifier(embeddings)
        return x

def create_data_loaders(data_dir="processed_dataset", batch_size=32, val_split=0.2):
    # Define image transformations with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load full dataset
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    
    # Calculate split sizes
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    # Split dataset
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Override validation transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, full_dataset.classes

def save_checkpoint(state, filename):
    """Helper function to save model checkpoint"""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return val_loss/len(val_loader), 100 * correct/total

def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    
    # Separate parameter groups for different learning rates
    backbone_params = list(model.backbone.parameters())[-10:]  # Only last few layers
    classifier_params = model.classifier.parameters()
    
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': 1e-5},  # Very small learning rate for backbone
        {'params': classifier_params, 'lr': 1e-4}  # Larger learning rate for classifier
    ], weight_decay=1e-4)  # Increased weight decay
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.2,  # More aggressive LR reduction
        patience=3,   # Reduced patience
        verbose=True
    )
    
    model = model.to(device)
    best_val_accuracy = 0.0
    patience_counter = 0
    max_patience = 10  # Reduced early stopping patience
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_accuracy = 100 * correct/total
            progress_bar.set_postfix({
                'loss': f'{running_loss/len(train_loader):.4f}',
                'accuracy': f'{train_accuracy:.2f}%'
            })
        
        # Validation phase
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Training - Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_accuracy)
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'accuracy': val_accuracy,
                'class_names': train_loader.dataset.dataset.classes
            }, 'models/best_model.pth')
            print(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    return model

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if processed dataset exists
    if not os.path.exists('processed_dataset'):
        print("Error: processed_dataset directory not found!")
        print("Please run preprocess.py first: python src/preprocess.py")
        return
    
    # Create data loaders
    train_loader, val_loader, class_names = create_data_loaders()
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    
    # Create model
    model = FaceRecognitionModel(num_classes)
    
    # Train the model
    model = train_model(model, train_loader, val_loader, device=device)
    
    print("Training completed!")

if __name__ == "__main__":
    main() 