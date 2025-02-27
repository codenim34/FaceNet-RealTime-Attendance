import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1
import os
from tqdm import tqdm

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

def create_data_loaders(data_dir="processed_dataset", batch_size=32):
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    # Create data loader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    return loader, dataset.classes

def save_checkpoint(state, filename):
    """Helper function to save model checkpoint"""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)

def train_model(model, train_loader, num_epochs=10, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model = model.to(device)
    best_accuracy = 0.0
    
    # Create models directory at the start
    os.makedirs('models', exist_ok=True)
    
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
            accuracy = 100 * correct/total
            progress_bar.set_postfix({
                'loss': f'{running_loss/len(train_loader):.4f}',
                'accuracy': f'{accuracy:.2f}%'
            })
        
        # Print epoch statistics
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Loss: {running_loss/len(train_loader):.4f}")
        print(f"Accuracy: {accuracy:.2f}%")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'class_names': train_loader.dataset.classes
            }, 'models/best_model.pth')
            print(f"New best model saved with accuracy: {accuracy:.2f}%")
    
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
    train_loader, class_names = create_data_loaders()
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    
    # Create model
    model = FaceRecognitionModel(num_classes)
    
    # Train the model
    model = train_model(model, train_loader, device=device)
    
    # Save the final model
    save_checkpoint({
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }, 'models/face_recognition_model.pth')
    print("Final model saved successfully!")

if __name__ == "__main__":
    main() 