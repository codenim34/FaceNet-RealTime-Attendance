import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from facenet_pytorch import InceptionResnetV1
import os
from tqdm import tqdm
import numpy as np

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))
        self.device = device
        
    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.centers.size(0)) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.centers.size(0), batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)
        
        classes = torch.arange(self.centers.size(0)).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.centers.size(0))
        mask = labels.eq(classes.expand(batch_size, self.centers.size(0)))
        
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss

class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionModel, self).__init__()
        # Load the pretrained FaceNet model
        self.backbone = InceptionResnetV1(pretrained='vggface2', classify=False)
        
        # Freeze most backbone layers
        for param in list(self.backbone.parameters())[:-10]:  # Fine-tune last 10 layers
            param.requires_grad = False
        
        # Feature processing
        self.feature_processor = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Dropout(0.7),  # Increased dropout
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

def create_data_loaders(data_dir="processed_dataset", batch_size=16, val_split=0.2):
    train_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.RandomApply([
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.GaussianBlur(3, sigma=(0.1, 0.2)),
        ], p=0.5),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    # Loss functions
    softmax_criterion = nn.CrossEntropyLoss()
    triplet_criterion = TripletLoss(margin=0.3)  # Increased margin
    center_criterion = CenterLoss(
        num_classes=len(train_loader.dataset.dataset.classes),
        feat_dim=512,
        device=device
    )
    
    # Optimizers
    backbone_params = list(model.backbone.parameters())[-10:]
    other_params = list(model.feature_processor.parameters()) + list(model.classifier.parameters())
    
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': 1e-4},
        {'params': other_params, 'lr': 1e-3},
        {'params': center_criterion.parameters(), 'lr': 1e-4}
    ], weight_decay=5e-4)  # Increased weight decay
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.2, patience=4, verbose=True
    )
    
    model = model.to(device)
    best_val_accuracy = 0.0
    patience_counter = 0
    max_patience = 8
    
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
            
            # Forward pass
            embeddings = model(images, return_embeddings=True)
            outputs = model.classifier(embeddings)
            
            # Calculate losses
            softmax_loss = softmax_criterion(outputs, labels)
            center_loss = center_criterion(embeddings, labels)
            
            # Triplet loss
            triplet_loss = 0.0
            if len(torch.unique(labels)) > 1:
                for label in torch.unique(labels):
                    mask = labels == label
                    if torch.sum(mask) >= 2:
                        pos_embeddings = embeddings[mask]
                        neg_embeddings = embeddings[~mask]
                        
                        pos_dists = torch.pdist(pos_embeddings, p=2)
                        if len(pos_dists) > 0:
                            hardest_pos_dist = torch.max(pos_dists)
                            neg_dists = torch.cdist(pos_embeddings, neg_embeddings, p=2)
                            hardest_neg_dist = torch.min(neg_dists)
                            loss = torch.relu(hardest_pos_dist - hardest_neg_dist + 0.3)
                            triplet_loss += loss
            
            # Combined loss with weighted components
            loss = softmax_loss + 0.3 * triplet_loss + 0.1 * center_loss
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_accuracy = 100 * correct/total
            progress_bar.set_postfix({
                'loss': f'{running_loss/len(train_loader):.4f}',
                'accuracy': f'{train_accuracy:.2f}%'
            })
        
        # Validation phase
        val_loss, val_accuracy = validate(model, val_loader, softmax_criterion, device)
        
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