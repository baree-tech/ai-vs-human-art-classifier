#-- AI vs real Art Classifier--
#------created by : Bareera Mushthak

#--------------------------------------------------------------------------------
# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import random

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset paths
data_dir = '/media/bareera/KINGSTON/Bareera/Dataset/AI_vs_Real' #path to dataset
batch_size = 32
image_size = 224

# Define transforms
transform = {
    'train': transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
}

# Load datasets
datasets_ = {
    split: datasets.ImageFolder(os.path.join(data_dir, split), transform=transform[split])
    for split in ['train', 'val', 'test']
}

dataloaders = {
    split: DataLoader(datasets_[split], batch_size=batch_size, shuffle=True)
    for split in ['train', 'val', 'test']
}

class_names = datasets_['train'].classes
print("Classes:", class_names)

# Define basic CNN model
class BasicCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(BasicCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = BasicCNN(num_classes=2).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, criterion, optimizer, epochs=10):
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss.append(running_loss / len(dataloaders['train']))
        train_acc.append(correct / total)

        # Validation phase
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in dataloaders['val']:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                v_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)

        val_loss.append(v_loss / len(dataloaders['val']))
        val_acc.append(v_correct / v_total)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss[-1]:.4f}, Acc: {train_acc[-1]*100:.2f}% | Val Loss: {val_loss[-1]:.4f}, Acc: {val_acc[-1]*100:.2f}%")

    return model, train_loss, train_acc, val_loss, val_acc

#  Train
model, train_loss, train_acc, val_loss, val_acc = train_model(model, criterion, optimizer, epochs=10)

# Save model
torch.save(model.state_dict(), "ai_vs_real_cnn.pth")
print("Model saved to ai_vs_real_cnn.pth")

