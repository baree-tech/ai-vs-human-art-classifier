# import libraries

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# paths and transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = '/media/bareera/KINGSTON/Bareera/Dataset/AI_vs_Real'
batch_size = 1
image_size = 224

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
class_names = test_dataset.classes

# Define CNN
class BasicCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(BasicCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
 # load the saved model
model = BasicCNN(num_classes=2).to(device)
model.load_state_dict(torch.load('ai_vs_real_cnn.pth', map_location=device))
model.eval()
 # evaluate model(confusion matrix and classification report)
from sklearn.metrics import confusion_matrix, classification_report

y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix.png")


import matplotlib.pyplot as plt

def imshow(img_tensor, ax, title):
    img = img_tensor.numpy().transpose((1, 2, 0))
    img = img * 0.5 + 0.5  # unnormalize from [-1, 1] to [0, 1]
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')

# Set to eval mode
model.eval()

shown = 0
max_show = 10  # Total images to show
shown_per_class = {'AI': 0, 'human': 0}
max_per_class = 5  # Show 5 from each class

fig, axs = plt.subplots(2, 5, figsize=(15, 6))
axs = axs.flatten()

with torch.no_grad():
    for images, labels in test_loader:
        if shown >= max_show:
            break

        for img, label in zip(images, labels):
            if shown >= max_show:
                break

            output = model(img.unsqueeze(0).to(device))
            _, pred = torch.max(output, 1)

            actual_class = class_names[label.item()]
            predicted_class = class_names[pred.item()]

            # Show max 5 from each class
            if shown_per_class[actual_class] < max_per_class:
                title = f"Pred: {predicted_class}\nActual: {actual_class}"
                imshow(img.cpu(), axs[shown], title)
                shown += 1
                shown_per_class[actual_class] += 1

plt.tight_layout()
plt.savefig("sample_predictions.png")  # Save instead of showing
print("Sample predictions saved as 'sample_predictions.png'")

