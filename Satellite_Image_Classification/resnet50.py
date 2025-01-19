import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torchvision
from torch import nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('Number of GPUs:', torch.cuda.device_count())
    print('GPU Model:', torch.cuda.get_device_name(0))
    print('Total GPU Memory [GB]:', torch.cuda.get_device_properties(0).total_memory / 1e9)
print(f"Using device: {device}")

# Constants
IMAGES_SOURCE = 'EuroSAT_RGB'
TRAINING_DESTINATION = 'training_images'
TESTING_DESTINATION = 'testing_images'
BATCH_SIZE = 64
NUM_WORKERS = 2

# Class mapping
CLASS_MAPPING = {
    0: 'AnnualCrop', 1: 'Forest', 2: 'HerbaceousVegetation', 3: 'Highway',
    4: 'Industrial', 5: 'Pasture', 6: 'PermanentCrop', 7: 'Residential',
    8: 'River', 9: 'SeaLake'
}

# Data transforms
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = ImageFolder(root=TRAINING_DESTINATION, transform=transform_train)
test_dataset = ImageFolder(root=TESTING_DESTINATION, transform=transform_test)

# Model definition
def get_model():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    return model.to(device)

# Training function
def train(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    scheduler.step()
    accuracy = 100. * correct / total
    return running_loss / len(loader), accuracy

# Evaluation function
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    return running_loss / len(loader), accuracy

# Cross-validation and training loop
NUM_EPOCHS = 20
NUM_FOLDS = 5
kfold = KFold(n_splits=NUM_FOLDS, shuffle=True)

for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
    print(f'FOLD {fold}')
    print('--------------------------------')

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_subsampler, num_workers=NUM_WORKERS)
    val_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=val_subsampler, num_workers=NUM_WORKERS)

    model = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # Save the model for each fold
    torch.save(model.state_dict(), f'resnet50_fold_{fold}.pth')

# Final evaluation on the test set
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
model = get_model()
model.load_state_dict(torch.load('resnet50_fold_0.pth', weights_only=True))
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"Final Test Accuracy: {test_acc:.2f}%")

# Visualize a prediction
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')


model.eval()
dataiter = iter(test_loader)
images, labels = next(dataiter)

with torch.no_grad():
    images = images.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

imshow(torchvision.utils.make_grid(images[:4]))
print('Predicted:', ' '.join(f'{CLASS_MAPPING[predicted[j].item()]}' for j in range(4)))
print('Actual:   ', ' '.join(f'{CLASS_MAPPING[labels[j].item()]}' for j in range(4)))
plt.show()
