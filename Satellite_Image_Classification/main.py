#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms


if torch.cuda.is_available():
    print('Number of GPUs:', torch.cuda.device_count())
    print('GPU Model:', torch.cuda.get_device_name(0))
    print('Total GPU Memory [GB]:', torch.cuda.get_device_properties(0).total_memory / 1e9)


images_source = 'EuroSAT_RGB'
training_destination = 'training_images'
testing_destination = 'testing_images'

image_class = 0
class_dict = {}

files = os.listdir(images_source)
files.sort()




transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   
])


training_dataset = torchvision.datasets.ImageFolder(root='training_images', transform=transform)

training_dataloader = torch.utils.data.DataLoader(
    training_dataset, batch_size=64, shuffle=True, num_workers=2
)

testing_dataset = torchvision.datasets.ImageFolder(root='testing_images', transform=transform)

testing_dataloader = torch.utils.data.DataLoader(
    testing_dataset, batch_size=1, shuffle=True, num_workers=2
)
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

data_iter = iter(training_dataloader)
images, labels = next(data_iter)


class_mapping = {
    0: 'AnnualCrop',
    1: 'Forest',
    2: 'HerbaceousVegetation',
    3: 'Highway',
    4: 'Industrial',
    5: 'Pasture',
    6: 'PermanentCrop',
    7: 'Residential',
    8: 'River',
    9: 'SeaLake'
}


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

data_iter = iter(training_dataloader)
images, labels = next(data_iter)

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(215296, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

model = CustomNet()

print(model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
num_epochs = 5

test_iter = iter(testing_dataloader)

print('Training Started!')

for epoch in range(num_epochs):
    running_loss = 0.0
    i = 0
    
    for data in training_dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        total_correct = 0
        total_samples = 0
        
        if i % 100 == 0:
            with torch.no_grad():
                test_images, test_labels = next(test_iter)
                test_images, test_labels = test_images.to(device), test_labels.to(device)
                test_outputs = model(test_images[:8])
                _, predicted = torch.max(test_outputs, 1)

        i += 1

    print(f"Epoch {epoch}, Loss: {running_loss / (i)}")

print('Training Completed!')


total_correct = 0
total_samples = 0

model.eval()
with torch.no_grad():
    for data in testing_dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

accuracy = total_correct / total_samples
print(accuracy)

with torch.no_grad():
    data_iter = iter(testing_dataloader)
    data = next(data_iter)
    inputs, _ = data
    image = inputs[0].unsqueeze(0)
    image = image.to(device)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    image_numpy = image.cpu().numpy()[0]
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    if image_numpy.shape[2] == 1:
        image_numpy = np.squeeze(image_numpy, axis=2)
    elif image_numpy.shape[2] == 3:
        image_numpy = (image_numpy - image_numpy.min()) / (image_numpy.max() - image_numpy.min())
    plt.figure(figsize=(6, 6))
    plt.imshow(image_numpy)
    plt.title(f'Prediction: {class_mapping[predicted.item()]}')
    plt.axis('off')
    plt.show()

