# -*- coding: utf-8 -*-
"""
Created for the book "Artificial Intelligence for Engineers" by Zhen (Leo) Liu.
The related theory, notation, and procedure are explained in the book.
Additional information is available at https://www.ai-engineer.org/.
This example trains a compact PyTorch CNN on MNIST.
"""


import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

transform = transforms.ToTensor()
training_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)
training_loader = DataLoader(Subset(training_data, range(12000)), batch_size=128, shuffle=True)
test_loader = DataLoader(Subset(test_data, range(2000)), batch_size=256)

model = nn.Sequential(
    nn.Conv2d(1, 16, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 5 * 5, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_function = nn.CrossEntropyLoss()

model.train()
for images, labels in training_loader:
    optimizer.zero_grad()
    loss = loss_function(model(images), labels)
    loss.backward()
    optimizer.step()

model.eval()
correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        correct += (model(images).argmax(dim=1) == labels).sum().item()
print(f"Test accuracy: {correct / len(test_loader.dataset):.3f}")
