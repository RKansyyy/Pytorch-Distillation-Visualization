import torch
import torchvision.transforms as transforms

import torchvision
from lucent.optvis import render
from lucent.modelzoo import inceptionv1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_set = torchvision.datasets.ImageNet(root='./data/', train=True, transform=train_transform, download=True)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True)

val_set = torchvision.datasets.ImageNet(root='./data/', train=False, transform=val_transform, download=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=10, shuffle=True)

