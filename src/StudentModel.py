import torch
import torch.nn as nn


class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1),         # (3 x 32 x 32)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                                                 # (32 x 16 x 16)
            nn.BatchNorm2d(num_features=32),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),        # (32 x 16 x 16)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                                                 # (64 x 8 x 8)
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),       # (64 x 8 x 8)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                                                 # (128 x 4 x 4)
            nn.BatchNorm2d(num_features=128),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=128*4*4, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(in_features=256, out_features=10),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class StudentModel2(nn.Module):
    def __init__(self):
        super(StudentModel2, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1),         # (3 x 32 x 32)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                                                 # (32 x 16 x 16)
            nn.BatchNorm2d(num_features=32),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),        # (32 x 16 x 16)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                                                 # (64 x 8 x 8)
            nn.BatchNorm2d(num_features=64),

        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=64*8*8, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(in_features=256, out_features=10),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

