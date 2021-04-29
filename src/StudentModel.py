import torch
import torch.nn as nn


class StudentModel(nn.Module):
    def __init__(self, out_features):
        super(StudentModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),         # (3 x 64 x 64)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                                                 # (64 x 32 x 32)
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),        # (64 x 32 x 32)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                                                 # (128 x 16 x 16)
            nn.BatchNorm2d(num_features=128),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),       # (128 x 16 x 16)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                                                 # (256 x 8 x 8)
            nn.BatchNorm2d(num_features=256),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=256*8*8, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(in_features=256, out_features=out_features),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class StudentModel_fc(nn.Module):
    def __init__(self, out_features):
        super(StudentModel_fc, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),         # (3 x 64 x 64)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                                                 # (64 x 32 x 32)
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),        # (64 x 32 x 32)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                                                 # (128 x 16 x 16)
            nn.BatchNorm2d(num_features=128),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),       # (128 x 16 x 16)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                                                 # (256 x 8 x 8)
            nn.BatchNorm2d(num_features=256),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=256*8*8, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(in_features=256, out_features=out_features),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class StudentModel3(nn.Module):
    def __init__(self, out_features):
        super(StudentModel3, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),         # (3 x 64 x 64)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                                                 # (64 x 32 x 32)
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),        # (64 x 32 x 32)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                                                 # (128 x 16 x 16)
            nn.BatchNorm2d(num_features=128),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),     # (128 x 16 x 16)
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),       # (128 x 16 x 16)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                                                 # (256 x 8 x 8)
            nn.BatchNorm2d(num_features=256),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),      # (256 x 8 x 8)
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=256),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=256*8*8, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(in_features=256, out_features=out_features),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out




class StudentModel4(nn.Module):
    def __init__(self, out_features):
        super(StudentModel4, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),         # (3 x 64 x 64)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                                                 # (64 x 32 x 32)
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),  # (64 x 32 x 32)
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),        # (64 x 32 x 32)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                                                 # (128 x 16 x 16)
            nn.BatchNorm2d(num_features=128),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),     # (128 x 16 x 16)
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),       # (128 x 16 x 16)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                                                 # (256 x 8 x 8)
            nn.BatchNorm2d(num_features=256),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),      # (256 x 8 x 8)
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=256),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=256*8*8, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(in_features=256, out_features=out_features),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out




class StudentModel2(nn.Module):
    def __init__(self, out_features):
        super(StudentModel2, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),         # (3 x 64 x 64)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                                                 # (64 x 32 x 32)
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),        # (64 x 32 x 32)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                                                 # (128 x 16 x 16)
            nn.BatchNorm2d(num_features=128),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=128*16*16, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(in_features=256, out_features=out_features),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

