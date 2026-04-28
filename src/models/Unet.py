import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=9, num_classes=9):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2) 
        )

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Classifier
        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x = self.enc1(x)
        x = self.enc2(x)

        # Decoder
        x = self.dec1(x)
        x = self.dec2(x)

        # Classifier
        x = self.classifier(x)
        return x