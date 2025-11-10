import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), # 16 channels, (480x480)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4), # Downsample by 4 -> 120x120

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # 32 channels, (120x120)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4), # Downsample by 4 -> 30x30

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 64 channels, (30x30)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Downsample by 2 -> 15x15

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 128 channels, (15x15)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3) # Downsample by 3 -> 5x5 (approx)
        )

        self.fc_input_size = 128 * 5 * 5

        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, 256), # Reduce to a reasonable hidden size
            nn.ReLU(),
            nn.Linear(256, 9) # Final layer for 9 outputs
        )
        
    def forward(self, x):
        x = self.features(x.unsqueeze(0))
        x = x.view(-1, self.fc_input_size)
        x = self.classifier(x)
        return x