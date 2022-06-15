import torch
from torch import nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, input_channels = [3, 6 ,16 ]):
        super(ConvNet, self).__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(input_channels[0], input_channels[1], 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(input_channels[1], input_channels[2], 5)
        self.fc1 = nn.Linear(input_channels[2] * 5 * 5, 1000)
        self.fc2 = nn.Linear(1000, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.input_channels[2] * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x