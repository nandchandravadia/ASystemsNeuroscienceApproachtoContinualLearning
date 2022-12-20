
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x1 = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x2 = F.max_pool2d(F.relu(self.conv2(x1)), (2, 2))
        x3 = x.view(-1, self.num_flat_features(x2))
        x4 = F.relu(self.fc1(x3))
        x5 = F.relu(self.fc2(x4))
        x6 = self.fc3(x5)

        return x6, x5

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features