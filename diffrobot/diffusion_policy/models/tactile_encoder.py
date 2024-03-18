import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class TactileEncoder(nn.Module):
    def __init__(self, out_dim):
        super(TactileEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        # Assuming a max pooling is applied after each conv layer, the size remains 4x4
        self.fc1 = nn.Linear(16*4*4, 120)  # Flattened 16 channels of 4x4 images
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dim)

    def forward(self, x):
        # x should be of shape (batch_size, 3, 4, 4)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features