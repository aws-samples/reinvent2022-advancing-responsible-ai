from torch import nn
from torch.nn.functional import max_pool2d, relu
import torch


class MLP(nn.Module):
    def __init__(self, colored, hidden_dim, model_weight_init_seed):
        torch.manual_seed(model_weight_init_seed)
        super().__init__()
        self.in_channels = 3 if colored else 1
        self.fc1 = nn.Linear(self.in_channels * 28 * 28, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(-1, self.in_channels * 28 * 28)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


class ConvNet(nn.Module):
    def __init__(self, colored, model_weight_init_seed):
        torch.manual_seed(model_weight_init_seed)
        super().__init__()
        self.in_chnanels = 3 if colored else 1
        self.conv1 = nn.Conv2d(self.in_channels, 20, 3, 1)
        self.conv2 = nn.Conv2d(20, 50, 4, 1)
        self.fc1 = nn.Linear(5 * 5 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, self.in_channels, 28, 28)
        x = relu(self.conv1(x))  # Converts to 20 wide 26 x 26
        x = max_pool2d(x, 2, 2)  # Converts to 20 wide 13 x 13
        x = relu(self.conv2(x))  # Converts to 50 wide 10 x 10
        x = max_pool2d(x, 2, 2)  # Converts to 50 wide 5 x 5
        # Flattens this for interpretation by linear layer
        x = x.view(-1, 5 * 5 * 50)
        x = relu(self.fc1(x))
        return self.fc2(x)
