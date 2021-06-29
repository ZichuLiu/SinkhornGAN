import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

DEVICE = 0


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=6):
        super(Generator, self).__init__()
        layers = [nn.Linear(input_size, hidden_size)]
        for i in range(num_layers-2):
            layers.extend([nn.ELU(),
                          nn.Linear(hidden_size, hidden_size)])
        layers.append(nn.Linear(hidden_size, output_size).cuda(device=DEVICE))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)