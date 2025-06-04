import numpy as np
import torch
from torch import nn

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parser.make_state import load_batch

from parser.make_layers import *


BATCH_SIZE = 10
TEST_IDX =  int(133801 * 0.8)
NUM_CHANNELS = 3
NUM_HIDDEN_CHANNELS = 64
NUM_LAYERS = 11


# # Check successfull opening
# states, moves = load_batch('parser/all_games.txt', batch_size=BATCH_SIZE, batch_idx=0)
# for i in range(8):
#     print(states[0][0][i])

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
  # Adjust for however many repeated blocks you want
        in_channels = NUM_CHANNELS
        out_channels = NUM_HIDDEN_CHANNELS
        layers = []
        for _ in range(NUM_LAYERS):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels  # Only the first layer uses 3 input channels
        # Final output layer
        layers.append(nn.Conv2d(out_channels, 2, kernel_size=1, stride=1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, input):
        return self.net(input)
    
model = ConvNet()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

states, moves = load_batch('parser/all_games.txt', batch_size=BATCH_SIZE, batch_idx=0)

input = []
for state in states[0]:
    # Stack channels: player -1, 0, 1
    layers = [positions_layer(state, i) for i in [-1, 0, 1]]  # List of three 8x8
    input.append(layers)  # Each entry is (3, 8, 8)

input = np.array(input)  # shape: (batch_size, 3, 8, 8)
input = torch.tensor(input, dtype=torch.float32)

output = model.forward(input)
print(output.shape)