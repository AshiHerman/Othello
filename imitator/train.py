from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
import tqdm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from othello.othello_game import OthelloGame
from parser.make_state import load_batch
from imitator.make_layers import get_feature_planes, turn  # use your revised version


# Learns to predict human othello moves

BATCH_SIZE = 40 #60
EPOCHS = 3

NUM_CHANNELS = 38
NUM_HIDDEN_CHANNELS = 64
NUM_LAYERS = 12

TRAIN_PATH = './parser/train.txt'
TEST_PATH = './parser/test.txt'


def process_boards(boards, moves=None):
    """
    Takes a list of boards (optionally with moves) and returns input tensor suitable for the model.
    """
    inputs = []
    for idx, board in enumerate(boards):
        # If you want to include move history, pass prev_moves here (for now: only most recent move)
        # prev_moves = moves[idx] if moves is not None else None
        player = turn(board)
        features = get_feature_planes(board, current_player=player)
        inputs.append(features)
    # (batch_size, num_channels, 8, 8)
    inputs = np.array(inputs)
    return torch.tensor(inputs, dtype=torch.float32)


def process_moves(moves):
    """
    Takes a list of moves and returns target tensor.
    Each move should be a tuple (row, col).
    """
    targets = [move[0]*8 + move[1] for move in moves]
    return torch.tensor(targets, dtype=torch.long)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = torch.relu(out)
        return out

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_conv = nn.Conv2d(NUM_CHANNELS, NUM_HIDDEN_CHANNELS, 3, 1, 1)
        self.input_bn = nn.BatchNorm2d(NUM_HIDDEN_CHANNELS)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(NUM_HIDDEN_CHANNELS) for _ in range(NUM_LAYERS)]
        )
        self.policy_conv = nn.Conv2d(NUM_HIDDEN_CHANNELS, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, 64)  # 64 board positions

    def forward(self, x):
        x = torch.relu(self.input_bn(self.input_conv(x)))
        x = self.res_blocks(x)
        x = torch.relu(self.policy_bn(self.policy_conv(x)))
        x = x.view(x.size(0), -1)
        out = self.policy_fc(x)
        return out



def train():
    model = ConvNet()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    test_losses = []
    test_accuracies = []
    train_losses = []   # <- ADD THIS LINE

    for epoch in range(EPOCHS):
        print(f"Epoch #{epoch + 1}")

        model.train()
        train_loss = 0.0
        gen = load_batch(TRAIN_PATH, BATCH_SIZE)
        total = 0
        for boards, moves in tqdm.tqdm(gen):
            input = process_boards(boards)
            target = process_moves(moves)
            optimizer.zero_grad()
            expected = model(input)
            loss = criterion(expected, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total += 1
            if total > 90:
                break

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            gen = load_batch(TEST_PATH, BATCH_SIZE)
            for boards, moves in tqdm.tqdm(gen):
                input = process_boards(boards)
                target = process_moves(moves)
                expected = model(input)
                test_loss += criterion(expected, target).item()
                predicted = torch.argmax(expected, dim=1)
                correct += (predicted == target).sum().item()
                total += 1
                if total > 30:
                    break
        
        train_loss /= (total*3/4)
        test_loss /= (total/4)
        accuracy = correct / total
        train_losses.append(train_loss)        # <- ADD THIS LINE
        test_losses.append(test_loss)
        test_accuracies.append(accuracy)
        print(
            f"Epoch {epoch+1}/{EPOCHS} — "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {test_loss:.4f}, "
            f"Val Acc: {accuracy:.4f}"
        )   

    torch.save(model.state_dict(), './imitator/model_saves/imitator_x.pth')

    # ------ Plotting Loss and Accuracy Curves ------
    plt.figure(figsize=(10, 4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss vs Epoch")

    plt.subplot(1,2,2)
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model
    # ----------------------------------------------

if __name__ == "__main__":
    model = train()
    gen = load_batch(TEST_PATH, batch_size=1)
    boards, moves = next(gen)
