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

BATCH_SIZE = 130
EPOCHS = 25

NUM_CHANNELS = 38
NUM_HIDDEN_CHANNELS = 64
NUM_LAYERS = 8

TRAIN_PATH = './parser/train.txt'
TEST_PATH = './parser/test.txt'
MODEL_PATH = './imitator/model_saves/imitator_y.pth'


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
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        print(f"Model loaded from: {MODEL_PATH}")
    else:
        print(f"Model file does not exist at: {MODEL_PATH}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    test_losses = []
    test_accuracies = []
    train_losses = []

    for epoch in range(EPOCHS):
        print(f"\nEpoch #{epoch + 1}")

        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        gen = load_batch(TRAIN_PATH, BATCH_SIZE)
        
        for boards, moves in tqdm.tqdm(gen):
            input = process_boards(boards)
            target = process_moves(moves)
            optimizer.zero_grad()
            expected = model(input)
            loss = criterion(expected, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
            if train_batches > 120:  # Use more training data
                break

        # Validation phase
        model.eval()
        test_loss = 0.0
        correct = 0
        total_samples = 0
        test_batches = 0
        
        with torch.no_grad():
            gen = load_batch(TEST_PATH, BATCH_SIZE)
            for boards, moves in tqdm.tqdm(gen):
                input = process_boards(boards)
                target = process_moves(moves)
                expected = model(input)
                test_loss += criterion(expected, target).item()
                predicted = torch.argmax(expected, dim=1)
                correct += (predicted == target).sum().item()
                total_samples += len(target)  # Count actual samples
                test_batches += 1
                if test_batches > 40:  # More validation data
                    break
        
        # Calculate metrics correctly
        avg_train_loss = train_loss / train_batches
        avg_test_loss = test_loss / test_batches
        accuracy = correct / total_samples
        
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        test_accuracies.append(accuracy)
        
        # Learning rate scheduling
        scheduler.step(avg_test_loss)
        
        print(
            f"Epoch {epoch+1}/{EPOCHS} — "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_test_loss:.4f}, "
            f"Val Acc: {accuracy:.4f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        torch.save(model.state_dict(), './imitator/model_saves/imitator_y.pth')

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
