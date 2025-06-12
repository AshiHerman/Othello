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
from imitator.make_layers import *

# Learns to predict human othello moves

BATCH_SIZE = 60
EPOCHS = 20

NUM_CHANNELS = 5
NUM_HIDDEN_CHANNELS = 64
NUM_LAYERS = 18

TRAIN_PATH = './parser/train.txt'
TEST_PATH = './parser/test.txt'

def process(boards, moves):
    input = []
    target = []
    for board, move in zip(boards, moves):
        # Build channels: 3 for positions, 1 for turn, 1 for available spots
        channels = [
            positions_layer(board, -1),
            positions_layer(board, 0),
            positions_layer(board, 1),
            turn_layer(board),
            available_spots_layer(board)
        ]  # Each is (8, 8)
        input.append(channels)  # shape (num_channels, 8, 8)
        target.append(move[0]*8 + move[1])
    input = np.array(input)  # shape: (batch_size, num_channels, 8, 8)
    input = torch.tensor(input, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.long)
    return input, target



class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = NUM_CHANNELS
        out_channels = NUM_HIDDEN_CHANNELS
        self.num_layers = NUM_LAYERS

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels if i==0 else out_channels + NUM_CHANNELS, out_channels, kernel_size=3, stride=1, padding=1)
            for i in range(NUM_LAYERS)
        ])
        self.relu = nn.ReLU()
        self.final = nn.Conv2d(out_channels + NUM_CHANNELS, 64, kernel_size=1, stride=1)

    
    def forward(self, input):
        x = input
        x = self.relu(self.convs[0](x))
        for conv in self.convs[1:]:
            x = torch.cat([x, input], dim=1)
            x = self.relu(conv(x))
        x = torch.cat([x, input], dim=1)
        out = self.final(x)
        out = out.mean([-2,-1])
        return out

import matplotlib.pyplot as plt  # Add this at the top

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
            input, target = process(boards, moves)
            optimizer.zero_grad()
            expected = model(input)
            loss = criterion(expected, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total += 1
            if total > 300:
                break

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            gen = load_batch(TEST_PATH, BATCH_SIZE)
            for boards, moves in tqdm.tqdm(gen):
                input, target = process(boards, moves)
                expected = model(input)
                test_loss += criterion(expected, target).item()
                predicted = torch.argmax(expected, dim=1)
                rows = predicted // 8
                cols = predicted % 8
                correct += (predicted == target).sum().item()
                total += 1
                if total > 100:
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

def find_probs(model, boards, moves):
    input, _ = process(boards, moves)
    expectations = model(input)

    all_probs = []
    all_valid_moves = []
    for i in range(len(moves)):
        board = boards[i]
        move = moves[i]
        expected = expectations[i]

        player = turn(board)
        valid_moves = get_valid(board, player)
        mask = np.array(valid_moves[:64]) == 0
        expected[mask] = -np.inf
        probs = torch.softmax(expected, dim=0).detach().numpy()  # shape (64,)
        
        predicted = torch.argmax(expected, dim=0).item()
        row = predicted // 8
        col = predicted % 8

        print(f"Example Prediction #{i+1}:")
        # print_board(boards[0])
        print(f'Predicted: ({row + 1}, {col + 1})')
        move = moves[i]
        print("Actual:", end=' ')
        print_move(move)
        print('\n')

        all_probs.append(probs)
        all_valid_moves.append(valid_moves)

    return all_probs, all_valid_moves


if __name__ == "__main__":
    model = train()
    
    gen = load_batch(TEST_PATH, batch_size=1)
    boards, moves = next(gen)
    find_probs(model, boards, moves)
