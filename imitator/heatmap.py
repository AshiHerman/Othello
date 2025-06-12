import torch
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parser.make_state import load_batch
from imitator.make_layers import *
from imitator.train import ConvNet, find_probs
from othello.othello_visualizer import show_state

# Uses model that predicts expert othello player moves to create a heatmap over probabilities of next actions

MODEL_PATH = './imitator/model_saves/imitator_3.5.pth'
TEST_PATH = './parser/test.txt'
BATCH_SIZE = 1

if __name__ == "__main__":
    model = ConvNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    gen = load_batch(TEST_PATH, batch_size=BATCH_SIZE)
    boards, moves = next(gen)
    all_probs, all_valid_moves = find_probs(model, boards, moves)

    for i in range(BATCH_SIZE):
        board = boards[i]
        player = turn(board)
        heatmap = all_probs[i].reshape(8, 8)

        show_state(
            board,
            player,
            valid_moves=all_valid_moves[i],
            message="Probabilities that expert player will make that move.",
            heatmap=heatmap,
            heatmap_cmap="viridis",
            heatmap_alpha=0.6,
            board_color="white"
        )

