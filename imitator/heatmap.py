import torch
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parser.make_state import load_batch
from imitator.make_layers import *
from imitator.train import ConvNet, process
from othello.othello_visualizer import show_state
from othello.othello_game import OthelloGame

MODEL_PATH = './imitator/model_saves/imitator_3.pth'
TEST_PATH = './parser/test.txt'

def main(mask_invalid_moves=True):
    model = ConvNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    gen = load_batch(TEST_PATH, batch_size=1)
    boards, moves = next(gen)
    input, target = process(boards, moves)
    board = boards[0]
    player = turn(board)
    expected = model(input)
    probs = torch.softmax(expected[0], dim=0).detach().numpy()  # shape (64,)

    valid_moves = OthelloGame(8).getValidMoves(board, player)
    if mask_invalid_moves:
        mask = np.array(valid_moves[:64]) == 0
        probs = probs.copy()
        probs[mask] = -np.inf

    pred_idx = np.argmax(probs)
    row = int(pred_idx // 8)
    col = int(pred_idx % 8)

    print("Example Prediction:")
    print_board(board)
    print(f'Predicted: ({row + 1}, {col + 1})')
    move = moves[0]
    print("Actual:", end=' ')
    print_move(move)

    heatmap = torch.softmax(expected[0], dim=0).detach().numpy().reshape(8, 8)
    show_state(
        board,
        player,
        valid_moves=valid_moves,
        heatmap=heatmap,
        heatmap_cmap="viridis",
        heatmap_alpha=0.6,
        board_color="white"
    )

if __name__ == "__main__":
    main(mask_invalid_moves=False)
