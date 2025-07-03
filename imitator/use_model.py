import torch
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parser.make_state import load_batch
from imitator.make_layers import *
from imitator.train import ConvNet, process_boards
from othello.othello_visualizer import show_state

# Uses model that predicts expert othello player moves to create a heatmap over probabilities of next actions

MODEL_PATH = './imitator/model_saves/imitator_y.pth'
TEST_PATH = './parser/test.txt'
BATCH_SIZE = 1

def find_best(model, board):
    model.eval()
    input = process_boards([board])
    expected = model(input)[0]
    player = -turn(board)
    valid_moves = get_valid(board, player)
    mask = np.array(valid_moves[:64]) == 0
    expected[mask] = -np.inf
    
    # probs = torch.softmax(expected, dim=0).detach().numpy()  # shape (64,)
    # print_board(probs.reshape(8, 8))
    predicted = torch.argmax(expected, dim=0).item()
    return predicted

def find_probs(model, boards, moves):
    input = process_boards(boards)
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

        # print(f"Example Prediction #{i+1}:")
        # # print_board(boards[0])
        # print(f'Predicted: ({row + 1}, {col + 1})')
        # move = moves[i]
        # print("Actual:", end=' ')
        # print_move(move)
        # print('\n')

        move = moves[i]
        message = f"Predicted: ({row + 1}, {col + 1})\n Actual: {(move[0] + 1, move[1] + 1)}"

        all_probs.append(probs)
        all_valid_moves.append(valid_moves)

    return all_probs, all_valid_moves, message


def heatmap():
    model = ConvNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    gen = load_batch(TEST_PATH, batch_size=BATCH_SIZE)
    boards, moves = next(gen)
    all_probs, all_valid_moves, message = find_probs(model, boards, moves)
    find_best(model, boards[0])

    for i in range(BATCH_SIZE):
        board = boards[i]
        player = turn(board)
        heatmap = all_probs[i].reshape(8, 8)

        show_state(
            board,
            player,
            valid_moves=all_valid_moves[i],
            message=message,
            heatmap=heatmap,
            heatmap_cmap="viridis",
            heatmap_alpha=0.6,
            board_color="white"
        )

if __name__ == "__main__":
    heatmap()