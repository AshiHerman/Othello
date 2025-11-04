import os
import sys
# If your alphazero code lives in a subfolder, make sure Python can find it:
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from parser.make_state import load_batch
from players.az import AlphaZero
from players.imit import turn  # to get current player


# Configuration
TEST_PATH   = './parser/test.txt'
BATCH_SIZE  = 1     # tune for speed vs memory
NUM_SIMS    = 25     # MCTS simulations per move

def evaluate_alphazero(test_path, batch_size, num_sims):
    # Initialize AlphaZero agent
    game = None  # not used directly
    az = AlphaZero(game, num_sims=num_sims, board_size=8)
    
    total = 0
    correct = 0

    gen = load_batch(test_path, batch_size=batch_size)
    for boards, moves in gen:
        for board, human_move in zip(boards, moves):
            total += 1
            print(f"Evaluating game #{total}")
            # human_move is (row, col) with 0-based indices
            human_idx = human_move[0] * 8 + human_move[1]

            # Prepare the state tuple for AlphaZero.choose_move:
            # choose_move expects (board_matrix, current_player)
            # convert the board list into a numpy array so negation works
            board_arr = np.array(board, dtype=np.int8)
            player    = turn(board_arr)
            az_idx    = az.choose_move((board_arr, player))

            if az_idx == human_idx:
                correct += 1

        # (optional) break early if you only want the first N positions
        if total >= 100: break

    return correct, total

if __name__ == '__main__':
    correct, total = evaluate_alphazero(TEST_PATH, BATCH_SIZE, NUM_SIMS)
    acc = 100 * correct / total if total else 0.0
    print(f"Evaluated {total} positions")
    print(f"AlphaZero vs. human move match rate: {acc:.2f}%")
