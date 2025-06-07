
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parser.make_state import load_batch
from othello.othello_visualizer import show_state
from othello.othello_game import get_valid

def get_state(layers):
    state = [[0 for _ in range(8)] for _ in range(8)]
    for i in range(len(state)):
        for j in range(len(state)):
            state[i][j] = layers[2][i][j]-layers[0][i][j]
    return state

# Returns 0 if black and 1 if white
def turn(state):
    count_pieces = 0
    for i in range(len(state)):
        for j in range(len(state)):
            count_pieces += 0 if state[i][j]==0 else 1
    return (count_pieces%2)*2 - 1

def positions_layer(state, type=0):
    layer = [[0 for _ in range(8)] for _ in range(8)]
    for i in range(len(state)):
        for j in range(len(state)):
            layer[i][j] = 1 if state[i][j]==type else 0
    return layer

def turn_layer(state):
    player = turn(state)
    return [[int((player+1)/2) for _ in range(8)] for _ in range(8)]

def available_spots_layer(state):
    player = turn(state)
    moves = get_valid(state, player)
    count = 0
    layer = [[0 for _ in range(8)] for _ in range(8)]
    for i in range(len(state)):
        for j in range(len(state)):
            layer[i][j] = int(moves[count])
            count += 1
    return layer

def print_board(layer):
    for j in range(8):
        print(layer[j])

def print_move(move):
    row, col = move
    print(f'({row+1}, {col+1})')

if __name__ == "__main__":
    gen = load_batch('./parser/all_games.txt', batch_size=1)
    input, moves = next(gen)


    board = input[0]
    player = turn(board)

    print_board(board)
    print_move(moves[0])
    print_board(available_spots_layer(board))

    show_state(board, player=player)

    
    
