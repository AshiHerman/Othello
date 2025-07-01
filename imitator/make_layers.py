import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parser.make_state import load_batch
from othello.othello_visualizer import show_state
from othello.othello_game import get_valid

def disc_layers(state, current_player):
    # Returns: [player_disc, opponent_disc, empty]
    return [
        (state == current_player).astype(np.float32),
        (state == -current_player).astype(np.float32),
        (state == 0).astype(np.float32)
    ]

def history_layers(prev_move):
    # Plane for most recent move
    plane = np.zeros((8,8), dtype=np.float32)
    if prev_move is not None:
        move = prev_move
        if move is not None and all(0 <= x < 8 for x in move):
            plane[move[0]][move[1]] = 1.0
    return [plane]

def get_valid_moves(board, player):
    valids = get_valid(board, player)
    valids = np.array(valids[:-1]).reshape(8, 8)
    return valids

def mobility_layers(state, current_player):
    # 8 planes, one-hot encoding of opponent's possible responses after each move
    planes = []
    moves = get_valid_moves(state, current_player)
    positions = np.argwhere(moves == 1)
    for idx in range(8):
        plane = np.zeros((8,8), dtype=np.float32)
        for pos in positions:
            tmp = np.copy(state)
            tmp[pos[0], pos[1]] = current_player
            opp_moves = get_valid_moves(tmp, -current_player)
            count = int(np.sum(opp_moves))
            if count == idx or (idx == 7 and count >= 7):
                plane[pos[0], pos[1]] = 1.0
        planes.append(plane)
    return planes

def stable_discs(board, player):
    # Simple dummy: only corners are always stable; replace with full stability logic
    stab = np.zeros((8,8), dtype=np.float32)
    for r, c in [(0,0), (0,7), (7,0), (7,7)]:
        if board[r][c] == player:
            stab[r][c] = 1.0
    return stab

def player_stability_gained_layers(state, current_player):
    # For each move, how many new stable discs does player gain? 8 one-hot planes
    planes = []
    moves = get_valid_moves(state, current_player)
    positions = np.argwhere(moves == 1)
    for idx in range(8):
        plane = np.zeros((8,8), dtype=np.float32)
        for pos in positions:
            tmp = np.copy(state)
            tmp[pos[0], pos[1]] = current_player
            before = np.sum(stable_discs(state, current_player))
            after = np.sum(stable_discs(tmp, current_player))
            gain = int(after - before)
            if gain == idx or (idx == 7 and gain >= 7):
                plane[pos[0], pos[1]] = 1.0
        planes.append(plane)
    return planes

def opponent_stability_gained_layers(state, current_player):
    # Like above, for opponent moves
    planes = []
    moves = get_valid_moves(state, -current_player)
    positions = np.argwhere(moves == 1)
    for idx in range(8):
        plane = np.zeros((8,8), dtype=np.float32)
        for pos in positions:
            tmp = np.copy(state)
            tmp[pos[0], pos[1]] = -current_player
            before = np.sum(stable_discs(state, -current_player))
            after = np.sum(stable_discs(tmp, -current_player))
            gain = int(after - before)
            if gain == idx or (idx == 7 and gain >= 7):
                plane[pos[0], pos[1]] = 1.0
        planes.append(plane)
    return planes

def frontier_layers(state):
    # For each occupied square, count empty neighbors (8 planes)
    def num_empty_neighbors(board, r, c):
        count = 0
        for dr in [-1,0,1]:
            for dc in [-1,0,1]:
                if dr==0 and dc==0: continue
                nr, nc = r+dr, c+dc
                if 0<=nr<8 and 0<=nc<8 and board[nr][nc]==0:
                    count += 1
        return count

    planes = []
    for idx in range(8):
        plane = np.zeros((8,8), dtype=np.float32)
        for i in range(8):
            for j in range(8):
                if state[i,j] != 0:
                    n = num_empty_neighbors(state, i, j)
                    if n == idx or (idx == 7 and n >= 7):
                        plane[i,j] = 1.0
        planes.append(plane)
    return planes

def legal_moves_layer(state, current_player):
    return [get_valid_moves(state, current_player).astype(np.float32)]

def ones_layer():
    return [np.ones((8,8), dtype=np.float32)]

def player_layer(current_player):
    return [np.ones((8,8), dtype=np.float32) if current_player == 1 else np.zeros((8,8), dtype=np.float32)]

def get_feature_planes(state, prev_move=None, current_player=1):
    state = np.array(state)
    features = []
    features.extend(disc_layers(state, current_player))                   # 3
    # features.extend(history_layers(prev_move))                          
    features.extend(mobility_layers(state, current_player))               # 8
    features.extend(player_stability_gained_layers(state, current_player))# 8
    features.extend(opponent_stability_gained_layers(state, current_player))# 8
    features.extend(frontier_layers(state))                               # 8
    features.extend(legal_moves_layer(state, current_player))             # 1
    features.extend(ones_layer())                                         # 1
    features.extend(player_layer(current_player))                         # 1
    return np.stack(features, axis=0)

def print_layer(layer):
    for row in layer:
        print(' '.join(f"{int(v):2}" for v in row))

def print_move(move):
    row, col = move
    print(f'({row+1}, {col+1})')

def turn(state):
    count_pieces = 0
    for i in range(len(state)):
        for j in range(len(state)):
            count_pieces += 0 if state[i][j]==0 else 1
    return (count_pieces%2)*2 - 1

if __name__ == "__main__":
    gen = load_batch('./parser/all_games.txt', batch_size=1)
    input, moves = next(gen)

    board = input[0]
    player = turn(board)
    prev_moves = moves[0] if moves else []

    features = get_feature_planes(board, prev_moves, player)

    layer_names = [
        "Player discs", "Opponent discs", "Empty",
        "History 1 (last move)", "History 2", "History 3", "History 4",
    ]
    layer_names += [f"Mobility: {i}" for i in range(8)]
    layer_names += [f"Player stability gained: {i}" for i in range(8)]
    layer_names += [f"Opponent stability gained: {i}" for i in range(8)]
    layer_names += [f"Frontier: {i}" for i in range(8)]
    layer_names += ["Legal moves", "Ones", "Player-is-black"]

    print_layer(board)

    print(f"Total feature layers: {features.shape[0]}")

    for i, layer in enumerate(features):
        print(f"\nLayer {i+1} - {layer_names[i] if i < len(layer_names) else 'Extra'}:")
        print_layer(layer)
