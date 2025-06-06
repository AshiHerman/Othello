import numpy as np
import random

def move_str_to_coords(move):
    col = ord(move[0].upper()) - ord('A')
    row = int(move[1]) - 1
    return (row, col)

def initial_board():
    board = np.zeros((8, 8), dtype=int)
    board[3, 3] = 1
    board[4, 4] = 1
    board[3, 4] = -1
    board[4, 3] = -1
    return board

DIRS = [(-1, -1), (-1, 0), (-1, 1),
        (0, -1),          (0, 1),
        (1, -1), (1, 0),  (1, 1)]

def opponent(player):
    return -player

def is_on_board(x, y):
    return 0 <= x < 8 and 0 <= y < 8

def apply_move(board, move, player):
    board = board.copy()
    flips = []
    row, col = move
    for dx, dy in DIRS:
        x, y = row + dx, col + dy
        this_flip = []
        while is_on_board(x, y) and board[x, y] == opponent(player):
            this_flip.append((x, y))
            x += dx
            y += dy
        if is_on_board(x, y) and board[x, y] == player:
            flips += this_flip
    for x, y in flips:
        board[x, y] = player
    board[row, col] = player
    return board

def get_board_states(moves_str):
    moves = [moves_str[i:i+2] for i in range(0, len(moves_str.strip()), 2)]
    board = initial_board()
    states = []
    player = -1
    for move_str in moves:
        states.append(board.tolist())
        move = move_str_to_coords(move_str)
        board = apply_move(board, move, player)
        player = -player
    return states

def get_moves(moves_str):
    moves = [moves_str[i:i+2] for i in range(0, len(moves_str.strip()), 2)]
    return [move_str_to_coords(m) for m in moves]

def load_batch(filename, batch_size):
    """
    Yields batches of (all_states, all_moves) from filename.
    Each batch is a tuple: (all_states, all_moves)
    """
    batch_lines = []
    with open(filename) as f:
        for line in f:
            game = line.strip()
            if not game:
                continue
            batch_lines.append(game)
            all_states = []
            all_moves = []
            if len(batch_lines) == batch_size:
                for game in batch_lines:
                    states = get_board_states(game)
                    idx = random.choice(range(len(states)))
                    all_states += [states[idx]]
                    all_moves += [get_moves(game)[idx]]
                yield all_states, all_moves
                batch_lines = []
        # yield final batch if any remaining lines
        all_states = []
        all_moves = []
        if batch_lines:
            for game in batch_lines:
                states = get_board_states(game)
                idx = random.choice(range(len(states)))
                all_states += [states[idx]]
                all_moves += [get_moves(game)[idx]]
            yield all_states, all_moves


# Usage example:
if __name__ == "__main__":
    gen = load_batch('./parser/all_games.txt', 10)
    all_states, all_moves = next(gen)
    print(all_states)
    print(all_moves)


    # all_states, all_moves = load_batch('./parser/all_games.txt', 10)

    # print(all_states)
    # print(all_moves)

    # with open('states.txt', 'w') as sf, open('moves.txt', 'w') as mf:
    #     for states, moves in zip(all_states, all_moves):
    #         for state, move in zip(states, moves):
    #             sf.write('\n'.join(map(str, state)) + '\n')
    #             sf.write('\n')
    #             mf.write(f"{move[0]} {move[1]}\n")
    #         sf.write('\n')  # blank line between games (optional)
    #         mf.write('\n')
    #         sf.write('\n')
