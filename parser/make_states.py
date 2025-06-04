import numpy as np

def movestr_to_tuple(movestr):
    col = ord(movestr[0].upper()) - ord('A')
    row = int(movestr[1]) - 1
    return row, col

def initial_board():
    board = np.zeros((8, 8), dtype=int)
    board[3, 3] = 1
    board[4, 4] = 1
    board[3, 4] = -1
    board[4, 3] = -1
    return board

DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

def apply_move(board, row, col, player):
    new_board = board.copy()
    new_board[row, col] = player
    opp = -player
    for dr, dc in DIRECTIONS:
        r, c = row + dr, col + dc
        path = []
        while 0 <= r < 8 and 0 <= c < 8:
            if new_board[r, c] == opp:
                path.append((r, c))
            elif new_board[r, c] == player:
                for rr, cc in path:
                    new_board[rr, cc] = player
                break
            else:
                break
            r += dr
            c += dc
    return new_board

def process_games(input_file, output_file):
    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            line = line.strip().upper()
            if not line:
                continue
            moves = [line[i:i+2] for i in range(0, len(line), 2)]
            board = initial_board()
            player = -1
            for mv in moves:
                state_flat = board.flatten().tolist()
                state_str = " ".join(map(str, state_flat))
                f_out.write(f"{state_str} {mv}\n")
                row, col = movestr_to_tuple(mv)
                board = apply_move(board, row, col, player)
                player *= -1

if __name__ == "__main__":
    process_games("./Othello-Board-Parser/all_games.txt", "./Othello-Board-Parser/states.txt")
