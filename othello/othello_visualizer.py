import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time

def play_interactive(game, initial_state, is_human_turn_fn, choose_ai_move_fn, ai_move_pause=0.2):
    """
    General interactive GUI game loop for any player setup.
    Adds a pause after the AI move before flipping opponent pieces.
    """
    plt.ion()
    n = game.n
    fig, ax = plt.subplots(figsize=(6, 6))
    move_result = {'selected': None}
    current_state = [initial_state]

    def draw_board(state, message=None):
        ax.clear()
        board, player = state
        valids = game.getValidMoves(board, player)
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_xticks(np.arange(n+1))
        ax.set_yticks(np.arange(n+1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, which='both', color='black', linewidth=1)
        ax.set_facecolor((0.0, 0.6, 0.0))
        piece_colors = {1: 'white', -1: 'black'}
        alpha_color = {1: (1.0, 1.0, 1.0, 0.4), -1: (0.0, 0.0, 0.0, 0.4)}
        for y in range(n):
            for x in range(n):
                piece = board[y][x]
                cx, cy = x + 0.5, n - y - 0.5
                idx = y * n + x
                if piece in piece_colors:
                    ax.add_patch(patches.Circle((cx, cy), 0.4, color=piece_colors[piece], ec='black'))
                elif valids[idx] == 1:
                    ax.add_patch(patches.Circle((cx, cy), 0.4, color=alpha_color[player], linewidth=0))
        ax.set_title(f"Othello - {'White' if player == 1 else 'Black'} to move")
        if message:
            ax.text(0.5, 0.5, message,
                    transform=ax.transAxes, fontsize=32, ha='center',
                    va='center', color='blue', weight='bold')
        ax.set_aspect('equal')
        fig.canvas.draw()
        fig.canvas.flush_events()

    def draw_ai_tile(board_before, action, player):
        """Draw the board with only the AI's new tile (no flips yet)."""
        temp = board_before.copy()
        if action != n * n:  # Not a pass move
            row, col = divmod(action, n)
            temp[row, col] = player
        draw_board((temp, -player))

    def onclick(event):
        if not is_human_turn_fn(current_state[0]):
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        row, col = n - y - 1, x
        idx = row * n + col
        valids = game.getValidMoves(current_state[0][0], current_state[0][1])
        if 0 <= row < n and 0 <= col < n and valids[idx] == 1:
            move_result['selected'] = idx

    fig.canvas.mpl_connect('button_press_event', onclick)

    def has_valid_moves(state):
        board, player = state
        valids = game.getValidMoves(board, player)
        return np.any(valids[:-1])

    while not game.isEnd(current_state[0]):
        draw_board(current_state[0])
        if not has_valid_moves(current_state[0]):
            print(f"Player {current_state[0][1]} has no moves. Passing.")
            current_state[0] = game.enact(current_state[0], n * n)
            continue
        if is_human_turn_fn(current_state[0]):
            move_result['selected'] = None
            while move_result['selected'] is None:
                plt.pause(0.05)
            action = move_result['selected']
        else:
            print("\nAI is thinking...")
            t0 = time.time()
            action = choose_ai_move_fn(game, current_state[0])
            print(f"AI plays {action} (in {time.time() - t0:.2f}s)")
            # --- PAUSE before flips ---
            time.sleep(ai_move_pause / 2)
            draw_ai_tile(current_state[0][0], action, current_state[0][1])
            time.sleep(ai_move_pause)
        current_state[0] = game.enact(current_state[0], action)

    # --- Game Over ---
    draw_board(current_state[0])
    plt.ioff()
    board = current_state[0][0]
    white = np.sum(board == 1)
    black = np.sum(board == -1)
    if white > black:
        result_msg = f"White wins!\n {white}-{black}"
        color = "blue"
    elif black > white:
        result_msg = f"Black wins!\n {black}-{white}"
        color = "red"
    else:
        result_msg = "Draw!"
        color = "green"

    draw_board(current_state[0], result_msg)
    print(result_msg)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(3)
    plt.show()
