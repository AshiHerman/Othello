import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time
from othello.othello_game import OthelloGame, get_valid

def show_state(
    board,
    player,
    valid_moves=None,
    message=None,
    pause=True,
    close_on_click=False,
    show_valid_moves=True,
    board_color="green",
    valid_move_color=None,
    heatmap=None,
    heatmap_cmap="hot",
    heatmap_alpha=0.55,
    show_colorbar=True,
):
    n = 8
    valid_moves = get_valid(board, player)
    fig, ax = plt.subplots(figsize=(6, 6))

    # Make extra space at the bottom for the message
    fig.subplots_adjust(bottom=0.18)

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks(np.arange(n+1))
    ax.set_yticks(np.arange(n+1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, which='both', color='black', linewidth=1)

    # Set background color
    if board_color == "gray":
        board_bg = (0.92, 0.92, 0.92)
    elif board_color == "green":
        board_bg = (0.0, 0.6, 0.0)
    else:
        board_bg = board_color
    ax.set_facecolor(board_bg)

    # Optionally overlay heatmap
    if heatmap is not None:
        hm = np.array(heatmap).reshape(n, n)
        vmin, vmax = np.nanmin(hm), np.nanmax(hm)
        img = ax.imshow(
            hm, cmap=heatmap_cmap, alpha=heatmap_alpha,
            extent=(0, n, 0, n), origin='upper', vmin=vmin, vmax=vmax
        )
        if show_colorbar:
            plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

    piece_colors = {1: 'white', -1: 'black'}
    alpha_color = {1: (1.0, 1.0, 1.0, 0.4), -1: (0.0, 0.0, 0.0, 0.4)}

    if valid_move_color is None:
        available_color = alpha_color[player]
    else:
        available_color = valid_move_color

    for y in range(n):
        for x in range(n):
            piece = board[y][x]
            cx, cy = x + 0.5, n - y - 0.5
            idx = y * n + x
            if piece in piece_colors:
                ax.add_patch(patches.Circle((cx, cy), 0.4, color=piece_colors[piece], ec='black'))
            elif show_valid_moves and valid_moves is not None and valid_moves[idx] == 1:
                ax.add_patch(patches.Circle((cx, cy), 0.4, color=available_color, linewidth=0))

    ax.set_title(f"Othello - {'White' if player == 1 else 'Black'} to move")

    # --- Draw message under the plot ---
    if message:
        # Place at bottom center (x=0.5, y=0.02 in figure coords)
        fig.text(0.5, 0.05, message,
                 ha='center', va='bottom', fontsize=8, color='black', weight='bold')

    ax.set_aspect('equal')
    fig.canvas.draw()
    fig.canvas.flush_events()

        # Add hover coordinate display
    hover_text = fig.text(0.5, -0.04, "", ha='center', va='top', fontsize=14, color='purple')

    def format_hover(event):
        # event.xdata and event.ydata are float board coordinates (None if out of axes)
        if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
            # Board is from (0,0) bottom left to (n, n) top right; but origin='upper' in imshow so we flip y
            # Othello usually wants (1,1) as top-left (y=0, x=0)
            col = int(event.xdata)
            row = n - int(event.ydata)
            if 0 <= col < n and 1 <= row <= n:
                hover_text.set_text(f"(x, y) = ({col+1}, {row})")
            else:
                hover_text.set_text("")
        else:
            hover_text.set_text("")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', format_hover)

    if close_on_click:
        def on_click(event):
            plt.close(fig)
        fig.canvas.mpl_connect('button_press_event', on_click)
    if pause:
        plt.show()
    else:
        plt.pause(0.001)
    return fig, ax



def play_interactive(game : OthelloGame, initial_state, is_human_turn_fn, choose_ai_move_fn, ai_move_pause=0, guidance=None):
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
            fig.subplots_adjust(bottom=0.18)  # Make space for the message
            fig.text(
                0.5, 0.05, message,
                ha='center', va='bottom', fontsize=8, color='black', weight='bold',
                bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2')
            )
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
        # Show guidance if it's the human's turn
        message = guidance(current_state[0]) if guidance and is_human_turn_fn(current_state[0]) else None
        draw_board(current_state[0], message=message)
        if message:
            print(message)
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
            action = choose_ai_move_fn(current_state[0])
            print(f"AI plays ({(action//8)+1}, {(action%8)+1}) (in {time.time() - t0:.2f}s)")
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
        result_msg = f"\t\t\t\t\tWHITE WINS!!!\t\t\t\t\t\n\t\t\t\t\t{white}-{black}\t\t\t\t\t"
    elif black > white:
        result_msg = f"\t\t\t\t\tBLACK WINS!!!\t\t\t\t\t\n\t\t\t\t\t{black}-{white}\t\t\t\t\t"
    else:
        result_msg = "\t\t\t\t\tDraw!\t\t\t\t\t"

    draw_board(current_state[0], result_msg)
    print(result_msg)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(3)
    plt.show()
