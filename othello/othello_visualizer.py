import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from .othello_game import OthelloGame

def play_interactive(game, initial_state, is_human_turn_fn, choose_ai_move_fn):
    """
    Starts and manages a persistent GUI game loop for Human vs AI Othello.

    Params:
    - game: OthelloGame instance
    - initial_state: (board, player)
    - is_human_turn_fn(state): returns True if human plays
    - choose_ai_move_fn(game, state): returns AI's move
    """
    plt.ion()  # turn on interactive mode
    board, player = initial_state
    n = game.n
    fig, ax = plt.subplots(figsize=(6, 6))
    move_result = {'selected': None}

    def draw_board(state):
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
        ax.set_aspect('equal')
        fig.canvas.draw()
        fig.canvas.flush_events()

    def onclick(event):
        if not is_human_turn_fn(current_state[0]):
            return
        x, y = int(event.xdata), int(event.ydata)
        row, col = n - y - 1, x
        idx = row * n + col
        valids = game.getValidMoves(current_state[0][0], current_state[0][1])
        if 0 <= row < n and 0 <= col < n and valids[idx] == 1:
            move_result['selected'] = idx

    fig.canvas.mpl_connect('button_press_event', onclick)

    current_state = [initial_state]

    while not game.isEnd(current_state[0]):
        draw_board(current_state[0])

        if is_human_turn_fn(current_state[0]):
            move_result['selected'] = None
            while move_result['selected'] is None:
                plt.pause(0.05)
            action = move_result['selected']
        else:
            print("\nAI is thinking...")
            import time
            t0 = time.time()
            action = choose_ai_move_fn(game, current_state[0])
            print(f"AI plays {action} (in {time.time() - t0:.2f}s)")

        current_state[0] = game.enact(current_state[0], action)

    draw_board(current_state[0])
    plt.ioff()
    # game.print_board(current_state[0])
    r = game.reward(current_state[0])
    if r == +1:
        print("🎉 You win!")
    elif r == -1:
        print("💻 AI wins!")
    else:
        print("🤝 Draw!")
    
        draw_board(current_state[0])

    # Show final outcome
    r = game.reward(current_state[0])
    if r == +1:
        text = "You Win!"
        color = "blue"
    elif r == -1:
        text = "AI Wins!"
        color = "blue"
    else:
        text = "Draw!"
        color = "blue"

    # Overlay the message
    ax.text(0.5, 0.5, text,
            transform=ax.transAxes,
            fontsize=32,
            ha='center',
            va='center',
            color=color,
            weight='bold')

    fig.canvas.draw()
    fig.canvas.flush_events()

    # Let it sit for a few seconds
    import time
    time.sleep(3)

    plt.ioff()
    plt.show()  # keep window open until user closes it manually




# def show_board(state, game):
#     """
#     Displays the Othello board and returns the clicked move index.

#     Parameters:
#     - state: tuple (board, player)
#     - game: OthelloGame instance

#     Returns:
#     - move: flat index of the selected move (int)
#     """
#     board, player = state
#     n = game.n
#     valids = game.getValidMoves(board, player)

#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.set_xlim(0, n)
#     ax.set_ylim(0, n)
#     ax.set_xticks(np.arange(n+1))
#     ax.set_yticks(np.arange(n+1))
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.grid(True, which='both', color='black', linewidth=1)
#     ax.set_facecolor((0.0, 0.6, 0.0))

#     piece_colors = {1: 'white', -1: 'black'}
#     alpha_color = {1: (1.0, 1.0, 1.0, 0.4), -1: (0.0, 0.0, 0.0, 0.4)}

#     for y in range(n):
#         for x in range(n):
#             piece = board[y][x]
#             cx, cy = x + 0.5, n - y - 0.5

#             if piece in piece_colors:
#                 ax.add_patch(patches.Circle((cx, cy), 0.4, color=piece_colors[piece], ec='black'))
#             idx = y * n + x
#             if valids[idx] == 1 and piece == 0:
#                 ax.add_patch(patches.Circle((cx, cy), 0.4, color=alpha_color[player], linewidth=0))

#     coord_text = ax.text(0.05, 1.01, '', transform=ax.transAxes, fontsize=12)
#     move_selected = {'index': None}

#     def on_motion(event):
#         if event.inaxes != ax:
#             coord_text.set_text('')
#             fig.canvas.draw_idle()
#             return
#         x, y = int(event.xdata), int(event.ydata)
#         if 0 <= x < n and 0 <= y < n:
#             board_y = n - y
#             coord_text.set_text(f"Hovering: ({board_y},{x + 1})")  # 1-indexed
#         else:
#             coord_text.set_text('')
#         fig.canvas.draw_idle()

#     def on_click(event):
#         if event.inaxes != ax:
#             return
#         x, y = int(event.xdata), int(event.ydata)
#         row, col = n - y - 1, x
#         idx = row * n + col
#         if 0 <= row < n and 0 <= col < n and valids[idx] == 1:
#             move_selected['index'] = idx
#             plt.close(fig)

#     fig.canvas.mpl_connect('motion_notify_event', on_motion)
#     fig.canvas.mpl_connect('button_press_event', on_click)

#     ax.set_aspect('equal')
#     plt.title(f"Othello - {'White' if player == 1 else 'Black'} to move")
#     plt.tight_layout()
#     plt.show()

#     return move_selected['index']
