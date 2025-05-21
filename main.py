from players.mcts import choose_move
from othello.othello_game import OthelloGame
# from othello.othello_logic import Board
import time

PLAYER_HUMAN = 1
PLAYER_AI = -1
BOARD_SIZE = 6

# ------------ Human vs AI Gameplay ------------

def get_valid_move(legal_moves, board_size):
    """
    Prompts the user for a valid move (row,col) until a correct one is entered.
    Returns the move as a flat index.
    """
    while True:
        try:
            move_str = input("Your move (row,col): ")
            row_str, col_str = move_str.split(",")
            row = int(row_str.strip()) - 1
            col = int(col_str.strip()) - 1
            move = row * board_size + col
            if move in legal_moves:
                return move
        except (ValueError, IndexError):
            pass
        print("Invalid move. Try again.")


def play_human_vs_ai(starting):
    """
    Allows a human player to play against the AI, alternating turns until the game ends.
    """
    # ------ Can be replaced by another game ------ #
    board_size = BOARD_SIZE
    game = OthelloGame(board_size)
    # --------------------------------------------- #
    human = PLAYER_HUMAN if starting == 'h' else PLAYER_AI
    state = game.startState(human)
    game.print_board(state)

    while not game.isEnd(state):
        game.print_board(state)
        if game.player(state) == PLAYER_HUMAN:
            moves = [a for a in game.actions(state)]
            print("Legal moves:", [(m//board_size+1, m%board_size+1) for m in moves])
            action = get_valid_move(moves, board_size)
            state = game.enact(state, action)
        else:
            print("\nAI is thinking...")
            start = time.time()
            action = choose_move(game, state)
            print(f"AI plays {action} (in {time.time() - start:.2f}s)\n")
            state = game.enact(state, action)

    # Game has ended, display final result
    game.print_board(state)
    r = game.reward(state)  # +1 if X wins, -1 if O wins, 0 if draw
    if r == +1:
        print("🎉 You win!")
    elif r == -1:
        print("💻 AI wins!")
    else:
        print("🤝 Draw!")


# ------------ AI Self-Play Simulation ------------

def simulate_self_play(num_games=100):
    """
    Simulates a number of self-play games between two AI players and summarizes the results.
    """
    # ------ Can be replaced by another game ------ #
    board_size = BOARD_SIZE
    game = OthelloGame(board_size)
    # --------------------------------------------- #
    tally = {"X": 0, "O": 0, "draw": 0}

    for _ in range(num_games):
        state = game.startState()

        # AI players alternate moves until game ends
        while not game.isEnd(state):
            action = choose_move(game, state)
            state = game.enact(state, action)

        # Tally result
        r = game.reward(state)
        if r == +1:
            tally["X"] += 1
        elif r == -1:
            tally["O"] += 1
        else:
            tally["draw"] += 1

    # Print summary of results
    print(f"After {num_games} self-play games:")
    print(f"  X wins:  {tally['X']}")
    print(f"  O wins:  {tally['O']}")
    print(f"  draws:   {tally['draw']}")


# ------------ Program Entry Point ------------

def main():
    mode = input("Enter 'h' for Human start, 'a' for AI start, 's' for self-play: ").strip().lower()
    if mode in ['h', 'a']:
        play_human_vs_ai(mode)
    elif mode == 's':
        try:
            n = int(input("How many self-play games? "))
        except ValueError:
            print("Invalid number, defaulting to 50.")
            n = 50
        simulate_self_play(n)
    else:
        print("Invalid mode selected. Exiting.")

if __name__ == "__main__":
    main()