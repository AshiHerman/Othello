from players.mcts import choose_move
from game.tictactoe import TicTacToe
import time

PLAYER_HUMAN = "X"
PLAYER_AI = "O"

# ------------ Human vs AI Gameplay ------------

def get_valid_move(legal_moves):
    """
    Prompts the user for a valid move until a correct one is entered.
    """
    while True:
        try:
            move = int(input("Your move: "))
            if move in legal_moves:
                return move - 1
        except ValueError:
            pass
        print("Invalid move. Try again.")

def play_human_vs_ai(starting):
    """
    Allows a human player to play against the AI, alternating turns until the game ends.
    """
    game = TicTacToe()
    human = PLAYER_HUMAN if starting == 'h' else PLAYER_AI
    state = game.startState(human)
    print(state)

    while not game.isEnd(state):
        if game.player(state) == PLAYER_HUMAN:
            game.print_board(state)
            moves = [a + 1 for a in game.actions(state)]
            print("Legal moves:", moves)
            action = get_valid_move(moves)
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
    game = TicTacToe()
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
