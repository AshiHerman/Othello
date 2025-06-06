import random
from players.mcts import choose_move
from othello.othello_game import OthelloGame
from othello.othello_visualizer import play_interactive
from tictactoe.tictactoe import TicTacToe

PLAYER_HUMAN = 1
PLAYER_AI = -1
BOARD_SIZE = 8

# ------------ Human vs AI Gameplay ------------

def is_human_turn(state):
    return state[1] == PLAYER_HUMAN

def play_human_vs_ai(game, starting):
    """
    Allows a human player to play against the AI, alternating turns until the game ends.
    """
    human = PLAYER_HUMAN if starting == 'h' else PLAYER_AI
    state = game.startState(human)
    play_interactive(game, state, is_human_turn, choose_move)


# ------------ AI Self-Play Simulation ------------

def simulate_self_play(game, num_games=100):
    """
    Simulates a number of self-play games between two AI players and summarizes the results.
    """
    tally = {"X": 0, "O": 0, "draw": 0}

    for i in range(num_games):
        print(f"Game #{i+1}")
        state = game.startState()

        # AI players alternate moves until game ends
        while not game.isEnd(state):
            # action = random.choice(game.actions(state))
            # state = game.enact(state, action)
            # game.print_board(state)

            action = choose_move(game, state)
            state = game.enact(state, action)
            # game.print_board(state)
        
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
    # ------ Can be replaced by another game ------ #
    board_size = BOARD_SIZE
    game = OthelloGame(board_size)
    # game = TicTacToe()
    # --------------------------------------------- #
    mode = input("Enter 'h' for Human start, 'a' for AI start, 's' for self-play: ").strip().lower()
    if mode in ['h', 'a']:
        play_human_vs_ai(game, mode)
    elif mode == 's':
        try:
            n = int(input("How many self-play games? "))
        except ValueError:
            print("Invalid number, defaulting to 50.")
            n = 50
        simulate_self_play(game, n)
    else:
        print("Invalid mode selected. Exiting.")

if __name__ == "__main__":
    main()