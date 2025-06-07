import random
from players.mcts import choose_move
from othello.othello_game import OthelloGame
from othello.othello_visualizer import play_interactive

BOARD_SIZE = 4

def play_human_vs_ai(game, human_player=1):
    state = game.startState(1)
    def is_human_turn(state): return state[1] == human_player
    print(human_player)
    print(is_human_turn(state))
    play_interactive(game, state, is_human_turn, choose_move)

def play_human_vs_human(game):
    state = game.startState(1)
    def is_always_human(state): return True
    def never_called_ai_move(game, state):
        raise RuntimeError("AI should not be called in human vs human mode.")
    play_interactive(game, state, is_always_human, never_called_ai_move)

def play_ai_vs_ai(game, num_games=100):
    tally = {"1": 0, "-1": 0, "draw": 0}
    for i in range(num_games):
        state = game.startState(1)
        while not game.isEnd(state):
            action = choose_move(game, state)
            state = game.enact(state, action)
        r = game.getGameEnded(state[0], 1)
        if r == +1:
            tally["1"] += 1
        elif r == -1:
            tally["-1"] += 1
        else:
            tally["draw"] += 1
    print(f"After {num_games} AI self-play games:")
    print(f"  Player 1 wins: {tally['1']}")
    print(f"  Player -1 wins: {tally['-1']}")
    print(f"  Draws: {tally['draw']}")

def main():
    game = OthelloGame(BOARD_SIZE)
    print("Modes: 'h' = Human vs AI, 'a' = AI vs Human, '2' = Human vs Human, 's' = AI self-play")
    mode = input("Select mode: ").strip().lower()
    if mode == 'h':
        play_human_vs_ai(game, human_player=1)
    elif mode == 'a':
        play_human_vs_ai(game, human_player=-1)
    elif mode == '2':
        play_human_vs_human(game)
    elif mode == 's':
        try:
            n = int(input("How many self-play games? "))
        except ValueError:
            n = 50
        play_ai_vs_ai(game, n)
    else:
        print("Invalid mode.")

if __name__ == "__main__":
    main()
