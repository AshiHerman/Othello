from players.alphazero import AlphaZero
from players.imitator import Imitator
from othello.othello_game import OthelloGame, get_valid

import numpy as np
from othello.othello_visualizer import play_interactive, show_state

AZ = AlphaZero(OthelloGame(8))
IM = Imitator()

def move_str(move):
    return f"({(move//8)+1}, {(move%8)+1})"

def show_probs(state):
    """Show move probabilities for the current state."""
    board = state[0]
    pi = AZ.mcts.getActionProb(board, temp=1)
    
    imitator_move = IM.choose_move(state)
    alpha_move = AZ.choose_move(state)

    alpha_move_str = f"AlphaZero move: {move_str(alpha_move)}, probability: {pi[alpha_move]:.3f}"
    imitator_move_str = f"Imitator move: {move_str(imitator_move)}, probability: {pi[imitator_move]:.3f}"

    if imitator_move == alpha_move:
        alpha_move_str = "AlphaZero and Imitator agree on the move."
        imitator_move_str = "Go ahead and play it!"
    else:
        alpha_move_str = f"AlphaZero suggests: {move_str(alpha_move)} at strength: {pi[alpha_move]:.3f}"
        imitator_move_str = f"While imitator expects: {move_str(imitator_move)} (strength: {pi[imitator_move]:.3f}). Watch out!"

    # print(alpha_move_str)
    # print(imitator_move_str)
    return "\n" + alpha_move_str + "\n" + imitator_move_str




# state = OthelloGame(8).startState(1)
# print(state)
# out = show_probs(state)
# show_state(board=state[0], player=1, message=out)

