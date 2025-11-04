"""
guidance.py - Provides move guidance and probability estimates for Othello using AlphaZero and Imitator models.
"""

from players.az import AlphaZero
from players.imit import Imitator
from othello.othello_game import OthelloGame

# Initialize game and AI models
GAME = OthelloGame(8)
AZ = AlphaZero()
IM = Imitator()

def move_str(move):
    """Convert move index to board coordinates (row, col)."""
    return f"({(move//8)+1}, {(move%8)+1})"

def get_win_prob(state, action):
    """Estimate win probability for a given state and action using AlphaZero's neural net."""
    next_state = GAME.enact(state, action)
    v = AZ.nnet.predict(next_state[0]*next_state[1])[1][0]
    return f"{(1-((v + 1) / 2)):.3f}"

def show_probs(state):
    """Show move suggestions and win probabilities for the current state from both models."""
    imitator_move = IM.choose_move(state)
    alpha_move = AZ.choose_move(state)

    # pi = AZ.mcts.getActionProb(board, temp=1)
    # alpha_move_str = f"AlphaZero move: {move_str(alpha_move)}, probability: {pi[alpha_move]:.3f}"
    # imitator_move_str = f"Imitator move: {move_str(imitator_move)}, probability: {pi[imitator_move]:.3f}"

    if imitator_move == alpha_move:
        alpha_move_str = f"                          Both models expect: {move_str(alpha_move)}                           "#               AlphaZero and Imitator agree on the move.               "
        imitator_move_str = f"                         With win probability: {get_win_prob(state, alpha_move)}                        "#               Go ahead and play it!               "
    else:
        alpha_move_str = f"AlphaZero suggests: {move_str(alpha_move)} with win probability: {get_win_prob(state, alpha_move)}"
        imitator_move_str = f"While imitator expects: {move_str(imitator_move)} with win probability: {get_win_prob(state, imitator_move)}"

    # print(alpha_move_str)
    # print(imitator_move_str)
    return "\n" + alpha_move_str + "\n" + imitator_move_str




# state = OthelloGame(8).startState(1)
# print(state)
# out = show_probs(state)
# show_state(board=state[0], player=1, message=out)

