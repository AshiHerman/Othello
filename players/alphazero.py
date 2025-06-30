import sys
import os
import logging

# Add alphazero directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'alphazero'))


from alphazero.othello.pytorch.NNet import NNetWrapper
from alphazero.MCTS import MCTS as AlphaZeroMCTS
from alphazero.utils import dotdict
from alphazero.othello.OthelloGame import OthelloGame as AlphaZeroOthelloGame
ALPHAZERO_AVAILABLE = True


class AlphaZero:
    """AlphaZero AI player using the 8x8 pretrained model"""
    
    def __init__(self, game, num_sims=25, board_size=8):
        if not ALPHAZERO_AVAILABLE:
            raise ImportError("AlphaZero dependencies not available")
        
        self.game = game
        self.board_size = board_size
        
        # Suppress AlphaZero logging
        logging.getLogger().setLevel(logging.WARNING)
        
        # Create AlphaZero game instance and neural network
        self.alphazero_game = AlphaZeroOthelloGame(board_size)
        self.nnet = NNetWrapper(self.alphazero_game)
        
        # Load the best pretrained model
        self._load_pretrained_model()
        
        # Initialize MCTS
        args = dotdict({'numMCTSSims': num_sims, 'cpuct': 1.0})
        self.mcts = AlphaZeroMCTS(self.alphazero_game, self.nnet, args)
    
    def _load_pretrained_model(self):
        """Load the best 8x8 pretrained model"""
        alphazero_dir = os.path.join(os.path.dirname(__file__), '')
        model_path = os.path.join(alphazero_dir, '../alphazero/pretrained_models/othello/pytorch')
        print(f"Loading model from {model_path}")
        model_file = '8x8_100checkpoints_best.pth.tar'
        
        try:
            self.nnet.load_checkpoint(model_path, model_file)
            print("✓ Loaded AlphaZero 8x8 pretrained model")
        except Exception as e:
            print(f"⚠ Warning: Could not load pretrained model: {e}")
            print("  Using untrained network")
    
    def choose_move(self, state):
        """Choose the best move using AlphaZero MCTS + neural network"""
        player = state[1]
        board = state[0] if player == 1 else -state[0]
        pi = self.mcts.getActionProb(board, temp=1)  # temp=0 for deterministic play
        # print(f"AlphaZero move probabilities: {pi}")
        return max(range(len(pi)), key=lambda i: pi[i])
    
    @staticmethod
    def is_available():
        """Check if AlphaZero can be used"""
        return ALPHAZERO_AVAILABLE