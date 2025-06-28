# players/alphazero.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pickle
import os
from typing import Tuple, List, Optional, Dict, Any

class PolicyValueNetwork(nn.Module):
    """Combined policy and value network for AlphaZero."""
    
    def __init__(self, input_size: int = 64, hidden_size: int = 64, action_size: int = 65):
        super(PolicyValueNetwork, self).__init__()
        
        # Shared layers
        self.shared1 = nn.Linear(input_size, hidden_size)
        self.shared2 = nn.Linear(hidden_size, hidden_size)
        
        # Policy head (outputs action probabilities)
        self.policy_head = nn.Linear(hidden_size, action_size)
        
        # Value head (outputs state value)
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 64)
            
        Returns:
            Tuple of (policy_logits, value) where:
            - policy_logits: Shape (batch_size, 65)
            - value: Shape (batch_size, 1)
        """
        # Shared layers with ReLU activation
        shared = F.relu(self.shared1(x))
        shared = F.relu(self.shared2(shared))
        
        # Policy head (raw logits)
        policy_logits = self.policy_head(shared)
        
        # Value head with tanh activation to bound output to [-1, 1]
        value = torch.tanh(self.value_head(shared))
        
        return policy_logits, value


class MCTSNode:
    """Node in the Monte Carlo Tree Search."""
    
    def __init__(self, state: Any, parent: Optional['MCTSNode'] = None, 
                 action: Optional[int] = None, prior_prob: float = 0.0):
        self.state = state
        self.parent = parent
        self.action = action  # Action that led to this node
        self.prior_prob = prior_prob  # P(s,a) from neural network
        
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, 'MCTSNode'] = {}
        self.is_expanded = False
    
    def get_value(self) -> float:
        """Get average value Q(s,a) for this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def get_ucb_score(self, c_puct: float, parent_visits: int) -> float:
        """Calculate UCB score for node selection."""
        if self.visit_count == 0:
            return float('inf')  # Prioritize unvisited nodes
        
        exploration_term = c_puct * self.prior_prob * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.get_value() + exploration_term
    
    def select_child(self, c_puct: float) -> 'MCTSNode':
        """Select child with highest UCB score."""
        if not self.children:
            raise ValueError("Cannot select child from leaf node")
        
        best_score = float('-inf')
        best_child = None
        
        # Randomize order for tie-breaking
        children_items = list(self.children.items())
        np.random.shuffle(children_items)
        
        for action, child in children_items:
            score = child.get_ucb_score(c_puct, self.visit_count)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def expand(self, game: Any, action_probs: np.ndarray):
        """Expand node by adding all legal children."""
        if self.is_expanded:
            return
        
        legal_actions = game.actions(self.state)
        
        for action in legal_actions:
            if action < len(action_probs):
                prior_prob = action_probs[action]
                new_state = game.enact(self.state, action)
                self.children[action] = MCTSNode(
                    state=new_state,
                    parent=self,
                    action=action,
                    prior_prob=prior_prob
                )
        
        self.is_expanded = True
    
    def backup(self, value: float):
        """Backup value through the path to root."""
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent is not None:
            # Value is from perspective of current player, flip for parent
            self.parent.backup(-value)


class MCTS:
    """Monte Carlo Tree Search implementation for AlphaZero."""
    
    def __init__(self, game: Any, network: PolicyValueNetwork, 
                 c_puct: float = 1.0, device: str = 'cpu'):
        self.game = game
        self.network = network
        self.c_puct = c_puct
        self.device = device
    
    def search(self, root_state: Any, num_simulations: int, 
               add_noise: bool = False, dirichlet_alpha: float = 0.3, 
               dirichlet_epsilon: float = 0.25) -> MCTSNode:
        """
        Perform MCTS search from root state.
        
        Args:
            root_state: Initial game state
            num_simulations: Number of MCTS simulations to run
            add_noise: Whether to add Dirichlet noise to root (for training)
            dirichlet_alpha: Alpha parameter for Dirichlet noise
            dirichlet_epsilon: Mixing coefficient for Dirichlet noise
            
        Returns:
            Root node with expanded tree
        """
        # Create root node
        root = MCTSNode(state=root_state)
        
        # Expand root node
        board, player = root_state
        canonical_board = self.game.getCanonicalForm(board, player)
        action_probs, _ = self._predict(canonical_board)
        
        # Add Dirichlet noise to root for exploration during training
        if add_noise:
            legal_actions = self.game.actions(root_state)
            noise = np.random.dirichlet([dirichlet_alpha] * len(legal_actions))
            noise_dict = {action: noise[i] for i, action in enumerate(legal_actions)}
            
            for i in range(len(action_probs)):
                if i in noise_dict:
                    action_probs[i] = (1 - dirichlet_epsilon) * action_probs[i] + \
                                    dirichlet_epsilon * noise_dict[i]
        
        root.expand(self.game, action_probs)
        
        # Run simulations
        for _ in range(num_simulations):
            self._simulate(root)
        
        return root
    
    def _simulate(self, root: MCTSNode):
        """Run a single MCTS simulation."""
        path = []
        node = root
        
        # Selection: traverse to leaf node
        while node.is_expanded and node.children:
            if self.game.isEnd(node.state):
                break
            node = node.select_child(self.c_puct)
            path.append(node)
        
        # Check if game is terminal
        if self.game.isEnd(node.state):
            # Terminal node - use game outcome
            board, player = node.state
            value = self.game.reward((node.state, player))
        else:
            # Leaf node - expand and evaluate
            board, player = node.state
            canonical_board = self.game.getCanonicalForm(board, player)
            action_probs, value = self._predict(canonical_board)
            
            # Expand node
            node.expand(self.game, action_probs)
            value = value.item()
        
        # Backup values
        node.backup(value)
    
    def _predict(self, board: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        """Get neural network predictions for board state."""
        self.network.eval()
        with torch.no_grad():
            board_tensor = torch.FloatTensor(board.flatten()).unsqueeze(0).to(self.device)
            policy_logits, value = self.network(board_tensor)
            
            # Convert policy logits to probabilities
            action_probs = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
            
        return action_probs, value.squeeze()


class AlphaZero:
    """AlphaZero player implementation for Othello."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = None):
        """
        Initialize AlphaZero player.
        
        Args:
            model_path: Path to saved model file. If None, creates new model.
            device: Device to run on ('cpu' or 'cuda'). Auto-detects if None.
        """
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize network
        self.network = PolicyValueNetwork()
        self.network.to(self.device)
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # MCTS parameters
        self.episodes_per_move = 100
        self.c_puct = 1.0
        
        # Import game class (assuming it's available in the environment)
        try:
            from othello import OthelloGame
            self.game_class = OthelloGame
        except ImportError:
            # Fallback - game will be passed to choose_move
            self.game_class = None
    
    def choose_move(self, game: Any, state: Any) -> int:
        """
        Choose the best move for the current state.
        
        Args:
            game: Game instance with methods actions(), enact(), isEnd(), reward()
            state: Current game state (board, current_player)
            
        Returns:
            Selected action (integer 0-64)
        """
        # Create MCTS instance
        mcts = MCTS(game, self.network, self.c_puct, self.device)
        
        # Run MCTS search
        root = mcts.search(state, self.episodes_per_move, add_noise=False)
        
        # Select action with highest visit count
        legal_actions = game.actions(state)
        
        if not legal_actions:
            return 64  # Pass move
        
        best_action = None
        best_visits = -1
        
        for action in legal_actions:
            if action in root.children:
                visits = root.children[action].visit_count
                if visits > best_visits:
                    best_visits = visits
                    best_action = action
        
        return best_action if best_action is not None else legal_actions[0]
    
    def get_action_probabilities(self, game: Any, state: Any, temperature: float = 1.0) -> np.ndarray:
        """
        Get action probabilities for training data collection.
        
        Args:
            game: Game instance
            state: Current game state
            temperature: Temperature for action selection (0 = deterministic)
            
        Returns:
            Array of action probabilities
        """
        # Create MCTS instance
        mcts = MCTS(game, self.network, self.c_puct, self.device)
        
        # Run MCTS search with exploration noise
        root = mcts.search(state, self.episodes_per_move, add_noise=True)
        
        # Get visit counts for all actions
        action_probs = np.zeros(65)  # 64 board positions + 1 pass
        
        total_visits = sum(child.visit_count for child in root.children.values())
        
        if total_visits == 0:
            # Fallback to uniform distribution over legal actions
            legal_actions = game.actions(state)
            for action in legal_actions:
                action_probs[action] = 1.0 / len(legal_actions)
        else:
            for action, child in root.children.items():
                if temperature == 0:
                    # Deterministic selection
                    action_probs[action] = child.visit_count / total_visits
                else:
                    # Temperature-based selection
                    action_probs[action] = (child.visit_count ** (1.0 / temperature)) / \
                                         sum(c.visit_count ** (1.0 / temperature) 
                                             for c in root.children.values())
        
        return action_probs
    
    def save_model(self, path: str):
        """Save model to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'model_config': {
                'input_size': 64,
                'hidden_size': 64,
                'action_size': 65
            }
        }, path)
    
    def load_model(self, path: str):
        """Load model from file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.network.to(self.device)
    
    def set_evaluation_mode(self):
        """Set network to evaluation mode."""
        self.network.eval()
    
    def set_training_mode(self):
        """Set network to training mode."""
        self.network.train()


# Utility function for easy import
def create_alphazero_player(model_path: str = None) -> AlphaZero:
    """
    Convenience function to create AlphaZero player.
    
    Args:
        model_path: Path to saved model. If None, creates new model.
        
    Returns:
        AlphaZero player instance
    """
    return AlphaZero(model_path=model_path)