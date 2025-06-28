import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from othello.othello_game import OthelloGame

# ------------ Configuration ------------
EPISODES_PER_MOVE = 100
EXPLORATION_CONSTANT = 1.0  # c_puct in AlphaZero
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25

# Fixed neural network definitions
class PolicyV(nn.Module):
    '''
    The Value Neural Network will approximate the Value of the node, given a State of the game.
    '''
    def __init__(self, input_size=64):
        super().__init__()
        self.relu = nn.ReLU()
        self.dense1 = nn.Linear(input_size, input_size)
        self.dense2 = nn.Linear(input_size, input_size)
        self.v_out = nn.Linear(input_size, 1)
        
    def forward(self, input):               
        x = self.dense1(input)
        x = self.relu(x)
        x = self.dense2(x)       
        x = self.relu(x)
        x = self.v_out(x)
        return torch.tanh(x)  # Value should be in [-1, 1]
    

class PolicyP(nn.Module):
    '''
    The Policy Neural Network will approximate the MCTS policy for the choice of nodes, given a State of the game.
    '''  
    def __init__(self, input_size=64, num_actions=65):  # 64 board positions + 1 pass move
        super().__init__()
        self.relu = nn.ReLU()
        self.dense1 = nn.Linear(input_size, input_size)
        self.dense2 = nn.Linear(input_size, input_size)
        self.p_out = nn.Linear(input_size, num_actions)
        
    def forward(self, input):               
        x = self.dense1(input)
        x = self.relu(x)
        x = self.dense2(x)       
        x = self.relu(x)
        x = self.p_out(x)
        return F.softmax(x, dim=-1)  # Policy should be probability distribution

# ------------ AlphaZero MCTS Node Definition ------------
class AlphaZeroNode:
    def __init__(self, problem, state, parent=None, action=None, prior_prob=0.0):
        """
        AlphaZero MCTS node with neural network guidance
        """
        self.problem = problem
        self.state = state
        self.parent = parent
        self.action = action
        self.prior_prob = prior_prob  # Prior probability from policy network
        
        self.children = {}  # Dict mapping action -> child node
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        
    def is_leaf(self):
        return len(self.children) == 0
    
    def value(self):
        """Average value of this node"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, c_puct=EXPLORATION_CONSTANT):
        """
        AlphaZero UCB formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        if self.visit_count == 0:
            return float('inf')
        
        exploration_term = (c_puct * self.prior_prob * 
                          math.sqrt(self.parent.visit_count) / (1 + self.visit_count))
        
        return self.value() + exploration_term
    
    def select_child(self, c_puct=EXPLORATION_CONSTANT):
        """
        Select child with highest UCB score - FIXED VERSION with tie-breaking
        """
        if not self.children:
            return None
            
        # Calculate UCB scores for all children
        children_with_scores = [(child, child.ucb_score(c_puct)) 
                               for child in self.children.values()]
        
        # Find maximum score
        max_score = max(score for _, score in children_with_scores)
        
        # Get all children with maximum score
        best_children = [child for child, score in children_with_scores 
                        if score == max_score]
        
        # Random tie-breaking
        return np.random.choice(best_children)
    
    def expand(self, policy_network, value_network):
        """
        Expand node using neural network predictions
        """
        if self.problem.isEnd(self.state):
            return self.problem.reward(self.state)
        
        # Get state representation for neural networks
        state_tensor = self.state_to_tensor(self.state)
        
        # Get policy and value predictions
        with torch.no_grad():
            action_probs = policy_network(state_tensor).squeeze()
            value = value_network(state_tensor).item()
        
        # Get valid actions
        valid_actions = self.problem.actions(self.state)
        
        # Create child nodes for valid actions
        for action in valid_actions:
            prior_prob = action_probs[action].item()
            
            next_state = self.problem.enact(self.state, action)
            child = AlphaZeroNode(
                self.problem, next_state, parent=self, 
                action=action, prior_prob=prior_prob
            )
            self.children[action] = child
        
        # Add Dirichlet noise to root node for exploration
        if self.parent is None:
            self.add_dirichlet_noise()
        
        self.is_expanded = True
        return value
    
    def add_dirichlet_noise(self):
        """Add Dirichlet noise to root node for exploration"""
        if len(self.children) == 0:
            return
            
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(self.children))
        for i, child in enumerate(self.children.values()):
            child.prior_prob = ((1 - DIRICHLET_EPSILON) * child.prior_prob + 
                              DIRICHLET_EPSILON * noise[i])
    
    def backup(self, value):
        """Backup value through the tree"""
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent is not None:
            # For Othello, we need to flip the value for the opponent
            # The value should be from the perspective of the player to move
            self.parent.backup(-value)
    
    def state_to_tensor(self, state):
        """Convert OthelloGame state to tensor"""
        board, player = state
        
        # Convert board to canonical form (from player's perspective)
        canonical_board = self.problem.getCanonicalForm(board, player)
        
        # Flatten the board to a 1D tensor
        board_flat = canonical_board.flatten()
        
        return torch.FloatTensor(board_flat).unsqueeze(0)
    
    def action_to_index(self, action):
        """Convert action to index - for Othello, action is already the index"""
        return action

# ------------ AlphaZero MCTS Algorithm ------------
def alphazero_mcts(problem, state, policy_network, value_network, 
                   num_simulations=EPISODES_PER_MOVE, c_puct=EXPLORATION_CONSTANT):
    """
    AlphaZero MCTS algorithm - FIXED VERSION with safety checks
    """
    root = AlphaZeroNode(problem, state)
    
    for simulation in range(num_simulations):
        node = root
        path_length = 0
        MAX_PATH_LENGTH = 200  # Safety limit to prevent infinite loops
        
        # Selection: traverse tree until leaf
        while not node.is_leaf() and node.is_expanded and path_length < MAX_PATH_LENGTH:
            selected_child = node.select_child(c_puct)
            if selected_child is None:
                break
            node = selected_child
            path_length += 1
        
        # Safety check for infinite loops
        if path_length >= MAX_PATH_LENGTH:
            print(f"Warning: Maximum path length reached in simulation {simulation}")
            continue
        
        # Expansion and Evaluation
        if not node.is_expanded:
            value = node.expand(policy_network, value_network)
        else:
            # If leaf is terminal, get true value
            if problem.isEnd(node.state):
                value = problem.reward(node.state)
            else:
                # Use value network for evaluation
                state_tensor = node.state_to_tensor(node.state)
                with torch.no_grad():
                    value = value_network(state_tensor).item()
        
        # Backup
        node.backup(value)
    
    return root

# ------------ AlphaZero Move Selection ------------
class AlphaZeroMCTS:
    def __init__(self, policy_network, value_network):
        self.policy_network = policy_network
        self.value_network = value_network
    
    def choose_move(self, problem, state, num_simulations=EPISODES_PER_MOVE, 
                   temperature=1.0, c_puct=EXPLORATION_CONSTANT):
        """
        Choose move using AlphaZero MCTS
        """
        root = alphazero_mcts(problem, state, self.policy_network, 
                            self.value_network, num_simulations, c_puct)
        
        # Get visit counts for each action
        actions = list(root.children.keys())
        visit_counts = [root.children[action].visit_count for action in actions]
        
        if temperature == 0:
            # Deterministic: choose most visited action
            best_idx = np.argmax(visit_counts)
            return actions[best_idx]
        else:
            # Stochastic: sample based on visit count distribution
            visit_counts = np.array(visit_counts)
            probs = visit_counts ** (1.0 / temperature)
            probs = probs / probs.sum()
            
            chosen_idx = np.random.choice(len(actions), p=probs)
            return actions[chosen_idx]
    
    def get_action_probabilities(self, problem, state, num_simulations=EPISODES_PER_MOVE,
                               temperature=1.0, c_puct=EXPLORATION_CONSTANT):
        """
        Get action probabilities for training
        """
        root = alphazero_mcts(problem, state, self.policy_network, 
                            self.value_network, num_simulations, c_puct)
        
        actions = list(root.children.keys())
        visit_counts = np.array([root.children[action].visit_count for action in actions])
        
        if temperature == 0:
            probs = np.zeros_like(visit_counts)
            probs[np.argmax(visit_counts)] = 1.0
        else:
            probs = visit_counts ** (1.0 / temperature)
            probs = probs / probs.sum()
        
        # Convert to full action space representation (65 actions: 64 board + 1 pass)
        action_probs = np.zeros(65)
        for action, prob in zip(actions, probs):
            action_probs[action] = prob
        
        return action_probs

# ------------ Example Usage ------------
if __name__ == "__main__":
    # Initialize networks
    policy_net = PolicyP(input_size=64, num_actions=65)  # 64 board + 1 pass
    value_net = PolicyV(input_size=64)
    
    # Create AlphaZero MCTS player
    alphazero_player = AlphaZeroMCTS(policy_net, value_net)
    
    # Initialize game
    problem = OthelloGame(8)
    current_state = problem.startState(first_player=1)
    
    print("Starting AlphaZero MCTS...")
    print("Current board:")
    problem.display(current_state[0])
    
    # Choose move
    move = alphazero_player.choose_move(problem, current_state, num_simulations=400)
    print(f"AlphaZero chose move: {move}")
    
    # Get action probabilities for training
    action_probs = alphazero_player.get_action_probabilities(problem, current_state, num_simulations=400)
    print(f"Action probabilities shape: {action_probs.shape}")
    print(f"Non-zero probabilities: {np.sum(action_probs > 0)}")
    
    # Apply the move
    new_state = problem.enact(current_state, move)
    print("\nBoard after move:")
    problem.display(new_state[0])