# training/alphazero_trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from collections import deque
from typing import List, Tuple, Dict, Any, Optional
import logging
import time
from dataclasses import dataclass

# Import the AlphaZero components
from players.alphazero import AlphaZero, PolicyValueNetwork, MCTS

@dataclass
class TrainingConfig:
    """Configuration for AlphaZero training."""
    episodes_per_move: int = 100
    exploration_constant: float = 1.0  # c_puct
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    learning_rate: float = 0.001
    batch_size: int = 32
    training_epochs: int = 10
    evaluation_games: int = 100
    max_buffer_size: int = 100000
    temperature_threshold: int = 15  # Moves after which temperature becomes 0
    update_threshold: float = 0.6  # Win rate needed to update model
    checkpoint_frequency: int = 10  # Save every N iterations


class ExperienceBuffer:
    """Circular buffer for storing training experiences."""
    
    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add(self, state: np.ndarray, action_probs: np.ndarray, value: float):
        """Add experience to buffer."""
        self.buffer.append((state.copy(), action_probs.copy(), value))
    
    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample random batch from buffer."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states = []
        action_probs = []
        values = []
        
        for idx in indices:
            state, probs, value = self.buffer[idx]
            states.append(state)
            action_probs.append(probs)
            values.append(value)
        
        return (np.array(states), np.array(action_probs), np.array(values))
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()


class AlphaZeroTrainer:
    """Complete training system for AlphaZero."""
    
    def __init__(self, game_class=None, config: TrainingConfig = None, 
                 model_dir: str = 'models', data_dir: str = 'data',
                 device: str = None):
        """
        Initialize AlphaZero trainer.
        
        Args:
            game_class: Game class (e.g., OthelloGame)
            config: Training configuration
            model_dir: Directory for saving models
            data_dir: Directory for saving training data
            device: Device to use ('cpu' or 'cuda')
        """
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Configuration
        self.config = config if config else TrainingConfig()
        
        # Directories
        self.model_dir = model_dir
        self.data_dir = data_dir
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        # Game initialization
        if game_class is None:
            try:
                from othello import OthelloGame
                self.game_class = OthelloGame
            except ImportError:
                raise ImportError("Game class must be provided or OthelloGame must be available")
        else:
            self.game_class = game_class
        
        self.game = self.game_class(8)  # 8x8 Othello
        
        # Initialize networks
        self.current_player = AlphaZero(device=self.device)
        self.best_player = AlphaZero(device=self.device)
        
        # Copy initial weights
        self.best_player.network.load_state_dict(self.current_player.network.state_dict())
        
        # Training components
        self.optimizer = optim.Adam(self.current_player.network.parameters(), 
                                  lr=self.config.learning_rate, weight_decay=1e-4)
        self.experience_buffer = ExperienceBuffer(self.config.max_buffer_size)
        
        # Training tracking
        self.training_history = {
            'iterations': [],
            'policy_losses': [],
            'value_losses': [],
            'total_losses': [],
            'win_rates': [],
            'buffer_sizes': []
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def generate_self_play_game(self, temperature_threshold: int = 15) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Generate a single self-play game and collect training data.
        
        Args:
            temperature_threshold: Move number after which to use temperature=0
            
        Returns:
            List of (state, action_probs, value) tuples
        """
        game_data = []
        state = self.game.startState()
        move_count = 0
        
        while not self.game.isEnd(state):
            move_count += 1
            
            # Use temperature for exploration in early moves
            temperature = 1.0 if move_count <= temperature_threshold else 0.0
            
            # Get canonical form for current player
            board, current_player = state
            canonical_board = self.game.getCanonicalForm(board, current_player)
            
            # Get action probabilities from current player
            action_probs = self.current_player.get_action_probabilities(
                self.game, state, temperature=temperature
            )
            
            # Store training example (we'll set values at the end)
            game_data.append((canonical_board.flatten(), action_probs, 0))
            
            # Sample action from probabilities
            legal_actions = self.game.actions(state)
            legal_action_probs = np.zeros_like(action_probs)
            for action in legal_actions:
                legal_action_probs[action] = action_probs[action]
            
            # Normalize probabilities
            if legal_action_probs.sum() > 0:
                legal_action_probs /= legal_action_probs.sum()
                action = np.random.choice(len(legal_action_probs), p=legal_action_probs)
            else:
                action = np.random.choice(legal_actions)
            
            # Apply action
            state = self.game.enact(state, action)
        
        # Get final game result and assign values
        board, final_player = state
        game_result = self.game.reward(state, 1)  # Get result for player 1
        
        # Assign values based on game outcome from each player's perspective
        final_game_data = []
        for i, (board_state, probs, _) in enumerate(game_data):
            # Determine which player made this move
            player_turn = 1 if i % 2 == 0 else -1
            # Value is game result from this player's perspective
            value = game_result * player_turn
            final_game_data.append((board_state, probs, value))
        
        return final_game_data
    
    def collect_self_play_data(self, num_games: int) -> int:
        """
        Collect self-play training data.
        
        Args:
            num_games: Number of self-play games to generate
            
        Returns:
            Number of training examples collected
        """
        self.logger.info(f"Collecting {num_games} self-play games...")
        
        total_examples = 0
        for game_idx in range(num_games):
            if game_idx % 10 == 0:
                self.logger.info(f"Generated {game_idx}/{num_games} games")
            
            game_data = self.generate_self_play_game(self.config.temperature_threshold)
            
            # Add to experience buffer
            for state, action_probs, value in game_data:
                self.experience_buffer.add(state, action_probs, value)
                total_examples += 1
        
        self.logger.info(f"Collected {total_examples} training examples")
        return total_examples
    
    def train_network(self) -> Tuple[float, float, float]:
        """
        Train the neural network on collected data.
        
        Returns:
            Tuple of (policy_loss, value_loss, total_loss)
        """
        if len(self.experience_buffer) < self.config.batch_size:
            self.logger.warning("Not enough data for training")
            return 0.0, 0.0, 0.0
        
        self.current_player.network.train()
        
        policy_losses = []
        value_losses = []
        total_losses = []
        
        num_batches = max(1, len(self.experience_buffer) // self.config.batch_size)
        
        for epoch in range(self.config.training_epochs):
            for batch_idx in range(num_batches):
                # Sample batch
                states, target_probs, target_values = self.experience_buffer.sample_batch(
                    self.config.batch_size
                )
                
                # Convert to tensors
                states_tensor = torch.FloatTensor(states).to(self.device)
                target_probs_tensor = torch.FloatTensor(target_probs).to(self.device)
                target_values_tensor = torch.FloatTensor(target_values).to(self.device)
                
                # Forward pass
                policy_logits, predicted_values = self.current_player.network(states_tensor)
                predicted_values = predicted_values.squeeze()
                
                # Calculate losses
                policy_loss = F.cross_entropy(policy_logits, target_probs_tensor)
                value_loss = F.mse_loss(predicted_values, target_values_tensor)
                total_loss = policy_loss + value_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.current_player.network.parameters(), 1.0)
                
                self.optimizer.step()
                
                # Track losses
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                total_losses.append(total_loss.item())
        
        avg_policy_loss = np.mean(policy_losses)
        avg_value_loss = np.mean(value_losses)
        avg_total_loss = np.mean(total_losses)
        
        self.logger.info(f"Training - Policy Loss: {avg_policy_loss:.4f}, "
                        f"Value Loss: {avg_value_loss:.4f}, Total Loss: {avg_total_loss:.4f}")
        
        return avg_policy_loss, avg_value_loss, avg_total_loss
    
    def evaluate_players(self, num_games: int) -> float:
        """
        Evaluate current player against best player.
        
        Args:
            num_games: Number of evaluation games to play
            
        Returns:
            Win rate of current player against best player
        """
        self.logger.info(f"Evaluating players over {num_games} games...")
        
        self.current_player.set_evaluation_mode()
        self.best_player.set_evaluation_mode()
        
        current_wins = 0
        draws = 0
        
        for game_idx in range(num_games):
            # Alternate who goes first
            if game_idx % 2 == 0:
                player1, player2 = self.current_player, self.best_player
            else:
                player1, player2 = self.best_player, self.current_player
            
            result = self.play_evaluation_game(player1, player2)
            
            # Count wins for current player
            if game_idx % 2 == 0:  # Current player was player 1
                if result > 0:
                    current_wins += 1
                elif result == 0:
                    draws += 1
            else:  # Current player was player 2
                if result < 0:
                    current_wins += 1
                elif result == 0:
                    draws += 1
        
        win_rate = (current_wins + 0.5 * draws) / num_games
        self.logger.info(f"Current player win rate: {win_rate:.3f} "
                        f"({current_wins} wins, {draws} draws)")
        
        return win_rate
    
    def play_evaluation_game(self, player1: AlphaZero, player2: AlphaZero) -> float:
        """
        Play a single evaluation game between two players.
        
        Args:
            player1: First player
            player2: Second player
            
        Returns:
            Game result from player1's perspective (1=win, 0=draw, -1=loss)
        """
        state = self.game.startState()
        
        while not self.game.isEnd(state):
            board, current_player = state
            
            # Choose player based on current turn
            if current_player == 1:
                action = player1.choose_move(self.game, state)
            else:
                action = player2.choose_move(self.game, state)
            
            # Apply action
            state = self.game.enact(state, action)
        
        # Get final result for player 1
        return self.game.reward(state, 1)
    
    def save_checkpoint(self, iteration: int):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(self.model_dir, f'checkpoint_iter_{iteration}.pth')
        best_model_path = os.path.join(self.model_dir, 'best_model.pth')
        
        # Save current model checkpoint
        self.current_player.save_model(checkpoint_path)
        
        # Save best model
        self.best_player.save_model(best_model_path)
        
        # Save training history
        history_path = os.path.join(self.data_dir, 'training_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(self.training_history, f)
        
        self.logger.info(f"Saved checkpoint at iteration {iteration}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        if os.path.exists(checkpoint_path):
            self.current_player.load_model(checkpoint_path)
            self.best_player.load_model(checkpoint_path)
            self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        # Load training history
        history_path = os.path.join(self.data_dir, 'training_history.pkl')
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                self.training_history = pickle.load(f)
    
    def plot_training_progress(self):
        """Plot training progress."""
        if not self.training_history['iterations']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss plots
        iterations = self.training_history['iterations']
        
        # Policy loss
        axes[0, 0].plot(iterations, self.training_history['policy_losses'])
        axes[0, 0].set_title('Policy Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Value loss
        axes[0, 1].plot(iterations, self.training_history['value_losses'])
        axes[0, 1].set_title('Value Loss')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # Total loss
        axes[1, 0].plot(iterations, self.training_history['total_losses'])
        axes[1, 0].set_title('Total Loss')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
        
        # Win rate
        axes[1, 1].plot(iterations, self.training_history['win_rates'])
        axes[1, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random')
        axes[1, 1].axhline(y=self.config.update_threshold, color='g', linestyle='--', 
                          alpha=0.7, label='Update Threshold')
        axes[1, 1].set_title('Win Rate Against Previous Best')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Win Rate')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.data_dir, 'training_progress.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Training progress plot saved to {plot_path}")
    
    def train(self, iterations: int, games_per_iteration: int = 25) -> Dict[str, Any]:
        """
        Main training loop for AlphaZero.
        
        Args:
            iterations: Number of training iterations
            games_per_iteration: Number of self-play games per iteration
            
        Returns:
            Dictionary with training statistics
        """
        self.logger.info(f"Starting AlphaZero training for {iterations} iterations")
        self.logger.info(f"Games per iteration: {games_per_iteration}")
        self.logger.info(f"Device: {self.device}")
        
        start_time = time.time()
        
        for iteration in range(1, iterations + 1):
            iter_start_time = time.time()
            
            self.logger.info(f"\n=== Iteration {iteration}/{iterations} ===")
            
            # 1. Self-play data collection
            num_examples = self.collect_self_play_data(games_per_iteration)
            
            # 2. Train neural network
            policy_loss, value_loss, total_loss = self.train_network()
            
            # 3. Evaluate current player against best player
            win_rate = self.evaluate_players(self.config.evaluation_games)
            
            # 4. Update best player if current player performs well
            if win_rate >= self.config.update_threshold:
                self.logger.info("Current player beats best player! Updating best model...")
                self.best_player.network.load_state_dict(
                    self.current_player.network.state_dict()
                )
            else:
                self.logger.info("Current player did not beat best player. Keeping best model.")
            
            # 5. Record training history
            self.training_history['iterations'].append(iteration)
            self.training_history['policy_losses'].append(policy_loss)
            self.training_history['value_losses'].append(value_loss)
            self.training_history['total_losses'].append(total_loss)
            self.training_history['win_rates'].append(win_rate)
            self.training_history['buffer_sizes'].append(len(self.experience_buffer))
            
            # 6. Save checkpoint
            if iteration % self.config.checkpoint_frequency == 0:
                self.save_checkpoint(iteration)
            
            # 7. Plot progress
            if iteration % (self.config.checkpoint_frequency * 2) == 0:
                self.plot_training_progress()
            
            iter_time = time.time() - iter_start_time
            self.logger.info(f"Iteration {iteration} completed in {iter_time:.2f}s")
            
            # Memory cleanup
            if iteration % 50 == 0:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Final save and plot
        self.save_checkpoint(iterations)
        self.plot_training_progress()
        
        total_time = time.time() - start_time
        self.logger.info(f"\nTraining completed in {total_time:.2f}s")
        
        return {
            'total_time': total_time,
            'final_buffer_size': len(self.experience_buffer),
            'history': self.training_history
        }
    
    def resume_training(self, checkpoint_path: str, additional_iterations: int, 
                       games_per_iteration: int = 25) -> Dict[str, Any]:
        """
        Resume training from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            additional_iterations: Number of additional iterations to train
            games_per_iteration: Number of self-play games per iteration
            
        Returns:
            Dictionary with training statistics
        """
        self.logger.info(f"Resuming training from {checkpoint_path}")
        self.load_checkpoint(checkpoint_path)
        
        current_iteration = len(self.training_history['iterations'])
        self.logger.info(f"Resuming from iteration {current_iteration}")
        
        return self.train(additional_iterations, games_per_iteration)
    
    def benchmark_player(self, player_path: str, num_games: int = 100) -> Dict[str, float]:
        """
        Benchmark a trained player against the current best player.
        
        Args:
            player_path: Path to player model
            num_games: Number of games to play
            
        Returns:
            Dictionary with benchmark results
        """
        # Load the player to benchmark
        benchmark_player = AlphaZero(model_path=player_path, device=self.device)
        benchmark_player.set_evaluation_mode()
        self.best_player.set_evaluation_mode()
        
        self.logger.info(f"Benchmarking {player_path} against best player")
        
        wins = 0
        draws = 0
        losses = 0
        
        for game_idx in range(num_games):
            if game_idx % 2 == 0:
                player1, player2 = benchmark_player, self.best_player
            else:
                player1, player2 = self.best_player, benchmark_player
            
            result = self.play_evaluation_game(player1, player2)
            
            # Count results for benchmark player
            if game_idx % 2 == 0:  # Benchmark player was player 1
                if result > 0:
                    wins += 1
                elif result == 0:
                    draws += 1
                else:
                    losses += 1
            else:  # Benchmark player was player 2
                if result < 0:
                    wins += 1
                elif result == 0:
                    draws += 1
                else:
                    losses += 1
        
        win_rate = (wins + 0.5 * draws) / num_games
        
        results = {
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'win_rate': win_rate,
            'total_games': num_games
        }
        
        self.logger.info(f"Benchmark results: {results}")
        return results


# Utility functions for easy usage
def train_new_model(iterations: int = 100, games_per_iteration: int = 25, 
                   model_dir: str = 'models', data_dir: str = 'data') -> AlphaZeroTrainer:
    """
    Train a new AlphaZero model from scratch.
    
    Args:
        iterations: Number of training iterations
        games_per_iteration: Number of self-play games per iteration
        model_dir: Directory to save models
        data_dir: Directory to save training data
        
    Returns:
        Trained AlphaZeroTrainer instance
    """
    trainer = AlphaZeroTrainer(model_dir=model_dir, data_dir=data_dir)
    trainer.train(iterations, games_per_iteration)
    return trainer


def resume_training_from_checkpoint(checkpoint_path: str, iterations: int = 50, 
                                  games_per_iteration: int = 25) -> AlphaZeroTrainer:
    """
    Resume training from a saved checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        iterations: Number of additional iterations
        games_per_iteration: Number of self-play games per iteration
        
    Returns:
        AlphaZeroTrainer instance
    """
    trainer = AlphaZeroTrainer()
    trainer.resume_training(checkpoint_path, iterations, games_per_iteration)
    return trainer