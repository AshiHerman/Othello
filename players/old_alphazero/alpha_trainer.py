import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# Assuming your AlphaZero implementation is in alphazero_mcts.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from players.alphazero import PolicyP, PolicyV, AlphaZeroMCTS
from othello.othello_game import OthelloGame

class AlphaZeroTrainer:
    def __init__(self, game, lr=0.001, dropout=0.3, epochs=10, batch_size=64, 
                 temp_threshold=15, update_threshold=0.6, maxlen_queue=200000,
                 num_mcts_sims=2, arena_compare=40, cpuct=1.0): # sims -> 25
        
        self.game = game
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        
        # Neural network parameters
        self.lr = lr
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Training parameters
        self.temp_threshold = temp_threshold
        self.update_threshold = update_threshold
        self.maxlen_queue = maxlen_queue
        self.num_mcts_sims = num_mcts_sims
        self.arena_compare = arena_compare
        self.cpuct = cpuct
        
        # Initialize networks
        self.policy_net = PolicyP(input_size=64, num_actions=65)
        self.value_net = PolicyV(input_size=64)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)
        
        # Training data storage
        self.training_examples = deque([], maxlen=self.maxlen_queue)
        
        # Create MCTS player
        self.mcts = AlphaZeroMCTS(self.policy_net, self.value_net)
        
    def execute_episode(self):
        """
        Execute one episode of self-play and return training examples
        """
        training_examples = []
        state = self.game.startState(1)
        episode_step = 0
        
        while not self.game.isEnd(state):
            episode_step += 1
            canonical_state = (self.game.getCanonicalForm(state[0], state[1]), state[1])
            
            # Get action probabilities from MCTS
            temp = int(episode_step < self.temp_threshold)
            action_probs = self.mcts.get_action_probabilities(
                self.game, canonical_state, 
                num_simulations=self.num_mcts_sims, 
                temperature=temp
            )
            
            # Store training example
            board_input = canonical_state[0].flatten()
            training_examples.append([board_input, action_probs, None])
            
            # Choose action based on probabilities
            action = np.random.choice(len(action_probs), p=action_probs)
            state = self.game.enact(state, action)
        
        # Get final reward and assign to all examples
        final_reward = self.game.reward(state)
        
        # Assign rewards to training examples (from perspective of each player)
        for i, example in enumerate(training_examples):
            # Reward alternates sign for each move (opponent's perspective)
            player_reward = final_reward * ((-1) ** i)
            example[2] = player_reward
            
        return training_examples
    
    def learn(self, examples):
        """
        Train the neural networks on a batch of examples
        """
        random.shuffle(examples)
        
        policy_losses = []
        value_losses = []
        
        for epoch in range(self.epochs):
            epoch_policy_loss = 0
            epoch_value_loss = 0
            batch_count = 0
            
            # Mini-batch training
            for i in range(0, len(examples), self.batch_size):
                batch = examples[i:i + self.batch_size]
                
                # Prepare batch data
                boards = torch.FloatTensor([example[0] for example in batch])
                target_pis = torch.FloatTensor([example[1] for example in batch])
                target_vs = torch.FloatTensor([example[2] for example in batch])
                
                # Forward pass
                pred_pis = self.policy_net(boards)
                pred_vs = self.value_net(boards).squeeze()
                
                # Calculate losses
                policy_loss = -torch.sum(target_pis * torch.log(pred_pis + 1e-8)) / target_pis.size()[0]
                value_loss = torch.sum((target_vs - pred_vs) ** 2) / target_vs.size()[0]
                
                # Backward pass and optimization
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()
                
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()
                
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                batch_count += 1
            
            policy_losses.append(epoch_policy_loss / batch_count)
            value_losses.append(epoch_value_loss / batch_count)
        
        return np.mean(policy_losses), np.mean(value_losses)
    
    def arena(self, new_mcts, num_games=2):#40):
        """
        Compare new MCTS with current MCTS
        """
        wins = 0
        losses = 0
        
        for i in range(num_games):
            # Alternate who goes first
            if i % 2 == 0:
                player1, player2 = new_mcts, self.mcts
            else:
                player1, player2 = self.mcts, new_mcts
            
            state = self.game.startState(1)
            while not self.game.isEnd(state):
                if state[1] == 1:
                    action = player1.choose_move(self.game, state, 
                                               num_simulations=self.num_mcts_sims, 
                                               temperature=0)
                else:
                    action = player2.choose_move(self.game, state, 
                                               num_simulations=self.num_mcts_sims, 
                                               temperature=0)
                state = self.game.enact(state, action)
            
            result = self.game.reward(state)
            
            # Count wins for new_mcts
            if i % 2 == 0:  # new_mcts was player 1
                if result == 1:
                    wins += 1
                elif result == -1:
                    losses += 1
            else:  # new_mcts was player 2
                if result == -1:
                    wins += 1
                elif result == 1:
                    losses += 1
        
        return wins, losses
    
    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        """Save model checkpoint"""
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'training_examples': list(self.training_examples)
        }, filepath)
    
    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        """Load model checkpoint"""
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            print(f"No checkpoint found at {filepath}")
            return False
        
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        self.training_examples = deque(checkpoint['training_examples'], maxlen=self.maxlen_queue)
        
        print(f"Checkpoint loaded from {filepath}")
        return True
    
    def train(self, iterations=100, episodes_per_iteration=25, save_interval=10):
        """
        Main training loop
        """
        policy_losses = []
        value_losses = []
        
        for iteration in range(1, iterations + 1):
            print(f"\n--- Iteration {iteration}/{iterations} ---")
            
            # Self-play
            iteration_examples = deque([], maxlen=self.maxlen_queue)
            
            print("Generating self-play games...")
            for episode in tqdm(range(episodes_per_iteration)):
                examples = self.execute_episode()
                iteration_examples.extend(examples)
            
            # Add to training examples
            self.training_examples.extend(iteration_examples)
            
            # Train neural networks
            if len(self.training_examples) > self.batch_size:
                print("Training neural networks...")
                policy_loss, value_loss = self.learn(list(self.training_examples))
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                print(f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
            
            # Arena comparison (every few iterations)
            if iteration % 5 == 0:
                print("Arena comparison...")
                # Create new MCTS with current networks
                new_mcts = AlphaZeroMCTS(self.policy_net, self.value_net)
                
                # Compare with previous version (simplified - just check if improving)
                wins, losses = self.arena(new_mcts, self.arena_compare)
                win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.5
                print(f"Arena results: {wins} wins, {losses} losses, win rate: {win_rate:.2f}")
                
                if win_rate >= self.update_threshold:
                    print("New model is better! Updating...")
                    self.mcts = new_mcts
                else:
                    print("Keeping previous model.")
            
            # Save checkpoint
            if iteration % save_interval == 0:
                self.save_checkpoint(filename=f'checkpoint_iter_{iteration}.pth.tar')
                print(f"Checkpoint saved at iteration {iteration}")
        
        # Final save
        self.save_checkpoint(filename='final_checkpoint.pth.tar')
        
        # Plot training progress
        if policy_losses:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(policy_losses)
            plt.title('Policy Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            
            plt.subplot(1, 2, 2)
            plt.plot(value_losses)
            plt.title('Value Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            
            plt.tight_layout()
            plt.savefig('training_progress.png')
            plt.show()


class AlphaZeroPlayer:
    """
    Wrapper class for using trained AlphaZero in your main game file
    """
    def __init__(self, checkpoint_path='checkpoint/final_checkpoint.pth.tar', num_mcts_sims=100):
        self.num_mcts_sims = num_mcts_sims
        
        # Initialize networks
        self.policy_net = PolicyP(input_size=64, num_actions=65)
        self.value_net = PolicyV(input_size=64)
        
        # Load trained weights
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.value_net.load_state_dict(checkpoint['value_state_dict'])
            print(f"Loaded trained AlphaZero from {checkpoint_path}")
        else:
            print(f"Warning: No checkpoint found at {checkpoint_path}. Using random weights.")
        
        # Set to evaluation mode
        self.policy_net.eval()
        self.value_net.eval()
        
        # Create MCTS player
        self.mcts = AlphaZeroMCTS(self.policy_net, self.value_net)
    
    def choose_move(self, game, state):
        """Interface for your main game file"""
        return self.mcts.choose_move(game, state, 
                                   num_simulations=self.num_mcts_sims, 
                                   temperature=0)


# Training script
if __name__ == "__main__":
    # Initialize game and trainer
    game = OthelloGame(8)
    trainer = AlphaZeroTrainer(game)
    
    # Try to load existing checkpoint
    trainer.load_checkpoint()
    
    print("Starting AlphaZero training...")
    print("This will take a long time. Consider running on GPU if available.")
    
    # Train the model
    trainer.train(iterations=4, episodes_per_iteration=2, save_interval=1)
    
    print("Training completed!")