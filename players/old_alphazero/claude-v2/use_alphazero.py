"""
Complete usage examples demonstrating how to train and use AlphaZero for Othello.

This file shows:
1. Training a new model from scratch
2. Loading and using a trained model
3. Resuming training from checkpoints
4. Evaluating and benchmarking models
5. Integration with existing game runners
6. Advanced training configurations
"""

import os
import numpy as np
from typing import Optional, Dict, Any

# Import our AlphaZero components

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from players.alphazero import AlphaZero, create_alphazero_player
from players.alphazero_trainer import AlphaZeroTrainer, TrainingConfig, train_new_model
from othello.othello_game import OthelloGame


def example_1_basic_training():
    """Example 1: Train a new AlphaZero model from scratch."""
    print("=== Example 1: Basic Training ===")
    
    # Create trainer with default configuration
    trainer = AlphaZeroTrainer(
        game_class=OthelloGame,
        model_dir='models',
        data_dir='data'
    )
    
    # Train for 50 iterations with 10 games per iteration
    # (Use smaller numbers for quick testing)
    results = trainer.train(iterations=50, games_per_iteration=10)
    
    print(f"Training completed in {results['total_time']:.2f} seconds")
    print(f"Final buffer size: {results['final_buffer_size']}")
    
    return trainer


def example_2_custom_training_config():
    """Example 2: Training with custom configuration."""
    print("\n=== Example 2: Custom Training Configuration ===")
    
    # Create custom training configuration
    config = TrainingConfig(
        episodes_per_move=50,  # Fewer MCTS simulations for faster training
        exploration_constant=1.5,  # Higher exploration
        learning_rate=0.0005,  # Lower learning rate
        batch_size=64,  # Larger batch size
        training_epochs=5,  # Fewer epochs per iteration
        evaluation_games=50,  # Fewer evaluation games
        update_threshold=0.55,  # Lower threshold for model updates
        checkpoint_frequency=5  # Save more frequently
    )
    
    # Create trainer with custom config
    trainer = AlphaZeroTrainer(
        game_class=OthelloGame,
        config=config,
        model_dir='models_custom',
        data_dir='data_custom'
    )
    
    # Train the model
    trainer.train(iterations=30, games_per_iteration=15)
    
    return trainer


def example_3_load_and_use_model():
    """Example 3: Load a trained model and use it for gameplay."""
    print("\n=== Example 3: Loading and Using Trained Model ===")
    
    # Load a trained model (assumes model exists)
    model_path = 'models/best_model.pth'
    
    if os.path.exists(model_path):
        # Load the trained AlphaZero player
        ai_player = AlphaZero(model_path=model_path)
        
        # Create game instance
        game = OthelloGame(8)
        
        # Example: Get a move for a specific game state
        initial_state = game.startState()
        board, current_player = initial_state
        
        print("Initial board:")
        game.print_board(board)
        
        # Get AI move
        move = ai_player.choose_move(game, initial_state)
        
        if move == 64:
            print("AI chooses to pass")
        else:
            row = move // 8
            col = move % 8
            print(f"AI chooses to play at position ({row}, {col})")
        
        return ai_player
    else:
        print(f"Model file {model_path} not found. Train a model first.")
        return None


def example_4_resume_training():
    """Example 4: Resume training from a checkpoint."""
    print("\n=== Example 4: Resume Training ===")
    
    checkpoint_path = 'models/checkpoint_iter_20.pth'
    
    if os.path.exists(checkpoint_path):
        # Resume training from checkpoint
        trainer = AlphaZeroTrainer()
        results = trainer.resume_training(
            checkpoint_path=checkpoint_path,
            additional_iterations=20,
            games_per_iteration=10
        )
        
        print(f"Resumed training completed in {results['total_time']:.2f} seconds")
        return trainer
    else:
        print(f"Checkpoint {checkpoint_path} not found.")
        return None



example_1_basic_training()